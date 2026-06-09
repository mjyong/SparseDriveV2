import logging
import os
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union
from functools import partial

import torch.distributed as dist
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.evaluate.pdm_score_fix_bug import pdm_score as pdm_score_fix_bug
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_fast"

# =============================================================================
# NAVSIM v2 (EPDMS) 快速评测脚本。整体流程：
#   1. 加载 agent + 权重，用 CacheOnlyDataset(从 data_cache 读特征) 跑 trainer.predict，
#      得到每个 token 的预测轨迹 merged_predictions = {token: Trajectory}
#   2. 按 log 把 token 分发给多个 worker(worker_map)，每个 worker 调 run_pdm_score：
#        读 metric_cache → PDMSimulator 仿真 + PDMScorer 闭环打分 → 每帧一个 score 行(DataFrame)
#   3. infer_start_adjacent_mapping: 找同一 log 内时间相邻的帧对
#      create_scene_aggregators: 据此算"两帧扩展舒适度"并回填
#      compute_final_scores: 用乘性指标×加权指标算最终 EPDMS 分
#   4. 求均值、存 csv。脚本会跑两遍：bug 版(pdm_score) 与 修正版(pdm_score_fix_bug)，
#      分别输出 navtest_v2.csv 和 navtest_v2_bug_fix.csv
# =============================================================================


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]], pdm_score_fn) -> List[pd.DataFrame]:
    """
    单个 worker 内运行的 PDMS 评测函数（被 worker_map 并行调用）。
    :param args: 该 worker 分到的数据点列表，每个含 cfg / log_file / tokens / model_trajectory
    :param pdm_score_fn: 实际打分函数（pdm_score 或 pdm_score_fix_bug）
    :return: 每个被评测 token 一行的 DataFrame 列表
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    # 汇总本 worker 负责的 log 与 token；只保留模型确实给出了预测轨迹的 token
    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = args[0]['model_trajectory']
    tokens = [t for t in tokens if t in model_trajectory]

    # 实例化仿真器与打分器（二者的 proposal_sampling 必须一致）
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    # v2 引入背景车策略：non_reactive(日志回放) 或 reactive(对自车反应)
    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
    # 加载 metric_cache(预计算的仿真上下文) 与场景；只评测两者交集且模型有预测的 token
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[pd.DataFrame] = []
    # 逐 token：取 metric_cache + 预测轨迹 → 闭环打分 → 组装一行结果（含 log_name/start_time 等元信息）
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]

            score_row, ego_simulated_states = pdm_score_fn(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            # 保存自车仿真状态序列，供后面跨两帧计算 two_frame_extended_comfort 用
            score_row["ego_simulated_states"] = [ego_simulated_states]

        except Exception:
            # 单 token 失败不影响整体：记一行空结果并标 valid=False
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token

        pdm_results.append(score_row)
    return pdm_results


def infer_start_adjacent_mapping(score_df: pd.DataFrame, time_gap_threshold: float = 0.55) -> Dict[str, str]:
    """
    Infers an adjacent mapping from the score_df DataFrame by start time.
    Each current-token is mapped to its previous-token if they are adjacent.
    Used to create the two-frame extended comfort score (reversed direction).

    :param score_df: DataFrame containing at least 'token', 'log_name', 'start_time'.
    :param time_gap_threshold: Maximum allowed gap (in seconds) between two frames to
                               consider them "adjacent".
    :return: Dictionary mapping each current-token to one previous-token.
    """
    adjacent_mapping: Dict[str, str] = {}

    # 注意：只在同一 log 内、且按 start_time 排序后时间差 ≤ 0.55s 的相邻帧之间建立映射。
    # 稀疏子集(mini/navmini)里每个 log 往往只有单帧或帧不相邻 → 返回空 dict，
    # 这会导致 create_scene_aggregators 里 all_updates 为空（已在该函数加保护）。
    for log_name, group_df in score_df[score_df["frame_type"] == SceneFrameType.ORIGINAL].groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)

        for i in range(1, len(group_df)):
            prev_row = group_df.iloc[i - 1]
            current_row = group_df.iloc[i]

            prev_token = prev_row["token"]
            current_token = current_row["token"]
            time_diff = current_row["start_time"] - prev_row["start_time"]

            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token

    return adjacent_mapping


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final scores for each row in pdm_score_df after updating
    the weighted metrics with two-frame extended comfort.

    If 'two_frame_extended_comfort' is NaN for a row, the corresponding
    metric and its weight are set to zero, effectively ignoring it
    during normalization.

    :param pdm_score_df: DataFrame containing PDM scores and metrics.
    :return: A new DataFrame with the computed final scores.
    """
    df = pdm_score_df.copy()

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()  # shape: (N, )
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())  # shape: (N, M)
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())  # shape: (N, M)

    mask = np.isnan(two_frame_scores)
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0

    non_mask = ~mask
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    weighted_metric_scores = weighted_sum / total_weight

    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(
        columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"],
        inplace=True,
    )

    return df


def create_scene_aggregators(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    """
    根据相邻帧映射(all_mappings)计算"两帧扩展舒适度"(two_frame_extended_comfort)，
    并把结果回填进 full_score_df。

    :param all_mappings: {当前帧token: 前一帧token}，由 infer_start_adjacent_mapping 生成
    :param full_score_df: 单帧 PDM 打分结果(含 ego_simulated_states 列，供跨帧聚合用)
    :param proposal_sampling: 轨迹采样参数
    :return: 增加了 two_frame_extended_comfort 列、去掉 ego_simulated_states 列的 DataFrame
    """

    # 先把该列全部置 NaN（NaN 表示"该帧没有可配对的前一帧"，compute_final_scores 会把它的权重归零）
    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []

    # 对每一对(当前帧, 前一帧)做跨帧聚合，得到该帧的两帧扩展舒适度
    for now_frame, previous_frame in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)

        all_updates.append(updated_rows)

    # ★ 保护：稀疏子集(如 mini/navmini)上可能没有任何时间相邻的帧对，
    #   此时 all_updates 为空，pd.concat([]) 会抛 "No objects to concatenate"。
    #   空时直接跳过回填——two_frame_extended_comfort 保持 NaN（其权重在后续被归零），不影响其余指标。
    if all_updates:
        all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
        full_score_df.update(all_updates_df)

    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])

    return full_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent
    )

    logger.info("Building Datasets")
    logger.info(f"Loading test set features from: {cfg.test_cache_path}")
    test_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    test_dataset = CacheOnlyDataset(
        cache_path=cfg.test_cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=test_scene_filter.log_names,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.params.batch_size,
        num_workers=cfg.dataloader.params.batch_size,
        shuffle=False,
        drop_last=False,
    )
    logger.info("Num test samples: %d", len(test_dataset))

    logger.info("Building Trainer")
    original_callbacks = agent.get_training_callbacks()
    callbacks = original_callbacks
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    # 第一阶段：前向推理。每个 batch 经 predict_step 输出 {token: Trajectory}
    logger.info("Starting Validation")
    predictions = trainer.predict(
        model=lightning_module,
        dataloaders=test_dataloader,
    )

    # 多卡时把各 rank 的预测 all_gather 后合并；单卡直接用
    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    # 合并成全局 {token: Trajectory}
    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    
    ################################## bug_version（用原始 pdm_score，复现官方 bug 前的口径）
    # 第二阶段：多进程闭环打分 → 合并所有 token 的分行
    score_rows: List[pd.DataFrame] = worker_map(worker, partial(run_pdm_score, pdm_score_fn=pdm_score), data_points)

    pdm_score_df = pd.concat(score_rows)

    # 第三阶段：两帧扩展舒适度聚合 + 计算最终 EPDMS 分
    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    # Calculate average score
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    # 末行追加均值，存为 navtest_v2.csv（bug 版口径）
    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / "navtest_v2.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / "navtest_v2.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )

    ################################## bug_fix version（与上面完全同流程，仅把打分函数换成 pdm_score_fix_bug，
    #   对应官方 issue #151 修复后的口径；结果存 navtest_v2_bug_fix.csv。两份口径都会报告。）
    score_rows: List[pd.DataFrame] = worker_map(worker, partial(run_pdm_score, pdm_score_fn=pdm_score_fix_bug), data_points)

    pdm_score_df = pd.concat(score_rows)

    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    # Calculate average score
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / "navtest_v2_bug_fix.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / "navtest_v2_bug_fix.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )



if __name__ == "__main__":
    main()
