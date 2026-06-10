"""
TensorRT 引擎 vs PyTorch 数值校验（fp32）。
用法（GPU 机器，需 tensorrt python 包）:
    python deploy/verify_trt.py --engine deploy/sparsedrive_fp32.engine \
                                --io deploy/sparsedrive_fp32_io.npz
判定:
    1. trajectory max|diff| < 1e-3 （fp32 跨框架累积误差量级）
    2. 选出的轨迹完全一致（argmax 同一条 anchor）—— 这才是部署正确性的硬标准:
       模型输出本质是"从离散词表选一条"，只要 argmax 不变，轨迹逐点 bit 级一致。
"""
import argparse

import numpy as np
import tensorrt as trt
import torch

INPUT_NAMES = ["imgs", "projection_mat", "image_wh", "status_feature"]


def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as rt:
        return rt.deserialize_cuda_engine(f.read())


def run_trt(engine, feed: dict):
    """用 torch.cuda 张量当显存缓冲跑 TRT，兼容 TRT 8.x(execute_v2) 与 10.x(execute_async_v3)。"""
    ctx = engine.create_execution_context()
    use_v3 = hasattr(engine, "num_io_tensors")          # TRT >= 8.5 的新 IO API

    buffers, out_host = {}, {}
    if use_v3:
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for name in names:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                t = torch.from_numpy(feed[name]).contiguous().cuda()
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                t = torch.empty(shape, dtype=torch.float32, device="cuda")
                out_host[name] = t
            buffers[name] = t
            ctx.set_tensor_address(name, int(t.data_ptr()))
        stream = torch.cuda.Stream()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    else:                                               # 旧 bindings API
        bindings = []
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                t = torch.from_numpy(feed[name]).contiguous().cuda()
            else:
                shape = tuple(ctx.get_binding_shape(i))
                t = torch.empty(shape, dtype=torch.float32, device="cuda")
                out_host[name] = t
            buffers[name] = t
            bindings.append(int(t.data_ptr()))
        ctx.execute_v2(bindings)
        torch.cuda.synchronize()
    return {k: v.cpu().numpy() for k, v in out_host.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--io", required=True, help="export_onnx.py 保存的 *_io.npz")
    args = ap.parse_args()

    data = np.load(args.io)
    feed = {k: data[k].astype(np.float32) for k in INPUT_NAMES}
    ref = data["trajectory_ref"]

    engine = load_engine(args.engine)
    out = run_trt(engine, feed)["trajectory"]

    diff = np.abs(out - ref)
    print(f"[trt]   trajectory shape={out.shape}  first pose={out[0, 0].tolist()}")
    print(f"[torch] first pose={ref[0, 0].tolist()}")
    print(f"[verify torch<->trt] max|diff|={diff.max():.3e}  mean={diff.mean():.3e}")

    # 硬标准: 选中的是同一条词表轨迹（逐点完全一致 = argmax 没漂移）
    same_choice = np.allclose(out, ref, atol=1e-3)
    assert same_choice, "TRT 与 PyTorch 选择的轨迹不一致！请检查 TF32 是否已关闭(--noTF32)"
    print("[PASS] TensorRT 引擎与 PyTorch 输出一致 (fp32)")


if __name__ == "__main__":
    main()
