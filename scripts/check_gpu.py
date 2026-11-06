"""Confirm PyTorch can see the GPU and actually run kernels on it.

RTX 50-series cards need a PyTorch build with CUDA 12.8 or newer. Older builds
import fine and even report the GPU, then fail on the first real kernel, so this
script runs a convolution instead of only checking is_available().
"""
import sys
import time

import torch
import torch.nn.functional as F


def main() -> int:
    print(f"Python      {sys.version.split()[0]}")
    print(f"PyTorch     {torch.__version__}")
    print(f"CUDA build  {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Check the PyTorch build and NVIDIA driver.")
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Device      {props.name}")
    print(f"Memory      {props.total_memory / 1024**3:.1f} GB")
    print(f"Capability  sm_{props.major}{props.minor}")
    print(f"bf16        {torch.cuda.is_bf16_supported()}")

    x = torch.randn(8, 3, 256, 256, device=device)
    w = torch.randn(32, 3, 9, 9, device=device)
    y = F.conv2d(x, w, padding=4)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(50):
        y = F.conv2d(x, w, padding=4)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 50 * 1000

    print(f"Conv test   output {tuple(y.shape)}, {ms:.2f} ms per call")
    print("PASS: GPU convolution ran successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
