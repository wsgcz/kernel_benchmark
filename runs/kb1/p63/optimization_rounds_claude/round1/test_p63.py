#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p63/optimization_rounds_claude/round1")

import torch
import torch.nn.functional as F
from output_model_new import ModelNew

def test_p63_kernel():
    """Test the p63 Conv2D kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Create model with benchmark parameters
    model = ModelNew(
        in_channels=16,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False
    ).cuda()

    # Create input with benchmark shape
    x = torch.randn((16, 16, 1024, 1024), dtype=torch.float32, device="cuda")

    # Run optimized kernel
    actual = model(x)

    # Compute expected result using PyTorch
    weight = model.conv2d.weight.to(dtype=torch.float32)
    expected = F.conv2d(x, weight, stride=1, padding=0)

    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
    mean_diff = torch.mean(torch.abs(actual_cpu - expected_cpu)).item()

    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Actual shape: {actual.shape}")
    print(f"Expected shape: {expected.shape}")

    passed = torch.allclose(actual_cpu, expected_cpu, rtol=1e-2, atol=0.1)
    print(f"Pass: {passed}")

    if not passed:
        # Check some specific locations
        print(f"\nSample comparisons:")
        print(f"actual[0,0,0,0]: {actual_cpu[0,0,0,0]}")
        print(f"expected[0,0,0,0]: {expected_cpu[0,0,0,0]}")
        print(f"actual[0,0,0,:4]: {actual_cpu[0,0,0,:4]}")
        print(f"expected[0,0,0,:4]: {expected_cpu[0,0,0,:4]}")

if __name__ == "__main__":
    test_p63_kernel()
