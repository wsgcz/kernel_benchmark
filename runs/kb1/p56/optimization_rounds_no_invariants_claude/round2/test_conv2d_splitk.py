#!/usr/bin/env python3
"""Test script for MFMA Split-K Conv2D kernel."""
import unittest
import sys

import torch
import torch.nn.functional as F

# Import the optimized kernel
sys.path.insert(0, "/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p56/optimization_rounds_no_invariants_claude/round2")
from output_model_new import ModelNew


class TestMFMASplitKConv2D(unittest.TestCase):
    """Test MFMA-optimized Split-K Conv2D kernel."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)
        self.rtol = 1e-2
        self.atol = 0.1  # Tolerance for bf16 precision

    def test_mfma_splitk_conv2d_benchmark_shape(self):
        """Test MFMA Split-K Conv2D for the benchmark shapes."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        if not hasattr(torch.version, "hip") or torch.version.hip is None:
            self.skipTest("HIP is not available.")

        # Create model with the kernel's expected parameters
        # BATCH=8, IN_CHANNELS=64, IN_H=512, IN_W=256, OUT_CHANNELS=128, KERNEL_H=5, KERNEL_W=7
        model = ModelNew(
            in_channels=64,
            out_channels=128,
            kernel_size=(5, 7),
            stride=1,
            padding=0,
            bias=False
        ).cuda()

        # Create input with the specific shape the kernel expects
        # Shape: (8, 64, 512, 256)
        x = torch.randn((8, 64, 512, 256), dtype=torch.float32, device="cuda")

        # Run optimized kernel
        actual = model(x)

        # Compute expected result using PyTorch
        weight = model.conv2d.weight.to(dtype=torch.float32)
        expected = F.conv2d(x, weight, stride=1, padding=0)

        actual_cpu = actual.cpu().to(torch.float32)
        expected_cpu = expected.cpu()

        max_diff = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
        mean_diff = torch.mean(torch.abs(actual_cpu - expected_cpu)).item()

        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Actual shape: {actual.shape}")
        print(f"Expected shape: {expected.shape}")

        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=self.rtol, atol=self.atol),
            msg=f"MFMA Split-K Conv2D results do not match.\n"
                f"Max absolute difference: {max_diff}\n"
                f"Mean absolute difference: {mean_diff}"
        )

    def test_mfma_splitk_conv2d_output_shape(self):
        """Test that output shape is correct."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")

        model = ModelNew(64, 128, (5, 7)).cuda()
        x = torch.randn((8, 64, 512, 256), dtype=torch.float32, device="cuda")

        output = model(x)

        # Expected output: (8, 128, 508, 250)
        expected_shape = (8, 128, 508, 250)
        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch: got {output.shape}, expected {expected_shape}")

    def test_mfma_splitk_conv2d_multiple_inputs(self):
        """Test with multiple random inputs."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        if not hasattr(torch.version, "hip") or torch.version.hip is None:
            self.skipTest("HIP is not available.")

        model = ModelNew(64, 128, (5, 7)).cuda()

        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            x = torch.randn((8, 64, 512, 256), dtype=torch.float32, device="cuda")

            actual = model(x)
            weight = model.conv2d.weight.to(dtype=torch.float32)
            expected = F.conv2d(x, weight, stride=1, padding=0)

            actual_f32 = actual.to(torch.float32)
            self.assertTrue(
                torch.allclose(actual_f32, expected, rtol=self.rtol, atol=self.atol),
                msg=f"Trial with seed {seed} failed"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
