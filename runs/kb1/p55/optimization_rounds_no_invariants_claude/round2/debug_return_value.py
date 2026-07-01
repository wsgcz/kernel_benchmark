#!/usr/bin/env python3
"""Test with correct return value handling."""
import torch
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

device = "cuda" if torch.cuda.is_available() else "cpu"
WARP_SIZE = 64
frag_size = 8

torch.manual_seed(42)

A = torch.randn((16, 64), dtype=torch.float16, device=device)
B = torch.randn((64, 16), dtype=torch.float16, device=device)
expected = torch.matmul(A.float(), B.float())

print("=" * 60)
print("Test: Capture return value")
print("=" * 60)

C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

for k_tile in range(2):
    k_start = k_tile * 32

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            A_shuffled[lane, t] = A[i, k_start + k]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            B_shuffled[lane, t] = B[k_start + k, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

    # CAPTURE the return value!
    C_shuffled = gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

C_result = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_result[4 * g + t, j] = C_shuffled[lane, t]

diff = torch.abs(C_result - expected)
print(f"With return value capture: max diff = {diff.max().item():.6f}")
print(f"Result matches: {torch.allclose(C_result, expected, rtol=1e-2, atol=1e-3)}")
