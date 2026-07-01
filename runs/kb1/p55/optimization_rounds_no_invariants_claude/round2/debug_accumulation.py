#!/usr/bin/env python3
"""Debug MFMA accumulation behavior."""
import torch
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

device = "cuda" if torch.cuda.is_available() else "cpu"
WARP_SIZE = 64
frag_size = 8

torch.manual_seed(42)

# Simple test: single MFMA call
A = torch.randn((16, 32), dtype=torch.float16, device=device)
B = torch.randn((32, 16), dtype=torch.float16, device=device)
expected = torch.matmul(A, B)

print("=" * 60)
print("Test: Single MFMA accumulation")
print("=" * 60)

# Single MFMA call
C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled[lane, t] = A[i, k]

B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    k_block = lane // 16
    j = lane % 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        B_shuffled[lane, t] = B[k, j]

A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

# Call MFMA
gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

# Unshuffle
C_result = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_result[4 * g + t, j] = C_shuffled[lane, t]

diff = torch.abs(C_result - expected)
print(f"Single MFMA: max diff = {diff.max().item():.6f}")

print("\n" + "=" * 60)
print("Test: MFMA with pre-initialized accumulator")
print("=" * 60)

# What if we pre-initialize the accumulator with non-zero values?
C_shuffled2 = torch.ones((WARP_SIZE, 4), dtype=torch.float32, device=device) * 10.0

gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled2)

C_result2 = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_result2[4 * g + t, j] = C_shuffled2[lane, t]

# Expected: C_result2 = expected + 10
expected_plus10 = expected + 10.0
diff2 = torch.abs(C_result2 - expected_plus10)
print(f"MFMA with pre-init accumulator: max diff = {diff2.max().item():.6f}")
print(f"C_result2[0,0] = {C_result2[0,0].item():.4f}, expected = {expected_plus10[0,0].item():.4f}")

print("\n" + "=" * 60)
print("Test: Two K tiles with single C_shuffled")
print("=" * 60)

# Test with K = 64, two K tiles of 32 each
A_k64 = torch.randn((16, 64), dtype=torch.float16, device=device)
B_k64 = torch.randn((64, 16), dtype=torch.float16, device=device)
expected_k64 = torch.matmul(A_k64.float(), B_k64.float())

C_shuffled3 = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

# Process first K tile (K=0-31)
A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled[lane, t] = A_k64[i, k]

B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    k_block = lane // 16
    j = lane % 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        B_shuffled[lane, t] = B_k64[k, j]

A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled3)

# Process second K tile (K=32-63)
A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled[lane, t] = A_k64[i, 32 + k]

B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    k_block = lane // 16
    j = lane % 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        B_shuffled[lane, t] = B_k64[32 + k, j]

A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled3)

C_result3 = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_result3[4 * g + t, j] = C_shuffled3[lane, t]

diff3 = torch.abs(C_result3 - expected_k64)
print(f"Two K tiles with same C_shuffled: max diff = {diff3.max().item():.6f}")
print(f"C_result3[0,0] = {C_result3[0,0].item():.4f}, expected = {expected_k64[0,0].item():.4f}")

print("\n" + "=" * 60)
print("Test: Check if C_shuffled is modified in-place")
print("=" * 60)

# Verify that MFMA modifies C_shuffled in-place
C_shuffled_before = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)
C_shuffled_after = C_shuffled_before.clone()

# Use the same packed tensors from the single K=32 test
A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled[lane, t] = A[i, k]

B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    k_block = lane // 16
    j = lane % 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        B_shuffled[lane, t] = B[k, j]

A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

print(f"C_shuffled_before id: {id(C_shuffled_before)}")
gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled_after)
print(f"C_shuffled_after id: {id(C_shuffled_after)}")

diff_ids = C_shuffled_before is C_shuffled_after
print(f"Same tensor: {diff_ids}")

# Check if C_shuffled_after has the result
C_check = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_check[4 * g + t, j] = C_shuffled_after[lane, t]

diff_check = torch.abs(C_check - expected)
print(f"C_shuffled_after has result: max diff = {diff_check.max().item():.6f}")
