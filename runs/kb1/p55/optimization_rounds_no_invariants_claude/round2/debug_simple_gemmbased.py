#!/usr/bin/env python3
"""Test MFMA with exact reference pattern."""
import torch
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Use EXACT dimensions from reference test
M, N, K = 16, 16, 32
WARP_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Test 1: Reference GEMM pattern")
print("=" * 60)

torch.manual_seed(0)
A = torch.randn((M, K), dtype=torch.float16, device=device)
B = torch.randn((K, N), dtype=torch.float16, device=device)
C = torch.zeros((M, N), dtype=torch.float16, device=device)

expected = torch.matmul(A, B)
print(f"Expected shape: {expected.shape}")

# EXACT code from reference test
frag_size = K // 4  # 8

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
C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C[4 * g + t, j] = C_shuffled[lane, t].to(torch.float16)

actual = C.to(device=device, dtype=torch.float32)
diff = torch.abs(actual - expected.float())
print(f"Reference pattern: max diff = {diff.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 2: Tiled GEMM pattern (simulating Conv2D tile)")
print("=" * 60)

# Now simulate the Conv2D scenario:
# - M_total = 16 (one spatial tile)
# - K_total = 288 (32 channels * 9 kernel area)
# - Process K in tiles of 32

M_total = 16  # One spatial tile
N_total = 16  # One OC tile
K_total = 288  # Full K dimension
MFMA_K = 32  # K per MFMA call

# Create larger matrices
A_large = torch.randn((M_total, K_total), dtype=torch.float16, device=device)
B_large = torch.randn((K_total, N_total), dtype=torch.float16, device=device)

# Expected result
expected_large = torch.matmul(A_large.float(), B_large.float())

# Process with tiled MFMA
C_large = torch.zeros((M_total, N_total), dtype=torch.float32, device=device)

k_tiles = K_total // MFMA_K  # 9

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    # Shuffle for this K tile
    frag_size_tile = MFMA_K // 4  # 8

    A_shuffled = torch.zeros((WARP_SIZE, frag_size_tile), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size_tile):
            k = k_block * frag_size_tile + t
            k_global = k_start + k
            A_shuffled[lane, t] = A_large[i, k_global]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size_tile), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size_tile):
            k = k_block * frag_size_tile + t
            k_global = k_start + k
            B_shuffled[lane, t] = B_large[k_global, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_tile // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_tile // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unshuffle and accumulate
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_large[4 * g + t, j] += C_shuffled[lane, t]

diff_large = torch.abs(C_large - expected_large)
print(f"Tiled GEMM (K_total=288): max diff = {diff_large.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 3: Split-K with tiled GEMM")
print("=" * 60)

SPLIT_K = 2
K_per_split = K_total // SPLIT_K  # 144

C_split = torch.zeros((M_total, N_total), dtype=torch.float32, device=device)

for split_id in range(SPLIT_K):
    k_split_start = split_id * K_per_split
    k_split_end = k_split_start + K_per_split

    print(f"Split {split_id}: K = {k_split_start} to {k_split_end - 1}")

    # Process K tiles within this split
    for k_tile in range(k_split_start // MFMA_K, k_split_end // MFMA_K):
        k_start = k_tile * MFMA_K

        # Shuffle
        frag_size_tile = MFMA_K // 4

        A_shuffled = torch.zeros((WARP_SIZE, frag_size_tile), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            i = lane % 16
            k_block = lane // 16
            for t in range(frag_size_tile):
                k = k_block * frag_size_tile + t
                k_global = k_start + k
                A_shuffled[lane, t] = A_large[i, k_global]

        B_shuffled = torch.zeros((WARP_SIZE, frag_size_tile), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            k_block = lane // 16
            j = lane % 16
            for t in range(frag_size_tile):
                k = k_block * frag_size_tile + t
                k_global = k_start + k
                B_shuffled[lane, t] = B_large[k_global, j]

        A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_tile // 2)
        B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_tile // 2)
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        # Unshuffle and accumulate
        for lane in range(WARP_SIZE):
            g = lane // 16
            j = lane % 16
            for t in range(4):
                C_split[4 * g + t, j] += C_shuffled[lane, t]

diff_split = torch.abs(C_split - expected_large)
print(f"Split-K GEMM: max diff = {diff_split.max().item():.6f}")
print(f"Result matches: {torch.allclose(C_split, expected_large, rtol=1e-2, atol=1e-3)}")
