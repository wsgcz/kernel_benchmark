#!/usr/bin/env python3
"""Detailed debug of single spatial tile."""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

device = "cuda" if torch.cuda.is_available() else "cpu"
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
frag_size = MFMA_K // 4  # 8

# Create Conv2D scenario
BATCH = 1
IN_CHANNELS = 32
OUT_CHANNELS = 16
IN_H = 19
IN_W = 19
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1
OUT_W = IN_W - KERNEL_W + 1
KERNEL_AREA = KERNEL_H * KERNEL_W
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 288

gemm_m = BATCH * OUT_H * OUT_W  # 289

torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)
w = conv.weight

expected = conv(x)

x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

# Build A matrix
A_gemm = torch.zeros((gemm_m, GEMM_K), dtype=torch.float16, device=device)
for hw_idx in range(gemm_m):
    batch = hw_idx // (OUT_H * OUT_W)
    hw_in_batch = hw_idx % (OUT_H * OUT_W)
    oh = hw_in_batch // OUT_W
    ow = hw_in_batch % OUT_W
    for k in range(GEMM_K):
        c = k // KERNEL_AREA
        spatial = k % KERNEL_AREA
        kh = spatial // KERNEL_W
        kw = spatial % KERNEL_W
        ih = oh + kh
        iw = ow + kw
        A_gemm[hw_idx, k] = x_f16[batch, c, ih, iw]

# Build B matrix
B_gemm = w_f16.reshape(OUT_CHANNELS, GEMM_K).t().contiguous()

# Reference GEMM
C_ref = torch.matmul(A_gemm.float(), B_gemm.float())

print("=" * 60)
print("Test: Compare direct MFMA vs tile-by-tile for same data")
print("=" * 60)

# Direct approach: process first 16 rows of A_gemm as if it were a standalone GEMM
A_first16 = A_gemm[0:16, :]
expected_first16 = C_ref[0:16, 0:16]

print(f"A_first16 shape: {A_first16.shape}")
print(f"Expected first16 shape: {expected_first16.shape}")

# Process with MFMA
C_result = torch.zeros((16, 16), dtype=torch.float32, device=device)
k_tiles = GEMM_K // MFMA_K

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k
            A_shuffled[lane, t] = A_first16[i, k_global]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k
            B_shuffled[lane, t] = B_gemm[k_global, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_result[4 * g + t, j] += C_shuffled[lane, t]

diff_direct = torch.abs(C_result - expected_first16)
print(f"Direct MFMA (first 16 rows): max diff = {diff_direct.max().item():.6f}")

print("\n" + "=" * 60)
print("Test: Process the same data but through tile iteration")
print("=" * 60)

# Now process the same data but using spatial_tile pattern
spatial_tile = 0
oc_tile = 0

C_result2 = torch.zeros((16, 16), dtype=torch.float32, device=device)

C_shuffled2 = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % MFMA_M
        k_block = lane // MFMA_M

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            row = spatial_tile * MFMA_M + i
            if row < gemm_m and k_global < GEMM_K:
                A_shuffled[lane, t] = A_gemm[row, k_global]  # A_gemm[0+i, k_global]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // MFMA_N
        j = lane % MFMA_N

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            col = oc_tile * MFMA_N + j
            if k_global < GEMM_K and col < OUT_CHANNELS:
                B_shuffled[lane, t] = B_gemm[k_global, col]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled2)

for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        row = g * 4 + t
        col = j

        hw_idx = spatial_tile * MFMA_M + row
        oc = oc_tile * MFMA_N + col

        if hw_idx < gemm_m and oc < OUT_CHANNELS:
            C_result2[hw_idx, oc] = C_shuffled2[lane, t]

diff_tile = torch.abs(C_result2 - expected_first16)
print(f"Tile iteration (spatial_tile=0): max diff = {diff_tile.max().item():.6f}")

# Compare the two results
diff_both = torch.abs(C_result - C_result2)
print(f"Difference between direct and tile: max diff = {diff_both.max().item():.6f}")

print("\n" + "=" * 60)
print("Test: Check if A_gemm[first 16 rows] matches A_first16")
print("=" * 60)

A_from_gemm = A_gemm[0:16, :]
diff_A = torch.abs(A_first16 - A_from_gemm)
print(f"A_gemm[first 16] vs A_first16: max diff = {diff_A.max().item():.6f}")

# They should be the same, but let's verify the actual values
print(f"A_first16[0, 0:5] = {A_first16[0, 0:5]}")
print(f"A_gemm[0, 0:5] = {A_gemm[0, 0:5]}")

print("\n" + "=" * 60)
print("Test: Check shuffle consistency")
print("=" * 60)

# Check if the shuffle produces the same values
k_start = 0  # First K tile

A_shuffled_direct = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled_direct[lane, t] = A_first16[i, k]

A_shuffled_tile = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % MFMA_M
    k_block = lane // MFMA_M

    for t in range(frag_size):
        k = k_block * frag_size + t
        k_global = k_start + k

        row = 0 * MFMA_M + i
        if row < gemm_m and k_global < GEMM_K:
            A_shuffled_tile[lane, t] = A_gemm[row, k_global]

diff_shuffle = torch.abs(A_shuffled_direct - A_shuffled_tile)
print(f"Shuffle consistency: max diff = {diff_shuffle.max().item():.6f}")
print(f"Shuffles equal: {torch.allclose(A_shuffled_direct, A_shuffled_tile)}")
