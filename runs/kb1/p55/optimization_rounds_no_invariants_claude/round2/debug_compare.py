#!/usr/bin/env python3
"""Compare working test with failing test directly."""
import torch
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

device = "cuda" if torch.cuda.is_available() else "cpu"
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
frag_size = MFMA_K // 4  # 8

print("=" * 60)
print("Test 1: Working pattern from debug_simple_gemmbased.py")
print("=" * 60)

# Create random A and B matrices
torch.manual_seed(42)
A_test = torch.randn((16, 288), dtype=torch.float16, device=device)
B_test = torch.randn((288, 16), dtype=torch.float16, device=device)

# Expected result
expected_test = torch.matmul(A_test.float(), B_test.float())

# Process with tiled MFMA
C_result = torch.zeros((16, 16), dtype=torch.float32, device=device)

k_tiles = 288 // 32

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k
            A_shuffled[lane, t] = A_test[i, k_global]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k
            B_shuffled[lane, t] = B_test[k_global, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_result[4 * g + t, j] += C_shuffled[lane, t]

diff_test = torch.abs(C_result - expected_test)
print(f"Working test: max diff = {diff_test.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 2: Same matrices, but with different access pattern")
print("=" * 60)

# Now let's see if the issue is with how we iterate over spatial tiles
# In the failing test, we have spatial_tile and access A_gemm[row, :]
# where row = spatial_tile * MFMA_M + i

# Let's simulate this with the same A_test matrix
spatial_tile = 0

C_result2 = torch.zeros((16, 16), dtype=torch.float32, device=device)

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            # This is the pattern from the failing test
            row = spatial_tile * MFMA_M + i
            A_shuffled[lane, t] = A_test[row, k_global]  # row = 0 + i

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            oc_tile = 0
            col = oc_tile * MFMA_N + j
            B_shuffled[lane, t] = B_test[k_global, col]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            row = g * 4 + t
            col = j

            hw_idx = spatial_tile * MFMA_M + row
            oc = oc_tile * MFMA_N + col
            C_result2[hw_idx, oc] += C_shuffled[lane, t]

diff_test2 = torch.abs(C_result2 - expected_test)
print(f"Same matrices, different pattern: max diff = {diff_test2.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 3: Check if the shuffle pattern matches")
print("=" * 60)

# Let's verify that the shuffle produces the same result
k_start = 0  # First K tile

# Pattern 1: Direct access
A_shuffled1 = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled1[lane, t] = A_test[i, k]

# Pattern 2: With spatial_tile offset
A_shuffled2 = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16

    for t in range(frag_size):
        k = k_block * frag_size + t

        row = spatial_tile * MFMA_M + i
        A_shuffled2[lane, t] = A_test[row, k]

diff_shuffle = torch.abs(A_shuffled1 - A_shuffled2)
print(f"Shuffle pattern difference: max diff = {diff_shuffle.max().item():.6f}")

# Check if they're exactly the same
print(f"Shuffle patterns equal: {torch.allclose(A_shuffled1, A_shuffled2)}")

print("\n" + "=" * 60)
print("Test 4: Check unshuffle pattern")
print("=" * 60)

# Create a test C_shuffled tensor
C_shuffled_test = torch.arange(64 * 4, dtype=torch.float32, device=device).reshape(64, 4)

# Unshuffle pattern 1: Direct
C_unshuffled1 = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C_unshuffled1[4 * g + t, j] = C_shuffled_test[lane, t]

# Unshuffle pattern 2: With tile offset
C_unshuffled2 = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16

    for t in range(4):
        row = g * 4 + t
        col = j

        hw_idx = spatial_tile * MFMA_M + row
        oc = 0 * MFMA_N + col
        C_unshuffled2[hw_idx, oc] += C_shuffled_test[lane, t]

diff_unshuffle = torch.abs(C_unshuffled1 - C_unshuffled2)
print(f"Unshuffle pattern difference: max diff = {diff_unshuffle.max().item():.6f}")
print(f"Unshuffle patterns equal: {torch.allclose(C_unshuffled1, C_unshuffled2)}")

print("\n" + "=" * 60)
print("Test 5: Check if the issue is with boundary checks")
print("=" * 60)

# In the failing test, we have boundary checks like:
# if row < gemm_m and k_global < GEMM_K:

# Let's see if adding boundary checks affects the result
C_result3 = torch.zeros((16, 16), dtype=torch.float32, device=device)
gemm_m = 289
GEMM_K = 288

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            row = spatial_tile * MFMA_M + i
            if row < gemm_m and k_global < GEMM_K:  # Boundary check
                A_shuffled[lane, t] = A_test[row, k_global]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            col = 0 * MFMA_N + j
            if k_global < GEMM_K and col < 16:  # Boundary check
                B_shuffled[lane, t] = B_test[k_global, col]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            row = g * 4 + t
            col = j

            hw_idx = spatial_tile * MFMA_M + row
            oc = 0 * MFMA_N + col

            if hw_idx < gemm_m and oc < 16:  # Boundary check
                C_result3[hw_idx, oc] += C_shuffled[lane, t]

diff_test3 = torch.abs(C_result3 - expected_test)
print(f"With boundary checks: max diff = {diff_test3.max().item():.6f}")
