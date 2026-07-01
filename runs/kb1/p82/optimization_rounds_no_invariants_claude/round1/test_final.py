#!/usr/bin/env python3
"""
Correct MFMA depthwise convolution using K for kernel positions.

Key insight:
- Use MFMA's K dimension for kernel positions (0..8, pad 9..15 with zeros)
- A[i][k] = X[i + k0_k][j + k1_k] (input value at output position i, j with kernel offset)
- B[k][j] = W[k0_k][k1_k] (kernel weight)

The result: C[i][j] = Σ_k A[i][k] * B[k][j] = Σ_k X[i+k0_k][j+k1_k] * W[k0_k][k1_k]
This is exactly the depthwise convolution formula!

Lane layout for MFMA_16x16x16:
- Lane l holds A[l % 16][4 * (l // 16) + t] for t in 0..3
- Lane l holds B[4 * (l // 16) + t][l % 16] for t in 0..3
- Lane l computes C[l % 16][4 * (l // 16) + t] for t in 0..3
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64
NUM_KERNEL_POS = 9  # 3x3 kernel

# Kernel position to (k0, k1) mapping
# k_pos: 0  1  2  3  4  5  6  7  8
# k0:    0  0  0  1  1  1  2  2  2
# k1:    0  1  2  0  1  2  0  1  2

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def depthwise_mfma_kernel(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Depthwise conv using MFMA with K as kernel position dimension."""
    lane = S.thread_id(0)

    row_idx = lane % 16      # i = output row, also j for B matrix
    k_grp = lane // 16       # which group of k values (0, 1, 2, 3)

    # Shared memory for MFMA fragments
    input_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    weight_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    input_shmem_f16 = S.view(input_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    weight_shmem_f16 = S.view(weight_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    # Load A matrix: A[i][k] = X[i + k0_k][j + k1_k]
    # Lane l holds A[row_idx][4*k_grp + t] where j = 4*k_grp + t (output column)
    for t in S.range(4):
        k_pos = 4 * k_grp + t  # kernel position index (0..15)

        # Compute kernel offsets (k0, k1) from k_pos
        # k_pos = k0 * 3 + k1
        k0 = k_pos // 3
        k1 = k_pos % 3

        # Output column j = 4*k_grp + t
        j = k_pos  # For simplicity, assume output column matches k_pos

        # But wait - each lane computes 4 different output columns!
        # j = 4 * k_grp + t, not k_pos

        # Actually, let me reconsider.
        # Lane l computes C[row_idx][4*k_grp + t'] for t' in 0..3
        # For each t', we need to sum over k.

        # A[row_idx][k] should be X[row_idx + k0_k][j + k1_k] where j = 4*k_grp + t'
        # But we're loading A for k = 4*k_grp + t (different t)

        # Let me fix this: A[i][k] for a fixed k should be the same across all i.
        # A[i][k] = X[i + k0_k][j + k1_k] where j varies based on which C column we're computing.

        # This is confusing. Let me think step by step:
        # For C[i][j], the formula is C[i][j] = Σ_k A[i][k] * B[k][j]
        # A[i][k] should be X[i + k0_k][j + k1_k]
        # B[k][j] should be W[k0_k][k1_k]

        # But A[i][k] depends on j, which is different for each output column!
        # Lane l computes C for columns j = 4*k_grp + t'
        # Lane l holds A[row_idx][k] for k = 4*k_grp + t (fixed k, not j)

        # So A[i][k] should be loaded independent of j.
        # A[i][k] = X[i + k0_k][k + k1_k] ??? That doesn't make sense either.

        # Let me reconsider the mapping.
        # For MFMA C = A * B:
        # - A is M x K
        # - B is K x N
        # - C is M x N
        #
        # For our problem:
        # - M = 16 (output rows)
        # - N = 16 (output columns)
        # - K = 9 (kernel positions) -> pad to 16
        #
        # A[i][k] should give the input value for output row i, kernel position k.
        # For kernel position k with offsets (k0_k, k1_k):
        #   A[i][k] = X[i + k0_k][j + k1_k] where j is ???
        #
        # The problem is that A[i][k] can only have one value, but the input
        # position depends on output column j.
        #
        # The solution: A[i][k] = X[i + k0_k][k + k1_k] (use k as column index)
        # This is wrong because k is kernel position, not output column.
        #
        # Alternative: Make A[i][k][j] a 3D tensor. But MFMA doesn't support that.
        pass

    # Actually, the correct approach is to recognize that for depthwise conv:
    # Each output position (i, j) is computed independently.
    # We can't use MFMA efficiently for this because MFMA computes C[i][j] = Σ_k A[i][k] * B[k][j]
    # which couples k across rows and columns.
    #
    # The best we can do with MFMA is:
    # For each kernel position (k0, k1):
    #   Load input patch into A (16x16)
    #   Set B = weight (scalar broadcast)
    #   C = A * B = input_patch * weight
    # But this is just scalar multiplication, not using MFMA's K reduction.
    #
    # Accumulate: Y[i][j] += input[i+k0][j+k1] * weight[k0][k1]

    # For this to work with MFMA:
    # A[i][k] = δ[i, some_relation] * input[i+k0][j+k1]
    # But this requires j to be known, which varies.
    #
    # Simplest working approach: Just compute directly without MFMA's K reduction.
    # Use MFMA to multiply a 16x16 input patch by a scalar weight.
    pass


@substrate.jit
def depthwise_simple_mfma(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """
    Simple approach: For each kernel position, load input and multiply by weight.
    Use MFMA to compute C = A * B where B is constant.
    This gives C[i][j] = A[i][j] * weight (incorrect, sums over k).
    """
    lane = S.thread_id(0)

    row_idx = lane % 16
    k_grp = lane // 16

    input_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    weight_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    input_shmem_f16 = S.view(input_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    weight_shmem_f16 = S.view(weight_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    acc = S.full((4,), 0.0, S.f32)

    # For each kernel position
    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load input: A[i][j] = X[i+k0][j+k1]
            # Lane l holds A[row_idx][4*k_grp + t]
            # j = 4*k_grp + t (output column)
            for t in S.range(4):
                j = 4 * k_grp + t
                i0 = row_idx + k0
                i1 = j + k1
                input_shmem_f16[lane, t] = X[0, 0, i0, i1]
                input_shmem_f16[lane, t + 4] = input_shmem_f16[lane, t]

            # Set B[k][j] = weight for all k
            # This means: C[i][j] = Σ_k A[i][k] * weight = weight * Σ_k A[i][k]
            # WRONG! This sums over k, giving weight * (A[i][0] + A[i][1] + ... + A[i][15])
            # We want: C[i][j] = A[i][j] * weight (not summed over k)

            # The only way to get C[i][j] = A[i][j] * weight with MFMA is:
            # B[j][j] = weight (diagonal), B[k][j] = 0 for k != j
            # Then C[i][j] = A[i][j] * weight

            # But as we discussed, we can't construct a proper diagonal B for all j.

            # Alternative: Use different k_grp assignments
            # Since lane l computes C for columns 4*k_grp + t, and lane l writes B[k][row_idx]
            # for k = 4*k_grp + t, we have:
            # Lane 0: writes B[0..3][0], computes C[0][0..3]
            # Lane 1: writes B[0..3][1], computes C[1][0..3]
            # ...
            # Lane 16: writes B[4..7][0], computes C[0][4..7]
            #
            # For C[0][0] = A[0][0] * weight:
            #   B[0][0] = weight (lane 0 can write this)
            # For C[0][4] = A[0][4] * weight:
            #   B[4][0] = weight (lane 16 writes this)
            # For C[0][8] = A[0][8] * weight:
            #   B[8][0] = weight (lane 32 writes this)
            # For C[0][12] = A[0][12] * weight:
            #   B[12][0] = weight (lane 48 writes this)
            #
            # For C[0][1] = A[0][1] * weight:
            #   B[1][0] = 0 (lane 0 can write B[1][0])
            #   But we want B[1][1] = weight, and lane 1 writes B[1][1]
            #
            # Wait! B[1][1] means lane 1 with k=1. Lane 1 has k_grp=0, k values 0..3.
            # Lane 1 can write B[1][1] (k=1, row_idx=1). Check: k==row_idx? 1==1? Yes!
            #
            # So the diagonal check does work for some cases.
            # Let me re-examine:
            # For lane l with row_idx = l % 16, k_grp = l // 16:
            #   Lane 0: row_idx=0, k_grp=0, k=0..3. Can write B[0][0]. ✓
            #   Lane 1: row_idx=1, k_grp=0, k=0..3. Can write B[1][1]. ✓
            #   Lane 2: row_idx=2, k_grp=0, k=0..3. Can write B[2][2]. ✓
            #   Lane 3: row_idx=3, k_grp=0, k=0..3. Can write B[3][3]. ✓
            #   Lane 4: row_idx=4, k_grp=0, k=0..3. Can write B[0..3][4]. But B[4][4] requires k=4!
            #   Lane 20: row_idx=4, k_grp=1, k=4..7. Can write B[4][4]. ✓
            #
            # So diagonal elements B[j][j] are written by lane (j//4)*16 + j.
            # And that lane computes C[j % 16][4*(j//4) + t] = C[j%16][j + t - (j%4)]
            #
            # For j=4: Lane 20 writes B[4][4], computes C[4][4..7].
            #   C[4][4] = A[4][4] * B[4][4] = A[4][4] * weight. ✓
            #   C[4][5] = A[4][5] * B[5][5] = A[4][5] * weight. (if B[5][5] is set)
            #
            # For B[5][5]: Lane 21 writes it. Lane 21 has row_idx=5, k_grp=1, k=4..7.
            #   B[5][5] is written when k=5, row_idx=5. k=5 is in 4..7. ✓
            #
            # So the diagonal matrix DOES work!
            # The issue was my earlier analysis was wrong.

            # Set diagonal B
            for t in S.range(4):
                k = 4 * k_grp + t
                if k == row_idx:
                    weight_shmem_f16[lane, t] = w_f16
                else:
                    weight_shmem_f16[lane, t] = S.convert(0.0, S.f16)
            for t in S.range(4, 8):
                weight_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            S.syncthreads()

            a_frag = S.view(input_shmem[lane], S.Tensor((2, 4, 1), S.f16))
            b_frag = S.view(weight_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag[0], acc)

    # Store output
    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running depthwise simple MFMA kernel...")
    try:
        depthwise_simple_mfma[launch_config](x, w, y)
        print('Kernel completed!')

        ref_full = F.conv2d(x.float(), w.float()).half()
        ref = ref_full[:, :, :16, :16]

        diff = torch.abs(y - ref)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f'Max diff: {max_diff}')
        print(f'Mean diff: {mean_diff}')

        print(f'Y[0,0,0,:4]: {y[0,0,0,:4]}')
        print(f'Ref[0,0,0,:4]: {ref[0,0,0,:4]}')

        print(f'Y[0,0,4,:4]: {y[0,0,4,:4]}')
        print(f'Ref[0,0,4,:4]: {ref[0,0,4,:4]}')

        if torch.allclose(y, ref, rtol=1e-2, atol=0.1):
            print('SUCCESS!')
        else:
            print('MISMATCH')
    except Exception as e:
        import traceback
        traceback.print_exc()
