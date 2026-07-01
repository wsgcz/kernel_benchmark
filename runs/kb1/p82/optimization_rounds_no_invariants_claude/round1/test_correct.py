#!/usr/bin/env python3
"""
Correct MFMA depthwise convolution kernel.

Key insight: Use MFMA K dimension for kernel positions, not spatial columns.

For depthwise conv: Y[o0][o1] = Σ_{k0,k1} X[o0+k0][o1+k1] * W[k0][k1]

Map to MFMA: C[i][j] = Σ_k A[i][k] * B[k][j]
where:
- i = output row
- j = output column
- k = kernel position index (0..8 for 3x3 kernel)
- A[i][k] = X[i + k0_k][j + k1_k] (input value at kernel position)
- B[k][j] = W[k0_k][k1_k] (weight, same for all j)

But wait - MFMA expects A[i][k] where k is summed over.
For A[i][k], lane l holds A[l%16][4*(l//16)+t].
So each lane can only access 4 specific k values.

With 9 kernel positions, we need all lanes to contribute.
Lanes 0-15: k=0..3
Lanes 16-31: k=4..7
Lanes 32-47: k=8..11 (but we only need k=8)
Lanes 48-63: k=12..15 (unused, pad with 0)

For B[k][j], lane l holds B[k][l%16] where k=4*(l//16)+t.
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def correct_mfma_kernel(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Correct MFMA kernel using K dimension for kernel positions."""
    lane = S.thread_id(0)

    row_idx = lane % 16      # i = output row
    col_idx = lane % 16      # j = output column (same as row_idx)
    k_grp = lane // 16       # which group of k values (0, 1, 2, 3)

    # Shared memory for MFMA fragments
    input_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    weight_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    input_shmem_f16 = S.view(input_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    weight_shmem_f16 = S.view(weight_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    # Accumulator for output
    acc = S.full((4,), 0.0, S.f32)

    # Kernel positions mapping: k_pos = k_grp * 4 + t
    # k_pos 0..8 map to kernel (k0, k1) as:
    # 0 -> (0,0), 1 -> (0,1), 2 -> (0,2), 3 -> (1,0), 4 -> (1,1)
    # 5 -> (1,2), 6 -> (2,0), 7 -> (2,1), 8 -> (2,2)

    # For A matrix: A[i][k] = X[i + k0_k][j + k1_k]
    # Lane l holds A[row_idx][4*k_grp + t] where j is the output column
    # We need to load X[row_idx + k0][j + k1] into A[row_idx][k_pos]
    # But j varies per output element! Each lane handles 4 different j values.

    # Actually, for output C[i][j], lane l holds C[i][4*k_grp + t] for fixed i.
    # So lane l computes multiple output columns j = 4*k_grp + t.

    # Let me reconsider:
    # Lane l stores C[l % 16][4 * (l // 16) + t]
    # So lane l computes output positions (row=l%16, col=4*(l//16)+t)

    # For each output position (i, j), C[i][j] = Σ_k A[i][k] * B[k][j]
    # A[i][k] should be X[i + k0_k][j + k1_k]
    # B[k][j] should be W[k0_k][k1_k]

    # But A[i][k] is independent of j, and we're computing C for multiple j values!
    # This is the issue - A[i][k] depends on j, but lane l computes C for multiple j.

    # Solution: For each (k0, k1) kernel position, compute a separate MFMA:
    # A[i][j] = X[i+k0][j+k1], B[j][j] = W[k0][k1] (diagonal)
    # Then C[i][j] = A[i][j] * W[k0][k1]
    # Accumulate over all kernel positions.

    # This is what I had before, but let me check the lane layout again.

    # Load A[i][j] = X[i+k0][j+k1]
    # Lane l holds A[l%16][4*(l//16)+t]
    # So j = 4*k_grp + t (but k_grp = l//16)
    # Wait, this means j = 4*(l//16) + t, and i = l%16

    # For MFMA, C[i][j] = Σ_k A[i][k] * B[k][j]
    # If B is diagonal with B[j][j] = weight:
    # C[i][j] = A[i][j] * weight

    # But lane l only stores A[l%16][4*(l//16)+t], not A[l%16][j] for all j!
    # So with diagonal B, we can only compute C[l%16][4*(l//16)+t], which is what lane l computes anyway.

    # So the approach should work. Let me fix the implementation.

    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load A[row_idx][j] = X[row_idx + k0][j + k1]
            # Lane l stores A[row_idx][4*k_grp + t] where j = 4*k_grp + t
            for t in S.range(4):
                j = 4 * k_grp + t  # output column
                i0 = row_idx + k0
                i1 = j + k1

                input_shmem_f16[lane, t] = X[0, 0, i0, i1]
                input_shmem_f16[lane, t + 4] = input_shmem_f16[lane, t]

            # Set diagonal B[j][j] = weight
            # Lane l holds B[k][col_idx] where k = 4*k_grp + t
            # For diagonal, need k == col_idx (which equals row_idx)
            for t in S.range(4):
                k = 4 * k_grp + t
                # B[k][j] where j = col_idx = row_idx
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

    # Store output: lane l writes Y[row_idx][4*k_grp + t]
    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running correct MFMA kernel...")
    try:
        correct_mfma_kernel[launch_config](x, w, y)
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

        if torch.allclose(y, ref, rtol=1e-2, atol=0.1):
            print('SUCCESS!')
        else:
            print('MISMATCH')
    except Exception as e:
        import traceback
        traceback.print_exc()
