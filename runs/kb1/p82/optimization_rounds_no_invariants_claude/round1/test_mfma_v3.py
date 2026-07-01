#!/usr/bin/env python3
"""
Follow the reference GEMM pattern exactly for depthwise conv.

In the reference test for 16x32 @ 32x16 GEMM:
- A[i][k] is packed with lane l holding A[l%16][k_block*8+t]
- B[k][j] is packed with lane l holding B[k_block*8+t][l%16]

For depthwise conv, we want:
- A[i][k] = X[i+k0_k][j+k1_k] ... but this depends on j!
- B[k][j] = W[k0_k][k1_k]

The issue: A[i][k] should be the input value at position (i + k0_k, j + k1_k).
But j is the output column, which varies.

Alternative: Don't use MFMA's K dimension for kernel positions.
Instead, process each kernel position separately and use the "im2col" approach:
- For each kernel position (k0, k1):
  - A[i][j] = X[i+k0][j+k1] (16x16 input patch)
  - B = identity * weight (scalar)
  - C[i][j] = A[i][j] * weight (element-wise multiply)

To get element-wise multiply from MFMA:
- A[i][k] = X[i+k0][k+k1] (map output column j to k)
- B[k][j] = weight if k == j, else 0 (diagonal)

This is what we tried before, but let me be more careful about the lane layout.
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def mfma_im2col_kernel(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """
    MFMA kernel using im2col approach.

    For each kernel position:
    - A[i][k] = X[i+k0][k+k1] (input at position i+k0, k+k1)
    - B[k][j] = weight if k == j (diagonal matrix)
    - C[i][j] = A[i][j] * weight (element-wise after sum over k collapses to single term)
    """
    lane = S.thread_id(0)

    row_idx = lane % 16  # i for A, j for B
    k_grp = lane // 16

    # Shared memory (must use u32 for MFMA compatibility)
    A_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    B_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    A_shmem_f16 = S.view(A_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    B_shmem_f16 = S.view(B_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    acc = S.full((4,), 0.0, S.f32)

    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load A: A[i][k] = X[i+k0][k+k1]
            # Lane l holds A[row_idx][4*k_grp + t] for k = 4*k_grp + t
            for t in S.range(4):
                k = 4 * k_grp + t
                i0 = row_idx + k0
                i1 = k + k1  # Use k as column index
                A_shmem_f16[lane, t] = X[0, 0, i0, i1]
                A_shmem_f16[lane, t + 4] = A_shmem_f16[lane, t]

            # Load B: B[k][j] = weight if k == j
            # Lane l holds B[k][row_idx] for k = 4*k_grp + t
            # Diagonal: B[k][j] = weight if k == j
            # For lane l with row_idx = j: set B[k][j] = weight if k == j
            for t in S.range(4):
                k = 4 * k_grp + t
                j = row_idx
                # Diagonal: k == j
                if k == j:
                    B_shmem_f16[lane, t] = w_f16
                else:
                    B_shmem_f16[lane, t] = S.convert(0.0, S.f16)
            for t in S.range(4, 8):
                B_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            S.syncthreads()

            a_frag = S.view(A_shmem[lane], S.Tensor((2, 4, 1), S.f16))
            b_frag = S.view(B_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag[0], acc)

    # Store output
    # Lane l holds C[row_idx][4*k_grp + t]
    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running MFMA im2col kernel...")
    try:
        mfma_im2col_kernel[launch_config](x, w, y)
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
            # Debug: print first row
            print(f'Y[0,0,0,:]: {y[0,0,0,:]}')
            print(f'Ref[0,0,0,:]: {ref[0,0,0,:]}')
    except Exception as e:
        import traceback
        traceback.print_exc()
