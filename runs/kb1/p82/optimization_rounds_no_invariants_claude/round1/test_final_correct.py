#!/usr/bin/env python3
"""
Correct MFMA depthwise conv based on proper understanding of lane layout.

From reference test output unpacking:
- Lane l stores C[4*(l//16) + t][l%16]
- Lane 0: C[0][0], C[1][0], C[2][0], C[3][0] (column 0, rows 0-3)
- Lane 1: C[0][1], C[1][1], C[2][1], C[3][1] (column 1, rows 0-3)
- Lane 16: C[4][0], C[5][0], C[6][0], C[7][0] (column 0, rows 4-7)

So each lane computes ONE COLUMN of C, for multiple rows.

For A matrix:
- Lane l holds A[l%16][4*(l//16) + t]
- Lane 0: A[0][0], A[0][1], A[0][2], A[0][3] (row 0, cols 0-3)
- Lane 1: A[1][0], A[1][1], A[1][2], A[1][3] (row 1, cols 0-3)

For B matrix:
- Lane l holds B[4*(l//16) + t][l%16]
- Lane 0: B[0][0], B[1][0], B[2][0], B[3][0] (column 0, rows 0-3)
- Lane 1: B[0][1], B[1][1], B[2][1], B[3][1] (column 1, rows 0-3)

MFMA computes C = A * B:
C[i][j] = Σ_k A[i][k] * B[k][j]

For depthwise conv:
C[i][j] = Σ_k A[i][k] * B[k][j] = Σ_k X[i+k0_k][j+k1_k] * W[k0_k][k1_k]

But with MFMA, A[i][k] and B[k][j] are distributed across lanes.
Each lane contributes part of the sum.

For diagonal B (B[k][j] = weight if k==j):
C[i][j] = A[i][j] * weight

Lane l computes C[:][j] for j = l % 16 (one column).
But lane l has A[l%16][:] (one row, multiple columns) and B[:][l%16] (one column, multiple rows).

The MFMA combines all lanes' A and B to compute C.

For C[0][1]:
- Computed by lane 1 (j=1)
- Lane 1 has B[:][1] (column 1 of B)
- Lane 0 has A[0][:] (row 0 of A)
- C[0][1] = Σ_k A[0][k] * B[k][1]

For diagonal B, B[1][1] = weight, others 0.
C[0][1] = A[0][1] * B[1][1] = A[0][1] * weight

A[0][1] is in lane 0 (row 0, col 1).
B[1][1] is in lane 1 (col 1, row 1).

So the computation IS correct across lanes!

The issue was my understanding of the output distribution.
Lane l stores C[4*g + t][j] where g = l//16, j = l%16.

Let me fix the implementation.
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def mfma_correct_v2(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Correct MFMA with proper lane layout understanding."""
    lane = S.thread_id(0)

    # Lane layout:
    # - A: lane l holds A[l%16][4*(l//16) + t] -> row = l%16, cols = 4*g + t
    # - B: lane l holds B[4*(l//16) + t][l%16] -> rows = 4*g + t, col = l%16
    # - C: lane l stores C[4*g + t][l%16] -> rows = 4*g + t, col = l%16

    row_idx = lane % 16       # Row index for A, column index for B and C
    col_idx = lane % 16       # Same as row_idx
    k_grp = lane // 16        # Row group for A columns, B rows, C rows

    A_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    B_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    A_shmem_f16 = S.view(A_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    B_shmem_f16 = S.view(B_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    # Accumulator for C[4*k_grp + t][col_idx]
    acc = S.full((4,), 0.0, S.f32)

    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load A: A[row_idx][k] = X[row_idx + k0][k + k1]
            # k = 4*k_grp + t (this is the column index in A)
            for t in S.range(4):
                k = 4 * k_grp + t  # K dimension index (also A column)
                i0 = row_idx + k0
                i1 = k + k1
                A_shmem_f16[lane, t] = X[0, 0, i0, i1]
                A_shmem_f16[lane, t + 4] = A_shmem_f16[lane, t]

            # Load B: B[k][col_idx] = weight if k == col_idx (diagonal)
            # k = 4*k_grp + t (this is the row index in B)
            for t in S.range(4):
                k = 4 * k_grp + t
                if k == col_idx:
                    B_shmem_f16[lane, t] = w_f16
                else:
                    B_shmem_f16[lane, t] = S.convert(0.0, S.f16)
            for t in S.range(4, 8):
                B_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            S.syncthreads()

            a_frag = S.view(A_shmem[lane], S.Tensor((2, 4, 1), S.f16))
            b_frag = S.view(B_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag[0], acc)

    # Store C: lane l stores C[4*k_grp + t][col_idx]
    for t in S.range(4):
        i = 4 * k_grp + t  # Row index in C
        j = col_idx         # Column index in C
        Y[0, 0, i, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running MFMA correct v2...")
    try:
        mfma_correct_v2[launch_config](x, w, y)
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
        print(f'Y[0,0,:4,0]: {y[0,0,:4,0]}')
        print(f'Ref[0,0,:4,0]: {ref[0,0,:4,0]}')

        if torch.allclose(y, ref, rtol=1e-2, atol=0.1):
            print('SUCCESS!')
        else:
            print('MISMATCH')
    except Exception as e:
        import traceback
        traceback.print_exc()
