import torch
import substrate
import substrate.language as S

# Test with 2x2 MFMA tiles (like the full kernel)
# 64x64 output per warp, split into 2x2 array of 32x32 MFMA tiles

M_FLAT = 64
OUT_CHANNELS = 64
K_FLAT = 16
KERNEL_AREA = 4  # 2x2 kernel for simplicity

WARP_SIZE = 64
NUM_WARPS = 1

@substrate.jit
def gemm_2x2_kernel(
    A: S.Tensor((64, 16), S.f32),  # Input A
    B: S.Tensor((64, 16), S.f32),  # Weights B (transposed for simplicity)
    C: S.Tensor((64, 64), S.f32),  # Output C
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Accumulators for 2x2 MFMA tiles
    acc00 = S.full((16,), 0.0, S.f32)
    acc01 = S.full((16,), 0.0, S.f32)
    acc10 = S.full((16,), 0.0, S.f32)
    acc11 = S.full((16,), 0.0, S.f32)

    # K tiles: K_FLAT = 16, MFMA_TILE_K = 8, so 2 tiles
    for k_tile in S.range(2):
        a_frag0 = S.make_local((4,), S.bf16)
        a_frag1 = S.make_local((4,), S.bf16)
        b_frag0 = S.make_local((4,), S.bf16)
        b_frag1 = S.make_local((4,), S.bf16)

        for e in S.range(4):
            a_frag0[e] = S.convert(0.0, S.bf16)
            a_frag1[e] = S.convert(0.0, S.bf16)
            b_frag0[e] = S.convert(0.0, S.bf16)
            b_frag1[e] = S.convert(0.0, S.bf16)

        # Apply permutation for MFMA input layout
        # Lane l should have A data for row permute_row(l % 32)
        high = (lane_col >> 2) & 0x7
        rotated = ((high & 0x1) << 2) | (high >> 1)
        permuted_row = (lane_col & 0x3) | (rotated << 2)

        # For A: permuted row (for MFMA layout)
        m0 = permuted_row  # Row 0-31 (permuted)
        m1 = permuted_row + 32  # Row 32-63 (permuted)

        # For B: no permutation needed for columns
        n0 = lane_col  # Column 0-31
        n1 = lane_col + 32  # Column 32-63

        for e in S.range(4):
            k_idx = k_tile * 8 + lane_k_base + e

            # Load A fragments
            if m0 < M_FLAT:
                a_frag0[e] = S.convert(A[m0, k_idx], S.bf16)
            if m1 < M_FLAT:
                a_frag1[e] = S.convert(A[m1, k_idx], S.bf16)

            # Load B fragments
            if n0 < OUT_CHANNELS:
                b_frag0[e] = S.convert(B[n0, k_idx], S.bf16)
            if n1 < OUT_CHANNELS:
                b_frag1[e] = S.convert(B[n1, k_idx], S.bf16)

        # MFMA operations
        acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc00)
        acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag1, acc01)
        acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag0, acc10)
        acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc11)

    # Writeback
    for acc_idx in S.range(16):
        row0 = 16 * (lane // 32) + acc_idx  # Row within first 32 rows
        row1 = 32 + 16 * (lane // 32) + acc_idx  # Row within second 32 rows
        col0 = lane_col  # Column within first 32 columns
        col1 = lane_col + 32  # Column within second 32 columns

        if row0 < M_FLAT and col0 < OUT_CHANNELS:
            C[row0, col0] = acc00[acc_idx]
        if row0 < M_FLAT and col1 < OUT_CHANNELS:
            C[row0, col1] = acc01[acc_idx]
        if row1 < M_FLAT and col0 < OUT_CHANNELS:
            C[row1, col0] = acc10[acc_idx]
        if row1 < M_FLAT and col1 < OUT_CHANNELS:
            C[row1, col1] = acc11[acc_idx]


if __name__ == "__main__":
    A = torch.randn((64, 16), dtype=torch.float32, device='cuda')
    B = torch.randn((64, 16), dtype=torch.float32, device='cuda')
    C = torch.zeros((64, 64), dtype=torch.float32, device='cuda')

    # Expected: C = A @ B.T
    expected = A @ B.T

    gemm_2x2_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    if max_diff > 0.1:
        print(f'\nFirst row comparison:')
        print(f'actual[0]: {actual[0, :8]}')
        print(f'expected[0]: {expected_cpu[0, :8]}')
        print(f'\nRow 32 comparison:')
        print(f'actual[32]: {actual[32, :8]}')
        print(f'expected[32]: {expected_cpu[32, :8]}')
