import torch
import substrate
import substrate.language as S

# Test 2x2 MFMA tiles WITHOUT permutation

M = 64
N = 64
K = 8
WARP_SIZE = 64

@substrate.jit
def gemm_2x2_no_perm_kernel(
    A: S.Tensor((64, 8), S.f32),
    B: S.Tensor((64, 8), S.f32),
    C: S.Tensor((64, 64), S.f32),
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

    a_frag0 = S.make_local((4,), S.bf16)
    a_frag1 = S.make_local((4,), S.bf16)
    b_frag0 = S.make_local((4,), S.bf16)
    b_frag1 = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e

        # NO permutation
        m0 = lane_col
        m1 = lane_col + 32
        n0 = lane_col
        n1 = lane_col + 32

        a_frag0[e] = S.convert(A[m0, k_idx], S.bf16)
        a_frag1[e] = S.convert(A[m1, k_idx], S.bf16)
        b_frag0[e] = S.convert(B[n0, k_idx], S.bf16)
        b_frag1[e] = S.convert(B[n1, k_idx], S.bf16)

    # MFMA operations
    acc00 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag0, acc00)
    acc01 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag1, acc01)
    acc10 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag0, acc10)
    acc11 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag1, acc11)

    # Writeback
    for acc_idx in S.range(16):
        row0 = 16 * (lane // 32) + acc_idx
        row1 = 32 + 16 * (lane // 32) + acc_idx
        col0 = lane_col
        col1 = lane_col + 32

        C[row0, col0] = acc00[acc_idx]
        C[row0, col1] = acc01[acc_idx]
        C[row1, col0] = acc10[acc_idx]
        C[row1, col1] = acc11[acc_idx]


if __name__ == "__main__":
    A = torch.randn((M, K), dtype=torch.float32, device='cuda')
    B = torch.randn((N, K), dtype=torch.float32, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    expected = A @ B.T

    gemm_2x2_no_perm_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'2x2 tile without permutation:')
    print(f'  Max diff: {max_diff}')
    print(f'  Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    print(f'\nRow 0 vs Row 32:')
    print(f'  Are they equal? {torch.allclose(actual[0], actual[32])}')

    # Check each tile
    print(f'\nTile (0,0) - rows 0-31, cols 0-31:')
    print(f'  Max diff: {torch.max(torch.abs(actual[:32, :32] - expected_cpu[:32, :32])).item()}')

    print(f'\nTile (1,0) - rows 32-63, cols 0-31:')
    print(f'  Max diff: {torch.max(torch.abs(actual[32:, :32] - expected_cpu[32:, :32])).item()}')
