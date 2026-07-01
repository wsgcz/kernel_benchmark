import torch
import substrate
import substrate.language as S

# Simple 32x32x8 GEMM test - NO PERMUTATION
M = 32
N = 32
K = 8
WARP_SIZE = 64

@substrate.jit
def gemm_kernel(
    A: S.Tensor((32, 8), S.bf16),
    B: S.Tensor((8, 32), S.bf16),
    C: S.Tensor((32, 32), S.f32),
):
    lane = S.thread_id(0)
    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    acc = S.full((16,), 0.0, S.f32)

    # NO permutation - direct row mapping
    a_row = lane_col
    b_col = lane_col

    # Load 4 bf16 values into local buffers
    a_frag_local = S.make_local((4,), S.bf16)
    b_frag_local = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k = lane_k_base + e
        a_frag_local[e] = A[a_row, k]
        b_frag_local[e] = B[k, b_col]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_local, b_frag_local, acc)

    # Writeback - standard MFMA output layout
    for acc_idx in S.range(16):
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col
        C[row, col] = acc[acc_idx]


if __name__ == "__main__":
    # Test
    A = torch.randn((32, 8), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((8, 32), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    expected = (A.float() @ B.float())

    gemm_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    # Check which rows match
    for r in range(0, 32, 4):
        row_diff = torch.max(torch.abs(actual[r] - expected_cpu[r])).item()
        print(f'Row {r} diff: {row_diff}')
