import torch
import substrate
import substrate.language as S

# Test single 32x32 MFMA first

M = 32
N = 32
K = 8
WARP_SIZE = 64

def permute_row(row: int) -> int:
    high = (row >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    return (row & 0x3) | (rotated << 2)

@substrate.jit
def gemm_single_kernel(
    A: S.Tensor((32, 8), S.f32),
    B: S.Tensor((32, 8), S.f32),
    C: S.Tensor((32, 32), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    acc = S.full((16,), 0.0, S.f32)

    # Apply permutation for MFMA input layout
    high = (lane_col >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    permuted_row = (lane_col & 0x3) | (rotated << 2)

    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = S.convert(A[permuted_row, k_idx], S.bf16)
        b_frag[e] = S.convert(B[lane_col, k_idx], S.bf16)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Writeback
    for acc_idx in S.range(16):
        row = 16 * (lane // 32) + acc_idx
        col = lane_col
        C[row, col] = acc[acc_idx]


if __name__ == "__main__":
    A = torch.randn((M, K), dtype=torch.float32, device='cuda')
    B = torch.randn((N, K), dtype=torch.float32, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    expected = A @ B.T

    gemm_single_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Single tile test:')
    print(f'  Max diff: {max_diff}')
    print(f'  Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')
