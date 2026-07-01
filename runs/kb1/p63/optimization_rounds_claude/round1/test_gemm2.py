import torch
import substrate
import substrate.language as S

# Simple 32x32x8 GEMM test
M = 32
N = 32
K = 8
WARP_SIZE = 64

def permute_row(row: int) -> int:
    high = (row >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    return (row & 0x3) | (rotated << 2)

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

    # Row permutation for A operand
    high = (lane_col >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    permuted_row = (lane_col & 0x3) | (rotated << 2)

    a_row = permuted_row
    b_col = lane_col

    # Load 4 bf16 values into a local buffer
    # Use S.make_local to create register storage
    a_frag_local = S.make_local((4,), S.bf16)
    b_frag_local = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k = lane_k_base + e
        a_frag_local[e] = A[a_row, k]
        b_frag_local[e] = B[k, b_col]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_local, b_frag_local, acc)

    # Writeback
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
    print(f'actual[0,0]: {actual[0,0]}')
    print(f'expected[0,0]: {expected_cpu[0,0]}')

    # Debug: check specific rows
    print(f'\nRow 0: actual={actual[0, :4]} expected={expected_cpu[0, :4]}')
    print(f'Row 4: actual={actual[4, :4]} expected={expected_cpu[4, :4]}')
