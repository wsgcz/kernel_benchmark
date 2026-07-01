import torch
import substrate
import substrate.language as S

# Test single MFMA with A1 data only

M = 32
N = 32
K = 8
WARP_SIZE = 64

@substrate.jit
def single_mfma_a1_kernel(
    A1: S.Tensor((32, 8), S.bf16),
    B: S.Tensor((8, 32), S.bf16),
    C1: S.Tensor((32, 32), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    acc = S.full((16,), 0.0, S.f32)
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = A1[lane_col, k_idx]
        b_frag[e] = B[k_idx, lane_col]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    for acc_idx in S.range(16):
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col
        C1[row, col] = acc[acc_idx]


if __name__ == "__main__":
    A1 = torch.randn((32, 8), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((8, 32), dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    expected = A1.float() @ B.float()

    single_mfma_a1_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A1, B, C1)

    actual = C1.cpu()
    expected_cpu = expected.cpu()

    diff = torch.max(torch.abs(actual - expected_cpu)).item()

    print(f'Single MFMA with A1 only:')
    print(f'  max_diff={diff:.6f}, pass={torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')
