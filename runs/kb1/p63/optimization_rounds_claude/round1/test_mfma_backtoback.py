import torch
import substrate
import substrate.language as S

# Test two MFMA operations with different A data

M = 32
N = 32
K = 8
WARP_SIZE = 64

@substrate.jit
def two_mfma_kernel(
    A0: S.Tensor((32, 8), S.f32),
    A1: S.Tensor((32, 8), S.f32),
    B: S.Tensor((32, 8), S.f32),
    C0: S.Tensor((32, 32), S.f32),
    C1: S.Tensor((32, 32), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # First MFMA with A0
    acc0 = S.full((16,), 0.0, S.f32)
    a_frag0 = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag0[e] = S.convert(A0[lane_col, k_idx], S.bf16)
        b_frag[e] = S.convert(B[lane_col, k_idx], S.bf16)

    acc0 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0, b_frag, acc0)

    # Second MFMA with A1
    acc1 = S.full((16,), 0.0, S.f32)
    a_frag1 = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag1[e] = S.convert(A1[lane_col, k_idx], S.bf16)

    acc1 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1, b_frag, acc1)

    # Writeback
    for acc_idx in S.range(16):
        row = 16 * (lane // 32) + acc_idx
        col = lane_col
        C0[row, col] = acc0[acc_idx]
        C1[row, col] = acc1[acc_idx]


if __name__ == "__main__":
    A0 = torch.randn((32, 8), dtype=torch.float32, device='cuda')
    A1 = torch.randn((32, 8), dtype=torch.float32, device='cuda')
    B = torch.randn((32, 8), dtype=torch.float32, device='cuda')
    C0 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')
    C1 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    expected0 = A0 @ B.T
    expected1 = A1 @ B.T

    two_mfma_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A0, A1, B, C0, C1)

    actual0 = C0.cpu()
    actual1 = C1.cpu()
    expected0_cpu = expected0.cpu()
    expected1_cpu = expected1.cpu()

    diff0 = torch.max(torch.abs(actual0 - expected0_cpu)).item()
    diff1 = torch.max(torch.abs(actual1 - expected1_cpu)).item()

    print(f'Two MFMA test:')
    print(f'  C0 (A0 @ B.T): max_diff={diff0:.4f}, pass={torch.allclose(actual0, expected0_cpu, rtol=1e-2, atol=0.1)}')
    print(f'  C1 (A1 @ B.T): max_diff={diff1:.4f}, pass={torch.allclose(actual1, expected1_cpu, rtol=1e-2, atol=0.1)}')

    # Check if C0 and C1 are different
    print(f'\n  C0 == C1? {torch.allclose(actual0, actual1)}')
    print(f'  C0[0, :4] = {actual0[0, :4].tolist()}')
    print(f'  C1[0, :4] = {actual1[0, :4].tolist()}')
