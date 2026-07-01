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
    A: S.Tensor((64, 2), S.u32),  # Packed as (64, 2) u32 = 64 lanes x 4 bf16
    B: S.Tensor((64, 2), S.u32),  # Packed as (64, 2) u32 = 64 lanes x 4 bf16
    C: S.Tensor((64, 16), S.f32), # Output as (64, 16) f32
):
    lane = S.thread_id(0)

    acc = S.full((16,), 0.0, S.f32)

    # View A[lane] and B[lane] as (1, 4, 1) bf16
    a_frag = S.view(A[lane], S.Tensor((1, 4, 1), S.bf16))
    b_frag = S.view(B[lane], S.Tensor((1, 4, 1), S.bf16))

    # MFMA instruction
    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag[0], b_frag[0], acc)

    # Writeback
    C[lane] = acc


if __name__ == "__main__":
    # Test - following the working test's data packing
    m = 32
    n = 32
    k = 8
    warp_size = 64

    A = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((k, n), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((m, n), dtype=torch.float32, device='cuda')

    expected = (A @ B).to(dtype=torch.float32, device='cpu')

    frag_size = k // (warp_size // m)

    # Pre-shuffle A
    A_shuffled = torch.zeros((warp_size, frag_size), dtype=torch.bfloat16, device='cuda')
    for lane in range(warp_size):
        i = permute_row(lane % m)
        k_block = lane // m
        for t in range(frag_size):
            kk = k_block * frag_size + t
            A_shuffled[lane, t] = A[i, kk]

    # Pre-shuffle B
    B_shuffled = torch.zeros((warp_size, frag_size), dtype=torch.bfloat16, device='cuda')
    for lane in range(warp_size):
        k_block = lane // n
        j = lane % n
        for t in range(frag_size):
            kk = k_block * frag_size + t
            B_shuffled[lane, t] = B[kk, j]

    # Pack to u32
    A_packed = A_shuffled.view(torch.int32).view(warp_size, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(warp_size, frag_size // 2)
    C_shuffled = torch.zeros((warp_size, 16), dtype=torch.float32, device='cuda')

    gemm_kernel[lambda: ((1, 1, 1), (warp_size, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unpack output
    row_block = m // (warp_size // m)
    for lane in range(warp_size):
        g = lane // n
        j = lane % n
        for t in range(row_block):
            C[row_block * g + t, j] = C_shuffled[lane, t]

    actual = C.to('cpu')

    max_diff = torch.max(torch.abs(actual - expected)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected, rtol=1e-2, atol=1e-3)}')
    print(f'actual[0,0]: {actual[0,0]}')
    print(f'expected[0,0]: {expected[0,0]}')
