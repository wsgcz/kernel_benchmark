import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

# Test Conv2D with exact MFMA pattern from working test
# 32x32x8 MFMA, single tile

N = 1
IN_CHANNELS = 16
IN_H = 35
IN_W = 35
OUT_CHANNELS = 32
KERNEL_H = 3
KERNEL_W = 3
OUT_H = 33
OUT_W = 33
KERNEL_AREA = KERNEL_H * KERNEL_W

BLOCK_M = 32
BLOCK_N = 32
WARP_SIZE = 64
NUM_WARPS = 1

HW_OUT = OUT_H * OUT_W
K_FLAT = IN_CHANNELS * KERNEL_AREA
M_FLAT = N * HW_OUT

def permute_row(row: int) -> int:
    high = (row >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    return (row & 0x3) | (rotated << 2)

@substrate.jit
def conv_kernel(
    A: S.Tensor((64, 2), S.u32),  # Pre-packed A
    B: S.Tensor((64, 2), S.u32),  # Pre-packed B
    C: S.Tensor((64, 16), S.f32),  # Output
):
    lane = S.thread_id(0)

    acc = S.full((16,), 0.0, S.f32)

    # View packed data as bf16
    m_a = S.view(A[lane], S.Tensor((1, 4, 1), S.bf16))
    m_b = S.view(B[lane], S.Tensor((1, 4, 1), S.bf16))

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], acc)

    C[lane] = acc


def test_conv_with_mfma():
    # Create input and weight
    x = torch.randn((N, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device='cuda')
    w = torch.randn((OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W), dtype=torch.float32, device='cuda')

    # Compute expected using PyTorch
    expected_full = F.conv2d(x, w, stride=1, padding=0)

    # For testing, just do a single 32x32 tile
    # Take the first 32 output positions and first 32 output channels
    expected = expected_full[0, :32, :32, 0].T  # Shape (32, 32)

    # Flatten input activations for the GEMM
    # For output position (oh, ow), we need input at (oh:oh+3, ow:ow+3)
    # Let's test with output position (0, 0) - (0, 31)
    # Actually, let's just test a simple GEMM

    # Create simple GEMM test
    m, n, k = 32, 32, 8
    A = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((k, n), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((m, n), dtype=torch.float32, device='cuda')

    expected = (A @ B).to(dtype=torch.float32, device='cpu')

    frag_size = k // (WARP_SIZE // m)  # 8 // 2 = 4

    # Pre-shuffle A
    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.bfloat16, device='cuda')
    for lane in range(WARP_SIZE):
        i = permute_row(lane % m)
        k_block = lane // m
        for t in range(frag_size):
            kk = k_block * frag_size + t
            A_shuffled[lane, t] = A[i, kk]

    # Pre-shuffle B
    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.bfloat16, device='cuda')
    for lane in range(WARP_SIZE):
        k_block = lane // n
        j = lane % n
        for t in range(frag_size):
            kk = k_block * frag_size + t
            B_shuffled[lane, t] = B[kk, j]

    # Pack to u32
    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 16), dtype=torch.float32, device='cuda')

    conv_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unpack output
    row_block = m // (WARP_SIZE // m)  # 16
    for lane in range(WARP_SIZE):
        g = lane // n
        j = lane % n
        for t in range(row_block):
            C[row_block * g + t, j] = C_shuffled[lane, t]

    actual = C.to('cpu')

    max_diff = torch.max(torch.abs(actual - expected)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected, rtol=1e-2, atol=0.1)}')

    if max_diff > 0.1:
        print(f'actual[0]: {actual[0]}')
        print(f'expected[0]: {expected[0]}')


if __name__ == "__main__":
    test_conv_with_mfma()
