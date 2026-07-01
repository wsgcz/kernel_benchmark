import torch
import torch.nn as nn
import substrate
import substrate.language as S

INPUT0_SHAPE = (16, 64, 1024, 1024)
OUTPUT_SHAPE = (16, 128, 1024, 1024)
WEIGHT_SHAPE = (128, 64, 1, 1)

# MFMA tile dimensions
MFMA_M = 32
MFMA_N = 32
MFMA_K = 8
WARP_SIZE = 64


@substrate.jit
def _mfma_matmul_kernel(
    A: S.Tensor((WARP_SIZE, 2), S.u32),
    B: S.Tensor((WARP_SIZE, 2), S.u32),
    C: S.Tensor((WARP_SIZE, 16), S.f32),
):
    """Simple 32x32x8 bf16 MFMA operation."""
    lane = S.thread_id(0)

    c_lane = S.full((16,), 0.0, S.f32)

    m_a = S.view(A[lane], S.Tensor((1, 4, 1), S.bf16))
    m_b = S.view(B[lane], S.Tensor((1, 4, 1), S.bf16))

    c_lane = S.amdgpu.mfma_32x32x8_bf16_f32(m_a[0], m_b[0], c_lane)

    C[lane] = c_lane


def _run_mfma_tile(A_bf16, B_bf16, C_f32, m_offset, n_offset, M, N, K):
    """Run a single 32x32 MFMA tile with K=8."""
    warp_size = 64
    tile_m = 32
    tile_n = 32
    tile_k = 8  # MFMA_K

    # Only handle K=8 case (single MFMA iteration)
    # For K=64, we need to accumulate 8 MFMA operations

    # Prepare A fragment (warp_size x 4 bf16 = warp_size x 2 u32)
    A_shuffled = torch.zeros((warp_size, 4), dtype=torch.bfloat16, device=A_bf16.device)
    for lane in range(warp_size):
        i = lane % tile_m  # row in 32x32 tile
        k_block = lane // tile_m  # 0 or 1 for k=8 (each lane gets 4 elements)
        for t in range(4):
            k = k_block * 4 + t
            global_i = m_offset + i
            global_k = k
            if global_i < M and global_k < K:
                A_shuffled[lane, t] = A_bf16[global_i, global_k]

    # Prepare B fragment (warp_size x 4 bf16 = warp_size x 2 u32)
    B_shuffled = torch.zeros((warp_size, 4), dtype=torch.bfloat16, device=B_bf16.device)
    for lane in range(warp_size):
        k_block = lane // tile_n  # 0 or 1 for k=8
        j = lane % tile_n  # col in 32x32 tile
        for t in range(4):
            k = k_block * 4 + t
            global_j = n_offset + j
            global_k = k
            if global_j < N and global_k < K:
                B_shuffled[lane, t] = B_bf16[global_j, global_k]

    # Pack into u32 for MFMA
    A_packed = A_shuffled.view(torch.int32).view(warp_size, 2)
    B_packed = B_shuffled.view(torch.int32).view(warp_size, 2)
    C_shuffled = torch.zeros((warp_size, 16), dtype=torch.float32, device=C_f32.device)

    # Run MFMA kernel
    _mfma_matmul_kernel[lambda: ((1, 1, 1), (warp_size, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unpack results
    for lane in range(warp_size):
        g = lane // tile_n
        j = lane % tile_n
        for t in range(4):
            i_offset = t
            global_i = m_offset + g * 4 + i_offset
            global_j = n_offset + j
            if global_i < M and global_j < N:
                C_f32[global_i, global_j] = C_shuffled[lane, t]


def _permute_row(row: int) -> int:
    """Permute row index for MFMA 32x32x8 lane layout."""
    high = (row >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    return (row & 0x3) | (rotated << 2)


def _run_mfma_tile_k64(A_bf16, B_bf16, C_f32, m_offset, n_offset, M, N, K):
    """Run a 32x32 MFMA tile accumulating over K."""
    warp_size = 64
    tile_m = 32
    tile_n = 32
    tile_k = 8

    # Accumulate C locally for this tile
    C_tile = torch.zeros((tile_m, tile_n), dtype=torch.float32, device=C_f32.device)

    # Loop over K in chunks of 8
    for k_chunk in range(0, K, tile_k):
        # Prepare A fragment for this k_chunk
        # Match the working test_debug.py logic
        A_shuffled = torch.zeros((warp_size, 4), dtype=torch.bfloat16, device=A_bf16.device)
        for lane in range(warp_size):
            i = _permute_row(lane % tile_m)
            k_block = lane // tile_m
            for t in range(4):
                kk = k_block * 4 + t
                global_i = m_offset + i
                global_k = k_chunk + kk
                if global_i < M and global_k < K:
                    A_shuffled[lane, t] = A_bf16[global_i, global_k]

        # Prepare B fragment
        B_shuffled = torch.zeros((warp_size, 4), dtype=torch.bfloat16, device=B_bf16.device)
        for lane in range(warp_size):
            k_block = lane // tile_n
            j = lane % tile_n
            for t in range(4):
                kk = k_block * 4 + t
                global_k = k_chunk + kk
                global_j = n_offset + j
                if global_k < K and global_j < N:
                    B_shuffled[lane, t] = B_bf16[global_k, global_j]

        # Pack and run MFMA
        A_packed = A_shuffled.view(torch.int32).view(warp_size, 2)
        B_packed = B_shuffled.view(torch.int32).view(warp_size, 2)
        C_shuffled = torch.zeros((warp_size, 16), dtype=torch.float32, device=C_f32.device)

        _mfma_matmul_kernel[lambda: ((1, 1, 1), (warp_size, 1, 1))](A_packed, B_packed, C_shuffled)

        # Unpack and accumulate
        for lane in range(warp_size):
            g = lane // tile_n
            j = lane % tile_n
            for t in range(16):
                local_i = g * 16 + t
                if local_i < tile_m:
                    C_tile[local_i, j] += C_shuffled[lane, t]

    # Copy tile results to global C
    for i in range(tile_m):
        for j in range(tile_n):
            global_i = m_offset + i
            global_j = n_offset + j
            if global_i < M and global_j < N:
                C_f32[global_i, global_j] = C_tile[i, j]


class ModelNew(nn.Module):
    """MFMA-optimized 1x1 Conv2D kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Create weight parameter (OIHW format)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels = self.out_channels

        # Convert to bfloat16
        x_bf16 = x.to(dtype=torch.bfloat16)

        # For 1x1 conv2d, we need to reshape input to (M, K) where:
        # M = batch * H * W (number of spatial positions)
        # K = in_channels
        # A[i, k] = input[b, k, h, w] where i = b * H * W + h * W + w
        # This requires permuting from NCHW to NHWC first
        x_nhwc = x_bf16.permute(0, 2, 3, 1)  # (batch, H, W, C)
        M = batch_size * in_h * in_w
        K = in_channels
        N = out_channels

        A = x_nhwc.reshape(M, K)  # (M, K)

        # Prepare weight as (K, N) - transposed for MFMA
        # Weight is stored as (N, K, 1, 1), we need (K, N) for MFMA
        w_bf16 = self.weight.to(dtype=torch.bfloat16, device=x.device)
        w_bf16_2d = w_bf16.squeeze(-1).squeeze(-1)  # (N, K)
        B = w_bf16_2d.T.contiguous()  # (K, N)

        # Output in float32
        C = torch.zeros((M, N), dtype=torch.float32, device=x.device)

        # Run MFMA tiles
        tile_size = 32
        for m_tile in range(0, M, tile_size):
            for n_tile in range(0, N, tile_size):
                _run_mfma_tile_k64(A, B, C, m_tile, n_tile, M, N, K)

        # Reshape output to NCHW
        # C[i, n] = output[b, n, h, w] where i = b * H * W + h * W + w
        C_nhwc = C.reshape(batch_size, in_h, in_w, out_channels)
        output = C_nhwc.permute(0, 3, 1, 2)  # (batch, C_out, H, W)

        return output
