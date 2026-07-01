"""MFMA-optimized depthwise Conv2D kernel.

This implementation uses MFMA 16x16x16 bf16->f32 instructions for the computation.
For depthwise convolution, each output channel corresponds to one input channel.

MFMA approach for depthwise conv:
- Each warp processes 16 positions × 16 channels
- Use multiple MFMA iterations to cover all 256 outputs
- Each MFMA computes a 16x16 matrix, we extract diagonal elements
- A[16x16]: input values for 16 positions across 16 kernel elements (padded)
- B[16x16]: weight values for 16 channels across 16 kernel elements (padded)
- C[m,n] = dot product for position m, channel n

Data layout for MFMA 16x16x16 bf16:
- Lane l provides A[l%16, k] and B[k, l%16] across K dimension
- Lane l holds C[4*(l//16) + t, l%16] for t=0..3
- Diagonal C[m,m] is extracted from lane m, accumulator index m%4
"""

import torch
import torch.nn as nn
import substrate
import substrate.language as S

# Constants for the fixed kernel shapes
BATCH = 64
CHANNELS = 128
IN_H = 256
IN_W = 512
OUT_H = 254
OUT_W = 510
K_H = 3
K_W = 3
K_ELEMS = K_H * K_W  # 9

# Warp size
WARP_SIZE = 64

# Thread configuration
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

# Output tile dimensions per CTA
TILE_M = 64  # Output positions per CTA (16 per warp * 4 warps)
TILE_N = 16  # Channels per CTA

# MFMA configuration
MFMA_K = 16  # Pad K=9 to K=16 for MFMA


def _launch():
    """Launch configuration for MFMA depthwise conv kernel."""
    total_outputs = BATCH * OUT_H * OUT_W
    m_tiles = (total_outputs + TILE_M - 1) // TILE_M
    n_tiles = (CHANNELS + TILE_N - 1) // TILE_N
    grid = (m_tiles * n_tiles, 1, 1)
    block = (THREADS, 1, 1)
    return lambda: (grid, block)


@substrate.jit
def mfma_depthwise_conv_kernel(
    X: S.Pointer(S.bf16),
    W: S.Pointer(S.bf16),
    Y: S.Pointer(S.bf16),
):
    """MFMA-based depthwise convolution kernel.

    Uses MFMA 16x16x16 bf16->f32 for dot product computation.

    Cooperative MFMA approach:
    - Each warp processes 16 positions × 16 channels = 256 outputs
    - Run 4 MFMA iterations per warp to cover all outputs
    - Each MFMA: lanes 0-15 cooperate on positions 0-15, channels q*4 to q*4+3
    - Lane l provides A[l%16, k] and B[k, l%16] across K dimension
    - Lane l accumulates C[4*(l//16) + t, l%16] for t=0..3

    For depthwise conv, we extract diagonal C[m,m] which gives dot product
    for position m, channel m (using the same index for position and channel).
    """
    total_outputs = BATCH * OUT_H * OUT_W
    hw_out = OUT_H * OUT_W

    # Create tensor views
    input_tensor = S.make_tensor(
        X, S.bf16,
        S.make_layout((BATCH, CHANNELS, IN_H, IN_W),
                      (CHANNELS * IN_H * IN_W, IN_H * IN_W, IN_W, 1))
    )
    weight_tensor = S.make_tensor(
        W, S.bf16,
        S.make_layout((CHANNELS, 1, K_H, K_W),
                      (K_H * K_W, K_H * K_W, K_W, 1))
    )
    output_tensor = S.make_tensor(
        Y, S.bf16,
        S.make_layout((BATCH, CHANNELS, OUT_H, OUT_W),
                      (CHANNELS * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1))
    )

    # Block identification
    linear_block_id = S.block_id(0)
    n_tiles = (CHANNELS + TILE_N - 1) // TILE_N
    tile_m_idx = linear_block_id // n_tiles
    tile_n_idx = linear_block_id - tile_m_idx * n_tiles

    # Thread identification
    tid = S.thread_id(0)
    wid = tid // WARP_SIZE
    lane = tid % WARP_SIZE

    # Position within warp for MFMA
    # For 16x16 MFMA, we use lanes 0-15, repeat 4 times
    mfma_lane = lane % 16
    mfma_group = lane // 16

    # Base coordinates for this CTA
    m_base = tile_m_idx * TILE_M
    n_base = tile_n_idx * TILE_N

    # Warp tiling - each warp processes 16 output positions
    outputs_per_warp = TILE_M // NUM_WARPS  # 16
    warp_m_start = m_base + wid * outputs_per_warp

    # Each warp does 4 iterations, each processing 4 channels
    # Lane l in iteration iter processes:
    # - Position: mfma_lane
    # - Channel: mfma_group + 4*iter (not used - we extract diagonal)
    for iter_idx in S.range(4):
        # For this iteration, each lane computes one output
        # Position index (0-15) based on mfma_lane
        pos_idx = mfma_lane

        # Channel index for storage: lane's position in the iteration
        # After MFMA, lane l will have C[4*mfma_group + t, mfma_lane]
        # We extract diagonal C[mfma_lane, mfma_lane] from the accumulator
        # Since we want position = channel, we use ch_idx = mfma_lane
        ch_idx = mfma_lane

        # Global position and channel
        m_idx = warp_m_start + pos_idx
        n_idx = n_base + ch_idx + iter_idx * 4

        # Decode position to (batch, oh, ow)
        batch_idx = m_idx // hw_out
        hw_rem = m_idx - batch_idx * hw_out
        oh_idx = hw_rem // OUT_W
        ow_idx = hw_rem - oh_idx * OUT_W

        # Bounds check
        in_bounds = m_idx < total_outputs and n_idx < CHANNELS

        # Load input values for A matrix
        # Lane l provides A[l%16, k] for position l%16
        a_frag = S.make_local((2, 4), S.bf16)  # 8 bf16 values for K=0..7 and K=8..15

        # Load weight values for B matrix
        # Lane l provides B[k, l%16] for channel l%16
        b_frag = S.make_local((2, 4), S.bf16)

        # Initialize to zero
        for i in S.range(2):
            for j in S.range(4):
                a_frag[i, j] = S.convert(0.0, S.bf16)
                b_frag[i, j] = S.convert(0.0, S.bf16)

        # Load K=9 kernel elements, pad to K=16
        for kh in S.range(K_H):
            for kw in S.range(K_W):
                k = kh * K_W + kw
                # Map k to (frag_idx, elem_idx) in the 2x4 layout
                frag_idx = k // 8
                elem_idx = k % 8
                # Further map elem_idx 0..7 to 2x4 layout
                row_in_frag = elem_idx // 4
                col_in_frag = elem_idx % 4

                ih = oh_idx + kh
                iw = ow_idx + kw

                # Load input value for this position and kernel element
                if in_bounds and ih < IN_H and iw < IN_W:
                    a_frag[frag_idx * 2 + row_in_frag, col_in_frag] = input_tensor[batch_idx, n_idx, ih, iw]

                # Load weight value for this channel and kernel element
                if n_idx < CHANNELS:
                    b_frag[frag_idx * 2 + row_in_frag, col_in_frag] = weight_tensor[n_idx, 0, kh, kw]

        # Initialize accumulator for MFMA result
        # For 16x16x16 bf16, each lane holds 4 f32 values
        acc = S.make_local((4,), S.f32)
        for i in S.range(4):
            acc[i] = S.convert(0.0, S.f32)

        # Execute MFMA operations
        # For 16x16x16 bf16, we need 2 MFMA calls (K=16 split into 2x8)
        # View bf16 arrays as required tensor shapes for MFMA
        a_view0 = S.view(a_frag[0], S.Tensor((1, 4, 1), S.bf16))
        b_view0 = S.view(b_frag[0], S.Tensor((1, 4, 1), S.bf16))
        a_view1 = S.view(a_frag[1], S.Tensor((1, 4, 1), S.bf16))
        b_view1 = S.view(b_frag[1], S.Tensor((1, 4, 1), S.bf16))

        # First MFMA: K elements 0-7
        acc = S.amdgpu.mfma_f32_16x16x16_bf16(a_view0[0], b_view0[0], acc)

        # Second MFMA: K elements 8-15
        acc = S.amdgpu.mfma_f32_16x16x16_bf16(a_view1[0], b_view1[0], acc)

        # Extract the result
        # Lane l holds C[4*(l//16) + t, l%16] for t=0..3
        # We want C[mfma_lane, mfma_lane] (diagonal element)
        # Since mfma_lane = l%16, and diagonal means row = col = mfma_lane
        # We need C[mfma_lane, mfma_lane] = C[4*mfma_group + t, mfma_lane]
        # This means 4*mfma_group + t = mfma_lane
        # So t = mfma_lane % 4
        acc_idx = mfma_lane % 4
        result = acc[acc_idx]

        # Store result
        if in_bounds:
            output_tensor[batch_idx, n_idx, oh_idx, ow_idx] = S.convert(result, S.bf16)


class ModelNew(nn.Module):
    """MFMA-optimized depthwise Conv2D using substrate."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                stride=stride, padding=padding, groups=in_channels, bias=bias)

        # Cache for cudagraph safety
        self._cached_x_ptr = None
        self._cached_w_ptr = None
        self._cached_y = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != (BATCH, CHANNELS, IN_H, IN_W):
            raise RuntimeError(f'This fused kernel only supports shape {(BATCH, CHANNELS, IN_H, IN_W)}.')
        if x.dtype != torch.float32:
            raise RuntimeError('This fused kernel only supports torch.float32 dtype.')

        # Convert to bf16 for computation
        x_bf16 = x.to(dtype=torch.bfloat16)
        w_bf16 = self.conv2d.weight.to(device=x.device, dtype=torch.bfloat16).squeeze(1)  # (128, 3, 3)

        # Allocate or reuse output buffer
        x_ptr = x_bf16.data_ptr()
        w_ptr = w_bf16.data_ptr()

        if self._cached_y is None or self._cached_x_ptr != x_ptr or self._cached_w_ptr != w_ptr:
            self._cached_y = torch.empty((BATCH, CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
            self._cached_x_ptr = x_ptr
            self._cached_w_ptr = w_ptr

        y_bf16 = self._cached_y

        # Run kernel
        mfma_depthwise_conv_kernel[_launch()](x_bf16, w_bf16, y_bf16)

        # Convert back to float32
        return y_bf16.to(dtype=torch.float32)
