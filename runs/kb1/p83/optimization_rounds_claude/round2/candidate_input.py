import torch
import torch.nn as nn
import substrate
import substrate.language as S


BATCH = 64
BATCH_SIZE = BATCH
IN_CHANNELS = 8
OUT_CHANNELS = 8
IN_H = 512
IN_W = 512
KERNEL_H = 3
KERNEL_W = 1
STRIDE = 1
PADDING = 0
DILATION = 1
GROUPS = 8
OUT_H = 510
OUT_W = 512

# Block tile dimensions - 128x128 as specified
BLOCK_M = 128
BLOCK_N = 128
WARP_SIZE = 64
NUM_WARPS = 4
THREADS_PER_BLOCK = WARP_SIZE * NUM_WARPS

# Grid: tile across (batch, channel, out_row) and out_col
# GEMM_M = batch * channel * out_row = 64 * 8 * 510 = 261120
# GEMM_N = out_col = 512
GEMM_M = BATCH * OUT_CHANNELS * OUT_H
GEMM_N = OUT_W

INPUT0_SHAPE = (BATCH, IN_CHANNELS, IN_H, IN_W)
WEIGHT_SHAPE = (OUT_CHANNELS, 1, KERNEL_H, KERNEL_W)
OUTPUT_SHAPE = (BATCH, OUT_CHANNELS, OUT_H, OUT_W)
OUTPUT_TORCH_DTYPE = torch.float32


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _launch():
    return ((_ceil_div(GEMM_N, BLOCK_N), _ceil_div(GEMM_M, BLOCK_M), 1), (THREADS_PER_BLOCK, 1, 1))


@substrate.jit
def fused_kernel(
    X: S.Tensor((64, 8, 512, 512), S.f32),
    W: S.Tensor((8, 1, 3, 1), S.f32),
    Y: S.Tensor((64, 8, 510, 512), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64
    warp = tid // 64
    warp_row = warp // 2
    warp_col = warp % 2

    group_m_base = S.block_id(1) * 128
    group_n_base = S.block_id(0) * 128

    # 2x2 warp grid, each warp handles 64x64 region
    # Each thread in warp (64 threads) handles 8x8 = 64 outputs
    # Total per warp: 64 * 64 = 4096 outputs

    for local_row in S.range(8):
        for local_col in S.range(8):
            # Position within warp's 64x64 region
            base_row_in_warp = (lane // 8) * 8
            base_col_in_warp = (lane % 8) * 8

            row = group_m_base + warp_row * 64 + base_row_in_warp + local_row
            col = group_n_base + warp_col * 64 + base_col_in_warp + local_col

            if row < 261120 and col < 512:
                # Decode row to (batch, channel, out_row)
                batch = row // (8 * 510)
                rem = row % (8 * 510)
                channel = rem // 510
                out_row = rem % 510
                out_col = col

                # Compute depthwise conv: sum over k
                result = S.convert(0.0, S.f32)
                for k in S.range(3):
                    in_row = out_row + k
                    if in_row >= 0 and in_row < 512:
                        result = result + X[batch, channel, in_row, out_col] * W[channel, 0, k, 0]

                Y[batch, channel, out_row, out_col] = S.convert(result, S.f32)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self._weight_cache = None
        self._weight_cache_key = None

    def _supports_optimized_path(self, x: torch.Tensor) -> bool:
        return (
            tuple(x.shape) == INPUT0_SHAPE
            and x.dtype == torch.float32
            and self.conv2d.in_channels == IN_CHANNELS
            and self.conv2d.out_channels == IN_CHANNELS
            and self.conv2d.kernel_size == (KERNEL_H, KERNEL_W)
            and self.conv2d.stride == (STRIDE, STRIDE)
            and self.conv2d.padding == (PADDING, PADDING)
            and self.conv2d.dilation == (DILATION, DILATION)
            and self.conv2d.groups == GROUPS
            and self.conv2d.bias is None
        )

    def _get_cached_weight(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.detach()
        key = (weight.data_ptr(), x.device.type, x.device.index, x.dtype, weight.is_contiguous())
        if self._weight_cache_key != key:
            self._weight_cache = weight.to(device=x.device, dtype=x.dtype).contiguous()
            self._weight_cache_key = key
        return self._weight_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._supports_optimized_path(x):
            raise RuntimeError(
                f"This optimized kernel only supports input shape {INPUT0_SHAPE}, "
                f"float32 inputs, and Conv2d({IN_CHANNELS}, {IN_CHANNELS}, ({KERNEL_H}, {KERNEL_W}), "
                f"stride={STRIDE}, padding={PADDING}, dilation={DILATION}, groups={GROUPS}, bias=False)."
            )
        x0 = x.contiguous()
        w = self._get_cached_weight(x0)
        y = torch.empty(OUTPUT_SHAPE, device=x0.device, dtype=OUTPUT_TORCH_DTYPE)
        fused_kernel[_launch](x0, w, y)
        return y
