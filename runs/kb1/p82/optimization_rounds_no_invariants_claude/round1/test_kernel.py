#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p82/optimization_rounds_no_invariants_claude/round1')
from output_model_new import ModelNew

torch.manual_seed(42)

# Create model with correct parameters for depthwise conv
model = ModelNew(in_channels=64, kernel_size=3, stride=1, padding=0, bias=False).cuda()

# Create input
x = torch.randn((16, 64, 512, 512), dtype=torch.float32, device='cuda')

print("Running optimized kernel...")
try:
    y = model(x)

    # Compute reference
    w = model.conv2d.weight.to(dtype=torch.float32)
    ref = F.conv2d(x, w, groups=64)

    # Check results
    diff = torch.abs(y - ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f'Output shape: {y.shape}')
    print(f'Reference shape: {ref.shape}')
    print(f'Max diff: {max_diff}')
    print(f'Mean diff: {mean_diff}')

    # Check a few values
    print(f'Y[0,0,0,:4]: {y[0,0,0,:4]}')
    print(f'Ref[0,0,0,:4]: {ref[0,0,0,:4]}')

    # Check if close
    if torch.allclose(y, ref, rtol=1e-2, atol=0.1):
        print('SUCCESS: Results match!')
    else:
        print('MISMATCH: Results do not match')
except Exception as e:
    import traceback
    traceback.print_exc()
