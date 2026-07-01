## 1: Prompt 1

Optimize the substrate kernel in /workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/round0/output_model_new.py

Use MFMA instructions `S.amdgpu.mfma_32x32x8_bf16_f32` to do matrix multiplications and loads the data from global memory using vectorized loads S.amdgpu.raw_buffer_load_x4. The MFMA instruction computes 32x32x8 matmul cooperatively in a wave. Build larger tiles by issuing multiple MFMA instructions across K and output subtiles.

Stage A and B through LDS. Each thread loads operand fragments in 16-byte chunks. Treat each 16-byte fragment as `(4, S.u32)` and reinterpret it as `2 x (4, S.bf16)`. Feed both `(4, S.bf16)` halves into MFMA in natural order. The intended effect is a cooperative `32x32x16` accumulation from two natural MFMA steps. Do not add lane-dependent or K-dependent control flow to select halves.
  - The two `(4, S.bf16)` halves from one 16-byte LDS load collectively represent a swizzled `32x16` operand contribution with 4-column interleaving.
  - Consuming them in natural order must produce the same final C as a naive conceptual layout because operand pairings remain consistent under MFMA swizzle.

Scale the kernel from one wave to four waves without changing the MFMA per-wave invariant. Interpret the 4 warps as a 2 x 2 warp grid. Keep MFMA math identical per warp. Only add warp ownership offsets at operand fetch and output writeback.

Note:

- The current snapshot of the repo under /workspace/substrate has examples of the substrate DSL. /workspace/substrate/test/examples/gemm/amdgpu/test_gemm_mfma.py has an implementation of GEMM / MFMA
- If optimizing for a fused kernel, do not directly call the amdgpu gemm kernel inside `substrate_kernel`. You can adopt the changes of it but the goal is to write a standalone fused substrate kernel.
- Make the optimized path cudagraph-safe: never build descriptor / metadata device tensors inside `forward()`. Prebuild or cache them and reuse them; only rebuild if the underlying storage pointer changes.
- Do not use torch native compute anywhere in `output_model_new.py` to perform multiplication or linear algebra, including fallback branches.
- This ban includes any direct or indirect torch compute path such as `torch.matmul`, `torch.mm`, `torch.mv`, `torch.bmm`, `torch.einsum`, `torch.mul`, `torch.addmm`, `torch.chain_matmul`, `torch.nn.functional.linear`, or equivalent compositions that let torch perform the multiply/linear work.
- The optimized kernel must actually issue MFMA instructions in the substrate kernel; a solution that does not use MFMA is not acceptable.
- Use the precreated `case.txt` in the current round directory. It already contains exactly one line: the absolute path of the current round directory.
- Do not modify `case.txt`. If you manually run correctness/debug evaluation, use only the exact evaluation command injected by the orchestrator in the final prompt for this round.
- Do not use git to try to find any old files!!!
Strict constraints:
  - Do not browse the web.
  - Do not search online for documentation, examples, repos, or references.
  - Do not use any network access at all.
  - Use only the files already present in the workspace dir.
  - Do not read any other kernel from anywhere in the space!!!

## 2: Prompt 2

Implement software pipelining to overlap MFMA, LDS access and global memory access. Use double buffering. Unroll the K-loop by 2 to minimize branching. Split the LDS access and overlap with MFMA in a fine-grain way to reduce the size of the working sets of the shared memroy A/B.

## 3: Prompt 3

Utilize range in S.amdgpu.raw_buffer_load_x4 / S.amdgpu.raw_buffer_store_* to remove the explicit branches guarding OOB access. The range is in the units of bytes. When range is set, S.amdgpu.raw_buffer_load_x4() returns 0 for OOB elements, and S.amdgpu.raw_buffer_store*() discards the OOB writes. The computations and LDS access still work with 0, and the OOB writes are discarded, therefore the optimization is safe. Removing the branches in the loop is more beneficial compared to reducing extra computations and LDS access.
