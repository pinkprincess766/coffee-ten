# coffee-ten

A tiny tensor compiler: Python builder → graph IR → loop IR with a schedule →
C → clang → a CPU kernel. The research question is whether an analytic cost model
(cache traffic plus roofline) can rank schedules well enough to replace most of a
random search.

## Status

Pre-MVP: workspace skeleton only, no compiler code yet. Numbers in this README are
a baseline, not results.

## Basis

Each section names the idea the code rests on, where it lives and what checks it.
`planned` means not implemented yet. References are numbered `[L#]`.

### Algorithm vs schedule — [L4], [L6]

What to compute is a graph; how to compute it is a schedule, a list of named
loop-IR transformations (`split`, `reorder`, `vectorize`). Each step either keeps
the program legal or returns an error. Code: planned. Checked by: planned.

### Graph IR and fusion — [L5], [L14]

A static-shape SSA DAG. A chain of elementwise ops lowers to one loop nest, so it
is fused for free; `matmul` is its own node. The backend emits C source, not
LLVM IR. Code: planned. Checked by: planned.

### Loop tiling for matmul — [L3], [L7]

`split` and `reorder` give cache tiles, `vectorize` widens the innermost `j` loop.
Packing and register micro-kernels are out of scope for the MVP, which caps the
reachable fraction of peak. Code: planned. Checked by: planned.

### Legality of loop reordering — [L24]

Reordering is allowed inside one perfect nest. In the matmul update nest the only
carried dependence is the reduction over `k`, a simple case of the dependence
vectors of the polyhedral model. Because float addition is not associative, steps
that keep the order of the `k` sum are tested bit for bit, the others with a
tolerance. Code: planned. Checked by: planned.

### Floating-point error bounds — [L29]

Tolerances are derived, not tuned: `|fl(xᵀy) − xᵀy| ≤ γ_n Σ|x_i y_i|` with
`γ_n = nu/(1 − nu)` and `u = 2⁻²⁴` for `f32`, plus an absolute term for subnormal
products. Code: planned. Checked by: planned.

### Roofline — [L2]

Speed is reported as a fraction of `min(peak, AI · BW)`, with peak FLOP/s and
memory bandwidth measured on the same machine. Code: planned. Checked by: planned.

### Tracing frontend — [L15], [L18]

A Python `Tensor` computes nothing; it records a graph that is run by a reference
interpreter or compiled. Code: planned. Checked by: planned.

### Cost-model autotuning (research, after the MVP) — [L21], [L22]

Hypothesis: the top 10 schedules by the model contain one within 90% of the best
found by random search with 1000 measurements. Code: planned. Checked by: planned.

## Baseline / Results

No baseline yet.

## Non-goals

Competing with BLAS, TVM, XLA or Inductor; training and autograd; sparse tensors;
dynamic shapes; convolutions and attention; types other than `f32`; a custom
language; ONNX or torch.fx import; MLIR and LLVM; threads and GPU in the MVP.

## References

- [L2] S. Williams, A. Waterman, D. Patterson. Roofline: an insightful visual
  performance model for multicore architectures. *CACM* 52(4), 2009.
  https://doi.org/10.1145/1498765.1498785
- [L3] S. Boehm. Fast Multidimensional Matrix Multiplication on CPU from Scratch.
  Blog post, 2022. https://siboehm.com/articles/22/Fast-MMM-on-CPU
- [L4] J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, S. Amarasinghe.
  Halide: A Language and Compiler for Optimizing Parallelism, Locality, and
  Recomputation in Image Processing Pipelines. PLDI 2013.
  https://doi.org/10.1145/2491956.2462176 (CACM version:
  https://doi.org/10.1145/3150211)
- [L5] T. Chen, T. Moreau, Z. Jiang, L. Zheng et al. TVM: An Automated End-to-End
  Optimizing Compiler for Deep Learning. OSDI 2018. https://arxiv.org/abs/1802.04799
- [L6] S. Feng, B. Hou, H. Jin, W. Lin et al. TensorIR: An Abstraction for
  Automatic Tensorized Program Optimization. ASPLOS 2023.
  https://arxiv.org/abs/2207.04296
- [L7] K. Goto, R. A. van de Geijn. Anatomy of High-Performance Matrix
  Multiplication. *ACM TOMS* 34(3), 2008. https://doi.org/10.1145/1356052.1356053
- [L14] J. Ansel, E. Yang, H. He, N. Gimelshein et al. PyTorch 2: Faster Machine
  Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.
  ASPLOS 2024. https://doi.org/10.1145/3620665.3640366
- [L15] tinygrad. https://github.com/tinygrad/tinygrad
- [L18] R. Frostig, M. J. Johnson, C. Leary. Compiling machine learning programs
  via high-level tracing. SysML 2018.
  https://mlsys.org/Conferences/2019/doc/2018/146.pdf
- [L21] T. Chen, L. Zheng, E. Yan, Z. Jiang et al. Learning to Optimize Tensor
  Programs. NeurIPS 2018. https://arxiv.org/abs/1805.08166 ; L. Zheng, C. Jia,
  M. Sun, Z. Wu et al. Ansor: Generating High-Performance Tensor Programs for Deep
  Learning. OSDI 2020. https://arxiv.org/abs/2006.06762
- [L22] A. Adams, K. Ma, L. Anderson, R. Baghdadi, T.-M. Li et al. Learning to
  Optimize Halide with Tree Search and Random Programs. *ACM TOG* 38(4), 2019.
  https://doi.org/10.1145/3306346.3322967
- [L24] U. Bondhugula, A. Hartono, J. Ramanujam, P. Sadayappan. A practical
  automatic polyhedral parallelizer and locality optimizer. PLDI 2008.
  https://doi.org/10.1145/1375581.1375595
- [L29] N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
  SIAM, 2002. https://doi.org/10.1137/1.9780898718027

## License

Apache-2.0, see `LICENSE`.
