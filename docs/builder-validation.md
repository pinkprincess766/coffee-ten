# Python expression validation (N18)

This project currently exposes matmul, single-axis reductions and elementwise expressions,
not a public `einsum` string parser. As in N7's current-operation corpus, N18 covers this
existing API; it adds no einsum syntax or new operation.

- Reduction axes (including negative axes), scalar reductions and empty reduced axes
  raise `ValueError` at operation construction, using the existing Graph messages.
- Matmul rank/inner dimensions and broadcast compatibility are checked at construction.
- N18 adds checked dense element counts at input, matmul and broadcast construction.
  Impossible shapes now raise `ValueError: shape (...) has too many elements` before
  lowering or allocation. Zero dimensions and scalar shapes retain their contracts.
- Distinct inputs with the same name are rejected during Graph construction, before C
  emission, cache creation or compiler startup. Reusing **one** tensor is valid and is
  deduplicated by expression identity. N18 keeps this existing validation stage rather
  than repeatedly walking entire expression trees at every binary operation.

Tests empty `PATH` to ensure shape/axis failures are independent of clang availability;
duplicate-name compilation also leaves the requested cache directory absent. Valid
sharing/negative-axis expressions retain exact output bits. Existing tests cover wrong
operand types, dimensions, depth limits, dtype and runtime input bindings. No arithmetic
or valid-expression behavior changes. A future public einsum API needs its own syntax
and validation contract; it is not implied by this task.
