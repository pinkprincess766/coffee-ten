# Loop IR and schedules

This page describes the two text formats of the `coffee-ten-loop` crate:

- the **kernel text**, the loop IR that `parse_kernel` reads and `Display for Kernel` prints
  (`crates/coffee-ten-loop/src/parse.rs`, `lib.rs`);
- the **schedule text**, which `Schedule::parse` reads and `apply` applies to a kernel
  (`schedule.rs`, `apply.rs`).

`ct.loop_ir` in Python returns the same kernel text. Every example below is a fenced block, and
a test parses, applies and prints each one ([§4](#4-how-the-examples-are-tested)).

## 1. Kernel text

A kernel is a header and a list of loop nests. A nest is a perfect nest of `for` loops, then
optional guards, then one store. This is `out = A @ B` for `A` of 8×4 and `B` of 4×32, as
`lower` prints it:

```loop-ir name=matmul
kernel matmul(A: f32[8, 4], B: f32[4, 32], out: f32[8, 32])
for i in 0..8:
  for j in 0..32:
    out[i, j] = 0.0
for i in 0..8:
  for k in 0..4:
    for j in 0..32:
      out[i, j] = out[i, j] + (A[i, k] * B[k, j])
```

Nest 0 zero-fills `out`. Nest 1 adds the products, with the reduction loop `k` in the middle.

### 1.1 Grammar

```grammar
kernel   = header { nest }
header   = "kernel " NAME "(" [ buffer { ", " buffer } ] ")"
buffer   = NAME ": f32[" [ NUMBER { ", " NUMBER } ] "]"
nest     = { loop } { guard } store
loop     = "for " NAME " in 0.." NUMBER [ mark ] ":"
mark     = " (vectorize)" | " (unroll)" | " (vectorize, unroll)"
guard    = "if " affine " < " [ "-" ] NUMBER ":"
store    = access " = " expr
access   = NAME "[" [ affine { ", " affine } ] "]"
affine   = [ "-" ] term { ( " + " | " - " ) term }
term     = NUMBER "*" NAME | NAME | NUMBER
expr     = operand [ ( " + " | " - " | " * " | " / " ) operand ]
operand  = access | CONSTANT | "(" expr ")" | "-(" expr ")"
         | "relu(" expr ")" | "exp(" expr ")" | "max(" expr ", " expr ")"
```

Rules for the text:

- `NAME` is an ASCII letter or `_`, then letters, digits and `_`. `NUMBER` is decimal digits.
- Each line is one `kernel`, `for`, `if` or store line. Blank lines are skipped, but they count
  in the line numbers of errors.
- A line is indented by two spaces for each `for` and `if` line above it in its nest. A store
  line ends its nest, and the next nest starts at column 0. Tabs are not indentation.
- Spaces are exact: one space around `+ - * /` and `=`, one after each comma, none at the end of
  a line.
- The parser checks syntax and that every buffer in a store or load is in the header. It does not
  check loop names, extents or index bounds. `Kernel::validate` does that ([§1.7](#17-errors)),
  and `apply` runs it on its result.

### 1.2 Header and buffers

`kernel NAME(` is followed by the buffers: `NAME: f32[d0, d1, ...]`. Every buffer is a row-major
`f32` array. Rank 0 is written `f32[]` and indexed as `Y[]`. The text does not say which buffers
are inputs: `lower` lists the inputs first, then temporaries, and puts the result in `out`, last.

```loop-ir
kernel scalar(Y: f32[])
Y[] = 0.0
```

A nest may have no loops. A kernel may have no nests.

### 1.3 Loops

`for NAME in 0..EXTENT:` runs `NAME` from 0 to `EXTENT - 1` in steps of 1. The extent is at least 1
and the names in a nest are distinct. A loop may carry a mark:

- `(vectorize)`: ask the backend to vectorize the loop. At most one loop of a nest has it, and it
  is the innermost.
- `(unroll)`: ask the backend to unroll the loop fully. Any loop may have it.
- `(vectorize, unroll)`: both, in this order.

The marks are requests to clang, not rewrites; the loop stays a loop in the IR.

### 1.4 Guards

`if AFFINE < BOUND:` follows the last loop and is indented one level deeper than the line above.
The store runs only when every guard holds. A `for` line after an `if` line is a syntax error.
`split` adds a guard when the factor does not divide the extent ([§3.3](#33-a-remainder-guard)).

### 1.5 Stores and expressions

A store is `BUF[index, ...] = value`. The value is built from:

- a load `BUF[index, ...]`;
- a constant;
- `relu(e)`, `exp(e)`, `max(e, e)` and the negation `-(e)`;
- one infix operator `+ - * /` between two operands.

An infix operand that is itself an infix expression is written in parentheses, so
`a + b + c` is a syntax error and `(a + b) + c` is not. `Display` adds these parentheses. A call
is never parenthesized.

```loop-ir
kernel ops(X: f32[4], Y: f32[4], Z: f32[4])
for i in 0..4:
  Z[i] = max(relu(X[i] - 1.0), -(Y[i])) / (exp(X[i]) + 1e-7)
```

Constants are `f32`. `Display` prints them as Rust's `{:?}` does (`0.0`, `-1.5`, `1e-7`, `inf`,
`NaN`), and the parser reads them back to the same bits, `-0.0` included.

### 1.6 Affine indices

Each index is `constant + sum of coeff * loop_name`. It is written as the terms in order, then
the constant, with a coefficient of 1 left out: `2*i + j - 3`, `-i + 3`. There is one index per
buffer dimension.

```loop-ir
kernel affine(X: f32[9], Y: f32[4], Z: f32[4])
for i in 0..4:
  Y[i] = X[2*i + 1]
for i in 0..4:
  Z[i] = X[-i + 3]
```

### 1.7 Errors

A syntax error carries a 1-based line number:

```loop-ir bad
kernel k(Y: f32[2])
for i in 0..2:
   Y[i] = 0.0
```

```error
loop IR line 3: bad indentation: expected 2 spaces, got 3
```

```loop-ir bad
kernel k(Y: f32[2])
for i in 0..2:
  if i < 1:
    for j in 0..2:
      Y[i] = 0.0
```

```error
loop IR line 4: loop after a guard
```

```loop-ir bad
kernel k(X: f32[2], Y: f32[2])
for i in 0..2:
  Y[i] = X[i] + X[i] + X[i]
```

```error
loop IR line 3: unexpected text " + X[i]"
```

`validate` returns the first of these problems: an extent of 0, a repeated loop name, an unknown
loop in an index, a vectorized loop that is not the only and innermost one, a guard that never
holds, an index with the wrong rank, and an index that leaves its dimension. A text that parses
can still fail here:

```loop-ir bad
kernel k(Y: f32[4])
for i in 0..5:
  Y[i] = 0.0
```

```error
nest 0: Y[dim 0] index range 0..=4 outside 0..4
```

## 2. Schedule text

A schedule is a list of steps separated by `;`. Steps run in order, each on the result of the
one before. `Schedule::parse` checks syntax only; the loops and the legality are checked by
`apply`.

### 2.1 Syntax

```grammar
schedule = [ step { ";" step } ]
step     = "split(" NEST "," NAME "," NAME "," NAME "," FACTOR ")"
         | "reorder(" NEST { "," NAME } ")"
         | "vectorize(" NEST "," NAME ")"
         | "unroll(" NEST "," NAME ")"
```

- Spaces, tabs and newlines are allowed between any two tokens, so a long schedule can be
  written one step per line. `Display` prints one canonical line, `step; step`.
- Empty text is the empty schedule. A `;` after the last step is an error.
- `NEST` and `FACTOR` are decimal numbers. `FACTOR` is at least 1.
- `reorder` needs at least one loop name. Within `split` the three names differ, and within
  `reorder` no name repeats.

```schedule
split(1, i, io, ii, 4);
split(1, j, jo, ji, 16);
reorder(1, io, jo, k, ii, ji);
vectorize(1, ji)
```

A syntax error carries the byte offset in the text:

```schedule bad
vectorize(1, j);
```

```error
schedule text at byte 16: expected identifier
```

```schedule bad
split(1, j, jo, ji, 0)
```

```error
schedule text at byte 20: factor must be at least 1
```

```schedule bad
split(1, j, j, ji, 4)
```

```error
schedule text at byte 12: repeated loop name
```

### 2.2 Steps and nest numbering

Nests are numbered from 0 in the order they appear in the kernel text. In the matmul above, nest
0 zero-fills `out` and nest 1 is the update. A step names its nest by number and its loops by
name. After a `split` the old name is gone and the new names are used by later steps.

- `split(nest, loop, outer, inner, factor)`: `loop` of extent `n` becomes `outer` of extent
  `ceil(n / factor)` and `inner` of extent `factor`, at the same place in the nest. Every use of
  `loop` in an index becomes `factor*outer + inner`. If `factor` does not divide `n`, the nest
  gets the guard `factor*outer + inner < n`. A `vectorize` mark moves to `inner`; an `unroll`
  mark is dropped, so the new loops are unrolled by their own steps.
- `reorder(nest, loop, ...)`: the new loop order, outermost first. Extents, guards and the body
  stay as they are.
- `vectorize(nest, loop)`: marks `loop` with `(vectorize)`.
- `unroll(nest, loop)`: marks `loop` with `(unroll)`.

`apply` works on a copy. If a step is refused, the input kernel is unchanged and no result is
returned. The README ([§2.2](../README.md#22-schedules)) says which schedules keep the order of
float sums (`preserving`) and which do not (`reordering`).

### 2.3 Legality

`apply` refuses a step with a `ScheduleError`. `step` counts from 0 in the schedule. The error
text has the step number and the nest, but no byte offset: the schedule was already parsed.

| Error kind | Text | When |
|---|---|---|
| `NoSuchNest` | `step S: no nest N` | The kernel has fewer than `N + 1` nests. |
| `NoSuchLoop` | `step S: nest N has no loop L` | The nest has no loop `L` (also after a `split` removed it). |
| `NameTaken` | `step S: nest N already has a loop X` | `outer` or `inner` of a `split` is a loop of the nest, or they are equal. |
| `BadReorder` | `step S: order is not a permutation of the loops of nest N` | `reorder` does not list every loop of the nest exactly once. |
| `IllegalReorder` | `step S: nest N reads its target at another index` | See below. |
| `IllegalVectorize` | `step S: cannot vectorize in nest N: REASON` | See below. |
| `IllegalUnroll` | `step S: cannot unroll in nest N: REASON` | See below. |
| `Invalid` | `schedule produced an invalid kernel: ...` | The result fails `validate`. With a valid input this is a bug in `apply`. |

`ZeroFactor` (`split factor is 0`) cannot come from text, as the parser refuses a factor of 0.
`NumberTooLarge` (`a number does not fit in i64`) means a factor or an extent does not fit the
IR's `i64`. `NotImplemented` exists in the enum, and `apply` never returns it.

**reorder.** A reduction loop is a loop whose name does not appear, with a non-zero coefficient,
in the store target. `reorder` is legal if

- different values of the non-reduction loops write different elements of the target (a
  sufficient check: each loop appears in one dimension, and the coefficients in a dimension are
  mixed-radix digits, as in `i` or `4*io + ii`);
- with no reduction loop, the body loads the target only at the index it stores to;
- with a reduction loop, the value is `target + e` or `e + target`, where `e` does not load the
  target buffer. `target * 2 + e` and an overwrite `target = e` are refused.

**vectorize.** `REASON` is one of these checks, in this order:

1. `already vectorized`: the loop has the mark.
2. `not the innermost loop`.
3. `reduction loop`: the loop is not in the store target (vectorized reductions are out of
   scope).
4. `target stride is not 1`: the loop must appear only in the last dimension of the target, with
   coefficient exactly 1, so that neighbouring lanes write neighbouring elements.
5. `unrolled and extent is not a power of two`.

**unroll.** `REASON` is one of these checks, in this order:

1. `already unrolled`.
2. `extent is larger than 64`: the limit is `MAX_UNROLL_EXTENT = 64`, a code-size choice that
   fits a 4×16 register tile.
3. `vectorized and extent is not a power of two`.

The power-of-two rule exists because the backend unrolls a vectorized loop by vectorizing it at
a width equal to its extent, and clang ignores a width that is not a power of two.

### 2.4 What the backend does with the marks

`coffee-ten-codegen-c` turns a mark into a `#pragma clang loop` line before the `for`
(`crates/coffee-ten-codegen-c/src/lib.rs`, `nest_c`):

- `(unroll)`: `unroll(full)`.
- `(vectorize)`: `vectorize(enable)`.
- A vectorized loop whose extent is a power of two of at most 16, or that is also unrolled:
  `vectorize(enable) vectorize_width(EXTENT) unroll(disable)`. Without the width, clang
  unrolls such a short loop before its loop vectorizer runs and the code stays scalar (#96,
  `CLANG_EARLY_UNROLL_LIMIT`).

The hint changes the speed and not the result: the backend keeps every float operation a
separate rounding.

## 3. Worked examples

### 3.1 The naive matmul

The kernel of [§1](#1-kernel-text) is the starting point, under the name `matmul`. Its nest 1
has the loops `i`, `k`, `j`. The target `out[i, j]` has no `k`, so `k` is the reduction loop.

### 3.2 A 4×16 register tile

Split `i` by 4 and `j` by 16, move the tile loops `ii` and `ji` innermost with `k` just outside
them, vectorize `ji`, and unroll `ii` and `ji`. This keeps a 4×16 tile of `out` in registers
across the `k` loop. The result below is what `apply` prints.

```schedule on=matmul
split(1, i, io, ii, 4); split(1, j, jo, ji, 16); reorder(1, io, jo, k, ii, ji); vectorize(1, ji); unroll(1, ii); unroll(1, ji)
```

```loop-ir
kernel matmul(A: f32[8, 4], B: f32[4, 32], out: f32[8, 32])
for i in 0..8:
  for j in 0..32:
    out[i, j] = 0.0
for io in 0..2:
  for jo in 0..2:
    for k in 0..4:
      for ii in 0..4 (unroll):
        for ji in 0..16 (vectorize, unroll):
          out[4*io + ii, 16*jo + ji] = out[4*io + ii, 16*jo + ji] + (A[4*io + ii, k] * B[k, 16*jo + ji])
```

Nest 0 is unchanged. In nest 1 every index has the form `4*io + ii` or `16*jo + ji`. The `ji` loop
has extent 16, a power of two, so `vectorize` and `unroll` agree on it.

### 3.3 A remainder guard

`k` has extent 4, and 3 does not divide 4. The split gives `ko` of extent `ceil(4 / 3) = 2` and
`ki` of extent 3, which covers 6 values of `k`. The guard `3*ko + ki < 4` skips the last two.

```schedule on=matmul
split(1, k, ko, ki, 3)
```

```loop-ir
kernel matmul(A: f32[8, 4], B: f32[4, 32], out: f32[8, 32])
for i in 0..8:
  for j in 0..32:
    out[i, j] = 0.0
for i in 0..8:
  for ko in 0..2:
    for ki in 0..3:
      for j in 0..32:
        if 3*ko + ki < 4:
          out[i, j] = out[i, j] + (A[i, 3*ko + ki] * B[3*ko + ki, j])
```

Two splits with remainders give two guards. They print in the order the splits ran, each one
level deeper:

```schedule on=matmul
split(1, i, io, ii, 3); split(1, k, ko, ki, 3)
```

```loop-ir
kernel matmul(A: f32[8, 4], B: f32[4, 32], out: f32[8, 32])
for i in 0..8:
  for j in 0..32:
    out[i, j] = 0.0
for io in 0..3:
  for ii in 0..3:
    for ko in 0..2:
      for ki in 0..3:
        for j in 0..32:
          if 3*io + ii < 8:
            if 3*ko + ki < 4:
              out[3*io + ii, j] = out[3*io + ii, j] + (A[3*io + ii, 3*ko + ki] * B[3*ko + ki, j])
```

### 3.4 Illegal schedules

Each schedule below is refused. The text under it is the exact error.

`reorder` must list every loop of the nest. Nest 1 has `i`, `k` and `j`:

```schedule on=matmul
reorder(1, i, k)
```

```error
step 0: order is not a permutation of the loops of nest 1
```

`vectorize` wants the innermost loop. `i` is the outermost:

```schedule on=matmul
vectorize(1, i)
```

```error
step 0: cannot vectorize in nest 1: not the innermost loop
```

A reduction loop is not vectorized. After the reorder `k` is innermost, and the step that fails
is step 1:

```schedule on=matmul
reorder(1, i, j, k); vectorize(1, k)
```

```error
step 1: cannot vectorize in nest 1: reduction loop
```

A vectorized or unrolled loop that is not a power of two cannot be both. `split` by 12 gives a
loop of extent 12 (and a guard, as 12 does not divide 32):

```schedule on=matmul
split(1, j, jo, ji, 12); vectorize(1, ji); unroll(1, ji)
```

```error
step 2: cannot unroll in nest 1: vectorized and extent is not a power of two
```

In the other order the same rule is reported by `vectorize`:

```schedule on=matmul
split(1, j, jo, ji, 12); unroll(1, ji); vectorize(1, ji)
```

```error
step 2: cannot vectorize in nest 1: unrolled and extent is not a power of two
```

`unroll` stops at an extent of 64. This kernel scales a vector of 100 elements:

```loop-ir name=scale
kernel scale(X: f32[100], Y: f32[100])
for i in 0..100:
  Y[i] = X[i] * 2.0
```

```schedule on=scale
unroll(0, i)
```

```error
step 0: cannot unroll in nest 0: extent is larger than 64
```

`reorder` is refused when iterations depend on each other. Here every iteration reads the element
the one before it wrote, so the order of the iterations matters (the target index is `i + 1`
and the load is at `i`):

```loop-ir name=shift
kernel shift(Y: f32[9])
for i in 0..8:
  Y[i + 1] = Y[i] + 1.0
```

```schedule on=shift
reorder(0, i)
```

```error
step 0: nest 0 reads its target at another index
```

## 4. How the examples are tested

`crates/coffee-ten-loop/tests/loop_ir_doc.rs` reads this file at compile time and checks every
fenced block. The info string after the opening fence tells the test what the block is:

| Info string | Check |
|---|---|
| `loop-ir` | `parse_kernel` accepts it, `validate` accepts it, and `Display` prints the same text. With `name=N`, later `schedule` blocks can use it. |
| `loop-ir bad` | The error of `parse_kernel` (or `validate`) equals the next `error` block. |
| `schedule` | `Schedule::parse` accepts it, and the printed schedule parses to the same steps. |
| `schedule on=N` | Also applied to the kernel named `N`. The next block is the printed result (`loop-ir`) or the error (`error`), and it must match exactly. |
| `schedule bad` | `Schedule::parse` fails with the text of the next `error` block. |
| `error` | The exact `Display` of the error before it. |
| `grammar` | Reference only. Not checked. |

A block with any other info string fails the test, so an example cannot be added without a check.
Fences must start at column 0. Prose that quotes a step inline (`reorder(1, i, k)` and the like)
is not checked; put an example in a block.

The test chose an integration test over a doc-test. A doc-test would need `include_str!` of this
file into a doc comment, and rustdoc would run every Rust block as code. These blocks are not
Rust, and a pairing of "schedule, result, error" needs a small parser anyway.
