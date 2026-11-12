"""Bounded, seeded test programs over the supported API; standalone generation is stdlib-only."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import struct

MASK = (1 << 64) - 1
DEFAULT_SEED = 0xC0FFEE01
DEFAULT_COUNT = 32
KINDS = ("matmul", "reduce", "elementwise", "chain")


class Rng:
    """Test-protocol xorshift64*, independent of live search/input RNG state."""
    def __init__(self, seed):
        self.state = (seed * 0x9E3779B97F4A7C15) & MASK if seed else 0x9E3779B97F4A7C15

    def next(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & MASK


@dataclass(frozen=True)
class Case:
    index: int
    seed: int
    kind: str
    dims: tuple
    axis: int
    pre: int
    post: int
    input_hash: str

    def row(self):
        return "\t".join((str(self.index), f"{self.seed:016x}", self.kind,
                          ",".join(map(str, self.dims)), str(self.axis),
                          str(self.pre), str(self.post), self.input_hash))

    @classmethod
    def parse(cls, line):
        fields = line.strip().split("\t")
        if len(fields) != 8:
            raise ValueError("case must have eight TSV fields")
        index, seed, kind, dims, axis, pre, post, digest = fields
        def number(token, base=10):
            # Match Rust's unsigned integer grammar, not Python int's extensions.
            digits = "0123456789abcdefABCDEF" if base == 16 else "0123456789"
            body = token[1:] if token.startswith("+") else token
            if not body or any(char not in digits for char in body):
                raise ValueError("invalid unsigned integer token")
            return int(token, base)
        case = cls(number(index), number(seed, 16), kind,
                   tuple(number(dim) for dim in dims.split(",")),
                   number(axis), number(pre), number(post), digest)
        rank = 2 if kind in ("matmul", "chain") else len(case.dims)
        if (not 0 <= case.index < 128 or not 0 <= case.seed <= MASK
                or kind not in KINDS or not 1 <= len(case.dims) <= 3
                or any(not 1 <= d <= 4 for d in case.dims)
                or (kind in ("matmul", "chain") and len(case.dims) != 3)
                or not 0 <= case.axis < rank or case.pre not in range(4)
                or case.post not in range(4) or len(digest) != 16
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("case exceeds the v1 grammar/bounds")
        if input_digest(case) != digest:
            raise ValueError("case input fingerprint mismatch")
        return case


def input_specs(case):
    if case.kind in ("matmul", "chain"):
        m, k, n = case.dims
        specs = [("A", (m, k)), ("B", (k, n))]
        if case.kind == "chain":
            specs.append(("C", (1, n)))
        if case.post == 1:
            specs.append(("bias", (n,)))
        return specs
    specs = [("X", case.dims)]
    if case.kind == "elementwise":
        specs.append(("B", (case.dims[-1],)))
    return specs


def input_words(case):
    for ordinal, (name, shape) in enumerate(input_specs(case)):
        rng = Rng(case.seed ^ (((ordinal + 1) * 0xD1B54A32D192ED03) & MASK))
        words = []
        for i in range(math.prod(shape)):
            u = rng.next()
            sign = (u & 1) << 31
            # Finite f32 values with diverse magnitudes/mantissas, plus signed zeros.
            word = sign if i % 11 == 10 else sign | ((120 + ((u >> 1) & 15)) << 23) | ((u >> 5) & 0x7FFFFF)
            words.append(word)
        yield name, shape, words


def input_digest(case):
    value = 0xCBF29CE484222325
    def feed(data):
        nonlocal value
        for byte in data:
            value = ((value ^ byte) * 0x100000001B3) & MASK
    for name, shape, words in input_words(case):
        name = name.encode("ascii")
        feed(struct.pack("<Q", len(name)))
        feed(name)
        feed(struct.pack("<Q", len(shape)))
        for dim in shape:
            feed(struct.pack("<Q", dim))
        for word in words:
            feed(struct.pack("<I", word))
    return f"{value:016x}"


def generate(seed=DEFAULT_SEED, count=DEFAULT_COUNT):
    if not 0 <= seed <= MASK or not 1 <= count <= 128:
        raise ValueError("seed must fit u64; count must be 1..128")
    source = Rng(seed)
    for index in range(count):
        case_seed = source.next()
        rng = Rng(case_seed)
        kind = KINDS[index % len(KINDS)]  # stratified families, randomized programs/shapes
        group = index // len(KINDS)
        rank = 3 if kind in ("matmul", "chain") else 1 + rng.next() % 3
        if kind == "reduce" and group % 4 == 3:
            rank = 3  # pin last-axis rank-3 coverage as well as randomized axes
        dims = tuple(1 + rng.next() % 4 for _ in range(rank))
        axis = rng.next() % (2 if kind in ("matmul", "chain") else rank)
        if kind == "reduce" and group % 4 == 3:
            axis = 2
        pre = (group + (seed & 3)) % 4
        post = (group + group // 4 + (index % 4) * 3 + ((seed >> 2) & 3)) % 4
        case = Case(index, case_seed, kind, dims, axis, pre, post, "")
        yield Case(case.index, case.seed, case.kind, case.dims, case.axis,
                   case.pre, case.post, input_digest(case))


def render(seed=DEFAULT_SEED, count=DEFAULT_COUNT):
    header = f"# coffee-ten-fuzz-v1 seed={seed:016x} count={count}\n"
    return header + "# id\tseed\tkind\tdims\taxis\tpre\tpost\tinput_hash\n" + "".join(c.row() + "\n" for c in generate(seed, count))


def read_cases(text):
    cases = [Case.parse(s) for s in text.splitlines() if s.strip() and not s.startswith("#")]
    if not 1 <= len(cases) <= 128 or len({c.index for c in cases}) != len(cases):
        raise ValueError("invalid/duplicate case IDs or case count")
    return cases


def numpy_inputs(case):
    import numpy as np
    return {name: np.array(words, dtype=np.uint32).view(np.float32).reshape(shape)
            for name, shape, words in input_words(case)}


def build(case, ct):
    tensors = {name: ct.input(name, shape) for name, shape in input_specs(case)}
    def pre(x):
        return (lambda: x, lambda: x * 0.1, lambda: ct.relu(x), lambda: -x)[case.pre]()
    if case.kind in ("matmul", "chain"):
        x = pre(tensors["A"]) @ tensors["B"]
        if case.kind == "chain":
            x = ct.sum(x, axis=1) @ tensors["C"]
        if case.post == 0:
            return x
        if case.post == 1:
            return ct.relu(x + tensors["bias"])
        if case.post == 2:
            return x - ct.sum(x, axis=case.axis)
        return x - ct.max(x, axis=case.axis)
    x = pre(tensors["X"])
    if case.kind == "reduce":
        if case.post == 0:
            return ct.sum(x, axis=case.axis)
        if case.post == 1:
            return ct.max(x, axis=case.axis)
        if case.post == 2:
            return x - ct.max(x, axis=case.axis)
        return ct.sum(ct.relu(x), axis=case.axis) + ct.sum(x, axis=case.axis)
    b = tensors["B"]
    if case.post == 0:
        return x + b
    if case.post == 1:
        return (x - b) * -0.3
    if case.post == 2:
        return x * b + x
    return x / (b * b + 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--output", type=Path)
    paths.add_argument("--check", type=Path)
    args = parser.parse_args()
    try:
        text = render(args.seed, args.count)
    except ValueError as e:
        parser.error(str(e))
    if args.check:
        if args.check.read_text() != text:
            parser.error("corpus differs from its documented generator/seed")
        print(f"verified {args.count} cases, seed {args.seed:#x}")
    elif args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
