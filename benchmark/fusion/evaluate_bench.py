# evaluate_bench.py — NumPy absolutes for the Wave 6.1 fusion gate (same box).
import time

import numpy as np

N = 4_000_000
a = np.arange(N, dtype=np.float64) + 1.0
b = (np.arange(N, dtype=np.float64) % 977.0) + 2.0
c = np.arange(N, dtype=np.float64) * 0.5
af = a.astype(np.float32)
bf = b.astype(np.float32)
ai = np.arange(N, dtype=np.int32)


def best(fn, rounds=9):
    out = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        out = min(out, (time.perf_counter() - t0) * 1000)
    return out


# warmup
_ = a * b + c
_ = (a - b) / (a + b)
_ = np.sum(a * b)

print(f"numpy {np.__version__}, 4M float64, best of 9:")
print(f"  a*b+c       {best(lambda: a * b + c):7.2f} ms")
print(f"  (a-b)/(a+b) {best(lambda: (a - b) / (a + b)):7.2f} ms")
print(f"  sum(a*b)    {best(lambda: np.sum(a * b)):7.2f} ms")
print(f"  sum(af*bf)  {best(lambda: np.sum(af * bf)):7.2f} ms  [f32]")
out = np.empty_like(a)


def muladd_out():
    np.multiply(a, b, out=out)
    np.add(out, c, out=out)


print(f"  a*b+c out=  {best(muladd_out):7.2f} ms  [two-pass with out=]")
print(f"  i4*2+f8     {best(lambda: ai * 2 + c):7.2f} ms")

# operand-layout context: a*b+c unfused absolutes per layout. NumPy has no fused
# evaluate (numexpr is not a pinned dep), so this is the unfused baseline the
# NumSharp fused/unfused ratios sit against — same 2-D 2000x2000 (=4M) operands.
LR = LC = 2000
a2 = (np.arange(LR * LC, dtype=np.float64) + 1.0).reshape(LR, LC)
b2 = (np.arange(LR * LC, dtype=np.float64) % 977.0 + 2.0).reshape(LR, LC)
c2 = (np.arange(LR * LC, dtype=np.float64) * 0.5).reshape(LR, LC)


def lay(x, l):
    return {"C": x, "F": np.asfortranarray(x), "T": x.T,
            "strided": x[:, ::2],
            "bcast": np.broadcast_to(x[0:1, :], (LR, LC))}[l]


print(f"  a*b+c across operand layouts (2-D {LR}x{LC}, unfused):")
for l in ("C", "F", "T", "strided", "bcast"):
    al, bl, cl = lay(a2, l), lay(b2, l), lay(c2, l)
    print(f"    [{l:<7}] {best(lambda al=al, bl=bl, cl=cl: al * bl + cl):7.2f} ms")
