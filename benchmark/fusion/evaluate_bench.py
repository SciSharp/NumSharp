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
