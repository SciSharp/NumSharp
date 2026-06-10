# route_audit_bench.py — NumPy twin of route_audit_bench.cs (same shapes/box).
import time

import numpy as np

N = 4_000_000
af = (np.arange(N, dtype=np.float32) % 997) + 1
bf = (np.arange(N, dtype=np.float32) % 877) + 2
afS = af[::2]
bfS = bf[::2]
m2d = np.arange(4_000_000, dtype=np.float32).reshape(2000, 2000)
col = np.arange(2000, dtype=np.float32).reshape(2000, 1)
ai = np.arange(N, dtype=np.int32)
bi = np.arange(N, dtype=np.int32) % 31
aiS = ai[::2]
biS = bi[::2]
ad = af.astype(np.float64)[:2_000_000]
adS = af.astype(np.float64)[::4]


def best(fn, rounds=7):
    fn()
    out = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        out = min(out, (time.perf_counter() - t0) * 1000)
    return out


def row(name, ms):
    print(f"  {name:<44} {ms:8.3f} ms")


print(f"numpy {np.__version__}")
print("== A. controls ==")
row("less   f32 contig 4M", best(lambda: af < bf))
row("less   f32 strided 2M", best(lambda: afS < bfS))
row("less   f32 bcast (2k,2k)<(2k,1)", best(lambda: m2d < col))
row("and    i32 contig 4M", best(lambda: np.bitwise_and(ai, bi)))
row("and    i32 strided 2M", best(lambda: np.bitwise_and(aiS, biS)))
row("invert i32 contig 4M", best(lambda: np.invert(ai)))
row("invert i32 strided 2M", best(lambda: np.invert(aiS)))
row("sinh   f64 contig 2M", best(lambda: np.sinh(ad)))
row("sinh   f64 strided 1M", best(lambda: np.sinh(adS)))

print("== B. maximum ==")
row("maximum f32 contig 4M", best(lambda: np.maximum(af, bf)))
row("maximum f32 strided 2M", best(lambda: np.maximum(afS, bfS)))
row("maximum f32 bcast", best(lambda: np.maximum(m2d, col)))

print("== C. shifts ==")
row("a<<3   i32 contig 4M", best(lambda: np.left_shift(ai, 3)))
row("a<<3   i32 strided 2M", best(lambda: np.left_shift(aiS, 3)))
row("a<<b   i32 contig 4M", best(lambda: np.left_shift(ai, bi)))
row("a<<b   i32 strided 2M", best(lambda: np.left_shift(aiS, biS)))

print("== D. reductions ==")
row("sum    axis=0 f32 (2k,2k)", best(lambda: np.sum(m2d, axis=0)))
row("cumsum axis=0 f32 (2k,2k)", best(lambda: np.cumsum(m2d, axis=0)))
row("cumsum axis=1 f32 (2k,2k)", best(lambda: np.cumsum(m2d, axis=1)))
row("var    axis=0 f32 (2k,2k)", best(lambda: np.var(m2d, axis=0)))
row("var    axis=1 f32 (2k,2k)", best(lambda: np.var(m2d, axis=1)))
row("all    axis=0 f32 (2k,2k)", best(lambda: np.all(m2d, axis=0)))
row("any    axis=1 f32 (2k,2k)", best(lambda: np.any(m2d, axis=1)))
print("[done]")
