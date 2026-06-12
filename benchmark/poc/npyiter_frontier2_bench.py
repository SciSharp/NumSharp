# =============================================================================
# npyiter_frontier2_bench.py — NumPy side of frontier round 2 (same ids as
# npyiter_frontier2_bench.cs): overlap/alias taxes, comparison->bool,
# early-exit reductions, broadcast-view reduce, mixed/scalar/empty small-N,
# 8-D construction, and single-thread np.sin as the parallel-dividend baseline
# (NumPy never threads its iterator).
#
# Run: python benchmark/poc/npyiter_frontier2_bench.py
# =============================================================================
import time
import numpy as np

fails = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        print(f"  CORRECTNESS FAIL: {what}")


def best_ms(body, iters, warm, rounds=5):
    for _ in range(warm):
        body()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            body()
        best = min(best, (time.perf_counter() - t0) * 1000.0 / iters)
    return best


def row(id_, label, ms, note=""):
    if ms >= 1.0:
        val = f"{ms:10.3f} ms"
    elif ms >= 0.001:
        val = f"{ms * 1000:10.2f} us"
    else:
        val = f"{ms * 1e6:10.1f} ns"
    print(f"{id_:<6} {label:<56} {val}  {note}")


print(f"NumPy {np.__version__} NpyIter frontier-2 bench")
print(f"{'id':<6} {'aspect':<56} {'per call':>13}")
print("-" * 95)

# C14 — 8-D construction
a8 = np.arange(65536, dtype=np.float64).reshape(4, 4, 4, 4, 4, 4, 4, 4)
o8 = np.empty((4, 4, 4, 4, 4, 4, 4, 4), dtype=np.float64)
row("C14", "ctor 2-op 8-D contig (4^8) EXLOOP",
    best_ms(lambda: np.nditer((a8, o8), flags=["external_loop"],
                              op_flags=[["readonly"], ["writeonly"]]), 100_000, 12_000))
del a8, o8

# V — overlap / aliasing taxes
M = 4_194_304
a = (np.arange(M, dtype=np.float64) % 97.0) + 1.0
b = (np.arange(M, dtype=np.float64) % 31.0) + 2.0
np.add(a, b, out=a)
row("V1", "in-place np.add(a, b, out=a) f64 4M (exact alias)",
    best_ms(lambda: np.add(a, b, out=a), 25, 8, 7))

x = (np.arange(M, dtype=np.float64) % 53.0) + 1.0
xs = x[:-1]
xd = x[1:]
np.add(xs, xs, out=xd)
row("V2", "np.add(x[:-1], x[:-1], out=x[1:]) 4M (forced copy)",
    best_ms(lambda: np.add(xs, xs, out=xd), 12, 4, 7))
del a, b, x, xs, xd

# D — comparison -> bool output
a = (np.arange(M, dtype=np.float64) % 97.0) + 1.0
b = (np.arange(M, dtype=np.float64) % 31.0) + 2.0
o = np.empty(M, dtype=np.bool_)
np.less(a, b, out=o)
check(o[777] == (a[777] < b[777]), "D1")
row("D1", "np.less(a, b, out=bool) f64 4M",
    best_ms(lambda: np.less(a, b, out=o), 25, 8, 7))
del a, b, o

# E — boolean reduce: full scan vs early exit
ME = 10_000_000
idx = np.arange(ME)
all_false = idx == -1
early_hit = idx == 1000
check(not bool(np.any(all_false)), "E1 result")
row("E1", "np.any(bool 10M, ALL-FALSE: full scan)",
    best_ms(lambda: np.any(all_false), 50, 12, 7))
check(bool(np.any(early_hit)), "E2 result")
row("E2", "np.any(bool 10M, TRUE at idx 1000: early exit)",
    best_ms(lambda: np.any(early_hit), 200, 50, 7))
del idx, all_false, early_hit

# F — reduce over a BROADCAST view
a8k = (np.arange(8192, dtype=np.float64) % 97.0) + 1.0
bc = np.broadcast_to(a8k, (1024, 8192))
expect = 1024.0 * float(np.sum(a8k))
got = float(np.sum(bc))
check(abs(got - expect) / expect < 1e-9, "F1")
row("F1", "np.sum over broadcast_to(8K -> (1024,8192))",
    best_ms(lambda: np.sum(bc), 25, 8, 7))
del a8k, bc

# M/O — small-N frontier
ai = np.arange(1000, dtype=np.int32)
bf = np.arange(1000, dtype=np.float64) + 0.5
of = np.empty(1000, dtype=np.float64)
np.add(ai, bf, out=of)
check(of[777] == 777 + 777.5, "M1")
row("M1", "np.add(i32 1K, f64 1K, out=f64) mixed small-N",
    best_ms(lambda: np.add(ai, bf, out=of), 100_000, 12_000))

a1k = np.arange(1000, dtype=np.float64)
o1k = np.empty(1000, dtype=np.float64)
np.add(a1k, 5.0, out=o1k)
check(o1k[777] == 782.0, "O3")
row("O3", "np.add(a 1K, scalar, out=) array+scalar small-N",
    best_ms(lambda: np.add(a1k, 5.0, out=o1k), 100_000, 12_000))

e1 = np.empty(0, dtype=np.float64)
e2 = np.empty(0, dtype=np.float64)
eo = np.empty(0, dtype=np.float64)
np.add(e1, e2, out=eo)
row("O4", "np.add on EMPTY (0,) arrays, out=",
    best_ms(lambda: np.add(e1, e2, out=eo), 200_000, 25_000))
del ai, bf, of, a1k, o1k, e1, e2, eo

# PAR — single-thread np.sin baseline (f64 sin is scalar libm at AVX2;
# NumPy never threads its iterator — this is the ceiling it offers)
src = (np.arange(M, dtype=np.float64) % 6.283185) - 3.1415926
dst = np.empty(M, dtype=np.float64)
np.sin(src, out=dst)
check(abs(dst[777] - np.sin(src[777])) < 1e-15, "PAR")
row("PARp", "np.sin(x, out=) f64 4M (single-thread — NumPy ceiling)",
    best_ms(lambda: np.sin(src, out=dst), 5, 2, 5))
del src, dst

print("-" * 95)
print("ALL CORRECTNESS CHECKS PASS" if fails == 0 else f"{fails} CORRECTNESS FAILURES")
