# =============================================================================
# npyiter_sizesweep_bench.py — NumPy side of the size-tier sweep (same ids as
# npyiter_sizesweep_bench.cs). Six operation families measured across four
# element-count tiers: scalar (1), 1K, 100K, 1M.
#
#   add   — np.add(a, b, out=o)             contiguous binary ufunc nditer
#   sqrt  — np.sqrt(a, out=o)               contiguous unary ufunc nditer
#   sum   — np.sum(a)                        full reduction (PyUFunc_ReduceWrapper)
#   copy  — np.positive(a, out=o)            real ufunc nditer copy (NOT np.copyto:
#                                            that is a stripped raw-array walker)
#   sadd  — np.add(a2[::2], b2[::2], out=o)  strided binary
#   bcast — np.add(a, b1, out=o)             stride-0 broadcast binary
#
# Run: python benchmark/poc/npyiter_sizesweep_bench.py
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


def row(id_, label, ms):
    if ms >= 1.0:
        val = f"{ms:10.3f} ms"
    elif ms >= 0.001:
        val = f"{ms * 1000:10.2f} us"
    else:
        val = f"{ms * 1e6:10.1f} ns"
    print(f"{id_:<12} {label:<46} {val}")


def pick(n):
    if n <= 1:
        return 400_000, 40_000
    if n <= 1_000:
        return 200_000, 25_000
    if n <= 100_000:
        return 4_000, 600
    return 400, 80


SIZES = [("1", 1), ("1K", 1_000), ("100K", 100_000), ("1M", 1_000_000)]

print(f"NumPy {np.__version__} NpyIter size-sweep")
print(f"{'id':<12} {'aspect':<46} {'per call':>13}")
print("-" * 86)

for tag, n in SIZES:
    iters, warm = pick(n)
    rounds = 7 if n >= 100_000 else 5

    a = (np.arange(n, dtype=np.float64) % 97.0) + 1.0
    b = (np.arange(n, dtype=np.float64) % 31.0) + 2.0
    o = np.empty(n, dtype=np.float64)
    b1 = np.array([3.0], dtype=np.float64)
    a2 = (np.arange(2 * n, dtype=np.float64) % 53.0) + 1.0
    b2 = (np.arange(2 * n, dtype=np.float64) % 17.0) + 1.0
    sa = a2[::2]
    sb = b2[::2]
    so = np.empty(n, dtype=np.float64)

    # add
    np.add(a, b, out=o)
    check(o[n - 1] == a[n - 1] + b[n - 1], f"add@{tag}")
    row(f"add@{tag}", f"binary add contig f64 N={n}",
        best_ms(lambda: np.add(a, b, out=o), iters, warm, rounds))

    # sqrt
    np.sqrt(a, out=o)
    check(abs(o[n - 1] - np.sqrt(a[n - 1])) < 1e-9, f"sqrt@{tag}")
    row(f"sqrt@{tag}", f"unary sqrt contig f64 N={n}",
        best_ms(lambda: np.sqrt(a, out=o), iters, warm, rounds))

    # sum
    s = np.sum(a)
    check(abs(float(s) - float(np.sum(a))) < 1e-6, f"sum@{tag}")
    row(f"sum@{tag}", f"reduce sum contig f64 N={n}",
        best_ms(lambda: np.sum(a), iters, warm, rounds))

    # copy (real ufunc nditer)
    np.positive(a, out=o)
    check(o[n - 1] == a[n - 1], f"copy@{tag}")
    row(f"copy@{tag}", f"unary copy contig f64 N={n}",
        best_ms(lambda: np.positive(a, out=o), iters, warm, rounds))

    # sadd
    np.add(sa, sb, out=so)
    check(so[n - 1] == sa[n - 1] + sb[n - 1], f"sadd@{tag}")
    row(f"sadd@{tag}", f"strided add a[::2]+b[::2] f64 N={n}",
        best_ms(lambda: np.add(sa, sb, out=so), iters, warm, rounds))

    # bcast
    np.add(a, b1, out=o)
    check(o[n - 1] == a[n - 1] + 3.0, f"bcast@{tag}")
    row(f"bcast@{tag}", f"broadcast add a+b1(1) f64 N={n}",
        best_ms(lambda: np.add(a, b1, out=o), iters, warm, rounds))

print("-" * 86)
print("ALL CORRECTNESS CHECKS PASS" if fails == 0 else f"{fails} CORRECTNESS FAILURES")
