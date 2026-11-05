"""NumPy twin of evaluate_samewidth_bench.cs (np.evaluate's P5.3 same-width vector plan): NumPy's statements for the same-width trees, the same value distributions
(age int64 0..96, price float64 with 10% NaN, i4 int32 -10..9, f4 float32 [0,1), m random bool), pinned to 0x14, the C#
side's schedule (best of 7 after a paused double warm-up)."""
import ctypes
import os
import time

import numpy as np

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessAffinityMask.restype = ctypes.c_int
assert k32.SetProcessAffinityMask(k32.GetCurrentProcess(), int(os.environ.get("NS_PROBE_AFFINITY") or "14", 16))


def best_us(f, n):
    """Best per-call microseconds over 7 rounds after a paused double warm-up."""
    for _ in range(20):
        f()
    time.sleep(0.1)
    for _ in range(20):
        f()
    reps = 3000 if n <= 1000 else 200 if n <= 100_000 else 25
    best = 1e9
    for _ in range(7):
        s = time.perf_counter()
        for _ in range(reps):
            f()
        best = min(best, (time.perf_counter() - s) / reps)
    return best * 1e6


rng = np.random.default_rng(42)
for n in (1_000, 100_000, 1_000_000):
    age = rng.integers(0, 97, n, dtype=np.int64)
    price = rng.random(n) * 100
    price[rng.integers(0, 10, n) == 0] = np.nan
    i4 = rng.integers(-10, 10, n, dtype=np.int32)
    f4 = rng.random(n, dtype=np.float32)
    m = rng.integers(0, 2, n).astype(bool)
    trees = [
        ("where(age>30,price,0)", lambda: np.where(age > 30, price, 0)),
        ("(age>30)*1.5", lambda: (age > 30) * 1.5),
        ("(age>30)&isnan(price)", lambda: (age > 30) & np.isnan(price)),
        ("logical_and(age,price)", lambda: np.logical_and(age, price)),
        ("where(i4<5,f4,f4*2)", lambda: np.where(i4 < 5, f4, f4 * 2)),
        ("m!=0", lambda: m != 0),
        ("where(m&(age>30),price,-1)", lambda: np.where(m & (age > 30), price, -1)),
    ]
    for name, f in trees:
        print(f"{name:28s} n={n:9d}: numpy {best_us(f, n):9.1f} us", flush=True)
