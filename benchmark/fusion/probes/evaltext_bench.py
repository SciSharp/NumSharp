"""NumPy twin of evaltext_bench.cs (docs/plans/evaluate-text.md, phase E1).

Times NumPy's own statements for the four trees over the identical data (the same 64-bit LCG picks the same city names),
pinned to the same CPUs, best of 7 rounds after a paused double warm-up.

    filter   (names == "paris") & (age > 30)
    count    np.count_nonzero(names == "paris")
    where    np.where(names < "m", price * 2, price)
    compare  names == "paris"

Run with the NumPy the oracle pins (numpy-version.txt; 2.5.4), on Windows (the pin is set through kernel32):
    python benchmark/fusion/probes/evaltext_bench.py > py.out
NS_PROBE_AFFINITY (hex) overrides the default mask 0x14, as in the C# side.
Output: one line per cell, kind,layout,n,tree,numpy_us.
"""
import ctypes
import os
import time

import numpy as np
from numpy.dtypes import StringDType

# Pin the process before any data exists, the same CPUs as the C# side, so both measure the same cores.
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessAffinityMask.restype = ctypes.c_int
assert k32.SetProcessAffinityMask(k32.GetCurrentProcess(), int(os.environ.get("NS_PROBE_AFFINITY") or "14", 16))

WORDS = ["paris", "london", "berlin", "madrid", "rome", "vienna", "oslo", "lisbon"]
M64 = (1 << 64) - 1


def best_us(f, n):
    """Best per-call time of f in microseconds: two warm-up bursts around a pause (the C# side's schedule, so both sides
    run the same number of calls), then the minimum over 7 timed rounds of a size-dependent repetition count."""
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


for kind in ["U", "S", "T"]:
    for layout in ["contig", "strided"]:
        for n in [1_000, 100_000, 1_000_000]:
            # A strided operand is every second element of a 2n array, as on the C# side.
            total = n if layout == "contig" else 2 * n
            state, strs = 12345, []
            for _ in range(total):
                state = (state * 6364136223846793005 + 1442695040888963407) & M64
                strs.append(WORDS[(state >> 33) % len(WORDS)])
            base = np.array(strs)
            if kind == "S":
                base = base.astype("S6")
            elif kind == "T":
                base = base.astype(StringDType())
            age_b = np.arange(total, dtype=np.int64) % 97
            price_b = np.arange(total, dtype=np.float64) * 0.5
            names = base if layout == "contig" else base[::2]
            age = age_b if layout == "contig" else age_b[::2]
            price = price_b if layout == "contig" else price_b[::2]
            # S compares with bytes, U and StringDType with str (NumPy has no S-with-str loop).
            paris = b"paris" if kind == "S" else "paris"
            m = b"m" if kind == "S" else "m"
            trees = [
                ("filter", lambda: (names == paris) & (age > 30)),
                ("count", lambda: np.count_nonzero(names == paris)),
                ("where", lambda: np.where(names < m, price * 2, price)),
                ("compare", lambda: names == paris),
            ]
            for name, f in trees:
                print(f"{kind},{layout},{n},{name},{best_us(f, n):.2f}", flush=True)
