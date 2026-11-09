"""NumPy twin of evaltext_e2_bench.cs (docs/plans/evaluate-text.md, phase E2).

Times NumPy's own statements for the eight phase-E2 trees over the identical data (the same 64-bit LCG picks the same
city names and digit texts), pinned to the same CPUs, best of 7 rounds after a paused double warm-up.

    truth_and    np.logical_and(names, age > 30)
    truth_where  np.where(names, price * 2, price)
    truth_count  np.count_nonzero(names)
    strlen       (np.strings.str_len(names) > 5) & (age > 30)
    starts       np.count_nonzero(np.strings.startswith(names, "pa") & (age > 30))
    find         np.where(np.strings.find(names, "o") >= 0, price, 0)
    isalpha      np.strings.isalpha(names)
    cast         digits.astype(np.int64) * 2

Run with the NumPy the oracle pins (numpy-version.txt; 2.5.4), on Windows (the pin is set through kernel32):
    python benchmark/fusion/probes/evaltext_e2_bench.py > py.out
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
    """Best per-call time of f in microseconds: two warm-up bursts around a pause (the C# side's schedule), then the
    minimum over 7 timed rounds of a size-dependent repetition count."""
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


def as_kind(texts, kind, width):
    """The text array of a kind: U as built, S bytes of ``width``, or StringDType."""
    a = np.array(texts)
    if kind == "S":
        return a.astype(width)
    if kind == "T":
        return a.astype(StringDType())
    return a


for kind in ["U", "S", "T"]:
    for layout in ["contig", "strided"]:
        for n in [1_000, 100_000, 1_000_000]:
            # A strided operand is every second element of a 2n array, as on the C# side.
            total = n if layout == "contig" else 2 * n
            state, strs, digit_texts = 12345, [], []
            for _ in range(total):
                state = (state * 6364136223846793005 + 1442695040888963407) & M64
                strs.append(WORDS[(state >> 33) % len(WORDS)])
                digit_texts.append(str((state >> 33) % 1000))
            base_names = as_kind(strs, kind, "S6")
            base_digits = as_kind(digit_texts, kind, "S3")
            age_b = np.arange(total, dtype=np.int64) % 97
            price_b = np.arange(total, dtype=np.float64) * 0.5
            step = slice(None) if layout == "contig" else slice(None, None, 2)
            names, digits, age, price = base_names[step], base_digits[step], age_b[step], price_b[step]
            # S compares with bytes, U and StringDType with str (NumPy has no S-with-str loop).
            pa = b"pa" if kind == "S" else "pa"
            o = b"o" if kind == "S" else "o"
            trees = [
                ("truth_and", lambda: np.logical_and(names, age > 30)),
                ("truth_where", lambda: np.where(names, price * 2, price)),
                ("truth_count", lambda: np.count_nonzero(names)),
                ("strlen", lambda: (np.strings.str_len(names) > 5) & (age > 30)),
                ("starts", lambda: np.count_nonzero(np.strings.startswith(names, pa) & (age > 30))),
                ("find", lambda: np.where(np.strings.find(names, o) >= 0, price, 0)),
                ("isalpha", lambda: np.strings.isalpha(names)),
                ("cast", lambda: digits.astype(np.int64) * 2),
            ]
            for name, f in trees:
                print(f"{kind},{layout},{n},{name},{best_us(f, n):.2f}", flush=True)
