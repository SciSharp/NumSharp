"""strings_bench.py — NumPy side of the benchmark/strings subsystem (strings phase 2, P2.7).

Times NumPy 2.4.2's text computations over the three string dtypes — S (bytes), U (UCS-4) and StringDType (T) — on
data kinds that separate the cases the kernels branch on, at 1K and 100K elements (1M for a core set), plus stepped
views. It WRITES the data first (np.save into the directory NUMSHARP_STRINGS_BENCH_DATA names; a StringDType array as
NumPy's pickle), so the NumSharp side (strings_bench.cs) times the SAME arrays, loaded with np.load_npy.

Output: one `key\\tms` line per cell, key `{op}|{kind}|{n}` (identical keys on both sides); a call NumPy refuses for a
kind emits nothing (the cell has no comparand). Driven by strings_sheet.py; standalone:
    NUMSHARP_STRINGS_BENCH_DATA=<dir> python benchmark/strings/strings_bench.py
"""
import os
import sys
import time
import warnings

import numpy as np

# Host pin: inherit the driving sheet's core when driven; honor NUMSHARP_BENCHMARK_AFFINITY standalone.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from benchmark_host import pin_current_process_from_env  # noqa: E402

pin_current_process_from_env()
warnings.simplefilter("ignore")   # np.save of a StringDType array warns that it pickles

T = np.dtypes.StringDType()
TNA = np.dtypes.StringDType(na_object=np.nan)
SIZES = (1_000, 100_000)
CORE_SIZE = 1_000_000
# The ops timed at 1M too (on u8 / t8): the most used text operations, one per family.
CORE_OPS = {"eq", "add", "sort", "str_len", "upper", "astype_f8"}
NON_ASCII = np.array(list("éßαωא中😀Ж"))


def best_ms(f):
    """Best-of-rounds milliseconds per call: a warm-up whose timed tail is the pilot, then ~200 ms of batched rounds.
    A call over 20 ms runs ~1 s of single-call rounds (5 to 100 of them) rather than the other subsystems' fixed 100:
    this sheet has a hundred such cells per side (NumPy's object-path functions, the 1M core), and 100 rounds each would
    take half an hour. The C# side applies the same policy."""
    depth = os.environ.get("NUMSHARP_BENCHMARK_DEPTH", "measure").lower()
    if depth == "pass":
        t = time.perf_counter()
        f()
        return (time.perf_counter() - t) * 1000.0
    wm = 3 if depth == "light" else 5
    f()
    t = time.perf_counter()
    for _ in range(wm - 1):
        f()
    per_call = (time.perf_counter() - t) * 1000.0 / (wm - 1)
    slow = per_call > 20.0
    it = 1 if slow else max(1, min(1_000_000, int(round(1.0 / max(per_call, 1e-6)))))
    budget = 200.0 / (6 if depth == "light" else 1)
    rd = (3 if depth == "light" else max(5, min(100, int(1000.0 / per_call) + 1))) if slow         else max(1, int(budget / max(it * per_call, 1e-9)) + 1)
    best = float("inf")
    for _ in range(rd):
        t = time.perf_counter()
        for _ in range(it):
            f()
        best = min(best, (time.perf_counter() - t) / it)
    return best * 1000.0


def text_array(rng, n, lo, hi, non_ascii=0.0):
    """n random lowercase texts of lo..hi characters as a U array, built from a code-point matrix (a row's zeros past
    its length are the text's end); with non_ascii, that share of the characters is drawn from NON_ASCII."""
    lengths = rng.integers(lo, hi + 1, n)
    codes = rng.integers(97, 123, (n, hi)).astype(np.uint32)
    if non_ascii:
        mask = rng.random((n, hi)) < non_ascii
        pool = np.array([ord(c) for c in NON_ASCII], dtype=np.uint32)
        codes[mask] = pool[rng.integers(0, len(pool), int(mask.sum()))]
    codes[np.arange(hi)[None, :] >= lengths[:, None]] = 0
    return codes.view(f"<U{hi}").reshape(n).copy()   # an owning U array, not a view of the code points


def make_kinds(n, seed, only=None):
    """The data kinds at n elements (each a fresh deterministic draw); only limits them to a subset."""
    rng = np.random.default_rng(seed)
    u8 = text_array(rng, n, 1, 8)
    u8n = text_array(rng, n, 1, 8, non_ascii=0.2)
    u40 = text_array(rng, n, 20, 40)
    digits = rng.integers(0, 1_000_000, n).astype("U")
    tna = u8.astype(TNA)
    tna[rng.random(n) < 0.1] = np.nan   # 10 % missing values
    kinds = {
        "u8": u8,
        "u8n": u8n,
        "u40": u40,
        "s8": u8.astype("S"),
        "t8": u8.astype(T),
        "t40": u40.astype(T),
        "tna": tna,
        "u8d": digits,
        "s8d": digits.astype("S"),
        "t8d": digits.astype(T),
    }
    return {k: v for k, v in kinds.items() if only is None or k in only}


def stepped(kinds2n):
    """Stepped views (a[::2]) of the 2n-element u8 / t8 arrays (the strided layout), each with its 2n-element base."""
    return {"u8@step2": (kinds2n["u8"][::2], kinds2n["u8"]), "t8@step2": (kinds2n["t8"][::2], kinds2n["t8"])}


def is_bytes(a):
    return a.dtype.kind == "S"


def needle(a, text):
    """A search / fill argument of the operand's text class: bytes for S, str otherwise."""
    return text.encode("ascii") if is_bytes(a) else text


S_TABLE = bytes.maketrans(b"abc", b"xyz")
U_TABLE = {97: "x", 98: "y", 99: "z"}


def ops_for(kind, a, b):
    """The (op, call) pairs timed for one data kind; b is the operand's partner (a rolled by one)."""
    text_kind = kind.split("@")[0]
    ops = [
        ("eq", lambda: a == b),
        ("lt", lambda: a < b),
        ("add", lambda: np.strings.add(a, b)),
        ("mul3", lambda: np.strings.multiply(a, 3)),
        ("max", lambda: np.max(a)),
        ("sort", lambda: np.sort(a)),
        ("argsort", lambda: np.argsort(a)),
        ("unique", lambda: np.unique(a)),
        ("isin", lambda: np.isin(a, a[:100])),
        ("str_len", lambda: np.strings.str_len(a)),
        ("find", lambda: np.strings.find(a, needle(a, "a"))),
        ("count", lambda: np.strings.count(a, needle(a, "a"))),
        ("startswith", lambda: np.strings.startswith(a, needle(a, "a"))),
        ("strip", lambda: np.strings.strip(a, needle(a, "ab"))),
        ("replace", lambda: np.strings.replace(a, needle(a, "a"), needle(a, "xy"))),
        ("center", lambda: np.strings.center(a, 12)),
        ("zfill", lambda: np.strings.zfill(a, 10)),
        ("partition", lambda: np.strings.partition(a, needle(a, "a"))),
        ("slice", lambda: np.strings.slice(a, 1, 4)),
        ("upper", lambda: np.strings.upper(a)),
        ("title", lambda: np.strings.title(a)),
        ("isalpha", lambda: np.strings.isalpha(a)),
        ("translate", lambda: np.strings.translate(a, S_TABLE if is_bytes(a) else U_TABLE)),
    ]
    # NumPy's StringDType sum is quadratic (155 s for 100K texts of 20-40 characters): 1K only (see main).
    if a.dtype.kind == "T":
        ops.append(("sum", lambda: np.sum(a)))
    if not is_bytes(a):
        fmt = np.full(a.shape, "<%s>", dtype=a.dtype if a.dtype.kind == "T" else "U4")
        ops.append(("mod", lambda: np.strings.mod(fmt, a)))
        ops.append(("encode", lambda: np.strings.encode(a, "utf-8")))
    else:
        ops.append(("decode", lambda: np.strings.decode(a, "utf-8")))
    if text_kind in ("u8d", "s8d", "t8d"):
        ops = [("astype_f8", lambda: a.astype(np.float64)), ("isdigit", lambda: np.strings.isdigit(a))]
    return ops


def main():
    data_dir = os.environ.get("NUMSHARP_STRINGS_BENCH_DATA")
    if not data_dir:
        print("NUMSHARP_STRINGS_BENCH_DATA must name a directory for the shared data", file=sys.stderr)
        sys.exit(2)
    os.makedirs(data_dir, exist_ok=True)

    sets = []
    stored = {}   # (kind, n) -> the array saved for the C# side
    for n in SIZES:
        kinds = make_kinds(n, seed=n)
        for kind, a in kinds.items():
            stored[(kind, n)] = a
        # A stepped kind saves its 2n-element BASE; the C# side takes the same [::2] view of it.
        for kind, (view, base) in stepped(make_kinds(2 * n, seed=2 * n + 1, only=("u8", "t8"))).items():
            kinds[kind] = view
            stored[(kind, n)] = base
        sets.append((n, kinds))
    core = make_kinds(CORE_SIZE, seed=CORE_SIZE, only=("u8", "t8", "u8d", "t8d"))
    for kind, a in core.items():
        stored[(kind, CORE_SIZE)] = a
    sets.append((CORE_SIZE, core))

    # The data first, every set, so the C# side reads exactly these arrays.
    for (kind, n), a in stored.items():
        np.save(os.path.join(data_dir, f"{kind.replace('@', '_')}_{n}.npy"), a, allow_pickle=True)

    for n, kinds in sets:
        for kind, a in kinds.items():
            b = np.roll(a, 1)
            for op, call in ops_for(kind, a, b):
                if (n == CORE_SIZE and op not in CORE_OPS) or (op == "sum" and n > SIZES[0]):
                    continue
                try:
                    call()
                except Exception:   # NumPy refuses this op for this kind: no comparand
                    continue
                print(f"{op}|{kind}|{n}\t{best_ms(call):.6f}", flush=True)


if __name__ == "__main__":
    main()
