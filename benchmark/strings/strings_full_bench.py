"""strings_full_bench.py - NumPy side of the full strings / chararray evaluation (benchmark/strings, the "full" sheet).

Every text operation NumSharp exposes, over S (bytes), U (UCS-4) and StringDType (T, with and without a NaN NA), timed
against NumPy 2.4.2 on the SAME arrays: the casts between every text class and the numbers (what NumPy's casting impls
run through get_strided_loop), every NDStringCopy route (dense block, raw 1-D overlap, element grid, the 3-D unit copy,
S <-> U, numbers <-> text, masked copies, scalar / broadcast / mask stores), the dtype layer (resolve_descriptors:
np.dtype, can_cast, promote_types, result_type), the ufuncs, reductions, sorting / searching / sets, all 46 np.strings
names, the np.char functions, the chararray members and operators, indexing, creation, printing and .npy / text I/O.

Modes (NUMSHARP_STRINGS_FULL_MODE, or argv[1]):
    write    writes every array both sides time into NUMSHARP_STRINGS_FULL_DATA (np.save; a StringDType array as NumPy's
             pickle), so the C# twin (strings_full_bench.cs) loads byte-identical data with np.load_npy.
    time     times every case and prints one line each:
                 key \t ms \t sig          sig = a result signature both sides compute (dtype, shape, a content number)
                 key \t REFUSED \t text    NumPy raised: no comparand
                 key \t CRASH \t text      NumPy 2.4.2 segfaults on this case (NUMPY_CRASHES): not run
                 key \t SKIP \t text       NumPy cannot finish it in reasonable time (NUMPY_STALLS): not run
    key = {group}.{op}|{kind}|{n}.

Environment:
    NUMSHARP_STRINGS_FULL_DATA     the shared data directory (or argv[2])
    NUMSHARP_STRINGS_FULL_FILTER   comma-separated key prefixes; an entry ending in '$' matches one key exactly
    NUMSHARP_STRINGS_FULL_SIZES    comma list of sizes (default 1000,100000)
    NUMSHARP_BENCHMARK_DEPTH       pass | light | measure (default measure), the house depth variable
    NUMSHARP_BENCHMARK_AFFINITY    the host pin the driving sheet exported (honored standalone too)
Driven by strings_full_sheet.py; standalone:
    NUMSHARP_STRINGS_FULL_DATA=<dir> python strings_full_bench.py write
    NUMSHARP_STRINGS_FULL_DATA=<dir> python strings_full_bench.py time
"""
import io
import os
import sys
import time
import warnings

import numpy as np

# Host pin: inherit the driving sheet's cores when driven; honor NUMSHARP_BENCHMARK_AFFINITY standalone.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from benchmark_host import pin_current_process_from_env  # noqa: E402

pin_current_process_from_env()
warnings.simplefilter("ignore")   # np.save of a StringDType array warns that it pickles

T = np.dtypes.StringDType()
TNA = np.dtypes.StringDType(na_object=np.nan)
NON_ASCII = np.array([ord(c) for c in "éßαωא中😀Ж"], dtype=np.uint32)
SIZES = tuple(int(s) for s in os.environ.get("NUMSHARP_STRINGS_FULL_SIZES", "1000,100000").split(","))


# ----------------------------------------------------------------------------------------------- data generation
def text_array(rng, n, lo, hi, non_ascii=0.0):
    """n random lowercase texts of lo..hi characters as an owning U array, built from a code-point matrix (a row's zeros
    past its length end the text); with non_ascii, that share of the characters comes from NON_ASCII (BMP and astral)."""
    lengths = rng.integers(lo, hi + 1, n)
    codes = rng.integers(97, 123, (n, hi)).astype(np.uint32)
    if non_ascii:
        mask = rng.random((n, hi)) < non_ascii
        codes[mask] = NON_ASCII[rng.integers(0, len(NON_ASCII), int(mask.sum()))]
    codes[np.arange(hi)[None, :] >= lengths[:, None]] = 0
    return codes.view(f"<U{hi}").reshape(n).copy()


def word_array(rng, n, sep):
    """2..5 lowercase words of 2..7 letters joined by sep (a U array)."""
    out = []
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    nw = rng.integers(2, 6, n)
    for i in range(n):
        ws = []
        for _ in range(nw[i]):
            ws.append("".join(letters[rng.integers(0, 26, rng.integers(2, 8))]))
        out.append(sep.join(ws))
    return np.array(out)


def write(datadir):
    """Writes every data kind at every size into datadir as {kind}_{n}.npy (np.save, allow_pickle for StringDType): the
    text kinds (short / non-ASCII / long / words / lines / digits / float texts in S, U and T, T with a NaN NA), the 2n
    bases of the stepped views, the number arrays cast to text, and the mask / index / count arrays."""
    os.makedirs(datadir, exist_ok=True)
    for n in SIZES:
        rng = np.random.default_rng(n)
        u8 = text_array(rng, n, 1, 8)
        u8n = text_array(rng, n, 1, 8, non_ascii=0.2)
        u40 = text_array(rng, n, 20, 40)
        uw = word_array(rng, n, " ")
        uln = word_array(rng, n, "\n")
        d2 = rng.integers(0, 100, n).astype("U")
        d6 = rng.integers(0, 1_000_000, n).astype("U")
        fl = (rng.random(n) * 1000.0).round(4).astype("U")
        tna = u8.astype(TNA)
        tna[rng.random(n) < 0.1] = np.nan
        rng2 = np.random.default_rng(2 * n + 1)
        u8_2n = text_array(rng2, 2 * n, 1, 8)
        kinds = {
            "u8": u8, "u8n": u8n, "u40": u40, "uw": uw, "uln": uln, "ud2": d2, "ud6": d6, "ufl": fl,
            "s8": u8.astype("S"), "s40": u40.astype("S"), "sw": uw.astype("S"), "sln": uln.astype("S"),
            "sd2": d2.astype("S"), "sd6": d6.astype("S"), "sfl": fl.astype("S"),
            "t8": u8.astype(T), "t8n": u8n.astype(T), "t40": u40.astype(T), "tw": uw.astype(T), "tln": uln.astype(T),
            "td2": d2.astype(T), "td6": d6.astype(T), "tfl": fl.astype(T), "tna": tna,
            "u8_2n": u8_2n, "s8_2n": u8_2n.astype("S"), "t8_2n": u8_2n.astype(T),
            "nb1": rng.random(n) < 0.5, "ni1": rng.integers(-128, 128, n).astype(np.int8),
            "ni4": rng.integers(-2**31, 2**31, n).astype(np.int32), "ni8": rng.integers(-2**62, 2**62, n),
            "nu8": rng.integers(0, 2**63, n).astype(np.uint64) * np.uint64(2),
            "nf2": (rng.standard_normal(n) * 100).astype(np.float16),
            "nf4": (rng.standard_normal(n) * 1e3).astype(np.float32),
            "nf8": rng.standard_normal(n) * 1e3,
            "nc16": rng.standard_normal(n) + 1j * rng.standard_normal(n),
            "mask": rng.random(n) < 0.5, "idx": rng.integers(0, n, n), "cnt": rng.integers(0, 4, n),
        }
        for name, a in kinds.items():
            np.save(os.path.join(datadir, f"{name}_{n}.npy"), a, allow_pickle=True)
    print("written", file=sys.stderr)


# ----------------------------------------------------------------------------------------------- timing
def best_ms(f):
    """Best-of-rounds milliseconds per call, the strings sheet's policy: a warm-up whose timed tail is the pilot, then
    ~200 ms of batched rounds; a call over 20 ms runs ~1 s of single-call rounds (5 to 100 of them)."""
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
    if slow:
        rd = 3 if depth == "light" else max(5, min(100, int(1000.0 / per_call) + 1))
    else:
        rd = max(1, int(budget / max(it * per_call, 1e-9)) + 1)
    best = float("inf")
    for _ in range(rd):
        t = time.perf_counter()
        for _ in range(it):
            f()
        best = min(best, (time.perf_counter() - t) / it)
    return best * 1000.0


# ----------------------------------------------------------------------------------------------- signatures
def dname(dt):
    """The dtype's signature name: kind plus item size (U32 for <U8), T for StringDType; the C# twin prints the same."""
    if dt.kind == "T":
        return "T"
    return f"{dt.kind}{dt.itemsize}"


def sig(r):
    """A result signature: dtype + shape + a content number (text: total length of the texts and the NA count; bool:
    the count of True; number: the sum). Both sides compute the same."""
    if r is None:
        return "none"
    if isinstance(r, np.dtype):
        return f"dtype:{r}"
    if isinstance(r, np.ndarray) and r.ndim == 0:
        return sig(r[()])
    if isinstance(r, tuple):
        return ";".join(sig(x) for x in r)
    if isinstance(r, (list,)):
        return f"list{len(r)}:{_list_items(r)}"
    if isinstance(r, (bool, np.bool_)):
        return f"bool:{int(r)}"
    if isinstance(r, (str, bytes, np.str_, np.bytes_)):
        return f"str:{len(r)}"
    if isinstance(r, (int, float, complex, np.number)):
        v = complex(r)
        return f"num:{v.real + v.imag:.6g}"
    if isinstance(r, np.ndarray):
        head = f"{dname(r.dtype)}[{'x'.join(str(s) for s in r.shape)}]"
        k = r.dtype.kind
        if k in "SUT":
            if k == "T" and getattr(r.dtype, "na_object", None) is not None and r.size:
                na = np.isnan(r) if r.dtype.na_object is not None else np.zeros(r.shape, bool)
                tot = int(np.strings.str_len(r[~na]).sum()) if (~na).any() else 0
                return f"{head}:{tot}/{int(na.sum())}"
            return f"{head}:{int(np.strings.str_len(r).sum()) if r.size else 0}"
        if k == "b":
            return f"{head}:{int(r.sum())}"
        if k in "iu":
            return f"{head}:{int(r.astype(np.float64).sum()):d}" if r.size < 2**20 else f"{head}:n"
        if k in "fc":
            v = complex(np.nansum(r.astype(np.complex128)))
            return f"{head}:{v.real + v.imag:.6g}"
        if k == "O":
            return f"{head}:{_list_items(r.ravel().tolist())}"
        return head
    return type(r).__name__


def _list_items(lst):
    """The item count of a list result: an inner list counts its length, anything else one (the C# twin's ListItems)."""
    tot = 0
    for x in lst:
        if isinstance(x, list):
            tot += len(x)
        else:
            tot += 1
    return tot


# ----------------------------------------------------------------------------------------------- cases
def load(datadir, name, n):
    """The array write() saved as {name}_{n}.npy."""
    return np.load(os.path.join(datadir, f"{name}_{n}.npy"), allow_pickle=True)


def two_d(n):
    """The 2-D shape a 1-D data kind is viewed in for the transposed / broadcast cases: rows * columns == n."""
    return (25, 40) if n == 1000 else (250, 400)


def is_s(a):
    """True for an S (bytes) array, whose text arguments are bytes."""
    return a.dtype.kind == "S"


def nd(a, s):
    """A text argument of a's class: bytes for S, str otherwise."""
    return s.encode("ascii") if is_s(a) else s


def cases(datadir, n):
    """Every case at size n as (key, call) pairs, key = {group}.{op}|{kind}|{n}, in the order the C# twin builds them;
    arguments and destination arrays are built here, outside the timed calls."""
    L = {}
    def get(name):
        """A data kind at size n, loaded once."""
        if name not in L:
            L[name] = load(datadir, name, n)
        return L[name]

    def kind(k):
        """A data kind, possibly a view: name@2 = [::2] of the 2n base, name@r = [::-1], name@T = 2-D transposed."""
        if k.endswith("@2"):
            return get(k[:-2] + "_2n")[::2]
        if k.endswith("@r"):
            return get(k[:-2])[::-1]
        if k.endswith("@T"):
            return get(k[:-2]).reshape(two_d(n)).T
        return get(k)

    out = []
    def add(op, k, fn):
        """Appends the case {op}|{k}|{n}."""
        out.append((f"{op}|{k}|{n}", fn))

    r, c = two_d(n)
    mask, idx, cnt = get("mask"), get("idx"), get("cnt")
    S_TABLE = bytes.maketrans(b"abc", b"xyz")
    U_TABLE = {97: "x", 98: "y", 99: "z"}

    # ---------------------------------------------------------------- cast: astype between every text class / numbers
    text_src = ["s8", "u8", "t8", "tna", "u8n", "t8n", "s40", "u40", "t40"]
    text_dst = {"S": "S", "U": "U", "T": T, "S4": "S4", "U4": "U4", "S16": "S16", "U16": "U16", "same": None}
    for k in text_src:
        for dn, dt in text_dst.items():
            def f(a=kind(k), dt=dt):
                """astype into the destination descriptor (None = the source's own: a copy)."""
                return a.astype(a.dtype if dt is None else dt)
            add(f"cast.to_{dn}", k, f)
    num_dst = {"b1": np.bool_, "i1": np.int8, "i2": np.int16, "i4": np.int32, "i8": np.int64, "u1": np.uint8,
               "u2": np.uint16, "u4": np.uint32, "u8": np.uint64, "f2": np.float16, "f4": np.float32,
               "f8": np.float64, "c16": np.complex128}
    for cls in "sut":
        for dn, dt in num_dst.items():
            add(f"cast.parse_{dn}", f"{cls}d2", lambda a=kind(f"{cls}d2"), dt=dt: a.astype(dt))
        for dn in ("i4", "i8", "u4", "u8", "f4", "f8"):
            add(f"cast.parse_{dn}", f"{cls}d6", lambda a=kind(f"{cls}d6"), dt=num_dst[dn]: a.astype(dt))
        for dn in ("f2", "f4", "f8", "c16"):
            add(f"cast.parse_{dn}", f"{cls}fl", lambda a=kind(f"{cls}fl"), dt=num_dst[dn]: a.astype(dt))
    for k in ("nb1", "ni1", "ni4", "ni8", "nu8", "nf2", "nf4", "nf8", "nc16"):
        for dn, dt in (("S", "S"), ("U", "U"), ("T", T)):
            add(f"cast.fmt_{dn}", k, lambda a=kind(k), dt=dt: a.astype(dt))
    # layouts through the cast loops
    for k in ("u8@2", "u8@r", "u8@T"):
        add("cast.to_S", k, lambda a=kind(k): a.astype("S"))
        add("cast.to_T", k, lambda a=kind(k): a.astype(T))
    for k in ("t8@2", "t8@r", "t8@T"):
        add("cast.to_U16", k, lambda a=kind(k): a.astype("U16"))
        add("cast.to_S16", k, lambda a=kind(k): a.astype("S16"))
    for k in ("s8@2", "s8@r"):
        add("cast.to_U", k, lambda a=kind(k): a.astype("U"))
    add("cast.fmt_U", "nf8@r", lambda a=get("nf8")[::-1]: a.astype("U"))
    add("cast.parse_f8", "ud6@r", lambda a=get("ud6")[::-1]: a.astype(np.float64))

    # ---------------------------------------------------------------- copy: NDStringCopy routes
    for k in ("s8", "u8", "t8", "tna", "s40", "u40", "t40"):
        a = kind(k)
        add("copy.copy", k, lambda a=a: a.copy())
        add("copy.copy_rev", k, lambda a=a: a[::-1].copy())
        add("copy.copy_T", k, lambda a=a: a.reshape(r, c).T.copy())
        dst = np.empty_like(a)
        add("copy.copyto", k, lambda a=a, dst=dst: np.copyto(dst, a))
        dstw = np.empty_like(a)
        add("copy.copyto_where", k, lambda a=a, dst=dstw: np.copyto(dst, a, where=mask))
        dst2 = np.empty(2 * n, a.dtype)
        add("copy.set_step", k, lambda a=a, dst=dst2: dst.__setitem__(slice(None, None, 2), a))
        dstm = np.empty((r, c), a.dtype)
        row = a[:c].copy()
        add("copy.set_bcast", k, lambda row=row, dst=dstm: dst.__setitem__(Ellipsis, row))
        dsts = np.empty_like(a)
        add("copy.set_scalar", k, lambda a=a, dst=dsts: dst.__setitem__(Ellipsis, nd(a, "xyz")))
        dstf = np.empty_like(a)
        add("copy.fill", k, lambda a=a, dst=dstf: dst.fill(nd(a, "xyz")))
        dsto = a.copy()
        add("copy.shift1", k, lambda dst=dsto: dst.__setitem__(slice(1, None), dst[:-1]))
        dstk = a.copy()
        add("copy.set_mask", k, lambda a=a, dst=dstk: dst.__setitem__(mask, nd(a, "zz")))
    for k in ("s8", "u8", "t8"):
        add("copy.copy_step", k + "@2", lambda a=kind(k + "@2"): a.copy())
        add("copy.ascontig_T", k + "@T", lambda a=kind(k + "@T"): np.ascontiguousarray(a))
        # three effective axes: past the element-grid route (at most two), NDStringCopy's general unit-view copy
        shp3 = (10, 10, 10) if n == 1000 else (40, 50, 50)
        add("copy.copy_3d", k, lambda a=kind(k), shp3=shp3: a.reshape(shp3).transpose(2, 0, 1).copy())
    u8, s8, t8 = get("u8"), get("s8"), get("t8")
    pairs = [
        ("copyto_U8toU16", "u8", u8, np.empty(n, "U16"), "same_kind"),
        ("copyto_U8toU4", "u8", u8, np.empty(n, "U4"), "unsafe"),
        ("copyto_S8toU8", "s8", s8, np.empty(n, "U8"), "same_kind"),
        ("copyto_U8toS8", "u8", u8, np.empty(n, "S8"), "unsafe"),
        ("copyto_S8toS16", "s8", s8, np.empty(n, "S16"), "same_kind"),
        ("copyto_U8toT", "u8", u8, np.empty(n, T), "same_kind"),
        ("copyto_TtoU8", "t8", t8, np.empty(n, "U8"), "unsafe"),
        ("copyto_S8toT", "s8", s8, np.empty(n, T), "same_kind"),
        ("copyto_TtoS8", "t8", t8, np.empty(n, "S8"), "unsafe"),
        ("copyto_i8toU", "ni8", get("ni8"), np.empty(n, "U21"), "unsafe"),
        ("copyto_f8toU", "nf8", get("nf8"), np.empty(n, "U32"), "unsafe"),
        ("copyto_Utof8", "ufl", get("ufl"), np.empty(n, np.float64), "unsafe"),
        ("copyto_Utoi8", "ud6", get("ud6"), np.empty(n, np.int64), "unsafe"),
        ("copyto_Ttof8", "tfl", get("tfl"), np.empty(n, np.float64), "unsafe"),
        ("copyto_i8toT", "ni8", get("ni8"), np.empty(n, T), "unsafe"),
    ]
    for op, k, a, dst, casting in pairs:
        add(f"copy.{op}", k, lambda a=a, dst=dst, casting=casting: np.copyto(dst, a, casting=casting))

    # ---------------------------------------------------------------- meta: resolve_descriptors / dtype layer (O(1))
    if n == SIZES[0]:
        dU8, dS8, dS3, dU5, dT, di8 = np.dtype("U8"), np.dtype("S8"), np.dtype("S3"), np.dtype("U5"), T, np.dtype("i8")
        dU8b = np.dtype("U8")
        meta = [
            ("dtype_U8", lambda: np.dtype("U8")), ("dtype_S8", lambda: np.dtype("S8")),
            ("dtype_T", lambda: np.dtypes.StringDType()), ("dtype_TNA", lambda: np.dtypes.StringDType(na_object=np.nan)),
            ("can_cast_U8_S8", lambda: np.can_cast(dU8, dS8)), ("can_cast_S8_U8", lambda: np.can_cast(dS8, dU8)),
            ("can_cast_U8_T", lambda: np.can_cast(dU8, dT)), ("can_cast_T_U8_sk", lambda: np.can_cast(dT, dU8, "same_kind")),
            ("can_cast_i8_U8", lambda: np.can_cast(di8, dU8)), ("can_cast_U8_i8_u", lambda: np.can_cast(dU8, di8, "unsafe")),
            ("promote_S3_U5", lambda: np.promote_types(dS3, dU5)), ("promote_U8_T", lambda: np.promote_types(dU8, dT)),
            ("result_type_uS", lambda: np.result_type(u8, s8)), ("result_type_tU", lambda: np.result_type(t8, u8)),
            ("dtype_eq", lambda: dU8 == dU8b),
        ]
        for op, fn in meta:
            out.append((f"meta.{op}|-|0", fn))

    # ---------------------------------------------------------------- ufunc: comparisons, add, multiply, maximum
    uf_kinds = ["s8", "u8", "t8", "tna", "u8n", "t8n", "s40", "u40", "t40", "s8@2", "u8@2", "t8@2", "u8@r", "t8@r"]
    for k in uf_kinds:
        a = kind(k)
        b = np.roll(a, 1)
        sc = nd(a, "abc")
        add("ufunc.eq", k, lambda a=a, b=b: a == b)
        add("ufunc.ne", k, lambda a=a, b=b: a != b)
        add("ufunc.lt", k, lambda a=a, b=b: a < b)
        add("ufunc.le", k, lambda a=a, b=b: a <= b)
        add("ufunc.gt", k, lambda a=a, b=b: a > b)
        add("ufunc.ge", k, lambda a=a, b=b: a >= b)
        add("ufunc.eq_sc", k, lambda a=a, sc=sc: a == sc)
        add("ufunc.add", k, lambda a=a, b=b: np.add(a, b))
        add("ufunc.add_sc", k, lambda a=a: np.add(a, nd(a, "xy")))
        add("ufunc.mul3", k, lambda a=a: np.strings.multiply(a, 3))
        add("ufunc.mul_arr", k, lambda a=a: np.strings.multiply(a, cnt))
        add("ufunc.maximum", k, lambda a=a, b=b: np.maximum(a, b))
        add("ufunc.minimum", k, lambda a=a, b=b: np.minimum(a, b))
    add("ufunc.eq_TU", "t8", lambda a=t8, b=u8: a == b)
    add("ufunc.add_TU", "t8", lambda a=t8, b=u8: np.add(a, b))

    # ---------------------------------------------------------------- reduce
    core = ["s8", "u8", "t8", "tna", "u40", "t40"]
    for k in core:
        a = kind(k)
        add("reduce.max", k, lambda a=a: np.max(a))
        add("reduce.min", k, lambda a=a: np.min(a))
        add("reduce.argmax", k, lambda a=a: np.argmax(a))
        add("reduce.argmin", k, lambda a=a: np.argmin(a))
        add("reduce.max_ax0", k, lambda a=a: np.max(a.reshape(r, c), axis=0))
        add("reduce.nonzero", k, lambda a=a: np.nonzero(a))
        add("reduce.count_nonzero", k, lambda a=a: np.count_nonzero(a))
        add("reduce.any", k, lambda a=a: np.any(a))
        add("reduce.all", k, lambda a=a: np.all(a))
        if n == SIZES[0]:
            add("reduce.sum", k, lambda a=a: np.sum(a))

    # ---------------------------------------------------------------- sort / search / sets
    for k in core + ["u8n"]:
        a = kind(k)
        b = np.roll(a, 1)
        hay = np.sort(a)
        add("sort.sort", k, lambda a=a: np.sort(a))
        add("sort.argsort", k, lambda a=a: np.argsort(a))
        add("sort.sort_stable", k, lambda a=a: np.sort(a, kind="stable"))
        add("sort.argsort_stable", k, lambda a=a: np.argsort(a, kind="stable"))
        add("sort.partition", k, lambda a=a: np.partition(a, n // 2))
        add("sort.argpartition", k, lambda a=a: np.argpartition(a, n // 2))
        add("sort.searchsorted", k, lambda hay=hay, b=b: np.searchsorted(hay, b))
        add("sort.lexsort", k, lambda a=a, b=b: np.lexsort((a, b)))
        add("sort.unique", k, lambda a=a: np.unique(a))
        add("sort.unique_counts", k, lambda a=a: np.unique(a, return_counts=True))
        add("sort.unique_inverse", k, lambda a=a: np.unique(a, return_inverse=True))
        add("sort.isin", k, lambda a=a: np.isin(a, a[:100]))
        add("sort.intersect1d", k, lambda a=a, b=b: np.intersect1d(a, b))
        add("sort.union1d", k, lambda a=a, b=b: np.union1d(a, b))
        add("sort.setdiff1d", k, lambda a=a, b=b: np.setdiff1d(a, b))
        add("sort.setxor1d", k, lambda a=a, b=b: np.setxor1d(a, b))
        # assume_unique over the raw (random-order) operands: NumPy's concatenated aux is then unsorted, which keeps its
        # StringDType introsort off the heapsort fallback that segfaults (the plain forms above crash NumPy over T).
        add("sort.intersect1d_au", k, lambda a=a, b=b: np.intersect1d(a, b, assume_unique=True))
        add("sort.setxor1d_au", k, lambda a=a, b=b: np.setxor1d(a, b, assume_unique=True))
        add("sort.quantile", k, lambda a=a: np.quantile(a, 0.5, method="lower"))

    # ---------------------------------------------------------------- np.strings
    st = np.strings
    str_kinds = ["s8", "u8", "t8", "tna", "u40", "t40"]
    uni_extra = ["u8n", "t8n"]
    strided = ["u8@2", "t8@2"]
    for k in str_kinds + uni_extra + strided:
        a = kind(k)
        A = lambda s, a=a: nd(a, s)  # noqa: E731
        full = k in str_kinds
        add("strings.str_len", k, lambda a=a: st.str_len(a))
        add("strings.find", k, lambda a=a, x=A("a"): st.find(a, x))
        add("strings.upper", k, lambda a=a: st.upper(a))
        add("strings.isalpha", k, lambda a=a: st.isalpha(a))
        add("strings.replace", k, lambda a=a, x=A("a"), y=A("xy"): st.replace(a, x, y))
        add("strings.strip", k, lambda a=a, x=A("ab"): st.strip(a, x))
        add("strings.center", k, lambda a=a: st.center(a, 12))
        add("strings.startswith", k, lambda a=a, x=A("a"): st.startswith(a, x))
        add("strings.slice", k, lambda a=a: st.slice(a, 1, 4))
        if k in strided:
            continue
        add("strings.lower", k, lambda a=a: st.lower(a))
        add("strings.swapcase", k, lambda a=a: st.swapcase(a))
        add("strings.capitalize", k, lambda a=a: st.capitalize(a))
        add("strings.title", k, lambda a=a: st.title(a))
        for p in ("isdigit", "isspace", "isalnum", "islower", "isupper", "istitle", "isdecimal", "isnumeric"):
            add(f"strings.{p}", k, lambda a=a, p=p: getattr(st, p)(a))
        if not is_s(a):
            add("strings.encode", k, lambda a=a: st.encode(a, "utf-8"))
        if not full:
            continue
        add("strings.rfind", k, lambda a=a, x=A("a"): st.rfind(a, x))
        add("strings.find_rng", k, lambda a=a, x=A("a"): st.find(a, x, 1, 5))
        add("strings.count", k, lambda a=a, x=A("a"): st.count(a, x))
        add("strings.endswith", k, lambda a=a, x=A("a"): st.endswith(a, x))
        if k != "tna":   # the first-character needle of an NA element is undefined (NumPy: Cannot slice null string)
            first = st.slice(a, 0, 1)
            add("strings.index", k, lambda a=a, x=first: st.index(a, x))
            add("strings.rindex", k, lambda a=a, x=first: st.rindex(a, x))
            add("strings.startswith_arr", k, lambda a=a, x=first: st.startswith(a, x))
            add("strings.find_arr", k, lambda a=a, x=first: st.find(a, x))
        add("strings.lstrip", k, lambda a=a, x=A("ab"): st.lstrip(a, x))
        add("strings.rstrip", k, lambda a=a, x=A("ab"): st.rstrip(a, x))
        add("strings.strip_ws", k, lambda a=a: st.strip(a))
        add("strings.ljust", k, lambda a=a: st.ljust(a, 12))
        add("strings.rjust", k, lambda a=a: st.rjust(a, 12))
        add("strings.center_fill", k, lambda a=a, x=A("*"): st.center(a, 12, x))
        add("strings.zfill", k, lambda a=a: st.zfill(a, 10))
        add("strings.expandtabs", k, lambda a=a: st.expandtabs(a))
        add("strings.replace_c1", k, lambda a=a, x=A("a"), y=A("xy"): st.replace(a, x, y, 1))
        add("strings.replace_del", k, lambda a=a, x=A("a"), y=A(""): st.replace(a, x, y))
        add("strings.partition", k, lambda a=a, x=A("a"): st.partition(a, x))
        add("strings.rpartition", k, lambda a=a, x=A("a"): st.rpartition(a, x))
        add("strings.slice_rev", k, lambda a=a: st.slice(a, None, None, -1))
        add("strings.slice_stop", k, lambda a=a: st.slice(a, 3))
        add("strings.translate", k, lambda a=a: st.translate(a, S_TABLE if is_s(a) else U_TABLE))
        if is_s(a):
            add("strings.translate_del", k, lambda a=a: st.translate(a, S_TABLE, b"ab"))
            add("strings.decode", k, lambda a=a: st.decode(a, "utf-8"))
            add("strings.decode_latin1", k, lambda a=a: st.decode(a, "latin-1"))
            fmt = np.full(a.shape, b"<%s>", dtype="S4")
            add("strings.mod", k, lambda a=a, fmt=fmt: st.mod(fmt, a))
        else:
            add("strings.encode_ascii", k, lambda a=a: st.encode(a, "ascii"))
            fmt = np.full(a.shape, "<%s>", dtype=a.dtype if a.dtype.kind == "T" else "U4")
            add("strings.mod", k, lambda a=a, fmt=fmt: st.mod(fmt, a))
    for cls in "sut":
        a = kind(f"{cls}d6")
        fmt = np.full(n, b"%07d" if cls == "s" else "%07d", dtype="S4" if cls == "s" else ("U4" if cls == "u" else T))
        add("strings.mod_int", f"{cls}d6", lambda fmt=fmt, v=get("ni4"): st.mod(fmt, v))
        add("strings.isdigit", f"{cls}d6", lambda a=a: st.isdigit(a))
        add("strings.isnumeric", f"{cls}d6", lambda a=a: st.isnumeric(a))

    # ---------------------------------------------------------------- np.char (the chararray module's functions)
    ch = np.char
    for k in ("s8", "u8", "t8"):
        a = kind(k)
        b = np.roll(a, 1)
        A = lambda s, a=a: nd(a, s)  # noqa: E731
        for op in ("equal", "not_equal", "less", "greater", "less_equal", "greater_equal"):
            add(f"char.{op}", k, lambda a=a, b=b, op=op: getattr(ch, op)(a, b))
        add("char.compare_chararrays", k, lambda a=a, b=b: ch.compare_chararrays(a, b, "==", True))
        add("char.add", k, lambda a=a, b=b: ch.add(a, b))
        add("char.multiply", k, lambda a=a: ch.multiply(a, 3))
        add("char.str_len", k, lambda a=a: ch.str_len(a))
        add("char.find", k, lambda a=a, x=A("a"): ch.find(a, x))
        add("char.count", k, lambda a=a, x=A("a"): ch.count(a, x))
        add("char.upper", k, lambda a=a: ch.upper(a))
        add("char.title", k, lambda a=a: ch.title(a))
        add("char.strip", k, lambda a=a, x=A("ab"): ch.strip(a, x))
        add("char.center", k, lambda a=a: ch.center(a, 12))
        add("char.zfill", k, lambda a=a: ch.zfill(a, 10))
        add("char.replace", k, lambda a=a, x=A("a"), y=A("xy"): ch.replace(a, x, y))
        add("char.isalpha", k, lambda a=a: ch.isalpha(a))
        add("char.partition", k, lambda a=a, x=A("a"): ch.partition(a, x))
        add("char.rpartition", k, lambda a=a, x=A("a"): ch.rpartition(a, x))
        add("char.join", k, lambda a=a, x=A("-"): ch.join(x, a))
        add("char.array", k, lambda a=a: ch.array(a))
        add("char.asarray", k, lambda a=a: ch.asarray(a))
        if is_s(a):
            add("char.decode", k, lambda a=a: ch.decode(a, "utf-8"))
        else:
            add("char.encode", k, lambda a=a: ch.encode(a, "utf-8"))
    for k in ("sw", "uw", "tw"):
        a = kind(k)
        add("char.split", k, lambda a=a: ch.split(a))
        add("char.rsplit", k, lambda a=a: ch.rsplit(a))
        add("char.split_sep", k, lambda a=a, x=nd(a, "a"): ch.split(a, x))
    for k in ("sln", "uln", "tln"):
        a = kind(k)
        add("char.splitlines", k, lambda a=a: ch.splitlines(a))

    # ---------------------------------------------------------------- chararray members and operators
    for k in ("s8", "u8", "s40", "u40", "sw", "uw"):
        a = kind(k)
        cc = ch.array(a)
        cb = ch.array(np.roll(a, 1))
        A = lambda s, a=a: nd(a, s)  # noqa: E731
        add("carr.eq", k, lambda cc=cc, cb=cb: cc == cb)
        add("carr.lt", k, lambda cc=cc, cb=cb: cc < cb)
        add("carr.eq_sc", k, lambda cc=cc, x=A("abc"): cc == x)
        add("carr.add", k, lambda cc=cc, cb=cb: cc + cb)
        add("carr.mul3", k, lambda cc=cc: cc * 3)
        cfmt = ch.array(np.full(n, b"<%s>" if is_s(a) else "<%s>", dtype="S4" if is_s(a) else "U4"))
        add("carr.mod", k, lambda cfmt=cfmt, a=a: cfmt % a)
        add("carr.upper", k, lambda cc=cc: cc.upper())
        add("carr.title", k, lambda cc=cc: cc.title())
        add("carr.strip", k, lambda cc=cc, x=A("ab"): cc.strip(x))
        add("carr.find", k, lambda cc=cc, x=A("a"): cc.find(x))
        add("carr.count", k, lambda cc=cc, x=A("a"): cc.count(x))
        add("carr.startswith", k, lambda cc=cc, x=A("a"): cc.startswith(x))
        add("carr.replace", k, lambda cc=cc, x=A("a"), y=A("xy"): cc.replace(x, y))
        add("carr.center", k, lambda cc=cc: cc.center(12))
        add("carr.zfill", k, lambda cc=cc: cc.zfill(10))
        add("carr.isalpha", k, lambda cc=cc: cc.isalpha())
        add("carr.partition", k, lambda cc=cc, x=A("a"): cc.partition(x))
        add("carr.split", k, lambda cc=cc: cc.split())
        add("carr.translate", k, lambda cc=cc, tb=(S_TABLE if is_s(a) else U_TABLE): cc.translate(tb))
        if is_s(a):
            add("carr.decode", k, lambda cc=cc: cc.decode("utf-8"))
        else:
            add("carr.encode", k, lambda cc=cc: cc.encode("utf-8"))
        add("carr.getitem_fancy", k, lambda cc=cc: cc[idx])
        add("carr.take", k, lambda cc=cc: cc.take(idx))
        add("carr.copy", k, lambda cc=cc: cc.copy())
        add("carr.astype_U16", k, lambda cc=cc: cc.astype("U16"))
        add("carr.argsort", k, lambda cc=cc: cc.argsort())
        add("carr.argmax", k, lambda cc=cc: cc.argmax())
        add("carr.flatten", k, lambda cc=cc: cc.flatten())
        add("carr.tolist", k, lambda cc=cc: cc.tolist())
        add("carr.iter", k, lambda cc=cc: sum(len(x) for x in cc))

    # ---------------------------------------------------------------- indexing / manipulation
    for k in core:
        a = kind(k)
        b = np.roll(a, 1)
        add("index.take", k, lambda a=a: np.take(a, idx))
        add("index.fancy", k, lambda a=a: a[idx])
        add("index.bool", k, lambda a=a: a[mask])
        add("index.where", k, lambda a=a, b=b: np.where(mask, a, b))
        add("index.concat", k, lambda a=a, b=b: np.concatenate((a, b)))
        add("index.stack", k, lambda a=a, b=b: np.stack((a, b)))
        add("index.repeat", k, lambda a=a: np.repeat(a, 2))
        add("index.tile", k, lambda a=a: np.tile(a, 2))
        add("index.flip_copy", k, lambda a=a: np.flip(a).copy())
        add("index.roll", k, lambda a=a: np.roll(a, 1))
        add("index.ravel_T", k, lambda a=a: a.reshape(r, c).T.ravel())
        add("index.compress", k, lambda a=a: np.compress(mask, a))
        dstp = a.copy()
        add("index.put", k, lambda a=a, dst=dstp: np.put(dst, idx, a))

    # ---------------------------------------------------------------- creation / conversion / printing
    for k in core:
        a = kind(k)
        add("create.tolist", k, lambda a=a: a.tolist())
        add("create.array2string", k, lambda a=a: np.array2string(a))
        add("create.repr", k, lambda a=a: repr(a))
        add("create.full_like", k, lambda a=a: np.full_like(a, nd(a, "zz")))
        add("create.zeros_like", k, lambda a=a: np.zeros_like(a))
        add("create.empty_like", k, lambda a=a: np.empty_like(a))
    for k, dt in (("u8", "U8"), ("s8", "S8"), ("t8", T), ("u40", "U40"), ("t40", T)):
        lst = get("u8" if k.endswith("8") else "u40").tolist()
        add("create.from_list", k, lambda lst=lst, dt=dt: np.array(lst, dtype=dt))
        add("create.full", k, lambda dt=dt: np.full(n, "abc", dtype=dt))
        add("create.zeros", k, lambda dt=dt: np.zeros(n, dtype=dt))
        add("create.empty", k, lambda dt=dt: np.empty(n, dtype=dt))
    add("create.from_list_infer", "u8", lambda lst=get("u8").tolist(): np.array(lst))

    # ---------------------------------------------------------------- I/O
    for k in core:
        a = kind(k)
        def save(a=a):
            """np.save into a fresh in-memory buffer; returns the byte count (the C# twin returns its stream length)."""
            buf = io.BytesIO()
            np.save(buf, a, allow_pickle=True)
            return buf.getbuffer().nbytes
        bufb = io.BytesIO()
        np.save(bufb, a, allow_pickle=True)
        raw = bufb.getvalue()
        add("io.save", k, save)
        add("io.load", k, lambda raw=raw: np.load(io.BytesIO(raw), allow_pickle=True))
        if not is_s(a) and k != "tna":
            def savetxt(a=a):
                """np.savetxt with fmt='%s' into a fresh in-memory text buffer; returns the character count."""
                s = io.StringIO()
                np.savetxt(s, a, fmt="%s")
                return len(s.getvalue())
            add("io.savetxt", k, savetxt)
            s = io.StringIO()
            np.savetxt(s, a, fmt="%s")
            text = s.getvalue()
            dt = a.dtype
            # A FRESH StringDType per call: NumPy's loadtxt keeps the caller's StringDType instance as the result's
            # descriptor, and a second call with the same instance corrupts the first result (MemoryError on read).
            add("io.loadtxt", k, lambda text=text, dt=dt: np.loadtxt(io.StringIO(text), comments=None, delimiter="\t",
                                                                    dtype=np.dtypes.StringDType() if dt.kind == "T" else dt))
    return out


# NumPy 2.4.2 SEGFAULTS on these cases (key without the size; found by running every case in its own pass and
# recording where the process died): np.intersect1d / np.setxor1d over StringDType sort the concatenation of two sorted
# runs, the default-kind introsort falls to its heapsort fallback, and that fallback crashes for StringDType. The
# assume_unique forms (sort.intersect1d_au / sort.setxor1d_au) keep the concatenation unsorted and give a comparand.
NUMPY_CRASHES = {
    "sort.intersect1d|t8", "sort.setxor1d|t8", "sort.intersect1d|tna", "sort.setxor1d|tna",
    "sort.intersect1d|t40", "sort.setxor1d|t40",
}

# Full keys NumPy cannot finish in reasonable time: np.setdiff1d over StringDType calls np.isin, which treats
# StringDType as an object dtype and loops `ar1 == a` over every element of ar2, O(n * m): 13.5 ms at 1K, ~135 s a call
# at 100K. The NumPy side prints a SKIP line; the C# side still times them (a NumSharp-only row).
NUMPY_STALLS = {"sort.setdiff1d|t8|100000", "sort.setdiff1d|tna|100000", "sort.setdiff1d|t40|100000"}


def run_time(datadir, filt):
    """Times every case of every size in order; see the module docstring for the output lines."""
    prefixes = None if filt in (None, "-", "") else filt.split(",")
    crashes = NUMPY_CRASHES
    slows = NUMPY_STALLS
    for n in SIZES:
        for key, fn in cases(datadir, n):
            if prefixes and not any((key == p[:-1]) if p.endswith("$") else key.startswith(p) for p in prefixes):
                continue
            if key.rsplit("|", 1)[0] in crashes:
                print(f"{key}\tCRASH\tNumPy 2.4.2 segfaults", flush=True)
                continue
            if key in slows:
                print(f"{key}\tSKIP\tNumPy too slow (quadratic)", flush=True)
                continue
            if os.environ.get("NUMSHARP_STRINGS_FULL_LIST"):
                print(key, flush=True)
                continue
            try:
                res = fn()
                s = sig(res)
            except Exception as e:  # NumPy refuses this case
                print(f"{key}\tREFUSED\t{type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            ms = best_ms(fn)
            print(f"{key}\t{ms:.6g}\t{s}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NUMSHARP_STRINGS_FULL_MODE", "time")
    datadir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NUMSHARP_STRINGS_FULL_DATA")
    if not datadir:
        print("NUMSHARP_STRINGS_FULL_DATA (or argv[2]) must name the shared data directory", file=sys.stderr)
        sys.exit(2)
    if mode == "write":
        write(datadir)
    else:
        run_time(datadir, sys.argv[3] if len(sys.argv) > 3 else os.environ.get("NUMSHARP_STRINGS_FULL_FILTER"))
