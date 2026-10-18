"""NumPy 2.4.2 twin of polyprint_bench.cs — the numpy.polynomial printing helpers and the scalar float formatters they
print with (U11: polyutils.format_float, set_default_printstyle; np.format_float_positional / np.format_float_scientific,
the port of NumPy's dragon4.c behind all three).

Writes data/polyprint/manifest.json (gitignored) — the cell list the C# side replays — then prints NumPy's best-of-R
per-call time for every cell (TSV: cell<TAB>us). Each cell records NumPy's text, which the C# runner compares BEFORE it
times the cell.

Usage (from this directory; pin both sides with the same mask, never run them concurrently):
    set NS_PROBE_AFFINITY=0x14
    python polyprint_numpy.py > numpy_print_times_1.tsv
    set DOTNET_TC_CallCountingDelayMs=0
    dotnet run -c Release --no-cache polyprint_bench.cs -- data/polyprint numpy_print_times_1.tsv > ns_print_1.tsv
    (the same for _2 and _3)
    python polyprint_report.py ns_print_1.tsv ns_print_2.tsv ns_print_3.tsv > polyprint_results.md
POLYPRINT_ONLY=FP/,FF/ (comma-separated label prefixes) restricts both sides to matching cells.

Sections (cell-label prefixes):
    FP  np.format_float_positional: float64 / float32 / float16 x 7 values x 8 option sets (unique, a unique cut, exact
        digits to 17 and 60 places, fractional=False, min_digits=20, padding, trim='-')
    FS  np.format_float_scientific: the widths x values x 5 option sets (unique, a unique cut, exact to 17 digits,
        exp_digits=5, min_digits=9)
    FF  polyutils.format_float: the widths x values x the default and floatmode='fixed' print options, plus non-floating
        x (a Python int, a 0-d array, a 1-D array: str(x))
    SP  set_default_printstyle('ascii' / 'unicode')
"""
import json
import math
import os
import sys
import timeit
from pathlib import Path

import numpy as np  # noqa: E402
from numpy.polynomial import polyutils as pu  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polyprint"
sys.path.insert(0, str(HERE))
from polyeval_numpy import pin_affinity  # noqa: E402  (the shared host pinning)

CELLS = []
ONLY = [p for p in os.environ.get("POLYPRINT_ONLY", "").split(",") if p]
WIDTHS = {"f64": np.float64, "f32": np.float32, "f16": np.float16}
VALUES = {"0.1": 0.1, "pi": math.pi, "1e-5": 1e-5, "123456.789": 123456.789, "1e300": 1e300, "5e-324": 5e-324,
          "-2.5": -2.5}
POS_OPTS = {"unique": {}, "p3": {"precision": 3}, "exact17": {"precision": 17, "unique": False},
            "exact60": {"precision": 60, "unique": False}, "frac5": {"precision": 5, "fractional": False},
            "min20": {"min_digits": 20}, "pad": {"pad_left": 4, "pad_right": 4}, "trim-": {"trim": "-"}}
SCI_OPTS = {"unique": {}, "p3": {"precision": 3}, "exact17": {"precision": 17, "unique": False},
            "exp5": {"exp_digits": 5}, "min9": {"min_digits": 9}}
PRINT_OPTS = {"default": {}, "fixed": {"floatmode": "fixed"}}


def width_value(width, v):
    """The value as the width's NumPy scalar (an np.float64 scalar is a Python float to format_float's test: both are
    floating, and the C# side spells both as a double).

    Args:
        width: f64 / f32 / f16.
        v: The Python float.

    Returns:
        The scalar.
    """
    with np.errstate(all="ignore"):
        return WIDTHS[width](v)


def cell(label, fn, x, kwargs, xspec, opts=None, calls=4000, rounds=9):
    """Registers one cell, its call and NumPy's text.

    Args:
        label: The cell's label ("<section>/<case>").
        fn: The formatter key: "pos", "sci", "ff" or "style".
        x: The NumPy argument.
        kwargs: The keyword arguments of the call.
        xspec: How the C# side builds x: {"kind": "f64"|"f32"|"f16", "bits": hex}, {"kind": "int", "v": n},
            {"kind": "array", "values": [...]} (a 0-d array when "zerod"), or {"kind": "str", "v": s}.
        opts: The print options the call runs under (format_float).
        calls: Calls per timing round.
        rounds: Rounds (the best one is reported).

    Returns:
        None.
    """
    if ONLY and not any(label.startswith(p) for p in ONLY):
        return
    f = {"pos": np.format_float_positional, "sci": np.format_float_scientific, "ff": pu.format_float,
         "style": np.polynomial.set_default_printstyle}[fn]
    if opts:
        def call():
            with np.printoptions(**opts):
                return f(x, **kwargs)
    else:
        def call():
            return f(x, **kwargs)
    r = call()
    spec = {"label": label, "fn": fn, "x": xspec, "kwargs": kwargs, "calls": calls, "rounds": rounds,
            "expect": "" if r is None else r}
    if opts:
        spec["opts"] = opts
    CELLS.append({**spec, "_call": call})


def bits_spec(width, x):
    """The C# spelling of a width's scalar: its bit pattern.

    Args:
        width: f64 / f32 / f16.
        x: The scalar.

    Returns:
        The spec.
    """
    u = {"f64": np.uint64, "f32": np.uint32, "f16": np.uint16}[width]
    return {"kind": width, "bits": hex(int(np.array(x).view(u)))}


def build():
    """Registers every cell of the four sections (see the module docstring).

    Returns:
        None.
    """
    for width in WIDTHS:
        for vlabel, v in VALUES.items():
            x = width_value(width, v)
            xs = bits_spec(width, x)
            for olabel, kw in POS_OPTS.items():
                cell(f"FP/{width}/{vlabel}/{olabel}", "pos", x, kw, xs)
            for olabel, kw in SCI_OPTS.items():
                cell(f"FS/{width}/{vlabel}/{olabel}", "sci", x, kw, xs)
            for olabel, po in PRINT_OPTS.items():
                cell(f"FF/{width}/{vlabel}/{olabel}", "ff", x, {}, xs, opts=po)
    cell("FF/obj/int", "ff", 7, {}, {"kind": "int", "v": 7})
    cell("FF/obj/zerod", "ff", np.array(0.25), {}, {"kind": "array", "values": [0.25], "zerod": True})
    cell("FF/obj/1d", "ff", np.array([0.1, 2.5, 1e-5]), {}, {"kind": "array", "values": [0.1, 2.5, 1e-5]}, calls=1000)
    cell("SP/ascii", "style", "ascii", {}, {"kind": "str", "v": "ascii"}, calls=20000)
    cell("SP/unicode", "style", "unicode", {}, {"kind": "str", "v": "unicode"}, calls=20000)


def main():
    """Pins the process, writes the manifest the C# runner replays, then prints NumPy's best per-call time per cell.

    Returns:
        None.
    """
    pin_affinity()
    DATA.mkdir(parents=True, exist_ok=True)
    build()
    # set_default_printstyle changes a class attribute: leave NumPy's default behind.
    np.polynomial.set_default_printstyle("unicode" if os.name != "nt" else "ascii")
    (DATA / "manifest.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "_call"} for c in CELLS],
                                                   indent=1), encoding="utf-8")
    print("cell\tnumpy_us")
    for c in CELLS:
        call, calls, rounds = c["_call"], c["calls"], c["rounds"]
        call()   # warm
        best = min(timeit.repeat(call, number=calls, repeat=rounds)) / calls
        print(f"{c['label']}\t{best * 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
