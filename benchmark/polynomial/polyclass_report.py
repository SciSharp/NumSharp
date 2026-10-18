"""Renders polyclass_results.md from one or more polyclass_bench.cs runs (the TSVs it prints).

Usage (from this directory, after full runs of both sides — see polyclass_numpy.py):
    python polyclass_report.py ns_class_1.tsv ns_class_2.tsv ns_class_3.tsv > polyclass_results.md

Each TSV is one alternating pair of runs (NumPy, then NumSharp joined with that NumPy run). The report takes each side's
per-cell minimum over the files — the best window of each side, so a background burst during one run does not decide a
ratio. The C# runner has already checked every cell's result against NumPy's SHA-256 before timing it; this script
refuses to render a run that reports a mismatch, so a committed report is always a bit-exact one.
"""
import collections
import math
import platform
import sys

SECTIONS = {
    "K": "construction and the class methods: the constructor (8 / 1,000 coefficients), identity, basis(5), "
         "fromroots(5 roots), cast (16 coefficients, from another class)",
    "A": "arithmetic: + and * of two series (8 / 64 coefficients, + also at 1,000), + / * / true division by a Python "
         "float, // and divmod (16 by 4), ** 3 (4 coefficients), unary -, ==",
    "E": "evaluation: p(x) at 1,000 and 100,000 points and at a Python float (8 coefficients), p(q) (4 and 4)",
    "C": "calculus and conversion: deriv, integ, integ with k and lbnd (64 coefficients), convert(kind=...) (16), "
         "linspace(100)",
    "M": "members: copy (1,000), trim, cutdeg(8), truncate(8) (64), mapparms, degree",
    "P": "printers: str (ascii), repr, _repr_latex_, format 'unicode' (8 coefficients)",
    "S": "pickling: __getstate__, __setstate__ (8 coefficients) — the section's floor: __getstate__ copies three "
         "arrays as NumPy does, at NumSharp's NDArray allocation floor (0.44 us a copy); __setstate__ stores a view per "
         "array (0.18 us each), so a series never disposes the caller's state arrays, where NumPy's is one "
         "`self.__dict__ = state` assignment (0.19 us in all)",
    "L": "LAPACK at one OpenBLAS thread: roots (8 coefficients), fit (1,000 points, degree 8)",
}


def icon(r):
    """The house NPY/NS status icon.

    Args:
        r: The NPY/NS ratio of a cell.

    Returns:
        ✅ at or above parity, 🟡 from 0.5, 🟠 from 0.2, 🔴 below.
    """
    return "✅" if r >= 1.0 else "🟡" if r >= 0.5 else "🟠" if r >= 0.2 else "🔴"


def geomean(xs):
    """The geometric mean.

    Args:
        xs: Positive numbers (at least one).

    Returns:
        exp(mean(log(x))).
    """
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def read(paths):
    """Reads the runner TSVs and merges them cell by cell.

    Args:
        paths: The TSV files (one per run pair).

    Returns:
        An ordered dict label -> (NumSharp us, NumPy us), each the minimum over the runs that measured the cell; exits
        with an error when any run reports a byte mismatch.
    """
    cells = collections.OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                # A cell row ends in its byte check; the header and the runner's per-section summary rows (also five
                # columns: section, cells, min, geomean, mismatches) do not.
                if len(parts) != 5 or parts[4] not in ("exact", "BYTES"):
                    continue
                label, ns, npy, _, check = parts
                if check != "exact":
                    sys.exit(f"{path}: {label} reports {check}; fix the mismatch before rendering a report")
                ns, npy = float(ns), float(npy)
                if label in cells:
                    a, b = cells[label]
                    cells[label] = (min(a, ns), min(b, npy))
                else:
                    cells[label] = (ns, npy)
    return cells


def main():
    """Prints the Markdown report.

    Returns:
        None.
    """
    cells = read(sys.argv[1:])
    if not cells:
        sys.exit("no cells")
    ratios = {k: npy / ns for k, (ns, npy) in cells.items()}
    print("# numpy.polynomial series classes (U10 / U11's class half) — NumSharp vs NumPy 2.4.2")
    print()
    print(f"{len(cells)} cells, every result SHA-256-identical to NumPy's (checked before timing). "
          f"Host: {platform.processor() or platform.machine()}, {len(sys.argv) - 1} alternating run pair(s), each side's "
          "best window per cell. NPY/NS = NumPy time / NumSharp time (higher = NumSharp faster).")
    print()
    print(f"**All cells: min {min(ratios.values()):.2f}x, geomean {geomean(list(ratios.values())):.2f}x.**")
    print()
    print("| section | what | cells | min NPY/NS | geomean NPY/NS |")
    print("|---|---|---:|---:|---:|")
    by_section = collections.OrderedDict()
    for k, r in ratios.items():
        by_section.setdefault(k.split("/")[0], []).append(r)
    for s, rs in by_section.items():
        print(f"| {s} | {SECTIONS.get(s, '')} | {len(rs)} | {min(rs):.2f} | {geomean(rs):.2f} |")
    print()
    print("## Every cell")
    print()
    print("| cell | NumSharp us | NumPy us | NPY/NS |")
    print("|---|---:|---:|---:|")
    for k, (ns, npy) in cells.items():
        r = npy / ns
        print(f"| {k} | {ns:.3f} | {npy:.3f} | {icon(r)} {r:.2f} |")
    slow = sorted((r, k) for k, r in ratios.items() if r < 1.5)
    print()
    print("## Cells below 1.5x")
    print()
    if not slow:
        print("None.")
    for r, k in slow:
        ns, npy = cells[k]
        print(f"- {k}: {r:.2f}x (NumSharp {ns:.3f} us, NumPy {npy:.3f} us)")


if __name__ == "__main__":
    main()
