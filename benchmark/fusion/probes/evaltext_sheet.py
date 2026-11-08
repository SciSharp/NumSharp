"""Joins the outputs of evaltext_bench.cs / evaltext_e2_bench.cs and their NumPy twins into the tables of
docs/plans/evaluate-text.md (section 5).

    python benchmark/fusion/probes/evaltext_sheet.py cs.out py.out [--full]

Default: one row per (text kind, layout, n), one column per tree (in the order the C# output first names them), each
cell "NPY/NS (np.evaluate us)" with the house icons (NPY/NS = NumPy time / NumSharp time, > 1 means NumSharp is faster;
>= 1.0 / >= 0.5 / >= 0.2 / below). --full: one row per cell with NumPy, np.evaluate and unfused NumSharp times, NPY/NS
and unfused/fused, then the worst NPY/NS cell. Lines that are not result rows (build output, warnings) are skipped.
"""
import sys


def rows(path, width):
    """Result rows of one output file: (kind, layout, n, tree) -> the numbers after the key. A row has `width` fields and
    starts with a text kind; anything else in the file is ignored."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != width or parts[0] not in ("U", "S", "T"):
                continue
            out[tuple(parts[:4])] = [float(x) for x in parts[4:]]
    return out


def icon(ratio):
    """The house icon of an NPY/NS ratio (CLAUDE.md, Performance convention)."""
    return "✅" if ratio >= 1.0 else "🟡" if ratio >= 0.5 else "🟠" if ratio >= 0.2 else "🔴"


def us(value):
    """Microseconds as the plan prints them: one decimal below 100, whole numbers with separators above."""
    return f"{value:,.1f}" if value < 100 else f"{value:,.0f}"


def main(argv):
    """Prints the table for the two files named in argv (C# output first); returns the process exit code."""
    if len(argv) < 3:
        print(__doc__)
        return 2
    cs = rows(argv[1], 6)
    py = rows(argv[2], 5)
    if "--full" in argv:
        print("| text | layout | n | tree | NumPy µs | np.evaluate µs | unfused µs | NPY/NS | unfused/fused |")
        print("|---|---|---|---|---|---|---|---|---|")
        worst = None
        for key, (fused, unfused) in cs.items():
            ratio = py[key][0] / fused
            if worst is None or ratio < worst[1]:
                worst = (key, ratio)
            kind, layout, n, tree = key
            print(f"| {kind} | {layout} | {int(n):,} | {tree} | {us(py[key][0])} | {us(fused)} | {us(unfused)} | "
                  f"{icon(ratio)} {ratio:.1f}× | {unfused / fused:.2f}× |")
        if worst:
            print(f"\nworst NPY/NS: {worst[0]} = {worst[1]:.2f}")
        return 0
    # The trees in the order the benchmark first names them (dicts keep insertion order).
    trees = list(dict.fromkeys(key[3] for key in cs))
    print("| text | layout | n | " + " | ".join(trees) + " |")
    print("|---|---|---|" + "---|" * len(trees))
    seen = []
    for kind, layout, n, _ in cs:
        if (kind, layout, n) in seen:
            continue
        seen.append((kind, layout, n))
        cells = []
        for tree in trees:
            fused = cs[(kind, layout, n, tree)][0]
            ratio = py[(kind, layout, n, tree)][0] / fused
            cells.append(f"{icon(ratio)} {ratio:.1f}× ({us(fused)})")
        print(f"| {kind} | {layout} | {int(n):,} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
