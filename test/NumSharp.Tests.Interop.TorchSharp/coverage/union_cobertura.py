"""
Union the line and branch coverage of several cobertura reports of ONE assembly, and print the totals per source file with
the line numbers no report covered.

The TorchSharp bridge needs the union: code that depends on the loaded TorchSharp (the 6-argument native THSTensor_new of
0.100.x, the 9-argument one from 0.101.0, the sparse / quantized / uint16 imports of newer releases) runs on one leg only.
A line counts as covered when any report hit it; a line's branch conditions count by the best report.

    python -I union_cobertura.py <floor>.cobertura.xml <current>.cobertura.xml

See bridge.runsettings beside this script for the collection commands.
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def union(paths):
    """Reads every report and unions it per (file, line).

    Args:
        paths: the cobertura XML files.

    Returns:
        A pair of dicts: (file, line) -> total hits, and (file, line) -> (covered conditions, total conditions).

    Raises:
        xml.etree.ElementTree.ParseError: a file is not XML.
        OSError: a file cannot be read.
    """
    hits = defaultdict(int)
    branches = {}
    for path in paths:
        root = ET.parse(path).getroot()
        for cls in root.iter("class"):
            fname = cls.get("filename")
            for line in cls.iter("line"):
                key = (fname, int(line.get("number")))
                hits[key] += int(line.get("hits"))
                condition = line.get("condition-coverage")
                if condition:
                    m = re.search(r"\((\d+)/(\d+)\)", condition)
                    covered, total = int(m.group(1)), int(m.group(2))
                    best = branches.get(key, (0, total))
                    branches[key] = (max(best[0], covered), total)
    return hits, branches


def main(paths):
    """Prints the per-file and total line / branch coverage of the union, with the uncovered line numbers.

    Args:
        paths: the cobertura XML files (at least one).

    Returns:
        0, or 2 when no report was given.
    """
    if not paths:
        print(__doc__)
        return 2
    hits, branches = union(paths)
    files = defaultdict(lambda: {"lines": 0, "covered": 0, "bcov": 0, "btot": 0, "missing": []})
    for (fname, number), h in sorted(hits.items()):
        f = files[fname]
        f["lines"] += 1
        if h > 0:
            f["covered"] += 1
        else:
            f["missing"].append(number)
        if (fname, number) in branches:
            covered, total = branches[(fname, number)]
            f["bcov"] += covered
            f["btot"] += total
    totals = {"lines": 0, "covered": 0, "bcov": 0, "btot": 0}
    for fname, f in sorted(files.items()):
        for k in totals:
            totals[k] += f[k]
        name = fname.replace("\\", "/").split("/")[-1]
        print(f"{name:40s} lines {f['covered']}/{f['lines']} ({100.0 * f['covered'] / max(f['lines'], 1):.1f}%)  "
              f"branches {f['bcov']}/{f['btot']} ({100.0 * f['bcov'] / max(f['btot'], 1):.1f}%)")
        if f["missing"]:
            print(f"    uncovered lines: {f['missing']}")
    print(f"TOTAL lines {totals['covered']}/{totals['lines']} ({100.0 * totals['covered'] / max(totals['lines'], 1):.1f}%)  "
          f"branches {totals['bcov']}/{totals['btot']} ({100.0 * totals['bcov'] / max(totals['btot'], 1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
