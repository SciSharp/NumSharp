#!/usr/bin/env python3
# =============================================================================
# render_dashboard.py — a dense, numbers-first NumPy-vs-NumSharp dashboard from the
# merged op-matrix (benchmark-report.json). Same ASCII-bar aesthetic as
# benchmark/npyiter/npyiter_results.md, applied to the full op × dtype × N comparison:
# headline geomean, by-size / by-suite / by-dtype bars, the status mix, and the top
# wins/losses. Graphs + stats + numbers, minimal prose.
#
#   python benchmark/scripts/render_dashboard.py
#     reads  benchmark/benchmark-report.json     (merged op-matrix, from merge-results.py)
#     writes benchmark/benchmark-dashboard.md     (the dense sheet, ```-fenced)
#
# speedup = NumPy ÷ NumSharp  (>1.0× = NumSharp faster). Only CREDIBLE comparisons
# (both sides ≥1µs, speedup ≤20×) are charted; negligible / no-data rows are excluded
# (see merge-results.py classify()).
# =============================================================================
import datetime
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "benchmark", "benchmark-report.json")
OUT = os.path.join(REPO, "benchmark", "benchmark-dashboard.md")

CREDIBLE = {"faster", "close", "slower", "much_slower"}
SCALE, WIDTH = 10.0, 20            # bar units: length 10 = parity (1.0×), 20 = 2.0× (then ▶)
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
HDR = "          slower ◄───────── 1.0 (parity) ─────────► faster"


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def bar(s):
    u = s * SCALE
    if u >= WIDTH:
        return "█" * (WIDTH - 1) + "▶"
    full = int(u)
    t = "█" * full + EIGHTHS[int((u - full) * 8)]
    pad = WIDTH - len(t)
    return t + (" " + "." * (pad - 1) if pad >= 3 else " " * pad)


def sizelabel(n):
    return {1000: "1K", 100000: "100K", 10000000: "10M"}.get(n, f"{n:,}")


def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    negligible = sum(1 for r in data if r["status"] == "negligible")
    no_data = sum(1 for r in data if r["status"] == "no_data")
    cred = [r for r in data if r["status"] in CREDIBLE and r.get("numsharp_ms") and r.get("numpy_ms")]
    for r in cred:
        r["sp"] = r["numpy_ms"] / r["numsharp_ms"]

    L = []

    def out(s=""):
        L.append(s)

    def barline(label, rows, width=13):
        sps = [r["sp"] for r in rows]
        if not sps:
            out(f"{label:<{width}}(no data)")
            return
        g = geomean(sps)
        win = sum(1 for x in sps if x > 1.0)
        tag = "   ◄ PARITY" if 0.97 <= g <= 1.03 else ("   ◄ SLOWER" if g < 0.97 else "")
        out(f"{label:<{width}}{bar(g)}  {g:5.2f}×   ({win:4d} win /{len(sps) - win:4d} lose){tag}")

    stamp = os.environ.get("BENCH_STAMP", datetime.date.today().isoformat())
    g_all = geomean([r["sp"] for r in cred])
    win = sum(1 for r in cred if r["sp"] > 1.0)

    out(f"NumSharp vs NumPy — operation matrix · {stamp} · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)")
    out(f"{len(cred)} credible comparisons of {total} ops · {negligible} negligible + {no_data} no-data excluded · BenchmarkDotNet vs NumPy 2.4.2")
    out()
    out(f"HEADLINE — {g_all:.2f}× geomean over {len(cred)} credible cells · {win} win / {len(cred) - win} lose")
    out()

    out("BY ARRAY-SIZE TIER  (geomean over all credible ops at that size)")
    out(HDR)
    for n in sorted({r["n"] for r in cred}):
        rows = [r for r in cred if r["n"] == n]
        if len(rows) >= 10:                         # skip stray one-off sizes (500/900/…)
            barline(sizelabel(n), rows)
    barline("ALL", cred)
    out()

    out("BY SUITE  (geomean, ranked fastest → slowest)")
    out(HDR)
    suites = {}
    for r in cred:
        suites.setdefault((r["suite"] or "?").lower(), []).append(r)
    for name, rows in sorted(suites.items(), key=lambda kv: geomean([r["sp"] for r in kv[1]]), reverse=True):
        barline(name, rows)
    out()

    out("BY DTYPE  (geomean over all credible ops of that type)")
    out(HDR)
    dts = {}
    for r in cred:
        dts.setdefault(r["dtype"], []).append(r)
    for name, rows in sorted(dts.items(), key=lambda kv: geomean([r["sp"] for r in kv[1]]), reverse=True):
        barline(name, rows)
    out()

    out("STATUS MIX  (NumSharp ÷ NumPy bands; credible only)")
    bands = [("✅ faster   ≥1.0×", "faster"), ("🟡 close    0.5–1.0×", "close"),
             ("🟠 slower   0.2–0.5×", "slower"), ("🔴 much     <0.2×", "much_slower")]
    counts = {s: sum(1 for r in data if r["status"] == s) for _, s in bands}
    mx = max(counts.values()) or 1
    for lab, s in bands:
        c = counts[s]
        out(f"{lab:<22}{'█' * round(18 * c / mx):<19}{c}")
    out()

    def row(r):
        sp = r["sp"]
        s = f"{sp:6.2f}×" if sp >= 1 else f"{sp:6.3f}× ({1 / sp:.0f}× slower)"
        op = r["operation"] if len(r["operation"]) <= 30 else r["operation"][:29] + "…"
        return f"  {op:<30} {r['dtype']:<8} {sizelabel(r['n']):>4}  {r['numpy_ms']:8.3f} →{r['numsharp_ms']:9.3f} ms  {s}"

    hdr = f"  {'operation':<30} {'dtype':<8} {'N':>4}  {'NumPy':>8}  {'NumSharp':>9}     speedup"
    out("TOP 12 WINS  (NumSharp fastest vs NumPy)")
    out(hdr)
    for r in sorted(cred, key=lambda r: r["sp"], reverse=True)[:12]:
        out(row(r))
    out()
    out("TOP 12 LOSSES  (NumSharp slowest vs NumPy)")
    out(hdr)
    for r in sorted(cred, key=lambda r: r["sp"])[:12]:
        out(row(r))
    out()
    out("note · speedup = NumPy ÷ NumSharp on one runner · negligible rows (<1µs work or >20× = call")
    out("       overhead / lazy alloc / views) excluded · absolute ms drift by hardware, ratios hold")

    sheet = "\n".join(L)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("```\n" + sheet + "\n```\n")
    print(sheet)


if __name__ == "__main__":
    main()
