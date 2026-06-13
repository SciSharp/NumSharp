#!/usr/bin/env python3
# =============================================================================
# npyiter_sheet.py — THE canonical NpyIter benchmark orchestrator + renderer.
#
# Runs every section of npyiter_bench.{cs,py} (NumSharp vs NumPy), each in its
# own short-lived process so the intermittent AV under heavy mixed load can't
# void the run (NumSharp sections retry up to 4x on a crash). Merges both sides
# and renders ONE results sheet: per-tier / per-category / per-family operation
# matrix, construction-vs-nditer, chunk-width dispatch, pathology canaries, and
# the NumSharp-only dividends. Saves the sheet to npyiter_results.md and the raw
# pairs to npyiter_results.tsv.
#
#   python benchmark/npyiter/npyiter_sheet.py                 # full run + sheet
#   python benchmark/npyiter/npyiter_sheet.py --skip-build    # reuse Release build
#   python benchmark/npyiter/npyiter_sheet.py --render-only   # re-render last .tsv
#   python benchmark/npyiter/npyiter_sheet.py --sections elementwise pathology
# =============================================================================
import argparse
import datetime
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CS = os.path.join(HERE, "npyiter_bench.cs")
PY = os.path.join(HERE, "npyiter_bench.py")
CORE_CSPROJ = os.path.join(REPO, "src", "NumSharp.Core", "NumSharp.Core.csproj")
TSV = os.path.join(HERE, "npyiter_results.tsv")
SHEET = os.path.join(HERE, "npyiter_results.md")

SECTIONS = ["elementwise", "reductions", "selection", "copycast", "indexmath",
            "dtypes", "dividends", "construction", "chunkwidth", "pathology"]

TIERS = ["1", "1K", "100K", "1M", "10M"]
TIER_LABEL = {"1": "scalar", "1K": "1K", "100K": "100K", "1M": "1M", "10M": "10M"}
CATEGORIES = [
    ("elementwise", ["add", "sqrt", "copy", "sadd", "bcast", "frev", "castbuf", "mixbuf"]),
    ("reductions",  ["psum", "sumax0", "sumax1", "sumdt", "amin", "cumsum", "anyff", "anyeh"]),
    ("selection",   ["where", "bread", "bassign", "cnz", "argw", "gather", "scatter"]),
    ("copy/cast",   ["flatten", "astype", "ravelT", "inplace", "lessbool"]),
    ("index-math",  ["unravel", "ravelmi"]),
    ("dtypes",      ["cplx", "f16", "i8"]),
]
MAIN = [f for _, fs in CATEGORIES for f in fs]
DIVIDENDS = ["fuse7", "reuse", "par8"]
CTOR = ["ctor.1op", "ctor.3op_exl", "ctor.ufunc", "ctor.bufcast", "ctor.multiindex",
        "ctor.8op", "ctor.4d", "ctor.8d", "ctor.strided2d"]
CW = ["cw.4", "cw.16", "cw.64", "cw.256", "cw.1024"]
PATH = ["path.bcast_reduce", "path.allocate", "path.overlap_copy", "path.forder_out", "path.zerodim"]
FAM_LABEL = {"sadd": "strided", "frev": "reversed", "psum": "sum", "sumax0": "sum ax0",
             "sumax1": "sum ax1", "sumdt": "sum dt=", "anyff": "any(F)", "anyeh": "any(hit)",
             "bread": "a[mask]", "bassign": "a[mask]=", "cnz": "count_nz", "argw": "argwhere",
             "gather": "a[idx]", "scatter": "a[idx]=", "ravelT": "ravel.T", "inplace": "in-place",
             "lessbool": "less->b", "ravelmi": "ravel_mi", "cplx": "complex", "f16": "float16", "i8": "int8"}

SCALE, WIDTH = 10.0, 20
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]


def log(m):
    print(m, file=sys.stderr, flush=True)


def parse(text):
    out = {}
    for ln in text.splitlines():
        if "\t" in ln:
            k, _, v = ln.partition("\t")
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


# A representative id per section — used by --resume to detect a covered section
# and by the merge to know which side produced what.
SECTION_PROBE = {
    "elementwise": "add@1M", "reductions": "psum@1M", "selection": "where@1M",
    "copycast": "flatten@1M", "indexmath": "unravel@1M", "dtypes": "cplx@1M",
    "dividends": "fuse7@1M", "construction": "ctor.1op", "chunkwidth": "cw.4",
    "pathology": "path.zerodim",
}
# DOTNET_DbgEnableMiniDump=0: an AV returns a non-zero exit IMMEDIATELY instead of
# hanging the process while the runtime writes a crash dump (the silent stall that
# voided the first full run). We never taskkill dotnet — fast clean exits + retry.
NS_ENV_EXTRA = {"DOTNET_DbgEnableMiniDump": "0", "DOTNET_EnableCrashReport": "0"}
NS_TIMEOUT = 360
NP_TIMEOUT = 240


def run_ns(section, retries=4):
    with open(CS, encoding="utf-8") as f:
        src = f.read()
    # Portability: the .cs pins #:project to an absolute Windows path so it can be
    # run directly with `dotnet run - < file` on the author's box. Rewrite it to
    # THIS checkout's csproj so the same bench runs unchanged on a Linux CI runner.
    src = src.replace("K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj",
                      CORE_CSPROJ.replace(os.sep, "/"))
    env = {**os.environ, "NPYITER_SECTION": section, **NS_ENV_EXTRA}
    for attempt in range(1, retries + 1):
        try:
            p = subprocess.run(["dotnet", "run", "-c", "Release", "-"], input=src,
                               capture_output=True, text=True, cwd=REPO, env=env, timeout=NS_TIMEOUT)
        except subprocess.TimeoutExpired:
            log(f"    NS {section}: attempt {attempt}/{retries} TIMED OUT ({NS_TIMEOUT}s) — retrying")
            continue
        if p.returncode == 0:
            return parse(p.stdout)
        log(f"    NS {section}: attempt {attempt}/{retries} crashed (exit {p.returncode}) — retrying")
    log(f"    NS {section}: FAILED after {retries} attempts — marked NA (ignored)")
    return {}


def run_np(section):
    env = {**os.environ, "NPYITER_SECTION": section}
    try:
        p = subprocess.run([sys.executable, PY], capture_output=True, text=True,
                           cwd=REPO, env=env, timeout=NP_TIMEOUT)
    except subprocess.TimeoutExpired:
        log(f"    NumPy {section}: TIMED OUT ({NP_TIMEOUT}s) — omitted")
        return {}
    if p.returncode != 0:
        log(f"    NumPy {section}: exit {p.returncode}\n{p.stderr[-400:]}")
    return parse(p.stdout)


def write_tsv(pairs):
    with open(TSV, "w", encoding="utf-8") as f:
        f.write("id\tns_ms\tnp_ms\n")
        for k, (a, b) in pairs.items():
            a_str = "NA" if a is None else repr(a)   # NA = NumSharp AV, ignored
            f.write(f"{k}\t{a_str}\t{b!r}\n")


def collect(sections, skip_build, resume):
    if not skip_build:
        log("[build] NumSharp.Core (Release)…")
        b = subprocess.run(["dotnet", "build", CORE_CSPROJ, "-c", "Release", "-v", "q", "--nologo",
                            "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0"],
                           capture_output=True, text=True, cwd=REPO)
        if b.returncode != 0:
            log("[build] FAILED:\n" + b.stdout[-1500:])
            sys.exit(1)
        log("[build] ok")
    # Resume: seed from any prior tsv so a crash mid-sweep doesn't lose progress.
    pairs = load_tsv() if (resume and os.path.exists(TSV)) else {}
    for s in sections:
        if resume and SECTION_PROBE.get(s) in pairs:
            log(f"[skip] {s} (already in tsv)")
            continue
        log(f"[run] {s} …")
        npp = run_np(s)            # NumPy first — reliable; gives the expected id set
        ns = run_ns(s)             # NumSharp — may crash (AV); {} if all retries fail
        if not ns and npp:
            # AV policy: a section that crashes all retries is IGNORED — record its
            # ids NA (NumSharp side absent), excluded from every geomean downstream.
            for k in npp:
                pairs[k] = (None, npp[k])
            log(f"    {s}: NA — NumSharp AccessViolation, ignored ({len(npp)} ids)")
        else:
            added = {k: (ns[k], npp[k]) for k in ns if k in npp and ns[k] > 0 and npp[k] > 0}
            pairs.update(added)
            log(f"    {s}: +{len(added)} pairs ({len(pairs)} total)")
        write_tsv(pairs)   # persist after EVERY section — partial progress survives
    log(f"[data] {len(pairs)} pairs -> {os.path.relpath(TSV, REPO)}")
    return pairs


def load_tsv():
    pairs = {}
    with open(TSV, encoding="utf-8") as f:
        next(f, None)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) == 3:
                ns = None if p[1] == "NA" else float(p[1])
                pairs[p[0]] = (ns, float(p[2]))
    return pairs


# ---- rendering --------------------------------------------------------------
def bar(s):
    u = s * SCALE
    if u >= WIDTH:
        return "█" * (WIDTH - 1) + "▶"
    full = int(u)
    t = "█" * full + EIGHTHS[int((u - full) * 8)]
    pad = WIDTH - len(t)
    return t + (" " + "." * (pad - 1) if pad >= 3 else " " * pad)


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def pct_str(pct):
    """Share of NumPy's time NumSharp uses; compact for the rare huge slowdowns."""
    return f"{pct:4.0f}%" if pct < 1000 else f"{pct / 100:3.0f}×NP"


SECTION_FAMS = {
    "elementwise": ["add", "sqrt", "copy", "sadd", "bcast", "frev", "castbuf", "mixbuf"],
    "reductions": ["psum", "sumax0", "sumax1", "sumdt", "amin", "cumsum", "anyff", "anyeh"],
    "selection": ["where", "bread", "bassign", "cnz", "argw", "gather", "scatter"],
    "copycast": ["flatten", "astype", "ravelT", "inplace", "lessbool"],
    "indexmath": ["unravel", "ravelmi"],
    "dtypes": ["cplx", "f16", "i8"],
    "dividends": ["fuse7", "reuse", "par8"],
}


def section_of(id_):
    if id_.startswith("ctor."):
        return "construction"
    if id_.startswith("cw."):
        return "chunkwidth"
    if id_.startswith("path."):
        return "pathology"
    fam = id_.split("@")[0]
    for sec, fams in SECTION_FAMS.items():
        if fam in fams:
            return sec
    return "?"


def render(pairs):
    NA = {k for k, (ns, _) in pairs.items() if ns is None}    # NumSharp AV — ignored
    SP = {k: np_ / ns for k, (ns, np_) in pairs.items() if ns is not None}

    def sp(fam, tier):
        return SP.get(f"{fam}@{tier}")

    def cell(fam, tier):
        key = f"{fam}@{tier}"
        if key in NA:
            return f"{'NA':>9}"
        v = SP.get(key)
        return f"{v:>8.2f}×" if v else f"{'-':>9}"

    def famsps(fam):
        return [sp(fam, t) for t in TIERS if sp(fam, t) is not None]

    def tiersps(tier, fams):
        return [sp(fam, tier) for fam in fams if sp(fam, tier) is not None]

    L, HDR = [], "        slower ◄───────── 1.0 (parity) ─────────► faster"

    def out(s=""):
        L.append(s)

    def barline(label, sps, width=11):
        if not sps:
            out(f"{label:<{width}}(no data)")
            return
        g = geomean(sps)
        win = sum(1 for x in sps if x > 1.0)
        tag = "  ◄ PARITY" if 0.97 <= g <= 1.03 else ("  ◄ SLOWER" if g < 0.97 else "")
        out(f"{label:<{width}}{bar(g)}  {g:5.2f}×  🕐{pct_str(100.0 / g)}  ({win:3d} win /{len(sps) - win:3d} lose){tag}")

    stamp = os.environ.get("NPYITER_STAMP", datetime.date.today().isoformat())
    out(f"NumSharp NpyIter — canonical benchmark · {stamp} · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)")
    out(f"{len(pairs)} measured pairs ({len(NA)} NA) · best-of-rounds, Release · matched kernels/ids")
    out("🕐 %NumPy = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)")
    out()
    ignored = sorted({section_of(k) for k in NA})
    out("AV POLICY — a NumSharp section that crashes all retries (known intermittent")
    out("AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED")
    out("and excluded from every geomean below."
        + (f"  THIS RUN: NA across {', '.join(ignored)}." if ignored else "  THIS RUN: none."))
    out()

    main_sps = [SP[f"{f}@{t}"] for f in MAIN for t in TIERS if f"{f}@{t}" in SP]
    if main_sps:
        g = geomean(main_sps)
        win = sum(1 for x in main_sps if x > 1.0)
        out(f"HEADLINE — operation matrix: {g:.2f}× geomean · 🕐 {100.0 / g:.0f}% of NumPy's time · {win} win / {len(main_sps) - win} lose over {len(main_sps)} cells")
        out()

    out("OPERATIONS — BY SIZE TIER  (geomean over all families)")
    out(HDR)
    for t in TIERS:
        barline(TIER_LABEL[t], tiersps(t, MAIN))
    barline("ALL", main_sps)
    out()

    out("OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)")
    out(HDR)
    for name, fams in CATEGORIES:
        barline(name, [SP[f"{f}@{t}"] for f in fams for t in TIERS if f"{f}@{t}" in SP])
    out()

    out("CATEGORY × TIER geomean")
    out(f"{'category':<12}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS))
    for name, fams in CATEGORIES:
        out(f"{name:<12}" + "".join(f"{geomean(tiersps(t, fams)):>8.2f}×" if tiersps(t, fams) else f"{'-':>9}" for t in TIERS))
    out()

    out("PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)")
    out(f"{'family':<11}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS) + "    geomean")
    for name, fams in CATEGORIES:
        out(f"-- {name}")
        for fam in fams:
            cells = "".join(cell(fam, t) for t in TIERS)
            g = famsps(fam)
            out(f"  {FAM_LABEL.get(fam, fam):<9}{cells}   {geomean(g):>6.2f}×" if g else f"  {FAM_LABEL.get(fam, fam):<9}{cells}")
    out()

    out("CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)")
    out(HDR)
    ctor_sps = [SP[c] for c in CTOR if c in SP]
    for c in CTOR:
        if c in SP:
            barline(c.replace("ctor.", ""), [SP[c]], width=13)
    if ctor_sps:
        barline("geomean", ctor_sps, width=13)
    out()

    out("CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)")
    out(HDR)
    for c in CW:
        if c in SP:
            barline("w=" + c.replace("cw.", ""), [SP[c]], width=13)
    out()

    out("PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)")
    for p in PATH:
        if p in SP:
            r = SP[p]
            verdict = "SLOWER" if r < 0.97 else ("faster" if r > 1.03 else "parity")
            mult = f"{1 / r:.1f}× slower" if r < 1 else f"{r:.1f}× faster"
            out(f"  {p.replace('path.', ''):<16}{r:6.2f}×   ({mult}, {verdict})")
    out()

    out("DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)")
    out(f"{'':13}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS) + "   note")
    DIV_NOTE = {"fuse7": "vs chained 6× add", "reuse": "vs rebuild each call", "par8": "vs single-thread"}
    for d in DIVIDENDS:
        cells = "".join(cell(d, t) for t in TIERS)
        out(f"{d:<13}{cells}   {DIV_NOTE[d]}")
    out()

    allmain = sorted(((k, SP[k]) for k in SP if k.split("@")[0] in MAIN and "@" in k), key=lambda kv: kv[1])
    if allmain:
        out("biggest NumSharp wins: " + " · ".join(f"{k} {v:.2f}×" for k, v in reversed(allmain[-5:])))
        out("most behind:           " + " · ".join(f"{k} {v:.2f}×" for k, v in allmain[:5]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse a prior tsv; skip already-collected sections")
    ap.add_argument("--sections", nargs="*", default=SECTIONS)
    args = ap.parse_args()

    if args.render_only:
        pairs = load_tsv()
    else:
        pairs = collect(args.sections, args.skip_build, args.resume)

    sheet = render(pairs)
    with open(SHEET, "w", encoding="utf-8") as f:
        f.write("```\n" + sheet + "\n```\n")
    log(f"[sheet] -> {os.path.relpath(SHEET, REPO)}")
    print(sheet)


if __name__ == "__main__":
    main()
