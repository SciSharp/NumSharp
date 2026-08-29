#!/usr/bin/env python3
"""Generate benchmark/scenarios.html from live BDN and NumPy scenario discovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmark"
CSHARP_PROJECT = BENCHMARK / "NumSharp.Benchmark.CSharp" / "NumSharp.Benchmark.CSharp.csproj"
PYTHON_BENCHMARK = BENCHMARK / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
DEFAULT_OUTPUT = BENCHMARK / "scenarios.html"

DTYPES = [
    {"key": "bool", "csharp": "Boolean", "numpy": "bool"},
    {"key": "uint8", "csharp": "Byte", "numpy": "uint8"},
    {"key": "int8", "csharp": "SByte", "numpy": "int8"},
    {"key": "int16", "csharp": "Int16", "numpy": "int16"},
    {"key": "uint16", "csharp": "UInt16", "numpy": "uint16"},
    {"key": "int32", "csharp": "Int32", "numpy": "int32"},
    {"key": "uint32", "csharp": "UInt32", "numpy": "uint32"},
    {"key": "int64", "csharp": "Int64", "numpy": "int64"},
    {"key": "uint64", "csharp": "UInt64", "numpy": "uint64"},
    {"key": "char", "csharp": "Char", "numpy": None},
    {"key": "float16", "csharp": "Half", "numpy": "float16"},
    {"key": "float32", "csharp": "Single", "numpy": "float32"},
    {"key": "float64", "csharp": "Double", "numpy": "float64"},
    {"key": "decimal", "csharp": "Decimal", "numpy": None},
    {"key": "complex128", "csharp": "Complex", "numpy": "complex128"},
]
DTYPE_KEYS = [item["key"] for item in DTYPES]
SUITE_ALIASES = {"Broadcasting": "Broadcast"}


def load_merge_normalizer():
    path = Path(__file__).with_name("merge-results.py")
    spec = importlib.util.spec_from_file_location("scenario_audit_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark normalizer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.normalize_op_name


NORMALIZE = load_merge_normalizer()


def run_json(command: list[str], *, cwd: Path = REPO) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Discovery command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Discovery command did not emit pure JSON: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}") from error


def discover_csharp(*, build: bool) -> dict[str, Any]:
    if build:
        completed = subprocess.run(
            ["dotnet", "build", str(CSHARP_PROJECT), "-c", "Release", "-f", "net10.0", "--no-restore"],
            cwd=REPO, text=True, encoding="utf-8", capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"C# benchmark build failed:\n{completed.stdout}\n{completed.stderr}")
    return run_json([
        "dotnet", "run", "-c", "Release", "--no-build", "-f", "net10.0",
        "--project", str(CSHARP_PROJECT), "--", "--audit-scenarios-json",
    ])


def discover_python() -> dict[str, Any]:
    return run_json([sys.executable, str(PYTHON_BENCHMARK), "--list-scenarios-json"])


def validate_source(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("rows"), list):
        raise RuntimeError(f"{source} discovery returned an unsupported schema")
    rows = payload["rows"]
    for index, row in enumerate(rows):
        if not row.get("title") or not row.get("suite") or not isinstance(row.get("dtypes"), list):
            raise RuntimeError(f"{source} row {index} is incomplete: {row}")
        unknown = sorted(set(row["dtypes"]) - set(DTYPE_KEYS))
        if unknown:
            raise RuntimeError(f"{source} row {row['title']!r} uses unknown dtypes: {unknown}")
    return rows


def index_source(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    owners: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in rows:
        identity = NORMALIZE(str(row["title"]))
        signature = (str(row["title"]), str(row.get("type", "")), str(row.get("method", "")))
        for dtype in row["dtypes"]:
            cell = (identity, dtype)
            previous = owners.get(cell)
            if previous is not None and previous != signature:
                raise RuntimeError(
                    f"{source} discovery has an ambiguous normalized cell {cell}: "
                    f"{previous} vs {signature}")
            owners[cell] = signature

        target = identities.setdefault(identity, {
            "titles": set(), "suites": set(), "categories": set(), "dtypes": set(), "cases": []})
        target["titles"].add(str(row["title"]))
        target["suites"].add(str(row["suite"]))
        if row.get("category"):
            target["categories"].add(str(row["category"]))
        target["dtypes"].update(row["dtypes"])
        target["cases"].append({
            key: row[key] for key in ("title", "suite", "category", "type", "method") if row.get(key)
        })
    return identities


def build_audit(csharp_payload: dict[str, Any], python_payload: dict[str, Any]) -> dict[str, Any]:
    csharp = index_source(validate_source(csharp_payload, "C# BDN"), "C# BDN")
    python = index_source(validate_source(python_payload, "Python"), "Python")
    rows = []
    for identity in sorted(set(csharp) | set(python)):
        cs = csharp.get(identity, {"titles": set(), "suites": set(), "categories": set(), "dtypes": set(), "cases": []})
        py = python.get(identity, {"titles": set(), "suites": set(), "categories": set(), "dtypes": set(), "cases": []})
        cs_titles = sorted(cs["titles"])
        py_titles = sorted(py["titles"])
        display_title = cs_titles[0] if cs_titles else py_titles[0]
        cells = {}
        for dtype in DTYPE_KEYS:
            has_cs = dtype in cs["dtypes"]
            has_py = dtype in py["dtypes"]
            cells[dtype] = {
                "csharp": has_cs,
                "python": has_py,
                "comparable": has_cs and has_py,
                "state": "both" if has_cs and has_py else "csharp_only" if has_cs else "python_only" if has_py else "none",
            }
        rows.append({
            "identity": identity,
            "title": display_title,
            "suites": sorted({SUITE_ALIASES.get(suite, suite) for suite in cs["suites"] | py["suites"]}),
            "categories": sorted(cs["categories"] | py["categories"]),
            "csharp_titles": cs_titles,
            "python_titles": py_titles,
            "title_match": cs_titles == py_titles and bool(cs_titles),
            "csharp_cases": cs["cases"],
            "python_cases": py["cases"],
            "cells": cells,
            "counts": {
                "csharp": sum(cell["csharp"] for cell in cells.values()),
                "python": sum(cell["python"] for cell in cells.values()),
                "comparable": sum(cell["comparable"] for cell in cells.values()),
                "csharp_only": sum(cell["state"] == "csharp_only" for cell in cells.values()),
                "python_only": sum(cell["state"] == "python_only" for cell in cells.values()),
            },
        })

    source_blob = json.dumps({"csharp": csharp_payload, "python": python_payload}, sort_keys=True).encode()
    totals = defaultdict(int)
    for row in rows:
        for key, value in row["counts"].items():
            totals[key] += value
    return {
        "schema_version": 1,
        "audit_id": hashlib.sha256(source_blob).hexdigest()[:16],
        "definition": "One row per normalized official scenario title; one column per NumSharp dtype.",
        "dtypes": DTYPES,
        "sources": {
            "csharp": {key: value for key, value in csharp_payload.items() if key != "rows"},
            "python": {key: value for key, value in python_payload.items() if key != "rows"},
        },
        "totals": {"scenarios": len(rows), **dict(totals)},
        "rows": rows,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NumSharp benchmark scenario × dtype audit</title>
<style>
:root{color-scheme:dark;--bg:#0b1017;--panel:#111923;--panel2:#16212e;--line:#293747;--text:#e7edf4;--muted:#91a1b3;--good:#35c982;--bad:#ef6573;--warn:#f1b84b;--blue:#57a8ff;--shadow:0 18px 55px #0008;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -10%,#19324a 0,transparent 34%),var(--bg);color:var(--text)}
main{max-width:1900px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:18px}.eyebrow{color:#78c6ff;text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:800}.hero h1{font-size:clamp(28px,4vw,48px);line-height:1.05;margin:7px 0}.hero p{margin:0;color:var(--muted);max-width:850px}.audit-id{font:12px ui-monospace,monospace;color:var(--muted);white-space:nowrap}
.cards{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px;margin:18px 0}.card{background:#111923d9;border:1px solid var(--line);border-radius:13px;padding:14px;box-shadow:var(--shadow)}.card span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em}.card strong{display:block;font-size:25px;margin-top:5px}.card.good strong{color:var(--good)}.card.bad strong{color:var(--bad)}
.controls{position:sticky;top:0;z-index:12;display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#0b1017ee;border:1px solid var(--line);padding:11px;border-radius:14px;backdrop-filter:blur(12px);box-shadow:var(--shadow)}input,select,button{background:var(--panel2);border:1px solid #34475b;color:var(--text);border-radius:9px;padding:9px 11px;font:inherit}input{min-width:300px;flex:1}button{cursor:pointer}button.active{background:#1f5c82;border-color:#51b7f2}.lens{display:flex;gap:5px}.legend{margin:12px 2px;color:var(--muted);font-size:13px}.legend b{color:var(--text)}
.table-wrap{overflow:auto;max-height:calc(100vh - 245px);border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:var(--shadow)}table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:12px}th,td{border-right:1px solid #223040;border-bottom:1px solid #223040;padding:7px 8px;text-align:center}thead th{position:sticky;top:0;z-index:5;background:#172331;color:#b8c7d7;font-weight:800}th.title,td.title{position:sticky;left:0;text-align:left;min-width:350px;max-width:350px;z-index:4;background:#121b26}thead th.title{z-index:8;background:#1a2938}th.suite,td.suite{position:sticky;left:350px;min-width:112px;z-index:3;background:#121b26}thead th.suite{z-index:7;background:#1a2938}.dtype{min-width:67px}.dtype small{display:block;color:#728398;font-weight:500}.title code{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:normal}.identity{display:block;color:#718396;font-size:10px;margin-top:3px}.counts{min-width:105px;color:var(--muted)}
tbody tr:hover td{background:#172433}tbody tr:hover td.title,tbody tr:hover td.suite{background:#1a2938}.mark{font-size:16px}.cell.yes{background:#0f2b22}.cell.no{background:#2a161c}.cell.state-csharp_only{box-shadow:inset 0 -2px var(--warn)}.cell.state-python_only{box-shadow:inset 0 -2px var(--blue)}.title-mismatch{color:var(--warn);font-size:10px;display:block;margin-top:4px}.empty{padding:40px;text-align:center;color:var(--muted)}
.foot{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;margin:12px 2px 30px}.foot code{color:#c2d4e6}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:900px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr)}.hero{display:block}.audit-id{margin-top:10px}.table-wrap{max-height:calc(100vh - 320px)}input{min-width:180px}}
</style>
</head>
<body><main>
<section class="hero"><div><div class="eyebrow">Executable coverage contract</div><h1>Scenario × dtype audit</h1><p>Exact official benchmark titles discovered from BenchmarkDotNet attributes and NumPy's live suite builders. Every row is checked against all 15 NumSharp dtypes.</p></div><div class="audit-id" id="audit-id"></div></section>
<section class="cards" aria-label="Audit summary">
  <div class="card"><span>Scenario titles</span><strong id="stat-scenarios">—</strong></div>
  <div class="card"><span>Selected cells</span><strong id="stat-selected">—</strong></div>
  <div class="card good"><span>✅ covered</span><strong id="stat-covered">—</strong></div>
  <div class="card bad"><span>❌ missing</span><strong id="stat-missing">—</strong></div>
  <div class="card"><span>C#-only cells</span><strong id="stat-csharp-only">—</strong></div>
  <div class="card"><span>Python-only cells</span><strong id="stat-python-only">—</strong></div>
</section>
<section class="controls" aria-label="Audit controls">
  <input id="search" type="search" placeholder="Filter exact title, normalized identity, class, or method…" aria-label="Filter scenarios">
  <select id="suite" aria-label="Filter suite"><option value="">All suites</option></select>
  <select id="coverage" aria-label="Filter coverage"><option value="all">All rows</option><option value="gaps">Has ❌ in selected lens</option><option value="complete">All 15 ✅ in selected lens</option><option value="csharp-only">Has C#-only cell</option><option value="python-only">Has Python-only cell</option><option value="title-mismatch">Title mismatch</option></select>
  <div class="lens" role="group" aria-label="Coverage lens"><button data-lens="comparable" class="active">Comparable</button><button data-lens="csharp">C# BDN</button><button data-lens="python">Python</button></div>
</section>
<p class="legend"><b id="lens-name">Comparable</b> lens: ✅ means the selected source coverage exists; ❌ means it does not. Underlined amber cells are C#-only, blue cells are Python-only. Hover any cell for both-source evidence.</p>
<div class="table-wrap"><table><thead><tr id="head-row"><th class="title">Exact benchmark title</th><th class="suite">Suite</th><th class="counts">C# / Py / Both</th></tr></thead><tbody id="body"></tbody></table><div id="empty" class="empty" hidden>No scenarios match these filters.</div></div>
<div class="foot"><span>Generated by <code>python benchmark/scripts/generate_scenarios_html.py</code></span><span id="source-note"></span></div>
</main>
<script id="scenario-audit-data" type="application/json">__AUDIT_DATA__</script>
<script>
(() => {
const audit=JSON.parse(document.getElementById('scenario-audit-data').textContent);let lens='comparable';
const $=id=>document.getElementById(id),body=$('body'),search=$('search'),suite=$('suite'),coverage=$('coverage');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const selected=(cell)=>lens==='comparable'?cell.comparable:cell[lens];
const allCells=row=>audit.dtypes.map(dtype=>row.cells[dtype.key]);
const searchable=row=>[row.title,row.identity,...row.csharp_titles,...row.python_titles,...row.csharp_cases.flatMap(x=>[x.type,x.method])].join(' ').toLowerCase();
audit.dtypes.forEach(dtype=>$('head-row').insertAdjacentHTML('beforeend',`<th class="dtype" title="C#: ${esc(dtype.csharp)} · NumPy: ${esc(dtype.numpy||'no direct peer')}">${esc(dtype.key)}<small>${esc(dtype.csharp)}</small></th>`));
[...new Set(audit.rows.flatMap(row=>row.suites))].sort().forEach(name=>suite.insertAdjacentHTML('beforeend',`<option>${esc(name)}</option>`));
function rowVisible(row){const q=search.value.trim().toLowerCase();if(q&&!searchable(row).includes(q))return false;if(suite.value&&!row.suites.includes(suite.value))return false;const cells=allCells(row);if(coverage.value==='gaps'&&cells.every(selected))return false;if(coverage.value==='complete'&&!cells.every(selected))return false;if(coverage.value==='csharp-only'&&!cells.some(c=>c.state==='csharp_only'))return false;if(coverage.value==='python-only'&&!cells.some(c=>c.state==='python_only'))return false;if(coverage.value==='title-mismatch'&&row.title_match)return false;return true}
function render(){const rows=audit.rows.filter(rowVisible);body.innerHTML=rows.map(row=>{const details=`C# title: ${row.csharp_titles.join(' | ')||'missing'}\nPython title: ${row.python_titles.join(' | ')||'missing'}\nC# cases: ${row.csharp_cases.map(x=>`${x.type}.${x.method}`).join(' | ')||'missing'}`;const twin=row.title_match?'':`<span class="title-mismatch">Python: <code>${esc(row.python_titles.join(' | ')||'missing')}</code></span>`;const cells=audit.dtypes.map(dtype=>{const cell=row.cells[dtype.key],yes=selected(cell),tip=`${dtype.key}\nC# BDN: ${cell.csharp?'scheduled':'missing'}\nPython: ${cell.python?'scheduled':'missing'}\nComparable: ${cell.comparable?'yes':'no'}`;return `<td class="cell ${yes?'yes':'no'} state-${cell.state}" title="${esc(tip)}" aria-label="${esc(tip)}"><span class="mark" aria-hidden="true">${yes?'✅':'❌'}</span></td>`}).join('');return `<tr><td class="title" title="${esc(details)}"><code>${esc(row.title)}</code>${twin}<span class="identity">${esc(row.identity)}</span></td><td class="suite">${esc(row.suites.join(', '))}</td><td class="counts">${row.counts.csharp} / ${row.counts.python} / ${row.counts.comparable}</td>${cells}</tr>`}).join('');$('empty').hidden=rows.length>0;updateStats(rows)}
function updateStats(rows){const cells=rows.flatMap(allCells),covered=cells.filter(selected).length;$('stat-scenarios').textContent=rows.length.toLocaleString();$('stat-selected').textContent=cells.length.toLocaleString();$('stat-covered').textContent=covered.toLocaleString();$('stat-missing').textContent=(cells.length-covered).toLocaleString();$('stat-csharp-only').textContent=cells.filter(c=>c.state==='csharp_only').length.toLocaleString();$('stat-python-only').textContent=cells.filter(c=>c.state==='python_only').length.toLocaleString();$('lens-name').textContent={comparable:'Comparable twin',csharp:'C# BDN',python:'Python'}[lens]}
document.querySelectorAll('[data-lens]').forEach(button=>button.addEventListener('click',()=>{lens=button.dataset.lens;document.querySelectorAll('[data-lens]').forEach(x=>x.classList.toggle('active',x===button));render()}));[search,suite,coverage].forEach(control=>control.addEventListener('input',render));
$('audit-id').textContent=`audit ${audit.audit_id}`;$('source-note').textContent=`${audit.sources.csharp.source} · ${audit.sources.python.source} · NumPy ${audit.sources.python.numpy_version}`;render();
})();
</script></body></html>
'''


def render_html(audit: dict[str, Any]) -> str:
    encoded = json.dumps(audit, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__AUDIT_DATA__", encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-build", action="store_true", help="Reuse the existing Release benchmark build")
    parser.add_argument("--check", action="store_true", help="Fail if the generated HTML is not current")
    args = parser.parse_args()

    csharp = discover_csharp(build=not args.no_build)
    python = discover_python()
    audit = build_audit(csharp, python)
    rendered = render_html(audit)
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print(f"Scenario audit is stale: {output}", file=sys.stderr)
            return 1
        print(f"Scenario audit is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    totals = audit["totals"]
    print(
        f"Scenario audit: {totals['scenarios']} titles · {totals['csharp']} C# cells · "
        f"{totals['python']} Python cells · {totals['comparable']} comparable -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
