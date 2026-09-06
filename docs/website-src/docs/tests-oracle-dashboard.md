<link rel="stylesheet" href="https://unpkg.com/tippy.js@6.3.7/dist/tippy.css">

<style>
.ns-bench-dashboard {
  --good: #178a5a;
  --good-soft: #dff5ea;
  --near: #b78318;
  --near-soft: #fff1cc;
  --slow: #c35721;
  --slow-soft: #ffe5d6;
  --bad: #b4232c;
  --bad-soft: #ffe1e4;
  --quiet: #6f7b86;
  --ink: var(--bs-body-color);
  --line: var(--bs-border-color);
  --panel: var(--bs-body-bg);
  --muted-bg: rgba(108, 117, 125, 0.10);
  --heat-text: #14212c;
  --heat-muted-text: #50606b;
  color: var(--ink);
}

html[data-bs-theme="dark"] .ns-bench-dashboard {
  --heat-text: #f3eee7;
  --heat-muted-text: #c9c0b8;
}

.ns-bench-dashboard .bench-intro { padding: 0.25rem 0 0; }
.ns-bench-dashboard .bench-kicker {
  color: var(--quiet); font-size: 0.82rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.ns-bench-dashboard .bench-title {
  font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.02; margin: 0.35rem 0 0.6rem;
}
.ns-bench-dashboard .bench-subtitle {
  color: var(--quiet); max-width: 76ch; font-size: 1.04rem; margin: 0;
}
.ns-bench-dashboard .snapshot-strip {
  display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1rem;
}
.ns-bench-dashboard .snapshot-pill,
.ns-bench-dashboard .snapshot-meta {
  border: 1px solid var(--line); border-radius: 999px; padding: 0.28rem 0.65rem;
  color: var(--quiet); background: var(--muted-bg); font-size: 0.82rem;
}

.ns-bench-dashboard .metric-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem; margin: 1.25rem 0 1.5rem;
}
.ns-bench-dashboard .metric-card,
.ns-bench-dashboard .story-card,
.ns-bench-dashboard .priority-card {
  border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
}
.ns-bench-dashboard .metric-card { padding: 0.85rem; min-height: 7rem; }
.ns-bench-dashboard .metric-label {
  color: var(--quiet); font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
}
.ns-bench-dashboard .metric-value {
  font-size: 2rem; line-height: 1.05; font-weight: 750; margin: 0.45rem 0 0.3rem;
}
.ns-bench-dashboard .metric-note { color: var(--quiet); font-size: 0.86rem; margin: 0; }
.ns-bench-dashboard .metric-good { color: var(--good); }
.ns-bench-dashboard .metric-near { color: var(--near); }
.ns-bench-dashboard .metric-slow { color: var(--slow); }
.ns-bench-dashboard .metric-bad { color: var(--bad); }

.ns-bench-dashboard .section-head {
  display: flex; align-items: baseline; justify-content: flex-start; gap: 0.65rem;
  border-top: 1px solid var(--line); padding-top: 1.2rem; margin: 1.45rem 0 0.75rem;
}
.ns-bench-dashboard .section-head h2 { font-size: 1.22rem; margin: 0; white-space: nowrap; }
.ns-bench-dashboard .section-note { color: var(--quiet); font-size: 0.86rem; margin: 0; }

.ns-bench-dashboard .read-guide { margin: 1.45rem 0 0; padding: 0 0 0.15rem; }
.ns-bench-dashboard .read-guide + section .section-head { border-top: 0; margin-top: 1rem; padding-top: 0; }
.ns-bench-dashboard .read-guide-head {
  align-items: end; display: flex; gap: 1rem; justify-content: space-between; margin-bottom: 0.7rem;
}
.ns-bench-dashboard .read-guide h2 { font-size: 1.22rem; margin: 0; }
.ns-bench-dashboard .guide-kicker {
  color: var(--quiet); font-size: 0.76rem; font-weight: 750;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.ns-bench-dashboard .guide-primer {
  border: 1px solid color-mix(in srgb, var(--line) 86%, transparent);
  border-radius: 8px; display: grid; margin: 0 0 0.85rem;
}
.ns-bench-dashboard .guide-primer-row {
  align-items: start; display: grid; gap: 0.7rem;
  grid-template-columns: 10rem minmax(0, 1fr); padding: 0.56rem 0.68rem;
}
.ns-bench-dashboard .guide-primer-row + .guide-primer-row {
  border-top: 1px solid color-mix(in srgb, var(--line) 74%, transparent);
}
.ns-bench-dashboard .guide-primer-term { color: var(--ink); font-size: 0.8rem; font-weight: 750; }
.ns-bench-dashboard .guide-primer-detail { color: var(--quiet); font-size: 0.82rem; }
.ns-bench-dashboard .guide-grid {
  display: grid; gap: 0.75rem; grid-template-columns: repeat(2, minmax(0, 1fr));
}
.ns-bench-dashboard .guide-block h3 { font-size: 0.82rem; margin: 0 0 0.25rem; }
.ns-bench-dashboard .guide-block p { color: var(--quiet); font-size: 0.84rem; margin: 0; }
.ns-bench-dashboard .guide-band-row { display: flex; flex-wrap: wrap; gap: 0.42rem; margin-top: 0.38rem; }
.ns-bench-dashboard .guide-band-block { margin-top: 0.78rem; }
.ns-bench-dashboard .guide-band {
  align-items: center; border: 1px solid color-mix(in srgb, var(--line) 78%, transparent);
  border-radius: 999px; display: inline-flex; font-size: 0.8rem; font-weight: 650;
  gap: 0.42rem; padding: 0.34rem 0.58rem; white-space: nowrap;
}
.ns-bench-dashboard .guide-band::before {
  border-radius: 999px; content: ""; display: block; height: 0.62rem; width: 0.62rem;
}
.ns-bench-dashboard .guide-band.faster::before { background: var(--good); }
.ns-bench-dashboard .guide-band.close::before { background: #0e7490; }
.ns-bench-dashboard .guide-band.slower::before { background: var(--near); }
.ns-bench-dashboard .guide-band.much::before { background: var(--bad); }
.ns-bench-dashboard .guide-band.nodata::before { background: #87909a; }

.ns-bench-dashboard .status-bar {
  display: flex; position: relative; height: 1.2rem; border-radius: 8px;
  border: 1px solid var(--line); background: var(--muted-bg); isolation: isolate;
}
.ns-bench-dashboard .status-segment {
  display: block; position: relative; flex: 0 0 var(--w); min-width: 0.4rem; height: 100%;
  cursor: pointer; border-left: 1px solid rgba(255, 255, 255, 0.32);
  transition: box-shadow 120ms ease, transform 120ms ease, filter 120ms ease;
}
.ns-bench-dashboard .status-segment:first-child { border-left: 0; border-radius: 7px 0 0 7px; }
.ns-bench-dashboard .status-segment:last-child { border-radius: 0 7px 7px 0; }
.ns-bench-dashboard .status-segment:hover,
.ns-bench-dashboard .status-segment:focus-visible {
  filter: saturate(1.12) brightness(1.04); outline: 2px solid color-mix(in srgb, var(--ink) 24%, transparent);
  outline-offset: 2px; transform: translateY(-1px); z-index: 5;
}
.ns-bench-dashboard .s-extreme { background: linear-gradient(90deg, #0e7490, #14b8a6); }
.ns-bench-dashboard .s-faster { background: var(--good); }
.ns-bench-dashboard .s-close { background: #d8a528; }
.ns-bench-dashboard .s-slower { background: var(--slow); }
.ns-bench-dashboard .s-much { background: var(--bad); }
.ns-bench-dashboard .s-nodata { background: #c7ccd1; }
.ns-bench-dashboard .legend-grid {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.45rem 0.7rem; margin: 0.65rem 0 0;
}
.ns-bench-dashboard .legend-item { font-size: 0.8rem; color: var(--quiet); }
.ns-bench-dashboard .legend-swatch {
  display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 3px;
  margin-right: 0.35rem; vertical-align: -0.08rem;
}

.ns-bench-dashboard .bar-table { display: grid; gap: 0.5rem; }
.ns-bench-dashboard .bar-row {
  display: grid; grid-template-columns: minmax(8rem, 14rem) minmax(12rem, 1fr) 5.2rem 6.4rem;
  align-items: center; gap: 0.7rem; border-radius: 8px; padding: 0.08rem 0.12rem;
}
.ns-bench-dashboard .bar-row:hover { background: var(--muted-bg); }
.ns-bench-dashboard .bar-label { font-weight: 650; font-size: 0.9rem; }
.ns-bench-dashboard .bar-track {
  position: relative; height: 0.72rem; border-radius: 999px; background: var(--muted-bg); overflow: hidden;
}
.ns-bench-dashboard .bar-fill {
  display: block; height: 100%; width: var(--w); background: var(--tone); border-radius: 999px;
}
.ns-bench-dashboard .bar-score { font-weight: 750; text-align: right; }
.ns-bench-dashboard .bar-count { color: var(--quiet); font-size: 0.82rem; text-align: right; }

.ns-bench-dashboard .dtype-carousel { display: grid; gap: 0.7rem; }
.ns-bench-dashboard .dtype-tabs {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.35rem; max-width: 30rem;
}
.ns-bench-dashboard .dtype-tab {
  appearance: none; border: 1px solid var(--line); border-radius: 8px;
  background: color-mix(in srgb, var(--panel) 92%, var(--heat-text) 8%);
  color: var(--heat-text) !important; cursor: pointer; font: inherit; font-size: 0.86rem;
  font-weight: 700; line-height: 1; padding: 0.58rem 0.55rem; text-align: center;
}
.ns-bench-dashboard .dtype-tab.is-active {
  background: var(--heat-text); border-color: var(--heat-text); color: var(--panel) !important;
}
.ns-bench-dashboard .dtype-lens-note {
  color: var(--quiet); font-size: 0.84rem; line-height: 1.45; margin: -0.1rem 0 0.05rem;
  max-width: 58rem;
}
.ns-bench-dashboard .dtype-panel {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(6.6rem, 1fr)); gap: 0.55rem; min-width: 0;
}
.ns-bench-dashboard .dtype-panel[hidden] { display: none; }
.ns-bench-dashboard .dtype-cell {
  --heat: var(--quiet); border-radius: 8px; border: 1px solid color-mix(in srgb, var(--heat) 42%, var(--line));
  background: linear-gradient(90deg, color-mix(in srgb, var(--heat) 18%, transparent) 0 var(--heat-width), transparent var(--heat-width)),
              color-mix(in srgb, var(--heat) 10%, var(--panel));
  color: var(--heat-text) !important; min-width: 0; padding: 0.62rem 0.58rem;
}
.ns-bench-dashboard .dtype-name { display: block; font-weight: 750; font-size: 0.86rem; overflow-wrap: anywhere; }
.ns-bench-dashboard .dtype-score { display: block; margin-top: 0.28rem; font-size: 1.08rem; font-weight: 780; }
.ns-bench-dashboard .dtype-count { display: block; color: var(--heat-muted-text); font-size: 0.76rem; margin-top: 0.18rem; }
.ns-bench-dashboard .dtype-meter {
  display: block; height: 0.28rem; margin-top: 0.5rem; border-radius: 999px;
  background: color-mix(in srgb, var(--ink) 10%, transparent); overflow: hidden;
}
.ns-bench-dashboard .dtype-meter span { display: block; width: var(--heat-width); height: 100%; background: var(--heat); }
.ns-bench-dashboard .heat-best { --heat: #109862; }
.ns-bench-dashboard .heat-good { --heat: #35a86e; }
.ns-bench-dashboard .heat-near { --heat: #c89422; }
.ns-bench-dashboard .heat-slow { --heat: #cf6728; }
.ns-bench-dashboard .heat-bad { --heat: #c92f3a; }

.ns-bench-dashboard .function-explorer-shell {
  border: 1px solid var(--line); border-radius: 8px; display: grid;
  grid-template-columns: minmax(15rem, 20rem) minmax(0, 1fr); min-height: 34rem; overflow: hidden;
}
.ns-bench-dashboard .function-sidebar {
  border-right: 1px solid var(--line); display: grid; grid-template-rows: auto 1fr; min-width: 0;
}
.ns-bench-dashboard .function-controls {
  background: color-mix(in srgb, var(--panel) 94%, var(--ink) 6%); display: grid; gap: 0.55rem; padding: 0.72rem;
}
.ns-bench-dashboard .function-search,
.ns-bench-dashboard .function-select {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px; color: var(--ink);
  font: inherit; font-size: 0.86rem; min-width: 0; padding: 0.48rem 0.55rem; width: 100%;
}
.ns-bench-dashboard .function-filter-row {
  display: grid; gap: 0.45rem; grid-template-columns: repeat(2, minmax(0, 1fr));
}
.ns-bench-dashboard .function-list-meta { color: var(--quiet); font-size: 0.76rem; }
.ns-bench-dashboard .function-list { max-height: 40rem; overflow: auto; }
.ns-bench-dashboard .function-list-item {
  --func-tone: var(--quiet); appearance: none; background: transparent; border: 0;
  border-bottom: 1px solid color-mix(in srgb, var(--line) 68%, transparent); border-left: 4px solid transparent;
  color: var(--ink); cursor: pointer; display: grid; gap: 0.18rem 0.55rem;
  grid-template-columns: minmax(0, 1fr) auto; padding: 0.55rem 0.68rem; text-align: left; width: 100%;
}
.ns-bench-dashboard .function-list-item:hover,
.ns-bench-dashboard .function-list-item.is-active {
  background: color-mix(in srgb, var(--func-tone) 10%, var(--panel)); border-left-color: var(--func-tone);
}
.ns-bench-dashboard .function-list-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.84rem;
  font-weight: 750; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ns-bench-dashboard .function-list-score { color: var(--func-tone); font-size: 0.84rem; font-weight: 800; }
.ns-bench-dashboard .function-list-detail {
  color: var(--quiet); font-size: 0.74rem; grid-column: 1 / -1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ns-bench-dashboard .function-detail { align-content: start; display: grid; gap: 0.85rem; min-width: 0; padding: 0.9rem; }
.ns-bench-dashboard .function-empty { color: var(--quiet); padding: 1rem; }
.ns-bench-dashboard .function-detail-head {
  align-items: start; display: flex; gap: 1rem; justify-content: space-between;
}
.ns-bench-dashboard .function-title {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.1rem; font-weight: 800;
}
.ns-bench-dashboard .function-subtitle { color: var(--quiet); font-size: 0.82rem; margin: 0.25rem 0 0; }
.ns-bench-dashboard .function-ratio-pill {
  --func-tone: var(--quiet); border: 1px solid color-mix(in srgb, var(--func-tone) 45%, var(--line));
  border-radius: 999px; color: var(--func-tone); font-size: 0.82rem; font-weight: 800;
  padding: 0.3rem 0.55rem; white-space: nowrap;
}
.ns-bench-dashboard .function-stat-strip {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.55rem;
}
.ns-bench-dashboard .function-stat { border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem; }
.ns-bench-dashboard .function-stat span { color: var(--quiet); display: block; font-size: 0.72rem; }
.ns-bench-dashboard .function-stat strong { display: block; font-size: 1rem; margin-top: 0.18rem; }
.ns-bench-dashboard .function-table-scroll { overflow: auto; }
.ns-bench-dashboard .function-table { border-collapse: collapse; font-size: 0.8rem; width: 100%; }
.ns-bench-dashboard .function-table th,
.ns-bench-dashboard .function-table td { border-bottom: 1px solid var(--line); padding: 0.43rem 0.5rem; text-align: left; }
.ns-bench-dashboard .function-table thead th {
  background: color-mix(in srgb, var(--panel) 92%, var(--ink) 8%); color: var(--quiet); font-weight: 750;
}
.ns-bench-dashboard .function-table .num { text-align: right; }
.ns-bench-dashboard .function-block-title { font-size: 0.82rem; font-weight: 800; margin-bottom: 0.45rem; }
.ns-bench-dashboard .func-tone-extreme { --func-tone: #0e7490; }
.ns-bench-dashboard .func-tone-good { --func-tone: var(--good); }
.ns-bench-dashboard .func-tone-near { --func-tone: var(--near); }
.ns-bench-dashboard .func-tone-slow { --func-tone: var(--slow); }
.ns-bench-dashboard .func-tone-bad { --func-tone: var(--bad); }
.ns-bench-dashboard .func-tone-empty { --func-tone: #87909a; }
.ns-bench-dashboard .ns-load-more-wrap { padding: 0.65rem 0; text-align: center; }
.ns-bench-dashboard .ns-load-more-button {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px; color: var(--ink);
  cursor: pointer; font: inherit; font-size: 0.8rem; padding: 0.4rem 0.7rem;
}

.ns-bench-dashboard .story-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem;
}
.ns-bench-dashboard .story-card { padding: 0.85rem; }
.ns-bench-dashboard .story-card h3 { font-size: 0.9rem; margin: 0; }
.ns-bench-dashboard .story-card strong { display: block; font-size: 1.35rem; margin-top: 0.45rem; }
.ns-bench-dashboard .story-card p { color: var(--quiet); font-size: 0.82rem; margin: 0.4rem 0 0; }
.ns-bench-dashboard .priority-grid { display: grid; gap: 0.75rem; }
.ns-bench-dashboard .priority-card { padding: 0.85rem; }
.ns-bench-dashboard .priority-list { columns: 2; margin: 0; padding-left: 1.25rem; }
.ns-bench-dashboard .priority-list li { break-inside: avoid; margin: 0 0 0.45rem; padding-right: 1rem; }
.ns-bench-dashboard .report-list { list-style: none; margin: 0; padding: 0; }
.ns-bench-dashboard .report-list li { margin: 0.45rem 0; padding-left: 1.1rem; position: relative; }
.ns-bench-dashboard .report-list li::before { color: var(--good); content: "•"; left: 0; position: absolute; }

@media (max-width: 991.98px) {
  .ns-bench-dashboard .metric-grid,
  .ns-bench-dashboard .story-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ns-bench-dashboard .function-explorer-shell { grid-template-columns: 1fr; }
  .ns-bench-dashboard .function-sidebar { border-bottom: 1px solid var(--line); border-right: 0; }
  .ns-bench-dashboard .function-list { max-height: 20rem; }
}
@media (max-width: 575.98px) {
  .ns-bench-dashboard .metric-grid,
  .ns-bench-dashboard .story-grid,
  .ns-bench-dashboard .legend-grid,
  .ns-bench-dashboard .function-stat-strip { grid-template-columns: 1fr; }
  .ns-bench-dashboard .guide-grid { grid-template-columns: 1fr; }
  .ns-bench-dashboard .guide-primer-row { grid-template-columns: 1fr; }
  .ns-bench-dashboard .bar-row { grid-template-columns: 1fr auto; }
  .ns-bench-dashboard .bar-track { grid-column: 1 / -1; grid-row: 2; }
  .ns-bench-dashboard .bar-count { display: none; }
  .ns-bench-dashboard .priority-list { columns: 1; }
}
</style>

<div class="ns-bench-dashboard" data-tests-oracle-dashboard>
  <section class="bench-intro">
    <div class="bench-kicker">NumSharp verification lab</div>
    <h2 class="bench-title">Unit Tests &amp; Oracle Dashboard</h2>
    <p class="bench-subtitle">
      A correctness inventory with unlike units kept separate: reflected unit tests and unit-test
      classes, committed NumPy 2.4.2 Oracle test cases, specialized flags/layout/format cases,
      execution gates, and live interoperability suites.
    </p>
    <div class="snapshot-strip" aria-label="Inventory details">
      <span class="snapshot-meta">Committed generated inventory</span>
      <span class="snapshot-meta" data-numpy-version>NumPy —</span>
      <span class="snapshot-meta" data-schema-version>Schema —</span>
      <span class="snapshot-meta">Reflected net8.0 assemblies</span>
    </div>
  </section>

  <section class="metric-grid" aria-label="Headline test and oracle metrics">
    <article class="metric-card">
      <div class="metric-label">Unit tests</div>
      <div class="metric-value metric-good" data-test-declarations>—</div>
      <p class="metric-note" data-test-note>One reflected method declaration; not an executed result</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Unit test classes</div>
      <div class="metric-value metric-good" data-test-classes>—</div>
      <p class="metric-note" data-class-note>Grouped ownership surface across the reflected assemblies</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Oracle test cases</div>
      <div class="metric-value metric-good" data-oracle-test-cases>—</div>
      <p class="metric-note" data-oracle-case-note>Corpus plus specialized flags, layout, and NPY/NPZ cases</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Oracle test classes</div>
      <div class="metric-value metric-good" data-oracle-test-classes>—</div>
      <p class="metric-note" data-oracle-class-note>Classes that run committed Oracle cases and harness checks</p>
    </article>
  </section>

  <section class="read-guide" aria-labelledby="tests-guide-title">
    <div class="read-guide-head"><div><div class="guide-kicker">Correctness evidence</div><h2 id="tests-guide-title">Legend &amp; How To Read</h2></div></div>
    <div class="guide-primer">
      <div class="guide-primer-row"><div class="guide-primer-term">Unit test</div><div class="guide-primer-detail">One reflected unit-test method. DataRow and DynamicData metadata is retained, but the dashboard does not invent or headline an execution count.</div></div>
      <div class="guide-primer-row"><div class="guide-primer-term">Oracle test case</div><div class="guide-primer-detail">One committed input/expected-result case: operands, dtype, shape, layout, parameters, result kind, bytes, or verbatim error. Two BLAS host-pin records are metadata and are explicitly excluded.</div></div>
      <div class="guide-primer-row"><div class="guide-primer-term">Not a pass rate</div><div class="guide-primer-detail">This is a deterministic inventory and evidence-strength view. Runtime pass/fail belongs to dotnet test; exclusions and open bugs are shown directly rather than counted as passes.</div></div>
    </div>
    <div class="guide-grid">
      <div class="guide-block"><h3>Unit test inventory</h3><p>Unit tests remain summarized by execution policy and capability group; they are intentionally kept out of the Oracle explorer.</p></div>
      <div class="guide-block"><h3>Oracle explorer</h3><p>Browse every Oracle operation by test cases, files, recorded layout labels, dtypes, parameters, result kinds, and explicit errors.</p></div>
    </div>
    <div class="guide-block guide-band-block">
      <h3>Status bands</h3>
      <div class="guide-band-row">
        <span class="guide-band faster">Default run: ordinary unit-test execution</span>
        <span class="guide-band close">Platform gated: conditional runner</span>
        <span class="guide-band slower">Manual gated: explicit or high-memory</span>
        <span class="guide-band much">Known bug gate: intentionally excluded</span>
        <span class="guide-band nodata">Ignored: no runtime execution</span>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Unit Test Execution Policy</h2><p class="section-note">Unit-test methods by runner policy; these are inventory counts, not pass/fail results</p></div>
    <div class="status-bar" data-status-bar></div>
    <div class="legend-grid" data-status-legend></div>
  </section>

  <section>
    <div class="section-head"><h2>Unit Test Groups</h2><p class="section-note">Source-folder suites deduplicated into broad capability groups; rows show unit tests and classes</p></div>
    <div class="bar-table" data-suite-scoreboard></div>
  </section>

  <section>
    <div class="section-head"><h2>Dtype Coverage</h2><p class="section-note">Non-exclusive Oracle test case↔dtype links; mixed-dtype cases can contribute to several dtypes</p></div>
    <div class="dtype-carousel" data-dtype-carousel>
      <div class="dtype-tabs" role="tablist">
        <button class="dtype-tab is-active" type="button" role="tab" aria-selected="true" data-panel="links">Test case links</button>
        <button class="dtype-tab" type="button" role="tab" aria-selected="false" data-panel="labels">Oracle operations</button>
        <button class="dtype-tab" type="button" role="tab" aria-selected="false" data-panel="files">Corpus files</button>
      </div>
      <p class="dtype-lens-note" data-dtype-lens-note aria-live="polite"></p>
      <div data-dtype-panels></div>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Specialized Oracle Tests</h2><p class="section-note">Oracle test cases outside the common operation/index corpus schema</p></div>
    <div class="story-grid" data-subsystem-signals></div>
  </section>

  <section class="function-explorer" aria-labelledby="evidence-explorer-title">
    <div class="section-head"><h2 id="evidence-explorer-title">Oracle Evidence Explorer</h2><p class="section-note">Oracle operations and their committed test cases; unit tests stay in the aggregate inventory above</p></div>
    <div class="function-explorer-shell">
      <aside class="function-sidebar">
        <div class="function-controls">
          <input class="function-search" type="search" placeholder="Search Oracle operations" aria-label="Search Oracle operations" data-evidence-search>
          <div class="function-filter-row">
            <select class="function-select" aria-label="Filter evidence group" data-evidence-filter><option value="">All groups</option></select>
            <select class="function-select" aria-label="Sort evidence" data-evidence-sort><option value="largest">Largest first</option><option value="review">Coverage review first</option><option value="name">Name</option></select>
          </div>
          <div class="function-list-meta" data-evidence-meta>Loading inventory...</div>
        </div>
        <div class="function-list" role="listbox" data-evidence-list></div>
      </aside>
      <article class="function-detail" aria-live="polite" data-evidence-detail><div class="function-empty">Loading correctness evidence...</div></article>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Coverage Review Queue</h2><p class="section-note">Factual evidence counts first; applicability-aware expansion techniques second</p></div>
    <div class="priority-grid"><article class="priority-card"><ol class="priority-list" data-priorities></ol></article></div>
  </section>

  <section>
    <div class="section-head"><h2>Full Reports</h2><p class="section-note">Generated data and the human-maintained oracle ledgers</p></div>
    <ul class="report-list">
      <li><a href="../../../test/inventory/generated/tests-oracle-report.csv">Full unit test inventory CSV</a> — one row per reflected unit-test method.</li>
      <li><a href="../../../test/inventory/generated/summary.md">Generated inventory summary</a> — headline counts and strength queue.</li>
      <li><a href="https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Oracle/Fuzz/README.md">Oracle architecture and divergence ledger</a>.</li>
      <li><a href="https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Oracle/Fuzz/COVERAGE_GAPS.md">Coverage map and expansion techniques</a>.</li>
      <li><a href="https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Oracle/Fuzz/JOURNEY3_TOUCHED_FUNCTIONS.md">Journey3 touched-function receipt</a>.</li>
    </ul>
  </section>
</div>

<script src="https://unpkg.com/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://unpkg.com/tippy.js@6.3.7/dist/tippy.umd.min.js"></script>
<script>
(() => {
  const root = document.querySelector("[data-tests-oracle-dashboard]");
  if (!root) return;
  const reportUrl = new URL(window.NUMSHARP_TESTS_ORACLE_REPORT_URL || "data/tests-oracle-report.json", window.location.href);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  const number = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));
  const pct = (part, total) => total ? Math.max(0.4, part * 100 / total) : 0;
  const statusMeta = {
    active: { label: "Default run", tone: "s-faster", func: "func-tone-good", color: "var(--good)" },
    platform: { label: "Platform gated", tone: "s-extreme", func: "func-tone-extreme", color: "#0e7490" },
    manual: { label: "Manual gated", tone: "s-close", func: "func-tone-near", color: "var(--near)" },
    open_bug: { label: "Known bug gate", tone: "s-much", func: "func-tone-bad", color: "var(--bad)" },
    ignored: { label: "Ignored", tone: "s-nodata", func: "func-tone-empty", color: "#87909a" }
  };
  const heatClass = (value, max) => {
    const ratio = max ? value / max : 0;
    return ratio >= 0.75 ? "heat-best" : ratio >= 0.45 ? "heat-good" : ratio >= 0.2 ? "heat-near" : ratio >= 0.08 ? "heat-slow" : "heat-bad";
  };
  const loadMore = (scope) => {
    scope.querySelector("[data-load-more]")?.addEventListener("click", (event) => {
      scope.querySelectorAll("[data-load-more-row][hidden]").forEach((row, index) => { if (index < 50) row.hidden = false; });
      const left = scope.querySelectorAll("[data-load-more-row][hidden]").length;
      if (!left) event.currentTarget.closest(".ns-load-more-wrap").remove();
      else event.currentTarget.querySelector("span").textContent = `(${left} remaining)`;
    });
  };

  fetch(reportUrl, { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`Unit Tests & Oracle report HTTP ${response.status}`);
    return response.json();
  }).then((data) => {
    const tests = data.summary.tests;
    const oracle = data.summary.oracle;
    root.querySelector("[data-numpy-version]").textContent = `NumPy ${data.numpyVersion}`;
    root.querySelector("[data-schema-version]").textContent = `Schema ${data.schemaVersion}`;
    root.querySelector("[data-test-declarations]").textContent = number(tests.methods);
    root.querySelector("[data-test-note]").textContent = `${number(tests.statuses.active)} default run · ${number(tests.methods - tests.statuses.active)} gated`;
    root.querySelector("[data-test-classes]").textContent = number(data.testClasses.length);
    root.querySelector("[data-class-note]").textContent = `${number(Object.keys(tests.projects).length)} reflected assemblies · ${number(tests.dynamicMethods)} DynamicData methods`;
    root.querySelector("[data-oracle-test-cases]").textContent = number(oracle.totalSerializedContracts);
    root.querySelector("[data-oracle-case-note]").textContent = `${number(oracle.corpusRows)} corpus · ${number(oracle.specializedOracleCases)} specialized`;
    root.querySelector("[data-oracle-test-classes]").textContent = number(tests.oracleTestClasses);
    root.querySelector("[data-oracle-class-note]").textContent = "Corpus and specialized runners; live Python interop excluded";

    const totalMethods = Object.values(tests.statuses).reduce((a, b) => a + Number(b), 0);
    const statusOrder = ["active", "platform", "manual", "open_bug", "ignored"];
    root.querySelector("[data-status-bar]").innerHTML = statusOrder.filter((key) => tests.statuses[key]).map((key) => {
      const count = Number(tests.statuses[key]); const meta = statusMeta[key];
      return `<span class="status-segment ${meta.tone}" tabindex="0" style="--w:${pct(count, totalMethods)}%" title="${meta.label}: ${number(count)} unit tests (${(count * 100 / totalMethods).toFixed(2)}%)"></span>`;
    }).join("");
    root.querySelector("[data-status-legend]").innerHTML = statusOrder.filter((key) => tests.statuses[key]).map((key) => {
      const meta = statusMeta[key];
      return `<span class="legend-item"><i class="legend-swatch" style="background:${meta.color}"></i>${meta.label}: ${number(tests.statuses[key])}</span>`;
    }).join("");

    const suiteGroupRules = [
      ["Engine, storage & views", ["Backends & kernels", "Shape & views", "Lifetime"]],
      ["Creation, casting & dtypes", ["Creation", "Casting", "NewDtypes", "Generic"]],
      ["Manipulation & indexing", ["Manipulation", "Indexing", "LongIndexing"]],
      ["Math, logic & statistics", ["Math", "Logic", "Statistics", "Operations"]],
      ["Linear algebra, FFT & polynomials", ["Linear algebra", "Fourier", "Polynomials"]],
      ["Selection, sorting & searching", ["Selection", "Sorting & searching", "Sorting Searching Counting"]],
      ["API surface & utilities", ["API & iteration", "API audit", "NpApiOverloads", "Utilities", "General", "Documentation", "Extensions", "Assembly"]],
      ["I/O & interoperability", ["I/O & formats", "Interop", "Python interoperability", "NumSharp.Bitmap"]],
      ["Random", ["Random"]],
      ["Oracle, parity & regressions", ["Differential oracle", "NumPy port regressions", "Open bugs", "Issues"]]
    ];
    const suiteGroupName = (area) => suiteGroupRules.find(([, areas]) => areas.includes(area))?.[0] || "Other";
    const suiteGroups = [...data.testClasses.reduce((groups, row) => {
      const name = suiteGroupName(row.area);
      if (!groups.has(name)) groups.set(name, { name, methods: 0, classes: 0, active: 0, openBugs: 0, oracleTagged: 0, projects: new Set(), areas: new Set() });
      const group = groups.get(name);
      group.methods += Number(row.methods);
      group.classes += 1;
      group.active += Number(row.statuses.active || 0);
      group.openBugs += Number(row.statuses.open_bug || 0);
      group.oracleTagged += Object.values(row.oracleKinds).reduce((sum, value) => sum + Number(value), 0);
      group.projects.add(row.project);
      group.areas.add(row.area);
      return groups;
    }, new Map()).values()].sort((a, b) => b.methods - a.methods);
    const maxSuite = Math.max(...suiteGroups.map((row) => row.methods));
    const suiteBoard = root.querySelector("[data-suite-scoreboard]");
    suiteBoard.innerHTML = suiteGroups.map((row) => {
      const openRatio = row.openBugs / Math.max(1, row.methods);
      const gated = row.methods - row.active;
      const status = openRatio > 0.20 ? "open_bug" : gated > 0 ? "manual" : "active";
      const meta = statusMeta[status];
      const projectText = [...row.projects].map((project) => project.replace("NumSharp.Tests.", "")).join(" + ");
      const oracleSuffix = row.oracleTagged ? ` · ${number(row.oracleTagged)} oracle-tagged` : "";
      const title = `${projectText} · ${number(row.active)} default · ${number(gated)} gated${oracleSuffix} · areas: ${[...row.areas].join(", ")}`;
      return `<div class="bar-row" title="${escapeHtml(title)}">
        <span class="bar-label">${escapeHtml(row.name)}</span><span class="bar-track"><i class="bar-fill" style="--w:${row.methods * 100 / maxSuite}%;--tone:${meta.color}"></i></span>
        <span class="bar-score">${number(row.methods)}</span><span class="bar-count">${number(row.classes)} classes</span></div>`;
    }).join("");

    const dtypeNames = Object.keys(oracle.dtypeRows).sort((a, b) => oracle.dtypeRows[b] - oracle.dtypeRows[a]);
    const dtypeOps = Object.fromEntries(dtypeNames.map((dtype) => [dtype, data.oracleOps.filter((op) => op.dtypes.includes(dtype)).length]));
    const dtypeFiles = Object.fromEntries(dtypeNames.map((dtype) => [dtype, data.oracleFiles.filter((file) => file.contractRows > 0 && file.dtypes.includes(dtype)).length]));
    const lenses = { links: oracle.dtypeRows, labels: dtypeOps, files: dtypeFiles };
    const lensLabels = { links: "test case links", labels: "Oracle operations", files: "corpus files" };
    const lensDescriptions = {
      links: "A test case link means one Oracle test case references this dtype as an input, requested dtype, or expected output. Mixed-dtype cases can appear under several dtypes, so these columns are not additive.",
      labels: "The number of distinct Oracle operations with at least one test case referencing this dtype.",
      files: "The number of committed JSONL corpus files containing at least one Oracle test case that references this dtype."
    };
    const panels = root.querySelector("[data-dtype-panels]");
    const lensNote = root.querySelector("[data-dtype-lens-note]");
    lensNote.textContent = lensDescriptions.links;
    panels.innerHTML = Object.entries(lenses).map(([lens, values], panelIndex) => {
      const max = Math.max(...Object.values(values));
      return `<div class="dtype-panel" id="dtype-panel-${lens}" role="tabpanel" ${panelIndex ? "hidden" : ""}>${dtypeNames.map((dtype) => {
        const value = Number(values[dtype] || 0); const width = max ? value * 100 / max : 0;
        return `<div class="dtype-cell ${heatClass(value, max)}" style="--heat-width:${width}%"><span class="dtype-name">${escapeHtml(dtype)}</span><span class="dtype-score">${number(value)}</span><span class="dtype-count">${lensLabels[lens]}</span><span class="dtype-meter"><span></span></span></div>`;
      }).join("")}</div>`;
    }).join("");
    root.querySelectorAll(".dtype-tab").forEach((tab) => tab.addEventListener("click", () => {
      root.querySelectorAll(".dtype-tab").forEach((candidate) => { const active = candidate === tab; candidate.classList.toggle("is-active", active); candidate.setAttribute("aria-selected", String(active)); });
      root.querySelectorAll(".dtype-panel").forEach((panel) => panel.hidden = panel.id !== `dtype-panel-${tab.dataset.panel}`);
      lensNote.textContent = lensDescriptions[tab.dataset.panel];
    }));

    root.querySelector("[data-subsystem-signals]").innerHTML = `
      <article class="story-card"><h3>Flags test cases</h3><strong class="metric-good">${number(oracle.flagsOracleRows)}</strong><p>Serialized ndarray flag expectations outside the common operation schema.</p></article>
      <article class="story-card"><h3>Layout test cases</h3><strong class="metric-good">${number(oracle.layoutOracleRows)}</strong><p>Dedicated layout-parity cases outside the common operation schema.</p></article>
      <article class="story-card"><h3>NPY/NPZ test cases</h3><strong class="metric-good">${number(oracle.npyOracleCases)}</strong><p>Byte and file-format test cases from the committed archive manifest.</p></article>
      <article class="story-card"><h3>Host-pin metadata</h3><strong class="metric-good">${number(oracle.hostPinRecords)}</strong><p>BLAS/platform provenance records; explicitly not counted as replay cases.</p></article>`;

    const explorer = {
      search: root.querySelector("[data-evidence-search]"), filter: root.querySelector("[data-evidence-filter]"),
      sort: root.querySelector("[data-evidence-sort]"), meta: root.querySelector("[data-evidence-meta]"),
      list: root.querySelector("[data-evidence-list]"), detail: root.querySelector("[data-evidence-detail]"), active: null
    };
    const opItems = data.oracleOps.map((row) => ({
      ...row, name: row.op, group: row.families.join(" + "), size: row.cases,
      review: Number(row.strength.thinContracts) + Number(row.strength.singleLayout) + Number(row.strength.singleDtype)
    }));
    const itemTone = (item) => item.strength.thinContracts ? "func-tone-bad"
      : item.strength.singleLayout || item.strength.singleDtype ? "func-tone-near" : "func-tone-good";
    const renderOpDetail = (item) => {
      const tone = item.strength.thinContracts ? "func-tone-bad" : item.strength.singleLayout || item.strength.singleDtype ? "func-tone-near" : "func-tone-good";
      const reviewFlags = [item.strength.thinContracts && "below 10 test cases", item.strength.singleLayout && "one layout label", item.strength.singleDtype && "one dtype label"].filter(Boolean);
      explorer.detail.innerHTML = `<div class="function-detail-head"><div><div class="function-title">${escapeHtml(item.op)}</div><p class="function-subtitle">${escapeHtml(item.families.join(" + "))} Oracle operation</p></div><span class="function-ratio-pill ${tone}">${number(item.cases)} test cases</span></div>
        <div class="function-stat-strip"><div class="function-stat"><span>Corpus files</span><strong>${number(item.files.length)}</strong></div><div class="function-stat"><span>Layout labels</span><strong>${number(item.layouts.length)}</strong></div><div class="function-stat"><span>Dtypes</span><strong>${number(item.dtypes.length)}</strong></div><div class="function-stat"><span>Error test cases</span><strong>${number(item.errorCases)}</strong></div></div>
        <div><div class="function-block-title">Coverage dimensions</div><p class="function-subtitle">${number(item.parameterSignatures)} parameter signatures · ${escapeHtml(item.resultKinds.join(", "))} result kinds · ${escapeHtml(item.valueClasses.join(", ") || "unclassified values")} · review: ${escapeHtml(reviewFlags.join(", ") || "no count-based flag")}</p></div>
        <div class="function-table-scroll"><table class="function-table"><thead><tr><th>Corpus file</th><th>Dtypes</th></tr></thead><tbody>${item.files.map((file)=>`<tr><td><a href="https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Oracle/Fuzz/corpus/${escapeHtml(file)}">${escapeHtml(file)}</a></td><td>${escapeHtml(item.dtypes.join(", ") || "schema-specific")}</td></tr>`).join("")}</tbody></table></div>`;
    };
    const selectItem = (item) => {
      explorer.active = item.op;
      explorer.list.querySelectorAll(".function-list-item").forEach((button) => { const active = button.dataset.id === explorer.active; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); });
      renderOpDetail(item);
    };
    const refreshFilters = () => {
      const groups = [...new Set(opItems.map((item) => item.group))].sort();
      explorer.filter.innerHTML = `<option value="">All groups</option>${groups.map((group)=>`<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`).join("")}`;
    };
    const renderList = () => {
      const query = explorer.search.value.trim().toLowerCase(); const group = explorer.filter.value;
      let rows = opItems.filter((item) => (!query || `${item.name} ${item.group}`.toLowerCase().includes(query)) && (!group || item.group === group));
      const sort = explorer.sort.value;
      rows.sort(sort === "name" ? (a,b)=>a.name.localeCompare(b.name) : sort === "review" ? (a,b)=>b.review-a.review || a.size-b.size : (a,b)=>b.size-a.size || a.name.localeCompare(b.name));
      explorer.meta.textContent = `${number(rows.length)} Oracle operations`;
      explorer.list.innerHTML = rows.map((item) => `<button class="function-list-item ${itemTone(item)}" role="option" aria-selected="false" data-id="${escapeHtml(item.op)}"><span class="function-list-name">${escapeHtml(item.name)}</span><span class="function-list-score">${number(item.size)}</span><span class="function-list-detail">${escapeHtml(`${item.layouts.length} layout labels · ${item.dtypes.length} dtypes · ${item.errorCases} error test cases`)}</span></button>`).join("");
      explorer.list.querySelectorAll(".function-list-item").forEach((button, index) => button.addEventListener("click", () => selectItem(rows[index])));
      if (rows.length) selectItem(rows.find((item)=> item.op===explorer.active) || rows[0]); else explorer.detail.innerHTML = `<div class="function-empty">No Oracle evidence matches these filters.</div>`;
    };
    explorer.search.addEventListener("input", renderList); explorer.filter.addEventListener("change", renderList); explorer.sort.addEventListener("change", renderList);
    refreshFilters(); renderList();

    const thin = data.oracleOps.filter((op) => op.strength.thinContracts).slice(0, 8).map((op) => `<code>${escapeHtml(op.op)}</code>`).join(", ");
    root.querySelector("[data-priorities]").innerHTML = `
      <li><strong>Reconciliation:</strong> all ${number(oracle.corpusRows)} corpus test cases map to ${number(oracle.opKeys)} Oracle operations; ${number(oracle.hostPinRecords)} host-pin metadata records are excluded from test-case totals.</li>
      <li><strong>Thin operations:</strong> ${number(oracle.thinOpLabels)} operations have fewer than ten test cases; first queue: ${thin}.</li>
      <li><strong>Layout review:</strong> ${number(oracle.singleLayoutOpLabels)} operations record exactly one layout. Expand only APIs whose semantics can vary by memory layout.</li>
      <li><strong>Dtype review:</strong> ${number(oracle.singleDtypeOpLabels)} operations record exactly one dtype. Schema-specific zero-dtype operations are not falsely counted as gaps.</li>
      <li><strong>Exception evidence:</strong> ${number(oracle.opLabelsWithErrors)} operations contain ${number(oracle.errorRows)} error test cases. Absence elsewhere is neutral unless the API specifies invalid inputs.</li>
      <li><strong>Parameter covering arrays:</strong> combine axis/order/mode/casting/out/where branches pairwise.</li>
      <li><strong>Boundary values:</strong> force NaN, signed zero, subnormal, wrap seams, singular matrices and empty contraction axes.</li>
      <li><strong>Protocol traces:</strong> materialize iterators, state objects, flags and planners into canonical arrays/tuples/text.</li>
      <li><strong>Backend variation:</strong> replay managed/OpenBLAS and deduplicate only equal semantic outcomes.</li>
      <li><strong>Metamorphic amplification:</strong> add round trips, inverse pairs, decomposition reconstruction and set identities.</li>
      <li><strong>Soak + shrink:</strong> bias nightly generation toward unseen strength cells and commit minimized regressions.</li>`;
    if (window.tippy) tippy(root.querySelectorAll("[title]"), { content: (node) => node.getAttribute("title"), allowHTML: false, delay: [250, 0] });
  }).catch((error) => {
    root.querySelector("[data-evidence-detail]").innerHTML = `<div class="function-empty">${escapeHtml(error.message)} Regenerate with: python test/inventory/generate_test_inventory.py</div>`;
    console.error(error);
  });
})();
</script>
