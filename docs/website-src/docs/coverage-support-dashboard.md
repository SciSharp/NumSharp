# NumPy API Coverage & Support

<link rel="stylesheet" href="https://unpkg.com/tippy.js@6.3.7/dist/tippy.css">

<style>
.ns-coverage-dashboard {
  --cov-good: #178a5a;
  --cov-good-soft: #dff5ea;
  --cov-alias: #2878b8;
  --cov-alias-soft: #e2f1ff;
  --cov-partial: #b78318;
  --cov-partial-soft: #fff1cc;
  --cov-bad: #b4232c;
  --cov-bad-soft: #ffe1e4;
  --cov-ext: #7256b5;
  --cov-ext-soft: #eee8ff;
  --cov-quiet: var(--bs-secondary-color, #66737e);
  --cov-line: var(--bs-border-color);
  --cov-panel: var(--bs-body-bg);
  --cov-muted: rgba(108, 117, 125, 0.09);
  color: var(--bs-body-color);
}

.ns-coverage-dashboard * { box-sizing: border-box; }
.ns-coverage-dashboard button,
.ns-coverage-dashboard input,
.ns-coverage-dashboard select { font: inherit; }

.ns-coverage-dashboard .cov-intro {
  border-bottom: 1px solid var(--cov-line);
  padding: .2rem 0 1.25rem;
}

.ns-coverage-dashboard .cov-kicker {
  color: var(--cov-quiet);
  font-size: .78rem;
  font-weight: 750;
  letter-spacing: .09em;
  text-transform: uppercase;
}

.ns-coverage-dashboard .cov-title {
  font-size: clamp(2rem, 4vw, 3.35rem);
  letter-spacing: -.035em;
  line-height: 1.02;
  margin: .35rem 0 .65rem;
}

.ns-coverage-dashboard .cov-lede {
  color: var(--cov-quiet);
  font-size: 1.02rem;
  margin: 0;
  max-width: 79ch;
}

.ns-coverage-dashboard .cov-meta,
.ns-coverage-dashboard .cov-legend {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin-top: .95rem;
}

.ns-coverage-dashboard .cov-pill,
.ns-coverage-dashboard .cov-badge {
  align-items: center;
  border: 1px solid var(--cov-line);
  border-radius: 999px;
  display: inline-flex;
  font-size: .78rem;
  font-weight: 680;
  gap: .34rem;
  line-height: 1;
  padding: .34rem .62rem;
}

.ns-coverage-dashboard .cov-pill { background: var(--cov-muted); color: var(--cov-quiet); }
.ns-coverage-dashboard .cov-dot { border-radius: 50%; height: .48rem; width: .48rem; }
.ns-coverage-dashboard .is-available { --status-color: var(--cov-good); --status-soft: var(--cov-good-soft); }
.ns-coverage-dashboard .is-partial { --status-color: var(--cov-partial); --status-soft: var(--cov-partial-soft); }
.ns-coverage-dashboard .is-unsupported,
.ns-coverage-dashboard .is-missing { --status-color: var(--cov-bad); --status-soft: var(--cov-bad-soft); }
.ns-coverage-dashboard .is-extension { --status-color: var(--cov-ext); --status-soft: var(--cov-ext-soft); }
.ns-coverage-dashboard .is-exact { --status-color: var(--cov-good); --status-soft: var(--cov-good-soft); }
.ns-coverage-dashboard .is-alias { --status-color: var(--cov-alias); --status-soft: var(--cov-alias-soft); }
.ns-coverage-dashboard .cov-badge { background: var(--status-soft); border-color: color-mix(in srgb, var(--status-color), transparent 62%); color: var(--status-color); }
.ns-coverage-dashboard .cov-badge .cov-dot { background: var(--status-color); }

.ns-coverage-dashboard .cov-metrics {
  display: grid;
  gap: .75rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 1.25rem 0;
}

.ns-coverage-dashboard .cov-metric,
.ns-coverage-dashboard .cov-panel,
.ns-coverage-dashboard .cov-surface,
.ns-coverage-dashboard .cov-category {
  background: var(--cov-panel);
  border: 1px solid var(--cov-line);
  border-radius: .7rem;
}

.ns-coverage-dashboard .cov-metric {
  color: inherit;
  min-height: 7rem;
  padding: .88rem;
  text-align: left;
}

.ns-coverage-dashboard button.cov-metric { cursor: pointer; }
.ns-coverage-dashboard button.cov-metric:hover,
.ns-coverage-dashboard button.cov-metric:focus-visible,
.ns-coverage-dashboard .cov-surface:hover,
.ns-coverage-dashboard .cov-surface:focus-visible,
.ns-coverage-dashboard .cov-category:hover,
.ns-coverage-dashboard .cov-category:focus-visible {
  border-color: color-mix(in srgb, var(--cov-alias), transparent 25%);
  box-shadow: 0 5px 18px rgba(22, 41, 57, .08);
  outline: none;
  transform: translateY(-1px);
}

.ns-coverage-dashboard .cov-metric-label { color: var(--cov-quiet); font-size: .76rem; font-weight: 750; letter-spacing: .055em; text-transform: uppercase; }
.ns-coverage-dashboard .cov-metric-value { font-size: clamp(1.7rem, 3vw, 2.25rem); font-weight: 780; letter-spacing: -.04em; line-height: 1.1; margin: .35rem 0 .25rem; }
.ns-coverage-dashboard .cov-metric-note { color: var(--cov-quiet); font-size: .79rem; line-height: 1.35; }

.ns-coverage-dashboard .cov-section { margin: 1.65rem 0; }
.ns-coverage-dashboard .cov-section-head { align-items: end; display: flex; flex-wrap: wrap; gap: .65rem 1rem; justify-content: space-between; margin-bottom: .7rem; }
.ns-coverage-dashboard .cov-section h2 { font-size: 1.26rem; margin: 0; }
.ns-coverage-dashboard .cov-section-copy { color: var(--cov-quiet); font-size: .83rem; margin: .2rem 0 0; }
.ns-coverage-dashboard .cov-section-count { background: var(--cov-muted); border: 1px solid var(--cov-line); border-radius: 999px; color: var(--cov-quiet); flex: 0 0 auto; font-size: .72rem; font-weight: 720; padding: .34rem .58rem; }

.ns-coverage-dashboard .cov-status-track {
  background: var(--cov-muted);
  border-radius: .55rem;
  display: flex;
  height: 3rem;
  isolation: isolate;
  overflow: visible;
}

.ns-coverage-dashboard .cov-status-segment {
  align-items: center;
  border: 0;
  color: #fff;
  cursor: pointer;
  display: flex;
  font-size: .78rem;
  font-weight: 750;
  justify-content: center;
  min-width: 2px;
  overflow: hidden;
  padding: 0 .45rem;
  position: relative;
  transition: filter .16s ease, box-shadow .16s ease, transform .16s ease;
  white-space: nowrap;
}
.ns-coverage-dashboard .cov-status-segment:first-child { border-radius: .55rem 0 0 .55rem; }
.ns-coverage-dashboard .cov-status-segment:last-child { border-radius: 0 .55rem .55rem 0; }
.ns-coverage-dashboard .cov-status-segment[data-status="available"] { background: var(--cov-good); }
.ns-coverage-dashboard .cov-status-segment[data-status="partial"] { background: var(--cov-partial); }
.ns-coverage-dashboard .cov-status-segment[data-status="unsupported"],
.ns-coverage-dashboard .cov-status-segment[data-status="missing"] { background: var(--cov-bad); }
.ns-coverage-dashboard .cov-status-segment[data-status="extension"] { background: var(--cov-ext); }
.ns-coverage-dashboard .cov-status-segment:hover,
.ns-coverage-dashboard .cov-status-segment:focus-visible {
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.52), 0 8px 20px rgba(20, 27, 34, .24);
  filter: brightness(1.1) saturate(1.08);
  transform: translateY(-2px);
  z-index: 2;
}
.ns-coverage-dashboard .cov-status-segment:focus-visible { box-shadow: inset 0 0 0 3px #fff; outline: 2px solid var(--cov-alias); outline-offset: 2px; }

.ns-coverage-dashboard .cov-surface-grid { display: grid; gap: .65rem; grid-template-columns: repeat(auto-fit, minmax(min(12rem, 100%), 1fr)); }
.ns-coverage-dashboard .cov-surface,
.ns-coverage-dashboard .cov-category { color: inherit; cursor: pointer; min-width: 0; padding: .8rem; text-align: left; transition: .15s ease; }
.ns-coverage-dashboard .cov-surface-title,
.ns-coverage-dashboard .cov-category-title { align-items: baseline; display: flex; flex-wrap: wrap; font-weight: 740; gap: .25rem .4rem; justify-content: space-between; }
.ns-coverage-dashboard .cov-surface-title code { color: inherit; font-size: .86rem; overflow-wrap: anywhere; }
.ns-coverage-dashboard .cov-category-title {
  align-items: start;
  display: grid;
  font-weight: 650;
  gap: .4rem;
  grid-template-columns: minmax(0, 1fr) auto;
  min-height: 2.15rem;
}
.ns-coverage-dashboard .cov-category-title > span:first-child {
  display: -webkit-box;
  font-size: .86rem;
  line-height: 1.16;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}
.ns-coverage-dashboard .cov-category-title .cov-percent { font-size: .78rem; font-weight: 720; line-height: 1.16; }
.ns-coverage-dashboard .cov-percent { color: var(--cov-good); }
.ns-coverage-dashboard .cov-mini-track { background: var(--cov-muted); border-radius: 999px; display: flex; height: .46rem; margin: .62rem 0 .55rem; overflow: hidden; }
.ns-coverage-dashboard .cov-mini-fill { background: var(--cov-good); border-radius: inherit; height: 100%; }
.ns-coverage-dashboard .cov-mini-segment { height: 100%; min-width: 0; }
.ns-coverage-dashboard .cov-mini-segment.is-available { background: var(--cov-good); }
.ns-coverage-dashboard .cov-mini-segment.is-partial { background: var(--cov-partial); }
.ns-coverage-dashboard .cov-mini-segment.is-gap { background: var(--cov-bad); }
.ns-coverage-dashboard .cov-mini-segment.is-extension { background: var(--cov-ext); }
.ns-coverage-dashboard .cov-small { color: var(--cov-quiet); font-size: .75rem; overflow-wrap: anywhere; }
.ns-coverage-dashboard .cov-card-breakdown { display: grid; gap: .35rem; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: .58rem; }
.ns-coverage-dashboard .cov-card-stat { background: var(--cov-muted); border: 1px solid color-mix(in srgb, var(--status-color), transparent 68%); border-radius: .42rem; color: var(--status-color); min-width: 0; padding: .38rem .42rem; }
.ns-coverage-dashboard .cov-card-stat strong { display: block; font-size: .9rem; line-height: 1; }
.ns-coverage-dashboard .cov-card-stat span { color: var(--cov-quiet); display: block; font-size: .64rem; margin-top: .18rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ns-coverage-dashboard .cov-card-foot { margin-top: .5rem; }
.ns-coverage-dashboard .cov-compact-breakdown { color: var(--cov-quiet); display: flex; flex-wrap: wrap; font-size: .72rem; gap: .22rem .42rem; line-height: 1.35; }
.ns-coverage-dashboard .cov-compact-breakdown span { align-items: center; display: inline-flex; gap: .24rem; white-space: nowrap; }
.ns-coverage-dashboard .cov-compact-breakdown span::before { background: var(--status-color); border-radius: 50%; content: ""; height: .4rem; width: .4rem; }

.ns-coverage-dashboard .cov-category-grid { display: grid; gap: .55rem; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.ns-coverage-dashboard .cov-surface-grid[data-scrollable="true"],
.ns-coverage-dashboard .cov-category-grid[data-scrollable="true"] { max-height: 32rem; overflow-y: auto; overscroll-behavior: contain; padding: .1rem .35rem .4rem .1rem; scrollbar-gutter: stable; }
.ns-coverage-dashboard .cov-surface { min-height: 8.25rem; }
.ns-coverage-dashboard .cov-category { min-height: 5.8rem; }

.ns-coverage-dashboard .cov-definition {
  align-items: start;
  background: var(--cov-muted);
  border-left: 4px solid var(--cov-alias);
  border-radius: .2rem .6rem .6rem .2rem;
  display: grid;
  gap: .35rem;
  grid-template-columns: auto 1fr;
  margin: 1.25rem 0;
  padding: .8rem .9rem;
}
.ns-coverage-dashboard .cov-definition strong { color: var(--cov-alias); }
.ns-coverage-dashboard .cov-definition p { font-size: .84rem; margin: 0; }

.ns-coverage-dashboard .cov-explorer { overflow: hidden; }
.ns-coverage-dashboard .cov-toolbar {
  align-items: end;
  border-bottom: 1px solid var(--cov-line);
  display: flex;
  flex-wrap: wrap;
  gap: .55rem;
  padding: .75rem;
}
.ns-coverage-dashboard .cov-control { display: grid; flex: 1 1 8.25rem; gap: .24rem; min-width: 0; }
.ns-coverage-dashboard .cov-control:first-child { flex: 2 1 16rem; }
.ns-coverage-dashboard .cov-scope-controls { align-items: end; display: flex; flex-wrap: wrap; gap: .75rem 1.1rem; margin: 1.15rem 0; }
.ns-coverage-dashboard .cov-scope-controls .cov-control { flex: 0 1 20rem; }
.ns-coverage-dashboard .cov-scope-note { color: var(--cov-quiet); flex: 1 1 20rem; font-size: .84rem; margin: 0 0 .25rem; }
.ns-coverage-dashboard .cov-control label { color: var(--cov-quiet); font-size: .7rem; font-weight: 750; letter-spacing: .05em; text-transform: uppercase; }
.ns-coverage-dashboard .cov-control input,
.ns-coverage-dashboard .cov-control select {
  background: var(--cov-panel);
  border: 1px solid var(--cov-line);
  border-radius: .42rem;
  color: inherit;
  min-height: 2.25rem;
  padding: .4rem .55rem;
  width: 100%;
}
.ns-coverage-dashboard .cov-control input:focus,
.ns-coverage-dashboard .cov-control select:focus { border-color: var(--cov-alias); box-shadow: 0 0 0 3px color-mix(in srgb, var(--cov-alias), transparent 78%); outline: none; }
.ns-coverage-dashboard .cov-reset { align-self: end; background: var(--cov-muted); border: 1px solid var(--cov-line); border-radius: .42rem; color: inherit; cursor: pointer; flex: 0 0 auto; min-height: 2.25rem; padding: .4rem .72rem; }

.ns-coverage-dashboard .cov-explorer-meta { align-items: center; border-bottom: 1px solid var(--cov-line); display: flex; gap: .6rem; justify-content: space-between; min-height: 2.8rem; padding: .55rem .8rem; }
.ns-coverage-dashboard .cov-result-count { color: var(--cov-quiet); font-size: .82rem; }
.ns-coverage-dashboard .cov-quick-gap { background: transparent; border: 0; color: var(--cov-alias); cursor: pointer; font-size: .8rem; font-weight: 700; }

.ns-coverage-dashboard .cov-explorer-body { display: grid; grid-template-columns: minmax(17rem, .82fr) minmax(24rem, 1.35fr); min-height: 34rem; }
.ns-coverage-dashboard .cov-results { border-right: 1px solid var(--cov-line); max-height: 43rem; overflow: auto; overscroll-behavior: contain; }
.ns-coverage-dashboard .cov-result {
  background: transparent;
  border: 0;
  border-bottom: 1px solid var(--cov-line);
  color: inherit;
  cursor: pointer;
  display: block;
  padding: .72rem .82rem;
  text-align: left;
  width: 100%;
}
.ns-coverage-dashboard .cov-result:hover { background: var(--cov-muted); }
.ns-coverage-dashboard .cov-result[aria-current="true"] { background: color-mix(in srgb, var(--cov-alias), transparent 90%); box-shadow: inset 3px 0 var(--cov-alias); }
.ns-coverage-dashboard .cov-result:focus-visible { outline: 2px solid var(--cov-alias); outline-offset: -2px; }
.ns-coverage-dashboard .cov-result-main { align-items: center; display: flex; gap: .45rem; justify-content: space-between; }
.ns-coverage-dashboard .cov-result-name { color: var(--bs-link-color); font-family: var(--bs-font-monospace); font-size: .83rem; font-weight: 690; overflow: hidden; text-decoration: none; text-overflow: ellipsis; white-space: nowrap; }
.ns-coverage-dashboard a.cov-result-name:hover,
.ns-coverage-dashboard a.cov-result-name:focus-visible,
.ns-coverage-dashboard .cov-detail-api-link:hover,
.ns-coverage-dashboard .cov-detail-api-link:focus-visible { text-decoration: underline; }
.ns-coverage-dashboard .cov-result-sub { color: var(--cov-quiet); display: flex; font-size: .71rem; gap: .45rem; margin-top: .25rem; }
.ns-coverage-dashboard .cov-result-status { background: var(--status-soft); border-radius: 999px; color: var(--status-color); flex: 0 0 auto; font-size: .67rem; font-weight: 750; padding: .22rem .42rem; text-transform: capitalize; }
.ns-coverage-dashboard .cov-more { background: var(--cov-muted); border: 0; color: var(--cov-alias); cursor: pointer; font-weight: 700; padding: .7rem; width: 100%; }
.ns-coverage-dashboard .cov-empty { color: var(--cov-quiet); padding: 2rem 1rem; text-align: center; }

.ns-coverage-dashboard .cov-detail { min-width: 0; padding: 1.05rem 1.15rem 1.4rem; }
.ns-coverage-dashboard .cov-detail-head { align-items: start; display: flex; gap: .7rem; justify-content: space-between; }
.ns-coverage-dashboard .cov-detail h3 { font-family: var(--bs-font-monospace); font-size: 1.28rem; margin: 0; overflow-wrap: anywhere; }
.ns-coverage-dashboard .cov-detail-api-link { color: var(--bs-link-color); text-decoration: none; }
.ns-coverage-dashboard .cov-detail-grid { display: grid; gap: .7rem; grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 1rem 0; }
.ns-coverage-dashboard .cov-detail-fact { background: var(--cov-muted); border-radius: .5rem; padding: .65rem; }
.ns-coverage-dashboard .cov-detail-label { color: var(--cov-quiet); font-size: .68rem; font-weight: 750; letter-spacing: .05em; margin-bottom: .25rem; text-transform: uppercase; }
.ns-coverage-dashboard .cov-detail-value { font-size: .85rem; font-weight: 680; }
.ns-coverage-dashboard .cov-code-block { background: #17212b; border-radius: .5rem; color: #e7edf3; font-family: var(--bs-font-monospace); font-size: .78rem; line-height: 1.45; margin: .35rem 0 .9rem; overflow: auto; padding: .72rem; white-space: pre-wrap; word-break: break-word; }
.ns-coverage-dashboard .cov-detail h4 { font-size: .8rem; letter-spacing: .045em; margin: 1rem 0 .3rem; text-transform: uppercase; }
.ns-coverage-dashboard .cov-notes { border-left: 3px solid var(--cov-partial); color: var(--cov-quiet); font-size: .84rem; margin-top: 1rem; padding: .2rem 0 .2rem .72rem; }
.ns-coverage-dashboard .cov-doc-link { display: inline-flex; font-size: .82rem; font-weight: 700; margin-top: .55rem; }
.ns-coverage-dashboard .cov-source-links { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .35rem; }
.ns-coverage-dashboard .cov-source-link { background: var(--cov-muted); border: 1px solid var(--cov-line); border-radius: .38rem; font-family: var(--bs-font-monospace); font-size: .75rem; padding: .35rem .5rem; text-decoration: none; }
.ns-coverage-dashboard .cov-source-link:hover,
.ns-coverage-dashboard .cov-source-link:focus-visible { border-color: var(--cov-alias); text-decoration: underline; }

.ns-coverage-dashboard .cov-loading,
.ns-coverage-dashboard .cov-error { border: 1px dashed var(--cov-line); border-radius: .6rem; color: var(--cov-quiet); margin: 1rem 0; padding: 1.2rem; }
.ns-coverage-dashboard .cov-error { background: var(--cov-bad-soft); color: var(--cov-bad); }
.ns-coverage-dashboard [hidden] { display: none !important; }

.tippy-box[data-theme~="ns-coverage"] { background: linear-gradient(145deg, #202832, #151b22); border: 1px solid #3b424a; border-radius: .72rem; box-shadow: 0 20px 52px rgba(0,0,0,.42), 0 5px 16px rgba(0,0,0,.3); color: #f4f7fa; font-size: .78rem; max-width: min(34rem, calc(100vw - 2rem)) !important; }
.tippy-box[data-theme~="ns-coverage"] .tippy-content { padding: 0; }
.tippy-box[data-theme~="ns-coverage"] .tippy-arrow { color: #17212b; }
.ns-cov-tip { min-width: min(27rem, calc(100vw - 2.5rem)); }
.ns-cov-tip-head { border-bottom: 1px solid rgba(255,255,255,.14); padding: .7rem .78rem .58rem; }
.ns-cov-tip-head strong { display: block; font-size: .88rem; }
.ns-cov-tip-head span { color: #aebbc6; }
.ns-cov-tip-tabs { display: flex; gap: .3rem; padding: .45rem .65rem 0; }
.ns-cov-tip-tab { background: transparent; border: 1px solid rgba(255,255,255,.18); border-radius: 999px; color: #d8e1e8; cursor: pointer; font-size: .7rem; padding: .25rem .55rem; }
.ns-cov-tip-tab[aria-selected="true"] { background: #fff; color: #17212b; }
.ns-cov-tip-panel { max-height: 18rem; overflow: auto; overscroll-behavior: contain; padding: .45rem .65rem .65rem; }
.ns-cov-tip-row { align-items: center; border-bottom: 1px solid rgba(255,255,255,.09); display: flex; gap: .6rem; justify-content: space-between; padding: .36rem .1rem; }
.ns-cov-tip-row:last-child { border-bottom: 0; }
.ns-cov-tip-row code { color: #fff; font-size: .72rem; }
.ns-cov-tip-row a { text-decoration-color: rgba(255,255,255,.4); text-underline-offset: .14rem; }
.ns-cov-tip-row a:hover code,
.ns-cov-tip-row a:focus-visible code { color: #8fc8ff; }
.ns-cov-tip-row span { color: #aebbc6; flex: 0 0 auto; font-size: .68rem; text-transform: capitalize; }

@media (max-width: 1150px) {
  .ns-coverage-dashboard .cov-toolbar .cov-control { flex-basis: 9rem; }
  .ns-coverage-dashboard .cov-toolbar .cov-control:first-child { flex-basis: 17rem; }
}

@media (max-width: 820px) {
  .ns-coverage-dashboard .cov-metrics,
  .ns-coverage-dashboard .cov-category-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ns-coverage-dashboard .cov-explorer-body { grid-template-columns: 1fr; }
  .ns-coverage-dashboard .cov-results { border-bottom: 1px solid var(--cov-line); border-right: 0; max-height: 25rem; }
}

@media (max-width: 560px) {
  .ns-coverage-dashboard .cov-metrics,
  .ns-coverage-dashboard .cov-category-grid,
  .ns-coverage-dashboard .cov-detail-grid { grid-template-columns: 1fr; }
  .ns-coverage-dashboard .cov-toolbar .cov-control,
  .ns-coverage-dashboard .cov-toolbar .cov-control:first-child { flex-basis: 100%; }
  .ns-coverage-dashboard .cov-reset { flex: 1 1 100%; }
  .ns-coverage-dashboard .cov-definition { grid-template-columns: 1fr; }
  .ns-coverage-dashboard .cov-status-track { height: 2.55rem; }
  .ns-coverage-dashboard .cov-status-segment span { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .ns-coverage-dashboard * { scroll-behavior: auto !important; transition: none !important; }
}
</style>

<div class="ns-coverage-dashboard" id="ns-coverage-dashboard">
  <section class="cov-intro" aria-labelledby="cov-dashboard-title">
    <div class="cov-kicker">NumPy 2.x parity · compiled API inventory</div>
    <h2 class="cov-title" id="cov-dashboard-title">See the supported surface. Find the next gap.</h2>
    <p class="cov-lede">Explore the NumPy catalog across functions, modules, and object members, with NumSharp equivalents, known limitations, and C# overloads. The page is generated from the same artifact published by CI.</p>
    <div class="cov-meta" id="cov-meta" aria-label="Artifact metadata"></div>
  </section>

  <div class="cov-loading" id="cov-loading" role="status">Loading the NumPy ↔ NumSharp coverage artifact…</div>
  <div class="cov-error" id="cov-error" role="alert" hidden></div>

  <div id="cov-content" hidden>
    <div class="cov-scope-controls">
      <div class="cov-control"><label for="cov-scope">Dashboard scope</label><select id="cov-scope" aria-describedby="cov-scope-note"><option value="numpy">All NumPy APIs</option><option value="default">Headline comparison</option><option value="extended">Extended NumPy APIs</option><option value="catalog">Full NumPy catalog</option><option value="extensions">NumSharp-only APIs</option><option value="all">Everything</option></select></div>
      <p class="cov-scope-note" id="cov-scope-note" aria-live="polite"></p>
    </div>
    <section class="cov-metrics" id="cov-metrics" aria-label="Coverage summary"></section>
    <div class="cov-definition">
      <strong>Coverage math</strong>
      <p><span id="cov-headline-reference"></span> Summaries, cards, and the explorer follow the selected scope. Availability is <code>available ÷ NumPy APIs in that scope</code>; supporting types, constants, modules, and NumSharp-only APIs do not enter this percentage. “Available” means an exact public member or a reviewed alias exists and is not marked partial/unsupported. The headline comparison also excludes extended APIs. API presence is not a blanket edge-case, dtype, layout, or signature parity claim. <span id="cov-applicability-note"></span></p>
    </div>
    <section class="cov-section" aria-labelledby="cov-status-heading">
      <div class="cov-section-head">
        <div><h2 id="cov-status-heading">Support mix</h2><p class="cov-section-copy">Select a segment to send that status into the explorer. Hover, focus, or click for its API breakdown.</p></div>
      </div>
      <div class="cov-status-track" id="cov-status-track"></div>
      <div class="cov-legend" id="cov-legend"></div>
    </section>
    <section class="cov-section" aria-labelledby="cov-surfaces-heading">
      <div class="cov-section-head">
        <div><h2 id="cov-surfaces-heading">Surface scoreboard</h2><p class="cov-section-copy">Every module and object surface in the selected scope. Select a card to explore its APIs.</p></div>
        <span class="cov-section-count" id="cov-surface-count"></span>
      </div>
      <div class="cov-surface-grid" id="cov-surface-grid"></div>
    </section>
    <section class="cov-section" aria-labelledby="cov-categories-heading">
      <div class="cov-section-head">
        <div><h2 id="cov-categories-heading">Capability map</h2><p class="cov-section-copy">Categories are assigned deterministically by the artifact generator. Select one to investigate it.</p></div>
        <span class="cov-section-count" id="cov-category-count"></span>
      </div>
      <div class="cov-category-grid" id="cov-category-grid"></div>
    </section>
    <section class="cov-section" aria-labelledby="cov-explorer-heading">
      <div class="cov-section-head">
        <div><h2 id="cov-explorer-heading">API explorer</h2><p class="cov-section-copy">Search NumPy names, NumSharp targets, signatures, categories, and review notes.</p></div>
      </div>
      <div class="cov-panel cov-explorer">
        <div class="cov-toolbar">
          <div class="cov-control"><label for="cov-search">Search</label><input id="cov-search" type="search" placeholder="np.add, ndarray.astype, FFT…" autocomplete="off"></div>
          <div class="cov-control"><label for="cov-surface">Surface</label><select id="cov-surface"><option value="all">All surfaces</option></select></div>
          <div class="cov-control"><label for="cov-category">Category</label><select id="cov-category"><option value="all">All categories</option></select></div>
          <div class="cov-control"><label for="cov-status">Status</label><select id="cov-status"><option value="all">All statuses</option></select></div>
          <div class="cov-control"><label for="cov-kind">Kind</label><select id="cov-kind"><option value="all">All kinds</option></select></div>
          <div class="cov-control"><label for="cov-mapping">Mapping</label><select id="cov-mapping"><option value="all">All mappings</option></select></div>
          <div class="cov-control"><label for="cov-sort">Sort</label><select id="cov-sort"><option value="gap">Gaps first</option><option value="name">API name</option><option value="surface">Surface</option><option value="coverage">Available first</option></select></div>
          <button class="cov-reset" id="cov-reset" type="button">Clear filters</button>
        </div>
        <div class="cov-explorer-meta"><span class="cov-result-count" id="cov-result-count" aria-live="polite"></span><button class="cov-quick-gap" id="cov-quick-gap" type="button">Show open gaps</button></div>
        <div class="cov-explorer-body">
          <div class="cov-results" id="cov-results" role="list" aria-label="Coverage APIs"></div>
          <article class="cov-detail" id="cov-detail" aria-live="polite"></article>
        </div>
      </div>
    </section>
  </div>
</div>

<script src="https://unpkg.com/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://unpkg.com/tippy.js@6.3.7/dist/tippy.umd.min.js"></script>
<script>
(() => {
  const root = document.getElementById("ns-coverage-dashboard");
  if (!root) return;

  const byId = (id) => root.querySelector(`#${id}`);
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const surfaceLabels = { np: "np.*", ndarray: "ndarray.*", random: "np.random.*", linalg: "np.linalg.*", fft: "np.fft.*", ufunc: "np.ufunc.*", ma: "np.ma.*", polynomial: "np.polynomial.*" };
  const scopeLabels = { numpy: "All NumPy APIs", default: "Headline comparison", extended: "Extended NumPy APIs", catalog: "Full NumPy catalog", extensions: "NumSharp-only APIs", all: "Everything" };
  const statusLabels = { available: "Available", partial: "Partial", unsupported: "Unsupported", missing: "Missing", extension: "NumSharp-only", gaps: "Open gaps (missing + unsupported)" };
  const statusOrder = { unsupported: 0, partial: 1, missing: 2, available: 3, extension: 4 };
  const mappingLabels = { exact: "Exact public name", alias: "Reviewed alias", missing: "No public mapping", extension: "NumSharp-only" };
  const dispositionLabels = { object: "Object members", type: "Type", constant: "Constant", module: "Module", extension: "NumSharp-only API", extended: "Extended catalog" };
  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const percent = (part, whole) => whole ? (part * 100 / whole) : 0;
  const apiLabel = (row) => row.id.startsWith("numpy.ndarray.")
    ? row.id.replace("numpy.ndarray.", "ndarray.")
    : row.id.startsWith("numpy.") ? row.id.replace("numpy.", "np.") : row.id;

  const state = { rows: [], filtered: [], selectedId: null, visible: 120, tooltipInstances: [], activeTooltip: null };

  function surfaceLabel(surface) {
    if (surfaceLabels[surface]) return surfaceLabels[surface];
    const sample = state.rows.find((row) => row.surface === surface && row.origin === "numpy");
    const prefix = sample?.id?.endsWith(`.${sample.name}`) ? sample.id.slice(0, -sample.name.length - 1) : "";
    const namespace = prefix || (surface.startsWith("numpy.") ? surface : `numpy.${surface}`);
    return surfaceLabels[surface] = `${namespace.replace(/^numpy\.ndarray(?=\.|$)/, "ndarray").replace(/^numpy\./, "np.")}.*`;
  }

  function scopedRows() {
    return state.rows.filter((row) => rowInScope(row, byId("cov-scope").value));
  }

  function isComparableApi(row) {
    return row.origin === "numpy" && (row.in_default_scope || (row.extended && !["class", "constant", "module"].includes(row.kind)));
  }

  function summarize(rows) {
    const counts = { total: rows.length, numpy: rows.filter(isComparableApi).length, availableApis: 0, available: 0, partial: 0, unsupported: 0, missing: 0, extension: 0, exact: 0, alias: 0 };
    rows.forEach((row) => {
      if (counts[row.status] !== undefined) counts[row.status] += 1;
      if (isComparableApi(row) && row.status === "available") counts.availableApis += 1;
      if (row.availability === "exact" || row.availability === "alias") counts[row.availability] += 1;
    });
    counts.coverage = percent(counts.availableApis, counts.numpy);
    return counts;
  }

  function defaultRows() {
    return state.rows.filter((row) => row.origin === "numpy" && row.in_default_scope);
  }

  function coverageLabel(counts) {
    return counts.numpy ? `${counts.coverage.toFixed(1)}%` : "—";
  }

  function statusBadge(status, text = statusLabels[status] || status) {
    return `<span class="cov-badge is-${escapeHtml(status)}"><span class="cov-dot"></span>${escapeHtml(text)}</span>`;
  }

  function metricCard(label, value, note, action = "") {
    const tag = action ? "button" : "div";
    const attrs = action ? ` type="button" data-metric-action="${escapeHtml(action)}"` : "";
    return `<${tag} class="cov-metric"${attrs}><div class="cov-metric-label">${escapeHtml(label)}</div><div class="cov-metric-value">${escapeHtml(value)}</div><div class="cov-metric-note">${escapeHtml(note)}</div></${tag}>`;
  }

  function miniSupportTrack(counts) {
    const gaps = counts.missing + counts.unsupported;
    const segments = [
      ["available", counts.available],
      ["partial", counts.partial],
      ["gap", gaps],
      ["extension", counts.extension]
    ];
    const label = `${counts.available} available, ${counts.partial} partial, ${gaps} gaps, ${counts.extension} NumSharp-only`;
    return `<div class="cov-mini-track" role="img" aria-label="${escapeHtml(label)}">${segments.filter(([, count]) => count > 0).map(([status, count]) => `<span class="cov-mini-segment is-${status}" style="width:${percent(count, counts.total)}%"></span>`).join("")}</div>`;
  }

  function cardBreakdown(counts) {
    const gaps = counts.missing + counts.unsupported;
    return `<div class="cov-card-breakdown">
      <div class="cov-card-stat is-available"><strong>${counts.available}</strong><span>available</span></div>
      <div class="cov-card-stat is-partial"><strong>${counts.partial}</strong><span>partial</span></div>
      <div class="cov-card-stat is-missing"><strong>${gaps}</strong><span>gaps</span></div>
    </div><div class="cov-small cov-card-foot">${counts.exact} direct names · ${counts.alias} aliases · ${counts.total} total${counts.extension ? ` · ${counts.extension} NumSharp-only` : ""}</div>`;
  }

  function compactCardBreakdown(counts) {
    const gaps = counts.missing + counts.unsupported;
    return `<div class="cov-compact-breakdown">
      <span class="is-available">${counts.available} available</span>
      <span class="is-partial">${counts.partial} partial</span>
      <span class="is-missing">${gaps} gaps</span>
      ${counts.extension ? `<span class="is-extension">${counts.extension} NumSharp-only</span>` : ""}
    </div>`;
  }

  function initializeMetadata(data) {
    const stats = summarize(defaultRows());
    const published = data.summary.default_scope;
    if (stats.total !== number(published.total) || stats.available !== number(published.available)) {
      throw new Error("Coverage summary does not match its row inventory.");
    }

    byId("cov-meta").innerHTML = [
      `NumPy ${data.numpy_version}`,
      `NumSharp assembly ${data.numsharp_assembly_version}`,
      `${data.summary.catalog_rows.toLocaleString()} searchable rows`,
      `schema v${data.schema_version}`
    ].map((item) => `<span class="cov-pill">${escapeHtml(item)}</span>`).join("");

    byId("cov-headline-reference").textContent = `Headline comparison: ${coverageLabel(stats)} (${stats.available}/${stats.total} APIs).`;
  }

  function renderSummary() {
    const rows = scopedRows();
    const stats = summarize(rows);
    const scope = byId("cov-scope").value;
    const surfaces = [...new Set(rows.map((row) => row.surface))];
    const categories = [...new Set(rows.map((row) => row.category))].sort((a, b) => a.localeCompare(b));
    const rejectedContracts = rows.filter((row) => row.applicability === "not_applicable").length;
    byId("cov-scope-note").textContent = `${scopeLabels[scope]}: ${stats.total.toLocaleString()} entries across ${surfaces.length} surfaces and ${categories.length} capability areas. This selection updates every summary and the explorer.`;
    byId("cov-applicability-note").textContent = rejectedContracts ? `This scope includes ${rejectedContracts.toLocaleString()} declared ufunc contract${rejectedContracts === 1 ? "" : "s"} whose calls NumPy rejects; gaps describe public API presence, not a count of computable operations.` : "";

    byId("cov-metrics").innerHTML = [
      metricCard("Scoped API availability", coverageLabel(stats), stats.numpy ? `${stats.availableApis} of ${stats.numpy} comparable NumPy APIs` : "No comparable NumPy APIs in this scope"),
      metricCard("Exact names", stats.exact.toLocaleString(), `${stats.alias} reviewed aliases bridge C#/NumPy naming`, stats.exact ? "exact" : ""),
      metricCard("Open gaps", (stats.missing + stats.unsupported).toLocaleString(), `${stats.partial} additional API${stats.partial === 1 ? " is" : "s are"} partial`, "gaps"),
      metricCard("Entries in scope", stats.total.toLocaleString(), `${surfaces.length} public surfaces · ${categories.length} capability areas`)
    ].join("");

    const statuses = ["available", "partial", "unsupported", "missing", "extension"];
    byId("cov-status-track").innerHTML = statuses.filter((status) => stats[status] > 0).map((status) => {
      const width = percent(stats[status], stats.total);
      const text = width >= 8 ? `<span>${statusLabels[status]} · ${stats[status]}</span>` : `<span>${stats[status]}</span>`;
      return `<button class="cov-status-segment" type="button" data-status="${status}" data-tooltip-group="status:${status}" style="width:${width}%" aria-label="Filter ${statusLabels[status]}: ${stats[status]} APIs">${text}</button>`;
    }).join("");
    byId("cov-legend").innerHTML = statuses.filter((status) => status !== "extension" || stats.extension).map((status) => statusBadge(status, `${statusLabels[status]} ${stats[status]}`)).join("");

    const priority = ["np", "ndarray", "random", "linalg", "fft"];
    const surfaceOrder = surfaces.sort((a, b) => {
      const ai = priority.includes(a) ? priority.indexOf(a) : priority.length;
      const bi = priority.includes(b) ? priority.indexOf(b) : priority.length;
      return ai - bi || surfaceLabel(a).localeCompare(surfaceLabel(b));
    });
    byId("cov-surface-count").textContent = `${surfaceOrder.length} public surfaces${surfaceOrder.length > 12 ? " · scroll to explore" : ""}`;
    byId("cov-surface-grid").dataset.scrollable = String(surfaceOrder.length > 12);
    byId("cov-surface-grid").innerHTML = surfaceOrder.map((surface) => {
      const subset = rows.filter((row) => row.surface === surface);
      const counts = summarize(subset);
      return `<button class="cov-surface" type="button" data-surface="${escapeHtml(surface)}" data-tooltip-group="surface:${escapeHtml(surface)}">
        <div class="cov-surface-title"><code>${escapeHtml(surfaceLabel(surface))}</code><span class="cov-percent">${coverageLabel(counts)}</span></div>
        ${miniSupportTrack(counts)}
        ${cardBreakdown(counts)}
      </button>`;
    }).join("");

    byId("cov-category-count").textContent = `${categories.length} capability areas${categories.length > 12 ? " · scroll to explore" : ""}`;
    byId("cov-category-grid").dataset.scrollable = String(categories.length > 12);
    byId("cov-category-grid").innerHTML = categories.map((category) => {
      const subset = rows.filter((row) => row.category === category);
      const counts = summarize(subset);
      return `<button class="cov-category" type="button" data-category="${escapeHtml(category)}" data-tooltip-group="category:${escapeHtml(category)}">
        <div class="cov-category-title"><span>${escapeHtml(category)}</span><span class="cov-percent">${coverageLabel(counts)}</span></div>
        ${miniSupportTrack(counts)}
        ${compactCardBreakdown(counts)}
      </button>`;
    }).join("");

    initializeTooltips();
  }

  function populateSelect(id, values, labels = {}) {
    const select = byId(id);
    const first = select.options[0];
    select.replaceChildren(first);
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = typeof labels === "function" ? labels(value) : labels[value] || value;
      select.appendChild(option);
    });
  }

  function initializeFilters() {
    const rows = scopedRows();
    populateSelect("cov-surface", [...new Set(rows.map((row) => row.surface))].sort(), surfaceLabel);
    populateSelect("cov-category", [...new Set(rows.map((row) => row.category))].sort());
    const statuses = [...new Set(rows.map((row) => row.status))].sort((a, b) => statusOrder[a] - statusOrder[b]);
    statuses.unshift("gaps");
    populateSelect("cov-status", statuses, statusLabels);
    populateSelect("cov-kind", [...new Set(rows.map((row) => row.kind))].sort());
    populateSelect("cov-mapping", [...new Set(rows.map((row) => row.availability))].sort(), mappingLabels);
  }

  function currentFilters() {
    return {
      search: byId("cov-search").value.trim().toLowerCase(),
      scope: byId("cov-scope").value,
      surface: byId("cov-surface").value,
      category: byId("cov-category").value,
      status: byId("cov-status").value,
      kind: byId("cov-kind").value,
      mapping: byId("cov-mapping").value,
      sort: byId("cov-sort").value,
    };
  }

  function rowInScope(row, scope) {
    if (scope === "default") return row.origin === "numpy" && row.in_default_scope;
    if (scope === "numpy") return isComparableApi(row);
    if (scope === "extended") return isComparableApi(row) && row.extended;
    if (scope === "catalog") return row.origin === "numpy";
    if (scope === "extensions") return row.origin === "numsharp";
    return true;
  }

  function filterRows() {
    const filters = currentFilters();
    const terms = filters.search.split(/\s+/).filter(Boolean);
    const results = state.rows.filter((row) => {
      if (!rowInScope(row, filters.scope)) return false;
      if (filters.surface !== "all" && row.surface !== filters.surface) return false;
      if (filters.category !== "all" && row.category !== filters.category) return false;
      if (filters.status === "gaps" && row.status !== "missing" && row.status !== "unsupported") return false;
      if (filters.status !== "all" && filters.status !== "gaps" && row.status !== filters.status) return false;
      if (filters.kind !== "all" && row.kind !== filters.kind) return false;
      if (filters.mapping !== "all" && row.availability !== filters.mapping) return false;
      if (!terms.length) return true;
      const haystack = [row.id, apiLabel(row), row.name, row.surface, surfaceLabel(row.surface), row.category, row.kind, row.disposition, row.status, row.availability, row.numsharp_target, row.numpy_signature, row.notes, ...(row.numsharp_signatures || [])].join(" ").toLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
    results.sort((a, b) => {
      if (filters.sort === "gap") return (statusOrder[a.status] - statusOrder[b.status]) || apiLabel(a).localeCompare(apiLabel(b));
      if (filters.sort === "coverage") return (statusOrder[b.status] - statusOrder[a.status]) || apiLabel(a).localeCompare(apiLabel(b));
      if (filters.sort === "surface") return a.surface.localeCompare(b.surface) || apiLabel(a).localeCompare(apiLabel(b));
      return apiLabel(a).localeCompare(apiLabel(b));
    });
    state.filtered = results;
    state.visible = 120;
    if (!results.some((row) => row.id === state.selectedId)) state.selectedId = results[0]?.id || null;
    renderResults();
    renderDetail();
  }

  function renderResults() {
    const target = byId("cov-results");
    const visible = state.filtered.slice(0, state.visible);
    byId("cov-result-count").textContent = `${state.filtered.length.toLocaleString()} of ${scopedRows().length.toLocaleString()} entries · ${scopeLabels[byId("cov-scope").value]}`;
    if (!visible.length) {
      target.innerHTML = `<div class="cov-empty">No API matches these filters.</div>`;
      return;
    }
    target.innerHTML = visible.map((row) => {
      const name = escapeHtml(apiLabel(row));
      const primaryUrl = row.documentation_url || row.numsharp_source_urls?.[0] || "";
      const linkTitle = row.documentation_url ? `Open latest NumPy documentation for ${name}` : `Open NumSharp source for ${name} on GitHub`;
      const nameMarkup = primaryUrl
        ? `<a class="cov-result-name" data-api-link href="${escapeHtml(primaryUrl)}" target="_blank" rel="noopener" title="${linkTitle}">${name}</a>`
        : `<span class="cov-result-name">${name}</span>`;
      return `<div class="cov-result" role="listitem" tabindex="0" data-row-id="${escapeHtml(row.id)}" aria-current="${row.id === state.selectedId}">
      <div class="cov-result-main">${nameMarkup}<span class="cov-result-status is-${escapeHtml(row.status)}">${escapeHtml(statusLabels[row.status] || row.status)}</span></div>
      <div class="cov-result-sub"><span>${escapeHtml(surfaceLabel(row.surface))}</span><span>·</span><span>${escapeHtml(row.category)}</span><span>·</span><span>${escapeHtml(row.kind)}</span></div>
    </div>`;
    }).join("") + (state.filtered.length > visible.length ? `<button class="cov-more" id="cov-more" type="button">Show ${Math.min(120, state.filtered.length - visible.length)} more</button>` : "");
  }

  function renderDetail() {
    const target = byId("cov-detail");
    const row = state.rows.find((item) => item.id === state.selectedId);
    if (!row) {
      target.innerHTML = `<div class="cov-empty">Choose an API to inspect its mapping.</div>`;
      return;
    }
    const signatures = row.numsharp_signatures?.length
      ? row.numsharp_signatures.map((signature) => `<div class="cov-code-block">${escapeHtml(signature)}</div>`).join("")
      : `<div class="cov-code-block">No NumSharp public member mapped</div>`;
    const mapping = row.availability === "exact" ? "Exact public name" : row.availability === "alias" ? "Reviewed alternate name or surface" : row.availability === "extension" ? "NumSharp-only public API" : "No public mapping found";
    const detailName = escapeHtml(apiLabel(row));
    const detailUrl = row.documentation_url || row.numsharp_source_urls?.[0] || "";
    const detailTitle = row.documentation_url ? `Open latest NumPy documentation for ${detailName}` : `Open NumSharp source for ${detailName} on GitHub`;
    const detailHeading = detailUrl
      ? `<a class="cov-detail-api-link" data-api-link href="${escapeHtml(detailUrl)}" target="_blank" rel="noopener" title="${detailTitle}">${detailName}</a>`
      : detailName;
    const sourceLinks = (row.numsharp_source_urls || []).map((url, index) => {
      const path = row.numsharp_source_paths?.[index] || url;
      const fileName = path.split("/").pop();
      return `<a class="cov-source-link" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${escapeHtml(path)}">${escapeHtml(fileName)}</a>`;
    }).join("");
    target.innerHTML = `
      <div class="cov-detail-head"><h3>${detailHeading}</h3>${statusBadge(row.status)}</div>
      <div class="cov-detail-grid">
        <div class="cov-detail-fact"><div class="cov-detail-label">Surface</div><div class="cov-detail-value">${escapeHtml(surfaceLabel(row.surface))}</div></div>
        <div class="cov-detail-fact"><div class="cov-detail-label">Category</div><div class="cov-detail-value">${escapeHtml(row.category)}</div></div>
        <div class="cov-detail-fact"><div class="cov-detail-label">Availability</div><div class="cov-detail-value">${escapeHtml(mapping)}</div></div>
        <div class="cov-detail-fact"><div class="cov-detail-label">Inventory group</div><div class="cov-detail-value">${row.in_default_scope ? "Headline comparison" : isComparableApi(row) ? "Extended NumPy API" : row.origin === "numsharp" ? "NumSharp-only API" : "Supporting NumPy export"}${row.disposition ? ` · ${escapeHtml(dispositionLabels[row.disposition] || row.disposition)}` : ""}</div></div>
        ${row.applicability ? `<div class="cov-detail-fact"><div class="cov-detail-label">NumPy applicability</div><div class="cov-detail-value">${escapeHtml(row.applicability === "not_applicable" ? "NumPy rejects this ufunc method" : row.applicability === "conditional" ? "Depends on operands and dtype" : row.applicability)}</div></div>` : ""}
      </div>
      ${row.origin === "numpy" ? `<h4>NumPy signature</h4><div class="cov-code-block">${escapeHtml(row.numpy_signature)}</div>` : ""}
      <h4>NumSharp target</h4><div class="cov-code-block">${escapeHtml(row.numsharp_target || "Not available")}</div>
      <h4>Compiled C# overload${row.numsharp_signatures?.length === 1 ? "" : "s"}</h4>${signatures}
      ${sourceLinks ? `<h4>NumSharp source</h4><div class="cov-source-links">${sourceLinks}</div>` : ""}
      ${row.notes ? `<div class="cov-notes"><strong>Review note:</strong> ${escapeHtml(row.notes)}</div>` : ""}
      ${row.documentation_url ? `<a class="cov-doc-link" href="${escapeHtml(row.documentation_url)}" target="_blank" rel="noopener">Open NumPy reference ↗</a>` : ""}
    `;
  }

  function applyPreset(preset, value = "") {
    clearExplorerFilters();
    if (preset === "status") byId("cov-status").value = value;
    if (preset === "surface") byId("cov-surface").value = value;
    if (preset === "category") byId("cov-category").value = value;
    if (preset === "gaps") byId("cov-status").value = "gaps";
    if (preset === "exact") byId("cov-mapping").value = "exact";
    filterRows();
    byId("cov-explorer-heading").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearExplorerFilters() {
    byId("cov-search").value = "";
    ["cov-surface", "cov-category", "cov-status", "cov-kind", "cov-mapping"].forEach((id) => byId(id).value = "all");
    byId("cov-sort").value = "gap";
  }

  function resetFilters() {
    clearExplorerFilters();
    filterRows();
  }

  function changeScope() {
    clearExplorerFilters();
    initializeFilters();
    renderSummary();
    filterRows();
  }

  function tooltipRowsHtml(rows) {
    if (!rows.length) return `<div class="ns-cov-tip-row"><span>No APIs in this side</span></div>`;
    return rows.slice(0, 18).map((row) => {
      const label = `<code>${escapeHtml(apiLabel(row))}</code>`;
      const linkedLabel = row.documentation_url
        ? `<a data-api-link href="${escapeHtml(row.documentation_url)}" target="_blank" rel="noopener" title="Open latest NumPy documentation">${label}</a>`
        : label;
      return `<div class="ns-cov-tip-row">${linkedLabel}<span>${escapeHtml(statusLabels[row.status] || row.status)}</span></div>`;
    }).join("")
      + (rows.length > 18 ? `<div class="ns-cov-tip-row"><span>+${rows.length - 18} more in explorer</span></div>` : "");
  }

  function tooltipContent(element) {
    const separator = element.dataset.tooltipGroup.indexOf(":");
    const type = element.dataset.tooltipGroup.slice(0, separator);
    const value = element.dataset.tooltipGroup.slice(separator + 1);
    let rows = scopedRows();
    if (type === "status") rows = rows.filter((row) => row.status === value);
    if (type === "surface") rows = rows.filter((row) => row.surface === value);
    if (type === "category") rows = rows.filter((row) => row.category === value);
    const available = rows.filter((row) => row.status === "available");
    const gaps = rows.filter((row) => row.status === "missing" || row.status === "unsupported");
    const other = rows.filter((row) => row.status === "partial" || row.status === "extension");
    const counts = summarize(rows);
    const title = type === "status" ? statusLabels[value] : type === "surface" ? surfaceLabel(value) : value;
    const tabs = [["available", "Available", available], ["gaps", "Gaps", gaps]];
    if (other.length) tabs.push(["other", "Partial / NumSharp-only", other]);
    const initialTab = tabs.find(([, , items]) => items.length)?.[0] || "available";
    return `<div class="ns-cov-tip"><div class="ns-cov-tip-head"><strong>${escapeHtml(title)}</strong><span>${counts.total} entries · ${counts.numpy ? `${counts.availableApis}/${counts.numpy} NumPy APIs available · ${coverageLabel(counts)}` : "No comparable NumPy APIs"}</span></div>
      <div class="ns-cov-tip-tabs" role="tablist">${tabs.map(([key, label, items]) => `<button class="ns-cov-tip-tab" role="tab" aria-selected="${key === initialTab}" data-tip-tab="${key}">${label} (${items.length})</button>`).join("")}</div>
      ${tabs.map(([key, , items]) => `<div class="ns-cov-tip-panel" data-tip-panel="${key}"${key === initialTab ? "" : " hidden"}>${tooltipRowsHtml(items)}</div>`).join("")}</div>`;
  }

  function initializeTooltips() {
    state.tooltipInstances.forEach((instance) => instance.destroy());
    state.tooltipInstances = [];
    state.activeTooltip = null;
    const elements = [...root.querySelectorAll("[data-tooltip-group]")];
    if (!window.tippy) {
      elements.forEach((element) => element.title = "Select to filter this breakdown in the explorer.");
      return;
    }
    elements.forEach((element) => {
      let timer = null;
      let pinned = false;
      const instance = window.tippy(element, {
        allowHTML: true,
        appendTo: () => document.body,
        content: () => tooltipContent(element),
        interactive: true,
        maxWidth: 560,
        placement: "auto",
        theme: "ns-coverage",
        trigger: "manual",
        onShow(current) { if (state.activeTooltip && state.activeTooltip !== current) state.activeTooltip.hide(); state.activeTooltip = current; },
        onHidden(current) { if (state.activeTooltip === current) state.activeTooltip = null; pinned = false; },
        onDestroy() { window.clearTimeout(timer); }
      });
      const showSoon = () => { window.clearTimeout(timer); timer = window.setTimeout(() => instance.show(), 700); };
      const hideSoon = () => { window.clearTimeout(timer); if (!pinned) timer = window.setTimeout(() => instance.hide(), 120); };
      element.addEventListener("mouseenter", showSoon);
      element.addEventListener("mouseleave", hideSoon);
      element.addEventListener("focus", showSoon);
      element.addEventListener("blur", hideSoon);
      element.addEventListener("click", () => { window.clearTimeout(timer); pinned = !instance.state.isVisible || !pinned; pinned ? instance.show() : instance.hide(); });
      element.addEventListener("keydown", (event) => { if (event.key === "Escape") { pinned = false; instance.hide(); } });
      instance.popper.addEventListener("mouseenter", () => window.clearTimeout(timer));
      instance.popper.addEventListener("mouseleave", hideSoon);
      instance.popper.addEventListener("click", (event) => {
        const tab = event.target.closest?.("[data-tip-tab]");
        if (!tab) return;
        const key = tab.dataset.tipTab;
        instance.popper.querySelectorAll("[data-tip-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
        instance.popper.querySelectorAll("[data-tip-panel]").forEach((panel) => panel.hidden = panel.dataset.tipPanel !== key);
        instance.popperInstance?.update?.();
      });
      state.tooltipInstances.push(instance);
    });
  }

  function bindEvents() {
    ["cov-search", "cov-surface", "cov-category", "cov-status", "cov-kind", "cov-mapping", "cov-sort"].forEach((id) => {
      byId(id).addEventListener(id === "cov-search" ? "input" : "change", filterRows);
    });
    byId("cov-scope").addEventListener("change", changeScope);
    document.addEventListener("click", (event) => {
      const active = state.activeTooltip;
      if (!active || active.reference.contains(event.target) || active.popper.contains(event.target)) return;
      active.hide();
    });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") state.activeTooltip?.hide(); });
    byId("cov-reset").addEventListener("click", resetFilters);
    byId("cov-quick-gap").addEventListener("click", () => applyPreset("gaps"));
    byId("cov-metrics").addEventListener("click", (event) => {
      const action = event.target.closest?.("[data-metric-action]")?.dataset.metricAction;
      if (action) applyPreset(action);
    });
    byId("cov-status-track").addEventListener("click", (event) => {
      const status = event.target.closest?.("[data-status]")?.dataset.status;
      if (status) applyPreset("status", status);
    });
    byId("cov-surface-grid").addEventListener("click", (event) => {
      const surface = event.target.closest?.("[data-surface]")?.dataset.surface;
      if (surface) applyPreset("surface", surface);
    });
    byId("cov-category-grid").addEventListener("click", (event) => {
      const category = event.target.closest?.("[data-category]")?.dataset.category;
      if (category) applyPreset("category", category);
    });
    byId("cov-results").addEventListener("click", (event) => {
      if (event.target.closest?.("[data-api-link]")) return;
      const rowId = event.target.closest?.("[data-row-id]")?.dataset.rowId;
      if (rowId) { state.selectedId = rowId; renderResults(); renderDetail(); return; }
      if (event.target.closest?.("#cov-more")) { state.visible += 120; renderResults(); }
    });
    byId("cov-results").addEventListener("keydown", (event) => {
      const currentRow = event.target.closest?.("[data-row-id]");
      if ((event.key === "Enter" || event.key === " ") && currentRow && event.target === currentRow) {
        event.preventDefault();
        state.selectedId = currentRow.dataset.rowId;
        renderResults();
        renderDetail();
        return;
      }
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      const buttons = [...byId("cov-results").querySelectorAll("[data-row-id]")];
      const index = buttons.indexOf(document.activeElement);
      const next = event.key === "ArrowDown" ? Math.min(buttons.length - 1, index + 1) : Math.max(0, index - 1);
      if (buttons[next]) {
        event.preventDefault();
        state.selectedId = buttons[next].dataset.rowId;
        renderResults();
        renderDetail();
        byId("cov-results").querySelectorAll("[data-row-id]")[next]?.focus();
      }
    });
  }

  async function initialize() {
    try {
      const url = new URL("data/coverage.json", window.location.href);
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`Coverage artifact request failed (${response.status}).`);
      const data = await response.json();
      if (data.schema_version !== 1 || !Array.isArray(data.rows)) throw new Error("Coverage artifact has an unsupported schema.");
      state.rows = data.rows;
      initializeMetadata(data);
      initializeFilters();
      renderSummary();
      bindEvents();
      filterRows();
      byId("cov-loading").hidden = true;
      byId("cov-content").hidden = false;
    } catch (error) {
      byId("cov-loading").hidden = true;
      byId("cov-error").hidden = false;
      byId("cov-error").textContent = "Coverage data could not be loaded. Reload this page to try again.";
      console.error(error);
    }
  }

  initialize();
})();
</script>
