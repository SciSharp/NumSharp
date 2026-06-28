# Benchmarks

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

html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-cell,
html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-name,
html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-score,
html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-tab {
  color: var(--heat-text) !important;
  -webkit-text-fill-color: var(--heat-text);
}

html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-tab.is-active {
  color: var(--panel) !important;
  -webkit-text-fill-color: var(--panel);
}

html[data-bs-theme="dark"] .ns-bench-dashboard .dtype-count {
  color: var(--heat-muted-text) !important;
  -webkit-text-fill-color: var(--heat-muted-text);
}

.ns-bench-dashboard .bench-intro {
  border-bottom: 1px solid var(--line);
  padding: 0.25rem 0 1.2rem;
  margin-bottom: 1.25rem;
}

.ns-bench-dashboard .bench-kicker {
  color: var(--quiet);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.ns-bench-dashboard .bench-title {
  font-size: clamp(2rem, 4vw, 3.4rem);
  line-height: 1.02;
  margin: 0.35rem 0 0.6rem;
}

.ns-bench-dashboard .bench-subtitle {
  color: var(--quiet);
  max-width: 76ch;
  font-size: 1.04rem;
  margin: 0;
}

.ns-bench-dashboard .snapshot-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 1rem;
}

.ns-bench-dashboard .snapshot-pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.28rem 0.65rem;
  color: var(--quiet);
  background: var(--muted-bg);
  font-size: 0.82rem;
}

.ns-bench-dashboard .metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 1.25rem 0 1.5rem;
}

.ns-bench-dashboard .metric-card,
.ns-bench-dashboard .story-card,
.ns-bench-dashboard .priority-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.ns-bench-dashboard .metric-card {
  padding: 0.85rem;
  min-height: 7rem;
}

.ns-bench-dashboard .metric-label {
  color: var(--quiet);
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
}

.ns-bench-dashboard .metric-value {
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 750;
  margin: 0.45rem 0 0.3rem;
}

.ns-bench-dashboard .metric-note {
  color: var(--quiet);
  font-size: 0.86rem;
  margin: 0;
}

.ns-bench-dashboard .metric-good { color: var(--good); }
.ns-bench-dashboard .metric-near { color: var(--near); }
.ns-bench-dashboard .metric-slow { color: var(--slow); }

.ns-bench-dashboard .section-head {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
  border-top: 1px solid var(--line);
  padding-top: 1.2rem;
  margin: 1.45rem 0 0.75rem;
}

.ns-bench-dashboard .section-head h2 {
  font-size: 1.22rem;
  margin: 0;
  white-space: nowrap;
}

.ns-bench-dashboard .section-note {
  color: var(--quiet);
  font-size: 0.86rem;
  margin: 0;
}

.ns-bench-dashboard .read-guide {
  border-top: 1px solid var(--line);
  margin: 1.45rem 0 0;
  padding: 1rem 0 0.15rem;
}

.ns-bench-dashboard .read-guide + section .section-head {
  border-top: 0;
  margin-top: 1rem;
  padding-top: 0;
}

.ns-bench-dashboard .read-guide-head {
  align-items: end;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
  margin-bottom: 0.7rem;
}

.ns-bench-dashboard .read-guide h2 {
  font-size: 1.22rem;
  margin: 0;
}

.ns-bench-dashboard .guide-kicker {
  color: var(--quiet);
  font-size: 0.76rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.ns-bench-dashboard .guide-formula {
  background: var(--muted-bg);
  border: 1px solid var(--line);
  border-radius: 8px;
  display: grid;
  gap: 0.08rem;
  min-width: 0;
  padding: 0.48rem 0.68rem;
  white-space: nowrap;
}

.ns-bench-dashboard .guide-formula span {
  color: var(--quiet);
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
}

.ns-bench-dashboard .guide-formula strong {
  font-size: 0.92rem;
  font-variant-numeric: tabular-nums;
}

.ns-bench-dashboard .guide-formula small {
  color: var(--quiet);
  font-size: 0.74rem;
  font-variant-numeric: tabular-nums;
}

.ns-bench-dashboard .guide-primer {
  border: 1px solid color-mix(in srgb, var(--line) 86%, transparent);
  border-radius: 8px;
  display: grid;
  margin: 0 0 0.85rem;
}

.ns-bench-dashboard .guide-primer-row {
  align-items: start;
  display: grid;
  gap: 0.7rem;
  grid-template-columns: 10rem minmax(0, 1fr);
  padding: 0.56rem 0.68rem;
}

.ns-bench-dashboard .guide-primer-row + .guide-primer-row {
  border-top: 1px solid color-mix(in srgb, var(--line) 74%, transparent);
}

.ns-bench-dashboard .guide-primer-term {
  color: var(--ink);
  font-size: 0.8rem;
  font-weight: 750;
}

.ns-bench-dashboard .guide-primer-detail {
  color: var(--quiet);
  font-size: 0.82rem;
}

.ns-bench-dashboard .guide-grid {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.ns-bench-dashboard .guide-block {
  border-left: 3px solid color-mix(in srgb, var(--line) 70%, var(--quiet) 30%);
  min-width: 0;
  padding-left: 0.7rem;
}

.ns-bench-dashboard .guide-block h3 {
  font-size: 0.82rem;
  margin: 0 0 0.25rem;
}

.ns-bench-dashboard .guide-block p {
  color: var(--quiet);
  font-size: 0.84rem;
  margin: 0;
}

.ns-bench-dashboard .guide-band-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.42rem;
  margin-top: 0.38rem;
}

.ns-bench-dashboard .guide-band-block {
  margin-top: 0.78rem;
}

.ns-bench-dashboard .guide-band {
  align-items: center;
  border: 1px solid color-mix(in srgb, var(--line) 78%, transparent);
  border-radius: 999px;
  display: inline-flex;
  font-size: 0.8rem;
  font-weight: 650;
  gap: 0.42rem;
  min-height: 0;
  padding: 0.34rem 0.58rem;
  white-space: nowrap;
}

.ns-bench-dashboard .guide-band::before {
  border-radius: 999px;
  content: "";
  display: block;
  height: 0.62rem;
  width: 0.62rem;
}

.ns-bench-dashboard .guide-band.faster::before { background: var(--good); }
.ns-bench-dashboard .guide-band.close::before { background: var(--near); }
.ns-bench-dashboard .guide-band.slower::before { background: var(--slow); }
.ns-bench-dashboard .guide-band.much::before { background: var(--bad); }
.ns-bench-dashboard .guide-band.nodata::before { background: #87909a; }

.ns-bench-dashboard .status-bar {
  display: flex;
  position: relative;
  height: 1.2rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--muted-bg);
  isolation: isolate;
}

.ns-bench-dashboard .status-segment {
  display: block;
  position: relative;
  flex: 0 0 var(--w);
  min-width: 0.4rem;
  height: 100%;
  cursor: pointer;
  border-left: 1px solid rgba(255, 255, 255, 0.32);
  transition: box-shadow 120ms ease, transform 120ms ease, filter 120ms ease;
}

.ns-bench-dashboard .status-segment:first-child {
  border-left: 0;
  border-radius: 7px 0 0 7px;
}

.ns-bench-dashboard .status-segment:last-child {
  border-radius: 0 7px 7px 0;
}

.ns-bench-dashboard .status-segment:hover,
.ns-bench-dashboard .status-segment:focus-visible,
.ns-bench-dashboard .status-segment.is-tooltip-open {
  filter: saturate(1.12) brightness(1.04);
  outline: 2px solid color-mix(in srgb, var(--ink) 24%, transparent);
  outline-offset: 2px;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--panel) 76%, transparent);
  transform: translateY(-1px);
  z-index: 5;
}

.ns-bench-dashboard .status-segment::after {
  content: attr(data-tip);
  position: absolute;
  left: 50%;
  bottom: calc(100% + 0.6rem);
  width: max-content;
  min-width: min(30rem, 92vw);
  max-width: min(44rem, 92vw);
  padding: 0.85rem 1rem;
  border-radius: 8px;
  background: var(--ink);
  color: var(--panel);
  box-shadow: 0 0.85rem 2.2rem rgba(15, 23, 42, 0.28);
  font-size: 0.86rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.45;
  opacity: 0;
  pointer-events: none;
  transform: translate(-50%, 0.2rem);
  transition: opacity 120ms ease, transform 120ms ease;
  white-space: pre-line;
  z-index: 20;
}

.ns-bench-dashboard .status-segment::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: calc(100% + 0.22rem);
  width: 0.55rem;
  height: 0.55rem;
  background: var(--ink);
  opacity: 0;
  pointer-events: none;
  transform: translate(-50%, 0.2rem) rotate(45deg);
  transition: opacity 120ms ease, transform 120ms ease;
  z-index: 19;
}

.ns-bench-dashboard .status-segment:hover::after,
.ns-bench-dashboard .status-segment:hover::before,
.ns-bench-dashboard .status-segment:focus-visible::after,
.ns-bench-dashboard .status-segment:focus-visible::before {
  opacity: 1;
  transform: translate(-50%, 0) rotate(0);
}

.ns-bench-dashboard .status-segment:hover::before,
.ns-bench-dashboard .status-segment:focus-visible::before {
  transform: translate(-50%, 0) rotate(45deg);
}

.ns-bench-dashboard .status-segment.tip-left::after {
  left: 0;
  transform: translate(0, 0.2rem);
}

.ns-bench-dashboard .status-segment.tip-left:hover::after,
.ns-bench-dashboard .status-segment.tip-left:focus-visible::after {
  transform: translate(0, 0);
}

.ns-bench-dashboard .status-segment.tip-right::after {
  left: auto;
  right: 0;
  transform: translate(0, 0.2rem);
}

.ns-bench-dashboard .status-segment.tip-right:hover::after,
.ns-bench-dashboard .status-segment.tip-right:focus-visible::after {
  transform: translate(0, 0);
}

.ns-bench-dashboard .status-segment::after,
.ns-bench-dashboard .status-segment::before {
  content: none !important;
  display: none !important;
}

.ns-bench-dashboard .s-faster { background: var(--good); }
.ns-bench-dashboard .s-close { background: #d8a528; }
.ns-bench-dashboard .s-slower { background: var(--slow); }
.ns-bench-dashboard .s-much { background: var(--bad); }
.ns-bench-dashboard .s-nodata { background: #c7ccd1; }

.ns-bench-dashboard .band-faster-100x { background: linear-gradient(90deg, #032a1f, #054936); }
.ns-bench-dashboard .band-faster-20 { background: linear-gradient(90deg, #054936, #066345); }
.ns-bench-dashboard .band-faster-10 { background: linear-gradient(90deg, #066345, #08734f); }
.ns-bench-dashboard .band-faster-5 { background: linear-gradient(90deg, #08734f, #0d865c); }
.ns-bench-dashboard .band-faster-2 { background: linear-gradient(90deg, #0d865c, #19996b); }
.ns-bench-dashboard .band-faster-125 { background: linear-gradient(90deg, #19996b, #35ad7c); }
.ns-bench-dashboard .band-faster-100 { background: linear-gradient(90deg, #35ad7c, #73c79d); }
.ns-bench-dashboard .band-close-090 { background: linear-gradient(90deg, #9a680c, #bd8619); }
.ns-bench-dashboard .band-close-075 { background: linear-gradient(90deg, #bd8619, #d8a528); }
.ns-bench-dashboard .band-close-050 { background: linear-gradient(90deg, #d8a528, #e7bf55); }
.ns-bench-dashboard .band-slower-040 { background: linear-gradient(90deg, #e19a65, #d6763f); }
.ns-bench-dashboard .band-slower-030 { background: linear-gradient(90deg, #d6763f, #c75b25); }
.ns-bench-dashboard .band-slower-020 { background: linear-gradient(90deg, #c75b25, #a93d13); }
.ns-bench-dashboard .band-much-010 { background: linear-gradient(90deg, #cf5a50, #bd3237); }
.ns-bench-dashboard .band-much-005 { background: linear-gradient(90deg, #bd3237, #9d1926); }
.ns-bench-dashboard .band-much-000 { background: linear-gradient(90deg, #9d1926, #6f0c16); }
.ns-bench-dashboard .band-nodata {
  background:
    repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.32) 0 0.25rem, transparent 0.25rem 0.5rem),
    #c7ccd1;
}

.ns-bench-dashboard .legend-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.45rem 0.7rem;
  margin: 0.65rem 0 0;
}

.ns-bench-dashboard .legend-item {
  font-size: 0.8rem;
  color: var(--quiet);
  white-space: normal;
}

.ns-bench-dashboard .legend-swatch {
  display: inline-block;
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 3px;
  margin-right: 0.35rem;
  vertical-align: -0.08rem;
}

.ns-bench-dashboard .bar-table {
  display: grid;
  gap: 0.5rem;
}

.ns-bench-dashboard .bar-row {
  display: grid;
  grid-template-columns: minmax(8rem, 12rem) minmax(12rem, 1fr) 4.2rem 5.8rem;
  align-items: center;
  gap: 0.7rem;
  border-radius: 8px;
  cursor: pointer;
  padding: 0.08rem 0.12rem;
  transition: background 120ms ease, box-shadow 120ms ease, outline-color 120ms ease, transform 120ms ease;
}

.ns-bench-dashboard .bar-row:hover,
.ns-bench-dashboard .bar-row:focus-visible,
.ns-bench-dashboard .bar-row.is-tooltip-open {
  background: var(--muted-bg);
  outline: 1px solid color-mix(in srgb, var(--ink) 16%, transparent);
  outline-offset: 2px;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--ink) 12%, transparent);
  transform: translateY(-1px);
}

.ns-bench-dashboard .bar-label {
  font-weight: 650;
  font-size: 0.9rem;
}

.ns-bench-dashboard .bar-track {
  position: relative;
  height: 0.72rem;
  border-radius: 999px;
  background: var(--muted-bg);
  overflow: hidden;
}

.ns-bench-dashboard .bar-track::after {
  content: "";
  position: absolute;
  left: 40%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(80, 80, 80, 0.35);
}

.ns-bench-dashboard .bar-fill {
  display: block;
  height: 100%;
  width: var(--w);
  background: var(--tone);
  border-radius: 999px;
}

.ns-bench-dashboard .bar-score {
  font-weight: 750;
  text-align: right;
}

.ns-bench-dashboard .bar-count {
  color: var(--quiet);
  font-size: 0.82rem;
  text-align: right;
}

.ns-bench-dashboard .dtype-carousel {
  display: grid;
  gap: 0.7rem;
}

.ns-bench-dashboard .dtype-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.35rem;
  max-width: 30rem;
}

.ns-bench-dashboard .dtype-tab {
  appearance: none;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: color-mix(in srgb, var(--panel) 92%, var(--heat-text) 8%);
  color: var(--heat-text) !important;
  -webkit-text-fill-color: var(--heat-text);
  cursor: pointer;
  font: inherit;
  font-size: 0.86rem;
  font-weight: 700;
  line-height: 1;
  padding: 0.58rem 0.55rem;
  text-align: center;
}

.ns-bench-dashboard .dtype-tab:hover,
.ns-bench-dashboard .dtype-tab:focus-visible {
  border-color: color-mix(in srgb, var(--heat-text) 40%, var(--line));
  outline: 2px solid color-mix(in srgb, var(--heat-text) 20%, transparent);
  outline-offset: 2px;
}

.ns-bench-dashboard .dtype-tab.is-active {
  background: var(--heat-text);
  border-color: var(--heat-text);
  color: var(--panel) !important;
  -webkit-text-fill-color: var(--panel);
}

.ns-bench-dashboard .dtype-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(6.6rem, 1fr));
  gap: 0.55rem;
  min-width: 0;
}

.ns-bench-dashboard .dtype-panel[hidden] {
  display: none;
}

.ns-bench-dashboard .dtype-cell {
  --heat: var(--quiet);
  border-radius: 8px;
  border: 1px solid color-mix(in srgb, var(--heat) 42%, var(--line));
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--heat) 18%, transparent) 0 var(--heat-width), transparent var(--heat-width)),
    color-mix(in srgb, var(--heat) 10%, var(--panel));
  color: var(--heat-text) !important;
  -webkit-text-fill-color: var(--heat-text);
  cursor: pointer;
  min-width: 0;
  padding: 0.62rem 0.58rem;
  transition: box-shadow 120ms ease, filter 120ms ease, outline-color 120ms ease, transform 120ms ease;
}

.ns-bench-dashboard .dtype-cell:hover,
.ns-bench-dashboard .dtype-cell:focus-visible,
.ns-bench-dashboard .dtype-cell.is-tooltip-open {
  filter: saturate(1.08);
  outline: 2px solid color-mix(in srgb, var(--heat) 26%, transparent);
  outline-offset: 2px;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--heat) 34%, transparent);
  transform: translateY(-1px);
}

.ns-bench-dashboard .dtype-name {
  display: block;
  color: var(--heat-text) !important;
  -webkit-text-fill-color: var(--heat-text);
  font-weight: 750;
  font-size: 0.86rem;
  overflow-wrap: anywhere;
}

.ns-bench-dashboard .dtype-score {
  display: block;
  color: var(--heat-text) !important;
  -webkit-text-fill-color: var(--heat-text);
  margin-top: 0.28rem;
  font-size: 1.08rem;
  font-weight: 780;
  line-height: 1.05;
}

.ns-bench-dashboard .dtype-count {
  display: block;
  color: var(--heat-muted-text) !important;
  -webkit-text-fill-color: var(--heat-muted-text);
  font-size: 0.76rem;
  margin-top: 0.18rem;
}

.ns-bench-dashboard .dtype-meter {
  display: block;
  height: 0.28rem;
  margin-top: 0.5rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--ink) 10%, transparent);
  overflow: hidden;
}

.ns-bench-dashboard .dtype-meter span {
  display: block;
  width: var(--heat-width);
  height: 100%;
  border-radius: inherit;
  background: var(--heat);
}

.ns-bench-dashboard .heat-best { --heat: #109862; }
.ns-bench-dashboard .heat-good { --heat: #35a86e; }
.ns-bench-dashboard .heat-near { --heat: #c89422; }
.ns-bench-dashboard .heat-slow { --heat: #cf6728; }
.ns-bench-dashboard .heat-bad { --heat: #c92f3a; }
.ns-bench-dashboard .heat-empty { --heat: #8b949e; }

.ns-bench-dashboard .story-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}

.ns-bench-dashboard .story-card {
  padding: 0.9rem;
}

.ns-bench-dashboard .story-card h3 {
  font-size: 0.98rem;
  margin: 0 0 0.55rem;
}

.ns-bench-dashboard .story-card strong {
  display: block;
  font-size: 1.45rem;
  line-height: 1.1;
  margin-bottom: 0.35rem;
}

.ns-bench-dashboard .story-card p {
  color: var(--quiet);
  font-size: 0.88rem;
  margin: 0;
}

.ns-bench-dashboard .priority-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}

.ns-bench-dashboard .priority-card {
  padding: 0.9rem;
}

.ns-bench-dashboard .priority-card h3 {
  font-size: 0.98rem;
  margin: 0 0 0.55rem;
}

.ns-bench-dashboard .priority-list {
  margin: 0;
  padding-left: 1.1rem;
}

.ns-bench-dashboard .priority-list li {
  margin: 0.28rem 0;
}

.ns-bench-dashboard .priority-grid.priority-grid-compact {
  display: block;
}

.ns-bench-dashboard .priority-card.priority-compact {
  padding: 0.8rem 0.9rem;
}

.ns-bench-dashboard .priority-list.priority-list-compact {
  columns: 4 10rem;
  column-gap: 0.9rem;
  font-size: 0.78rem;
  line-height: 1.22;
  padding-left: 1.25rem;
}

.ns-bench-dashboard .priority-list-compact li {
  break-inside: avoid;
  margin: 0 0 0.32rem;
  padding-left: 0.1rem;
}

.ns-bench-dashboard .priority-list-compact strong {
  color: var(--ink);
  font-weight: 760;
}

.ns-bench-dashboard .link-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.ns-bench-dashboard .dash-link {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.45rem 0.65rem;
  text-decoration: none;
  font-weight: 650;
}

.tippy-box[data-theme~="ns-bench"] {
  --ns-tip-bg: var(--bs-body-bg, #ffffff);
  --ns-tip-fg: var(--bs-body-color, #1f2933);
  --ns-tip-border: var(--bs-border-color, #d8dee4);
  --ns-tip-muted: #6f7b86;
  --ns-tip-accent: #178a5a;
  --ns-tip-shadow-wide: rgba(2, 6, 23, 0.42);
  --ns-tip-shadow-tight: rgba(2, 6, 23, 0.24);
  --ns-tip-shadow-ring: rgba(2, 6, 23, 0.14);
  background-color: var(--ns-tip-bg) !important;
  background-image: linear-gradient(180deg, color-mix(in srgb, var(--ns-tip-bg) 96%, var(--ns-tip-accent) 4%) 0, var(--ns-tip-bg) 7.5rem) !important;
  background-clip: padding-box;
  border: 1px solid var(--ns-tip-border);
  border-radius: 10px;
  color: var(--ns-tip-fg);
  box-shadow:
    0 1.55rem 4.5rem var(--ns-tip-shadow-wide),
    0 0.55rem 1.45rem var(--ns-tip-shadow-tight),
    0 0 0 1px var(--ns-tip-shadow-ring) !important;
  font-size: 0.78rem;
  line-height: 1.28;
  opacity: 1 !important;
  overflow: hidden;
}

html[data-bs-theme="dark"] .tippy-box[data-theme~="ns-bench"] {
  --ns-tip-shadow-wide: rgba(0, 0, 0, 0.72);
  --ns-tip-shadow-tight: rgba(0, 0, 0, 0.48);
  --ns-tip-shadow-ring: rgba(255, 255, 255, 0.08);
}

.tippy-box[data-theme~="ns-bench"] > .tippy-svg-arrow {
  fill: var(--ns-tip-bg);
}

.tippy-box[data-theme~="ns-bench"] > .tippy-arrow {
  color: var(--ns-tip-bg);
}

.tippy-box[data-theme~="ns-bench"] .tippy-content {
  background-color: var(--ns-tip-bg);
  padding: 0;
}

.ns-tip {
  background-color: var(--ns-tip-bg);
  background-image: linear-gradient(180deg, color-mix(in srgb, var(--ns-tip-bg) 96%, var(--ns-tip-accent) 4%) 0, var(--ns-tip-bg) 7.5rem);
  max-width: min(760px, calc(100vw - 24px));
}

.ns-tip-title {
  font-size: 0.92rem;
  font-weight: 760;
  padding: 0.76rem 0.86rem 0.3rem;
}

.ns-tip-meta {
  color: var(--ns-tip-muted);
  font-size: 0.74rem;
  padding: 0 0.86rem 0.54rem;
}

.ns-tip-switch {
  background: color-mix(in srgb, var(--ns-tip-bg) 88%, var(--ns-tip-fg) 12%);
  border: 1px solid color-mix(in srgb, var(--ns-tip-border) 84%, var(--ns-tip-fg) 16%);
  border-radius: 999px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin: 0 0.86rem 0.68rem;
  overflow: hidden;
}

.ns-tip-switch-button {
  appearance: none;
  background: transparent;
  border: 0;
  color: var(--ns-tip-fg);
  cursor: pointer;
  font: inherit;
  font-size: 0.74rem;
  font-weight: 760;
  line-height: 1;
  padding: 0.5rem 0.72rem;
  text-align: center;
}

.ns-tip-switch-button + .ns-tip-switch-button {
  border-left: 1px solid color-mix(in srgb, var(--ns-tip-border) 84%, transparent);
}

.ns-tip-switch-button:hover,
.ns-tip-switch-button:focus-visible {
  background: color-mix(in srgb, var(--ns-tip-fg) 8%, transparent);
  outline: none;
}

.ns-tip-switch-button.is-active {
  background: var(--ns-tip-fg);
  color: var(--ns-tip-bg);
}

.ns-tip-panel[hidden] {
  display: none;
}

.ns-tip-panel-note {
  border-top: 1px solid color-mix(in srgb, var(--ns-tip-border) 68%, transparent);
  color: var(--ns-tip-muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  padding: 0.44rem 0.86rem;
  text-transform: uppercase;
}

.ns-tip-scroll {
  max-height: min(420px, 68vh);
  overscroll-behavior: contain;
  overflow: auto;
}

.ns-tip-table {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  min-width: 43rem;
  width: 100%;
}

.ns-tip-table th,
.ns-tip-table td {
  border-top: 1px solid color-mix(in srgb, var(--ns-tip-border) 72%, transparent);
  padding: 0.34rem 0.46rem;
  text-align: left;
  vertical-align: top;
}

.ns-tip-table th {
  background: color-mix(in srgb, var(--ns-tip-bg) 92%, var(--ns-tip-fg) 8%);
  color: var(--ns-tip-muted);
  font-size: 0.68rem;
  font-weight: 750;
  position: sticky;
  text-transform: uppercase;
  top: 0;
  z-index: 1;
}

.ns-tip-table .num {
  text-align: right;
  white-space: nowrap;
}

.ns-perf-ratio {
  align-items: center;
  display: inline-flex;
  gap: 0.34rem;
  justify-content: flex-end;
}

.ns-perf-dot {
  background: var(--perf-color, var(--ns-tip-muted));
  border-radius: 999px;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--perf-color, var(--ns-tip-muted)) 18%, transparent);
  display: inline-block;
  flex: 0 0 auto;
  height: 0.54rem;
  width: 0.54rem;
}

.ns-perf-excellent { --perf-color: #0f8f68; }
.ns-perf-good { --perf-color: #178a5a; }
.ns-perf-near { --perf-color: #bd8b14; }
.ns-perf-slow { --perf-color: #c65f20; }
.ns-perf-bad { --perf-color: #b4232c; }
.ns-perf-empty { --perf-color: #87909a; }

.ns-tip-table .op {
  max-width: 22rem;
  min-width: 14rem;
}

.ns-tip-empty {
  color: var(--ns-tip-muted);
  padding: 0.8rem;
}

.contribution + .next-article {
  display: none !important;
}

@media (max-width: 1200px) {
  .ns-bench-dashboard .metric-grid,
  .ns-bench-dashboard .story-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .ns-bench-dashboard .guide-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 760px) {
  .ns-bench-dashboard .metric-grid,
  .ns-bench-dashboard .story-grid,
  .ns-bench-dashboard .priority-grid,
  .ns-bench-dashboard .legend-grid {
    grid-template-columns: 1fr;
  }

  .ns-bench-dashboard .section-head {
    align-items: flex-start;
    flex-direction: column;
    gap: 0.35rem;
  }

  .ns-bench-dashboard .read-guide-head {
    align-items: flex-start;
    flex-direction: column;
    gap: 0.55rem;
  }

  .ns-bench-dashboard .guide-formula {
    align-self: stretch;
    width: 100%;
  }

  .ns-bench-dashboard .guide-grid {
    grid-template-columns: 1fr;
  }

  .ns-bench-dashboard .guide-primer-row {
    gap: 0.22rem;
    grid-template-columns: 1fr;
  }

  .ns-bench-dashboard .guide-band {
    white-space: normal;
  }

  .ns-bench-dashboard .bar-row {
    grid-template-columns: 1fr 4.2rem;
    gap: 0.35rem 0.6rem;
  }

  .ns-bench-dashboard .bar-track {
    grid-column: 1 / -1;
    order: 3;
  }

  .ns-bench-dashboard .bar-count {
    display: none;
  }

  .ns-bench-dashboard .dtype-tabs {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .ns-bench-dashboard .priority-list.priority-list-compact {
    columns: 1;
  }
}
</style>

<div class="ns-bench-dashboard">
  <section class="bench-intro">
    <div class="bench-kicker">NumSharp performance lab</div>
    <h2 class="bench-title">Benchmark Dashboard</h2>
    <p class="bench-subtitle">
      A compact operating view of the NumSharp vs NumPy benchmark suite: 14 official op suites,
      all supported dtypes where applicable, three cache tiers, and five subsystem scans for iterator,
      layout, operand, cast, and fusion behavior.
    </p>
    <div class="snapshot-strip" aria-label="Benchmark snapshot details">
      <span class="snapshot-pill">Snapshot: 2026-06-23</span>
      <span class="snapshot-pill">Commit: e3b7c268</span>
      <span class="snapshot-pill">NumPy: 2.4.2</span>
      <span class="snapshot-pill">Ratio: NumPy_ms / NumSharp_ms</span>
      <span class="snapshot-pill">Higher is better</span>
    </div>
  </section>

  <section class="metric-grid" aria-label="Headline benchmark metrics">
    <article class="metric-card">
      <div class="metric-label">Operation cells</div>
      <div class="metric-value">1,851</div>
      <p class="metric-note">op x dtype x size rows in the official matrix</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Large arrays</div>
      <div class="metric-value metric-good">1.26x</div>
      <p class="metric-note">10M-element geomean, 80% of NumPy time</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Cache tier</div>
      <div class="metric-value metric-near">0.90x</div>
      <p class="metric-note">100K-element geomean, current main pressure point</p>
    </article>
    <article class="metric-card">
      <div class="metric-label">Cast subsystem</div>
      <div class="metric-value metric-good">1,439</div>
      <p class="metric-note">wins out of 1,568 comparable cast cells</p>
    </article>
  </section>

  <section class="read-guide" aria-labelledby="read-guide-title">
    <div class="read-guide-head">
      <div>
        <div class="guide-kicker">Operation cells</div>
        <h2 id="read-guide-title">Legend &amp; How To Read</h2>
      </div>
      <div class="guide-formula" aria-label="Ratio formula">
        <span>Ratio</span>
        <strong>NumPy / NumSharp</strong>
        <small>Example: x3 = NumSharp 10s, NumPy 30s</small>
      </div>
    </div>
    <div class="guide-primer" aria-label="Technical reading guide">
      <div class="guide-primer-row">
        <div class="guide-primer-term">Benchmark row</div>
        <div class="guide-primer-detail">Each result is one operation x dtype x size timing cell from the latest benchmark snapshot; rollups use comparable measured rows.</div>
      </div>
      <div class="guide-primer-row">
        <div class="guide-primer-term">Timing fields</div>
        <div class="guide-primer-detail">Ratios come from raw NumPy_ms and NumSharp_ms timings. A value above 1.00x is a NumSharp win; below 1.00x means NumPy was faster.</div>
      </div>
      <div class="guide-primer-row">
        <div class="guide-primer-term">Drill-down</div>
        <div class="guide-primer-detail">Click Status Mix segments, Suite Scoreboard rows, and Dtype Heatmap cards to open the internal top-25 best and worst benchmark rows for that grouping.</div>
      </div>
    </div>
    <div class="guide-grid">
      <div class="guide-block">
        <h3>Cell</h3>
        <p>One benchmark row: operation, dtype, and size tier. The dashboard uses best timed runs after warmup.</p>
      </div>
      <div class="guide-block">
        <h3>Reading Ratios</h3>
        <p>Higher is better. 2.00x means NumSharp is twice as fast as NumPy; 0.50x means NumSharp takes about twice as long.</p>
      </div>
    </div>
    <div class="guide-block guide-band-block">
      <h3>Performance Bands</h3>
      <div class="guide-band-row" aria-label="Performance color bands">
        <span class="guide-band faster">Faster: 1.00x and above</span>
        <span class="guide-band close">Close: 0.50x to 1.00x</span>
        <span class="guide-band slower">Slower: 0.20x to 0.50x</span>
        <span class="guide-band much">Much slower: below 0.20x</span>
        <span class="guide-band nodata">No data: pending C# measurement</span>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Status Mix</h2>
      <p class="section-note">All measured rows classified by NumPy / NumSharp, including sub-microsecond rows; only no-data cells are separate</p>
    </div>
    <div class="status-bar" role="list" aria-label="Status mix ratio bands ordered from fastest measured rows to slowest measured rows, followed by no-data rows">
      <span class="status-segment band-faster-100x tip-left" role="listitem" tabindex="0" style="--w:0.216%" aria-label="4 rows at 100x or faster" data-tip="4 faster rows, &gt;=100x&#10;1. np.zeros_like (float64) | dtype=float64 | N=10M | 2150.69x | NS &lt;0.1% NP&#10;2. np.zeros_like (int64) | dtype=int64 | N=10M | 1585.88x | NS 0.1% NP&#10;3. np.zeros_like (int32) | dtype=int32 | N=10M | 1230.61x | NS 0.1% NP&#10;4. np.zeros_like (float32) | dtype=float32 | N=10M | 1061.51x | NS 0.1% NP"></span>
      <span class="status-segment band-faster-20 tip-left" role="listitem" tabindex="0" style="--w:0.054%" aria-label="1 row at 20x to 100x faster" data-tip="1 faster row, 20-100x&#10;1. np.prod axis=1 (float64) | dtype=float64 | N=10M | 23.97x | NS 4.2% NP"></span>
      <span class="status-segment band-faster-10 tip-left" role="listitem" tabindex="0" style="--w:1.243%" aria-label="23 rows at 10x to 20x faster" data-tip="23 faster rows, 10-20x (top 10)&#10;1. np.dot(a, b) (float64) | dtype=float64 | N=100K | 14.17x | NS 7.1% NP&#10;2. np.prod (float64) | dtype=float64 | N=100K | 13.75x | NS 7.3% NP&#10;3. np.nanstd(a) (float64) | dtype=float64 | N=1K | 13.33x | NS 7.5% NP&#10;4. np.std (float32) | dtype=float32 | N=1K | 12.79x | NS 7.8% NP&#10;5. np.nanstd(a) (float16) | dtype=float16 | N=1K | 12.50x | NS 8.0% NP&#10;6. np.percentile(a, 50) (float64) | dtype=float64 | N=1K | 12.18x | NS 8.2% NP&#10;7. np.nanquantile(a, 0.5) (float32) | dtype=float32 | N=1K | 11.86x | NS 8.4% NP&#10;8. np.nanvar(a) (float16) | dtype=float16 | N=1K | 11.67x | NS 8.6% NP&#10;9. np.nanpercentile(a, 50) (float32) | dtype=float32 | N=1K | 11.51x | NS 8.7% NP&#10;10. np.nanpercentile(a, 50) (float64) | dtype=float64 | N=1K | 11.42x | NS 8.8% NP"></span>
      <span class="status-segment band-faster-5 tip-left" role="listitem" tabindex="0" style="--w:4.376%" aria-label="81 rows at 5x to 10x faster" data-tip="81 faster rows, 5-10x (top 10)&#10;1. np.nanprod(a) (float32) | dtype=float32 | N=10M | 9.91x | NS 10.1% NP&#10;2. np.sum axis=1 (int16) | dtype=int16 | N=100K | 9.75x | NS 10.3% NP&#10;3. np.sum axis=1 (int8) | dtype=int8 | N=100K | 9.66x | NS 10.3% NP&#10;4. np.nanquantile(a, 0.5) (float64) | dtype=float64 | N=1K | 9.59x | NS 10.4% NP&#10;5. np.std (float16) | dtype=float16 | N=1K | 9.46x | NS 10.6% NP&#10;6. np.quantile(a, 0.5) (float64) | dtype=float64 | N=1K | 9.44x | NS 10.6% NP&#10;7. np.count_nonzero(a) (float32) | dtype=float32 | N=1K | 9.41x | NS 10.6% NP&#10;8. np.sum axis=0 (uint8) | dtype=uint8 | N=100K | 9.40x | NS 10.6% NP&#10;9. np.nansum(a) (float32) | dtype=float32 | N=10M | 9.30x | NS 10.8% NP&#10;10. np.var (float16) | dtype=float16 | N=1K | 9.03x | NS 11.1% NP"></span>
      <span class="status-segment band-faster-2 tip-left" role="listitem" tabindex="0" style="--w:16.694%" aria-label="309 rows at 2x to 5x faster" data-tip="309 faster rows, 2-5x (top 10)&#10;1. np.amin axis=0 (uint8) | dtype=uint8 | N=1K | 4.95x | NS 20.2% NP&#10;2. np.var (float32) | dtype=float32 | N=10M | 4.91x | NS 20.4% NP&#10;3. np.std axis=0 (float32) | dtype=float32 | N=1K | 4.90x | NS 20.4% NP&#10;4. np.std axis=0 (float64) | dtype=float64 | N=10M | 4.90x | NS 20.4% NP&#10;5. np.sum axis=1 (int16) | dtype=int16 | N=10M | 4.90x | NS 20.4% NP&#10;6. np.nanmin(a) (float16) | dtype=float16 | N=1K | 4.89x | NS 20.4% NP&#10;7. np.std (float16) | dtype=float16 | N=100K | 4.85x | NS 20.6% NP&#10;8. np.var (float16) | dtype=float16 | N=10M | 4.81x | NS 20.8% NP&#10;9. np.nansum(a) (float16) | dtype=float16 | N=1K | 4.79x | NS 20.9% NP&#10;10. np.average(a) (float64) | dtype=float64 | N=10M | 4.79x | NS 20.9% NP"></span>
      <span class="status-segment band-faster-125 tip-left" role="listitem" tabindex="0" style="--w:15.181%" aria-label="281 rows at 1.25x to 2x faster" data-tip="281 faster rows, 1.25-2x (top 10)&#10;1. np.amax axis=0 (int8) | dtype=int8 | N=100K | 2.00x | NS 50.1% NP&#10;2. np.mean (int32) | dtype=int32 | N=100K | 2.00x | NS 50.1% NP&#10;3. np.amax (int16) | dtype=int16 | N=100K | 1.99x | NS 50.2% NP&#10;4. a ^ b (uint16) | dtype=uint16 | N=10M | 1.99x | NS 50.2% NP&#10;5. np.argmax (float16) | dtype=float16 | N=100K | 1.98x | NS 50.4% NP&#10;6. a | b (uint16) | dtype=uint16 | N=100K | 1.98x | NS 50.4% NP&#10;7. a ^ b (int16) | dtype=int16 | N=100K | 1.98x | NS 50.5% NP&#10;8. np.prod axis=1 (int64) | dtype=int64 | N=1K | 1.98x | NS 50.5% NP&#10;9. np.amin axis=0 (int16) | dtype=int16 | N=100K | 1.98x | NS 50.5% NP&#10;10. a - b (element-wise) (uint16) | dtype=uint16 | N=100K | 1.98x | NS 50.6% NP"></span>
      <span class="status-segment band-faster-100" role="listitem" tabindex="0" style="--w:12.426%" aria-label="230 rows at 1x to 1.25x faster" data-tip="230 faster rows, 1.00-1.25x (top 10)&#10;1. np.add(a, b) (uint8) | dtype=uint8 | N=10M | 1.25x | NS 80.1% NP&#10;2. np.mean axis=1 (float32) | dtype=float32 | N=100K | 1.25x | NS 80.1% NP&#10;3. np.var axis=0 (float16) | dtype=float16 | N=10M | 1.25x | NS 80.3% NP&#10;4. np.nanmedian(a) (float16) | dtype=float16 | N=10M | 1.25x | NS 80.3% NP&#10;5. np.searchsorted(a, v) (int32) | dtype=int32 | N=100K | 1.25x | NS 80.2% NP&#10;6. np.log1p (float32) | dtype=float32 | N=100K | 1.25x | NS 80.3% NP&#10;7. np.amax axis=0 (uint64) | dtype=uint64 | N=10M | 1.25x | NS 80.3% NP&#10;8. np.sin (float32) | dtype=float32 | N=1K | 1.24x | NS 80.5% NP&#10;9. np.log1p (float32) | dtype=float32 | N=1K | 1.24x | NS 80.8% NP&#10;10. np.mean (complex128) | dtype=complex128 | N=10M | 1.24x | NS 80.9% NP"></span>
      <span class="status-segment band-close-090" role="listitem" tabindex="0" style="--w:5.781%" aria-label="107 close rows at 0.90x to 1.00x" data-tip="107 close rows, 0.90-1.00x (top 10)&#10;1. a != b (float64) | dtype=float64 | N=10M | 1.00x | NS 100.1% NP&#10;2. np.argmax (int8) | dtype=int8 | N=100K | 1.00x | NS 100.2% NP&#10;3. np.mean axis=1 (complex128) | dtype=complex128 | N=100K | 1.00x | NS 100.4% NP&#10;4. np.array_equal(a, b) (float32) | dtype=float32 | N=10M | 1.00x | NS 100.4% NP&#10;5. np.sin (float32) | dtype=float32 | N=100K | 1.00x | NS 100.4% NP&#10;6. a &lt; b (float32) | dtype=float32 | N=10M | 1.00x | NS 100.5% NP&#10;7. np.cos (float32) | dtype=float32 | N=100K | 1.00x | NS 100.5% NP&#10;8. np.cbrt(a) (float64) | dtype=float64 | N=100K | 0.99x | NS 100.6% NP&#10;9. a &lt; b (float64) | dtype=float64 | N=10M | 0.99x | NS 100.8% NP&#10;10. a &gt; b (float64) | dtype=float64 | N=10M | 0.99x | NS 101.0% NP"></span>
      <span class="status-segment band-close-075" role="listitem" tabindex="0" style="--w:5.619%" aria-label="104 close rows at 0.75x to 0.90x" data-tip="104 close rows, 0.75-0.90x (top 10)&#10;1. a - scalar (uint8) | dtype=uint8 | N=1K | 0.90x | NS 111.2% NP&#10;2. np.amin axis=0 (float32) | dtype=float32 | N=10M | 0.90x | NS 111.3% NP&#10;3. a - scalar (uint16) | dtype=uint16 | N=1K | 0.90x | NS 111.4% NP&#10;4. np.amax (int8) | dtype=int8 | N=10M | 0.90x | NS 111.4% NP&#10;5. np.cbrt(a) (float16) | dtype=float16 | N=10M | 0.90x | NS 111.7% NP&#10;6. np.quantile(a, 0.5) (float64) | dtype=float64 | N=10M | 0.90x | NS 111.7% NP&#10;7. a % 7 (literal) (float32) | dtype=float32 | N=10M | 0.89x | NS 111.9% NP&#10;8. a % b (element-wise) (float64) | dtype=float64 | N=100K | 0.89x | NS 112.2% NP&#10;9. np.amin (float32) | dtype=float32 | N=10M | 0.89x | NS 112.4% NP&#10;10. np.log (float64) | dtype=float64 | N=1K | 0.89x | NS 112.4% NP"></span>
      <span class="status-segment band-close-050 tip-right" role="listitem" tabindex="0" style="--w:12.048%" aria-label="223 close rows at 0.50x to 0.75x" data-tip="223 close rows, 0.50-0.75x (top 10)&#10;1. scalar / a (int32) | dtype=int32 | N=10M | 0.75x | NS 133.7% NP&#10;2. a - b (element-wise) (uint8) | dtype=uint8 | N=1K | 0.75x | NS 133.8% NP&#10;3. np.invert(a) (int64) | dtype=int64 | N=10M | 0.75x | NS 133.7% NP&#10;4. a / scalar (int32) | dtype=int32 | N=10M | 0.75x | NS 134.1% NP&#10;5. np.log2 (float16) | dtype=float16 | N=1K | 0.75x | NS 134.3% NP&#10;6. np.reciprocal(a) (float16) | dtype=float16 | N=1K | 0.75x | NS 134.2% NP&#10;7. np.where(cond) (float64) | dtype=float64 | N=1K | 0.74x | NS 134.4% NP&#10;8. a + scalar (int64) | dtype=int64 | N=10M | 0.74x | NS 134.6% NP&#10;9. np.argmin (uint16) | dtype=uint16 | N=10M | 0.74x | NS 134.9% NP&#10;10. a * 2 (literal) (float16) | dtype=float16 | N=1K | 0.74x | NS 135.2% NP"></span>
      <span class="status-segment band-slower-040 tip-right" role="listitem" tabindex="0" style="--w:4.106%" aria-label="76 slower rows at 0.40x to 0.50x" data-tip="76 slower rows, 0.40-0.50x (top 10)&#10;1. a + scalar (int32) | dtype=int32 | N=1K | 0.50x | NS 200.2% NP&#10;2. a &amp; b (uint8) | dtype=uint8 | N=1K | 0.50x | NS 201.4% NP&#10;3. np.mean axis=0 (float16) | dtype=float16 | N=10M | 0.50x | NS 201.5% NP&#10;4. scalar - a (float32) | dtype=float32 | N=1K | 0.50x | NS 202.2% NP&#10;5. np.argmin (int64) | dtype=int64 | N=100K | 0.49x | NS 202.2% NP&#10;6. np.nanmax(a) (float64) | dtype=float64 | N=10M | 0.49x | NS 202.8% NP&#10;7. a - b (element-wise) (int64) | dtype=int64 | N=100K | 0.49x | NS 203.2% NP&#10;8. a - scalar (uint64) | dtype=uint64 | N=1K | 0.49x | NS 203.3% NP&#10;9. np.outer(a, b) (float64) | dtype=float64 | N=100K | 0.49x | NS 203.5% NP&#10;10. np.argmin (uint64) | dtype=uint64 | N=10M | 0.49x | NS 203.6% NP"></span>
      <span class="status-segment band-slower-030 tip-right" role="listitem" tabindex="0" style="--w:4.916%" aria-label="91 slower rows at 0.30x to 0.40x" data-tip="91 slower rows, 0.30-0.40x (top 10)&#10;1. a * a (square) (uint16) | dtype=uint16 | N=1K | 0.40x | NS 251.4% NP&#10;2. a | b (int16) | dtype=int16 | N=1K | 0.40x | NS 251.2% NP&#10;3. a * b (element-wise) (uint64) | dtype=uint64 | N=1K | 0.40x | NS 252.3% NP&#10;4. a + scalar (int64) | dtype=int64 | N=100K | 0.39x | NS 254.7% NP&#10;5. a * a (square) (int32) | dtype=int32 | N=1K | 0.39x | NS 256.0% NP&#10;6. a &gt;= b (float64) | dtype=float64 | N=100K | 0.39x | NS 255.9% NP&#10;7. np.isfinite(a) (float64) | dtype=float64 | N=1K | 0.39x | NS 255.7% NP&#10;8. a &gt; b (float64) | dtype=float64 | N=100K | 0.39x | NS 255.5% NP&#10;9. a * 2 (literal) (int32) | dtype=int32 | N=1K | 0.39x | NS 256.1% NP&#10;10. a * a (square) (uint8) | dtype=uint8 | N=1K | 0.39x | NS 256.9% NP"></span>
      <span class="status-segment band-slower-020 tip-right" role="listitem" tabindex="0" style="--w:6.699%" aria-label="124 slower rows at 0.20x to 0.30x" data-tip="124 slower rows, 0.20-0.30x (top 10)&#10;1. np.argmax (int64) | dtype=int64 | N=100K | 0.30x | NS 334.9% NP&#10;2. a - scalar (float64) | dtype=float64 | N=1K | 0.30x | NS 334.6% NP&#10;3. a * a (square) (int64) | dtype=int64 | N=100K | 0.30x | NS 335.6% NP&#10;4. a / scalar (float32) | dtype=float32 | N=1K | 0.30x | NS 336.3% NP&#10;5. np.dot(a, b) (float64) | dtype=float64 | N=10M | 0.30x | NS 337.5% NP&#10;6. np.exp (float32) | dtype=float32 | N=1K | 0.30x | NS 338.9% NP&#10;7. a * 2 (literal) (int16) | dtype=int16 | N=1K | 0.30x | NS 338.5% NP&#10;8. a * b (element-wise) (complex128) | dtype=complex128 | N=1K | 0.29x | NS 340.6% NP&#10;9. a * 2 (literal) (uint16) | dtype=uint16 | N=1K | 0.29x | NS 343.6% NP&#10;10. a + 5 (literal) (int32) | dtype=int32 | N=1K | 0.29x | NS 343.6% NP"></span>
      <span class="status-segment band-much-010 tip-right" role="listitem" tabindex="0" style="--w:3.944%" aria-label="73 much slower rows at 0.10x to 0.20x" data-tip="73 much slower rows, 0.10-0.20x (top 10)&#10;1. a + 5 (literal) (uint32) | dtype=uint32 | N=1K | 0.20x | NS 502.0% NP&#10;2. np.floor (float64) | dtype=float64 | N=100K | 0.20x | NS 501.8% NP&#10;3. np.left_shift(a, 2) (uint16) | dtype=uint16 | N=1K | 0.20x | NS 502.8% NP&#10;4. np.left_shift(a, 2) (uint32) | dtype=uint32 | N=1K | 0.20x | NS 502.6% NP&#10;5. np.sign (float64) | dtype=float64 | N=1K | 0.20x | NS 504.1% NP&#10;6. a * 2 (literal) (float32) | dtype=float32 | N=1K | 0.20x | NS 504.5% NP&#10;7. np.left_shift(a, 2) (int16) | dtype=int16 | N=1K | 0.20x | NS 510.5% NP&#10;8. np.add(a, b) (float64) | dtype=float64 | N=1K | 0.20x | NS 514.0% NP&#10;9. np.right_shift(a, 2) (uint16) | dtype=uint16 | N=1K | 0.20x | NS 511.7% NP&#10;10. np.exp2 (float32) | dtype=float32 | N=100K | 0.19x | NS 515.7% NP"></span>
      <span class="status-segment band-much-005 tip-right" role="listitem" tabindex="0" style="--w:2.215%" aria-label="41 much slower rows at 0.05x to 0.10x" data-tip="41 much slower rows, 0.05-0.10x (top 10)&#10;1. np.right_shift(a, 2) (int16) | dtype=int16 | N=100K | 0.10x | NS 1021.3% NP&#10;2. np.zeros_like (int32) | dtype=int32 | N=1K | 0.10x | NS 1051.8% NP&#10;3. np.zeros_like (int64) | dtype=int64 | N=1K | 0.10x | NS 1056.4% NP&#10;4. np.invert(a) (bool) | dtype=bool | N=100K | 0.09x | NS 1068.5% NP&#10;5. np.ones (float32) | dtype=float32 | N=1K | 0.09x | NS 1080.3% NP&#10;6. a[::-1] (reversed) | dtype=float64 | N=10M | 0.09x | NS 1091.4% NP&#10;7. a | b (uint64) | dtype=uint64 | N=1K | 0.09x | NS 1094.0% NP&#10;8. np.invert(a) (uint64) | dtype=uint64 | N=1K | 0.09x | NS 1161.5% NP&#10;9. np.isnan(a) (float32) | dtype=float32 | N=100K | 0.09x | NS 1177.4% NP&#10;10. np.full (float32) | dtype=float32 | N=1K | 0.08x | NS 1199.1% NP"></span>
      <span class="status-segment band-much-000 tip-right" role="listitem" tabindex="0" style="--w:0.756%" aria-label="14 much slower rows below 0.05x" data-tip="14 much slower rows, &lt;0.05x (slowest 10)&#10;1. np.empty (float64) | dtype=float64 | N=1K | 0.02x | NS 4784.9% NP&#10;2. np.empty (int32) | dtype=int32 | N=100K | 0.02x | NS 4659.1% NP&#10;3. np.empty (int64) | dtype=int64 | N=100K | 0.02x | NS 4388.7% NP&#10;4. np.empty (float64) | dtype=float64 | N=100K | 0.02x | NS 4137.0% NP&#10;5. np.empty (int64) | dtype=int64 | N=1K | 0.03x | NS 3888.0% NP&#10;6. np.empty (float32) | dtype=float32 | N=100K | 0.03x | NS 3787.7% NP&#10;7. np.zeros (float64) | dtype=float64 | N=1K | 0.03x | NS 3534.8% NP&#10;8. np.empty (int32) | dtype=int32 | N=1K | 0.03x | NS 3239.4% NP&#10;9. np.zeros (int32) | dtype=int32 | N=1K | 0.04x | NS 2881.0% NP&#10;10. np.copy (int64) | dtype=int64 | N=1K | 0.04x | NS 2479.3% NP"></span>
      <span class="status-segment band-nodata tip-right" role="listitem" tabindex="0" style="--w:3.73%" aria-label="69 no-data rows pending C# measurements" data-tip="69 no-data rows, pending C# measurements (first 10)&#10;1. matrix + scalar | dtype=float64 | N=1K | pending C#&#10;2. matrix + row_vector (N,M)+(M,) | dtype=float64 | N=1K | pending C#&#10;3. matrix + col_vector (N,M)+(N,1) | dtype=float64 | N=1K | pending C#&#10;4. np.broadcast_to(row, (N,M)) | dtype=float64 | N=1K | pending C#&#10;5. matrix + scalar | dtype=float64 | N=100K | pending C#&#10;6. matrix + row_vector (N,M)+(M,) | dtype=float64 | N=100K | pending C#&#10;7. matrix + col_vector (N,M)+(N,1) | dtype=float64 | N=100K | pending C#&#10;8. np.broadcast_to(row, (N,M)) | dtype=float64 | N=100K | pending C#&#10;9. reshape 1D-&gt;2D | dtype=float64 | N=1K | pending C#&#10;10. reshape 2D-&gt;1D | dtype=float64 | N=1K | pending C#"></span>
    </div>
    <div class="legend-grid">
      <span class="legend-item"><span class="legend-swatch s-faster"></span>929 faster, 1.00-2150.69x</span>
      <span class="legend-item"><span class="legend-swatch s-close"></span>434 close, 0.50-1.00x</span>
      <span class="legend-item"><span class="legend-swatch s-slower"></span>291 slower, 0.20-0.50x</span>
      <span class="legend-item"><span class="legend-swatch s-much"></span>128 much slower, &lt;0.20x</span>
      <span class="legend-item"><span class="legend-swatch s-nodata"></span>69 no data</span>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Suite Scoreboard</h2>
      <p class="section-note">Geomean across credible rows. Parity marker is 1.0x.</p>
    </div>
    <div class="bar-table">
      <div class="bar-row">
        <span class="bar-label">Statistics</span>
        <span class="bar-track"><span class="bar-fill" style="--w:89.6%; --tone:var(--good)"></span></span>
        <span class="bar-score metric-good">2.24x</span>
        <span class="bar-count">39 / 10</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Reduction</span>
        <span class="bar-track"><span class="bar-fill" style="--w:72.4%; --tone:var(--good)"></span></span>
        <span class="bar-score metric-good">1.81x</span>
        <span class="bar-count">385 / 110</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Broadcasting</span>
        <span class="bar-track"><span class="bar-fill" style="--w:44.0%; --tone:var(--good)"></span></span>
        <span class="bar-score metric-good">1.10x</span>
        <span class="bar-count">3 / 0</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Sorting</span>
        <span class="bar-track"><span class="bar-fill" style="--w:43.2%; --tone:var(--good)"></span></span>
        <span class="bar-score metric-good">1.08x</span>
        <span class="bar-count">23 / 13</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Creation</span>
        <span class="bar-track"><span class="bar-fill" style="--w:41.6%; --tone:var(--good)"></span></span>
        <span class="bar-score metric-good">1.04x</span>
        <span class="bar-count">23 / 21</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Unary</span>
        <span class="bar-track"><span class="bar-fill" style="--w:33.2%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.83x</span>
        <span class="bar-count">93 / 120</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Arithmetic</span>
        <span class="bar-track"><span class="bar-fill" style="--w:32.0%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.80x</span>
        <span class="bar-count">145 / 197</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Selection</span>
        <span class="bar-track"><span class="bar-fill" style="--w:30.4%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.76x</span>
        <span class="bar-count">2 / 4</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Logic</span>
        <span class="bar-track"><span class="bar-fill" style="--w:27.2%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.68x</span>
        <span class="bar-count">14 / 25</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Comparison</span>
        <span class="bar-track"><span class="bar-fill" style="--w:25.6%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.64x</span>
        <span class="bar-count">17 / 31</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Linear algebra</span>
        <span class="bar-track"><span class="bar-fill" style="--w:22.8%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.57x</span>
        <span class="bar-count">2 / 6</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Bitwise</span>
        <span class="bar-track"><span class="bar-fill" style="--w:21.6%; --tone:var(--near)"></span></span>
        <span class="bar-score metric-near">0.54x</span>
        <span class="bar-count">45 / 68</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Manipulation</span>
        <span class="bar-track"><span class="bar-fill" style="--w:16.0%; --tone:var(--slow)"></span></span>
        <span class="bar-score metric-slow">0.40x</span>
        <span class="bar-count">1 / 1</span>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Dtype Heatmap</h2>
      <p class="section-note">Credible operation-matrix rows by dtype and cache tier</p>
    </div>
    <div class="dtype-carousel" data-dtype-carousel>
      <div class="dtype-tabs" role="tablist" aria-label="Dtype heatmap views">
        <button class="dtype-tab is-active" type="button" role="tab" id="dtype-tab-overall" aria-controls="dtype-panel-overall" aria-selected="true" data-panel="overall">Geomean</button>
        <button class="dtype-tab" type="button" role="tab" id="dtype-tab-n1k" aria-controls="dtype-panel-n1k" aria-selected="false" tabindex="-1" data-panel="n1k">1K</button>
        <button class="dtype-tab" type="button" role="tab" id="dtype-tab-n100k" aria-controls="dtype-panel-n100k" aria-selected="false" tabindex="-1" data-panel="n100k">100K</button>
        <button class="dtype-tab" type="button" role="tab" id="dtype-tab-n10m" aria-controls="dtype-panel-n10m" aria-selected="false" tabindex="-1" data-panel="n10m">10M</button>
      </div>
      <div class="dtype-panel is-active" id="dtype-panel-overall" role="tabpanel" aria-labelledby="dtype-tab-overall">
        <div class="dtype-cell heat-best" style="--heat-width:89%"><span class="dtype-name">uint8</span><span class="dtype-score">2.00x</span><span class="dtype-count">61 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:82%"><span class="dtype-name">int8</span><span class="dtype-score">1.84x</span><span class="dtype-count">63 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:64%"><span class="dtype-name">uint16</span><span class="dtype-score">1.44x</span><span class="dtype-count">65 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:62%"><span class="dtype-name">int16</span><span class="dtype-score">1.39x</span><span class="dtype-count">66 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:57%"><span class="dtype-name">float16</span><span class="dtype-score">1.28x</span><span class="dtype-count">224 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:48%"><span class="dtype-name">uint32</span><span class="dtype-score">1.09x</span><span class="dtype-count">65 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:46%"><span class="dtype-name">complex128</span><span class="dtype-score">1.04x</span><span class="dtype-count">64 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:45%"><span class="dtype-name">float32</span><span class="dtype-score">1.01x</span><span class="dtype-count">233 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:42%"><span class="dtype-name">float64</span><span class="dtype-score">0.95x</span><span class="dtype-count">257 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:41%"><span class="dtype-name">int32</span><span class="dtype-score">0.93x</span><span class="dtype-count">111 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:33%"><span class="dtype-name">int64</span><span class="dtype-score">0.75x</span><span class="dtype-count">118 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:33%"><span class="dtype-name">uint64</span><span class="dtype-score">0.74x</span><span class="dtype-count">63 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-slow" style="--heat-width:13%"><span class="dtype-name">bool</span><span class="dtype-score">0.30x</span><span class="dtype-count">8 rows</span><span class="dtype-meter"><span></span></span></div>
      </div>
      <div class="dtype-panel" id="dtype-panel-n1k" role="tabpanel" aria-labelledby="dtype-tab-n1k" hidden>
        <div class="dtype-cell heat-best" style="--heat-width:74%"><span class="dtype-name">float16</span><span class="dtype-score">1.67x</span><span class="dtype-count">70 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:72%"><span class="dtype-name">uint64</span><span class="dtype-score">1.61x</span><span class="dtype-count">3 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:64%"><span class="dtype-name">float32</span><span class="dtype-score">1.44x</span><span class="dtype-count">41 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:56%"><span class="dtype-name">float64</span><span class="dtype-score">1.26x</span><span class="dtype-count">45 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:56%"><span class="dtype-name">uint8</span><span class="dtype-score">1.26x</span><span class="dtype-count">2 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:41%"><span class="dtype-name">int8</span><span class="dtype-score">0.92x</span><span class="dtype-count">4 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:36%"><span class="dtype-name">complex128</span><span class="dtype-score">0.82x</span><span class="dtype-count">16 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:26%"><span class="dtype-name">int64</span><span class="dtype-score">0.59x</span><span class="dtype-count">14 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:25%"><span class="dtype-name">int16</span><span class="dtype-score">0.56x</span><span class="dtype-count">6 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-slow" style="--heat-width:22%"><span class="dtype-name">uint32</span><span class="dtype-score">0.49x</span><span class="dtype-count">5 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-slow" style="--heat-width:19%"><span class="dtype-name">int32</span><span class="dtype-score">0.43x</span><span class="dtype-count">13 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-slow" style="--heat-width:19%"><span class="dtype-name">uint16</span><span class="dtype-score">0.43x</span><span class="dtype-count">5 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-empty" style="--heat-width:0%"><span class="dtype-name">bool</span><span class="dtype-score">n/a</span><span class="dtype-count">0 rows</span><span class="dtype-meter"><span></span></span></div>
      </div>
      <div class="dtype-panel" id="dtype-panel-n100k" role="tabpanel" aria-labelledby="dtype-tab-n100k" hidden>
        <div class="dtype-cell heat-best" style="--heat-width:97%"><span class="dtype-name">uint8</span><span class="dtype-score">2.19x</span><span class="dtype-count">29 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:92%"><span class="dtype-name">int8</span><span class="dtype-score">2.07x</span><span class="dtype-count">29 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:75%"><span class="dtype-name">uint16</span><span class="dtype-score">1.69x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:74%"><span class="dtype-name">int16</span><span class="dtype-score">1.66x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:53%"><span class="dtype-name">complex128</span><span class="dtype-score">1.19x</span><span class="dtype-count">24 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:49%"><span class="dtype-name">uint32</span><span class="dtype-score">1.10x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:48%"><span class="dtype-name">float16</span><span class="dtype-score">1.08x</span><span class="dtype-count">77 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:37%"><span class="dtype-name">int32</span><span class="dtype-score">0.84x</span><span class="dtype-count">49 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:30%"><span class="dtype-name">float64</span><span class="dtype-score">0.68x</span><span class="dtype-count">105 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:28%"><span class="dtype-name">int64</span><span class="dtype-score">0.63x</span><span class="dtype-count">52 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:28%"><span class="dtype-name">float32</span><span class="dtype-score">0.62x</span><span class="dtype-count">96 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:24%"><span class="dtype-name">uint64</span><span class="dtype-score">0.54x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-bad" style="--heat-width:6%"><span class="dtype-name">bool</span><span class="dtype-score">0.12x</span><span class="dtype-count">4 rows</span><span class="dtype-meter"><span></span></span></div>
      </div>
      <div class="dtype-panel" id="dtype-panel-n10m" role="tabpanel" aria-labelledby="dtype-tab-n10m" hidden>
        <div class="dtype-cell heat-best" style="--heat-width:84%"><span class="dtype-name">uint8</span><span class="dtype-score">1.89x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:80%"><span class="dtype-name">int8</span><span class="dtype-score">1.80x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-best" style="--heat-width:67%"><span class="dtype-name">uint16</span><span class="dtype-score">1.51x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:63%"><span class="dtype-name">float32</span><span class="dtype-score">1.41x</span><span class="dtype-count">96 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:62%"><span class="dtype-name">int16</span><span class="dtype-score">1.39x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:56%"><span class="dtype-name">int32</span><span class="dtype-score">1.26x</span><span class="dtype-count">49 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:55%"><span class="dtype-name">uint32</span><span class="dtype-score">1.23x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:53%"><span class="dtype-name">float16</span><span class="dtype-score">1.20x</span><span class="dtype-count">77 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:51%"><span class="dtype-name">float64</span><span class="dtype-score">1.15x</span><span class="dtype-count">107 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-good" style="--heat-width:47%"><span class="dtype-name">complex128</span><span class="dtype-score">1.05x</span><span class="dtype-count">24 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:43%"><span class="dtype-name">int64</span><span class="dtype-score">0.97x</span><span class="dtype-count">52 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:42%"><span class="dtype-name">uint64</span><span class="dtype-score">0.94x</span><span class="dtype-count">30 rows</span><span class="dtype-meter"><span></span></span></div>
        <div class="dtype-cell heat-near" style="--heat-width:34%"><span class="dtype-name">bool</span><span class="dtype-score">0.77x</span><span class="dtype-count">4 rows</span><span class="dtype-meter"><span></span></span></div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Subsystem Signals</h2>
      <p class="section-note">Result models that the op matrix cannot express</p>
    </div>
    <div class="story-grid">
      <article class="story-card">
        <h3>NDIter</h3>
        <strong class="metric-good">1.18x</strong>
        <p>Iterator operation geomean. Strong reductions and dtype loops, with copy/cast and index math still visible as overhead canaries.</p>
      </article>
      <article class="story-card">
        <h3>Layout</h3>
        <strong class="metric-near">0.50-1.80x</strong>
        <p>Layout scans expose the real split: large elementwise wins, while strided/broadcast reductions and decimal sums trail.</p>
      </article>
      <article class="story-card">
        <h3>Cast</h3>
        <strong class="metric-good">1,439 wins</strong>
        <p>The full src-to-dst astype grid is broadly ahead. Remaining lag clusters around same-type diagonal copy and bool conversion cases.</p>
      </article>
      <article class="story-card">
        <h3>Fusion</h3>
        <strong class="metric-good">4.16x</strong>
        <p>The best fixed expression speedup for fused <code>np.evaluate</code> over NumSharp's unfused chain; broadcast fusion reaches 3.60x.</p>
      </article>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Optimization Priorities</h2>
      <p class="section-note">Current optimization priorities from the latest snapshot</p>
    </div>
    <div class="priority-grid priority-grid-compact">
      <article class="priority-card priority-compact">
        <ol class="priority-list priority-list-compact">
          <li><strong>Shift kernels:</strong> vectorize 100K int left/right shift.</li>
          <li><strong>Bool bitwise:</strong> lift <code>invert</code>, <code>&amp;</code>, <code>|</code>, <code>^</code>.</li>
          <li><strong>Decimal reduce:</strong> fix broadcast/sliced sum axis cliffs.</li>
          <li><strong>i32 broadcast:</strong> repair stride-0 axis sum.</li>
          <li><strong>f64 sum:</strong> close the 100K reduction gap.</li>
          <li><strong>Float predicates:</strong> accelerate 100K <code>isnan/isinf/isfinite</code>.</li>
          <li><strong>f32 add/mul:</strong> improve 100K scalar/literal paths.</li>
          <li><strong>f64 add/mul:</strong> improve 100K scalar/literal paths.</li>
          <li><strong>Float abs/neg:</strong> remove 100K C/strided cliffs.</li>
          <li><strong>Rounding:</strong> tighten 100K f32/f64 floor/ceil/trunc.</li>
          <li><strong>Exp/log:</strong> reduce f32/f64 mid-tier overhead.</li>
          <li><strong>f16 unary:</strong> lift sign/math scalar fallback.</li>
          <li><strong>Int mean:</strong> improve 10M int64/uint64 mean.</li>
          <li><strong>Linear algebra:</strong> revisit large f64 matmul/dot.</li>
          <li><strong>Flatten:</strong> fix the 100K copy/cast trough.</li>
          <li><strong>Astype small:</strong> reduce scalar/1K setup cost.</li>
          <li><strong>Ravel T:</strong> close transpose ravel mid-size gap.</li>
          <li><strong>less-&gt;bool:</strong> specialize bool output loops.</li>
          <li><strong>Index math:</strong> speed scalar/1K unravel/ravel-multi.</li>
          <li><strong>NDIter copy:</strong> raise 100K copy/cast geomean.</li>
          <li><strong>Chunk width:</strong> optimize tiny inner width dispatch.</li>
          <li><strong>f16 operands:</strong> improve strided/reversed/broadcast cases.</li>
          <li><strong>Cast diagonal:</strong> speed same-type copy cells.</li>
          <li><strong>Cast bool:</strong> clean remaining <code>* -&gt; bool</code> losses.</li>
          <li><strong>Fusion:</strong> broaden fused-expression coverage.</li>
        </ol>
      </article>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>Full Reports</h2>
      <p class="section-note">Detailed tables remain available for traceability</p>
    </div>
    <div class="link-strip">
      <a class="dash-link" href="../../../benchmark/history/latest/MANIFEST.md">Snapshot manifest</a>
      <a class="dash-link" href="../../../benchmark/history/latest/benchmark-report.md">Unified report</a>
      <a class="dash-link" href="../../../benchmark/history/latest/cast_results.md">Cast matrix</a>
      <a class="dash-link" href="../../../benchmark/history/latest/nditer_results.md">NDIter results</a>
      <a class="dash-link" href="../../../benchmark/history/latest/layout_results.md">Layout matrix</a>
      <a class="dash-link" href="../../../benchmark/history/latest/operand_results.md">Operand layouts</a>
      <a class="dash-link" href="../../../benchmark/history/latest/fusion_results.md">Fusion results</a>
      <a class="dash-link" href="il-generation.md">IL generation</a>
    </div>
  </section>
</div>

<script src="https://unpkg.com/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://unpkg.com/tippy.js@6.3.7/dist/tippy.umd.min.js"></script>

<script>
(() => {
  document.querySelectorAll("[data-dtype-carousel]").forEach((root) => {
    const tabs = Array.from(root.querySelectorAll(".dtype-tab"));
    const panels = Array.from(root.querySelectorAll(".dtype-panel"));
    const primaryText = root.querySelectorAll(".dtype-cell, .dtype-name, .dtype-score");
    const mutedText = root.querySelectorAll(".dtype-count");

    const applyThemeText = () => {
      primaryText.forEach((node) => {
        node.style.setProperty("color", "var(--heat-text)", "important");
        node.style.setProperty("-webkit-text-fill-color", "var(--heat-text)", "important");
      });

      mutedText.forEach((node) => {
        node.style.setProperty("color", "var(--heat-muted-text)", "important");
        node.style.setProperty("-webkit-text-fill-color", "var(--heat-muted-text)", "important");
      });
    };

    applyThemeText();
    new MutationObserver(applyThemeText).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-bs-theme"]
    });

    const showPanel = (name, focusTab = false) => {
      tabs.forEach((tab) => {
        const isActive = tab.dataset.panel === name;
        tab.classList.toggle("is-active", isActive);
        tab.setAttribute("aria-selected", String(isActive));
        tab.tabIndex = isActive ? 0 : -1;
        if (isActive && focusTab) {
          tab.focus();
        }
      });

      panels.forEach((panel) => {
        const isActive = panel.id === `dtype-panel-${name}`;
        panel.classList.toggle("is-active", isActive);
        panel.hidden = !isActive;
      });
    };

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => showPanel(tab.dataset.panel));
      tab.addEventListener("keydown", (event) => {
        const isForward = event.key === "ArrowRight" || event.key === "ArrowDown";
        const isBackward = event.key === "ArrowLeft" || event.key === "ArrowUp";
        const isHome = event.key === "Home";
        const isEnd = event.key === "End";

        if (!isForward && !isBackward && !isHome && !isEnd) {
          return;
        }

        event.preventDefault();
        let nextIndex = index;
        if (isForward) nextIndex = (index + 1) % tabs.length;
        if (isBackward) nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (isHome) nextIndex = 0;
        if (isEnd) nextIndex = tabs.length - 1;

        showPanel(tabs[nextIndex].dataset.panel, true);
      });
    });
  });

  const reportUrl = new URL("data/benchmark-report.json", window.location.href);
  const topCount = 25;
  const tooltipHoverDelayMs = 5000;
  let tooltipId = 0;
  const bandDefinitions = [
    { className: "band-faster-100x", title: "100x or faster", min: 100, sort: "desc" },
    { className: "band-faster-20", title: "20x to 100x faster", min: 20, max: 100, sort: "desc" },
    { className: "band-faster-10", title: "10x to 20x faster", min: 10, max: 20, sort: "desc" },
    { className: "band-faster-5", title: "5x to 10x faster", min: 5, max: 10, sort: "desc" },
    { className: "band-faster-2", title: "2x to 5x faster", min: 2, max: 5, sort: "desc" },
    { className: "band-faster-125", title: "1.25x to 2x faster", min: 1.25, max: 2, sort: "desc" },
    { className: "band-faster-100", title: "1.00x to 1.25x faster", min: 1, max: 1.25, sort: "desc" },
    { className: "band-close-090", title: "0.90x to 1.00x", min: 0.9, max: 1, sort: "desc" },
    { className: "band-close-075", title: "0.75x to 0.90x", min: 0.75, max: 0.9, sort: "desc" },
    { className: "band-close-050", title: "0.50x to 0.75x", min: 0.5, max: 0.75, sort: "desc" },
    { className: "band-slower-040", title: "0.40x to 0.50x", min: 0.4, max: 0.5, sort: "asc" },
    { className: "band-slower-030", title: "0.30x to 0.40x", min: 0.3, max: 0.4, sort: "asc" },
    { className: "band-slower-020", title: "0.20x to 0.30x", min: 0.2, max: 0.3, sort: "asc" },
    { className: "band-much-010", title: "0.10x to 0.20x", min: 0.1, max: 0.2, sort: "asc" },
    { className: "band-much-005", title: "0.05x to 0.10x", min: 0.05, max: 0.1, sort: "asc" },
    { className: "band-much-000", title: "Below 0.05x", max: 0.05, sort: "asc" },
    { className: "band-nodata", title: "No data", noData: true, sort: "index" }
  ];

  const tierByPanel = {
    "dtype-panel-overall": null,
    "dtype-panel-n1k": 1000,
    "dtype-panel-n100k": 100000,
    "dtype-panel-n10m": 10000000
  };

  const escapeHtml = (value) => String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

  const numberOrNull = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };

  const formatRatio = (value) => Number.isFinite(value) ? `${value.toFixed(value >= 10 ? 2 : 3)}x` : "n/a";
  const formatMs = (value) => {
    if (!Number.isFinite(value)) return "n/a";
    if (value > 0 && value < 0.001) return "<0.001";
    if (value < 0.1) return value.toFixed(4);
    if (value < 10) return value.toFixed(3);
    return value.toFixed(2);
  };

  const formatN = (value) => {
    if (!Number.isFinite(value)) return "n/a";
    if (value >= 1000000 && value % 1000000 === 0) return `${value / 1000000}M`;
    if (value >= 1000 && value % 1000 === 0) return `${value / 1000}K`;
    return String(value);
  };

  const formatBandLabel = (row) => {
    const ratio = row.ratio;
    if (!Number.isFinite(ratio)) return "No data";
    if (ratio >= 100) return "100x or faster";
    if (ratio >= 20) return "20x to 100x faster";
    if (ratio >= 10) return "10x to 20x faster";
    if (ratio >= 5) return "5x to 10x faster";
    if (ratio >= 2) return "2x to 5x faster";
    if (ratio >= 1.25) return "1.25x to 2x faster";
    if (ratio >= 1) return "1.00x to 1.25x faster";
    if (ratio >= 0.9) return "0.90x to 1.00x";
    if (ratio >= 0.75) return "0.75x to 0.90x";
    if (ratio >= 0.5) return "0.50x to 0.75x";
    if (ratio >= 0.4) return "0.40x to 0.50x";
    if (ratio >= 0.3) return "0.30x to 0.40x";
    if (ratio >= 0.2) return "0.20x to 0.30x";
    if (ratio >= 0.1) return "0.10x to 0.20x";
    if (ratio >= 0.05) return "0.05x to 0.10x";
    return "Below 0.05x";
  };

  const performanceClass = (row) => {
    const ratio = row.ratio;
    if (!Number.isFinite(ratio)) return "ns-perf-empty";
    if (ratio >= 2) return "ns-perf-excellent";
    if (ratio >= 1) return "ns-perf-good";
    if (ratio >= 0.5) return "ns-perf-near";
    if (ratio >= 0.2) return "ns-perf-slow";
    return "ns-perf-bad";
  };

  const renderRatio = (row) => `
    <span class="ns-perf-ratio ${performanceClass(row)}">
      <span class="ns-perf-dot" aria-hidden="true"></span>
      <span>${formatRatio(row.ratio)}</span>
    </span>`;

  const sortRows = (rows, direction = "desc") => {
    const copy = rows.slice();
    if (direction === "index") {
      return copy.sort((a, b) => a.index - b.index);
    }

    return copy.sort((a, b) => {
      const left = a.ratio ?? Number.NEGATIVE_INFINITY;
      const right = b.ratio ?? Number.NEGATIVE_INFINITY;
      return direction === "asc" ? left - right : right - left;
    });
  };

  const buildDataTable = (visibleRows, isNoData = false) => {
    const body = visibleRows.map((row, index) => isNoData
      ? `<tr>
          <td class="num">${index + 1}</td>
          <td class="op">${escapeHtml(row.operation)}</td>
          <td>${escapeHtml(row.suite)}</td>
          <td>${escapeHtml(row.dtype)}</td>
          <td class="num">${formatN(row.n)}</td>
        </tr>`
      : `<tr>
          <td class="num">${index + 1}</td>
          <td class="op">${escapeHtml(row.operation)}</td>
          <td>${escapeHtml(row.suite)}</td>
          <td>${escapeHtml(row.dtype)}</td>
          <td class="num">${formatN(row.n)}</td>
          <td class="num">${renderRatio(row)}</td>
          <td class="num">${formatMs(row.numpyMs)}</td>
          <td class="num">${formatMs(row.numSharpMs)}</td>
          <td>${escapeHtml(formatBandLabel(row))}</td>
        </tr>`).join("");

    const header = isNoData
      ? `<tr><th class="num">#</th><th>Operation</th><th>Suite</th><th>Dtype</th><th class="num">N</th></tr>`
      : `<tr><th class="num">#</th><th>Operation</th><th>Suite</th><th>Dtype</th><th class="num">N</th><th class="num">Ratio</th><th class="num">NumPy ms</th><th class="num">NumSharp ms</th><th>Band</th></tr>`;

    return `
      <div class="ns-tip-scroll">
        <table class="ns-tip-table">
          <thead>${header}</thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;
  };

  const buildTooltipTable = (title, rows, options = {}) => {
    const isNoData = Boolean(options.noData);
    const sortedRows = isNoData ? sortRows(rows, "index") : rows;
    const visibleRows = sortedRows.slice(0, topCount);
    const subtitle = isNoData
      ? `${visibleRows.length} of ${rows.length} pending rows`
      : `${Math.min(topCount, rows.length)} best and ${Math.min(topCount, rows.length)} worst of ${rows.length} rows · ratio = NumPy_ms / NumSharp_ms · higher is better`;

    if (visibleRows.length === 0) {
      return `
        <div class="ns-tip">
          <div class="ns-tip-title">${escapeHtml(title)}</div>
          <div class="ns-tip-empty">No matching benchmark rows.</div>
        </div>`;
    }

    if (isNoData) {
      return `
        <div class="ns-tip">
          <div class="ns-tip-title">${escapeHtml(title)}</div>
          <div class="ns-tip-meta">${escapeHtml(subtitle)}</div>
          ${buildDataTable(visibleRows, true)}
        </div>`;
    }

    const bestRows = sortRows(rows, "desc").slice(0, topCount);
    const worstRows = sortRows(rows, "asc").slice(0, topCount);
    const id = `ns-tip-${++tooltipId}`;

    return `
      <div class="ns-tip">
        <div class="ns-tip-title">${escapeHtml(title)}</div>
        <div class="ns-tip-meta">${escapeHtml(subtitle)}</div>
        <div class="ns-tip-switch" role="tablist" aria-label="${escapeHtml(title)} ranking">
          <button class="ns-tip-switch-button is-active" type="button" role="tab" aria-selected="true" aria-controls="${id}-best" data-tip-tab-target="${id}-best">Best performers</button>
          <button class="ns-tip-switch-button" type="button" role="tab" aria-selected="false" aria-controls="${id}-worst" data-tip-tab-target="${id}-worst" tabindex="-1">Worst performers</button>
        </div>
        <div class="ns-tip-panel is-active" id="${id}-best" role="tabpanel">
          <div class="ns-tip-panel-note">Highest ratio in this grouping</div>
          ${buildDataTable(bestRows)}
        </div>
        <div class="ns-tip-panel" id="${id}-worst" role="tabpanel" hidden>
          <div class="ns-tip-panel-note">Lowest ratio in this grouping</div>
          ${buildDataTable(worstRows)}
        </div>
      </div>`;
  };

  const attachTooltip = (element, html, plainLabel) => {
    element.dataset.benchmarkTooltip = "true";
    element.dataset.benchmarkTooltipHtml = html;
    element.setAttribute("aria-label", plainLabel);
    if (!element.hasAttribute("tabindex")) {
      element.tabIndex = 0;
    }
  };

  const bandForElement = (element) => bandDefinitions.find((band) => element.classList.contains(band.className));

  const rowMatchesBand = (row, band) => {
    if (band.noData) {
      return row.status === "no_data" || row.ratio === null;
    }

    if (row.ratio === null) return false;
    if (band.min !== undefined && row.ratio < band.min) return false;
    if (band.max !== undefined && row.ratio >= band.max) return false;
    return true;
  };

  const initializeBenchmarkTooltips = async () => {
    let rawRows;
    try {
      const response = await fetch(reportUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      rawRows = await response.json();
    } catch (error) {
      document.querySelectorAll("[data-tip]").forEach((element) => {
        element.setAttribute("title", element.getAttribute("data-tip").replace(/\n/g, " | "));
      });
      console.warn("Benchmark tooltip data could not be loaded", error);
      return;
    }

    const rows = rawRows.map((row, index) => ({
      index,
      operation: row.operation || "",
      suite: row.suite || "",
      category: row.category || "",
      dtype: row.dtype || "",
      n: numberOrNull(row.n),
      ratio: numberOrNull(row.ratio),
      pctNumPy: numberOrNull(row.pct_numpy),
      numpyMs: numberOrNull(row.numpy_ms),
      numSharpMs: numberOrNull(row.numsharp_ms),
      status: row.status || ""
    }));

    const measuredRows = rows.filter((row) => row.ratio !== null);

    document.querySelectorAll(".status-segment").forEach((element) => {
      const band = bandForElement(element);
      if (!band) return;
      const bandRows = rows.filter((row) => rowMatchesBand(row, band));
      const title = band.noData ? "No-data benchmark rows" : `${band.title} benchmark rows`;
      attachTooltip(element, buildTooltipTable(title, bandRows, { noData: band.noData }), title);
    });

    document.querySelectorAll(".bar-row").forEach((element) => {
      const suite = element.querySelector(".bar-label")?.textContent.trim();
      if (!suite) return;
      const suiteRows = measuredRows.filter((row) => row.suite === suite);
      const title = `${suite} benchmark rows`;
      attachTooltip(element, buildTooltipTable(title, suiteRows), title);
    });

    document.querySelectorAll(".dtype-cell").forEach((element) => {
      const dtype = element.querySelector(".dtype-name")?.textContent.trim();
      const panelId = element.closest(".dtype-panel")?.id;
      if (!dtype || !(panelId in tierByPanel)) return;

      const tier = tierByPanel[panelId];
      const dtypeRows = measuredRows.filter((row) => {
        if (row.dtype !== dtype) return false;
        return tier === null || row.n === tier;
      });

      const title = tier === null
        ? `${dtype} benchmark rows`
        : `${dtype} at ${formatN(tier)} benchmark rows`;
      attachTooltip(element, buildTooltipTable(title, dtypeRows), title);
    });

    const targets = document.querySelectorAll("[data-benchmark-tooltip]");
    if (!window.tippy) {
      targets.forEach((element) => element.setAttribute("title", element.getAttribute("aria-label")));
      return;
    }

    const targetElements = Array.from(targets);
    const hoverTimers = new WeakMap();
    const hideTimers = new WeakMap();
    const containsNode = (container, node) => container instanceof Node && node instanceof Node && container.contains(node);

    const clearHoverTimer = (element) => {
      const timer = hoverTimers.get(element);
      if (timer) {
        window.clearTimeout(timer);
        hoverTimers.delete(element);
      }
    };

    const clearHideTimer = (element) => {
      const timer = hideTimers.get(element);
      if (timer) {
        window.clearTimeout(timer);
        hideTimers.delete(element);
      }
    };

    const closeTooltip = (element, force = false) => {
      if (!force && element.dataset.tooltipPinned === "true") return;
      clearHoverTimer(element);
      clearHideTimer(element);
      element.dataset.tooltipPinned = "false";
      element.classList.remove("is-tooltip-open");
      element._tippy?.hide();
    };

    const closeOtherTooltips = (activeElement) => {
      targetElements.forEach((element) => {
        if (element !== activeElement) closeTooltip(element, true);
      });
    };

    const openTooltip = (element, pinned = false) => {
      if (!element._tippy) return;
      closeOtherTooltips(element);
      clearHoverTimer(element);
      clearHideTimer(element);
      element.dataset.tooltipPinned = pinned ? "true" : "false";
      element.classList.add("is-tooltip-open");
      element._tippy.show();
    };

    const scheduleHoverTooltip = (element) => {
      if (element.dataset.tooltipPinned === "true") return;
      clearHoverTimer(element);
      clearHideTimer(element);
      hoverTimers.set(element, window.setTimeout(() => openTooltip(element), tooltipHoverDelayMs));
    };

    const scheduleHideTooltip = (element) => {
      if (element.dataset.tooltipPinned === "true") return;
      clearHoverTimer(element);
      clearHideTimer(element);
      hideTimers.set(element, window.setTimeout(() => closeTooltip(element), 140));
    };

    const togglePinnedTooltip = (element) => {
      if (element.dataset.tooltipPinned === "true") {
        closeTooltip(element, true);
        return;
      }

      openTooltip(element, true);
    };

    const activateTooltipTab = (button) => {
      const tip = button.closest(".ns-tip");
      const targetId = button.dataset.tipTabTarget;
      if (!tip || !targetId) return;

      tip.querySelectorAll(".ns-tip-switch-button").forEach((candidate) => {
        const isActive = candidate === button;
        candidate.classList.toggle("is-active", isActive);
        candidate.setAttribute("aria-selected", String(isActive));
        candidate.tabIndex = isActive ? 0 : -1;
      });

      tip.querySelectorAll(".ns-tip-panel").forEach((panel) => {
        const isActive = panel.id === targetId;
        panel.classList.toggle("is-active", isActive);
        panel.hidden = !isActive;
      });
    };

    const handleTooltipTabKeydown = (event) => {
      const button = event.target instanceof Element
        ? event.target.closest(".ns-tip-switch-button")
        : null;
      if (!button) return;

      const buttons = Array.from(button.closest(".ns-tip-switch")?.querySelectorAll(".ns-tip-switch-button") || []);
      if (buttons.length === 0) return;

      const currentIndex = buttons.indexOf(button);
      let nextIndex = currentIndex;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (currentIndex + 1) % buttons.length;
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (currentIndex - 1 + buttons.length) % buttons.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = buttons.length - 1;
      if (nextIndex === currentIndex) return;

      event.preventDefault();
      event.stopPropagation();
      buttons[nextIndex].focus();
      activateTooltipTab(buttons[nextIndex]);
    };

    const getTooltipScrollContainer = (target, popper) => {
      const directScroller = target instanceof Element
        ? target.closest(".ns-tip-scroll")
        : null;

      if (directScroller && containsNode(popper, directScroller)) {
        return directScroller;
      }

      return popper.querySelector(".ns-tip-panel:not([hidden]) .ns-tip-scroll")
        || popper.querySelector(".ns-tip-scroll");
    };

    const getWheelScale = (event, scroller) => {
      if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) return 16;
      if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) return scroller?.clientHeight || window.innerHeight;
      return 1;
    };

    const containTooltipWheel = (event, popper) => {
      event.preventDefault();
      event.stopPropagation();

      const scroller = getTooltipScrollContainer(event.target, popper);
      if (!scroller) return;

      const scale = getWheelScale(event, scroller);
      scroller.scrollTop += event.deltaY * scale;
      scroller.scrollLeft += event.deltaX * scale;
    };

    window.tippy(targets, {
      allowHTML: true,
      appendTo: () => document.body,
      arrow: true,
      content: (reference) => reference.dataset.benchmarkTooltipHtml,
      duration: [140, 90],
      hideOnClick: false,
      interactive: true,
      maxWidth: "none",
      onHidden(instance) {
        if (instance.reference.dataset.tooltipPinned !== "true") {
          instance.reference.classList.remove("is-tooltip-open");
        }
      },
      placement: "bottom-start",
      popperOptions: {
        modifiers: [
          { name: "flip", options: { fallbackPlacements: ["top-start", "right-start", "left-start"] } },
          { name: "preventOverflow", options: { altAxis: true, boundary: "viewport", padding: 12, tether: false } }
        ]
      },
      theme: "ns-bench",
      trigger: "manual"
    });

    targetElements.forEach((element) => {
      const instance = element._tippy;
      if (!instance) return;

      element.addEventListener("mouseenter", () => scheduleHoverTooltip(element));
      element.addEventListener("mouseleave", (event) => {
        if (containsNode(instance.popper, event.relatedTarget)) return;
        scheduleHideTooltip(element);
      });

      element.addEventListener("focusin", () => scheduleHoverTooltip(element));
      element.addEventListener("focusout", (event) => {
        if (containsNode(instance.popper, event.relatedTarget)) return;
        scheduleHideTooltip(element);
      });

      element.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        togglePinnedTooltip(element);
      });

      element.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          togglePinnedTooltip(element);
        }
        if (event.key === "Escape") {
          event.preventDefault();
          closeTooltip(element, true);
        }
      });

      instance.popper.addEventListener("mouseenter", () => clearHideTimer(element));
      instance.popper.addEventListener("mouseleave", (event) => {
        if (containsNode(element, event.relatedTarget)) return;
        scheduleHideTooltip(element);
      });
      instance.popper.addEventListener("click", (event) => {
        const button = event.target instanceof Element
          ? event.target.closest(".ns-tip-switch-button")
          : null;
        if (!button || !containsNode(instance.popper, button)) return;

        event.preventDefault();
        event.stopPropagation();
        activateTooltipTab(button);
        instance.popperInstance?.update?.();
      });
      instance.popper.addEventListener("keydown", handleTooltipTabKeydown);
      instance.popper.addEventListener("wheel", (event) => containTooltipWheel(event, instance.popper), { passive: false });
    });

    document.addEventListener("click", (event) => {
      const clickedInsideTooltip = targetElements.some((element) =>
        containsNode(element, event.target) || containsNode(element._tippy?.popper, event.target));

      if (!clickedInsideTooltip) {
        targetElements.forEach((element) => closeTooltip(element, true));
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        targetElements.forEach((element) => closeTooltip(element, true));
      }
    });
  };

  initializeBenchmarkTooltips();
})();
</script>
