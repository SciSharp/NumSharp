// Run with: node --test coverage/test_dashboard.cjs
// Execute the shipped inline script so scope/filter regressions cannot hide in a
// second implementation. The small DOM harness records rendered markup; the
// DocFX/browser check remains responsible for layout and native interactions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const markdown = fs.readFileSync(path.join(__dirname, '../docs/website-src/docs/coverage-support-dashboard.md'), 'utf8');
const inline = markdown.match(/<script>\s*([\s\S]*?)<\/script>/)[1];

function row(id, category, overrides = {}) {
  return {
    id, origin: 'numpy', name: id.split('.').at(-1), surface: 'np',
    category, kind: 'function', status: 'missing', availability: 'missing',
    in_default_scope: false, extended: true, numsharp_signatures: [],
    ...overrides,
  };
}

const fixture = [
  row('numpy.add', 'Arithmetic', { status: 'available', availability: 'exact', in_default_scope: true, extended: false }),
  row('numpy.subtract', 'Arithmetic', { status: 'partial', availability: 'exact', in_default_scope: true, extended: false }),
  row('numpy.random.Generator.normal', 'Random: generators', { surface: 'random.Generator', kind: 'method', status: 'available', availability: 'exact', disposition: 'object' }),
  row('numpy.random.Generator.integers', 'Random: generators', { surface: 'random.Generator', kind: 'method' }),
  row('numpy.add.reduce', 'Universal function objects', { surface: 'ufunc', name: 'add.reduce', kind: 'method', status: 'unsupported' }),
  row('numpy.strings.upper', 'String operations', { surface: 'strings' }),
  row('numpy.random.PCG64', 'Random bit generators', { surface: 'random', kind: 'class', status: 'available', availability: 'exact' }),
  row('numpy.pi', 'Constants', { kind: 'constant', extended: false, status: 'available', availability: 'exact' }),
  row('numpy.testing', 'Modules', { kind: 'module', extended: false }),
  row('NumSharp.NDArray.ToArray', 'Conversion extensions', { origin: 'numsharp', surface: 'ndarray', kind: 'method', extended: false, status: 'extension', availability: 'extension' }),
];

function dashboard() {
  const elements = new Map();
  const ids = [...markdown.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  for (const id of ids) {
    elements.set(id, {
      id, value: 'all', innerHTML: '', textContent: '', dataset: {}, options: [{ value: 'all' }],
      replaceChildren(first) { this.options = [first]; this.value = first.value; },
      appendChild(option) { this.options.push(option); },
      querySelectorAll() { return []; }, addEventListener() {}, scrollIntoView() {},
    });
  }
  const element = (id) => elements.get(id);
  element('cov-search').value = '';
  element('cov-scope').value = 'numpy';
  element('cov-sort').value = 'gap';
  const root = { querySelector: (selector) => element(selector.slice(1)), querySelectorAll: () => [] };
  const document = {
    getElementById: () => root,
    createElement: () => ({ value: '', textContent: '' }),
    addEventListener() {},
  };
  const context = vm.createContext({ document, window: {}, console });
  const exposed = inline.replace(/\n  initialize\(\);\s*\n\}\)\(\);\s*$/, '\n  globalThis.dashboard = { state, rowInScope, summarize, surfaceLabel, apiLabel, scopedRows, initializeMetadata, initializeFilters, renderSummary, filterRows, changeScope, resetFilters, applyPreset, tooltipContent, parityClass };\n})();');
  assert.notEqual(exposed, inline, 'test harness must replace only the initialization call');
  new vm.Script(exposed).runInContext(context);
  const api = context.dashboard;
  api.state.rows = fixture.map((item) => ({ ...item }));
  api.initializeFilters();
  api.renderSummary();
  api.filterRows();
  return { api, element };
}

test('expanded API scope includes object methods and excludes supporting exports', () => {
  const { api } = dashboard();
  assert.equal(api.scopedRows().length, 6);
  assert.equal(api.summarize(api.scopedRows()).coverage, 100 / 3);
  const all = api.summarize(api.state.rows);
  assert.equal(all.numpy, 6);
  assert.equal(all.availableApis, 2);
  assert.equal(all.coverage, 100 / 3, 'constants/classes/extensions must not inflate API availability');
  assert.equal(api.state.rows.filter((item) => api.rowInScope(item, 'catalog')).length, 9);
  assert.equal(api.state.rows.filter((item) => api.rowInScope(item, 'extended')).length, 4);
});

test('scoreboards display every scoped surface and category', () => {
  const { api, element } = dashboard();
  for (const item of api.scopedRows()) {
    assert.ok(element('cov-surface-grid').innerHTML.includes(`data-surface="${item.surface}"`));
    assert.ok(element('cov-category-grid').innerHTML.includes(`data-category="${item.category}"`));
  }
  assert.match(element('cov-surface-grid').innerHTML, /np\.random\.Generator\.\*/);
  assert.match(element('cov-surface-grid').innerHTML, /np\.ufunc\.\*/);
  assert.match(element('cov-scope-note').textContent, /6 entries across 4 surfaces and 4 capability areas/);
});

test('scope changes update scoreboards, filters, tooltip rows and explorer together', () => {
  const { api, element } = dashboard();
  element('cov-search').value = 'stale';
  element('cov-surface').value = 'strings';
  element('cov-scope').value = 'default';
  api.changeScope();
  assert.equal(api.state.filtered.length, 2);
  assert.equal(element('cov-search').value, '');
  assert.equal(element('cov-surface').value, 'all');
  assert.deepEqual(Array.from(element('cov-surface').options, (item) => item.value), ['all', 'np']);
  assert.ok(!element('cov-surface-grid').innerHTML.includes('random.Generator'));
  assert.match(element('cov-metrics').innerHTML, /1 of 2 comparable NumPy APIs/);
  assert.match(api.tooltipContent({ dataset: { tooltipGroup: 'surface:np' } }), /2 entries/);
  assert.match(element('cov-result-count').textContent, /2 of 2 entries · Headline comparison/);
});

test('card presets clear conflicting explorer filters while preserving scope', () => {
  const { api, element } = dashboard();
  element('cov-search').value = 'np.strings.upper';
  element('cov-category').value = 'String operations';
  element('cov-status').value = 'missing';
  element('cov-kind').value = 'function';
  element('cov-mapping').value = 'missing';
  api.applyPreset('surface', 'random.Generator');
  assert.equal(element('cov-scope').value, 'numpy');
  assert.equal(api.state.filtered.length, 2);
  assert.ok(api.state.filtered.every((item) => item.surface === 'random.Generator'));
  api.applyPreset('gaps');
  assert.equal(api.state.filtered.length, 3);
  assert.ok(api.state.filtered.some((item) => item.status === 'unsupported'));
  api.applyPreset('exact');
  assert.equal(api.state.filtered.length, 3, 'exact mapping is independent of partial support');
});

test('search accepts displayed np aliases and qualified object identifiers', () => {
  const { api, element } = dashboard();
  element('cov-search').value = 'np.random.Generator.normal';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.equal(api.apiLabel(api.state.filtered[0]), 'np.random.Generator.normal');
  assert.match(element('cov-detail').innerHTML, /Extended NumPy API · Object members/);
  element('cov-search').value = 'numpy.add.reduce';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.match(element('cov-results').innerHTML, /np\.add\.reduce/);
});

test('extension-only scope has no NumPy percentage and gaps do not jump scopes', () => {
  const { api, element } = dashboard();
  element('cov-scope').value = 'extensions';
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /No comparable NumPy APIs in this scope/);
  assert.match(element('cov-status-track').innerHTML, /data-status="extension"/);
  api.applyPreset('gaps');
  assert.equal(element('cov-scope').value, 'extensions');
  assert.equal(api.state.filtered.length, 0);
  api.resetFilters();
  assert.equal(api.state.filtered.length, 1);
  assert.equal(element('cov-scope').value, 'extensions');
});

test('tooltips preserve colon category names and start on a populated tab', () => {
  const { api } = dashboard();
  const category = api.tooltipContent({ dataset: { tooltipGroup: 'category:Random: generators' } });
  assert.match(category, /2 entries/);
  assert.match(category, /np\.random\.Generator\.normal/);
  const gaps = api.tooltipContent({ dataset: { tooltipGroup: 'status:unsupported' } });
  assert.match(gaps, /aria-selected="true" data-tip-tab="gaps"/);
  assert.match(gaps, /np\.add\.reduce/);
});

test('headline metadata retains the published denominator and validates it', () => {
  const { api, element } = dashboard();
  const data = { numpy_version: '2.x', numsharp_assembly_version: 'test', schema_version: 1, summary: { default_scope: { total: 2, available: 1 }, catalog_rows: fixture.length } };
  api.initializeMetadata(data);
  assert.equal(element('cov-headline-reference').textContent, 'Headline comparison: 50.0% (1/2 APIs).');
  data.summary.default_scope.total = 100;
  assert.throws(() => api.initializeMetadata(data), /does not match its row inventory/);
});

test('declared but inapplicable ufunc methods stay searchable with a scoped disclosure', () => {
  const { api, element } = dashboard();
  api.state.rows.push(row('numpy.sin.reduce', 'Universal function objects', { surface: 'ufunc', name: 'sin.reduce', kind: 'method', applicability: 'not_applicable' }));
  api.changeScope();
  assert.match(element('cov-applicability-note').textContent, /1 declared ufunc contract whose calls NumPy rejects/);
  assert.equal(api.scopedRows().length, 7, 'all declared API contracts remain countable');
  element('cov-search').value = 'np.sin.reduce';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.match(element('cov-detail').innerHTML, /NumPy rejects this ufunc method/);
  element('cov-scope').value = 'default';
  api.changeScope();
  assert.equal(element('cov-applicability-note').textContent, '');
});

test('large card grids scroll without dropping any surfaces or categories', () => {
  const { api, element } = dashboard();
  for (let index = 0; index < 13; index++) {
    api.state.rows.push(row(`numpy.family${index}.operation`, `Capability ${index}`, { surface: `family${index}` }));
  }
  api.changeScope();
  assert.equal(element('cov-surface-grid').dataset.scrollable, 'true');
  assert.equal(element('cov-category-grid').dataset.scrollable, 'true');
  assert.match(element('cov-surface-count').textContent, /scroll to explore/);
  assert.match(element('cov-category-count').textContent, /scroll to explore/);
  assert.match(element('cov-surface-grid').innerHTML, /data-surface="family12"/);
  assert.match(element('cov-category-grid').innerHTML, /data-category="Capability 12"/);
  element('cov-scope').value = 'default';
  api.changeScope();
  assert.equal(element('cov-surface-grid').dataset.scrollable, 'false');
  assert.equal(element('cov-category-grid').dataset.scrollable, 'false');
});

test('projected polynomial and masked-array groups stay merged across cards, filters and tooltips', () => {
  const { api, element } = dashboard();
  const polynomialIds = [
    'numpy.polyfit', 'numpy.polyval', 'numpy.poly1d.roots',
    'numpy.polynomial.polynomial.polyval', 'numpy.polynomial.chebyshev.chebval',
    'numpy.polynomial.legendre.legval', 'numpy.polynomial.hermite.hermval',
    'numpy.polynomial.hermite_e.hermeval', 'numpy.polynomial.laguerre.lagval',
    'numpy.polynomial.chebyshev.Chebyshev.deriv',
    'numpy.polynomial.polyutils.as_series',
  ];
  api.state.rows = [
    ...polynomialIds.map((id, index) => row(id, 'Polynomials', {
      surface: 'polynomial', in_default_scope: index < 2, extended: index >= 2,
    })),
    row('numpy.polynomial.polynomial.Polynomial', 'Polynomials', { surface: 'polynomial', kind: 'class' }),
    // Object first: the shared label must not be inferred as np.ma.MaskedArray.*.
    row('numpy.ma.MaskedArray.sum', 'Masked arrays', { surface: 'ma', kind: 'method' }),
    row('numpy.ma.masked_where', 'Masked arrays', { surface: 'ma' }),
  ];
  for (const scope of ['numpy', 'catalog', 'all']) {
    element('cov-scope').value = scope;
    api.changeScope();
    assert.deepEqual(Array.from(element('cov-surface').options, (item) => item.value), ['all', 'ma', 'polynomial']);
    assert.deepEqual(Array.from(element('cov-category').options, (item) => item.value), ['all', 'Masked arrays', 'Polynomials']);
    assert.equal((element('cov-surface-grid').innerHTML.match(/data-surface="polynomial"/g) || []).length, 1);
    assert.equal((element('cov-surface-grid').innerHTML.match(/data-surface="ma"/g) || []).length, 1);
    assert.equal((element('cov-category-grid').innerHTML.match(/data-category="Polynomials"/g) || []).length, 1);
    assert.match(element('cov-surface-grid').innerHTML, /<code>np\.polynomial\.\*<\/code>/);
    assert.match(element('cov-surface-grid').innerHTML, /<code>np\.ma\.\*<\/code>/);
    assert.doesNotMatch(element('cov-surface-grid').innerHTML, /data-surface="(?:ma\.MaskedArray|poly1d|polynomial\.)/);
    api.applyPreset('surface', 'polynomial');
    const expectedPolynomials = polynomialIds.length + (scope === 'numpy' ? 0 : 1);
    assert.equal(api.state.filtered.length, expectedPolynomials);
    const polynomialTooltip = api.tooltipContent({ dataset: { tooltipGroup: 'surface:polynomial' } });
    assert.ok(polynomialTooltip.includes(`${expectedPolynomials} entries`));
    assert.match(polynomialTooltip, /np\.polyfit/);
    assert.match(polynomialTooltip, /np\.polynomial\.chebyshev\.Chebyshev\.deriv/);
    api.applyPreset('category', 'Polynomials');
    assert.equal(api.state.filtered.length, expectedPolynomials);
    api.applyPreset('surface', 'ma');
    assert.equal(api.state.filtered.length, 2);
    const maskedTooltip = api.tooltipContent({ dataset: { tooltipGroup: 'surface:ma' } });
    assert.match(maskedTooltip, /2 entries/);
    assert.match(maskedTooltip, /np\.ma\.MaskedArray\.sum/);
    assert.match(maskedTooltip, /np\.ma\.masked_where/);
  }
  element('cov-scope').value = 'default';
  api.changeScope();
  assert.deepEqual(Array.from(element('cov-surface').options, (item) => item.value), ['all', 'polynomial']);
  assert.equal(api.state.filtered.length, 2);
  assert.match(element('cov-results').innerHTML, /np\.polyfit/);
  assert.doesNotMatch(element('cov-results').innerHTML, /np\.polynomial\.polyfit/);
});

// ---- NumPy-oracle parity evidence (generator 1.10+: rows carry an `oracle` record) ----

function direct(contracts, extra = {}) {
  return {
    status: 'direct', contracts, error_contracts: 2, sources: { fuzz: contracts }, keys: ['sqrt'],
    files: ['unary.jsonl'], dtypes: ['float32', 'float64'], layouts: 26, pinned: false,
    pinned_contracts: 0, pins: [], ...extra,
  };
}

function aliasVia(id, contracts) {
  return {
    status: 'alias', contracts: 0, error_contracts: 0, sources: {}, keys: [], files: [], dtypes: [],
    layouts: 0, pinned: false, pinned_contracts: 0, pins: [],
    via: { id, contracts, rule: 'reviewed', reason: 'np.amax is an alias of np.max.', pinned: false },
  };
}

const oracleRows = () => [
  row('numpy.sqrt', 'Math', { status: 'available', availability: 'exact', in_default_scope: true, extended: false, oracle: direct(1200) }),
  row('numpy.exp', 'Math', { status: 'available', availability: 'exact', in_default_scope: true, extended: false,
    oracle: direct(40, { keys: ['exp'], files: ['unary.jsonl'], pinned: true, pinned_contracts: 40, pins: ['host-libm'] }) }),
  row('numpy.amax', 'Reductions', { status: 'available', availability: 'exact', in_default_scope: true, extended: false, oracle: aliasVia('numpy.max', 1648) }),
  row('numpy.histogram', 'Statistics', { status: 'available', availability: 'exact', in_default_scope: true, extended: false }),
  row('numpy.nextafter', 'Math', { status: 'missing', availability: 'missing', in_default_scope: true, extended: false }),
  row('numpy.ma.MaskedArray.flatten', 'Masked arrays', { surface: 'ma', kind: 'method', oracle: direct(408, { keys: ['ma.flatten'], files: ['ma_manip.jsonl'] }) }),
  row('numsharp.np.evaluate', 'NumSharp-only APIs', { origin: 'numsharp', kind: 'method', extended: false, status: 'extension', availability: 'extension', oracle: direct(29563, { keys: ['evaluate'], files: ['evaluate.jsonl'] }) }),
];

test('oracle parity classes mirror the generator classify() for every status', () => {
  const { api } = dashboard();
  const rows = oracleRows();
  const classes = Object.fromEntries(rows.map((item) => [item.id, api.parityClass(item)]));
  assert.deepEqual(classes, {
    'numpy.sqrt': 'verified-direct',
    'numpy.exp': 'verified-direct',
    'numpy.amax': 'verified-alias',
    'numpy.histogram': 'unverified',
    'numpy.nextafter': 'none',
    'numpy.ma.MaskedArray.flatten': 'uncredited',
    'numsharp.np.evaluate': 'verified-direct',
  });
  // Partial rows are credited too; a row the catalog calls missing is never "verified".
  assert.equal(api.parityClass(row('numpy.shape', 'X', { status: 'partial' })), 'unverified');
  assert.equal(api.parityClass(row('numpy.shape', 'X', { status: 'partial', oracle: direct(3) })), 'verified-direct');
  assert.equal(api.parityClass(row('numpy.x', 'X', { status: 'unsupported', oracle: direct(3) })), 'uncredited');
  const counts = api.summarize(rows);
  assert.equal(counts.oracle, 4, 'the NumSharp-only row counts as verified, the uncredited row does not');
  assert.equal(counts.oracleApis, 3, 'only comparable NumPy APIs enter the oracle percentage');
  assert.equal(counts.oracleDirect, 3);
  assert.equal(counts.oracleAlias, 1);
  assert.equal(counts.oracleUnverified, 1);
  assert.equal(counts.oracleUncredited, 1);
  assert.equal(counts.oracleCoverage, 50, '3 verified of 6 comparable APIs');
});

test('oracle metric, filter, badges and search follow the selected scope', () => {
  const { api, element } = dashboard();
  api.state.rows = oracleRows();
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /Oracle-verified parity/);
  assert.match(element('cov-metrics').innerHTML, /50\.0%/);
  assert.match(element('cov-metrics').innerHTML, /3 of 6 replay committed NumPy oracle contracts · 1 available without/);
  assert.match(element('cov-metrics').innerHTML, /data-metric-action="oracle"/);
  assert.deepEqual(Array.from(element('cov-oracle').options, (item) => item.value),
    ['all', 'verified', 'verified-direct', 'verified-alias', 'uncredited', 'unverified', 'none']);
  assert.match(element('cov-surface-grid').innerHTML, /3 oracle-verified/);
  api.applyPreset('oracle');
  assert.equal(element('cov-oracle').value, 'verified');
  assert.deepEqual(api.state.filtered.map((item) => item.id).sort(), ['numpy.amax', 'numpy.exp', 'numpy.sqrt']);
  assert.equal((element('cov-results').innerHTML.match(/class="cov-result-oracle"/g) || []).length, 3);
  element('cov-oracle').value = 'unverified';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id), ['numpy.histogram']);
  assert.doesNotMatch(element('cov-results').innerHTML, /cov-result-oracle/);
  api.resetFilters();
  assert.equal(element('cov-oracle').value, 'all');
  element('cov-search').value = 'ma_manip.jsonl';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id), ['numpy.ma.MaskedArray.flatten'], 'oracle files are searchable');
  element('cov-scope').value = 'extensions';
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /1 oracle-verified entries in this scope/);
});

test('detail pane renders contracts, host pins, alias via and the two gap readings', () => {
  const { api, element } = dashboard();
  api.state.rows = oracleRows();
  api.changeScope();
  const show = (id) => { element('cov-search').value = id; api.filterRows(); return element('cov-detail').innerHTML; };
  const sqrt = show('numpy.sqrt');
  assert.match(sqrt, /<strong>1,200<\/strong> committed NumPy 2\.4\.2 contracts/);
  assert.match(sqrt, /2 error-message contracts/);
  assert.match(sqrt, /<code>float32<\/code> <code>float64<\/code> · 26 memory layouts/);
  assert.doesNotMatch(sqrt, /pinned host/);
  const exp = show('numpy.exp');
  assert.match(exp, /40 of 40 contracts are hard-gated only on the pinned host \(the win-amd64 CRT libm\)/);
  assert.match(exp, /40 contracts · pinned host only/);
  const amax = show('numpy.amax');
  assert.match(amax, /Verified through <code>np\.max<\/code>'s 1,648 contracts \(reviewed delegation alias\)/);
  assert.match(amax, /Via np\.max/);
  const flatten = show('numpy.ma.MaskedArray.flatten');
  assert.match(flatten, /credits no NumSharp member under this NumPy name/);
  assert.match(flatten, /408 contracts, not credited/);
  const histogram = show('numpy.histogram');
  assert.match(histogram, /No committed NumPy-oracle contract exercises this API/);
  const nextafter = show('numpy.nextafter');
  assert.doesNotMatch(nextafter, /NumPy oracle evidence/, 'a missing API without contracts gets no evidence block');
  assert.match(nextafter, /No oracle evidence/);
});

test('metadata publishes oracle totals and rejects a classification drift', () => {
  const { api, element } = dashboard();
  api.state.rows = oracleRows();
  const data = {
    numpy_version: '2.4.2', numsharp_assembly_version: 'test', schema_version: 1,
    oracle: { contracts: 251498 },
    summary: { default_scope: { total: 5, available: 4, oracle_verified: 3 }, catalog_rows: 7 },
  };
  api.initializeMetadata(data);
  assert.match(element('cov-meta').innerHTML, /251,498 NumPy oracle contracts/);
  assert.equal(element('cov-oracle-reference').textContent,
    'Headline oracle-verified: 60.0% (3/5 APIs; 1 available APIs without contracts).');
  data.summary.default_scope.oracle_verified = 4;
  assert.throws(() => api.initializeMetadata(data), /oracle summary does not match/);
  delete data.summary.default_scope.oracle_verified;
  api.initializeMetadata(data);
  assert.equal(element('cov-oracle-reference').textContent, '', 'pre-1.10 artifacts render without the oracle line');
});
