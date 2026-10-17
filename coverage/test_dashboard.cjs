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

// Run the page's inline script against a stub DOM (one recording element per id in the page) and
// return its exposed functions with the fixture loaded; every function a test calls must be listed
// in the `globalThis.dashboard` replacement below.
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
  const exposed = inline.replace(/\n  initialize\(\);\s*\n\}\)\(\);\s*$/, '\n  globalThis.dashboard = { state, rowInScope, summarize, surfaceLabel, apiLabel, scopedRows, initializeMetadata, initializeFilters, renderSummary, filterRows, changeScope, resetFilters, applyPreset, tooltipContent, parityClass, signatureClass, signatureMarks };\n})();');
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
  assert.match(element('cov-applicability-note').textContent, /1 declared contract whose calls NumPy rejects \(ufunc methods/);
  assert.equal(api.scopedRows().length, 7, 'all declared API contracts remain countable');
  element('cov-search').value = 'np.sin.reduce';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.match(element('cov-detail').innerHTML, /NumPy rejects this ufunc method/);
  element('cov-scope').value = 'default';
  api.changeScope();
  assert.equal(element('cov-applicability-note').textContent, '');
});

test('a refused Python protocol is disclosed as a protocol, not as a ufunc method', () => {
  const { api, element } = dashboard();
  api.state.rows.push(row('numpy.ndarray.__delitem__', 'Array protocols', { surface: 'ndarray', name: '__delitem__', kind: 'method', applicability: 'not_applicable', python_protocol: true }));
  api.state.rows.push(row('numpy.sin.reduce', 'Universal function objects', { surface: 'ufunc', name: 'sin.reduce', kind: 'method', applicability: 'not_applicable' }));
  api.changeScope();
  assert.match(element('cov-applicability-note').textContent, /2 declared contracts whose calls NumPy rejects/);
  element('cov-search').value = 'ndarray.__delitem__';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.match(element('cov-detail').innerHTML, /NumPy rejects every call of this protocol/);
  assert.doesNotMatch(element('cov-detail').innerHTML, /ufunc method/);
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

// The site renders whatever the data branch's inventory/latest holds, and master's CI publishes the
// inventory its own generator writes. A generator without the oracle join (master's 1.6.0) records no
// `oracle` field on any row: the page must not read that as "no contracts" for every API (it showed
// "Oracle-verified parity 0.0% (0 of 560)" and an "Available, no oracle contracts" filter).
test('an artifact without oracle evidence (generator < 1.10) hides every oracle element', () => {
  const { api, element } = dashboard();
  api.state.rows = oracleRows().map(({ oracle, ...rest }) => rest);
  api.initializeMetadata({
    numpy_version: '2.4.2', numsharp_assembly_version: '0.70.0.0', schema_version: 1,
    summary: { default_scope: { total: 5, available: 4 }, catalog_rows: 7 },
  });
  assert.equal(api.state.hasOracle, false);
  assert.ok(api.state.rows.every((item) => api.parityClass(item) === 'none'), 'no row reads as unverified');
  assert.equal(element('cov-oracle-reference').textContent, '');
  assert.equal(element('cov-lede-oracle').hidden, true, 'the lede no longer promises oracle contracts');
  assert.equal(element('cov-oracle-explainer').hidden, true, 'the reading guide drops the oracle paragraph');
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /Scoped API availability/);
  assert.doesNotMatch(element('cov-metrics').innerHTML, /Oracle-verified parity/);
  assert.doesNotMatch(element('cov-surface-grid').innerHTML, /oracle-verified/);
  assert.doesNotMatch(element('cov-category-grid').innerHTML, /oracle-verified/);
  assert.doesNotMatch(api.tooltipContent({ dataset: { tooltipGroup: 'surface:np' } }), /oracle-verified/);
  assert.deepEqual(Array.from(element('cov-oracle').options, (item) => item.value), ['all', 'none']);
  element('cov-search').value = 'numpy.histogram';
  api.filterRows();
  assert.equal(api.state.filtered.length, 1);
  assert.doesNotMatch(element('cov-detail').innerHTML, /[Oo]racle/, 'no oracle fact and no "no contracts" block');
  assert.doesNotMatch(element('cov-results').innerHTML, /cov-result-oracle/);
});

test('an artifact with oracle evidence keeps the oracle elements after metadata', () => {
  const { api, element } = dashboard();
  api.state.rows = oracleRows();
  api.initializeMetadata({
    numpy_version: '2.4.2', numsharp_assembly_version: 'test', schema_version: 1, oracle: { contracts: 251498 },
    summary: { default_scope: { total: 5, available: 4, oracle_verified: 3 }, catalog_rows: 7 },
  });
  assert.equal(api.state.hasOracle, true);
  assert.equal(element('cov-lede-oracle').hidden, false);
  assert.equal(element('cov-oracle-explainer').hidden, false);
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /Oracle-verified parity/);
  assert.match(element('cov-surface-grid').innerHTML, /3 oracle-verified/);
  assert.match(api.tooltipContent({ dataset: { tooltipGroup: 'surface:np' } }), /3 oracle-verified/);
});

// ---- Signature parity (generator 1.15+: credited calls carry a `signature` record) ----

// A credited headline row with the given signature record (coverage/signature_parity.py's shape).
function signed(id, signature, overrides = {}) {
  return row(id, 'Math', { status: 'available', availability: 'exact', in_default_scope: true, extended: false, signature, ...overrides });
}

// One row per record shape the generator emits: identical (with and without informational notes),
// each mark (M X O T N), a closest-overload-only difference, not compared, and an uncredited API.
const signatureRows = () => [
  signed('numpy.sum', { status: 'differs', spelling: 'exact', overloads: 7, missing: ['out', 'initial', 'where'],
    order: ['keepdims: NumPy #4, C# #3'], closest: 'NDArray sum(NDArray a, int? axis = null)', numpy: 'sum(a, axis=None, ...)' }),
  signed('numpy.transpose', { status: 'differs', spelling: 'exact', overloads: 1, missing: ['axes (C#: premute)'],
    foreign: ['premute'], closest: 'NDArray transpose(NDArray a, int[] premute = null)', numpy: 'transpose(a, axes=None)' }),
  signed('numpy.inner', { status: 'identical', spelling: 'exact', overloads: 1, closest: 'NDArray inner(NDArray a, NDArray b)', numpy: 'inner(a, b, /)' }),
  signed('numpy.acos', { status: 'differs', spelling: 'alias', overloads: 1, excluded: ['casting', 'order'],
    closest: 'NDArray arccos(NDArray x)', numpy: 'acos(x, /, out=None, ...)' }, { availability: 'alias' }),
  signed('numpy.zeros', { status: 'differs', spelling: 'exact', overloads: 1, typecode: ['System.Type dtype'],
    excluded: ['like'], closest: 'NDArray zeros(Shape shape, System.Type dtype = null)', numpy: 'zeros(shape, dtype=None, ...)' }),
  signed('numpy.full', { status: 'identical', spelling: 'exact', overloads: 1, null_default: ["order: 'C' vs null"],
    excluded: ['like'], closest: 'NDArray full(Shape shape, object fill_value)', numpy: 'full(shape, fill_value, ...)' }),
  signed('numpy.std', { status: 'differs', spelling: 'exact', overloads: 1, order: ['ddof: NumPy #4, C# #3'],
    required: ['ddof'], closest: 'NDArray std(NDArray a, int ddof)', numpy: 'std(a, axis=None, ...)' }),
  signed('numpy.astype', { status: 'not-compared', spelling: 'exact', reason: 'credited through NumSharp.NDArray.astype, whose receiver is an argument of the NumPy call or the other way round' }),
  row('numpy.nextafter', 'Math', { status: 'missing', availability: 'missing', in_default_scope: true, extended: false }),
];

test('signature classes, marks and counts mirror the generator for every record shape', () => {
  const { api } = dashboard();
  const rows = signatureRows();
  const shown = Object.fromEntries(rows.map((item) => [item.id, `${api.signatureClass(item)}:${api.signatureMarks(item)}`]));
  assert.deepEqual(shown, {
    'numpy.sum': 'differs:MO', 'numpy.transpose': 'differs:MX', 'numpy.inner': 'identical:',
    'numpy.acos': 'differs:N', 'numpy.zeros': 'differs:T', 'numpy.full': 'identical:',
    'numpy.std': 'differs:', 'numpy.astype': 'not-compared:', 'numpy.nextafter': 'none:',
  });
  // An alias that could not be compared still shows N: its spelling is another name either way.
  assert.equal(api.signatureMarks(signed('numpy.shape', { status: 'not-compared', spelling: 'alias', reason: 'credited to a property' })), 'N');
  const counts = api.summarize(rows);
  assert.equal(counts.signature, 2);
  assert.equal(counts.signatureApis, 2);
  assert.equal(counts.signatureDiffers, 5);
  assert.equal(counts.signatureNotCompared, 1);
  assert.deepEqual([counts.signatureMissing, counts.signatureForeign, counts.signatureOverloaded, counts.signatureTypecode, counts.signatureOtherName], [2, 1, 1, 1, 1]);
  assert.equal(counts.signatureCoverage, 100 * 2 / 9, 'identical over every comparable API of the scope, as the generator divides');
});

test('signature metric, filter, badges and search follow the selected scope', () => {
  const { api, element } = dashboard();
  api.state.rows = signatureRows();
  api.changeScope();
  assert.match(element('cov-metrics').innerHTML, /NumPy-identical signatures/);
  assert.match(element('cov-metrics').innerHTML, /22\.2%/);
  assert.match(element('cov-metrics').innerHTML, /2 of 9 take NumPy&#039;s exact parameters · 5 differ/);
  assert.match(element('cov-metrics').innerHTML, /data-metric-action="signature"/);
  assert.deepEqual(Array.from(element('cov-signature').options, (item) => item.value),
    ['all', 'identical', 'differs', 'mark:M', 'mark:X', 'mark:O', 'mark:T', 'mark:N', 'not-compared', 'none']);
  assert.match(element('cov-surface-grid').innerHTML, /2 identical signatures/);
  assert.match(api.tooltipContent({ dataset: { tooltipGroup: 'surface:np' } }), /2 identical signatures/);
  api.applyPreset('signature');
  assert.equal(element('cov-signature').value, 'differs');
  assert.deepEqual(api.state.filtered.map((item) => item.id).sort(), ['numpy.acos', 'numpy.std', 'numpy.sum', 'numpy.transpose', 'numpy.zeros']);
  // One badge per differing row, its letters the marks (a closest-overload-only difference reads "sig").
  const badges = [...element('cov-results').innerHTML.matchAll(/class="cov-result-signature"[^>]*>([^<]*)</g)].map((match) => match[1]).sort();
  assert.deepEqual(badges, ['MO', 'MX', 'N', 'T', 'sig']);
  element('cov-signature').value = 'mark:M';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id).sort(), ['numpy.sum', 'numpy.transpose']);
  element('cov-signature').value = 'identical';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id).sort(), ['numpy.full', 'numpy.inner']);
  assert.doesNotMatch(element('cov-results').innerHTML, /cov-result-signature/, 'an identical row carries no badge');
  api.resetFilters();
  assert.equal(element('cov-signature').value, 'all');
  element('cov-search').value = 'premute';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id), ['numpy.transpose']);
  // A foreign name that appears nowhere else in the row: only the `foreign` list can match it.
  api.state.rows.push(signed('numpy.flip', { status: 'differs', spelling: 'exact', overloads: 1, foreign: ['axis_list'] }));
  element('cov-search').value = 'axis_list';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id), ['numpy.flip'], 'foreign parameter names are searchable');
  element('cov-search').value = 'initial';
  api.filterRows();
  assert.deepEqual(api.state.filtered.map((item) => item.id), ['numpy.sum'], 'missing parameter names are searchable');
});

test('detail pane renders marks, closest-overload differences and the not-compared reason', () => {
  const { api, element } = dashboard();
  api.state.rows = signatureRows();
  api.changeScope();
  // Select one API by searching for it and return the detail pane it renders.
  // id: the row id to search for (the first result becomes the selected row).
  const show = (id) => { element('cov-search').value = id; api.filterRows(); return element('cov-detail').innerHTML; };
  const transpose = show('numpy.transpose');
  assert.match(transpose, /Differs · M X/);
  assert.match(transpose, /<strong>M<\/strong> · no overload offers <code>axes \(C#: premute\)<\/code>/);
  assert.match(transpose, /<strong>X<\/strong> · NumPy has no <code>premute<\/code>/);
  assert.match(transpose, /Compared overload: <code>NDArray transpose\(NDArray a, int\[\] premute = null\)<\/code>/);
  const sum = show('numpy.sum');
  assert.match(sum, /<strong>O<\/strong> · 7 C# overloads; NumPy has one signature/);
  assert.match(sum, /Closest overload: <code>at another position: keepdims: NumPy #4, C# #3<\/code>/);
  const inner = show('numpy.inner');
  assert.match(inner, /Identical to NumPy/);
  assert.match(inner, /One C# overload has exactly NumPy's parameters under NumPy's name/);
  assert.match(inner, /cov-signature-block is-match/);
  const acos = show('numpy.acos');
  assert.match(acos, /<strong>N<\/strong> · credited through another name/);
  assert.match(acos, /not counted: <code>casting<\/code> <code>order<\/code>/);
  const zeros = show('numpy.zeros');
  assert.match(zeros, /<strong>T<\/strong> · dtype taken as <code>System\.Type dtype<\/code> instead of <code>DType<\/code>/);
  const full = show('numpy.full');
  assert.match(full, /Informational, a null default where NumPy has a value: <code>order: &#039;C&#039; vs null<\/code>/);
  const std = show('numpy.std');
  assert.match(std, /Differs \(closest overload\)/);
  assert.match(std, /<code>C# requires ddof<\/code>/);
  const astype = show('numpy.astype');
  assert.match(astype, /Not compared: credited through NumSharp\.NDArray\.astype/);
  const nextafter = show('numpy.nextafter');
  assert.doesNotMatch(nextafter, /NumPy signature parity/, 'an uncredited API gets no signature block');
  assert.match(nextafter, /No signature record/);
});

test('metadata recounts the signature headline and rejects a drift', () => {
  const { api, element } = dashboard();
  api.state.rows = signatureRows();
  const data = {
    numpy_version: '2.4.2', numsharp_assembly_version: 'test', schema_version: 1,
    summary: { default_scope: { total: 9, available: 8, signature_identical: 2, signature_differs: 5 }, catalog_rows: 9 },
  };
  api.initializeMetadata(data);
  assert.equal(api.state.hasSignature, true);
  assert.equal(element('cov-signature-reference').textContent,
    'Headline NumPy-identical signatures: 22.2% (2/9 APIs; 5 differ: 2 M, 1 X, 1 O, 1 T, 1 N).');
  assert.equal(element('cov-lede-signature').hidden, false);
  assert.equal(element('cov-signature-explainer').hidden, false);
  data.summary.default_scope.signature_identical = 3;
  assert.throws(() => api.initializeMetadata(data), /signature summary does not match/);
  data.summary.default_scope.signature_identical = 2;
  data.summary.default_scope.signature_differs = 4;
  assert.throws(() => api.initializeMetadata(data), /signature summary does not match/);
});

// An artifact published before generator 1.15 has no `signature` record on any row; reading that as
// "no record" for every API would claim 0.0% identical signatures.
test('an artifact without signature records (generator < 1.15) hides every signature element', () => {
  const { api, element } = dashboard();
  api.state.rows = signatureRows().map(({ signature, ...rest }) => rest);
  api.initializeMetadata({
    numpy_version: '2.4.2', numsharp_assembly_version: 'test', schema_version: 1,
    summary: { default_scope: { total: 9, available: 8 }, catalog_rows: 9 },
  });
  assert.equal(api.state.hasSignature, false);
  assert.equal(element('cov-signature-reference').textContent, '');
  assert.equal(element('cov-lede-signature').hidden, true);
  assert.equal(element('cov-signature-explainer').hidden, true);
  api.changeScope();
  assert.doesNotMatch(element('cov-metrics').innerHTML, /NumPy-identical signatures/);
  assert.doesNotMatch(element('cov-surface-grid').innerHTML, /identical signatures/);
  assert.doesNotMatch(api.tooltipContent({ dataset: { tooltipGroup: 'surface:np' } }), /identical signatures/);
  assert.deepEqual(Array.from(element('cov-signature').options, (item) => item.value), ['all', 'none']);
  element('cov-search').value = 'numpy.transpose';
  api.filterRows();
  assert.doesNotMatch(element('cov-detail').innerHTML, /NumPy signature<\/div>|NumPy signature parity/);
  assert.doesNotMatch(element('cov-results').innerHTML, /cov-result-signature/);
});
