# Random oracle coverage: the whole random world against NumPy 2.4.2

Living plan and state document. Branch `journey4`. Started 2026-09-27.

**Task (verbatim):** "bringing oracle to completely cover the random world on NumSharp vs NumPy. Plan and work according
to plan, keeping the state up to speed. Be sure to not miss an overload or a parameter of any kind." Earlier in the same
thread: every API, every generator type, 10 fixed seeds plus 10 random seeds.

**State:** see [§9 State log](#9-state-log) (newest entry first) and the phase checklists in [§8](#8-phases).

---

## 1. Goal and acceptance criteria

Every public member of NumSharp's random world is checked against NumPy 2.4.2 by the committed differential oracle
(no Python at test time), and CI fails when that stops being true. Concretely:

1. **Every overload.** Each public constructor, method overload, property getter/setter, field and operator of the 20
   random types (402 members, inventory in §2) is invoked by at least one oracle case, by its exact C# signature. A
   member without a NumPy counterpart is listed in an exemption table with a reason; an exemption that gains a case
   fails CI (self-retiring).
2. **Every parameter.** For each overload: every optional parameter is exercised both omitted (the default path) and
   with a non-default value; every required value parameter takes at least two distinct values; nullable parameters
   take null and non-null; `params` arrays take zero and several elements; every enumerated value NumPy accepts (dtype
   names, `method=`, `check_valid=`, …) appears, and at least one value NumPy rejects, with NumPy's error text.
3. **Every generator type.** Every `Generator` member runs on all five engines (MT19937, PCG64, PCG64DXSM, Philox,
   SFC64). Every legacy `RandomState` sampler runs on the legacy-seeded MT19937 and on `RandomState(bit_generator)` over
   each of the five engines. Every bit-generator member runs on every engine that has it.
4. **Seeds.** Every stream member is recorded under the 10 fixed seeds (§5) in the committed corpus. A nightly job
   draws 10 fresh random seeds, regenerates the same families with NumPy, and replays them (§6.7).
5. **Stream position.** Every stream case also records the receiver's full state after the call (engine state, the
   buffered 32-bit half, the legacy Gaussian cache), so over- and under-drawing is caught, not only wrong values.
6. **Errors.** Every NumPy error a member can raise from its parameters is recorded with NumPy's exception type and
   verbatim text, in NumPy's check order.
7. **Gates in CI.** Inventory freshness, overload coverage, parameter coverage, engine coverage, seed coverage and
   state-observation presence are FuzzMatrix tests (§6.6).

Non-goals: statistical quality of the streams (byte identity subsumes it); C# overload *resolution* at compile time
(reflection targets each overload directly; the resolution cases that matter are unit-tested).

## 2. Scope: the inventory

Reflected on 2026-09-27 (`scratchpad/p7/inventory.cs`), committed as `test/oracle/random_surface.json` in P0.

| Type | Members | Notes |
|---|---:|---|
| `NumPyRandom` (NumPy `RandomState`; `np.random` is one) | 181 | 35 samplers × scalar/sized/array overloads, basics, 13 `default_rng`, 8 `RandomState`, 8 `seed`, 3 `set_state`, `randn<T>`, `Seed`, `_bit_generator`, `_poisson_lam_max` |
| `NumPyRandom.State` (the dict form of `get_state(legacy: false)`) | 6 | |
| `NativeRandomState` (the legacy state tuple) | 7 | incl. an obsolete `byte[]` ctor that always throws |
| `Generator` | 84 | 36 samplers × scalar/array overloads, `random`, `integers`×2, `choice`×2, `bytes`, `shuffle`, `permutation`×2, `permuted`, `spawn`, `bit_generator`, ctor, repr |
| `BitGenerator` (abstract) | 5 | `random_raw`, `spawn`, `lock`, `seed_seq`, `state` |
| `BitGeneratorState` (abstract) | 1 | |
| `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`, `MT19937` | 12 + 12 + 13 + 9 + 15 | ctors per seed form, `advance`, `jumped`, `state`, `_legacy_seeding`, `Philox(seed, counter, key)` |
| their `State` classes | 7 + 7 + 9 + 6 + 6 | ctors, properties, MT19937's implicit conversion from the legacy tuple |
| `SeedSequence`, `SeedlessSeedSequence`, `ISeedSequence`, `ISpawnableSeedSequence` | 17 + 3 + 1 + 1 | |

NumPy side: the coverage catalog's 218 `numpy.random.*` rows (module functions, `RandomState`, `Generator`, the five
engines, `BitGenerator`, `SeedSequence`). Rows NumSharp cannot have (`capsule`, `cffi`, `ctypes`) are out of scope.

## 3. Baseline (before this plan)

Measured 2026-09-27 on `099dd982`:

- Oracle tiers `random_parity(_host)` and `generator_parity(_host)`: 2,196 cases.
- Engines in the oracle: PCG64 (under `Generator`) and MT19937 (under `RandomState`) only. PCG64DXSM, Philox, SFC64,
  and MT19937 under `Generator`: 0 cases.
- Seeds: 42 and 987654321 on scalar paths, 42 only on array-parameter paths; five more legacy seeds only in the 12
  `seed`/`get_state`/`set_state` cases.
- The nightly soak's seeds drive the array-operation fuzzer; it generates no random-sampling case.
- Members with no oracle case: `Generator.permuted`, `Generator.multivariate_normal`, `Generator.spawn`,
  `RandomState.multivariate_normal`, `tomaxint`, every bit-generator member (`random_raw`, `advance`, `jumped`,
  `spawn`, `state` get/set, `seed_seq`), every `SeedSequence` member, `MT19937._legacy_seeding`,
  `RandomState(bit_generator)` over non-MT engines, the dict forms of `get_state`/`set_state`, all state classes.
- Overload granularity: the hand-written dispatch calls one C# overload per NumPy call pattern, so most overloads
  (the `int size` / `int[] size` / `long[] size` / scalar-return forms, `choice(int)`, …) were never invoked.
- The surface guard inventories only `NumPyRandom`'s method names, not `Generator`, the engines or `SeedSequence`, and
  not overloads.

## 4. Principles

- NumPy 2.4.2 is the truth; the corpus is generated by running NumPy and committed; the C# side never runs Python.
- A case names the exact C# overload it tests; the harness resolves that signature by reflection and invokes exactly
  that member, so coverage cannot be claimed by a dispatch that calls a different overload.
- Observations are canonical and identical on both sides (§6.4); nondeterministic results (OS entropy) are observed
  through their deterministic projection only (types, sizes, ranges, invariants).
- A divergence found while building this is a NumSharp bug to fix (reproduce in a test, then fix), unless C# cannot
  express NumPy's behavior; those are recorded as intended divergences with the behavior NumSharp exhibits instead,
  and that alternative behavior is asserted too.
- Inputs on which NumPy never returns (legacy `zipf(a >= 1025)`, legacy `vonmises(kappa >= 2**511)`, Windows NumPy's
  legacy HRUA with a population past 2**31, `SeedSequence.spawn` past 2**32 children) cannot be oracle cases; they
  stay unit-tested and are listed in §7.

## 5. Seeds and engines

- **Fixed seeds (10):** `0, 1, 7, 42, 1234, 65535, 2147483647, 2147483648, 987654321, 4294967295` — every one valid
  for the legacy `RandomState(int)` (below `2**32`) and for every engine; both ends of the 32-bit range included.
- Seed *forms* (big integers past 64 bits, int/long/uint arrays, `SeedSequence` with a spawn key, `NDArray` seeds,
  `object` seeds) are parameters of the constructor / `seed` / `default_rng` members and are covered by their domains.
- **Engines (5):** MT19937, PCG64, PCG64DXSM, Philox, SFC64. `RandomState` also runs on the legacy-seeded MT19937
  (`RandomState(seed)`), which differs from `RandomState(MT19937(seed))` only in seeding.

## 6. Design

### 6.1 Case schema (op key `random_api`)

The common corpus schema, so every existing tool (CorpusSurvey, the coverage join, the leak sweep) keeps working:

```json
{"id": "...", "op": "random_api",
 "params": {"member": "Generator.normal",               // NumPy-level member: the coverage join's param
            "sig": "Generator.normal(double,double,Shape)", // the exact C# overload the harness invokes
            "recv": {"k": "Generator", "engine": "Philox", "seed": 42},
            "prime": "none",                             // receiver priming before the call (§6.3)
            "args": [{"n": "loc", "t": "double", "v": 0.5}, {"n": "scale", "omit": true}, ...],
            "calls": 1},
 "operands": [ ... NDArray arguments, referenced by index ... ],
 "expected": {"kind": "random_api", "result": {...}, "state": "..."},
 "layout": "random_api/Generator", "valueclass": "stream"}
```

Error cases carry `expects_throw` and `error: {type, text}` like every other tier.

### 6.2 Arguments

Typed, named JSON values decoded against the resolved parameter's actual type: `double`, `long`, `int`, `uint`,
`ulong`, `bool`, `string`, `BigInteger` (decimal string), `UInt128`, nullable forms, `Shape` (dims / default),
`Shape?`, `DType` (name / null), `NDArray` (operand index / null), arrays (`int[]`, `long[]`, `uint[]`, `ulong[]`,
`double[]`, `double[,]`), `params` arrays, and object-typed arguments encoded with their Python type (int, big int,
float, str, bool, None, complex, list, ndarray, SeedSequence, bit generator, Generator, RandomState, state objects).
An omitted argument is passed as `Type.Missing` (the parameter's default), exactly as a C# call that leaves it out.

### 6.3 Receivers and priming

- `RandomState` legacy-seeded (`np.random.RandomState(seed)`), or `RandomState(ENGINE(seed))` for each engine.
- `Generator(ENGINE(seed))` for each engine.
- A bit generator `ENGINE(seed)`; a `SeedSequence(...)`; a state object; none (constructors).
- Priming puts the receiver in a non-trivial state before the observed call: `none`, `f32` (one float32 draw: a
  buffered 32-bit half pending), `gauss` (one legacy normal: the Gaussian cache full), `raw3` (three raw words).
- `calls: N` repeats the observed call N times and records the last result (stream advancement, the legacy binomial
  cache).

### 6.4 Observations

Canonical on both sides: arrays as (dtype, shape, bytes); Python float / int / bool scalars against NumSharp's 0-d
results by value; `None` against `void`; strings verbatim; sequences item by item; dicts in NumPy's key order; bit
generators and `Generator`s by type + state (+ `seed_seq`); state as canonical text (MT19937's 624-word key by SHA-256,
the other engines in full; RandomState adds `has_gauss` and the cached Gaussian's bits). After every stream call the
receiver's state text is recorded too.

### 6.5 Tiers

| File | Content | Gate |
|---|---|---|
| `random_api.jsonl` | libm-free members: bits, integers, uniform, choice/shuffle/permutation, bytes, raw, state, seeding, `SeedSequence` | hard, every host |
| `random_api_host.jsonl` | samplers whose transform consumes libm | hard on win-amd64, Inconclusive elsewhere |
| `random_api_mvn.jsonl` | `multivariate_normal` on both APIs | win-amd64 with the pinned OpenBLAS (the `linalg_parity` pin), Inconclusive elsewhere |
| `random_api_lp64.jsonl` | outcomes that depend on C `long` being 64-bit, authored with Linux NumPy (WSL `~/np242/bin/python`) | hard on x64 Windows and Linux (both replayed green), Inconclusive elsewhere |

### 6.6 Gates (FuzzMatrix)

G1 inventory freshness (reflection == `random_surface.json`); G2 overload coverage; G3 parameter coverage; G4 engine
coverage; G5 seed coverage; G6 every stream case carries a state observation; plus the existing floors (`MinCases`)
and the leak sweep over the new files.

### 6.7 Nightly soak with 10 fresh seeds

A new job in `fuzz-soak.yml` on `windows-latest` (the host the libm tier is authored on): draws 10 fresh seeds below
`2**32`, runs the generator in soak mode with them, replays the result through the same harness (a test that reads the
generated directory from an environment variable and is Inconclusive without it), and uploads failing corpora.

### 6.8 Coverage join and docs

`coverage/oracle_map.json` gains the `random_api` key (param `member`, ids `numpy.random.{value}`, overrides for
NumSharp-only members) and host pins for the new tiers; the Fuzz README, CLAUDE.md's differential-fuzz section and the
compliance page describe the new family.

## 7. Exemptions and intended divergences (filled in as found)

| Member / region | Why no oracle case | Where it is tested instead |
|---|---|---|
| `NativeRandomState(byte[])` | obsolete NumSharp constructor that always throws; no NumPy counterpart | unit tests |
| `NumPyRandom.Seed` | NumSharp bookkeeping property, not NumPy API | unit tests |
| `BitGenerator.lock` | a threading primitive; nothing NumPy-observable | unit tests (lock identity, concurrency) |
| legacy `zipf(a >= 1025)`, legacy `vonmises(kappa >= 2**511)` | NumPy never returns | `RandomBroadcast.Test.cs` |
| `SeedSequence.spawn` past `2**32` children | NumPy hangs (uint32 loop variable wraps) | `BitGeneratorFamily.Test.cs` |
| `randint(dtype='l')` (the explicit C-long char) | `np.dtype('l')` is the HOST's C long in both libraries (int32 on Windows, int64 on Linux), so no portable expectation exists; the omitted dtype (NumSharp's LP64 long) is covered | `RandomTypeParity.Test.cs` |
| `multivariate_hypergeometric(method='count')` with `sum(colors)` near `10**9` | both libraries fill a `sum(colors)`-entry index array (8 GB); the corpus keeps it at `10**6` and pins the pre-allocation limits | — |

Intended divergences (in the corpus, excused by `RandomApiDivergences` with a key AND a bound or condition, printed by
every replay):

| Member | NumPy | NumSharp | Why |
|---|---|---|---|
| `Generator.pareto`, `Generator.power` | the win-amd64 CRT `expm1` | Sun's `s_expm1`: at most 2 ULP (pareto), `ceil(2/a) + 3` ULP (power); stream positions identical | the CRT's in-band `expm1` is not portable |
| `NumPyRandom.get_state()` on a non-MT19937 engine | warns, returns the dict | `ValueError` naming `get_state(legacy: false)` | the overload is typed as the legacy tuple |
| `set_state` / `RandomState(NativeRandomState)` with `pos` outside `[0, 624]` | stores it; the next draw reads past the key | `ValueError` | undefined behavior in NumPy |

## 8. Phases

- [x] **P0 — plan and inventory.** This document; the inventory is written by the G1 test itself
  (`RandomApiSurfaceTests`, with `NUMSHARP_WRITE_RANDOM_SURFACE=1`) rather than a separate script, so the file and its
  gate share one reflection; G1 test.
- [x] **P1 — harness core.** C#: case model, receiver factory, typed argument decoding, reflective invocation, the
  observation serializer, the comparator (with the per-element ULP bounds pareto/power already carry, for arrays and
  scalar Python floats), error mapping, replay tests per tier, leak-sweep routing (`random_api:<member>` coverage
  key). Python (`test/oracle/gen_random_oracle.py`): receivers, argument encoders, observation serializers, binding
  table, writer, the LP64 sub-run and merge. `MinCases` floors move to P7 (set once the corpus is final).
- [x] **P2 — `NumPyRandom` bindings** (all 181 members + `NumPyRandom.State` + `NativeRandomState`), with errors —
  except the 13 `default_rng` overloads, which need the P4 seed-sequence/generator arguments.
- [x] **P3 — `Generator` bindings** (all 84 members) × 5 engines, with errors.
- [ ] **P4 — bit generators, seed sequences, state classes, `default_rng`, `RandomState` factories.**
- [ ] **P5 — gates G2–G6 on;** every divergence they surface triaged: fixed with a regression test, or recorded in §7.
- [ ] **P6 — nightly soak job** with 10 fresh seeds.
- [ ] **P7 — coverage join, docs, floors;** full verification (both TFMs, FuzzMatrix, coverage generator); commit.

## 9. State log

- **2026-09-27 (P2 + P3 done, checkpoint commit)** — Every `NumPyRandom`, `NumPyRandom.State`, `NativeRandomState` and
  `Generator` member is generated (`default_rng` waits for P4). The generator gained a per-overload emitter
  (`emit_m`) and an API-generic `sweep`; object arguments (bit generators, state tuples/dicts built identically on
  both sides, fresh per case), argument aliasing (`permuted(x, out=x)`), watched in-place operands (`shuffle`, `out=`),
  typed generic arguments (`randn<T>` as NumPy's cast of the draw), entropy masks (`seed()`, `RandomState()`), object
  and sequence observations, and a representability check (a value no C# argument can carry is not a case). A new
  `random_api_mvn` tier replays `multivariate_normal` through NumPy's own OpenBLAS (705 cases, byte-exact). Corpus:
  portable 6,511, host 15,733, mvn 705, LP64 206. Findings fixed in NumSharp (each with a unit pin in
  `RandomOracleFindings.Test.cs`): §10 items 3–9. The harness itself had a bug the oracle exposed: an omitted
  `Shape size = default` was passed as `Activator.CreateInstance(typeof(Shape))` — Shape's parameterless constructor,
  i.e. `Shape.Scalar` (`size=()`) — where the compiler passes `default(Shape)` (`size=None`). The coverage join maps
  the new op key (`coverage/oracle_map.json`: 61 value overrides, host pins for the three pinned tiers); four aliases
  it made stale (`random`/`ranf`/`sample` onto `random_sample`) were deleted, as the join requires. Headline
  oracle-verified APIs 486 to 493 of 560. Verified: worktree full Oracle and unit suites on net8.0 and net10.0, WSL
  Linux replay of the portable and LP64 tiers, the coverage generator and its tests. A parallel session's `9aac6ee9`
  (random type parity: every legacy C long is int64, `long` single-dimension sizes, get/set_bit_generator) landed
  during P2; the G1 gate caught the surface drift and the inventory was regenerated on it. Unclaimed: 162 members (P4).
- **2026-09-27 (P1 done)** — The 29 distribution samplers are generated for every overload on both APIs: legacy on
  `RandomState(seed)` and `RandomState(ENGINE(seed))` × 5 engines, `Generator(ENGINE(seed))` × 5, each × the 10 fixed
  seeds; size forms, omitted optionals, nulls, priming (`u32`/`gauss`/`raw3`), `calls: 2`, array-parameter layouts,
  and NumPy's errors. Corpus: `random_api.jsonl` 914 (portable), `random_api_host.jsonl` 13,042 (win-amd64 libm),
  `random_api_lp64.jsonl` 32. All green on Windows; under Linux .NET (WSL, glibc) the portable and LP64 tiers are green
  too (host tier Inconclusive there by design). The LP64 sub-run: the legacy integer families are regenerated by Linux
  NumPy in WSL, and a case moves to the LP64 tier when Linux's answer leaves int32 or the platforms disagree on raising;
  ids are stable (content-derived, `#k` only for true duplicates), and the merge refuses an id naming different cases
  on the two platforms. The leak sweep now replays `random_api*` through the harness (measured, no escapes).
  Remaining unclaimed: 275 members (P2–P4). Verification runs in a detached worktree (`scratchpad/p7/wt`) because the
  parallel session's uncommitted polynomial WIP leaves the shared tree's build red.
- **2026-09-27** — Plan written. Inventory reflected: 402 public members over 20 types. Baseline measured (§3).
  Existing infrastructure read: `gen_oracle.py` random sections, `OpRegistry(.Generator/.RandomBroadcast).cs`,
  `FuzzCorpusTests` replay, the coverage join (`param`-driven keys), the leak sweep (replays every corpus file), the
  soak workflow (ubuntu-latest, 1 fixed + 9 random seeds for the array fuzzer). A parallel session is active on the
  polynomial review (`da84683f`); commits stage only this plan's files.

## 10. Findings

1. **Legacy integer dtype policy** — RESOLVED by the parallel `9aac6ee9`: every legacy C `long` is now NumSharp's LP64
   int64 (`randint`/`permutation`/`choice`/`random_integers`/`multinomial` had returned the win-amd64 int32). The oracle
   widens Windows NumPy's int32 answers for all of them, and the LP64 tier (206 cases) carries the answers only a 64-bit
   `long` gives (`randint(0, 2**40)`, `choice(2**40)`, `tomaxint`, `geometric(p=5e-324)`, `zipf(1.5)` past `LONG_MAX`,
   the LP64 `poisson` lam bound, ...).
2. **NumPy's own uniform error wording differs by path**: the scalar path says "high - low range exceeds valid
   bounds", the array path "Range exceeds valid bounds". Both recorded verbatim; NumSharp matches both.
3. **`rand(default(Shape))` threw `ArgumentNullException`** (it allocated from a shape without dimensions). NumPy's
   counterpart is `rand()`: now one draw, 0-d. FIXED.
4. **`None` arguments threw `NullReferenceException`** in legacy `choice`/`permutation`/`shuffle` and
   `Generator.choice`/`permutation`/`permuted`/`shuffle`. NumPy's `np.asarray(None)` is a 0-d object array, so each
   raises the error a 0-d argument gets (`permuted` including its `out` checks and the object-to-dtype cast). FIXED.
5. **A null seed array raised "Seed must be non-empty"** in `seed(int[]/long[]/uint[])`, `RandomState(...)` and
   `MT19937._legacy_seeding(...)`; NumPy's `seed(None)` draws OS entropy. FIXED (an EMPTY array still raises).
6. **`has_gauss` was stored as a bool**, so a `set_state` tuple with `has_gauss=2` read back 1; NumPy keeps the C int
   until the cache is consumed. FIXED (`_hasGauss` is the int).
7. **`permuted`'s cast error said "array data" for a 0-d source**; NumPy's `copyto` says "scalar". FIXED.
8. **Legacy `multinomial` leaked its output on the `n < 0` error path** (NumPy allocates before checking `n`, and so
   did NumSharp, but never released it). Caught by the leak sweep replaying the new tier. FIXED.
9. **.NET 8's double `%` returns the DEFAULT NaN (`0xfff8...`, sign set) for a NaN dividend**, where C's `fmod` (NumPy)
   and .NET 10 propagate the input NaN. `vonmises(mu=NaN)` therefore returned `-NaN` on net8.0 only. FIXED in both
   vonmises wraps. The same `%` is NumSharp's float `np.fmod`/`np.mod` kernel on net8.0; the ordinary oracle tokenizes
   NaN bits, so that difference is invisible there and worth a separate look.
