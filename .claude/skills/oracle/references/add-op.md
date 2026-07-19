# Adding a new op to the differential-fuzz oracle

Goal: make NumSharp's `np.<op>` provably bit-identical to NumPy 2.4.2 across the layout × dtype matrix, wired into
the `FuzzMatrix` gate exactly like every other op. Two edits (generator + registry), one regeneration, one gate run.

## 0. Prerequisite

`python -c "import numpy; print(numpy.__version__)"` must print `2.4.2`. Anything else can shift the recorded bytes
and make the committed corpus disagree with real NumPy.

## 1. Choose the tier (`gen_oracle.py`)

Each tier is a `gen_<mode>(...)` function that loops `layout × dtype`, builds `(base, view)` fixtures from
`layout_catalog.py`, runs a list of `jobs`, and records NumPy's `(dtype, shape, C-contiguous bytes)`. Pick the tier
that matches your op family:

| Op family | `gen_<mode>` | Corpus file | Mode arg |
|-----------|--------------|-------------|----------|
| shape/view manipulation | `gen_manip` | `manip.jsonl` | `manip` |
| elementwise binary | `gen_binary` | `binary_arith.jsonl` etc. | `binary` |
| elementwise unary | `gen_unary` | `unary.jsonl` | `unary` |
| reductions | `gen_reduce` | `reduce.jsonl` | `reduce` |
| scans / diff | `gen_scan` | `scan.jsonl` | `scan` |
| sorting / searching | `gen_argsort` etc. | `sort.jsonl` | `sort` |

The full mode list is in `gen_oracle.py`'s `main()` (`elif mode == "<mode>":`). `grep -n "def gen_" gen_oracle.py`.

## 2. Add job(s) to the tier

A job is a tuple `(opname, params_dict, lambda)`. The `lambda` takes the fixture `view` and returns the NumPy
result; `params_dict` is what the C# side reads back to reconstruct the same call. Guard by `nd` (ndim) / `sz`
(size) where NumPy would raise — the generator's `try/except` skips raising cases and prints a skipped count.

**Worked example (the flip/trim_zeros/transpose-alias additions to `gen_manip`):**

```python
# base job list (dtype-agnostic, any ndim incl. 0-d):
("flip", {}, lambda v: np.flip(v)),                       # reverse ALL axes (0-d -> scalar)

if nd >= 1:
    ("flipud", {}, lambda v: np.flipud(v)),
    ("flip", {"axis": 0}, lambda v: np.flip(v, 0)),       # scalar-axis form -> int overload
    ("trim_zeros", {"trim": "fb"}, lambda v: np.trim_zeros(v, "fb")),
    ("trim_zeros", {"trim": "fb", "axis": 0}, lambda v: np.trim_zeros(v, "fb", axis=0)),

if nd >= 2:
    ("fliplr", {}, lambda v: np.fliplr(v)),
    ("flip", {"axes": [0, nd - 1]}, lambda v, nd=nd: np.flip(v, (0, nd - 1))),   # tuple-axis -> int[] overload
    ("permute_dims", {}, lambda v: np.permute_dims(v)),
    ("matrix_transpose", {}, lambda v: np.matrix_transpose(v)),
    ("trim_zeros", {"trim": "fb", "axes": [nd - 1]}, lambda v, nd=nd: np.trim_zeros(v, "fb", axis=(nd - 1,))),
```

Notes that matter:
- **Bind loop variables with defaults** (`lambda v, nd=nd: ...`) — Python closures capture by reference, so a bare
  `nd` in the lambda would see the loop's final value.
- **Pick params that reach real code paths.** Above, `{"axis": …}` vs `{"axes": […]}` deliberately exercise the
  scalar-int vs int[] overloads of `np.flip`/`np.trim_zeros`. One representative call per distinct path is enough;
  the layout × dtype loop multiplies each job into hundreds of cases.
- **Value-dependent ops need the right fixtures.** `trim_zeros` only trims when an edge is zero — the pools in
  `layout_catalog.py` are front-loaded with `0` (int) and `0.0/-0.0` amid `nan/inf` (float), so `f`/`b`/`fb` each
  exercise real cropping. If your op depends on values NOT present in the pools, add a dedicated `gen_<op>` tier
  (see `gen_clip`/`gen_quantile` for the pattern) instead of relying on `gen_manip`.

## 3. Add the matching case (`OpRegistry.cs`)

`OpRegistry.Apply(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)` is a big switch that pairs
**1:1** with the generator opnames. Add your case next to the related ops:

```csharp
case "flip":
    if (p.ContainsKey("axes")) return np.flip(ops[0], ParseIntArray(p["axes"]));
    return p.ContainsKey("axis") ? np.flip(ops[0], p["axis"].GetInt32()) : np.flip(ops[0]);
case "fliplr": return np.fliplr(ops[0]);
case "permute_dims": return np.permute_dims(ops[0], p.ContainsKey("axes") ? ParseIntArray(p["axes"]) : null);
case "trim_zeros":
    if (p.ContainsKey("axes")) return np.trim_zeros(ops[0], p["trim"].GetString(), ParseIntArray(p["axes"]));
    return p.ContainsKey("axis")
        ? np.trim_zeros(ops[0], p["trim"].GetString(), p["axis"].GetInt32())
        : np.trim_zeros(ops[0], p["trim"].GetString());
```

Param-reading helpers (all on `JsonElement p["…"]`): `.GetInt32()`, `.GetDouble()`, `.GetString()`,
`ParseIntArray(p["axes"])`, `ParseAxis(p)` (nullable axis), `ParseKeepdims(p)`. `ops[0]`, `ops[1]`, … are the
operands FuzzCorpus rebuilt from the recorded bytes. Mutating ops (`place`/`put`/`copyto`) return the mutated
operand as the result.

## 4. Regenerate the corpus

```bash
cd test/oracle
python gen_oracle.py <mode>          # e.g. manip  -> writes ../NumSharp.UnitTest/Fuzz/corpus/manip.jsonl
```

Sanity-check the op landed (compact JSON — parse it, don't grep for `"op": "…"`):

```bash
python - <<'PY'
import json, collections
c = collections.Counter()
for line in open("../NumSharp.UnitTest/Fuzz/corpus/manip.jsonl", encoding="utf-8"):
    if line.strip(): c[json.loads(line)["op"]] += 1
print({k: c[k] for k in ["flip","fliplr","trim_zeros"]})
PY
```

## 5. Build + run the gate

```bash
cd ../NumSharp.UnitTest
dotnet build -c Debug -f net10.0            # the csproj glob copies corpus/*.jsonl into bin/.../Fuzz/corpus/
dotnet test --no-build -f net10.0 --filter "FullyQualifiedName~FuzzCorpusTests.<Tier>"
```

A pass means every new case (all layouts, all 15 dtypes, Char woven via `char_tier`) is bit-identical to NumPy.
If a case is red, go to `references/triage.md`.

## 6. What you do NOT need to do

- No new test method — `FuzzCorpusTests.<Tier>()` already replays the whole `<tier>.jsonl`; you added cases to it.
- No Char wiring — `char_tier("<mode>")` re-runs `gen_<mode>` with the Char pool automatically.
- No CI change — CI replays the committed corpus; committing the regenerated `.jsonl` is the whole delivery.

## Adding a whole new TIER (rare)

When no existing `gen_<mode>` fits (a new op family with its own fixtures — mirror `gen_clip`/`gen_quantile`):

1. **New generator** — add `def gen_<newmode>(dtypes, layout_names): ...` in `gen_oracle.py`, returning the same
   case dicts. Reuse `layout_catalog.py` fixtures, or build op-specific ones if values matter.
2. **Dispatch it** — add an `elif mode == "<newmode>":` branch in `main()` that calls
   `write_jsonl(corpus_dir + "/<newmode>.jsonl", gen_<newmode>(...) [+ char_tier("<newmode>")])`, and add the
   `char_tier` branch if you want Char coverage.
3. **New test method** — add `[FuzzMatrix] public void <Tier>() => RunCorpus("<newmode>.jsonl");` in
   `FuzzCorpusTests.cs`.
4. **Register the opnames** in `OpRegistry.cs` as usual, regenerate (`python gen_oracle.py <newmode>`), build, run.

## Adding a new DTYPE or LAYOUT

- **Dtype**: widen the mode's dtype axis toward `ALL_DTYPES` in `gen_oracle.py` (most tiers already use it). Char
  and Decimal are handled by their own paths (see `regenerate.md`). Then regenerate + rerun.
- **Layout**: add a `(base, view)` builder to `layout_catalog.py` AND mirror it under the same name in
  `test/NumSharp.UnitTest/Fuzz/LayoutCatalog.cs` (the C# side must rebuild the identical view). Regenerate every
  affected tier.
