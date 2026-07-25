# Adding a NEW ufunc (elementwise op) end-to-end

The reusable ufunc plumbing (out=/where=/dtype=) covers *existing* ops. A **new** elementwise op is
a ~12-touchpoint change, and two of the touchpoints are non-obvious (the NDExpr fusion DSL and
DecimalMath). Archetype: `BinaryOp.ATan2` (np.arctan2) — `grep -rn "ATan2" src/NumSharp.Core` and
mirror every hit. For ops with a `Vector<T>` primitive (min/max-like), see how `Maximum`/`FMax`
emit instead of ATan2's scalar-Math-call pattern.

## The checklist (dependency order)

| # | File | What goes there |
|---|------|-----------------|
| 1 | `Backends/Kernels/KernelOp.cs` | `BinaryOp` (or `UnaryOp`) enum member, doc-commented with the np name and NaN semantics |
| 2 | `Utilities/DecimalMath.cs` | Decimal implementation (decimal-via-double bridge, or true decimal math when precision allows) |
| 3 | `Backends/Kernels/Direct/DirectILKernelGenerator.cs` | MethodInfo cache entry (fail-fast `?? throw` pattern) + the emission branch (`EmitATan2Operation`-style for scalar-Math-call ops; vector-method emission for SIMD-able ops) |
| 4 | `Backends/TensorEngine.cs` | The two abstract overloads: `(x1, x2, Type dtype)` and `(x1, x2, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)` |
| 5 | `Backends/Default/Math/Default.<Op>.cs` | The engine op implementation (promotion, kernel dispatch) |
| 6 | `Backends/Default/Math/DefaultEngine.BinaryOp.cs` | Engine dispatch wiring for the new enum member |
| 7 | `Backends/Default/Math/DefaultEngine.UfuncOut.cs` | Name registration (`BinaryOp.X => "npname"`) — this string appears in the verbatim ufunc cast-error texts |
| 8 | `Math/np.<family>.cs` (or a new `Math/np.<name>.cs`) | THREE overloads, matching arctan2: NumPy-shaped `(x1, x2, NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)` + positional-dtype `NPTypeCode` + `Type` (source-compat) |
| 9 | `Backends/Iterators/NDExpr.cs` | Factory (`public static NDExpr X(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.X, a, b)`) — without this, `np.evaluate` fusion silently can't use the op |
| 10 | `Backends/Iterators/NDExpr.Typing.cs` | The per-node result_type rule (e.g. `BinaryOp.X when intish => NDExprTypeRules.FloatTier(common)`) — must equal the ufunc's loop-dtype policy exactly |
| 11 | oracle (Phase 4) | `gen_oracle.py` binary/unary tier job + `OpRegistry.cs` case — see the `oracle` skill |
| 12 | benchmark (Phase 5) | C# `[Benchmark]` + NumPy twin — see the `benchmark` skill |

Unary ops mirror the same list through `UnaryOp` / the unary partials; the np API shape drops `x2`.

## Loop-signature policy — decide it from `ufunc.types`, not intuition

`print(np.<name>.types)` gives the whole dtype policy in one line. Example, copysign:

```
['ee->e', 'ff->f', 'dd->d', 'gg->g']     # float16/32/64/longdouble only
```

Read it as the NumSharp policy:
- **bool** rides the lowest float loop (`e` → Half): `copysign(True, True)` → `1.0` float16.
- **all ints** ride `d` (Double): `copysign(1, -1)` → `-1.0` float64.
- **complex absent** → input rejection with the **coercion** TypeError:
  `ufunc 'copysign' not supported for the input types, and the inputs could not be safely coerced
  to any supported types according to the casting rule ''safe''`
- **`dtype=` outside the loops** → the **no-loop** TypeError:
  `No loop matching the specified signature and casting was found for ufunc copysign`

Those are TWO DIFFERENT verbatim texts (input-coercion vs dtype-no-loop) — probe both, implement
both. The house error-order (probed for bitwise, project CLAUDE.md): bad `where` → no-loop →
out-cast → shape.

- **Decimal** (no NumPy loop): house decision — usually follow the float policy via the DecimalMath
  bridge; document it.
- **Char** (no NumPy dtype): house decision — usually the uint16 integer policy; the oracle's char
  proxy will then cover it automatically.

## Value-semantics traps to pin (probe, then test)

- Signed zero: `copysign(0.0, -1.0) = -0.0`, `copysign(-0.0, 1.0) = 0.0` — bit-exact, the oracle
  will catch a wrong sign bit.
- NaN: sign transfer applies to NaN too (`signbit(copysign(nan, -1)) = True`); NaN-propagating vs
  NaN-ignoring pairs (Maximum vs FMax) are distinct enum members, not flags.
- NEP50 weak scalars: `copysign(f2_array, -2.0)` stays float16 — python-float literals must not
  promote the loop.
