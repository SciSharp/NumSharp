# Differential-oracle coverage map

Measured on `journey3` against NumPy **2.4.2** on 2026-08-22. This file describes the live
oracle, not the 2026-07 remediation baseline. Mechanical enforcement lives in
`OracleSurfaceCoverageTests.cs`: a new public `np`, `np.linalg`, `np.fft`, or `np.random` method
fails `FuzzMatrix` until it is directly covered or explicitly classified below.

## Current state

| Measure | Current |
|---|---:|
| Committed JSONL corpus | **116,339 cases / 64 files** |
| Ordinary op corpus (excluding index, Decimal, host-pin metadata) | **103,208** |
| Advanced indexing | **12,426** |
| Independent Decimal oracle | **703** |
| Char proxy rows | **5,506 across 20 files** |
| Distinct corpus op keys | **363** |
| Public surfaces inventoried | `np` 321 · `np.linalg` 31 · `np.fft` 18 · `np.random` 48 |
| Journey3 changed-file receipt | **186/186 touched callables have direct cases** |
| FuzzMatrix gate | **85/85 green on net8.0 and net10.0** |

The main corpus now includes deterministic creation (`creation.jsonl`, 302), conversion,
file artifacts, and finite-check errors (`conversion.jsonl`, 1,078), and arity-complete public
multi-output APIs (`multioutput.jsonl`, 64). The older claim that creation or tuple-returning functions are
structurally out of scope is obsolete: corpus cases may have zero operands and `kind:"tuple"`
records every result slot.

## Direct coverage added in the 2026-08-21 completeness pass

- Complex accessors: `conjugate`, `real`, `imag`, `angle` (radians and degrees) over the full
  dtype × 26-layout unary matrix, including the Char proxy.
- Deterministic creation: `arange`, `linspace`, `zeros`, `ones`, `full`, `eye`, `identity`,
  `zeros_like`, `ones_like`, `full_like`, dense/sparse `indices`.
- Conversion: `array`, `asarray`, `asanyarray`, `ascontiguousarray`, `asfortranarray`,
  `asarray_chkfinite`, `asmatrix`, `require`, `frombuffer`, `fromstring`.
- Multi-output contracts with arity: `split`, `array_split`, `hsplit`, `vsplit`, `dsplit`,
  `unstack`, `broadcast_arrays`, `meshgrid`, `unravel_index`, `modf`, `average(returned=True)`,
  and `unique_values`/`unique_counts`/`unique_inverse`/`unique_all`.
- Remaining public array surface: `copy`, `resize`, `column_stack`, `block`, `broadcast_to`,
  dtype metadata (`dtype`, `common_type`, `isdtype`, `issubdtype`, `mintypecode`, `isfortran`,
  `iterable`) and Array-API `vector_norm`/`matrix_norm`.
- Journey3 direct-coverage closure: the alias entry points `acosh`/`asinh`/`atanh`/`conj`;
  `empty`/`empty_like` via deterministic post-allocation fill; binary `fromfile`, parsed `loadtxt`,
  verbatim `savetxt`; materialized `nditer`/`nested_iters` traces; and full MT19937
  `seed`/`get_state`/`set_state` traces/restored draws. The exact 186-name list is
  `JOURNEY3_TOUCHED_FUNCTIONS.md` and is executable in `Journey3TouchedOracleCoverageTests`.

## Coverage-strength gates

Naming an op once is not sufficient. `OracleCoverageStrengthTests` now holds all **361 ordinary
op keys** (the two advanced-indexing schema keys are separate) to both:

1. at least **four committed cases**, and
2. variation in at least one independent axis: layout, parameter signature, operand-dtype
   signature, value class, or result/error outcome.

The eight former sub-four keys were expanded: `ravel_multi_index`, `unravel_index`,
`unravel_index_all`, `unique_values`, `broadcast_arrays`, and random `seed`/`get_state`/`set_state`.
The separate indexing engine keeps explicit floors of 2,000 curated, 100 getter-dtype, 10
setter-dtype, and 10,000 seeded-random cases. Per-file corpus floors remain additive.

## Techniques for increasing every op's coverage

Use these as orthogonal axes; adding more rows on the same happy path is the weakest option.

1. **Layout multiplication:** C/F contiguous, transpose, positive/negative stride, non-zero
   offset, stride-0 broadcast, scalar, empty, singleton, and high-rank. This is the default
   `layout_catalog.py` technique and should be widened before inventing bespoke fixtures.
2. **Dtype/promotion multiplication:** all 13 NumPy dtypes, Char through the uint16 proxy, and
   Decimal through the independent scalar oracle. Pairwise ops need mixed-width/signedness pairs,
   not only same-dtype rows; result dtype is asserted before values.
3. **Parameter-path covering arrays:** enumerate each overload/branch (`axis` scalar/tuple/None,
   keepdims, order, mode, casting, ddof, endpoint), then use pairwise or IPOG-style combinations
   where the full Cartesian product is too large. Parameter signatures are now measured by the
   strength gate.
4. **Boundary value classes:** NaN/±inf/±0/subnormal/max, integer min/max and wrap seams, ties,
   duplicates, singular/ill-conditioned matrices, zero-length contraction axes, and values that
   force every selection branch. Add a dedicated tier when the generic pool cannot make the op
   bite.
5. **Error-space inversion:** retain the cells value generators normally skip and compare NumPy's
   exception type and message. Generate one invalid neighbour per valid boundary: axis ±1 past the
   edge, malformed shapes/text, impossible broadcasts, bad casting/order/device, OOB indices.
6. **Object/protocol materialization:** turn iterators, state objects, named tuples, dtypes, flags,
   and planners into canonical arrays/tuples/text traces. `nditer`, `nested_iters`, and
   `get_state` demonstrate this technique.
7. **Artifact and side-effect replay:** compare emitted text/bytes or replay an exact input artifact
   through a temp stream/file. `fromfile`, `loadtxt`, `savetxt`, and the separate byte-identical
   NPY/NPZ oracle use this technique.
8. **Mutation/alias observation:** record the return value *and* the complete backing buffer for
   `out=`, `where=`, copy/scatter, overlap, and strided destinations. This detects corruption
   outside the logical view and proves write-through semantics.
9. **Metamorphic amplification:** add NumPy-free invariants beside direct examples—round trips,
   inverse pairs, transpose/reshape identities, decomposition reconstruction, set identities,
   and seeded state save/restore. These catch coherent generator/registry mistakes.
10. **Independent truth channels:** for numerically unstable reductions/transcendentals/products,
    carry a correctly rounded `Fraction`/mpmath/scalar reference used only after a NumPy
    divergence. It distinguishes parity debt from a real precision regression.
11. **Implementation variation:** replay managed C# and OpenBLAS, deduplicate only equal semantic
    outcomes (throw/result + dtype + shape + bytes), and adjudicate genuine flips against the
    pinned NumPy backend. Add CPU width/path forcing where one kernel has V128/V256/V512 routes.
12. **Seeded soak + shrink + pin:** nightly generation explores fresh shape/dtype/layout/parameter
    combinations; every failure is minimized and committed as a regression. Coverage-guided seed
    selection can favour still-unseen registry branches and strength-matrix cells.
13. **Alias/overload entry-point replay:** even when two names share an implementation, call the
    public alias/overload directly at least four times. This catches wrong default forwarding and
    signature drift that canonical-op coverage cannot see.

## Public `np` methods without their own corpus op name

These are deliberate classifications asserted by `OracleSurfaceCoverageTests`; none is an
unexplained omission.

### Equivalent aliases / synthetic corpus spellings

`absolute→abs` · `amax→max` · `amin→min` · `around→round_` ·
`bitwise_not→invert` ·
`broadcast→broadcast_values+broadcast_shape` · `common_type_code→common_type` ·
`concat→concatenate` · `degrees→rad2deg` · `radians→deg2rad` ·
`result_type→result_type_arrays/result_type_dtypes` ·
`true_divide→divide`.

### Owned by a stronger sibling gate or non-deterministic by definition

- Printing: `array2string`, `format_float_positional`, `format_float_scientific`,
  `get_printoptions`, `set_printoptions`, `printoptions` (the dedicated printing fuzz suite;
  `array_str`/`array_repr` are also in this corpus).
- Iterator objects: `flat`, `nditer_chunks` (NDIter unit/parity suites; materialized
  `ndindex`/`ndenumerate`/`nditer`/`nested_iters`/`broadcast` traces are in `iter.jsonl`).
- I/O: `load`, `load_npy`, `load_npz`, `save`, `savez`, `savez_compressed` (`NpyOracle`, reverse
  interoperability, and I/O tests). `fromfile`/`loadtxt`/`savetxt` now also have direct op keys.
- `evaluate` (NDExpr suite); `finfo`/`iinfo` (typing tests plus pinned open bugs).

### Compatibility-only / NumSharp-only names

`are_broadcastable`, `asscalar`, `find_common_type`, `issctype`, `issubsctype`,
`maximum_sctype`, `sctype2char`, `multithreading`, low-level `ndarray`, and `save_version` have no
same-named NumPy 2.4.2 callable. They are covered by compatibility/unit tests or indirectly by the
canonical operation.

All 31 `np.linalg` methods and all 18 `np.fft` methods now have direct corpus keys.

## Managed C# versus OpenBLAS variation and deduplication

The ordinary tiers always run with `TensorEngine.Blas == null`, proving NumSharp.Core's managed
kernels. Backend coverage is additive:

1. `matmul_parity.jsonl` (589) and `linalg_parity.jsonl` (366) are host-pinned byte-parity gates
   for the exact NumPy scipy-openblas binary/kernel/thread configuration.
2. `BlasBackendDelta` replays the **1,775** affected ordinary cases from `matmul`, `products`,
   `groupa`, `specials`, and `einsum` twice. Current result: **1,747 deduplicated** outcomes and
   **28 genuine backend changes**, all 28 byte-identical to NumPy on the pinned host.
3. Dedup compares outcome kind (threw/result), dtype, shape, and bytes. Equal payload bytes can no
   longer hide a backend-induced dtype/shape change. A change on a non-CBLAS dtype is always red;
   changed float32/float64/complex128 results are compared to NumPy when the host pin matches.

The factorisation family is not duplicated through the delta runner because Core has no managed
LU/QR/SVD/eigen fallback; its backend-on behavior is owned entirely by `linalg_parity`.

Corpus-level exact duplicate rows are small and intentional: **428 duplicate instances (0.373%)**
after ignoring case IDs, mainly the smoke subset of `astype_full` and seeded random collisions.
Runtime backend dedup is the material deduplication boundary.

## Remaining addable coverage gaps

These are the highest-value next expansions; they are parameter/value gaps, not unclassified
public methods.

The live strength census reports **64 ops below 10 cases**, **74 ops with one serialized layout**,
**39 ops with one operand-dtype signature**, and only **37 ops with a recorded error outcome**.
The sub-10 queue is:

- 4 cases: `array_split`, `average_returned`, `block`, `broadcast_arrays`, `column_stack`,
  `corrcoef`, `extract`, `get_state`, `indices`, `indices_sparse`, `insert`, `intersect1d`,
  `meshgrid`, `modf`, `nditer_values` (error-only compatibility key), `poly1d_coeffs`,
  `poly1d_fromroots`, `polyadd`, `polydiv`, `polyfit`, `polymul`, `polysub`,
  `ravel_multi_index`, `seed`, `set_state`, `setdiff1d`, `setxor1d`, `tensorsolve`,
  `union1d`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`,
  `unravel_index_all`.
- 5–7 cases: `einsum_path`, `roots`, `diag_indices_from`, `dot_aat`, `dot_ata`, `matmul_aat`,
  `matmul_ata`, `mintypecode`, `nested_iters`, `std_ddof`, `tensorinv`, `var_ddof`, `cond`,
  `unique`, `unravel_index`.
- 8–9 cases: `append`, `broadcast_to`, `compress`, `flatten`, `lstsq`, `matrix_norm`,
  `matrix_rank`, `polyder`, `put`, `unstack`, `vector_norm`, `cov`, `eig`, `eigvals`,
  `right_shift`.

Best next moves are layout multiplication for the set/poly/manipulation groups; singular,
batched, empty and complex matrices for linalg; cached-Gaussian/array-seed states for random;
additional order/mode/broadcast forms for index transforms; and invalid-neighbour generation for
the 324 ops with no error row.

1. Ufunc `dtype=` across the full ufunc × input-dtype × out-dtype matrix. `out_where.jsonl`
   already covers 28 ufuncs × seven output layouts × nine mask layouts, including the whole base
   buffer, but explicit loop-dtype selection is narrower.
2. Reduction tuple/multi-axis APIs once corresponding NumSharp overloads exist; broader `ddof` /
   correction values and output-buffer paths.
3. `clip` with one-sided and array-broadcast bounds; more `pad` modes and parameter combinations.
4. Creation/conversion error grids: invalid `device`, order/copy conflicts, negative/overflowing
   shapes, malformed text/buffer offsets. The value paths and `asarray_chkfinite` errors are now
   covered.
5. Structured `finfo`/`iinfo` field parity after the pinned `minexp` and uint64-max product bugs
   are fixed; these results need a structured/text comparator rather than an ndarray.
6. Iterator-object-only features (ranged/buffered iterator controls) beyond the materialized
   `nditer` and `nested_iters` traversal traces.
7. Decimal counterparts for newly-added creation/conversion/accessor cases. Decimal has no NumPy
   analog and requires expansion of `gen_decimal_oracle.cs`, not proxy relabelling.
8. More adversarial values for `unique_*`, `average(returned=True)`, `matrix_norm`/`vector_norm`,
   and creation boundaries (`linspace` subnormal/large endpoints, `arange` length boundaries).

## Random surface gaps

The stream corpus directly covers 35 distributions/methods. `random` aliases the covered
`random_sample`; `seed`/`get_state`/`set_state` now have direct corpus keys (boundary-seeded draws,
the full 624-word canonical state, and restored-state draws), while `RandomState` keeps its dedicated
constructor/state tests; Bernoulli
is a NumSharp extension (NumPy spells it `binomial(1,p)`). Seven public samplers remain carved and
pinned because their stream algorithms diverge: `binomial`, `f`, `multinomial`,
`multivariate_normal`, `negative_binomial`, `pareto`, and `standard_cauchy`. In addition,
`gamma(shape&lt;1, scale)` is a parameter carve while `standard_gamma` and `gamma(shape≥1)` are
directly byte-gated.

## Divergence policy

- Product bug: fix it when bounded and understood; otherwise carve the exact cell and pin it under
  `[OpenBugs]`. Never add a `MisalignedRegistry` excuse for a real defect.
- Intended/algorithmic difference: a tightly bounded `(op,dtype,kind)` registry branch, printed on
  every run, mirrored in `README.md`, and pinned from both sides in
  `MisalignedRegistryTightnessTests`.
- Generator/harness mistake: repair and regenerate; never excuse it.
- Host-dependent/undefined NumPy behavior: defuse or host-pin it rather than teaching NumSharp one
  machine's accident.

The full live divergence ledger remains `README.md`; `COMPLETENESS_PLAN.md` is retained as the
historical 2026-07 remediation record, not the current coverage source of truth.
