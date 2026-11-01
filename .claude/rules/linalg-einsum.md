---
paths:
  - "src/NumSharp.Core/LinearAlgebra/**"
  - "src/NumSharp.Core/Backends/Default/LinearAlgebra/**"
  - "src/NumSharp.Core/Backends/{IBlasBackend,LinalgEngine,TensorEngine.LinearAlgebra,TensorEngine.EinsumFunc}.cs"
  - "src/NumSharp.Core/Exceptions/LinAlgError.cs"
  - "test/NumSharp.Tests/LinearAlgebra/**"
---

# Linear algebra and `np.einsum`

Sources: the facades `LinearAlgebra/np.{inner,vdot,vecdot,matvec,vecmat,outer,tensordot,cross,kron}.cs` and
`LinearAlgebra/linalg/np.linalg.{solve,inv,det,eig,svd,qr,cholesky,lstsq,norm,multi_dot,matrix_power,arrayapi}.cs`; the
product members of `Backends/TensorEngine.LinearAlgebra.cs`; np.outer's and np.tensordot's statements as
`TensorEngine.Outer` and the two `TensorEngine.Tensordot` forms (`Backends/TensorEngine.Numeric.cs`, run by
`Backends/Default/LinearAlgebra/Default.{Outer,Tensordot}.cs`); the gufuncs
`Backends/Default/LinearAlgebra/Default.Gufuncs.cs` with `GufuncGuard.cs` / `GufuncSignature.cs`; the `np.linalg` bodies
on the `TensorEngine.Linalg` sub-engine (`Backends/LinalgEngine.cs`, implemented by
`Backends/Default/LinearAlgebra/DefaultLinalgEngine.*.cs`, whose `DefaultLinalgEngine.cs` holds the `_assert_*` /
`_commonType` ports and the `CoreMatmul` / `CoreDot` calls of the main-namespace products); `Exceptions/LinAlgError.cs`.
The sub-engine's statements call members, never an np.* facade: a nested numpy.linalg function is its own virtual member
(`Inv`, `Solve`, `Svdvals`, `Norm`, ...), a main-namespace function the member its facade dispatches to (the Array-API
`outer` and `tensordot` forms call `TensorEngine.Outer` / `Tensordot` of their first operand's engine).

## What computes with and without a backend

Everything NumPy computes through OpenBLAS has its public API, NumPy's validation and an engine seam; whether the
numerics exist WITHOUT a backend depends on what a portable managed algorithm costs:

| | Members | Behaviour |
|---|---|---|
| Products (CBLAS) | `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `linalg.multi_dot`, `matrix_power` (n ≥ 0), the `linalg` Array-API forms, `linalg.norm` except matrix ord ∈ {2, -2, 'nuc'} | Managed kernels. With `NumSharp.Interop.OpenBLAS`, `dot`/`matmul`/`inner`/`vdot`/`vecdot`/`matvec`/`vecmat` run through cblas for float32/float64/complex128, byte-identical to NumPy (`tensordot`/`multi_dot`/`matrix_power` compose over `dot`/`matmul`). |
| LU family (LAPACK LU) | `det`, `slogdet`, `solve`, `inv`, and `tensorinv`/`tensorsolve`/`matrix_power` (n < 0) composed on them | A managed fallback (`Backends/Default/LinearAlgebra/ManagedLu.cs`): a right-looking unblocked LU (`getf2`) plus triangular substitution (`getrs`), allclose to NumPy (bit-exact for tiny matrices, a few ULP apart as they grow: Core has no blocked LU reproducing LAPACK's `gemm` accumulation). det is the `sign·exp(Σ log|Uᵢᵢ|)` fold (`det([[5.]])` is `4.999999999999999`); an exact-0 pivot → `LinAlgError("Singular matrix")` for solve/inv, `(0,-inf)` / det 0 for slogdet/det. The generic `<T,TOps>` core monomorphises for double and Complex (Single upcasts to double, NumPy's lite rule); the double hot loops are hand-written `Vector256<double>` + FMA kernels (a register-blocked GER for the O(m³) rank-1 elimination: 4 trailing rows a pass, the pivot-row vector reused, 4 independent `Fma.MultiplyAddNegated` chains; 4×-unrolled FMA axpys for the substitution), and past `BlockedThreshold` = 256 the double factorisation is blocked (`FactorDoubleBlocked`: panel getf2 + laswp + TRSM + a Schur GEMM on `SimdMatMul.MatMulDouble`); complex stays scalar. Faster than NumPy for small matrices, slower for large ones (the managed GEMM is the ceiling). With the backend they run LAPACK `gesv`/`getrf`, byte-identical to NumPy. `[Misaligned]`: a matrix whose zero pivot appears only after several eliminations (`[[1,2,3],[4,5,6],[7,8,9]]`) lands on ~1e-16 in the unblocked kernel where blocked LAPACK cancels to an exact 0, so det is ~0 and inv/solve return a large finite inverse instead of raising. |
| Other factorisations (LAPACK) | `cholesky`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `pinv`, `lstsq`, `qr`, `svd`, `svdvals`, `matrix_rank`, `cond`, `norm` matrix ord ∈ {2, -2, 'nuc'} | Without a backend: `NotSupportedException` from the `TensorEngine.Linalg` kernels, naming the NumPy API, the LAPACK routine and the `IBlasBackend.Try*` member (Core has no managed QR/SVD/eigensolver). Except an EMPTY operand (an empty matrix or a zero-size batch): NumPy's answer needs no arithmetic (shapes, zeros, the identity a full SVD's `U`/`Vh` or a complete QR's `Q` of an M×0 matrix carries, `lstsq`'s sequential sum of squares of `b` for an M×0 matrix), so every factorisation gives it in Core before the engine asks a backend (`Backends/Default/LinearAlgebra/LinAlgEmpty.cs`; gate the portable `linalg_empty.jsonl` tier, zero excuses, strict error types); `lstsq` with a non-empty `a` and an empty `b` still needs the backend (gelsd computes `a`'s singular values). With `NumSharp.Interop.OpenBLAS`: LAPACK for float32/float64/complex128 (`potrf`/`geev`/`syevd`/`heevd`/`gesdd`/`geqrf`/`orgqr`/`gelsd`), byte-identical to NumPy; float32 upcast to double and cast back as NumPy's `_commonType` forces (complex through the `z` routines). A float32 operand's complex `eig`/`eigvals` result is NumPy's complex64 (each component rounded to float32) carried in complex128 (`CollapseEig`). |

- The seam is `TensorEngine.Blas`, one property. `IBlasBackend.cs` declares the 15 linear-algebra `Try*` members (5
  products, 10 factorisations) as default interface methods returning false; `OpenBlasBackend` overrides all 15, and a
  third-party or real-only CBLAS backend still compiles. Default interface members are interface-dispatched and invisible
  on the implementing class (a test types its variable as `IBlasBackend` for that reason). LAPACK belongs on this
  property because it lives in the bundled binary: scipy-openblas ships `gesv`/`getrf`/`potrf`/`geev`/`syevd`/`heevd`/
  `gesdd`/`geqrf`/`orgqr`/`gelsd` beside the BLAS symbols. The callers are the product members of
  `Backends/TensorEngine.LinearAlgebra.cs` (with the shared `LinAlgHelper`) and the ten `protected virtual` kernels of
  `DefaultLinalgEngine` (`Backends/Default/LinearAlgebra/DefaultLinalgEngine.Kernels.cs`), and each reads `Blas` into a
  local before calling it.
- Validation runs before the throw, so the error contract is the same with or without a backend: `_assert_stacked_2d`,
  `_assert_stacked_square` and `_commonType` are ported verbatim (`LinAlgError("{n}-dimensional array given. Array must
  be at least two-dimensional")`, `"Last 2 dimensions of the array must be square"`, `TypeError("array type float16 is
  unsupported in linalg")`; Decimal and Char take float16's exit).

## NumPy behaviours to keep

- `np.inner` reports `b`'s shape already transposed: it is `matrixproduct(a, swapaxes(b,-1,-2))`, so
  `np.inner(ones((2,3)), ones((3,2)))` raises "shapes (2,3) and (2,3) not aligned".
- `np.vdot`'s length mismatch is a RESHAPE error, and its own length check is dead code: `array_vdot` flattens both
  operands through one reused `PyArray_Dims` buffer holding `-1`, and `_fix_unknown_dimension` writes the resolved length
  back into it, so the second flatten asks for `(a.size,)`: `vdot(ones(3), ones(5))` → "cannot reshape array of size 5
  into shape (3,)".
- `vecdot` reduces in the LOOP dtype, not NEP 50's accumulator (its loops are `'ii->i'`, so an int32 pair stays int32).
  `vecdot`/`vecmat`/`vdot` conjugate their first operand; `inner`/`matvec` do not.
- `axes=` names the core axes per operand and is the general form of `axis=`: one entry per input AND output
  (`np.vecdot(m, v, axes: [[-1],[-1],[]])`); the output entry may be omitted only where the output has no core axes
  (`vecdot` alone); both `axes` and `axis` raise `TypeError("cannot specify both 'axis' and 'axes'")`. For
  `matvec`/`vecmat` the output entry says where the result's core axis lands
  (`np.matvec(a, b, axes: [[-2,-1],[-1],[0]])` transposes the answer).
- `axis=`/`keepdims=` compute only on `vecdot` but exist on all three: `matvec`/`vecmat` raise `TypeError` for both
  (offered so that a ported call fails the same way with the same message). The keepdims text names input 1's core rank
  (1 for `matvec`, 2 for `vecmat`).
- `keepdims=True` composes with `axes=` on `vecdot`: the output gains the reduced core dimension (size 1), so a provided
  output entry must be length 1 (`[[1],[1],[1]]`) and says where the kept axis lands; omitted (`[[1],[1]]`), the kept axis
  goes to the end. An explicit empty output entry (`[[1],[1],[]]`) contradicts the kept dimension: AxisError "operand 2
  has 1 core dimensions, but 0 dimensions are specified" (the mirror image without keepdims: "has 0 … but 1 specified").
  The output axis is validated against the output rank. The whole interaction is NumPy's own check sequence, ported once
  for the four gufuncs (`GufuncGuard.CheckCoreDimensions`: keepdims / axis support, `_validate_num_dims`,
  `_parse_axes_arg` / `_parse_axis_arg`, `_get_coredim_sizes`, then the loop broadcast).
- `np.linalg.matrix_rank` below rank 2 is a PREDICATE: `int(not all(A == 0))`, so `matrix_rank([1,2,3])` is 1 and an
  empty operand is 0.
- `tensordot` has three rejections in NumPy's order: a repeated RAW axis value → `ValueError("duplicate axes are not
  allowed in tensordot")`, first (`([5,5],[0,1])` is a duplicate, not an index error); an out-of-range contraction axis →
  `IndexError("tuple index out of range")` (NumPy indexes the shape tuple with the raw axis, so `tensordot(a2d, b2d, 3)`
  and `([5,0],[0,1])` are index errors); mismatched extents and unequal list lengths → `ValueError("shape-mismatch for
  sum")`, and since the extent comparison short-circuits, a mismatch found earlier hides a later out-of-range axis
  (`([0,1],[0,5])` is the shape mismatch). A negative count contracts nothing (`axes=-1` ≡ `axes=0` ≡ the outer product).
- `np.linalg.multi_dot` requires every non-endpoint operand to be 2-D: NumPy promotes a 1-D first/last operand (row /
  column) and then runs `_assert_2d` over all of them, so a 0-D, a 3-D+, or a 1-D operand in a middle position raises
  `LinAlgError("{ndim}-dimensional array given. Array must be two-dimensional")` (without it the Cormen cost model would
  read a phantom `dimensions[1]`). `out=` is threaded into the final dot, so it takes the TWO-dimensional product shape
  even for a vector endpoint (`multi_dot([v, B, C], out)` wants `(1,k)` and returns a raveled `(k,)` view). Exactly two
  operands is a plain `np.dot` and skips the 2-D check.
- The `np.linalg` Array-API forms are not aliases: `linalg.trace`/`linalg.diagonal` reduce the LAST two axes where
  `np.trace`/`np.diagonal` take the first (a `(2,3,3)` stack gives `(2,)` against `(3,)`); `linalg.outer` demands 1-D
  where `np.outer` flattens anything; `linalg.cross` demands 3-vectors.
- `NDArray.matrix_power(n)` with n < 0 is `inv(a)**|n|` (binary exponentiation), raises `LinAlgError` for non-square
  input, and its identity has the operand's dtype. `np.dot`/`np.matmul` "not aligned" messages render shapes as `(2,3)`
  through `Shape.ToPythonTuple()` (`ToString()` renders for humans).
- `np.stack` expands each operand as given (three 0-d arrays stack to `(3,)`, gate `Creation/np.stack.Test.cs`);
  `linalg.cross` builds its result with `expand_dims` + `concatenate`, which gives the same answer.
- Type-only divergences (texts identical): reshape/alignment errors raise `IncorrectShapeException` where NumPy raises
  `ValueError`; `AxisError` derives from `ArgumentOutOfRangeException` with `paramName: "axis"`, so .NET appends
  `(Parameter 'axis')` to `.Message` (assert axis texts with wildcards).

## `np.einsum`

`LinearAlgebra/np.einsum.cs` (the facade), the abstract `TensorEngine.Einsum` (`Backends/TensorEngine.EinsumFunc.cs`,
NumPy's `_core/einsumfunc.py`), which `DefaultEngine` implements in `Backends/Default/LinearAlgebra/Default.Einsum.cs`
(the argument checks, the parse, the view path, then the contraction), and `LinearAlgebra/EinsumSubscripts.cs` (the
parser).
- The parser is a port of `PyArray_EinsteinSum`'s parsing block with `parse_operand_subscripts` and
  `parse_output_subscripts` (`numpy/_core/src/multiarray/einsum.cpp`). Every expression NumPy accepts parses and
  contracts; every expression NumPy rejects is rejected with NumPy's text. Both spellings work: the subscripts string and
  the sublist form `np.einsum(a, [0,1], b, [1,2], [0,2])`.
- The contraction goes through OpenBLAS like the other products, by construction: it is a port of NumPy's own
  `optimize=` path (`bmm_einsum` + `_parse_eq_to_batch_matmul` in `einsumfunc.py`), which reduces every pairwise
  contraction to a `matmul` (plus a single-operand einsum for diagonals, sums and transposes first), or to a broadcast
  `multiply` when nothing is contracted. With `NumSharp.Interop.OpenBLAS` the float32/float64/complex128 contractions are
  byte-identical to NumPy's `einsum(..., optimize=True)`. NumPy's default `einsum` (`optimize=False`) is its own C
  `sum_of_products` loop and not BLAS-identical even inside NumPy, so a float comparison with a plain NumPy einsum call is
  a ≤ 3-ULP comparison. There is no `TryEinsum` seam: N operands fold left to right in `TensorEngine.Einsum`, each step
  through `PairwiseContract`.
- Values: integer and boolean contractions are byte-exact everywhere; product-shaped cases (matmul, batched matmul,
  matvec, `ij,kj->ik`, tensor contractions, inner/outer/kron) equal `np.matmul`/`np.multiply` bit for bit. A pure float
  summation outside a product (`ij->i`, a diagonal then a sum, the accumulation across three or more float operands) can
  differ in the last ULP (it follows `np.sum` and the left-to-right fold). All 15 dtypes contract (Half/Decimal/Char
  through the scalar matmul paths; complex UNCONJUGATED). einsum keeps the operand dtype (`einsum('ij->i', int32)` is
  int32, not `np.sum`'s int64); two-operand promotion is `result_type`.
- The single-operand VIEW path is NumPy's: one operand + no `out=` + nothing summed → a view: `einsum('ii->i', a)` is
  writeable (`…[:] = 1` sets a's diagonal; einsum re-aliases from the OPERAND's storage, because `Storage.Alias()`
  inherits read-onlyness from the storage it aliases), a broadcast operand's view stays read-only, and
  `einsum('ij->ij', a)` is a distinct view object, never `a` itself (so `result.resize()` refuses, as NumPy's view
  does, instead of resizing the operand). On this path NumPy ignores `order=` and discards a
  `dtype=` request (`einsum('ii->i', int32, dtype=int64)` is the int32 view).
- `out=` joins NumPy's iterator, and `casting=` gates every cast with its texts. NumPy's `c_einsum` runs one NpyIter
  over the operands and `out=` (`NPY_ITER_COMMON_DTYPE`, out READWRITE), in this order: a read-only out raises `operand
  array with iterator write flag set is read-only`; the loop dtype is the common dtype of the operands AND the out when
  no `dtype=` is given (int8 operands with an int64 out are summed in int64); each operand must reach the loop dtype
  under the rule (`Iterator operand 0 dtype could not be cast from dtype('int64') to dtype('float64') according to the
  rule 'no'`, which is what makes `casting='no'` reject mixed dtypes); then the out is read into the loop dtype
  (`Iterator operand 1 dtype could not be cast from dtype('float64') to dtype('int32') ...` under `dtype=int32`) and
  written back from it (`Iterator requested dtype could not be cast from dtype('float64') to dtype('int32'), the operand
  1 dtype, according to the rule 'safe'`; out is operand `nop` in NumPy's numbering), all before anything is computed.
  A text `out=` goes to the text refusal, which takes the out into the common dtype and its casts. A same-rank `out=`
  extent mismatch is worded about the contraction (`einsum() output parameter has shape (…) but the contraction
  produces (…)`) where NumPy leaks NpyIter's "remapped shapes" text.
- `order=` resolves the computed result's layout: `'A'` and `'K'` give F-contiguous when every input is F-contiguous (an
  all-F matmul, hadamard or 3-operand chain is F even at the default `'K'`), C otherwise.
- The grammar half of the parse is cached by `(subscripts, operand ranks)` in a 4096-entry `ConcurrentDictionary`
  (NumPy caches its equation parse the same way, `lru_cache(2**12)`); the extent/diagonal validation runs per call
  against the shared immutable grammar.
- NumPy has two einsum parsers (the C one behind `optimize=False`, a Python one in `einsumfunc.py`) that word the same
  rejection differently; NumSharp reproduces the C one whatever `optimize` says.
- The label encoding is NumPy's: one `sbyte` per operand dimension; positive = the label's ASCII code at first
  occurrence, negative = the offset back to that first occurrence (a repeated label, a diagonal), 0 = a broadcast
  dimension from an ellipsis (`"abbcbc"` → `[97,98,-1,99,-3,-2]`).
- Grammar facts:
  - The two operand-count messages read backwards: one operand against two terms raises "more operands provided to
    einstein sum function than specified in the subscripts string".
  - An impossible diagonal has two messages: `nop == 1 && out == null &&` nothing summed takes NumPy's view attempt
    ("dimensions in single operand for collapsing index …"); anything else says "dimensions in operand {i} for …".
  - Every diagonal is validated BEFORE any cross-operand extent (NumPy collapses all diagonals first).
  - The implicit output is ASCII-ordered, so an upper-case label sorts before a lower-case one (`einsum("zA", a)`
    transposes a `(2,3)`; `einsum("Az", a)` does not).
  - The sublist alphabet is `'ABC…Zabc…z'`, upper case first (NumPy 2.4.2's `einsum_symbols`), so index order equals
    ASCII order (`einsum(a, [0,26])` on `(2,3)` keeps `(2,3)`).
  - Spaces are legal anywhere; a `>` without its `-` is scanned as a label and rejected as an invalid subscript; a lone
    `-` gets the missing-arrow message.
- `[Misaligned]`: when a label's extents disagree between operands, NumSharp raises NumPy's `einsumfunc.py` wording
  (`Size of label 'j' for operand 1 (3) does not match previous terms (4).`) where NumPy's C path leaks NpyIter's
  "operands could not be broadcast together with remapped shapes …".
- Pass the keywords by name (`np.einsum("ij->i", ops, @out: dst)`): they are keyword-only in NumPy, and a fully positional
  7-argument call matches both the `params` overload and the full one (scalars convert to `NDArray` implicitly).
- `np.einsum_path` (the statements: `TensorEngine.EinsumPath`, `Backends/Default/LinearAlgebra/Default.EinsumPath.cs`,
  on the first operand's engine; the facades, the planner and the path type:
  `LinearAlgebra/{np.einsum_path,EinsumPathPlanner,EinsumPath}.cs`), a route-for-route port of NumPy
  2.4.2's `einsumfunc.py`: `_parse_einsum_input`, the `_greedy_path`/`_optimal_path` searches with the `_flop_count` cost
  model, and the info-string builder. `np.einsum_path(subscripts, operands, optimize)` returns `(EinsumPath path, string
  repr)`; `EinsumPath` is a `readonly struct` (Deconstruct, indexer, `Steps`/`Count`, `ToList()` for NumPy's raw list,
  NumPy-format `ToString`). The path and every metric are byte-identical to NumPy. `optimize=` takes NumPy's whole
  `path_type` vocabulary in NumPy's decision order (`false`/`null`, `true`/`"greedy"`/`"optimal"`, a precomputed
  `EinsumPath`, a `("greedy"|"optimal", memory)` tuple), with NumPy's errors (`ValueError` for parse/shape rejections;
  `KeyError` (`Exceptions/KeyError.cs`) for an unknown path name, reproducing NumPy's `('Path name %s not found', '…')`;
  `RuntimeError` for an incomplete explicit path; `TypeError` "Did not understand the path: …" / "object of type '…' has
  no len()"; a bool memory limit is Python's int, accepted). `np.einsum` runs the same resolution
  (`EinsumPathPlanner.ResolveOptimize`) for any `optimize` but `False`, first, before the operands, `order=` and
  `casting=`, as NumPy's `einsum` calls `einsum_path` before `c_einsum`. The returned `EinsumPath` round-trips into
  `np.einsum(…, optimize: path)`, which validates it but still contracts pairwise left to right. The cost model is `BigInteger` (NumPy uses Python ints: products past `long` would
  wrap and derail the comparisons); extents stay `long`. The `.3e` FLOP/size formatting rounds half to even like Python
  (.NET's `"0.000e+00"` rounds half away). Ellipsis placeholder letters are drawn deterministically (NumPy's come from a
  hash-randomized set; only those display letters differ). `einsum_call=True` is not ported (NumPy-internal).
- Gates: `LinearAlgebra/EinsumPathTests.cs`, `EinsumSubscriptParityTests.cs` (the grammar), `EinsumContractionTests.cs`
  (the values against NumPy 2.4.2), and the `einsum` tier.

## Gates of the linear-algebra surface

- `LinAlgSignatureParityTests` reflects over the whole surface and asserts each function has an overload whose parameter
  NAMES appear in NumPy's ORDER, plus the defaults NumPy states concretely (renaming `UPLO` to `uplo` or swapping
  `keepdims` and `ord` compiles, passes every value test, and breaks every named-argument call ported from Python), and
  that no `np.linalg` member is missing. Two defaults C# cannot spell are a null sentinel, pinned behaviourally:
  `matrix_norm`'s `ord='fro'` and `vector_norm`'s `ord=2`. Four ufunc keywords are absent from the gufuncs on purpose:
  `casting`, `order`, `subok`, `signature` (NumSharp models none of them in its ufunc surface).
- `LinearAlgebra/{BlasProductApiTests, LinAlgErrorParityTests, LinAlgEngineSeamTests, LinAlgSignatureParityTests,
  EinsumSubscriptParityTests}.cs`; `LinAlgEngineSeamTests` pins the no-backend contract (every factorisation raises
  `NotSupportedException` with Core alone).
- The host-pinned `linalg_parity` tier (`oracle.md`): the eigen/SVD/QR/Cholesky family and the LU family, byte-exact on
  NumPy's pinned scipy-openblas at threads = 1, Inconclusive off the pinned host.
