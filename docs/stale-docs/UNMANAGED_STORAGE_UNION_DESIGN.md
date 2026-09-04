# UnmanagedStorage Typed-Slice Union — Design & Safety Proof

**Status:** IMPLEMENTED (journey3, 2026-08-24) · **Proposed:** 2026-08-24
**Change class:** internal memory-layout optimization · **Public API:** unchanged
**One-line:** collapse `UnmanagedStorage`'s 15 per-dtype `ArraySlice<T>` fields into a single
`[StructLayout(LayoutKind.Explicit)]` union — **−896 B per `UnmanagedStorage` instance**, zero
behavior change, zero public-API change.

> **Implementation outcome (measured, net10.0 x64 Release):** `UnmanagedStorage` BaseSize
> **1088 → 192 B**; per-array managed footprint (`int32×10`, storage + box + NDArray + Disposer +
> shape arrays) **1424 → 528 B** — the −896 B delta is exactly the 14 dead lanes. Scalar
> `GetInt32` ×100M: **130 → 113 ms** (no regression); `Alias(Shape)` 32.8 ns/alias (the replaced
> IL copier added a `ConcurrentDictionary` lookup + delegate invoke on top of the same field copy).
> Gates: full suite **14,043/0** (net10.0, CI filter) · **FuzzMatrix 88/88 zero diffs** ·
> net8.0 union/ARC/lifetime/weave leg 89/0 · GC stress **WKS + SVR + GCStress=0xC**: 2.4M
> arrays/views churned, 2.43M typed-lane verifications, 0 corruption · committed gate
> `test/NumSharp.Tests/Backends/Unmanaged/TypedSlicesUnionTests.cs` (6).
> Two corrections to §4 found during implementation: `GeneratedDelegates.cs` (kernel-cache census)
> also referenced `_storageAliasFieldCopiers` — its count/clear entries were removed with the file —
> and the grep's 4th hit (`test/NumSharp.Benchmark/ArrayAssignmentUnspecifiedType.cs`) is a false
> positive (its `_arrayBoolean` is an unrelated local `Boolean[]` field).
> §5.7 CI matrix: GREEN — Build and Release succeeded on the union merge commit `793949f6`
> (run 32761521466) and on the branch tip `8a039e8f` (run 32762118299, incl. the refreshed
> test-inventory dashboard the Deploy Docs gate requires for the 6 new tests). §5.8 ILVerify:
> woven-vs-unwoven delta **0** (4194 = 4194, normalized per-method sets identical), and
> pre-union-vs-post-union delta **0** modulo one compiler-generated closure renumbering
> (`<>c__1110`→`<>c__1106`, from deleting StorageAlias.cs). Post-merge with the `~NDArray`
> abandon fix (09b47a49 — methods only, no lane-struct field changes, layout proof intact):
> union/ARC/lifetime/weave gates 110/0 + a fresh-build GC-stress leg (600K arrays, 0 corruption).

---

## 1. Problem

Every `UnmanagedStorage` (one per `NDArray`) declares **15 strongly-typed `ArraySlice<T>` fields**,
one per supported dtype:

```csharp
protected ArraySlice<bool>    _arrayBoolean;   // UnmanagedStorage.cs:32
protected ArraySlice<sbyte>   _arraySByte;
...                                            // 15 total
protected ArraySlice<Complex> _arrayComplex;   // UnmanagedStorage.cs:46
```

Each `ArraySlice<T>` is a **64-byte** struct (it holds only pointers/counts, so its size is
independent of `T`). Fifteen of them = **960 bytes embedded by value in every storage instance**, of
which **exactly one is ever populated** (the lane matching the array's dtype). The other 14
(~896 B) are permanently `default` — dead weight carried by every array of every dtype.

### Measured baseline (net10.0, x64, `MethodTable.BaseSize`)

| Object | Size |
|---|--:|
| `UnmanagedStorage` instance | **1088 B** |
| — of which: 15 × `ArraySlice<T>` | 960 B |
| — of which: embedded `Shape` | 56 B |
| — of which: header + `InternalArray`/`Address`/`Count`/`_dtype`/`_typecode`/`_baseStorage`/`Engine`/flags | ~72 B |
| Per-array managed footprint, `int32×10` (storage + box + `NDArray` + `Disposer` + 2 shape arrays) | **1344 B** |

The 960 B also underlies the documented **"three synchronized address caches" footgun** (CLAUDE.md,
nditer `it[0]`): `Address`, `InternalArray`, and the 15 `_arrayXxx` are three redundant encodings of
the same pointer that must be kept in sync.

---

## 2. Chosen solution — the explicit-layout union

Replace the 15 fields with **one 64-byte union** in which all 15 lanes overlap at offset 0:

```csharp
[StructLayout(LayoutKind.Explicit)]
private struct TypedSlices
{
    [FieldOffset(0)] public ArraySlice<bool>    Boolean;
    [FieldOffset(0)] public ArraySlice<sbyte>   SByte;
    [FieldOffset(0)] public ArraySlice<byte>    Byte;
    [FieldOffset(0)] public ArraySlice<short>   Int16;
    [FieldOffset(0)] public ArraySlice<ushort>  UInt16;
    [FieldOffset(0)] public ArraySlice<int>     Int32;
    [FieldOffset(0)] public ArraySlice<uint>    UInt32;
    [FieldOffset(0)] public ArraySlice<long>    Int64;
    [FieldOffset(0)] public ArraySlice<ulong>   UInt64;
    [FieldOffset(0)] public ArraySlice<char>    Char;
    [FieldOffset(0)] public ArraySlice<Half>    Half;
    [FieldOffset(0)] public ArraySlice<float>   Single;
    [FieldOffset(0)] public ArraySlice<double>  Double;
    [FieldOffset(0)] public ArraySlice<decimal> Decimal;
    [FieldOffset(0)] public ArraySlice<Complex> Complex;
}

private TypedSlices _slices;   // replaces all 15 fields
```

Then a **mechanical rename** everywhere the typed fields are touched: `_arrayInt32` → `_slices.Int32`,
`_arrayDouble` → `_slices.Double`, etc. The getters keep their exact shape
(`_slices.Int32[_shape.GetOffset(indices)]`), so the read path is byte-for-byte identical.

### Why union (B) over drop-fields (C) / non-generic (D)

The three champions were measured end-to-end (see Appendix A). The union was chosen deliberately for
its risk profile:

| Criterion | **B — union** | C — drop fields | D — non-generic |
|---|---|---|---|
| Buffer / read semantics | **byte-identical (rename only)** | getter read-path changes | getter + owner change |
| ARC refcount reach | **untouched** (`InternalArray`) | untouched | **rerouted** (risk) |
| Public API (`Data<T>`/`GetData<T>`) | **untouched** | untouched | brushes `IArraySlice` surface |
| Per-array saving | −896 B | −960 B | −1016 B |
| Mechanical risk | **lowest** (rename) | medium | highest |

The union gives ~93% of the maximum saving while changing **no semantics** — every lane is still an
`ArraySlice<T>` written and read exactly as before; they merely share storage. That is the safest
path to the win.

---

## 3. Why it is safe — the argument

### 3.1 The overlap is CLR-legal (statically proven by successful type load)

`[StructLayout(LayoutKind.Explicit)]` with overlapping fields that **contain a managed reference** is
legal **iff**, at every byte offset, all overlapping fields agree on "object reference vs
non-reference," and overlapping references are of a compatible type. `ArraySlice<T>`'s layout is
**identical for every `T`** (all fields are `T*`/`void*`/`long`/`bool` plus an
`UnmanagedMemoryBlock<T>` whose only reference is a **non-generic `Disposer`** at offset 0). So across
all 15 lanes:

- **offset 0** — a `Disposer` object reference, **same type in every lane** ✓
- **all other offsets** — pointers/longs/bool, **non-GC in every lane** ✓

Because the single managed slot is same-type/same-offset, the loader accepts the type. **This has been
verified empirically**: instantiating a union of the 15 real `ArraySlice<T>` loads with no
`TypeLoadException` and round-trips a written lane (Appendix A, Exp 1). A successful load is the CLR's
own guarantee that the GC ref map is well-formed.

### 3.2 Aliasing is never crossed

A given `UnmanagedStorage` has exactly one dtype for its entire lifetime, and **only the matching lane
is ever written or read** (`_typecode` selects it in every ctor, setter, getter, and the alias copy).
The union's other 14 lanes are never touched, so lane-vs-lane aliasing is never exercised — the union
is used purely as "the one typed slice, in less space." Cross-lane reads/writes cannot occur by
construction.

### 3.3 The ARC refcount is untouched

Reference counting is reached **entirely through `InternalArray`**, never through the typed fields:

```
NDArray.InitializeArc() → Storage.InternalArray.TryAddRef()
                        → ArraySlice<T>.TryAddRef()  (ArraySlice`1.cs:583)
                        → MemoryBlock.TryAddRef() → Disposer.TryAddRef()
Dispose()/finalizer     → Storage.InternalArray.Release() → … → Disposer.Release()
```

`InternalArray` (the boxed `IArraySlice`) is **kept verbatim** by this change. The union only affects
the *typed convenience field*, which does not participate in `TryAddRef`/`Release`/
`IsUniquelyReferenced`/`IsReleased`. Therefore the refcounter cannot break. (Verified on baseline:
Appendix A, Exp 3 — create → view → shared refcount → dispose base → view survives → dispose view →
freed.)

### 3.4 Slicing and views are untouched

NumSharp's view model is driven by **`Shape` (offset + strides) + `Address`**, not by the typed field.
A view shares the buffer via `InternalArray.Slice()` (same `MemoryBlock`/`Disposer`) and carries its
window in `Shape`. The union changes none of that; the alias path merely copies the one typed slice as
before (now one struct assignment instead of a per-dtype IL copy — §4.3). Write-through, broadcast,
negative-stride, transpose, offset views all behave identically.

### 3.5 Byte-for-byte buffer semantics

No allocation path, no pointer arithmetic, no kernel, and no cast is touched. `Address`, `Count`,
`InternalArray`, `Shape`, and the unmanaged buffer are all unchanged. The differential-fuzz corpus
(bit-exact vs NumPy 2.4.2) must remain green with zero diffs — this is the primary correctness gate
(§5.3).

---

## 4. The change — exact edit inventory

Blast radius is **4 files** (the typed fields are referenced nowhere else — grep-confirmed: 196
occurrences across `UnmanagedStorage.cs`, `UnmanagedStorage.Getters.cs`,
`DirectILKernelGenerator.StorageAlias.cs`).

### 4.1 `UnmanagedStorage.cs`
- Delete the 15 field declarations (lines 32–46).
- Add the `TypedSlices` struct + `private TypedSlices _slices;`.
- Rename `_arrayXxx` → `_slices.Xxx` in: the 15 scalar ctors, the 15 array ctors, the
  `SetInternalArray(Array)` 15-case switch, and the `SetInternalArray(IArraySlice)` 15-case switch.
  (The `CopyTo` switches use `CopyTo<T>` — they do **not** reference the typed fields and need no
  change.)

### 4.2 `UnmanagedStorage.Getters.cs`
- Rename `_arrayXxx` → `_slices.Xxx` in the ~30 scalar getters (int[] + long[] overloads). Shape is
  unchanged: `public int GetInt32(int[] indices) => _slices.Int32[_shape.GetOffset(indices)];`

### 4.3 `UnmanagedStorage.Cloning.cs` — `Alias(Shape)` fast path
Replace the per-dtype IL copier with a single struct copy:
```csharp
// before (Cloning.cs:93):
DirectILKernelGenerator.GetStorageAliasFieldCopier(_typecode)(r, this);
// after:
r._slices = _slices;      // one 64 B copy; copies the one live lane, by value
```
`Alias()` and `Alias(ref Shape)` route through `SetInternalArray(InternalArray)` (which now assigns
`_slices.Xxx` via its switch) and need no further change; they may optionally be simplified to
`r._slices = _slices` as well.

### 4.4 `DirectILKernelGenerator.StorageAlias.cs` — retire
After §4.3 the IL copier has no caller (grep-confirmed single caller). Delete the file (and its
`_storageAliasFieldCopiers` cache). This removes ~140 lines and a `DynamicMethod` emit path.

### 4.5 Nothing else
`GetData<T>()`, `Data<T>()`, `AsSpan<T>()`, `ToArray<T>()`, the kernels, `InternalArray`, the ARC
plumbing, and the public API are all untouched.

---

## 5. Extensive verification plan (the proof of "stable and safe")

All steps run in an **isolated `git worktree`** (`git worktree add --detach HEAD`) so the main tree
and any parallel session are untouched. Build with the standard silent build; the NDScope weaver runs
post-compile as usual.

### 5.1 Build + type-load
- `dotnet build` Core clean (net8.0 **and** net10.0), 0 errors.
- A startup smoke that instantiates every dtype's storage forces the `TypedSlices` type to load on
  each target runtime → proves no `TypeLoadException` on the real type in the real assembly.

### 5.2 Size verification (the win)
- Re-run the `MethodTable.BaseSize` probe: `UnmanagedStorage` **1088 → 192 B**; per-array `int32×10`
  footprint **1344 → 448 B** (−67%). Assert the numbers.

### 5.3 Functional correctness — the primary gate
- **Full suite**: `dotnet test --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"` — all green.
- **Differential fuzz (bit-exact vs NumPy 2.4.2)**: `dotnet test --filter "TestCategory=FuzzMatrix"`
  — **zero diffs**. This is the strongest correctness proof; it exercises all 15 dtypes × layouts ×
  ops through the storage.
- **All-15-dtype storage exercise**: a dedicated test that allocates, writes, reads, slices, clones,
  and disposes one array of *each* dtype — the whole point of the union is that every lane still works.

### 5.4 ARC / lifetime
- `test/.../Backends/Unmanaged/ArcLifecycleTests.cs`, `DeallocationTests.cs`,
  `CloneRegressionTests.cs`, and `test/NumSharp.Tests.Interop/ShutdownLeakTests.cs` — all green.
- Explicit assertions: base+view share one `Disposer` (`IsUniquelyReferenced` flips 1→shared→1),
  dispose-base-view-survives, double-dispose idempotent, finalizer path frees.

### 5.5 GC stress — the key explicit-layout safety concern
The one novel risk of a union with a managed reference is **GC tracking of the overlapped `Disposer`
ref under a compacting/concurrent collector**. Static legality (§3.1) guarantees the ref map is
correct; this step proves it *dynamically*:
- Allocate ≥1e6 union-backed arrays + views across all dtypes; hold a rolling live set; force
  aggressive `GC.Collect(2, Forced, blocking)` + `WaitForPendingFinalizers` in a loop.
- Run under **Server GC + Concurrent GC** and **Workstation GC** (`DOTNET_gcServer=1/0`,
  `DOTNET_gcConcurrent=1`) so the compacting mover is exercised.
- After each GC wave, assert: every live array still reads its expected values (no corruption from a
  mis-updated `Disposer` ref), refcounts are consistent, no premature free, no crash/AV.
- Optionally run one pass under `DOTNET_GCStress=0xC` (JIT/GC stress) as a belt-and-braces check.

### 5.6 Slicing / views / broadcast
- Slicing, strided/negative-stride/transpose/broadcast views, write-through, `reshape`, `flip`,
  `np.split`/`unstack` children — existing tests + a targeted view battery, all green and value-exact.

### 5.7 CI matrix
- The PR must be green on the full CI matrix: **Windows / Ubuntu / macOS × net8.0 / net10.0**, and the
  **arm64** legs — explicit-layout + managed-ref-overlap is a CLR-uniform rule, but it is *proven* on
  each leg, not assumed.

### 5.8 Weaver / ILVerify interaction
- The NDScope weaver processes Core post-compile; confirm it weaves cleanly with the new struct and
  the ILVerify gate (woven-vs-unwoven delta = 0) stays green.

### 5.9 Perf neutrality
- Scalar-get microbench (100M reads) and a small `run_benchmark.py` slice: the union lane read must
  match the old typed-field read (≈175 ms baseline; Appendix A, Exp 4). No elementwise/reduction
  regression (the kernels use `Address`, unaffected). Confirm alias/view construction is no slower
  (the IL copier is replaced by a cheaper single struct copy).

---

## 6. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| GC mis-tracks the overlapped `Disposer` ref | very low (loader-enforced) | §5.5 GC stress on WKS+Server+Concurrent; static proof §3.1 |
| A missed `_arrayXxx` rename site | low | grep is exhaustive (3 files); build fails closed on any miss |
| Platform/arch CLR layout difference | very low | §5.7 full CI matrix incl. arm64 |
| Weaver chokes on the new struct | low | §5.8 |
| Hidden reflection on field name `_arrayInt32` | low | only `StorageAlias.cs` did (by name) and it is retired; grep confirms no other name-based access |

## 7. Rollback
Trivial and total: revert the 4-file diff. No data format, no serialized state, no public API changed,
so rollback is a pure code revert with no migration.

## 8. Acceptance criteria (all required)
1. Core builds clean on net8.0 + net10.0, 0 errors/new warnings.
2. `UnmanagedStorage` measures **192 B** (was 1088 B); per-array `int32×10` **448 B** (was 1344 B).
3. Full test suite green (minus OpenBugs/HighMemory).
4. **`FuzzMatrix` green with zero diffs** (bit-exact vs NumPy 2.4.2, all dtypes).
5. ARC/lifetime + slicing/view suites green.
6. GC-stress harness clean under WKS + Server + Concurrent GC (no corruption, no AV, refcounts sound).
7. CI matrix green (3 OS × 2 TFM + arm64).
8. Perf-neutral (scalar-get ≈ baseline; no elementwise/reduction regression).
9. `StorageAlias.cs` removed; no `_arrayXxx` references remain.

---

## Appendix A — measured evidence (POCs, this investigation)

All figures net10.0, x64, Release, `MethodTable.BaseSize` / `GC.GetAllocatedBytesForCurrentThread`.

- **Exp 1 — union loads & works.** A `[StructLayout(Explicit)]` union of the **15 real
  `ArraySlice<T>`** loads with **no `TypeLoadException`**, is **64 B** (vs 960 B), and round-trips a
  written `Int32` lane. → §3.1.
- **Exp 2 — projected storage sizes** (faithful mock classes): baseline **1088 B** → union **192 B**
  → drop-fields 128 B → non-generic 152 B.
- **Exp 3 — slicing + ARC invariants on baseline**: fresh owned array `IsUniquelyReferenced=true`;
  strided view shares buffer (both `false`, same base address, `offset=2`, `stride=2`); write-through
  works; dispose base → view still reads, now unique; dispose view → `IsReleased=true`. The typed
  fields participate in **none** of it. → §3.3, §3.4.
- **Exp 4 — scalar-get throughput** (100M reads): typed-field lane **175 ms** ≈ raw pointer;
  interface `GetIndex<T>` 567 ms (3.2× slower, and irrelevant here since the union keeps the typed
  lane). → §5.9.
- **Exp 5 — 4-layout POC battery**: baseline / union / drop / non-generic each pass the identical
  slice + write-through + full refcount-lifecycle battery; mini-model sizes 1016 / 120 / 56 / 80 track
  the real 1088 / 192 / 128 / 152.

POC sources: `scratchpad/{memdiag,exp_storage,exp_slice_arc,poc}.cs` (this session).
