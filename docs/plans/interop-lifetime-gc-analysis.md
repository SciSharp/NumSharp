# Lifetime and garbage-collection analysis: the ONNX Runtime and ML.NET bridges

- **Status:** DONE 2026-10-03, branch `journey4` on top of `fbc8de92`. Scope: `NumSharp.Interop.OnnxRuntime` and
  `NumSharp.Interop.MLNet`, both UNRELEASED (see `.claude/CLAUDE.md`, "Unreleased packages").
- **Gates:**
  - `test/NumSharp.Tests.Interop.OnnxRuntime/LifetimeGcTests.cs` (9 tests);
  - `test/NumSharp.Tests.Interop.MLNet/LifetimeGcTests.cs` (2 tests);
  - the shared harness `test/NumSharp.Tests.Interop.OnnxRuntime/LifetimeHarness.cs`, linked into the ML.NET suite.
- **The question.** Can any sequence of caller actions, collections and finalizations, in any order the runtime may pick,
  make a bridge:
  1. read or write freed memory;
  2. free memory that something reachable still uses;
  3. give a reference back twice, or never;
  4. dispose an object the caller can still reach?

  This document states the model, lists every object that holds a reference on a buffer, and proves six properties for
  each. It names what the bridges cannot see. It also records the ten violations the analysis found, all fixed, with the
  evidence before and after each fix.

## 1. The model

### 1.1 NumSharp's reference counting

A buffer is an `UnmanagedMemoryBlock<T>`, and its `Disposer` holds the count (`_refCount`) and a single-shot `_freed` flag
(`src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock`1.cs`):

| Operation | What it does | Who uses it |
|---|---|---|
| `TryAddRef()` | CAS loop; fails once the count is −1 (released). The only liveness check. | every reference taker |
| `Release()` | decrement; at 0, CAS 0 → −1 and free (the buffer returns to `SizeBucketedBufferPool`) | deterministic paths: `Dispose`, `NDScope` |
| `Abandon()` | decrement, never free; the count may rest at 0 with the buffer allocated | finalizers |
| `~Disposer()` | frees the buffer if it was not freed yet | runs only once NOTHING reaches the buffer |

Two kinds of reference reach a buffer:

- **Counted**, holding one count: an `NDArray` takes one in its constructor, releases it in `Dispose`, and abandons it in
  its finalizer. Every bridge object of section 2 is counted.
- **Uncounted**: a bare `IArraySlice` or `UnmanagedStorage` (`GetData()`, `Storage.InternalArray`). It keeps the
  `Disposer` reachable, so `~Disposer` cannot free the buffer under it. It holds no count, though, so a deterministic
  `Release` to zero does free under it.

A **forward alias** (`ArraySlice.WrapShared`) is a slice whose reference counting is its owner's: every count, release and
abandon goes to the owner's `Disposer`, and the alias keeps that `Disposer` reachable.

An **external block** (`new UnmanagedMemoryBlock<T>(pointer, count, Action)`) runs its action in two cases: when its count
is released to zero, and when its `~Disposer` runs. The action cannot tell which.

**The rule NumSharp's core settled** (the `SlicingWithNegativeIndex1` use-after-free): a finalizer proves only that ITS
object became unreachable. It must give its reference back with `Abandon`, never claim the eager free, because an
uncounted alias may still be reachable. A deterministic `Release` keeps the eager free, because there the caller asserts
that nothing it is responsible for still reads the buffer.

### 1.2 What the runtime guarantees, and what it does not

- **Reachability and finalization order.** An object is collectable once no root reaches it. When a group becomes
  unreachable together, one collection queues every finalizer in the group. They run on the finalizer thread in an
  **unspecified order**, so one finalizer may find its peers already finalized.
- **Weak handles.**
  - A short weak handle (`GCHandleType.Weak`, `WeakReference`) is cleared by that same collection, before any finalizer
    runs.
  - A long weak handle (`WeakTrackResurrection`) survives until the object is actually reclaimed.
- **`ConditionalWeakTable<K, V>` is an ephemeron.** The table keeps V alive exactly as long as K is reachable, and a
  K → V → K cycle does not keep K alive.
- **JIT liveness.**
  - **Optimized code.** A local or argument (including `this`) is live only up to its last use. A raw pointer, an
    `IntPtr` or a `Span<T>` over native memory roots nothing. A caller does not report a delegate as live during that
    delegate's own invocation; inside the call, the delegate's target is live only up to the target's last use.
  - **Tier-0 code.** This is a method's first ~30 calls, until a background compile replaces it. Tier-0 reports every
    local and argument as live for the whole method, and a Debug build behaves the same. Liveness bugs therefore show
    only in code that has run for a while.
  - **Address-taken locals.** The current JIT reports a local whose address is passed to a call (a struct used as a
    return buffer, say) for the whole method. The runtime does not promise this.
- **Finalizer registration.** An object with a finalizer is registered when it is ALLOCATED. A constructor that throws
  still leaves the finalizer to run unless the constructor suppresses it.

### 1.3 What ONNX Runtime and ML.NET do

Probed at ORT 1.16.0 and 1.29.0, ML.NET DataView 2.0.1 and 4.0.2, and read in `refs/onnxruntime/csharp/`.

- **Which types have finalizers.** `OrtValue` and `InferenceSession` declare one. `DisposableNamedOnnxValue`,
  `NamedOnnxValue`, `OrtValueTensor<T>`, `DisposableList<T>` and `FixedBufferOnnxValue` do not.
- **`~OrtValue()` calls `Dispose(false)`.** That releases the native value (when owned) and **skips `_memHandle`**, the
  pin of a value made with `CreateTensorValueFromMemory`. `Dispose()` calls `Dispose(true)`, which disposes `_memHandle`
  too.
- **`InferenceSession.Run(…, IReadOnlyCollection<OrtValue> inputValues, …)` drops its managed inputs.** It copies each
  `v.Handle` into an `IntPtr[]` and runs on those, so nothing references the managed inputs during the native run.
- **`OrtValueTensor<T>.Pin()` returns a bare-pointer `MemoryHandle`.** This manager backs a legacy output's memory. Its
  pin roots nothing: the native buffer lives exactly as long as the tensor → manager → `OrtValue` chain.
- **ML.NET's `DataViewRow` declares no finalizer.** Its `Dispose()` calls `Dispose(true)` and then
  `GC.SuppressFinalize`.
- **ML.NET abandons cursors, and transforms hold getters rather than cursors.** `ShuffleRows` never disposes the cursor it
  shuffles.

## 2. Who holds what

| Object | References it owns | Taken by | Given back deterministically | Given back when unreachable | Kept reachable by |
|---|---|---|---|---|---|
| `OrtTensor` (`AsOrtValue`) | 1 on the exported buffer | `Pin` (`TryAddRef`) before the `OrtValue` exists | `Dispose`: disposes the `OrtValue`, removes it from the table, `Release`s, disposes an owned conversion | `~OrtTensor`: `Abandon` and the export counter, nothing else | the caller, or `ValueOwners[Value]` (a `ConditionalWeakTable`) while the `OrtValue` is reachable |
| `OrtTensor<T>` (`AsDenseTensor`) | 1 | the same | `Dispose`: disposes the manager (new spans and pins refused), `Release`s, disposes an owned conversion | `~OrtTensor`: `Abandon` | the caller, or the manager's back-reference while the `Tensor`/`Memory` is reachable |
| `UnmanagedMemoryManager<T>` | none | — | refuses new access once its handle is disposed | — | the `Memory<T>` and the `DenseTensor` |
| `PinToken`, one per `Memory.Pin()` | 1 | `Pin` (`TryAddRef`) | `Unpin`, once: `Release` when a live holder disposed the pin, `Abandon` when finalization did | `~PinToken`: `Abandon` | the `MemoryHandle` copies, ORT's `_memHandle` included |
| view of a NumSharp-backed tensor (`h.Tensor.AsNDArray()`) | 1, forwarded to the buffer | `TryAddRef` (the liveness check), then the view's own | `Dispose` → `Release` | `~NDArray` → `Abandon` | an ordinary NumSharp view |
| import of an `OrtValue` (`value.AsNDArray(ownsValue)`) | the view's external block | — | last `Release` → lease: disposes the value (when owning) | `~Disposer` → the same lease | the view; the lease roots the `OrtValue` |
| view of a foreign tensor (`AsNDArray(DenseTensor)`) | external block plus the tensor's pin | `tensor.Buffer.Pin()` | lease: unpin, then `KeepAlive(tensor)` | `~Disposer` → the same lease | the lease roots the tensor |
| `ToNDArray(OrtValue)`, `ToNDArray<T>(DenseTensor)` | none (copies), or the pin's | — | — | — | the call keeps its source reachable until the copy completes (`GC.KeepAlive`) |
| `NDArrayDataView` | 1 on the fed array's buffer | `Pin` | `Dispose`: `Release`, disposes an owned fed array | `~NDArrayDataView`: `Abandon` | the caller, its cursors, transforms, getters |
| ML.NET cursor | 1 (its own pin) | `TakeCursorPin` | `Dispose(true)`: `Release` | `Dispose(false)`: `Abandon` | its getters (closure → cursor); the cursor holds the view |
| `vbuffer.AsNDArray<T>()` | the view's external block plus a pinned `GCHandle` | `GCHandle.Alloc(Pinned)` | last `Release` → `handle.Free` | `~Disposer` → `handle.Free` | the view |

## 3. The properties, and why they hold

### P1. No finalizer frees memory that a reachable alias still reads, in any order

- **Finalizers abandon.** `~OrtTensor` (both handles), `~NDArrayDataView`, the cursor's `Dispose(false)` and
  `~PinToken` give their reference back with `Abandon` only.
- **Deterministic disposals reachable from a finalizer.** An import view's external block runs its lease from
  `~Disposer`. The leases dispose an owned `OrtValue` (its `Dispose(true)` disposes the pin the value holds) or a foreign
  tensor's `MemoryHandle`. The only thing on those paths that can touch NumSharp memory is a `PinToken`'s `Unpin`, reached
  when the value or the pin lies over a handle's `Memory`. `PinToken.Unpin` reads a short weak handle to itself: if the
  target is null, the call comes from finalization, so the token abandons. If the target is live, a live holder disposed
  the pin, so it releases.

  The test is sound in both directions:
  - The collection that made the token unreachable cleared its handle before any finalizer ran.
  - A holder able to call `Unpin` holds the token, so its handle is live. `GC.KeepAlive(this)` keeps the frame from
    dropping the token before the read.

  A long weak handle would survive into finalization and fail this test (mutant M21).
- **The leases' other actions free no NumSharp memory.** `handle.Free` releases the GC pin of a managed array, and
  `GC.KeepAlive` frees nothing.
- **`~Disposer` frees only an unreachable buffer.** Every alias, counted or not, references the `Disposer`.

So the only frees left are a deterministic `Release` to zero (the caller's assertion, as in NumSharp's core) and the
`~Disposer` of an unreachable block. Every finalizer body only decrements a count: an `Abandon` after a free is a no-op on
memory, and `_freed` makes the free single-shot. Therefore every order has the same outcome.

**Tested:** every finalization order of 8 ONNX scenarios (212 staged real collections) and 6 ML.NET scenarios (42); see
§6.1.

### P2. Every counted reference is given back exactly once

- **One flag per holder.** Each holder guards its give-back with one `Interlocked.Exchange` on a flag that `Dispose` and
  the finalizer share: `OrtTensor`, `OrtTensor<T>`, `NDArrayDataView`, the cursor, `PinToken`, and both `ImportLease`s.
- **A `MemoryHandle` is a struct, so its copies share one pin.** They share one `PinToken`, whose flag turns a second
  `Unpin` into a no-op. Before the fix the manager was every handle's pinnable, and each copy released a reference.
- **Constructors that take over a reference suppress their own finalizer if they throw.** This covers `OrtTensor` (its
  table registration) and `PinToken` (its handle allocation). Otherwise the finalizer of an object never handed out would
  give back the reference that the caller's cleanup also gives back. `UnmanagedMemoryManager<T>.Pin` releases the
  reference itself when no token could take it.

**Tested:** after every staged order, the count must be exactly 0 (−1 means it was freed under the alias, more than 0
means a reference was never given back). Also `TwoCopiesOfOnePin_Disposed_GiveOneReferenceBack`.

### P3. Nothing leaks

- **Dropped objects.** A dropped handle, view, cursor or pin abandons its reference in its finalizer, the count can reach
  0, and `~Disposer` frees the buffer once it is unreachable.
- **Pins ORT skips.** ORT's finalizer skips the pin of an undisposed value; the pin's own token gives the reference back.
- **Cycles.** The `ConditionalWeakTable` cycle is ephemeral, and the manager ↔ handle cycle is collected as a whole.

**Tested:**
- `DroppedPins_GiveTheirReferenceBack_WhenCollected`;
- the final collection of each held-object test;
- the count-0 check after every staged order.

### P4. Reachable means valid

| What the caller holds | Why it stays valid |
|---|---|
| `h.Value` | the table keeps the handle, and with it the handle's reference; the handle's finalizer no longer disposes the `OrtValue` |
| `h.Tensor` or `h.Memory` | the manager references the handle; the handle's finalizer no longer disposes the manager |
| `h.Source` (an owned conversion) | its own `NDArray` reference; the handle's finalizer no longer disposes it |
| a pin | its token's reference |
| a NumSharp-backed tensor's view | its own counted reference |
| an `OrtValue` made over `h.Memory` | its pin |
| an ML.NET view, cursor or getter | the closure → cursor → view chain, each link with a pin |

The count stays truthful while any of these lives, so `resize(refcheck: true)` refuses to reallocate.

**Tested:** `HeldOrtValue_…`, `HeldDenseTensor_…`, `HeldSource_…`.

### P5. No read outlives its owner's root

| Read | What keeps the owner reachable |
|---|---|
| `ToNDArray(OrtValue)` | `GC.KeepAlive(value)` after the copy |
| `ToNDArray<T>(DenseTensor)` | `GC.KeepAlive(tensor)` in the `finally` |
| the ML.NET vector and scalar getters | `GC.KeepAlive(this)` after the read |
| the ML.NET conversions (`ConvertPinned`) | a counted reference held across the read |

**Tested:** the in-flight tests (§6.2).

### P6. Concurrent transitions are safe

- **`Dispose` against the finalizer.** Each holder has one `Interlocked` flag.
- **`TryAddRef` against `Release`.** The CAS on the count against `Release`'s 0 → −1 CAS (core).
- **A pin racing the handle's `Dispose`.** `Pin` refuses once the manager is disposed, and `TryAddRef` refuses once every
  reference is gone.
- **A NumSharp-tensor view racing the handle's `Dispose`.** The view checks `IsDisposed` and then calls `TryAddRef`; the
  second check closes a window of a few instructions (M14).

## 4. What the bridges cannot see: the caller's part

- **B1. Native references are invisible to the GC.**
  - ORT's `Run` does not keep its managed inputs alive (§1.3); neither do `OrtIoBinding`, `SessionOptions.AddInitializer`
    and `OrtValue.CreateSequence`.
  - Keep the handle, or its `Value`, reachable for as long as native code may read it: a `using` around the run.
  - `InferenceSessionExtensions.Run` does this itself: it disposes its handles in a `finally` after the run.
- **B2. Raw pointers and spans do not root.** `pin.Pointer`, or a `Span<T>` taken from `Memory.Span`, is valid only while
  the object it came from is reachable and undisposed.
- **B3. A deterministic `Dispose` is an assertion.** Disposing the last counted holder frees the buffer at once, even
  under an uncounted alias the caller still holds. This is NumSharp core's rule.
- **B4. A user finalizer that disposes a bridge object** (against .NET's guidance) runs the deterministic path from
  finalization. The pin token detects this and abandons. The handles', views' and cursors' `Dispose` do not detect it,
  exactly like NumSharp's own `NDArray.Dispose` called from a finalizer.
- **B5. Resurrection.** An object resurrected by a user finalizer has already lost its short weak handle, so its token
  takes the abandon path. That frees later, never earlier.

## 5. Violations found, all fixed

| # | Violation | Evidence before | After | Fix |
|---|---|---|---|---|
| V1 | Bridge finalizers freed the buffer under a reachable alias (`Release` at zero). | Probe, real collection: the alias read another array's data, for `OrtTensor` and `OrtTensor<T>`. `NDArrayDataView` and the cursor depended on the order (the collection happened to pick the safe one). | Every staged order passes. | `Abandon` in every finalizer |
| V2 | The handle's finalizer disposed the owned Decimal conversion the caller held through `Source`. | `IsDisposed=True, [5]=777` | `IsDisposed=False, [5]=5` | the finalizer touches only the count |
| V3 | The handle's finalizer disposed the `OrtValue` the caller held; any native call on it then faults. | probe: the value was disposed | alive | `ValueOwners` table |
| V4 | The handle's finalizer disposed the manager under the caller's `DenseTensor`, and its dropped reference let `resize(refcheck)` reallocate under the tensor. | `ObjectDisposedException` after a mere GC | `[5]=5` | the manager → handle back-reference |
| V5 | An undisposed `OrtValue` over the handle's `Memory` kept its pin forever. | after collection: `IsUniquelyReferenced=False, resize refused` | `True, resize works` | `PinToken` and its finalizer |
| V6 | Two disposed copies of one `MemoryHandle` released two references, the second someone else's. | by analysis | test passes | the `PinToken` once-guard |
| V7 | `ToNDArray(OrtValue)` read freed native memory mid-copy. | the process died: `Internal CLR error (0x80131506) at System.Buffer.MemmoveInternal … at NDArrayOnnxInterop.ToNDArray(OrtValue)` | 7,302 rounds in 40 s, none bad | `GC.KeepAlive(value)` |
| V8 | The ML.NET cursor was finalized during its own getter, so its release could free the row under the copy. | 117 of 279 getter-only cursors | 0 of 322 | `GC.KeepAlive(this)` |
| V9 | A view of a NumSharp-backed tensor was leased like foreign memory: its pin's release ran from the view block's FINALIZER and could claim the eager free, and the import counter counted NumSharp memory. | by analysis; restoring the lease (mutant M13) fails the suite | passes | forward alias (`WrapShared`) |
| V10 | Found while writing this analysis. An owning import of an `OrtValue` made over a handle's `Memory`: when the import view was collected, its block's finalizer ran the lease, the lease called `OrtValue.Dispose(true)`, that disposed the pin, and the pin's `Release` ran from finalization. | 10 of 120 orders freed the buffer under the alias | 0 of 120 | `PinToken`'s weak handle to itself |

`ToNDArray<T>(DenseTensor)` also gained a `GC.KeepAlive(tensor)`. The window exists by analysis, but it could not be opened
on the current runtime (§7, M12).

## 6. How it is tested

### 6.1 Staged collections: the finalization-order harness

`LifetimeHarness.StageCollection` reproduces ONE real collection of a scenario's objects with the order chosen:

1. Each participant's own finalizer is suppressed, and the participants are handed to a finalizable conductor that nothing
   references.
2. The scenario is built in its own method, which returns only what the test keeps: the uncounted alias, a text and a
   report. A Debug build keeps every local alive until its method returns, hence the separate method.
3. `RunStagedCollection` collects. That makes the participants genuinely unreachable, so the collection clears their
   short weak handles, as it would in production.
4. The conductor's finalizer runs the participants' finalizers on the finalizer thread in the chosen order.

The test then takes the freed buffers back out of the pool and fills them with a sentinel. It reads the alias, then checks
three things: that the buffer was not freed, that the count is exactly 0, and that the export and import counters are back
where they started.

An earlier version ran the finalizers by hand on the test thread while every participant was still reachable. That cannot
see behavior that depends on reachability, and V10 depends on exactly that: under the hand-run harness, the fixed token
saw its handle alive and released.

| Suite | Scenario | Participants | Orders |
|---|---|---|---|
| ONNX | `AsOrtValue` | source, handle, `OrtValue` | 6 |
| ONNX | `AsOrtValue` of Decimal | Decimal source, conversion, handle, `OrtValue` | 24 |
| ONNX | `AsDenseTensor` | source, handle | 2 |
| ONNX | `AsDenseTensor` of Decimal | Decimal source, conversion, handle | 6 |
| ONNX | `Tensor.AsNDArray()` view | source, handle, view, the view's `Disposer` | 24 |
| ONNX | undisposed `OrtValue` over the `Memory` | source, handle, value, its pin token | 24 |
| ONNX | dropped `Memory.Pin()` | source, handle, token | 6 |
| ONNX | owning import of that `OrtValue` (V10) | source, import view, its `Disposer`, value, token | 120 |
| ML.NET | vector-column view | source, view | 2 |
| ML.NET | scalar-column view | source, view | 2 |
| ML.NET | Decimal view | Decimal source, conversion, view | 6 |
| ML.NET | `ToDataView` snapshot | snapshot, view | 2 |
| ML.NET | view with a cursor | source, view, cursor | 6 |
| ML.NET | Decimal view with a cursor | Decimal source, conversion, view, cursor | 24 |

That is 212 orders for ONNX and 42 for ML.NET. Both suites run their permutation test in under a second, because the
collections are young-generation ones.

### 6.2 In-flight probes

An in-flight window exists only when a collection lands between the owner's last use and the end of the read. Three things
decide whether a test can see one. Each of the first in-flight tests got at least one of them wrong and could not fail:
M11 and M18 survived them.

- **Optimized code.** Tier-0 code reports every local as live for the whole method, which closes the window in the code
  under test AND in the test's own caller frame. Both interop test hosts therefore run with tiered compilation off
  (`<TieredCompilation>false</TieredCompilation>` in their csproj), so every method is optimized from its first call.
  Warming the code up to tier 1 is not a substitute:
  - run alone, the warm-up sufficed;
  - in a whole-suite run, the getters never reached tier 1 (other tests' compile backlog and jitting), and the
    no-KeepAlive control was caught in 0 of 12 rounds, against 12 of 12 with tiering off.

  The posture costs about 2.5 s per suite (ONNX 13.3 → 15.6 s, ML.NET 36.3 → 38.9 s), and it runs every other bridge test
  against optimized code too. The getter probe's reader frames are also marked `AggressiveOptimization`, and both probes
  still warm up, for a host that does tier.
- **A window wide enough, and a free that shows.** The `OrtValue` copy now uses 16 MB values, where it used 128 KB ones.
  Freeing a block that size returns it to the operating system, so a read of it faults; the copy also lasts long enough
  for a collection to land in it. With the `KeepAlive` removed:

  | Value size | Collector | Outcome |
  |---|---|---|
  | 128 KB (the old size) | back to back | 1,335 pressured rounds, no fault |
  | 1 MB | back to back | four ~2.5 s runs survived (a single 8 s run crashed) |
  | 2 MB | back to back | four ~2.5 s runs survived |
  | 4 MB | back to back | crashed in 4 of 4 runs |
  | 16 MB | paced | crashed in 3 of 3 runs |
- **A collector that lets the work run.** One young collection per millisecond. Back-to-back collections starved the
  legacy round to 4 rounds in 3 s.

The ML.NET getter probe's verdict has no false positive. A round counts only if the cursor's export disappeared before
the getter **committed** its row into the caller's buffer, and the getter's `KeepAlive` comes after that commit. A
**control** getter of the same shape without a `KeepAlive` runs through the same harness first, and it is caught in 12 of
12 rounds. If the control were never caught, the window would not exist in that process and the test would be
inconclusive instead of green.

### 6.3 Held-object tests

Each held-object test keeps exactly one thing (`Value`, `Tensor` or `Source`), collects fully, and then checks three
things:
- the thing is undisposed;
- it reads its own values with the pool poisoned;
- the buffer's count still includes the handle, so `resize` refuses.

Each test then drops the thing and checks that the handle is collected and its reference given back.

## 7. Mutation campaign

Each mutant reverts one fix (multi-edit where one fix spans two places), in a detached worktree. It runs the affected
project's whole suite in Release, net10.0 (`scratchpad/mutate4.py`, the runner of the 2026-10-02 interop campaigns).

**18 of 21 killed. The 3 survivors are windows that cannot be opened today, explained in §9.**

| # | Mutant | Verdict |
|---|---|---|
| M01 | `~OrtTensor` releases instead of abandoning | killed (1 test) |
| M02 | `~OrtTensor<T>` releases instead of abandoning | killed (1) |
| M03 | `~OrtTensor` disposes the owned conversion again | killed (2) |
| M04 | `~OrtTensor<T>` disposes the owned conversion again | killed (2) |
| M05 | the `OrtValue` does not keep its handle reachable (no table entry) | killed (1) |
| M06 | the memory does not keep its handle reachable (no back-reference) | killed (1) |
| M07 | a dropped pin's finalizer releases instead of abandoning | killed (1) |
| M08 | a dropped pin has no finalizer | killed (2) |
| M09 | a pin releases on every `Unpin` (no once-guard) | killed (2) |
| M10 | the manager is every pin's pinnable again (the pre-fix pin) | killed (3) |
| M11 | `ToNDArray(OrtValue)` without its `KeepAlive` | killed (the test host crashes in the copy) |
| M12 | `ToNDArray<T>(DenseTensor)` without its `KeepAlive` | **survived**: unobservable on the current runtime |
| M13 | a view of a NumSharp-backed tensor is leased like foreign memory again | killed (1) |
| M14 | the NumSharp-tensor alias skips its liveness check | **survived**: race-only |
| M15 | `~NDArrayDataView` releases instead of abandoning | killed (1) |
| M16 | `~NDArrayDataView` disposes the owned conversion again | killed (1) |
| M17 | the cursor's finalizer releases instead of abandoning | killed (1) |
| M18 | the vector getter without its `KeepAlive` | killed (1) |
| M19 | the scalar getter without its `KeepAlive` | **survived**: no safepoint inside the window |
| M20 | a pin disposed from finalization releases (no reachability test) | killed (1) |
| M21 | the token's handle to itself is a long weak handle, so it survives into finalization | killed (1) |

The first run of the campaign, before the in-flight probes were redesigned (§6.2), killed 14 of 19. M11 and M18 survived
it: those tests could not fail, because their windows were too narrow or ran in tier-0 code. V10 was found by reading the
code while that run was in progress. Its fix depends on a weak handle, which the hand-run harness of the time could not
model, so the harness moved to staged real collections and the import-back scenario was added. M20 and M21 pin that fix.

## 8. Version and configuration matrix

The release configuration is what CI runs: the interop-test job's exact steps were replayed, with the global
`-p:NoWarn`, the floor on both TFMs from one build, then the current version on net10.0. No test was skipped there, so
the in-flight probes ran and their controls were caught.

| Suite | Dependency | net8.0 | net10.0 |
|---|---|---|---|
| ONNX Runtime, Release (CI replay) | ORT 1.16.0 (floor) | 209/209 | 209/209 |
| ONNX Runtime, Release (CI replay) | ORT 1.29.0 (current) | — | 209/209 |
| ML.NET, Release (CI replay) | ML.NET 2.0.1 (floor) | 207/207 | 207/207 |
| ML.NET, Release (CI replay) | ML.NET 4.0.2 (current) | — | 207/207 |
| ONNX Runtime, Debug | ORT 1.16.0 and 1.29.0 | 207 + 2 inconclusive | 207 + 2 inconclusive |
| ML.NET, Debug | ML.NET 2.0.1 and 4.0.2 | 206 + 1 inconclusive | 206 + 1 inconclusive |

The Debug rows are the in-flight tests declining to run: a Debug build keeps every local alive, so the windows they
probe do not exist there, and the tests say so.

## 9. What remains

- **M12 is not observable today.** The tensor's `Memory<T>` sits in an address-taken stack slot, so the JIT keeps the
  manager alive anyway. The `KeepAlive` is the contract, and the test remains as the net.
- **M14 is race-only.** The window between `IsDisposed` and `TryAddRef` is a few instructions long.
- **M19 has no safepoint inside its window.** After the scalar getter's last field read there is only one load: no call
  and no loop, so the collector cannot stop the thread there. The `KeepAlive` guards against a JIT that makes the method
  fully interruptible, and against a future edit that adds a call.
- **B1 to B5 are the caller's part.** The README and the docs page state B1 and B3 for users.
