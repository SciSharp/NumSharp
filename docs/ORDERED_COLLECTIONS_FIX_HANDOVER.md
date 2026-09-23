# Handover — revalidate and fix the ordered-collection bugs

**Branch:** `journey4` (draft PR #631). **Never commit on `master`.** Base: `fe756f9e` (the test commit).
**Scope:** `src/NumSharp.Core/Collections/Concurrent/` — `OrderedDictionary<TKey,TValue>` (lean, lock-free replace, value
stored once; 0 production users) and `ConcurrentOrderedDictionary<TKey,TValue>` (COD, the shipped node type).
`ConcurrentOrderedCompactDictionary` (COCD) is correct in every scenario below and is the reference behaviour.
`ConcurrentPointingDict` was deleted in `5635182d` — ignore it.

**Mission, in order:** (1) REVALIDATE each problem yourself — do not trust this document; (2) if confirmed, FIX it;
(3) flip its test from `[OpenBugs]` to a normal CI test in the same commit as the fix; (4) measure and record the cost.
If a problem does NOT reproduce (a parallel session may have fixed it), record that and skip it.

---

## 0. Ground rules (this repo runs several concurrent sessions)

- Build/test in an **isolated worktree**, never the shared tree:
  `git worktree add --detach <scratchpad>/wt HEAD`, build there, copy only your changed files back.
- Commit **path-limited**: `git add -- <paths> && git commit --only -F <msgfile> -- <paths>`, then
  `git show --name-only --format= HEAD` to prove only your files are in it. Never `git add -A`, never `--amend`.
- Before copying files back, re-check `git log <base>..HEAD -- <your files>` for overlapping commits.
- Commit messages: extensive; NO `Co-Authored-By` / "Generated with" lines (user rule).
- Every member you write or edit must carry full XML docs (summary with consequence/tradeoff, every param/return/
  exception) plus WHY comments on non-obvious bodies (user rule).
- `NumSharp.Core.csproj` NoWarns the whole CS1574 cref family, so the compiler never reports a broken `<see cref>`.
  Validate crefs with `docfx metadata docs/website-src/docfx.json` **in the scratch worktree** (never the shared tree —
  it rewrites the tracked `api/toc.yml`).
- Test files that use `OrderedDictionary<,>` must put `using NumSharp.Collections;` INSIDE the namespace block: on
  net10 it is otherwise ambiguous (CS0104) with `System.Collections.Generic.OrderedDictionary<,>` (.NET 9+).
- `dotnet run` probes: Release only (`dotnet run -c Release`), use a fresh filename after editing Core (the `#:project`
  runfile cache can run stale Core), and set `DOTNET_TC_CallCountingDelayMs=0` for timings.

## 1. Revalidate (step 1 — run before touching code)

All ten problems are pinned by committed tests (`fe756f9e`) that assert the CORRECT outcome and fail today:

```
dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj -c Release -f net10.0 \
  --filter "FullyQualifiedName~OrderedDictionaryOpenBugsTests|FullyQualifiedName~OrderedCollectionsComparerExceptionSafetyTests" \
  --logger "console;verbosity=detailed"
```

Expected today on net10.0 AND net8.0: **10 failed / 5 passed** (the 5 passing are COCD/OrderedDictionary controls that
must stay green). Read every failure message — each names its violated clause. Baseline for the whole folder under the
CI filter (`FullyQualifiedName~NumSharp.Tests.Collections&TestCategory!=OpenBugs`): **172/172** on both TFMs.

| # | Test (file) | Must fail today with |
|---|---|---|
| B1 | `Enumerate_WhileAWriterRegrowsTheTable_…` (OrderedDictionaryOpenBugsTests) | IndexOutOfRange + phantom default values within ms |
| B2 | `WideValueReplaceThenAppend_ReaderHoldingTheOlderGeneration_…` | `found=True, value=0` (expected not-found or 150.25) |
| B3 | `LockFreeReplace_RacingCopyingResizes_NeverLosesACompletedWrite` | "SetByKey(n) returned, then a read saw n-1" |
| B4 | `NullKey_IsRejectedOnEveryKeyEntryPoint_…` | all 11 entry points accept null |
| B5 | `HugeCapacity_IsRefusedWithArgumentOutOfRange_…` | ctor does not return within 10 s (spins) |
| B6 | `ComparerThrowDuringAResize_DoesNotLeaveTheReplacePermanentlyLocked` | replace blocks behind the lock after a comparer throw |
| B7 | `ConcurrentOrderedDictionary_InteriorRemoval_RefusedWriteBackMidReindex_…` (OrderedCollectionsComparerExceptionSafetyTests) | key 3 enumerates but does not resolve |
| B8 | `ConcurrentOrderedDictionary_SwapBackRemoval_…` | key 9 enumerates twice |
| B9 | `ConcurrentOrderedDictionary_WideSwapBackRemoval_…` | key 3 enumerates but does not resolve |
| B10 | `ConcurrentOrderedDictionary_RemoveWhere_…` | keys 0/2/4/6 enumerate but do not resolve |

Note B5 leaves a lowest-priority thread spinning until the test host exits (unavoidable on the buggy code).

## 2. Fixes

Line numbers are at `fe756f9e`. Remove `[OpenBugs]` from each test in the same commit as its fix.

### OrderedDictionary (`OrderedDictionary.cs`)

**B1 — enumerator reads the generation twice** (`:536`
`GetEnumerator() => new Enumerator(_t._values, Volatile.Read(ref _t._count))`). Fix: `Tables t = _t; return new
Enumerator(t._values, Volatile.Read(ref t._count));` (as `ToArray`/`AsValuesSpan` already do). Grep the type for any
other expression that reads `_t` twice.

**B2 — wide-value replace shares index/keys across generations** (`:361-368`: `Clone()` of values only, then
`new Tables(t._index, t._keys, values, t._count)`). The next in-place append writes the SHARED index word and key, and
an older-generation reader (no count gate by design) returns its own unwritten `values[slot]`. Fix: copy the WHOLE
generation (`t._index`, `t._keys`, `t._values` all cloned) — the rule `ConcurrentOrderedDictionary.COMPACT.md` §4.7
states and COCD follows. Same O(n) class as today. Do NOT fix it with a reader-side count gate (that throws away the
count-gate-free read the design is built on).

**B3 — lost updates: the lock-free replace is a store-buffering (Dekker) race.** Replacer (`:345-346`): plain value
store, then volatile reads of `_resizing` / `_t`. Resizer (`:428`, `:455`, `:364`, `:597`): `_resizing = true`
(volatile write), then copy. Release/acquire cannot forbid "each side's load misses the other side's store" (x86-TSO or
ARM64). Fix — a full fence on BOTH sides:
- replacer: `Interlocked.MemoryBarrier()` between `t._values[slot] = value;` and the `_resizing`/`_t` re-check;
- every resize: a full fence right after setting the flag, before the copy (make `_resizing` an `int` and use
  `Interlocked.Exchange(ref _resizing, 1)`, clear with `Volatile.Write(ref _resizing, 0)`).
Proof sketch: with both fences, "replacer read flag clear" ⇒ the resizer's flag store was not yet visible ⇒ its copy
had not started ⇒ the copy sees the value store. Measured in a scratch A/B: 3.26% lost → **0**, cost **+3.3 ns** per
replace (4.9 → 8.2 ns), still ~5× faster than a locked replace. Update the type remarks (they describe the protocol) and
proposal §7.1/§7.2.

**B4 — null keys accepted.** Add the siblings' `NullCheck(key)` (`if (!typeof(TKey).IsValueType && key is null) throw
new ArgumentNullException(nameof(key));` — the `typeof` guard avoids a Debug-only box) to: `TryGetValue` `:199`,
`GetByKey` `:231`, the `this[TKey]` accessors, `ContainsKey` `:266`, `IndexOf` `:271`, `TryAdd` `:278`, `SetByKey`
`:334`, `GetOrAdd` `:387`, `TryRemove` `:411`, and per pair in `AddRange` `:469`.

**B5 — capacity loop overflows `int`** (`:174` `while (len < cap*100/70+1) len <<= 1;`; also `AppendGrow` `:596`
`newLen <<= 1`). Compute the index length in `long`; if it exceeds `1 << 30` (the largest power-of-two `int[]`) throw
`ArgumentOutOfRangeException` — mirror COCD's `IndexLengthFor`. Guard the growth path the same way.

**B6 — `_resizing` left stuck when a resize throws.** Each resize sets the flag, allocates, re-hashes survivors through
the user comparer (`InsertIndex` → `GetHashCode`), publishes, then clears — no `try/finally` (`:428-439` TryRemove,
`:455-457` Clear, `:364-368` wide SetByKey, `:597-608` AppendGrow). Wrap each in `try { … } finally { clear flag }`.
Clearing after a failed resize is safe: nothing was published, so a replacer that stored into the live generation is
correct either way. Combine with B3's `int` flag.

### ConcurrentOrderedDictionary (`ConcurrentOrderedDictionary.cs`) — B7–B10

**Root cause.** Interior removal (`RemoveCoreUnderLock` `:1530`, map removal `:1563`, re-index loop `:1569`), both
swap-back branches (`TryRemoveSwapBack` `:827`; `_byKey.TryRemove` `:873`/`:895` then `RewriteIndexUnderLock`
`:879`/`:896`) and `RemoveWhere` (`:918`; removals `:962`, re-index `:971`, and the `finally` re-index `:992` before the
publish at `:998`) mutate the key map FIRST, then re-index the shifted or moved keys through
`RewriteIndexUnderLock` (`:1579`) → `GetValueRefOrNullRef` → **the user comparer**, and only then publish the store.
Any comparer throw there — including COD's own `LockRecursionException` from the documented evil-comparer refusal —
leaves the paths permanently inconsistent.

**Fix (recommended): resolve every node BEFORE the first mutation, then mutate throw-free.** A `ref` cannot be stored,
so add node-handle members to the NumSharp partial of the vendored clone (`ConcurrentDictionary.RefAccessors.cs`, never
the vendored body). It can see the private `Node`:
- `internal object? FindNode(TKey key)` — the same walk as `GetValueRefOrNullRef`, returning the node (calls the comparer);
- `internal static ref TValue NodeValue(object node)` — `ref Unsafe.As<Node>(node)._value` (no comparer);
- `internal bool TryRemoveNode(object node)` — unlink BY REFERENCE using the node's stored `_hashcode` for the bucket,
  under the clone's own stripe lock, updating counts exactly like `TryRemoveInternal` but without a comparer call.
Then, in COD:
- interior removal: collect the shifted keys' nodes (one lookup each — the same number of lookups as today, plus an
  O(n) handle array), remove the key, write indices through the handles, publish;
- swap-back: resolve the moved key's node before removing anything;
- RemoveWhere: pre-collect all nodes; in the loop, unlink by handle and write indices by handle, so the predicate is
  the only user code left (its throw → the existing `finally` publishes a consistent prefix, now throw-free).

Before changing RemoveWhere, read the existing test `RemoveWhere_ThrowingPredicate_LeavesBothPathsConsistent` and keep
its "removed everything matched so far" semantics. Keep the COCD controls green.

Simpler fallback (weaker): run a comparer pre-pass over every key to be re-indexed before mutating. It catches
deterministic comparers and the refusal attack, but not stateful comparers, and it doubles the re-index lookups.

**B11 (inspection only, no test) — out-of-memory can strand a wide-value replace.** `ReplaceExistingUnderLock` `:1492`:
for a non-atomic `TValue`, the map node is swapped (`:1511`) BEFORE `values.Clone()` (`:1514`) and the new `Store`
allocation, so an out-of-memory error leaves the key path ahead of the list path. Reorder: allocate and fill the new
values array and `Store` first, then swap the node, then publish (the order `AddRange`'s wide branch already uses).

## 3. Acceptance gates

1. The 10 tests pass on net10.0 **and** net8.0 (Release; also run Debug once), and the `[OpenBugs]` attributes are removed.
2. Re-run the three storm tests (B1, B3, and B6's harness) about 20 times in a loop: zero flakes.
3. The CI filter over the Collections folder is green: 172 → **182** tests on both TFMs.
4. The full `NumSharp.Tests` suite under the CI filter (`TestCategory!=OpenBugs&TestCategory!=HighMemory`) is green.
5. Measure and record in `ConcurrentOrderedDictionary.TODO.md` (COD) and the type remarks (OrderedDictionary):
   - OrderedDictionary `SetByKey` replace, before vs after (expect about +3 ns);
   - COD interior removal and swap-back, before vs after the node-handle change (expect about neutral);
   - the probes in `benchmark/collections/probes/` can be adapted for this; follow their methodology (pinned P-core,
     2.5 s spin-up, 150 ms warm-up per operation).
6. `docfx metadata` in the scratch worktree reports no new collection cref warnings.
7. Update `docs/proposals/ConcurrentOrderedDictionary.md`: §7.1/§7.2 must describe the fenced protocol. It must stop
   calling the implementation "complete" (its proposed API is a superset) and must cite these tests as the evidence.

## 4. Decisions for the user — do NOT change without asking

- The name `NumSharp.Collections.OrderedDictionary<,>` breaks compilation (CS0104) on .NET 9+ for consumers that import
  both namespaces and name the type (verified). Options: keep it and document, or rename it.
- `AddRange` semantics differ inside the family: `OrderedDictionary` keeps the first value (add if absent); COD and
  COCD keep the last value (upsert).
- `OrderedDictionary` accepts a negative capacity (documented as "use default"); the siblings and the BCL throw.
- Performance opportunity: `OrderedDictionary.TryRemove` is O(n) even for the last entry (no O(1) tail path), and both
  removal and growth re-hash every key (no stored hashes); COCD does neither.
- `GetValueRefOrNullRef` is public on the public vendored `ConcurrentDictionary`. This is intentional (`4499d3ea`); no
  action unless the user reverses it.

## 5. Background

- Design ledgers: `ConcurrentOrderedDictionary.TODO.md` (COD contract, memory, gates) and
  `ConcurrentOrderedDictionary.COMPACT.md` (compact-table research; §4 = the publication rules a port must keep).
- Test harness notes:
  - B2 is deterministic because `OrderedDictionary.TryGetValue` captures `_t` BEFORE calling `GetHashCode`, so a
    comparer hook can run the competing writer inside that window.
  - B3 needs exactly one writer per key re-reading its own writes; shared-key storms hide lost updates.
- A differential fuzz of all three types against a list-based reference (765K ops) found **no** functional bugs. The
  problems are concurrency, exception-safety and edge-contract only.
