# CounterExamples — Level D (the analyzer)

The compile-time guardrails cannot live in a runnable project, because their whole point is that the
**build fails**. This sibling project is therefore *deliberately un-buildable*: every method in
[`BadShapes.cs`](BadShapes.cs) places `[NDScoped]` / `[NDScopedAsync]` on a target the weaver cannot
weave, so the bundled Roslyn analyzer (`NumSharp.Weaver.Analyzer`) reports a build **error** at
`CoreCompile` — in the editor and before the weave.

It is **excluded from `SciSharp.NumSharp.sln` and from CI** (it would break the build). Exercise it
with the script, which turns "the build fails" into a green assertion:

```bash
bash show-analyzer.sh          # or:  dotnet run --project examples/NDScoping -- level D
# → prints each error NDW00x, then: ANALYZER-OK
```

## The shapes and their errors

| Method (`BadShapes.cs`) | Attribute + shape | Expected error |
|---|---|---|
| `RefEgress` | `[NDScoped] void M(ref NDArray a)` | **NDW002** — a hidden `ref NDArray` egress the weaver cannot see |
| `BadCarrier` | `[NDScoped] List<NDArray> M(…)` | **NDW003** — an unsupported carrier (collection); no way to see every NDArray |
| `NoBody` | `[NDScoped] extern NDArray M(…)` | **NDW005** — no body to weave |
| `SetterOnly` | `[NDScoped]` on a set-only property | **NDW006** — nothing to weave; put it on the getter |
| `AsyncUnderSync` | `[NDScoped] async Task<NDArray> M(…)` | **NDW009** — async belongs on `[NDScopedAsync]` |
| `SyncUnderAsync` | `[NDScopedAsync] NDArray M(…)` | **NDW010** — a plain sync method belongs on `[NDScoped]` |
| `HasBoth` | `[NDScoped][NDScopedAsync] …` | **NDW011** — a method has exactly one scoping model |

The IL-only rejections (NDW001 NDScope unresolved, NDW004 an unrecognized state-machine shape, NDW007
a tail-call, NDW008 a NumSharp too old for the async seam) are NOT source-detectable, so they stay
with the weaver post-compile and are not demonstrated here — see `tools/verify_weaver_package.sh`.

## Why only the analyzer is wired

The csproj references the analyzer (`OutputItemType="Analyzer"`) and NumSharp.Core, and nothing else —
no weaver import. The build never gets past `CoreCompile`, so there is nothing to weave; the analyzer
is independent of the weave (`-p:SkipNDScopeWeave` has no effect here). This is the same layer split
`verify_weaver_package.sh` step 11 uses: the analyzer preempts the weaver.
