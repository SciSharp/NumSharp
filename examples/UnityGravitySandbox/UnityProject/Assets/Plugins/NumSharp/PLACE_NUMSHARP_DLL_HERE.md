# Put `NumSharp.dll` in this folder

Unity references any managed assembly dropped under `Assets/Plugins/` from the default
`Assembly-CSharp`, which is where this sample's scripts compile. So this is where the NumSharp
backend goes.

## Build the DLL

From the repository root:

```bash
dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0
```

Then copy the output into this folder:

```
src/NumSharp.Core/bin/Release/net8.0/NumSharp.dll   ->   Assets/Plugins/NumSharp/NumSharp.dll
```

`NumSharp.Core` has **no external runtime dependencies** (it is 100% managed C# — see the project's
`CLAUDE.md`), so `NumSharp.dll` is the only file you need.

## Runtime requirement (read this)

`NumSharp.Core` targets **.NET 8** and generates some of its compute kernels as IL at runtime
(`System.Reflection.Emit`). That means it needs a **JIT-capable, modern-.NET scripting backend**:

- ✅ **Unity 6's `.NET` / CoreCLR scripting backend** (the configuration this sample targets) — runs
  .NET 8 assemblies and JITs the runtime kernels. This is the recommended and tested path.
- ⚠️ **Mono** — Unity's Mono is `.NET Standard 2.1`-era; the `net8.0` build references APIs it does
  not provide (`System.Runtime.Intrinsics.Vector512`, generic-math interfaces, …) and will fail to
  load. You would need a `netstandard2.1`-targeted NumSharp build, which the stock package is not.
- ❌ **IL2CPP** — ahead-of-time compiled, so `Reflection.Emit` is unavailable and NumSharp's IL
  kernels cannot be generated at runtime.

See `../../../README.md` → "Getting NumSharp into Unity" for the full setup, and note that the
physics engine is independently verified without Unity via the `Verification/` project.

> This `.md` file is a placeholder so the folder is tracked in git. Delete it once `NumSharp.dll` is
> in place, or leave it — Unity ignores non-code files it isn't told to import.
