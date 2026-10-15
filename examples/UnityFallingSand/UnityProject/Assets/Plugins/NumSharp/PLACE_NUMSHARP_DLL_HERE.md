# Put `NumSharp.dll` in this folder

Unity references any managed assembly under `Assets/Plugins/` from the default `Assembly-CSharp`, where
this sample's scripts compile — so this is where the NumSharp backend goes.

## Build the DLL

From the repository root:

```bash
dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0
# copy bin/Release/net8.0/NumSharp.dll  ->  Assets/Plugins/NumSharp/NumSharp.dll
```

`NumSharp.Core` has no external runtime dependencies, so `NumSharp.dll` is the only file needed.

## Runtime requirement (read this)

`NumSharp.Core` targets **.NET 8** and generates some kernels as IL at runtime
(`System.Reflection.Emit`), so it needs a **JIT-capable, modern-.NET scripting backend**:

- ✅ **Unity 6's `.NET` / CoreCLR scripting backend** (this sample's target) — runs .NET 8 assemblies and
  JITs the kernels.
- ⚠️ **Mono** — `.NET Standard 2.1`-era; the `net8.0` build references APIs it lacks and won't load.
- ❌ **IL2CPP** — ahead-of-time; `Reflection.Emit` is unavailable.

See `../../../README.md` → "Getting NumSharp into Unity". The simulation is independently verified
without Unity via the `Verification/` project.

> This placeholder keeps the folder tracked in git. Delete it once `NumSharp.dll` is here, or leave it.
