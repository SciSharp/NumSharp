# OpenBLAS Dependency — Delivery & Discovery Design

> **Status:** DESIGN (proposed) · 2026-08-13 · branch `journey3`
> Supersedes the discovery/precedence model documented in `docs/GEMM_PARITY.md` §6.1.
> Some pieces exist today (the bundle, the runtime discovery scan); the override system, the
> transitive delivery and the renames are NEW. Each section marks **[EXISTS]**, **[CHANGE]** or
> **[NEW]**.

This document defines how `NumSharp.Interop.OpenBLAS` obtains, delivers and binds an OpenBLAS/CBLAS
shared library, and how a *dependent* package can reuse the same delivery. It is the single source of
truth for the two questions that keep coming up: **"where does the binary come from?"** and **"which
one wins?"**

---

## 1. Goals & non-goals

**Goals**

1. **Bundled default, from PyPI, shipped in the nupkg** — a pinned scipy-openblas version is the
   zero-config default, packaged as per-RID NuGet runtime assets. No network at first run. **[EXISTS]**
2. **A consumer override, on the `PackageReference`** — pick a *version* (downloaded at build) or a
   *directory* (read in place), without a second package reference. **[NEW]**
3. **The override takes priority over the bundle**, and a version override is a **hard requirement**
   (fail if it cannot be satisfied). **[NEW]**
4. **Bundle is the last resort** — used only when no other tooling is found. **[CHANGE]**
5. **Never grab OpenBLAS out of a numpy installation** — remove the `numpy.libs` discovery tier. **[CHANGE]**
6. **The bundled binary and the build-downloaded binary are byte-for-byte the same artifact**, dropped
   into the same layout, so one transparently replaces the other. **[NEW/INVARIANT]**
7. **Keep no stray artifacts** — the downloaded `.whl` is extracted and discarded; only the extracted
   native library is cached. **[NEW]**
8. **Transitive delivery** — a package that depends on `NumSharp.Interop.OpenBLAS` can offer the exact
   same delivery to *its* consumers (metadata on *its* `PackageReference`), and can bake a chosen
   version into *its own* nupkg. Enabled by default, tunable per reference. **[NEW]**

**Non-goals**

- **No network at runtime, ever.** Autoinstall (§4) discovers and binds a *local* file; it never
  downloads. All fetching is a build/restore concern. (Verified today: zero network code in the C#
  runtime.)
- **No building OpenBLAS from source** at build time (a Fortran/C toolchain in `dotnet build` is out).
- **No non-scipy distributions** as a default — the pin is the scipy-openblas wheels NumPy itself uses.

---

## 2. The two phases

Everything below lives in exactly one of two phases. Keeping them apart is the whole design.

| Phase | Who runs it | What it decides | Network? |
|---|---|---|---|
| **BUILD** (MSBuild props/targets) | the consumer's `dotnet build`/`pack` (or a dependent package's) | *which* native binary lands in the output `runtimes/<rid>/native/` (and, optionally, in a nupkg), plus a **source marker** | **yes** (only for a version override) |
| **RUNTIME** (`BundleAutoinstall`) | the app process, at assembly load | *which* on-disk library to bind, and wiring the backend | **never** |

> "tooling" in this document = an OpenBLAS/CBLAS shared library on disk (`*.dll`/`*.so`/`*.dylib`).

The build phase writes files (and one marker); the runtime phase only ever *reads and loads*. A
version override is fetched at build and staged locally, so at runtime it is indistinguishable from a
bundle except by the marker (§8).

---

## 3. Delivery modes & priority — the heart of the design

Four ways a binary can be present, in strict priority order. **The first that yields a loadable
library wins.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME priority (BundleAutoinstall)                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ 1. BINDING     NUMSHARP_PARITY_BLAS / Blas.Enable(path)   exclusive, fatal miss│  [EXISTS]
│ 2. OVERRIDE    a) path   OpenBlasPath / NUMSHARP_OPENBLAS_PATH   read in place  │  [NEW]
│                b) version staged binary + REQUIRED marker → hard-required      │  [NEW]
│ 3. MACHINE     system installs (apt/brew/MacPorts/conda/vcpkg/source) + bare   │  [CHANGE]
│    TOOLING     loader names + PATH sweep     (NO numpy.libs)                    │
│ 4. BUNDLE      the nupkg's runtimes/<rid>/native default   ← LAST RESORT       │  [CHANGE]
└──────────────────────────────────────────────────────────────────────────────┘
```

Two things changed from the current implementation:

- **The bundle moved from #2 to #4.** Any tooling found on the machine now beats it. The bundle is a
  safety net, not the preferred answer. *(See §10.1 — this changes the parity-by-default story and
  needs your explicit sign-off.)*
- **numpy.libs is gone.** We do not read a pip-installed numpy's `numpy.libs/`. A conda/system
  *OpenBLAS* package is still machine tooling (tier 3); a *numpy* is not.

Build-time resolution decides what actually sits in tiers 2 and 4:

| `OpenBlasVersion` | `OpenBlasPath` | Build does | Runtime mode | Enforced? |
|---|---|---|---|---|
| — | — | nothing (ships bundle) | bundle (tier 4) | no |
| set | — | download → temp cache → copy to output `runtimes/<rid>/native`; write marker | version override (tier 2b) | **yes** |
| set | set | download → into `OpenBlasPath` (cache + read dir); write marker | version override (tier 2b) at `OpenBlasPath` | **yes** |
| — | set | record `OpenBlasPath` as the read dir; write marker | path override (tier 2a) | no |

**Environment always wins over `PackageReference` metadata** for the same knob
(`NUMSHARP_OPENBLAS_VERSION` beats `<OpenBlasVersion>`, `NUMSHARP_OPENBLAS_PATH` beats `<OpenBlasPath>`).

---

## 4. Bundle delivery (the default) — `BundleAutoinstall` **[EXISTS + CHANGE]**

The bundle is the pinned scipy-openblas version shipped as per-RID NuGet runtime assets
(`runtimes/<rid>/native/`), staged at *pack* time by `tools/fetch_openblas.py` from the PyPI wheels
NumPy pins (`openblas-manifest.json`). This is unchanged as the packaging mechanism.

**Renames [CHANGE]:**

| Today | New | Why |
|---|---|---|
| `Blas.AutoInstall()` (`[ModuleInitializer]`) | **`Blas.BundleAutoinstall()`** | it exists to make the *bundle* work with zero config; the name should say so |
| `NUMSHARP_BLAS_AUTOINSTALL=0` | **`NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0`** | matches the method | 
| — | old env kept as a **deprecated alias** for one release | migration |

`BundleAutoinstall` is still the single runtime entry point that runs the whole tier-1→4 discovery and
wires `OpenBlasBackend` onto `TensorEngine.Blas`; it is *named* for its defining default job (the
bundle), not limited to it. Referencing the package remains the entire opt-in, and a total failure
remains silent (fall back to NumSharp's managed SIMD GEMM).

**Behavioural changes:**

- **Bundle is tier 4** (§3): `BundleAutoinstall` only reaches the shipped `runtimes/<rid>/native`
  asset after an override and all machine tooling have declined.
- **`PythonLibDirectories()` is deleted** (numpy.libs), along with its use of `VIRTUAL_ENV`/`PYTHONHOME`
  for that purpose. `CONDA_PREFIX` survives only as a machine-tooling root (a conda-installed *openblas*,
  not numpy's).
- `NUMSHARP_BLAS_BUNDLED=0` continues to drop the bundle entirely.

---

## 5. Override system — `PackageReference` metadata + env **[NEW]**

The override is configured **on the `PackageReference`**, not with a second package:

```xml
<PackageReference Include="NumSharp.Interop.OpenBLAS" Version="0.60.0">
  <OpenBlasVersion>0.3.34.106.0</OpenBlasVersion>   <!-- env NUMSHARP_OPENBLAS_VERSION wins -->
  <!-- optional: read-from / download-to directory (relative or absolute) -->
  <OpenBlasPath>$(MSBuildProjectDirectory)/native/openblas</OpenBlasPath>
</PackageReference>
```

### 5.1 Knobs

| Metadata | Env (wins) | Meaning | Default |
|---|---|---|---|
| `OpenBlasVersion` | `NUMSHARP_OPENBLAS_VERSION` | scipy-openblas version to fetch from PyPI (enforced) | — (bundle) |
| `OpenBlasPath` | `NUMSHARP_OPENBLAS_PATH` | directory the tooling is **read from** — and **downloaded to** if a version is also set (relative resolves against the project dir) | — |
| `OpenBlasDistribution` | `NUMSHARP_OPENBLAS_DISTRIBUTION` | `scipy-openblas64` \| `scipy-openblas32` (`64`/`32` aliases) | per-RID from the manifest (win-arm64 = 32) |
| `OpenBlasFeed` | `NUMSHARP_OPENBLAS_FEED` | PyPI JSON base (mirror hook) | `pypi.org` |
| `OpenBlasSha256` | `NUMSHARP_OPENBLAS_SHA256` | expected extracted-lib hash | manifest hash (only for the pinned default) |
| `OpenBlasDelivery` | `NUMSHARP_OPENBLAS_DELIVERY` | `none` \| `build` \| `package` (§6) | `build` |

### 5.2 Path mode vs version mode

- **Path only (`OpenBlasPath`, no version):** read tooling from that directory at runtime (tier 2a),
  exactly like a build-baked `NUMSHARP_OPENBLAS_PATH`. **No version enforcement** — a path is "look
  here", not a contract. If the directory holds no loadable BLAS, discovery falls through to machine
  tooling and then the bundle.
- **Version (`OpenBlasVersion`):** the build downloads that version and stages it (§5.3). It is a
  **hard requirement** — if the build cannot download/verify/stage it, **the build fails** (`§10.5`),
  and at runtime a required override that fails to load **throws** rather than silently degrading.
- **Version + path:** download the version *into* `OpenBlasPath` (that directory doubles as the cache
  and the read location) instead of the default output folder.

### 5.3 Build-time pipeline (version mode)

```
resolve (env > metadata) → (distribution, version, rid)
  → PyPI JSON  <feed>/pypi/<distribution>/<version>/json   → pick the wheel for this RID's platform tag
  → download the .whl to a TEMP file
  → verify wheel sha256 (from the PyPI index)
  → extract the single  <dist>/lib/<native lib>  member
  → verify the extracted lib sha256 (against OpenBlasSha256 when provided)
  → write the extracted lib into the GLOBAL TEMP CACHE, keyed by (distribution, version, rid, sha)
  → DELETE the .whl                                   ← keep no artifact beyond the lib (Goal 7)
  → copy cache → output  runtimes/<rid>/native/<file>   (or → OpenBlasPath)
  → write the source marker (§8)
```

- Implemented as an inline `RoslynCodeTaskFactory` C# task shipped in the package's
  `buildTransitive` targets — **no Python, no extra dependency** (`System.IO.Compression` +
  `System.Text.Json` from the SDK).
- The **default path stages nothing and hits no network**: when no version/path is set, the bundle
  already ships. Only an actual override downloads.
- **Caching:** the global temp cache (`%LOCALAPPDATA%/NumSharp/openblas/` or `$XDG_CACHE_HOME`)
  makes repeated builds and multiple projects reuse one download. The cache holds **extracted libs
  only**, never wheels.

### 5.4 The "same binary" invariant (Goal 6)

The wheel→lib extraction is **identical** whether run at pack time (`fetch_openblas.py`, the maintainer)
or at build time (the inline task, the consumer): same per-RID distribution/platform-tag map, same
`/lib/` member pick, same output file name, same `runtimes/<rid>/native/` layout. That is *why* a
version override can transparently overwrite the bundle in the same folder — they are the same kind of
artifact. The **`openblas-manifest.json` is the shared contract** and is packed into the package so the
build-time task reads the exact same RID map the maintainer script used.

*(The source marker in §8 is separate metadata, not part of "the binary"; the binaries stay identical.)*

---

## 6. Transitive / third-party delivery **[NEW]**

The whole point of this section: a package `MyPackage` that depends on `NumSharp.Interop.OpenBLAS`
should give *its* consumers the identical override experience, and should be able to ship a chosen
version in *its own* nupkg.

### 6.1 The mechanism: `buildTransitive`

The download/stage logic ships in **`buildTransitive/NumSharp.Interop.OpenBLAS.{props,targets}`**.
NuGet flows `buildTransitive` assets to any project that references the package **directly OR
transitively**, so the targets run in the final consumer's build even when the reference is
`MyPackage → NumSharp.Interop.OpenBLAS`.

The targets scan **every `@(PackageReference)`** for the OpenBLAS metadata, not just the
`NumSharp.Interop.OpenBLAS` one. So this works unchanged:

```xml
<!-- consumer of MyPackage, which depends on NumSharp.Interop.OpenBLAS -->
<PackageReference Include="MyPackage" Version="1.0.0">
  <OpenBlasVersion>0.3.34.106.0</OpenBlasVersion>   <!-- honoured, delivers exactly as the direct ref -->
</PackageReference>
```

This is **enabled by default** (the transitive targets are active), and tunable with the
`OpenBlasDelivery` metadata (`none` to opt a reference out).

### 6.2 Baking into a dependent's own nupkg (`OpenBlasDelivery=package`)

When `MyPackage` is itself **built/packed**, the same transitive targets run (MyPackage references
NumSharp.Interop.OpenBLAS). If MyPackage sets `OpenBlasVersion` and `OpenBlasDelivery=package`, the
targets stage the binaries into MyPackage's output **and add them as pack content**
(`runtimes/<rid>/native/**`, `Pack=true`), so **`MyPackage.nupkg` ships its own OpenBLAS** at the
pinned version. MyPackage's consumers then get it from MyPackage's nupkg with no build-time download.

| `OpenBlasDelivery` | Effect |
|---|---|
| `none` | ignore OpenBLAS metadata on this reference |
| `build` (default) | stage into the **consuming app's** output only |
| `package` | ALSO include the staged assets in **this project's** nupkg (for package authors) |

> **Note — no double-bundling by default.** A dependent does **not** need `package` merely to expose
> the *default* version: NuGet already flows `NumSharp.Interop.OpenBLAS`'s bundled runtime assets
> transitively to the final app. `package` is for a dependent that wants a *different, pinned* version
> baked into its own nupkg. (See §10.4.)

---

## 7. Caching & cleanup **[NEW]**

- **Global temp cache** (per user), keyed by `(distribution, version, rid, sha256)`. Shared across
  projects and repeated builds. Holds **extracted native libraries only**.
- **Wheels are transient:** downloaded to a temp file, verified, extracted, then **deleted** (Goal 7).
  Nothing wheel-shaped survives a build.
- **Output staging** is a copy from the cache into `runtimes/<rid>/native/` (or `OpenBlasPath`),
  overwritable and idempotent (skipped when the destination already matches the keyed sha).

---

## 8. Runtime discovery & version enforcement **[NEW mechanism]**

`BundleAutoinstall` runs the tier-1→4 scan (§3). Tiers 2 and 4 both look at
`runtimes/<rid>/native/` (or `OpenBlasPath`); a **source marker** written by the build tells them
apart and carries the enforcement flag.

**Source marker** (recommended: a small sidecar, `openblas.source.json`, next to the app / the staged
binary — a pure "file placement" handoff, no `runtimeconfig` needed; alternative: a
`RuntimeHostConfigurationOption` AppContext value):

```json
{ "mode": "version|path|none", "distribution": "scipy-openblas64",
  "version": "0.3.34.106.0", "sha256": "…", "required": true, "path": "…" }
```

Runtime rules:

- `mode = version` → the folder holds a **required override** → check it at **tier 2b**, ahead of
  machine tooling. If it fails to load, **throw** (do not fall back) — the version was a contract.
  Optionally assert the loaded library's sha against the marker.
- `mode = path` → read from `path` at **tier 2a**; **not required** → fall through on miss.
- `mode = none` / no marker → the `runtimes/<rid>/native` folder is the **bundle** → it is reached
  only at **tier 4**, after machine tooling.

This is the one subtle bit: the *same folder* is high-priority when it is a required override and
last-resort when it is the bundle, and the marker is what flips it. It exists precisely because Goal 6
makes the two binaries indistinguishable by content.

The **binding** path (`NUMSHARP_PARITY_BLAS` / explicit `Blas.Enable(path)`) is unchanged and still
short-circuits everything (tier 1, exclusive, fatal on miss). Symbol-scheme probing inside a bound
library (`scipy_*64_` ILP64 → … → plain `cblas_*` LP64) is unchanged.

---

## 9. Naming & environment migration **[CHANGE]**

| Kind | Today | New |
|---|---|---|
| Method | `Blas.AutoInstall()` | `Blas.BundleAutoinstall()` |
| Env (opt-out) | `NUMSHARP_BLAS_AUTOINSTALL=0` | `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0` (old kept as deprecated alias 1 release) |
| Env (drop bundle) | `NUMSHARP_BLAS_BUNDLED=0` | unchanged |
| Env (binding) | `NUMSHARP_PARITY_BLAS` | unchanged |
| Env (override path) | `NUMSHARP_OPENBLAS_PATH` (runtime additive) | **override path** (runtime tier 2a **and** build read/download dir) |
| Env (override version) | — | `NUMSHARP_OPENBLAS_VERSION` (build) |
| Env (override misc) | — | `NUMSHARP_OPENBLAS_{DISTRIBUTION,FEED,SHA256,DELIVERY}` |
| Removed | numpy.libs scan (`PythonLibDirectories`) | deleted |

---

## 10. Consequences & open decisions

These are the calls that need your sign-off before implementation. Each has a recommendation.

### 10.1 Bundle-last changes the parity-by-default story ⚠ **biggest**

Moving the bundle to tier 4 means: on a machine with *any* system OpenBLAS (apt/brew/conda/vcpkg, or
one on `PATH`), the package now binds **that**, not the pinned bundle — and a distro OpenBLAS is **not**
byte-identical to NumPy 2.4.2. The current design's headline promise ("reference the package → parity
by default") no longer holds unconditionally. Parity is recovered explicitly by **pinning a version**
(`OpenBlasVersion` — a hard requirement, so it always wins over machine tooling) or by
`NUMSHARP_PARITY_BLAS`.
**Recommendation:** accept it as stated (this is a "use available tooling, bundle as fallback" model),
and add a one-line doc/log note that byte-parity now requires a pinned version. *Alternative to
consider:* keep the bundle **above** ambient machine tooling but **below** an explicit override (i.e.
"tooling" = override only), which preserves parity-by-default. **Confirm which you want.**

### 10.2 Multiple `PackageReference`s carrying conflicting `OpenBlasVersion`

With transitive scanning (§6.1), two references could disagree.
**Recommendation:** a direct reference wins over a transitive one; among peers, the **highest** version
wins; log the resolution. Hard-error only on an explicit `path` vs `version` conflict.

### 10.3 Marker mechanism: sidecar file vs `runtimeconfig`

**Recommendation:** sidecar `openblas.source.json` (file-placement, travels through publish, no
host-config plumbing). `RuntimeHostConfigurationOption` is the alternative if you prefer nothing on
disk beside the binary.

### 10.4 Does `OpenBlasDelivery=package` bundle even the *default* version?

**Recommendation:** no — `package` bundles only a **pinned** version into a dependent's nupkg; the
default flows transitively for free (§6.2 note). Avoids every dependent nupkg gaining 62 MB.

### 10.5 Offline / air-gapped builds with a version override

A version override needs the network (or a mirror via `OpenBlasFeed`) at build. By Goal 3 a failure is
a **hard build error**, so an air-gapped build that pins a version *and* lacks the cache/mirror fails
loudly. **Recommendation:** keep hard-fail (silent fallback to a different binary is worse); document
the mirror/cache-priming escape hatch.

### 10.6 Security

Downloading + later loading native code selected by build metadata is a supply-chain surface.
**Recommendation:** always verify the wheel sha (PyPI index) *and* the extracted-lib sha; require
`OpenBlasSha256` for any non-`pypi.org` `OpenBlasFeed`; never silently substitute.

---

## 11. Implementation status / checklist

| Piece | State |
|---|---|
| Bundle: per-RID nupkg runtime assets, `fetch_openblas.py`, manifest, double-hash verify | **EXISTS** |
| Runtime discovery scan (`AutoCandidates`) incl. system + PATH tiers, 32/64 names | **EXISTS** |
| Rename `AutoInstall` → `BundleAutoinstall` (+ env, deprecated alias) | new |
| Move bundle to tier 4 (last resort) | new |
| Remove `PythonLibDirectories` (numpy.libs) | new |
| `buildTransitive` props/targets + inline download/extract task | new |
| Override resolution (env > metadata), path & version modes, hard-fail | new |
| Global temp cache (extracted libs only) + wheel cleanup | new |
| Source marker + runtime tier-2b enforcement | new |
| Transitive metadata scan over all `PackageReference`s | new |
| `OpenBlasDelivery=package` pack hook (dependent's own nupkg) | new |
| Gates: build-override integration test; runtime priority unit tests; `MatmulParityBackendTests` unaffected | new |

---

## 12. References

- Current runtime discovery & binding: `src/NumSharp.Interop.OpenBLAS/CBlasNative.cs`
  (`AutoCandidates`, `BundledDirectories`, `SystemBlasDirectories`, `PathEnvDirectories`,
  `PythonLibDirectories` ← to be removed).
- Current autoinstall & backend wiring: `src/NumSharp.Interop.OpenBLAS/Blas.cs`
  (`AutoInstall` ← to be renamed, `Enable`, `Disable`).
- Bundle pack rules: `src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj`.
- Pack-time fetch & the shared extraction contract: `tools/fetch_openblas.py`,
  `tools/openblas-manifest.json`.
- Parity rationale (why the binary must match, three-knob determinism): `docs/GEMM_PARITY.md`.
