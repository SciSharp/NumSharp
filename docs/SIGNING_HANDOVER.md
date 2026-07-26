# Assembly signing — state, release-day runbook, and continuation

**Status as of 2026-07-26 (branch `journey3`, commits `478d550d` + `4ea939d9`)**

| Tier | What it gives you | State |
|------|-------------------|-------|
| **1 — Strong naming** | Assembly *identity*: version binding, `InternalsVisibleTo`, distinguishing this `NumSharp` from another | **DONE**, verified locally, gated in CI |
| **2 — Authenticode + NuGet author signing** | *Authenticity*: proof a binary came from SciSharp and was not tampered with | **NOT STARTED** — needs a code-signing certificate |

The two are independent. Tier 1 is a compile-time property of the assembly and needs no secret;
Tier 2 needs a purchased certificate and CI secrets. **Tier 1 does not make NumSharp "signed" in
the security sense** — see "What Tier 1 is not" below before writing any release note that claims
it does.

---

## 1. What Tier 1 changed

- **`Directory.Build.props`** (repo root, new) — `SignAssembly` defaults to `true` for every
  project; `AssemblyOriginatorKeyFile` resolves through `$(MSBuildThisFileDirectory)` so it is
  correct from any working directory. Opt out per project with `<SignAssembly>false</SignAssembly>`.
- **`Open.snk`** — consolidated from 5 byte-identical copies to 1 at the repo root.
- **All 7 `InternalsVisibleTo` now carry `PublicKey=`** — 5 in
  `src/NumSharp.Core/Assembly/Properties.cs`, 2 in `src/NumSharp.Interop.BLAS/AssemblyInfo.cs`.
- **`NumSharp.Interop.BLAS` is signed** (it consumes Core's internals).
- Removed the dead `#if !SIGNING` guard and the misplaced `SIGNING` define.
- **Gates:** `test/NumSharp.UnitTest/Assembly/StrongNameTests.cs` (6 tests) and
  `tools/verify_strong_name.cs` (package-level, run by CI).

### The key, precisely

`Open.snk` is **Microsoft's published open-source key** — byte-identical to
`dotnet/arcade`'s `src/Microsoft.DotNet.Arcade.Sdk/tools/snk/Open.snk` (596 bytes,
sha256 `b897629e0b20090c9219ac80392c037fa4cfcc2bdc21d3c45d9bf74b8df0f671`, verified by download).

```
PublicKeyToken : cc7b13ffcd2ddd51
```

Same identity as `netstandard`, `System.Memory`, `System.Buffers`, `System.Text.Json`. **TensorFlow.NET
signs with the identical key**, which is why NumSharp keeps it: a NumSharp-owned key would break the
cross-repo `InternalsVisibleTo("TensorFlowNET.UnitTest")` for no gain.

### What Tier 1 is **not**

The private half of this key is **public by design** — Microsoft publishes it so OSS builds can
produce strong-named output. Therefore:

- Anyone can mint an assembly with NumSharp's exact identity. That was *already* true when NumSharp
  was unsigned (`PublicKeyToken=null` is trivially impersonated), so this is not a regression — but
  it is not protection either.
- **`InternalsVisibleTo` is not an access control.** Private reflection defeats `internal` without
  any key anyway.
- **.NET Core and later do not verify strong-name signatures at load time**; the strong name is an
  identity string only.

Do not describe the 0.61.0 release as "signed for security". It is "strong-named", full stop.

---

## 2. What CI does now

| Job | Trigger | Signing relevance |
|-----|---------|-------------------|
| `test` (windows/ubuntu/macos) | every push, PR, tag | Runs `StrongNameTests` — **this is the cross-platform proof**, since the tests assert identity on all three OSes. Not excluded by the CI category filter. |
| **`verify-signing`** (new) | every push, PR, tag | Asserts `Open.snk` size+sha256, packs a **throwaway** Core/Bitmap/BLAS, runs `tools/verify_strong_name.cs`. Publishes nothing. |
| `validate-release` | tag `v*` only | unchanged |
| `build-nuget` | tag `v*` only | Builds+packs Core and Bitmap, then **`Verify strong naming of packed assemblies`** (new) before upload. |
| `create-release`, `publish-nuget` | tag `v*` only | unchanged |

`verify-signing` exists specifically because a release-only gate is how the original defect survived
six years. It costs ~1 minute and moves detection from release day to the offending PR.

### `tools/verify_strong_name.cs`

Cross-platform, dependency-free (reads PE metadata straight out of the `.nupkg`; no `sn.exe`, no
temp files). Checks each `lib/**/*.dll` for: the **full public key** (not just the token), the
correct token, and the `StrongNameSigned` CorFlag — which is what separates a real signature from a
**delay-signed** image that carries the key and an empty signature blob. Skips `runtimes/**/native/`
(OpenBLAS is unmanaged).

**It fails rather than passing vacuously** — no packages found, or a package with no `lib` assembly,
is an error. Verified to have teeth against three negative controls:

```
packages/NumSharp.0.40.0.nupkg (real, historically shipped)  → exit 1, "no public key"  ✔ caught
empty directory                                              → exit 1, refuses to pass   ✔
nonexistent path                                             → exit 1                    ✔
```

Run locally any time:

```bash
dotnet run tools/verify_strong_name.cs -- <dir-or-nupkg>     # default: artifacts/nuget
```

---

## 3. Verified locally (2026-07-26)

| Claim | Evidence |
|---|---|
| Solution builds clean, Debug **and** Release | 0 errors; no `CS8002`; all 553/1199 warnings pre-existing (CA1416, nullable, vulnerable transitives) |
| Every built assembly signed | **24/24** carry `cc7b13ffcd2ddd51` |
| Every **packed** assembly signed | **6/6** across `NumSharp` / `NumSharp.Bitmap` / `NumSharp.Interop.BLAS` |
| Release path unaffected by version overrides | Simulated `build-nuget` with `Version=0.61.0-rc.1 AssemblyVersion=0.61.0`, incl. `pack --no-build` → verify green |
| No test regressions | 12158 passed / 2 failed; **both attributed elsewhere** by rebuilding pristine `HEAD` in a throwaway worktree — `ErrorsFull` fails on `HEAD` too, `Index_Random` passes on `HEAD` and fails only against uncommitted in-flight indexing edits |
| In-repo `dotnet run` scripts inherit signing | Ran one with no signing properties → token `cc7b13ffcd2ddd51`, internal access worked |

---

## 4. What CANNOT be verified until release day

Everything above is local or PR-level. These only exercise on a real `v*` tag push:

1. **The tag → `validate-release` → `build-nuget` → `create-release` → `publish-nuget` chain**, with
   the new verify step inside it. The step is simulated locally with the exact commands, but not on
   a GitHub runner.
2. **nuget.org accepting a strong-named package.** No reason it would not — nuget.org is indifferent
   to strong names — but it has never been done for this package.
3. **Real consumer rebinding.** The identity change only bites downstream.

---

## 5. Release-day runbook

**Pre-flight (before tagging)**

- [ ] Resolve the open decisions in §6 — especially the `NumSharp.Interop.BLAS` packaging gap.
- [ ] Confirm `verify-signing` is green on the release commit.
- [ ] Version is **0.61.0** or **1.0.0** — *not* a patch bump. See §6.2.
- [ ] Release notes at `docs/releases/RELEASE_<version>.md` state the breaking identity change.
- [ ] Local dry run:
      ```bash
      dotnet pack src/NumSharp.Core/NumSharp.Core.csproj -c Release -o /tmp/rel
      dotnet pack src/NumSharp.Bitmap/NumSharp.Bitmap.csproj -c Release -o /tmp/rel
      dotnet run tools/verify_strong_name.cs -- /tmp/rel        # expect: verify_strong_name: OK
      ```

**Tag and watch**

- [ ] Tag from `master` (`validate-release` enforces this) and push.
- [ ] Watch `build-nuget` → step **"Verify strong naming of packed assemblies"**.
      Expect `assemblies checked: N   packages: M   failures: 0` and `verify_strong_name: OK`.
- [ ] **Abort criteria** — if that step fails, do **not** re-run the job. The artifact is wrong, not
      flaky. Delete the tag, fix, re-tag. The likely causes, in order: a csproj set
      `SignAssembly=false`; `Open.snk` missing from checkout; the `*.snk binary` rule removed from
      `.gitattributes` (line-ending corruption — the key-integrity step catches this first).

**Post-publish**

- [ ] Download the published package from nuget.org and verify the *shipped* bytes, not the CI copy:
      ```bash
      curl -sL -o /tmp/pub.nupkg https://www.nuget.org/api/v2/package/NumSharp/<version>
      dotnet run tools/verify_strong_name.cs -- /tmp/pub.nupkg
      ```
- [ ] Confirm `dotnet add package NumSharp` resolves and a trivial consumer builds.
- [ ] Notify TensorFlow.NET (§6.3).

---

## 6. Open decisions — settle these BEFORE tagging

### 6.1 `NumSharp.Interop.BLAS` is not in the release pipeline ⚠ **biggest gap**

`build-nuget` builds and packs **only** Core and Bitmap. The release-notes generator likewise lists
only those two packages. `NumSharp.Interop.BLAS` is a fully configured shipping package
(`PackageId`, `GeneratePackageOnBuild`, description, license expression) that exists **only on
`journey3`** — the release workflow predates it.

**As things stand, merging `journey3` and tagging would ship NumSharp 0.61.0 without its BLAS
package.** This is a packaging/release-content decision, not a signing one, so it was deliberately
**not** changed as part of the signing work. Decide one of:

- add the two `dotnet build` / `dotnet pack` invocations plus a release-notes row (the new
  `verify-signing` job already packs it and tolerates its absence on other branches); or
- ship it out-of-band; or
- hold it back from 0.61.0 deliberately.

Note it also bundles ~62 MB of OpenBLAS runtime assets per RID, staged by
`python tools/fetch_openblas.py` — which CI does **not** currently run. Packing it in CI without
that step produces an asset-free package (a warning, not an error) unless
`-p:RequireOpenBlasAssets=true` is passed.

### 6.2 Version must not be a patch

Identity changes `PublicKeyToken=null` → `cc7b13ffcd2ddd51`. Every consumer binding to NumSharp must
recompile; existing `bindingRedirect` entries and any assembly compiled against the unsigned
identity will not resolve. Ship as **0.61.0** or **1.0.0**.

### 6.3 Coordinate with TensorFlow.NET

Same maintainer, tightest coupling, and it references NumSharp *and* signs with the same key. It
needs a rebuild against the strong-named NumSharp. Warn before publishing, not after.

### 6.4 Make `verify-signing` a required check

It only protects `master` if branch protection requires it. Not configured.

---

## 7. Tier 2 continuation — Authenticode + NuGet author signing

This is the tier that delivers actual authenticity. **None of it exists yet.** It is additive and
non-breaking, so it can land in any later release independently of 0.61.0.

**Step 1 — obtain a certificate** (org decision, has a cost, cannot be done by an agent)

| Option | Rough cost | Notes |
|---|---|---|
| Azure Trusted Signing | ~$10/month | Cheapest sane path; no HSM to manage; integrates with a GitHub Action |
| OV code-signing cert | ~$200–400/year | Now requires hardware token / cloud HSM — awkward in CI |
| EV code-signing cert | ~$300–600/year | Immediate SmartScreen reputation; same HSM friction |

Must be issued to the organisation (**SciSharp STACK**), not an individual.

**Step 2 — sign the DLLs (Authenticode)** before `dotnet pack`, with a timestamp server so
signatures outlive certificate expiry.

**Step 3 — sign the package (NuGet author signing)**

```bash
dotnet nuget sign artifacts/nuget/*.nupkg \
  --certificate-path "$CERT" --certificate-password "$PW" \
  --timestamper http://timestamp.digicert.com
dotnet nuget verify artifacts/nuget/*.nupkg --all
```

**Step 4 — secrets**: certificate in GitHub Secrets or Azure Key Vault; never in the repo.
`.gitignore:220` already ignores `*.pfx` — keep it that way. A committed `.pfx` *would* be a real
security incident requiring revocation, unlike `Open.snk`.

**Step 5 — extend `tools/verify_strong_name.cs`** (or add a sibling) to assert the Authenticode and
NuGet signatures too, so Tier 2 gets the same package-level gate Tier 1 has.

---

## 8. Rollback

If a release ships and breaks consumers: nuget.org packages **cannot be deleted**, only **unlisted**.
Unlist the bad version and publish a follow-up. Note that *reverting* strong-naming is itself another
identity change — it does not restore compatibility with pre-0.61 consumers, it creates a third
identity. Prefer fixing forward.

---

## 9. Reference

| Thing | Where |
|---|---|
| Signing configuration | `Directory.Build.props` (repo root) |
| The key | `Open.snk` (repo root), `*.snk binary` in `.gitattributes:30` |
| Friend declarations | `src/NumSharp.Core/Assembly/Properties.cs`, `src/NumSharp.Interop.BLAS/AssemblyInfo.cs` |
| Unit gate | `test/NumSharp.UnitTest/Assembly/StrongNameTests.cs` |
| Package gate | `tools/verify_strong_name.cs` |
| CI | `.github/workflows/build-and-release.yml` → jobs `verify-signing`, `build-nuget` |
| Background | `.claude/CLAUDE.md` → "Strong naming (identity, NOT authenticity)" |
