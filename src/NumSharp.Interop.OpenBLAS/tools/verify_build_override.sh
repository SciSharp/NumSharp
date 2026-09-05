#!/usr/bin/env bash
# =============================================================================
# verify_build_override.sh — the BUILD-phase override delivery integration gate
# (docs/stale-docs/OPENBLAS_DELIVERY_DESIGN.md §5–§8; manual, like test/oracle/verify_npy_interop.py)
#
# Packs the REAL nupkg, restores it into a scratch consumer from a local feed, and
# proves the whole chain: buildTransitive import → PackageReference-metadata /
# env knobs → PyPI download + double-hash verify → global cache (wheel deleted) →
# staging over the bundle → source marker → runtime binds it as a REQUIRED
# override (OpenBlasEngine.IsBundledLibrary == false). Plus the adversarial half: the
# wrong-version and tampered-artifact hard fails, the mirror-needs-sha and
# invalid-knob rejections, the delivery=none opt-out, the two-reference conflict,
# poisoned-cache self-healing, marker lifecycle across mode switches, and both
# publish layouts.
#
# Needs: network (PyPI; the happy path caches after the first run — the tamper
# steps re-download one ~20 MB wheel per run by design), the .NET SDK.
#
#   bash src/NumSharp.Interop.OpenBLAS/tools/verify_build_override.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
REPO="$(cd "$PKG_DIR/../.." && pwd)"
WORK="$(mktemp -d)"
# On Git Bash/MSYS mktemp yields a /tmp/... path the WINDOWS dotnet/NuGet processes
# cannot resolve (it would silently break the restore of the scratch consumer) —
# normalize to a mixed C:/... form both worlds accept.
command -v cygpath > /dev/null 2>&1 && WORK="$(cygpath -m "$WORK")"
trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/build.log"
run() { "$@" > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "command failed: $*"; }; }

# The version is the repo-root Directory.Build.props VersionPrefix (the csproj carries no
# <Version> of its own any more) — ask MSBuild rather than grep a file that no longer has it.
VERSION="$(cd "$REPO" && dotnet msbuild src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj -getProperty:Version | tr -d '\r')"
[ -n "$VERSION" ] || fail "could not read the package version"
PINNED="$(sed -n 's/.*"distribution_version": "\([^"]*\)".*/\1/p' "$HERE/openblas-manifest.json" | head -1)"
FEED="$WORK/feed"; mkdir -p "$FEED"
CONS="$WORK/consumer"; mkdir -p "$CONS"

# The knobs must come from THIS script, not the ambient environment.
unset NUMSHARP_OPENBLAS_VERSION NUMSHARP_OPENBLAS_SEARCH_PATH NUMSHARP_OPENBLAS_DISTRIBUTION \
      NUMSHARP_OPENBLAS_PYPI_FEED_URL NUMSHARP_OPENBLAS_SHA256 NUMSHARP_OPENBLAS_DELIVERY \
      NUMSHARP_OPENBLAS_LIBRARY NUMSHARP_OPENBLAS_USE_BUNDLED NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL 2>/dev/null || true

step() { printf '\n== %s ==\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*|Windows_NT) OS_NAME=Windows ;; *) OS_NAME="$(uname -s)" ;; esac

# The consumer csproj is REWRITTEN per scenario (not sed-mutated): each step states its
# whole reference block, so no step depends on the previous one's edits.
write_csproj() {
cat > "$CONS/consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>disable</Nullable>
  </PropertyGroup>
  <ItemGroup>
$1
  </ItemGroup>
</Project>
EOF
}
REF_BARE="    <PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\" />"
REF_VERSION="    <PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\">
      <OpenBlasVersion>$PINNED</OpenBlasVersion>
    </PackageReference>"

step "pack NumSharp $VERSION + NumSharp.Interop.OpenBLAS $VERSION into a local feed"
# -t:Rebuild first: this repo's incremental build is known to declare stale outputs
# up-to-date after source changes, and dotnet pack defaults to Release on .NET 8+ —
# a stale Release DLL would get packed silently.
run dotnet build "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet build "$REPO/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet pack "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" --no-build -c Release -o "$FEED" \
    -v q --nologo "-clp:ErrorsOnly"
run dotnet pack "$REPO/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj" --no-build -c Release -o "$FEED" \
    -v q --nologo "-clp:ErrorsOnly"
ls "$FEED"/NumSharp.Interop.OpenBLAS."$VERSION".nupkg > /dev/null || fail "interop nupkg missing"

# The scratch consumer must restore THIS pack, not a same-versioned cache entry.
NUGET_ROOT="$(dotnet nuget locals global-packages --list | sed 's/.*: //' | tr -d '\r')"
command -v cygpath > /dev/null 2>&1 && NUGET_ROOT="$(cygpath -m "$NUGET_ROOT")"
rm -rf "$NUGET_ROOT/numsharp/$VERSION" "$NUGET_ROOT/numsharp.interop.openblas/$VERSION"

step "scaffold the consumer"
cat > "$CONS/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$FEED" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
EOF
write_csproj "$REF_BARE"
cat > "$CONS/Program.cs" <<'EOF'
using System;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

Console.WriteLine("enabled=" + OpenBlasEngine.Enabled);
Console.WriteLine("library=" + OpenBlasEngine.LibraryPath);
Console.WriteLine("bundled=" + OpenBlasEngine.IsBundledLibrary);
Console.WriteLine("dot00=" + (float)np.dot(
    np.arange(6).astype(NPTypeCode.Single).reshape(2, 3),
    np.arange(6).astype(NPTypeCode.Single).reshape(3, 2))[0, 0]);
EOF

OUT="$CONS/bin/Debug/net10.0"
RID="$(cd "$CONS" && dotnet msbuild consumer.csproj -getProperty:NETCoreSdkPortableRuntimeIdentifier | tr -d '\r')"
RID_MARKER="$OUT/runtimes/$RID/native/openblas.source.json"
ROOT_MARKER="$OUT/openblas.source.json"

step "1. default build: bundle ships, NO marker, no task"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
ls "$OUT"/runtimes/"$RID"/native/*openblas* > /dev/null 2>&1 || fail "bundled asset did not reach the output"
[ ! -f "$RID_MARKER" ] && [ ! -f "$ROOT_MARKER" ] || fail "a default build must write no marker"
# Where the build puts the per-user cache is the PACKAGE's decision (its buildTransitive props;
# readable only once the consumer has restored the package) — ask MSBuild rather than re-derive
# the rule here. It caught the Linux/macOS drift: MSBuild synthesizes a LOCALAPPDATA property on
# Unix, so an unguarded `$(LOCALAPPDATA)` rule landed the cache in ~/.local/share while every
# other party expected ~/.cache.
CACHE="$(cd "$CONS" && dotnet msbuild consumer.csproj -getProperty:NumSharpOpenBlasCacheDir | tr -d '\r')"
[ -n "$CACHE" ] || fail "the package's props did not define NumSharpOpenBlasCacheDir"
case "$OS_NAME" in
    Windows) ;;
    *) case "$CACHE" in "${XDG_CACHE_HOME:-$HOME/.cache}"/NumSharp/openblas) ;; *) fail "on Unix the build cache must be ~/.cache/NumSharp/openblas (the runtime loader's root), got $CACHE" ;; esac ;;
esac
echo "    cache: $CACHE"
echo "ok"

step "2. version override via PackageReference metadata ($PINNED): stage + marker + runtime override"
write_csproj "$REF_VERSION"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "version"' "$RID_MARKER" || fail "version marker missing/wrong at $RID_MARKER"
grep -q '"required": true' "$RID_MARKER" || fail "version marker must be required"
RUN="$(cd "$CONS" && dotnet run --no-build 2>/dev/null)"
echo "$RUN" | grep -q "enabled=True" || fail "backend not enabled at runtime: $RUN"
echo "$RUN" | grep -q "bundled=False" || fail "a marker-declared override must NOT report IsBundledLibrary: $RUN"
echo "$RUN" | grep -q "dot00=10" || fail "wrong product: $RUN"
echo "ok"

step "3. rebuild is a cache hit (no download)"
DL="$(cd "$CONS" && dotnet build -t:Rebuild -v n --nologo 2>&1 | grep -c "NumSharp.Interop.OpenBLAS: downloading" || true)"
[ "$DL" = "0" ] || fail "expected no downloads on a warm cache, saw $DL"
echo "ok"

step "4. a wrong version is a HARD build failure"
( cd "$CONS" && NUMSHARP_OPENBLAS_VERSION=9.9.9.9 dotnet build -v q --nologo "-clp:ErrorsOnly" > "$WORK/wrong.log" 2>&1 ) \
    && fail "a bogus version override must fail the build"
grep -q "hard requirement" "$WORK/wrong.log" || fail "the failure must state the contract; got: $(cat "$WORK/wrong.log")"
echo "ok"

step "5. a tampered artifact (wrong expected sha) is a HARD build failure"
# The wrong sha misses the cache, forcing a real download whose extracted library then
# fails verification — the tamper-detection proof (§10.6). Costs one wheel per run.
( cd "$CONS" && NUMSHARP_OPENBLAS_SHA256="$(printf '0%.0s' {1..64})" dotnet build -v q --nologo "-clp:ErrorsOnly" > "$WORK/sha.log" 2>&1 ) \
    && fail "a sha mismatch must fail the build"
grep -q "CHECKSUM MISMATCH" "$WORK/sha.log" || fail "the failure must name the mismatch; got: $(cat "$WORK/sha.log")"
echo "ok"

step "6. a non-pypi.org feed without an explicit sha is refused (before any network)"
( cd "$CONS" && NUMSHARP_OPENBLAS_PYPI_FEED_URL=https://mirror.example.invalid dotnet build -v q --nologo "-clp:ErrorsOnly" > "$WORK/feedx.log" 2>&1 ) \
    && fail "a mirror without OpenBlasSha256 must fail the build"
grep -q "requires an explicit" "$WORK/feedx.log" || fail "the failure must demand the sha; got: $(cat "$WORK/feedx.log")"
echo "ok"

step "7. an invalid OpenBlasDelivery value is refused"
( cd "$CONS" && NUMSHARP_OPENBLAS_DELIVERY=banana dotnet build -v q --nologo "-clp:ErrorsOnly" > "$WORK/deliv.log" 2>&1 ) \
    && fail "an invalid delivery value must fail the build"
grep -q "none|build|package" "$WORK/deliv.log" || fail "the failure must list the valid values; got: $(cat "$WORK/deliv.log")"
echo "ok"

step "8. OpenBlasDelivery=none opts the reference out entirely"
write_csproj "    <PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\">
      <OpenBlasVersion>$PINNED</OpenBlasVersion>
      <OpenBlasDelivery>none</OpenBlasDelivery>
    </PackageReference>"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
[ ! -f "$RID_MARKER" ] && [ ! -f "$ROOT_MARKER" ] || fail "delivery=none must ignore the reference's OpenBLAS metadata"
echo "ok"

step "9. a version on one reference vs a bare path on another is a HARD conflict (§10.2)"
write_csproj "    <PackageReference Include=\"NumSharp\" Version=\"$VERSION\">
      <OpenBlasVersion>$PINNED</OpenBlasVersion>
    </PackageReference>
    <PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\">
      <OpenBlasPath>$WORK/somewhere</OpenBlasPath>
    </PackageReference>"
( cd "$CONS" && dotnet build -v q --nologo "-clp:ErrorsOnly" > "$WORK/conflict.log" 2>&1 ) \
    && fail "an enforced download and a read-in-place directory on different references cannot both win"
grep -q "cannot both" "$WORK/conflict.log" || fail "the failure must explain the conflict; got: $(cat "$WORK/conflict.log")"
echo "ok"

step "10. a poisoned cache entry is discarded and re-downloaded, never staged"
write_csproj "$REF_VERSION"
# -print -quit, not `| head -1`: under `set -o pipefail` a `find | head` pipeline can end with
# find's SIGPIPE status (141) and abort the script with no message at all. The entry holds the
# main library, the vendored runtime (Linux/macOS) and the .entry.json sidecar — poisoning ANY
# of them must be detected, so whichever file comes first is fine.
CACHED="$(find "$CACHE/scipy-openblas64/$PINNED/$RID" -type f ! -name '*.tmp' -print -quit 2>/dev/null || true)"
[ -n "$CACHED" ] || fail "expected a cache entry from step 2 under $CACHE/scipy-openblas64/$PINNED/$RID"
echo "    poisoning $CACHED"
printf 'garbage' > "$CACHED"
# The rebuild's output is what carries the diagnosis; a failed build must show it, not vanish
# behind a captured substitution.
POISON="$(cd "$CONS" && dotnet build -t:Rebuild -v n --nologo 2>&1)" \
    || { echo "$POISON" | grep -i "error\|NumSharp.Interop.OpenBLAS:" | tail -30; fail "the rebuild after poisoning the cache failed (output above)"; }
echo "$POISON" | grep -q "DISCARDING poisoned cache entry" || fail "the tampered entry must be detected and discarded"
echo "$POISON" | grep -q "NumSharp.Interop.OpenBLAS: downloading" || fail "the discarded entry must be re-downloaded"
grep -q '"mode": "version"' "$RID_MARKER" || fail "staging must complete after the self-heal"
echo "ok"

step "11. version + path combined (metadata version, env path): download INTO the dir, marker points there"
PATH_DIR="$WORK/pathdir"; mkdir -p "$PATH_DIR"
( cd "$CONS" && NUMSHARP_OPENBLAS_SEARCH_PATH="$PATH_DIR" run dotnet build -v q --nologo "-clp:ErrorsOnly" )
ls "$PATH_DIR"/*openblas* > /dev/null 2>&1 || fail "the version must be staged INTO OpenBlasPath (§5 table row 3)"
grep -q '"mode": "version"' "$ROOT_MARKER" || fail "root marker must record the version override at $ROOT_MARKER"
grep -q '"path"' "$ROOT_MARKER" || fail "root marker must point at the custom directory"
[ ! -f "$RID_MARKER" ] || fail "the stale default-staged marker must be cleared when staging moves to OpenBlasPath"
COMBRUN="$(cd "$CONS" && NUMSHARP_OPENBLAS_SEARCH_PATH="$PATH_DIR" dotnet run --no-build 2>/dev/null)"
echo "$COMBRUN" | grep -qi "library=.*pathdir" || fail "runtime must bind from the custom directory: $COMBRUN"
echo "$COMBRUN" | grep -q "bundled=False" || fail "a custom-dir override is not the bundle: $COMBRUN"
echo "ok"

step "12. override removed: all markers cleared"
write_csproj "$REF_BARE"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
[ ! -f "$RID_MARKER" ] && [ ! -f "$ROOT_MARKER" ] || fail "markers must be cleaned when the override is removed"
echo "ok"

step "13. path-only mode (env, empty dir): non-binding marker, runtime falls through to the bundle"
EMPTY_DIR="$WORK/emptydir"; mkdir -p "$EMPTY_DIR"
( cd "$CONS" && NUMSHARP_OPENBLAS_SEARCH_PATH="$EMPTY_DIR" run dotnet build -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "path"' "$ROOT_MARKER" || fail "path marker missing at $ROOT_MARKER"
PATHRUN="$(cd "$CONS" && dotnet run --no-build 2>/dev/null)"
echo "$PATHRUN" | grep -q "bundled=True" || fail "an empty path override must fall through to the bundle: $PATHRUN"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
[ ! -f "$ROOT_MARKER" ] || fail "the path marker must clear once the env override is gone"
echo "ok"

step "14. publish layouts (version override via env)"
( cd "$CONS" && NUMSHARP_OPENBLAS_VERSION="$PINNED" run dotnet publish -o "$WORK/pub-portable" -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "version"' "$WORK/pub-portable/runtimes/$RID/native/openblas.source.json" \
    || fail "portable publish must carry the rid-dir marker"
( cd "$CONS" && NUMSHARP_OPENBLAS_VERSION="$PINNED" run dotnet publish -r "$RID" --self-contained false -o "$WORK/pub-rid" -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "version"' "$WORK/pub-rid/openblas.source.json" \
    || fail "flattened publish must carry the root marker"
PUBRUN="$(dotnet "$WORK/pub-rid/consumer.dll" 2>/dev/null)"
echo "$PUBRUN" | grep -q "bundled=False" || fail "published app must bind the override: $PUBRUN"
echo "ok"

step "15. the cache holds extracted libraries only (Goal 7: no wheels survive)"
if [ -d "$CACHE" ]; then
    # BSD wc (macOS) pads the count with leading spaces — compare numerically, never as a string.
    WHEELS="$(find "$CACHE" -name '*.whl' | wc -l | tr -d '[:space:]')"
    [ "${WHEELS:-0}" -eq 0 ] || fail "$WHEELS wheel(s) found under $CACHE"
fi
echo "ok"

printf '\nALL CHECKS PASSED — the override delivery chain is intact.\n'
