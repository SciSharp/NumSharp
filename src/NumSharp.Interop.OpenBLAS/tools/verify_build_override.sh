#!/usr/bin/env bash
# =============================================================================
# verify_build_override.sh — the BUILD-phase override delivery integration gate
# (docs/OPENBLAS_DELIVERY_DESIGN.md §5–§8; manual, like test/oracle/verify_npy_interop.py)
#
# Packs the REAL nupkg, restores it into a scratch consumer from a local feed, and
# proves the whole chain: buildTransitive import → PackageReference-metadata /
# env knobs → PyPI download + double-hash verify → global cache (wheel deleted) →
# staging over the bundle → source marker → runtime binds it as a REQUIRED
# override (Blas.IsBundledLibrary == false). Also: the wrong-version hard fail,
# marker lifecycle across mode switches, and both publish layouts.
#
# Needs: network (PyPI, first run only — the cache serves afterwards), the .NET SDK.
# Cost:  ~1 min warm, plus one ~20 MB wheel download cold.
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

VERSION="$(sed -n 's/.*<Version>\([^<]*\)<\/Version>.*/\1/p' "$PKG_DIR/NumSharp.Interop.OpenBLAS.csproj" | head -1)"
PINNED="$(sed -n 's/.*"distribution_version": "\([^"]*\)".*/\1/p' "$HERE/openblas-manifest.json" | head -1)"
FEED="$WORK/feed"; mkdir -p "$FEED"
CONS="$WORK/consumer"; mkdir -p "$CONS"

# The knobs must come from THIS script, not the ambient environment.
unset NUMSHARP_OPENBLAS_VERSION NUMSHARP_OPENBLAS_PATH NUMSHARP_OPENBLAS_DISTRIBUTION \
      NUMSHARP_OPENBLAS_FEED NUMSHARP_OPENBLAS_SHA256 NUMSHARP_OPENBLAS_DELIVERY \
      NUMSHARP_PARITY_BLAS NUMSHARP_BLAS_BUNDLED NUMSHARP_BLAS_BUNDLE_AUTOINSTALL \
      NUMSHARP_BLAS_AUTOINSTALL 2>/dev/null || true

step() { printf '\n== %s ==\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

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
cat > "$CONS/consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>disable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp.Interop.OpenBLAS" Version="$VERSION" />
  </ItemGroup>
</Project>
EOF
cat > "$CONS/Program.cs" <<'EOF'
using System;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

Console.WriteLine("enabled=" + Blas.Enabled);
Console.WriteLine("library=" + Blas.LibraryPath);
Console.WriteLine("bundled=" + Blas.IsBundledLibrary);
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
echo "ok"

step "2. version override via PackageReference metadata ($PINNED): stage + marker + runtime override"
sed -i "s#<PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\" />#<PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\"><OpenBlasVersion>$PINNED</OpenBlasVersion></PackageReference>#" "$CONS/consumer.csproj"
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

step "5. version + path combined (metadata version, env path): download INTO the dir, marker points there"
PATH_DIR="$WORK/pathdir"; mkdir -p "$PATH_DIR"
( cd "$CONS" && NUMSHARP_OPENBLAS_PATH="$PATH_DIR" run dotnet build -v q --nologo "-clp:ErrorsOnly" )
ls "$PATH_DIR"/*openblas* > /dev/null 2>&1 || fail "the version must be staged INTO OpenBlasPath (§5 table row 3)"
grep -q '"mode": "version"' "$ROOT_MARKER" || fail "root marker must record the version override at $ROOT_MARKER"
grep -q '"path"' "$ROOT_MARKER" || fail "root marker must point at the custom directory"
[ ! -f "$RID_MARKER" ] || fail "the stale default-staged marker must be cleared when staging moves to OpenBlasPath"
COMBRUN="$(cd "$CONS" && NUMSHARP_OPENBLAS_PATH="$PATH_DIR" dotnet run --no-build 2>/dev/null)"
echo "$COMBRUN" | grep -qi "library=.*pathdir" || fail "runtime must bind from the custom directory: $COMBRUN"
echo "$COMBRUN" | grep -q "bundled=False" || fail "a custom-dir override is not the bundle: $COMBRUN"
echo "ok"

step "6. override removed: all markers cleared"
sed -i "s#<PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\">.*</PackageReference>#<PackageReference Include=\"NumSharp.Interop.OpenBLAS\" Version=\"$VERSION\" />#" "$CONS/consumer.csproj"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
[ ! -f "$RID_MARKER" ] && [ ! -f "$ROOT_MARKER" ] || fail "markers must be cleaned when the override is removed"
echo "ok"

step "7. path-only mode (env, empty dir): non-binding marker, runtime falls through to the bundle"
EMPTY_DIR="$WORK/emptydir"; mkdir -p "$EMPTY_DIR"
( cd "$CONS" && NUMSHARP_OPENBLAS_PATH="$EMPTY_DIR" run dotnet build -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "path"' "$ROOT_MARKER" || fail "path marker missing at $ROOT_MARKER"
PATHRUN="$(cd "$CONS" && dotnet run --no-build 2>/dev/null)"
echo "$PATHRUN" | grep -q "bundled=True" || fail "an empty path override must fall through to the bundle: $PATHRUN"
( cd "$CONS" && run dotnet build -v q --nologo "-clp:ErrorsOnly" )
[ ! -f "$ROOT_MARKER" ] || fail "the path marker must clear once the env override is gone"
echo "ok"

step "8. publish layouts (version override via env)"
( cd "$CONS" && NUMSHARP_OPENBLAS_VERSION="$PINNED" run dotnet publish -o "$WORK/pub-portable" -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "version"' "$WORK/pub-portable/runtimes/$RID/native/openblas.source.json" \
    || fail "portable publish must carry the rid-dir marker"
( cd "$CONS" && NUMSHARP_OPENBLAS_VERSION="$PINNED" run dotnet publish -r "$RID" --self-contained false -o "$WORK/pub-rid" -v q --nologo "-clp:ErrorsOnly" )
grep -q '"mode": "version"' "$WORK/pub-rid/openblas.source.json" \
    || fail "flattened publish must carry the root marker"
PUBRUN="$(dotnet "$WORK/pub-rid/consumer.dll" 2>/dev/null)"
echo "$PUBRUN" | grep -q "bundled=False" || fail "published app must bind the override: $PUBRUN"
echo "ok"

step "9. the cache holds extracted libraries only (Goal 7: no wheels survive)"
CACHE="${LOCALAPPDATA:-$HOME/.cache}/NumSharp/openblas"
if [ -d "$CACHE" ]; then
    WHEELS="$(find "$CACHE" -name '*.whl' | wc -l)"
    [ "$WHEELS" = "0" ] || fail "$WHEELS wheel(s) found under $CACHE"
fi
echo "ok"

printf '\nALL CHECKS PASSED — the override delivery chain is intact.\n'
