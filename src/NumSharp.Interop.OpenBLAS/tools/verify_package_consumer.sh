#!/usr/bin/env bash
# =============================================================================
# verify_package_consumer.sh — the nupkg-CONSUMER gate for NumSharp.Interop.OpenBLAS
#
# Packs the REAL package into a local feed, restores it into a scratch consumer by
# PackageReference — NOT ProjectReference — and proves the bundled OpenBLAS actually LOADS
# from what NuGet delivers, on this OS, in every layout a consumer can end up with.
#
# Why this exists: the test suites consume the package by ProjectReference, where a
# Content item copied files a PackageReference restore never sees. That is how the macOS
# package shipped a dylib whose vendored Fortran runtime (@loader_path/../.dylibs/…) sat in
# a SIBLING folder NuGet silently drops — "Library not loaded" on every Mac, green CI.
#
#   1  pack NumSharp + NumSharp.Interop.OpenBLAS (assets REQUIRED — an asset-free pack
#      would pass every later step vacuously by falling back to the managed kernels)
#   2  package SHAPE: every manifest RID's library under runtimes/<rid>/native/; the macOS
#      deps NESTED under native/.dylibs/; NO runtimes/<rid>/.dylibs/ sibling anywhere
#   3  consumer (PackageReference, local feed) -> build -> run: the bundle is loaded and
#      computes NumPy's answer (portable layout; on macOS the loader symlinks the sibling)
#   4  read-only install (non-Windows): a fresh process still loads — via the per-user
#      cache copy on macOS (LoadedImagePath under NUMSHARP_OPENBLAS_CACHE_DIR), directly
#      elsewhere — and writes NOTHING into the read-only tree
#   5  RID-specific publish (the FLATTENED layout, deps beside the main) -> run -> loads
#   6  the cache entry is self-verifying (macOS): a tampered dep is rebuilt, never mapped
#
# Needs: the .NET SDK, python3 (fetch_openblas.py is stdlib-only), network once (PyPI).
#
#   bash src/NumSharp.Interop.OpenBLAS/tools/verify_package_consumer.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
REPO="$(cd "$PKG_DIR/../.." && pwd)"
WORK="$(mktemp -d)"
# On Git Bash/MSYS mktemp yields a /tmp/... path the WINDOWS dotnet/NuGet processes cannot
# resolve — normalize to a mixed C:/... form both worlds accept.
command -v cygpath > /dev/null 2>&1 && WORK="$(cygpath -m "$WORK")"
trap 'chmod -R u+w "$WORK" 2>/dev/null || true; rm -rf "$WORK"' EXIT
LOG="$WORK/build.log"
run() { "$@" > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "command failed: $*"; }; }
step() { printf '\n== %s ==\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
case "$OS" in Darwin) IS_MAC=1 ;; *) IS_MAC=0 ;; esac
case "$OS" in MINGW*|MSYS*|CYGWIN*|Windows_NT) IS_WIN=1 ;; *) IS_WIN=0 ;; esac

# The knobs must come from THIS script, not the ambient environment.
unset NUMSHARP_OPENBLAS_VERSION NUMSHARP_OPENBLAS_SEARCH_PATH NUMSHARP_OPENBLAS_DISTRIBUTION \
      NUMSHARP_OPENBLAS_PYPI_FEED_URL NUMSHARP_OPENBLAS_SHA256 NUMSHARP_OPENBLAS_DELIVERY \
      NUMSHARP_OPENBLAS_LIBRARY NUMSHARP_OPENBLAS_USE_BUNDLED NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL \
      NUMSHARP_OPENBLAS_CACHE_DIR OPENBLAS_HOME OPENBLAS_ROOT 2>/dev/null || true

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || fail "python3 is required (fetch_openblas.py stages the bundled assets)"

VERSION="$(cd "$REPO" && dotnet msbuild src/NumSharp.Core/NumSharp.Core.csproj -getProperty:Version | tr -d '\r')"
[ -n "$VERSION" ] || fail "could not read the package version"
FEED="$WORK/feed"; mkdir -p "$FEED"
CONS="$WORK/consumer"; mkdir -p "$CONS"
CACHE="$WORK/cache"; mkdir -p "$CACHE"
export NUMSHARP_OPENBLAS_CACHE_DIR="$CACHE"

step "0. stage the bundled OpenBLAS assets (every RID, manifest-verified)"
run "$PY" "$HERE/fetch_openblas.py"
run "$PY" "$HERE/fetch_openblas.py" --check
echo "ok"

step "1. pack NumSharp $VERSION + NumSharp.Interop.OpenBLAS $VERSION into a local feed (assets required)"
# -t:Rebuild first: this repo's incremental build is known to declare stale outputs up-to-date
# after source changes, and dotnet pack defaults to Release on .NET 8+.
run dotnet build "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet build "$REPO/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet pack "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" --no-build -c Release -o "$FEED" \
    -v q --nologo "-clp:ErrorsOnly"
run dotnet pack "$REPO/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj" --no-build -c Release -o "$FEED" \
    -v q --nologo "-clp:ErrorsOnly" -p:RequireOpenBlasAssets=true
NUPKG="$FEED/NumSharp.Interop.OpenBLAS.$VERSION.nupkg"
[ -f "$NUPKG" ] || fail "interop nupkg missing: $NUPKG"
echo "ok"

step "2. package shape: native assets where NuGet delivers them, macOS deps NESTED, no sibling folder"
LISTING="$("$PY" - "$NUPKG" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in sorted(z.namelist()):
        if n.startswith("runtimes/"):
            print(n)
EOF
)"
echo "$LISTING" | sed 's/^/    /'
for rid in win-x64 win-arm64 linux-x64 linux-arm64 linux-musl-x64 linux-musl-arm64 osx-x64 osx-arm64; do
    echo "$LISTING" | grep -q "^runtimes/$rid/native/[^/]*openblas[^/]*$" || fail "no main library for $rid under runtimes/$rid/native/"
done
for rid in osx-x64 osx-arm64; do
    for dep in libgfortran.5.dylib libquadmath.0.dylib libgcc_s.1.1.dylib; do
        echo "$LISTING" | grep -q "^runtimes/$rid/native/.dylibs/$dep$" \
            || fail "$rid: vendored $dep must be packed NESTED as runtimes/$rid/native/.dylibs/$dep (NuGet delivers nested paths)"
    done
done
echo "$LISTING" | grep -q "^runtimes/[^/]*/\.dylibs/" \
    && fail "a runtimes/<rid>/.dylibs/ SIBLING folder is packed — NuGet drops it from every consumer, so it is dead weight that hides the defect"
for rid in linux-x64 linux-musl-x64; do
    echo "$LISTING" | grep -q "^runtimes/$rid/native/libgfortran-" || fail "$rid: vendored libgfortran must sit beside the main (RUNPATH \$ORIGIN)"
done
echo "ok"

step "3. consumer by PackageReference: the bundle loads and computes NumPy's answer (portable layout)"
# The scratch consumer must restore THIS pack, not a same-versioned cache entry.
NUGET_ROOT="$(dotnet nuget locals global-packages --list | sed 's/.*: //' | tr -d '\r')"
command -v cygpath > /dev/null 2>&1 && NUGET_ROOT="$(cygpath -m "$NUGET_ROOT")"
rm -rf "$NUGET_ROOT/numsharp/$VERSION" "$NUGET_ROOT/numsharp.interop.openblas/$VERSION"
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
    <UseAppHost>false</UseAppHost>
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

// Loading happens in the module initializer when the assembly is first touched; make the
// outcome — and the reason for a failure — visible either way.
string reason = null;
try { OpenBlasEngine.Enable(); } catch (Exception e) { reason = e.GetType().Name + ": " + e.Message; }
Console.WriteLine("enabled=" + OpenBlasEngine.Enabled);
Console.WriteLine("library=" + OpenBlasEngine.LibraryPath);
Console.WriteLine("image=" + OpenBlasEngine.LoadedImagePath);
Console.WriteLine("bundled=" + OpenBlasEngine.IsBundledLibrary);
Console.WriteLine("info=" + OpenBlasEngine.Info);
if (reason != null) Console.WriteLine("reason=" + reason);
Console.WriteLine("dot00=" + (float)np.dot(
    np.arange(6).astype(NPTypeCode.Single).reshape(2, 3),
    np.arange(6).astype(NPTypeCode.Single).reshape(3, 2))[0, 0]);
EOF
( cd "$CONS" && run dotnet build -c Release -v q --nologo "-clp:ErrorsOnly" )
OUT="$CONS/bin/Release/net10.0"
RID="$(cd "$CONS" && dotnet msbuild consumer.csproj -getProperty:NETCoreSdkPortableRuntimeIdentifier | tr -d '\r')"
ls "$OUT"/runtimes/"$RID"/native/*openblas* > /dev/null 2>&1 || fail "the bundled asset did not reach the consumer output for $RID"
[ ! -d "$OUT/runtimes/$RID/.dylibs" ] || fail "a fresh restore must not carry a runtimes/$RID/.dylibs sibling (it is the loader's job to make it)"
RUN="$(cd "$CONS" && dotnet "$OUT/consumer.dll" 2>&1)"; echo "$RUN" | sed 's/^/    /'
echo "$RUN" | grep -q "^enabled=True" || fail "the bundled OpenBLAS did not load from a PackageReference restore: $RUN"
echo "$RUN" | grep -q "^bundled=True" || fail "the loaded library must be the bundle: $RUN"
echo "$RUN" | grep -q "^dot00=10" || fail "wrong product: $RUN"
if [ "$IS_MAC" = 1 ]; then
    [ -e "$OUT/runtimes/$RID/.dylibs" ] || fail "macOS: the loader must have materialized runtimes/$RID/.dylibs in place"
    [ -L "$OUT/runtimes/$RID/.dylibs" ] && echo "    (materialized as a symlink -> $(readlink "$OUT/runtimes/$RID/.dylibs"))"
    IMG="$(echo "$RUN" | sed -n 's/^image=//p')"; LIB="$(echo "$RUN" | sed -n 's/^library=//p')"
    [ "$IMG" = "$LIB" ] || fail "macOS writable layout: the library must be mapped IN PLACE, not from the cache (image=$IMG library=$LIB)"
    [ -z "$(ls -A "$CACHE" 2>/dev/null)" ] || fail "macOS writable layout must not touch the cache: $(ls -R "$CACHE")"
fi
echo "ok"

if [ "$IS_WIN" = 0 ]; then
    step "4. read-only install: a fresh process still loads, and writes nothing into the tree"
    RO="$WORK/ro"; rm -rf "$RO"; cp -R "$OUT" "$RO"
    rm -rf "$RO/runtimes/$RID/.dylibs"          # back to exactly what NuGet delivered
    chmod -R a-w "$RO"
    RORUN="$(cd "$CONS" && dotnet "$RO/consumer.dll" 2>&1)"; echo "$RORUN" | sed 's/^/    /'
    echo "$RORUN" | grep -q "^enabled=True" || fail "the bundle must load from a read-only install: $RORUN"
    echo "$RORUN" | grep -q "^dot00=10" || fail "wrong product from the read-only install: $RORUN"
    [ ! -e "$RO/runtimes/$RID/.dylibs" ] || fail "nothing may be written into the read-only tree"
    if [ "$IS_MAC" = 1 ]; then
        IMG="$(echo "$RORUN" | sed -n 's/^image=//p')"
        case "$IMG" in "$CACHE"/*) ;; *) fail "macOS read-only layout: the image must come from the cache under $CACHE, got $IMG" ;; esac
        echo "$RORUN" | grep -q "^library=$RO/" || fail "LibraryPath must keep naming the install's own file (the parity pin hashes it): $RORUN"
        [ -f "$IMG" ] && cmp -s "$IMG" "$RO/runtimes/$RID/native/$(basename "$IMG")" || fail "the cached image must be byte-identical to the install's library"
        ls "$(dirname "$(dirname "$IMG")")/.dylibs/libgfortran.5.dylib" > /dev/null || fail "the cache entry must carry ../.dylibs/ beside native/"
        echo "    (mapped from the cache: $IMG)"
    fi
    chmod -R u+w "$RO"
    echo "ok"
fi

step "5. RID-specific publish (flattened layout): the bundle loads from beside the app"
PUB="$WORK/pub-rid"
( cd "$CONS" && run dotnet publish -c Release -r "$RID" --self-contained false -p:UseAppHost=false -o "$PUB" -v q --nologo "-clp:ErrorsOnly" )
ls "$PUB"/*openblas* > /dev/null 2>&1 || fail "a RID-specific publish must flatten the library next to the app"
[ ! -d "$PUB/runtimes" ] || fail "a RID-specific publish must not carry a runtimes/ tree"
if [ "$IS_MAC" = 1 ]; then
    ls "$PUB"/libgfortran.5.dylib > /dev/null 2>&1 || fail "macOS: the flatten must have dropped native/.dylibs/ beside the main"
fi
PUBRUN="$(cd "$CONS" && dotnet "$PUB/consumer.dll" 2>&1)"; echo "$PUBRUN" | sed 's/^/    /'
echo "$PUBRUN" | grep -q "^enabled=True" || fail "the bundle must load from a flattened publish: $PUBRUN"
echo "$PUBRUN" | grep -q "^dot00=10" || fail "wrong product from the flattened publish: $PUBRUN"
[ ! -e "$WORK/.dylibs" ] || fail "the loader must never write ../.dylibs OUTSIDE the application folder"
if [ "$IS_MAC" = 1 ]; then
    IMG="$(echo "$PUBRUN" | sed -n 's/^image=//p')"
    case "$IMG" in "$CACHE"/*) ;; *) fail "macOS flattened layout: the image must come from the cache, got $IMG" ;; esac

    step "6. the cache entry is self-verifying: a tampered dependency is rebuilt, never mapped"
    ENTRY="$(dirname "$(dirname "$IMG")")"
    printf 'garbage' > "$ENTRY/.dylibs/libgfortran.5.dylib"
    AGAIN="$(cd "$CONS" && dotnet "$PUB/consumer.dll" 2>&1)"
    echo "$AGAIN" | grep -q "^enabled=True" || fail "the loader must rebuild a tampered cache entry: $AGAIN"
    cmp -s "$ENTRY/.dylibs/libgfortran.5.dylib" "$PUB/libgfortran.5.dylib" || fail "the tampered dep must have been restored from its source"
    echo "ok"
fi
echo "ok"

printf '\nALL CHECKS PASSED — NumSharp.Interop.OpenBLAS %s loads from a PackageReference restore on %s (%s).\n' "$VERSION" "$OS" "$RID"
