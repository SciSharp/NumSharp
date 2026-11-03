#!/usr/bin/env bash
# =============================================================================
# verify_package_consumer.sh — the nupkg-CONSUMER gate for NumSharp.Interop.MLNet
#
# Packs the REAL package into a local feed, restores it into a scratch consumer by
# PackageReference — NOT ProjectReference — and runs it, on this OS, on net8.0 and net10.0, at
# the floor and at the current ML.NET. The test suite consumes the bridge by ProjectReference, so
# a packaging regression (a dependency range, a TFM folder, a missing XML file, a NumSharp
# dependency on the wrong version) would otherwise reach nuget.org unseen.
#
#   1  pack NumSharp + NumSharp.Interop.OnnxRuntime + NumSharp.Interop.MLNet into a local feed: the
#      ML.NET bridge depends on the ONNX Runtime bridge (Postprocess lives there alone), so its consumer
#      cannot restore without it. Both bridges pack with -p:IsPackable=true: while they are unreleased
#      their csprojs say false, and `dotnet pack` of such a project is a silent no-op, so the gate packs
#      what WILL ship without publishing anything
#   2  package shape: lib/<tfm>/ dll + xml for net8.0 and net10.0, no native asset, the
#      dependency groups name NumSharp and NumSharp.Interop.OnnxRuntime at THIS version and
#      Microsoft.ML.DataView from 2.0.0 with NO upper bound (a cap below a consumer's Microsoft.ML is a
#      restore error, NU1107)
#   3  consumer by PackageReference + Microsoft.ML 2.0.1 (the floor), net8.0 and net10.0:
#      AsDataView + NormalizeMinMax + ToNDArray, Postprocess.Softmax from the ONNX Runtime bridge
#      WITHOUT loading ONNX Runtime (an ML.NET app references no native ORT flavour), the zero-copy
#      VBuffer import (write-through: the packaged assembly still reaches VBuffer<T>'s private array),
#      the live counter back to 0
#   4  the same consumer against the current ML.NET
#
# Needs: the .NET SDK and network (nuget.org, for ML.NET). No Python.
# It deletes this version of NumSharp / NumSharp.Interop.OnnxRuntime / NumSharp.Interop.MLNet from the
# global NuGet packages folder (NUGET_PACKAGES when set) so the consumer restores the fresh pack.
#
#   bash src/NumSharp.Interop.MLNet/tools/verify_package_consumer.sh
# =============================================================================
set -euo pipefail
# Checks read captured output through a here-string, `grep -q PAT <<< "$OUT"`, never `echo | grep -q`:
# under pipefail a SIGPIPE'd echo fails the pipeline for a pattern grep FOUND (see the OpenBLAS script).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
REPO="$(cd "$PKG_DIR/../.." && pwd)"
WORK="$(mktemp -d)"
# On Git Bash/MSYS mktemp yields a /tmp/... path the WINDOWS dotnet/NuGet processes cannot resolve.
command -v cygpath > /dev/null 2>&1 && WORK="$(cygpath -m "$WORK")"
trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/build.log"
run() { "$@" > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "command failed: $*"; }; }
step() { printf '\n== %s ==\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

MLNET_FLOOR=2.0.1
MLNET_CURRENT="${MLNET_CURRENT:-5.0.1}"
VERSION="$(cd "$REPO" && dotnet msbuild src/NumSharp.Core/NumSharp.Core.csproj -getProperty:Version | tr -d '\r')"
[ -n "$VERSION" ] || fail "could not read the package version"
FEED="$WORK/feed"; mkdir -p "$FEED"
CONS="$WORK/consumer"; mkdir -p "$CONS"

ORT_PROJ="$REPO/src/NumSharp.Interop.OnnxRuntime/NumSharp.Interop.OnnxRuntime.csproj"
step "1. pack NumSharp $VERSION + NumSharp.Interop.OnnxRuntime $VERSION + NumSharp.Interop.MLNet $VERSION into a local feed"
# -t:Rebuild first: this repo's incremental build is known to declare stale outputs up-to-date
# after source changes, and dotnet pack defaults to Release on .NET 8+. Rebuilding the ML.NET bridge
# cleans and rebuilds its references too (Core and the ONNX Runtime bridge), so every --no-build pack
# below reads outputs of this run.
run dotnet build "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet build "$PKG_DIR/NumSharp.Interop.MLNet.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet pack "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" --no-build -c Release -o "$FEED" -v q --nologo "-clp:ErrorsOnly"
# -p:IsPackable=true overrides the csprojs' UNRELEASED false (a global property wins over the project's own),
# so the gate checks the packages the owner's release would publish; nothing here leaves the throwaway feed.
run dotnet pack "$ORT_PROJ" --no-build -c Release -o "$FEED" -v q --nologo "-clp:ErrorsOnly" -p:IsPackable=true
run dotnet pack "$PKG_DIR/NumSharp.Interop.MLNet.csproj" --no-build -c Release -o "$FEED" -v q --nologo "-clp:ErrorsOnly" \
    -p:IsPackable=true
[ -f "$FEED/NumSharp.Interop.OnnxRuntime.$VERSION.nupkg" ] \
    || fail "ONNX Runtime bridge nupkg missing (the ML.NET package depends on it): did the pack honour -p:IsPackable=true?"
NUPKG="$FEED/NumSharp.Interop.MLNet.$VERSION.nupkg"
[ -f "$NUPKG" ] || fail "bridge nupkg missing: $NUPKG (did the pack honour -p:IsPackable=true?)"
echo "ok"

step "2. package shape: lib per TFM, no native asset, the dependency groups"
# Unpacked with `tar` (bsdtar reads zip on Windows; standard elsewhere), `unzip` as the fallback.
X="$WORK/x"; mkdir -p "$X"
( cd "$X" && { tar -xf "$NUPKG" 2> /dev/null || unzip -q "$NUPKG"; } ) || fail "cannot unpack $NUPKG"
for tfm in net8.0 net10.0; do
    for f in NumSharp.Interop.MLNet.dll NumSharp.Interop.MLNet.xml; do
        [ -f "$X/lib/$tfm/$f" ] || fail "lib/$tfm/$f is not in the package"
    done
done
[ ! -d "$X/runtimes" ] || fail "the bridge must carry no native asset: $(ls -R "$X/runtimes" | head -5)"
NUSPEC="$(cat "$X"/*.nuspec)"
for tfm in net8.0 net10.0; do
    GROUP="$(tr -d '\r' <<< "$NUSPEC" | tr '\n' ' ' | grep -o "<group targetFramework=\"$tfm\">.*" | sed 's#</group>.*##')"
    [ -n "$GROUP" ] || fail "no dependency group for $tfm"
    grep -q "id=\"NumSharp\" version=\"$VERSION\"" <<< "$GROUP" || fail "$tfm: the NumSharp dependency must be $VERSION: $GROUP"
    # The ProjectReference that brings Postprocess packs as this dependency, co-versioned like NumSharp's.
    grep -q "id=\"NumSharp.Interop.OnnxRuntime\" version=\"$VERSION\"" <<< "$GROUP" \
        || fail "$tfm: the NumSharp.Interop.OnnxRuntime dependency (Postprocess) must be $VERSION: $GROUP"
    # An OPEN range packs as the bare floor (version="2.0.0"); a capped one as "[2.0.0, X)". The exact quote
    # after the floor refuses both a moved floor and a cap coming back.
    grep -q "id=\"Microsoft.ML.DataView\" version=\"2.0.0\"" <<< "$GROUP" \
        || fail "$tfm: Microsoft.ML.DataView must be 2.0.0 (the compiled reference) or later with NO upper bound - a cap below a consumer's Microsoft.ML is a restore error (NU1107): $GROUP"
    echo "    $tfm: $(grep -o '<dependency [^>]*>' <<< "$GROUP" | sed -E 's/ exclude="[^"]*"//' | tr '\n' ' ')"
done
echo "ok"

# The scratch consumer must restore THIS pack, not a same-versioned cache entry.
NUGET_ROOT="$(dotnet nuget locals global-packages --list | sed 's/.*: //' | tr -d '\r')"
command -v cygpath > /dev/null 2>&1 && NUGET_ROOT="$(cygpath -m "$NUGET_ROOT")"
rm -rf "$NUGET_ROOT/numsharp/$VERSION" "$NUGET_ROOT/numsharp.interop.onnxruntime/$VERSION" \
    "$NUGET_ROOT/numsharp.interop.mlnet/$VERSION"
cat > "$CONS/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$FEED" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="local"><package pattern="NumSharp" /><package pattern="NumSharp.*" /></packageSource>
    <packageSource key="nuget.org"><package pattern="*" /></packageSource>
  </packageSourceMapping>
</configuration>
EOF
cat > "$CONS/consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFrameworks>net8.0;net10.0</TargetFrameworks>
    <Nullable>disable</Nullable>
    <UseAppHost>false</UseAppHost>
    <MLNetVersion Condition="'\$(MLNetVersion)' == ''">$MLNET_FLOOR</MLNetVersion>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp.Interop.MLNet" Version="$VERSION" />
    <PackageReference Include="Microsoft.ML" Version="\$(MLNetVersion)" />
  </ItemGroup>
</Project>
EOF
cat > "$CONS/Program.cs" <<'EOF'
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp;
using NumSharp.Interop.MLNet;
using NumSharp.Interop.OnnxRuntime;   // Postprocess: the ONNX Runtime bridge's, which this package depends on

// The verbs a consumer reaches first, from the restored package: a lazy view through a real transform,
// a column read back, Postprocess over it, and the zero-copy VBuffer import (which reflects VBuffer<T>'s
// private array).
var ml = new MLContext(seed: 1);
NDArray x = np.array(new float[,] { { 0, 10 }, { 5, 20 }, { 10, 30 } });
using (NDArrayDataView view = x.AsDataView("Features"))
using (NDArray norm = ml.Transforms.NormalizeMinMax("Features").Fit(view).Transform(view).ToNDArray("Features"))
using (NDArray p = Postprocess.Softmax(norm, axis: -1))
using (NDArray rows = np.sum(p, axis: -1))
{
    Console.WriteLine("norm=" + string.Join(",", norm.ToArray<float>().Select(v => v.ToString("0.###"))));
    Console.WriteLine("softmax_rows=" + string.Join(",", rows.ToArray<float>().Select(v => v.ToString("0.000"))));
}
var buffer = new VBuffer<float>(3, new float[] { 1, 2, 3 });
using (NDArray alias = buffer.AsNDArray())
{
    alias.SetSingle(42f, 0);
    Console.WriteLine("writethrough=" + (buffer.GetValues()[0] == 42f));
}
Console.WriteLine("live=" + NDArrayMLNetInterop.LiveExports);
// The ML.NET bridge's promise to an app that never runs ONNX Runtime: Postprocess touches no ORT type, so the
// managed ORT assembly (restored transitively) is never loaded, and the native runtime the app does not have is
// never looked for.
Console.WriteLine("ort_loaded=" + AppDomain.CurrentDomain.GetAssemblies().Any(a => a.GetName().Name == "Microsoft.ML.OnnxRuntime"));
Console.WriteLine("framework=" + RuntimeInformation.FrameworkDescription);
EOF

consume() {   # $1 = ML.NET version
    ( cd "$CONS" && run dotnet build -c Release -p:MLNetVersion="$1" -v q --nologo "-clp:ErrorsOnly" )
    for tfm in net8.0 net10.0; do
        RUN="$(cd "$CONS" && dotnet "bin/Release/$tfm/consumer.dll" 2>&1)" || fail "ML.NET $1 / $tfm: the consumer failed: $RUN"
        echo "$RUN" | sed "s/^/    [$tfm, ML.NET $1] /"
        # NormalizeMinMax divides by the maximum (no shift), so each column is x / max: 0, .5, 1 and 1/3, 2/3, 1.
        grep -q "^norm=0,0.333,0.5,0.667,1,1$" <<< "$RUN" || fail "ML.NET $1 / $tfm: wrong NormalizeMinMax result: $RUN"
        grep -q "^softmax_rows=1.000,1.000,1.000$" <<< "$RUN" || fail "ML.NET $1 / $tfm: Postprocess.Softmax rows do not sum to 1: $RUN"
        grep -q "^writethrough=True$" <<< "$RUN" || fail "ML.NET $1 / $tfm: the VBuffer import is no longer zero-copy (VBuffer<T>._values not reached): $RUN"
        grep -q "^live=0$" <<< "$RUN" || fail "ML.NET $1 / $tfm: live exports did not return to 0: $RUN"
        grep -q "^ort_loaded=False$" <<< "$RUN" \
            || fail "ML.NET $1 / $tfm: Postprocess loaded ONNX Runtime, which an ML.NET-only app does not have: $RUN"
    done
}

step "3. consumer by PackageReference + Microsoft.ML $MLNET_FLOOR (the floor), net8.0 and net10.0"
consume "$MLNET_FLOOR"
echo "ok"

step "4. the same consumer + Microsoft.ML $MLNET_CURRENT (current)"
consume "$MLNET_CURRENT"
echo "ok"

printf '\nPASS: NumSharp.Interop.MLNet %s restores and runs from its nupkg (net8.0 / net10.0, ML.NET %s and %s)\n' \
    "$VERSION" "$MLNET_FLOOR" "$MLNET_CURRENT"
