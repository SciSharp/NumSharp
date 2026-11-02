#!/usr/bin/env bash
# =============================================================================
# verify_package_consumer.sh — the nupkg-CONSUMER gate for NumSharp.Interop.OnnxRuntime
#
# Packs the REAL package into a local feed, restores it into a scratch consumer by
# PackageReference — NOT ProjectReference — and runs it, on this OS, on net8.0 and net10.0, at
# the floor and at the current ONNX Runtime. The test suite consumes the bridge by
# ProjectReference, so a packaging regression (a dependency range, a TFM folder, a missing XML
# file, a NumSharp dependency on the wrong version) would otherwise reach nuget.org unseen.
#
#   1  pack NumSharp + NumSharp.Interop.OnnxRuntime into a local feed
#   2  package shape: lib/<tfm>/ dll + xml for net8.0 and net10.0, no native asset (the consumer
#      picks the native ORT package), the dependency groups name NumSharp at THIS version and
#      Microsoft.ML.OnnxRuntime.Managed from 1.16.0
#   3  consumer by PackageReference + Microsoft.ML.OnnxRuntime 1.16.0 (the floor), net8.0 and
#      net10.0: session.Run over a committed test model, AsOrtValue + ORT's own Run + ToNDArray,
#      Postprocess, the live counters back to 0
#   4  the same consumer against the current ONNX Runtime
#
# Needs: the .NET SDK and network (nuget.org, for ONNX Runtime). No Python.
# It deletes this version of NumSharp / NumSharp.Interop.OnnxRuntime from the global NuGet packages
# folder (NUGET_PACKAGES when set) so the consumer restores the fresh pack.
#
#   bash src/NumSharp.Interop.OnnxRuntime/tools/verify_package_consumer.sh
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

ORT_FLOOR=1.16.0
ORT_CURRENT="${ORT_CURRENT:-1.31.1}"
VERSION="$(cd "$REPO" && dotnet msbuild src/NumSharp.Core/NumSharp.Core.csproj -getProperty:Version | tr -d '\r')"
[ -n "$VERSION" ] || fail "could not read the package version"
FEED="$WORK/feed"; mkdir -p "$FEED"
CONS="$WORK/consumer"; mkdir -p "$CONS"

step "1. pack NumSharp $VERSION + NumSharp.Interop.OnnxRuntime $VERSION into a local feed"
# -t:Rebuild first: this repo's incremental build is known to declare stale outputs up-to-date
# after source changes, and dotnet pack defaults to Release on .NET 8+.
run dotnet build "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet build "$PKG_DIR/NumSharp.Interop.OnnxRuntime.csproj" -t:Rebuild -c Release \
    -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
run dotnet pack "$REPO/src/NumSharp.Core/NumSharp.Core.csproj" --no-build -c Release -o "$FEED" -v q --nologo "-clp:ErrorsOnly"
run dotnet pack "$PKG_DIR/NumSharp.Interop.OnnxRuntime.csproj" --no-build -c Release -o "$FEED" -v q --nologo "-clp:ErrorsOnly"
NUPKG="$FEED/NumSharp.Interop.OnnxRuntime.$VERSION.nupkg"
[ -f "$NUPKG" ] || fail "bridge nupkg missing: $NUPKG (is the project packable?)"
echo "ok"

step "2. package shape: lib per TFM, no native asset, the dependency groups"
# The listing and the nuspec are read with the SDK's own tools only (unzip is not on every runner):
# a throwaway C# script would need a restore, so the nupkg is unpacked with `tar`, which reads zip
# archives on Windows (bsdtar) and is standard elsewhere; fall back to `unzip` when tar cannot.
X="$WORK/x"; mkdir -p "$X"
( cd "$X" && { tar -xf "$NUPKG" 2> /dev/null || unzip -q "$NUPKG"; } ) || fail "cannot unpack $NUPKG"
for tfm in net8.0 net10.0; do
    for f in NumSharp.Interop.OnnxRuntime.dll NumSharp.Interop.OnnxRuntime.xml; do
        [ -f "$X/lib/$tfm/$f" ] || fail "lib/$tfm/$f is not in the package"
    done
done
[ ! -d "$X/runtimes" ] || fail "the bridge must carry no native asset (the consumer picks the ORT flavour): $(ls -R "$X/runtimes" | head -5)"
NUSPEC="$(cat "$X"/*.nuspec)"
for tfm in net8.0 net10.0; do
    GROUP="$(tr -d '\r' <<< "$NUSPEC" | tr '\n' ' ' | grep -o "<group targetFramework=\"$tfm\">.*" | sed 's#</group>.*##')"
    [ -n "$GROUP" ] || fail "no dependency group for $tfm"
    grep -q "id=\"NumSharp\" version=\"$VERSION\"" <<< "$GROUP" || fail "$tfm: the NumSharp dependency must be $VERSION: $GROUP"
    grep -q "id=\"Microsoft.ML.OnnxRuntime.Managed\" version=\"\[1.16.0," <<< "$GROUP" \
        || fail "$tfm: Microsoft.ML.OnnxRuntime.Managed must start at 1.16.0 (the compiled reference): $GROUP"
    echo "    $tfm: $(grep -o '<dependency [^>]*>' <<< "$GROUP" | sed -E 's/ exclude="[^"]*"//' | tr '\n' ' ')"
done
echo "ok"

# The scratch consumer must restore THIS pack, not a same-versioned cache entry.
NUGET_ROOT="$(dotnet nuget locals global-packages --list | sed 's/.*: //' | tr -d '\r')"
command -v cygpath > /dev/null 2>&1 && NUGET_ROOT="$(cygpath -m "$NUGET_ROOT")"
rm -rf "$NUGET_ROOT/numsharp/$VERSION" "$NUGET_ROOT/numsharp.interop.onnxruntime/$VERSION"
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
    <OrtVersion Condition="'\$(OrtVersion)' == ''">$ORT_FLOOR</OrtVersion>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp.Interop.OnnxRuntime" Version="$VERSION" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="\$(OrtVersion)" />
    <None Include="add_f32.onnx" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
</Project>
EOF
cp "$REPO/test/NumSharp.Tests.Interop.OnnxRuntime/Models/add_f32.onnx" "$CONS/"
cat > "$CONS/Program.cs" <<'EOF'
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

// add_f32: C = A + B with ORT's broadcasting. Every verb a consumer reaches first, from the restored package.
using var session = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "add_f32.onnx"));
NDArray a = np.arange(6).astype(np.float32).reshape(2, 3);
NDArray b = np.array(new float[] { 10, 20, 30 });
var outs = session.Run(new Dictionary<string, NDArray> { ["A"] = a, ["B"] = b });
using NDArray c = outs["C"];
Console.WriteLine("c=" + string.Join(",", c.ToArray<float>()));
using (OrtTensor ha = a.AsOrtValue())
using (OrtTensor hb = b.AsOrtValue())
using (var runOptions = new RunOptions())
using (var raw = session.Run(runOptions, new[] { "A", "B" }, new[] { ha.Value, hb.Value }, new[] { "C" }))
using (NDArray back = raw[0].ToNDArray())
    Console.WriteLine("roundtrip=" + np.array_equal(back, c));
using NDArray p = Postprocess.Softmax(c, axis: -1);
using NDArray rows = np.sum(p, axis: -1);
Console.WriteLine("softmax_rows=" + string.Join(",", rows.ToArray<float>().Select(v => v.ToString("0.000"))));
Console.WriteLine("live=" + NDArrayOnnxInterop.LiveExports + "/" + NDArrayOnnxInterop.LiveImports);
Console.WriteLine("framework=" + RuntimeInformation.FrameworkDescription);
EOF

consume() {   # $1 = ONNX Runtime version
    ( cd "$CONS" && run dotnet build -c Release -p:OrtVersion="$1" -v q --nologo "-clp:ErrorsOnly" )
    for tfm in net8.0 net10.0; do
        RUN="$(cd "$CONS" && dotnet "bin/Release/$tfm/consumer.dll" 2>&1)" || fail "ORT $1 / $tfm: the consumer failed: $RUN"
        echo "$RUN" | sed "s/^/    [$tfm, ORT $1] /"
        grep -q "^c=10,21,32,13,24,35$" <<< "$RUN" || fail "ORT $1 / $tfm: wrong Run result: $RUN"
        grep -q "^roundtrip=True$" <<< "$RUN" || fail "ORT $1 / $tfm: AsOrtValue + Run + ToNDArray disagree with Run: $RUN"
        grep -q "^softmax_rows=1.000,1.000$" <<< "$RUN" || fail "ORT $1 / $tfm: Postprocess.Softmax rows do not sum to 1: $RUN"
        grep -q "^live=0/0$" <<< "$RUN" || fail "ORT $1 / $tfm: live exports / imports did not return to 0: $RUN"
    done
}

step "3. consumer by PackageReference + Microsoft.ML.OnnxRuntime $ORT_FLOOR (the floor), net8.0 and net10.0"
consume "$ORT_FLOOR"
echo "ok"

step "4. the same consumer + Microsoft.ML.OnnxRuntime $ORT_CURRENT (current)"
consume "$ORT_CURRENT"
echo "ok"

printf '\nPASS: NumSharp.Interop.OnnxRuntime %s restores and runs from its nupkg (net8.0 / net10.0, ORT %s and %s)\n' \
    "$VERSION" "$ORT_FLOOR" "$ORT_CURRENT"
