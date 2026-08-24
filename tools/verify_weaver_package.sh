#!/usr/bin/env bash
# =================================================================================================
# verify_weaver_package.sh — end-to-end gate for the NumSharp.Weaver NuGet package.
#
# The OpenBLAS verify_build_override.sh pattern: a scripted nupkg-flow run in which every claim the
# package makes ("install it and your [NDScoped] methods are woven; it is a weaver on the project
# it is installed on, not a dependency") is exercised against a REAL consumer project restored from
# a local feed.
#
#   1  pack the weaver exactly as CI does (build -t:Rebuild, then pack --no-build — this also
#      proves Publish-under-NoBuild stages the tools/ payload)
#   2  package SHAPE: developmentDependency=true, NO dependency entries, NO lib/, the tool +
#      Mono.Cecil under tools/net8.0/any/, build/NumSharp.Weaver.targets
#   3  scaffold a multi-TFM (net8.0;net10.0) consumer referencing NumSharp.Core by ProjectReference
#   4  `dotnet add package NumSharp.Weaver` from the local feed → NuGet writes PrivateAssets="all"
#      by itself (the developmentDependency install UX — the not-a-dependency half of the contract)
#   5  build → the packaged NDScopeWeave target runs per TFM and reports the woven methods
#   6  run the consumer on BOTH TFMs → it self-verifies: every [NDScoped] method carries an NDScope
#      local (reflection over the shipped IL), and every result survives its scope (values correct)
#   7  incremental rebuild → the weave is SKIPPED (per-TFM marker up to date)
#   8  -t:Rebuild -p:SkipNDScopeWeave=true → no weave, and the consumer's own check now FAILS —
#      proving both the escape hatch and that step 6 is non-vacuous
#   9  pack the consumer → its nuspec depends on NumSharp but NOT on NumSharp.Weaver
#
# Usage:  tools/verify_weaver_package.sh [workdir]     (workdir kept on failure, or with KEEP=1)
# Needs:  dotnet SDK (net8 + net10 targeting), python (zip inspection; no unzip dependency).
# =================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/nsweaver.XXXXXX")}"
FEED="$WORK/feed"
CONSUMER="$WORK/consumer"
mkdir -p "$FEED" "$CONSUMER"

# Windows-native spellings for values embedded in XML files / handed to dotnet.
winpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
ROOTW="$(winpath "$ROOT")"
FEEDW="$(winpath "$FEED")"

step() { echo; echo "===== $*"; }
fail() { echo "FAIL: $*" >&2; echo "(work dir kept for inspection: $WORK)" >&2; exit 1; }

command -v dotnet >/dev/null 2>&1 || fail "dotnet not on PATH"
command -v python >/dev/null 2>&1 || fail "python not on PATH (used for zip inspection)"

step "1/9 pack NumSharp.Weaver the way CI does (build -t:Rebuild, then pack --no-build)"
dotnet build "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release -t:Rebuild -v q --nologo
dotnet pack  "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release --no-build -o "$FEEDW" -v q --nologo
PKG="$(ls "$FEED"/NumSharp.Weaver.*.nupkg 2>/dev/null | head -1)"
[ -n "$PKG" ] && [ -f "$PKG" ] || fail "pack produced no NumSharp.Weaver nupkg in $FEED"
echo "packed: $(basename "$PKG")"

step "2/9 package shape: developmentDependency, no dependencies, no lib/, tools+targets payload"
python - "$(winpath "$PKG")" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = set(z.namelist())
nuspec = next(n for n in names if n.endswith(".nuspec"))
text = z.read(nuspec).decode("utf-8")
def need(cond, msg):
    if not cond:
        print("FAIL:", msg); sys.exit(1)
need("<developmentDependency>true</developmentDependency>" in text, "developmentDependency=true missing from nuspec")
need("<dependencies" not in text, "nuspec must carry NO dependency entries (the weaver is not a dependency)")
need(not any(n.startswith("lib/") for n in names), "a lib/ payload would make the weaver a runtime dependency")
for required in ("build/NumSharp.Weaver.targets",
                 "tools/net8.0/any/NumSharp.Weaver.dll",
                 "tools/net8.0/any/NumSharp.Weaver.runtimeconfig.json",
                 "tools/net8.0/any/NumSharp.Weaver.deps.json",
                 "tools/net8.0/any/Mono.Cecil.dll"):
    need(required in names, "missing " + required)
print("package shape OK")
PY

step "3/9 scaffold the consumer (net8.0;net10.0, ProjectReference to NumSharp.Core)"
cat > "$CONSUMER/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="weaver-local" value="$FEEDW" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
EOF
cat > "$CONSUMER/Consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFrameworks>net8.0;net10.0</TargetFrameworks>
    <AssemblyName>Consumer</AssemblyName>
    <IsPackable>true</IsPackable>
    <Version>1.2.3</Version>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="$ROOTW/src/NumSharp.Core/NumSharp.Core.csproj" />
  </ItemGroup>
</Project>
EOF
cat > "$CONSUMER/Program.cs" <<'EOF'
using System;
using System.Linq;
using System.Reflection;
using NumSharp;

// Three [NDScoped] shapes, one per weave path a consumer exercises: a bare NDArray return, a
// ValueTuple-of-NDArrays return, and a consumer-defined INDArrayCarrier result struct.
internal static class Demo
{
    [NDScoped]
    public static NDArray Normalize(NDArray a)
    {
        var mean = np.mean(a);           // transient
        var std = np.std(a);             // transient
        var centered = a - mean;         // transient
        return centered / std;           // the one we keep
    }

    [NDScoped]
    public static (NDArray, NDArray) PlusTimes(NDArray a)
    {
        var t = a + 100.0;               // transient
        return (t - 99.0, a * 2.0);
    }

    [NDScoped]
    public static PairResult SquareStats(NDArray a)
    {
        var sq = a * a;                  // transient
        return new PairResult(a + sq, sq - a);
    }
}

internal readonly struct PairResult : INDArrayCarrier
{
    public readonly NDArray PlusSquare;
    public readonly NDArray SquareMinus;

    public PairResult(NDArray plusSquare, NDArray squareMinus)
    {
        PlusSquare = plusSquare;
        SquareMinus = squareMinus;
    }

    void INDArrayCarrier.YieldTo(NDScope scope)
    {
        scope.Returns(PlusSquare);
        scope.Returns(SquareMinus);
    }
}

internal static class Program
{
    private static int _failures;

    private static void Check(bool ok, string what)
    {
        Console.WriteLine((ok ? "ok      " : "FAIL    ") + what);
        if (!ok) _failures++;
    }

    private static int Main()
    {
        // 1. Weave coverage — the invariant NumSharp's own NDScopeWeaveTests pin for NumSharp.Core:
        //    every [NDScoped] method in the SHIPPED assembly carries a local of type NDScope.
        var scoped = typeof(Demo)
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
            .Where(m => m.GetCustomAttribute<NDScopedAttribute>() != null)
            .OrderBy(m => m.Name)
            .ToList();
        Check(scoped.Count == 3, "found " + scoped.Count + "/3 [NDScoped] methods");
        foreach (var m in scoped)
        {
            bool woven = m.GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope));
            Console.WriteLine((woven ? "ok      " : "UNWOVEN ") + m.Name);
            if (!woven) _failures++;
        }

        // 2. Values survive the scope: every result was yielded through scope.Returns, so nothing
        //    the caller receives is disposed; transients were reclaimed without corrupting them.
        var a = np.arange(1.0, 6.0);                                       // [1 2 3 4 5]
        var n = Demo.Normalize(a);
        Check(Math.Abs(n.GetDouble(0) + Math.Sqrt(2.0)) < 1e-9, "Normalize[0] == -sqrt(2)");
        Check(Math.Abs(n.GetDouble(4) - Math.Sqrt(2.0)) < 1e-9, "Normalize[4] == +sqrt(2)");

        var (x, y) = Demo.PlusTimes(a);
        Check(Math.Abs(x.GetDouble(0) - 2.0) < 1e-12, "PlusTimes.x[0] == 2");
        Check(Math.Abs(y.GetDouble(4) - 10.0) < 1e-12, "PlusTimes.y[4] == 10");

        var r = Demo.SquareStats(a);
        Check(Math.Abs(r.PlusSquare.GetDouble(2) - 12.0) < 1e-12, "SquareStats.PlusSquare[2] == 12");
        Check(Math.Abs(r.SquareMinus.GetDouble(4) - 20.0) < 1e-12, "SquareStats.SquareMinus[4] == 20");

        // 3. The input was constructed BEFORE any scope opened, so it was never tracked.
        Check(Math.Abs(a.GetDouble(2) - 3.0) < 1e-12, "input survives every call");

        Console.WriteLine(_failures == 0 ? "CONSUMER-OK" : "CONSUMER-FAIL (" + _failures + ")");
        return _failures == 0 ? 0 : 1;
    }
}
EOF

# The dev-loop trap: the package version is constant (0.60.0), and NuGet's GLOBAL cache keys by
# id+version — a previously restored copy would shadow the freshly packed one on every iteration.
# The id exists only for this gate, so evicting it is safe and makes the run hermetic.
rm -rf "$HOME/.nuget/packages/numsharp.weaver"

step "4/9 dotnet add package → PrivateAssets=all written automatically (not-a-dependency UX)"
(cd "$CONSUMER" && dotnet add Consumer.csproj package NumSharp.Weaver --source "$FEEDW" > "$WORK/add.log" 2>&1) \
  || { tail -20 "$WORK/add.log"; fail "dotnet add package NumSharp.Weaver failed"; }
grep -qi "PrivateAssets" "$CONSUMER/Consumer.csproj" \
  || fail "dotnet add package did not write PrivateAssets — developmentDependency metadata missing?"
echo "PrivateAssets written by NuGet itself:"
grep -i -A2 "NumSharp.Weaver" "$CONSUMER/Consumer.csproj" | sed 's/^/    /'

step "5/9 build → the packaged NDScopeWeave target weaves per TFM"
(cd "$CONSUMER" && dotnet build -c Release -v n > "$WORK/build1.log" 2>&1) \
  || { tail -40 "$WORK/build1.log"; fail "consumer build failed"; }
WOVEN=$(grep -c "woven 3, already-scoped 0" "$WORK/build1.log" || true)
[ "$WOVEN" -eq 2 ] || { grep -i "NumSharp.Weaver" "$WORK/build1.log" || true; fail "expected 'woven 3' for both TFMs, saw $WOVEN"; }
echo "woven 3 methods on net8.0 AND net10.0"

step "6/9 run the woven consumer on both TFMs"
for tfm in net8.0 net10.0; do
  (cd "$CONSUMER" && dotnet run --no-build -c Release -f "$tfm" > "$WORK/run.$tfm.log" 2>&1) \
    || { cat "$WORK/run.$tfm.log"; fail "consumer run failed on $tfm"; }
  grep -q "CONSUMER-OK" "$WORK/run.$tfm.log" || { cat "$WORK/run.$tfm.log"; fail "no CONSUMER-OK on $tfm"; }
  echo "$tfm: CONSUMER-OK"
done

step "7/9 incremental rebuild → weave skipped via the per-TFM marker"
# One settling build first: the P2P chain (Core self-weave + its weaver bootstrap) can legitimately
# recompile ONCE after the surrounding property-set context changes; the invariant under test is
# that a build in which CoreCompile is up to date does not re-weave.
(cd "$CONSUMER" && dotnet build -c Release -v q --nologo > "$WORK/build2a.log" 2>&1) \
  || { tail -40 "$WORK/build2a.log"; fail "settling build failed"; }
(cd "$CONSUMER" && dotnet build -c Release -v n > "$WORK/build2.log" 2>&1) \
  || { tail -40 "$WORK/build2.log"; fail "incremental build failed"; }
if grep -q "woven 3" "$WORK/build2.log"; then
  fail "weave RERAN on an up-to-date build — the NDScopeWeave marker is not incremental"
fi
echo "up-to-date build did not re-weave"

step "8/9 -p:SkipNDScopeWeave=true → no weave, and the consumer's own gate goes red (non-vacuous)"
(cd "$CONSUMER" && dotnet build -c Release -t:Rebuild -p:SkipNDScopeWeave=true -v n > "$WORK/build3.log" 2>&1) \
  || { tail -40 "$WORK/build3.log"; fail "SkipNDScopeWeave build failed"; }
if grep -qi "NumSharp.Weaver:" "$WORK/build3.log"; then
  fail "SkipNDScopeWeave=true did not skip the weave"
fi
if (cd "$CONSUMER" && dotnet run --no-build -c Release -f net8.0 > "$WORK/run.skip.log" 2>&1); then
  fail "UNWOVEN consumer passed its own weave check — the step-6 gate is vacuous"
fi
grep -q "UNWOVEN" "$WORK/run.skip.log" || { cat "$WORK/run.skip.log"; fail "expected UNWOVEN report from the skipped build"; }
echo "escape hatch works; step-6 gate proven non-vacuous"

step "9/9 pack the consumer → depends on NumSharp, NOT on NumSharp.Weaver"
(cd "$CONSUMER" && dotnet build -c Release -t:Rebuild -v q --nologo > "$WORK/build4.log" 2>&1) \
  || { tail -40 "$WORK/build4.log"; fail "re-weave rebuild failed"; }
(cd "$CONSUMER" && dotnet pack -c Release --no-build -o "$WORK/out" -v q --nologo > "$WORK/pack.log" 2>&1) \
  || { tail -40 "$WORK/pack.log"; fail "consumer pack failed"; }
python - "$(winpath "$WORK/out/Consumer.1.2.3.nupkg")" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
text = z.read(next(n for n in z.namelist() if n.endswith(".nuspec"))).decode("utf-8")
if 'id="NumSharp"' not in text:
    print("FAIL: consumer nuspec lost its NumSharp dependency"); sys.exit(1)
if "NumSharp.Weaver" in text:
    print("FAIL: NumSharp.Weaver leaked into the consumer's dependencies"); sys.exit(1)
print("consumer nuspec OK: NumSharp present, NumSharp.Weaver absent")
PY

echo
echo "verify_weaver_package: ALL 9 STEPS OK"
if [ "${KEEP:-0}" != "1" ]; then rm -rf "$WORK"; else echo "(KEEP=1: work dir kept at $WORK)"; fi
