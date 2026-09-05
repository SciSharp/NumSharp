#!/usr/bin/env bash
# =================================================================================================
# verify_build_package.sh — end-to-end gate for the NumSharp.Build NuGet package.
#
# The OpenBLAS verify_build_override.sh pattern: a scripted nupkg-flow run in which every claim the
# package makes ("install it and your [NDScoped] methods are woven; it is a weaver on the project
# it is installed on, not a dependency") is exercised against a REAL consumer project restored from
# a local feed.
#
#   1  pack the weaver exactly as CI does (build -t:Rebuild, then pack --no-build — this also
#      proves Publish-under-NoBuild stages the tools/ payload)
#   2  package SHAPE: developmentDependency=true, NO dependency entries, NO lib/, the tool +
#      Mono.Cecil under tools/net8.0/any/, build/NumSharp.Build.targets — and NO analyzers/
#      (the compile-time analyzer ships in the NumSharp package itself, asserted here on the
#      NumSharp nupkg: analyzers/dotnet/cs/NumSharp.Build.Analyzer.dll alone — the compiler
#      host provides Microsoft.CodeAnalysis); 2b: packing Core with
#      -p:EnableNDArrayLeakAnalyzer=false still ships the analyzer (payload toggle-independent)
#      while Core's own compile draws no NDW012 (the toggle governs analysis, never the payload)
#   3  scaffold a multi-TFM (net8.0;net10.0) consumer referencing NumSharp.Core by ProjectReference
#   4  `dotnet add package NumSharp.Build` from the local feed → NuGet writes PrivateAssets="all"
#      by itself (the developmentDependency install UX — the not-a-dependency half of the contract)
#   5  build → the packaged NDScopeWeave target runs per TFM and reports the woven methods
#   6  run the consumer on BOTH TFMs → it self-verifies: every [NDScoped] method carries an NDScope
#      local (reflection over the shipped IL), and every result survives its scope (values correct)
#   7  incremental rebuild → the weave is SKIPPED (per-TFM marker up to date)
#   8  -t:Rebuild -p:SkipNDScopeWeave=true → no weave, and the consumer's own check now FAILS —
#      proving both the escape hatch and that step 6 is non-vacuous
#   9  pack the consumer → its nuspec depends on NumSharp but NOT on NumSharp.Build; then the
#      NOT-CONTAGIOUS guard (NDW018) for the references `dotnet add package` did NOT write: 9b a
#      HAND-WRITTEN <PackageReference Include="NumSharp.Build"> without PrivateAssets FAILS to pack
#      (no nupkg), 9c -p:NumSharpBuildAllowAsDependency=true packs it as a dependency on purpose AND
#      a consumer of that nupkg is still NOT woven (weaving never flows, even then), 9d adding
#      PrivateAssets="all" by hand packs clean (no NumSharp.Build dependency) and the Central Package
#      Management spelling without PrivateAssets draws NDW018 too
#  10  the REAL product path: a consumer taking NumSharp itself as a PackageReference (lib/
#      resolution, not the P2P ref-assembly path) weaves and runs; 10b repeats it on net10.0
#      (lib/net10.0 + the TFM-agnostic analyzers/ and build/ assets, everything from the nupkgs)
#  11  error shapes FAIL the build in the right LAYER: 11a the compile-time ANALYZER reports the
#      source-detectable rejections (NDW002 ref egress, NDW003 unsupported carrier, NDW009/010 the
#      WRONG attribute, NDW011 both attributes) and preempts the weaver; 11b the post-compile WEAVER
#      reports the IL-only NDW004 (a hand-rolled bogus [AsyncStateMachine] the analyzer passes)
#  12  a consumer with NumSharp but no [NDScoped]/[NDScopedAsync] methods no-ops — assembly untouched
#  13  transitive isolation: AppB references woven LibA but does NOT install the weaver —
#      LibA stays woven, AppB's own [NDScoped] method stays inert (never woven, never an error)
#  14  a strong-named consumer is re-signed with its own key and still runs
#  15  the STATE-MACHINE weave: an async method + a non-async-Task method ([NDScopedAsync]) and a
#      synchronous iterator ([NDScoped]) are woven through their compiler state machines and behave
#      (temps reclaimed at completion, results and yielded elements survive, in-flight operands
#      protected by deferral)
#  16  the LEAK analyzer (NDW012): a NON-scoped method that drops an NDArray it never returns/disposes
#      draws a build WARNING; a [NDScoped] method with the same body does NOT; 16b the OWNERSHIP
#      analyzer (NDW016/NDW017): a type that stores an NDArray without IDisposable, and a disposable
#      that never disposes its member, each draw a build WARNING; [NDBorrowed] silences both
#  17  the WEAVER-MISSING guard (NDW013, shipped in NumSharp itself): a consumer that uses [NDScoped]
#      but does NOT install NumSharp.Build draws a build WARNING — from the ANALYZER the NumSharp
#      package applies (per member, with a source location) while the MSBuild fallback scan in
#      build/NumSharp.targets stays silent (no double report); installing the weaver silences it;
#      17b: a compile the analyzer does not reach (a bare <Reference> to the PACKAGED NumSharp.dll
#      importing the PACKAGED build/NumSharp.targets) draws the project-wide MSBuild fallback scan
#      instead — and NOT via ExcludeAssets="analyzers", which the SDK does not honour for analyzers
#  18  the analyzer rides the NumSharp package ALONE: a consumer that references NumSharp but NOT
#      NumSharp.Build still gets the compile-time gate (a bad [NDScoped] target FAILS the build,
#      NDW003) and the NDW012 leak warning — no weaver install required
#
# Usage:  tools/verify_build_package.sh [workdir]     (workdir kept on failure, or with KEEP=1)
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

step "1/18 pack NumSharp.Build the way CI does (build -t:Rebuild, then pack --no-build)"
dotnet build "$ROOTW/tools/NumSharp.Build/NumSharp.Build.csproj" -c Release -t:Rebuild -v q --nologo
dotnet pack  "$ROOTW/tools/NumSharp.Build/NumSharp.Build.csproj" -c Release --no-build -o "$FEEDW" -v q --nologo
PKG="$(ls "$FEED"/NumSharp.Build.*.nupkg 2>/dev/null | head -1)"
[ -n "$PKG" ] && [ -f "$PKG" ] || fail "pack produced no NumSharp.Build nupkg in $FEED"
WVER="$(basename "$PKG" .nupkg | sed 's/^NumSharp\.Build\.//')"
echo "packed: $(basename "$PKG")"

# NumSharp itself joins the feed for the PackageReference consumer (step 10) — the real-world
# composition, where the weaver resolves NDScope out of lib/net8.0/NumSharp.dll rather than the
# ProjectReference ref assembly. GeneratePackageOnBuild=false avoids the multi-TFM NU5026 trap.
dotnet pack "$ROOTW/src/NumSharp.Core/NumSharp.Core.csproj" -c Release -o "$FEEDW" \
  -p:GeneratePackageOnBuild=false -v q --nologo > "$WORK/corepack.log" 2>&1 \
  || { tail -20 "$WORK/corepack.log"; fail "NumSharp.Core pack failed"; }
CPKG="$(ls "$FEED"/NumSharp.[0-9]*.nupkg 2>/dev/null | head -1)"
[ -n "$CPKG" ] || fail "no NumSharp nupkg in the feed"
CVER="$(basename "$CPKG" .nupkg | sed 's/^NumSharp\.//')"
echo "packed: $(basename "$CPKG")"

# Evict same-id/same-version entries from the GLOBAL package cache: the cache is consulted before
# any source, so a stale extraction from a previous run would silently shadow the packages just
# packed and this script would verify LAST run's bits (the constant-version local-package trap).
GPF="${NUGET_PACKAGES:-$HOME/.nuget/packages}"
rm -rf "$GPF/numsharp.build/$WVER" "$GPF/numsharp/$CVER" 2>/dev/null || true

step "2/18 package shape: developmentDependency, no dependencies, no lib/, tools+targets payload; analyzer rides the NumSharp package"
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
for required in ("build/NumSharp.Build.targets",
                 "tools/net8.0/any/NumSharp.Build.dll",
                 "tools/net8.0/any/NumSharp.Build.runtimeconfig.json",
                 "tools/net8.0/any/NumSharp.Build.deps.json",
                 "tools/net8.0/any/Mono.Cecil.dll"):
    need(required in names, "missing " + required)
# Weaver-ONLY: the compile-time analyzer ships in the NumSharp package (checked next), never here —
# two copies across the two packages would double-apply on any consumer that installs both.
need(not any(n.startswith("analyzers/") for n in names),
     "NumSharp.Build must ship NO analyzers/ (the analyzer rides the NumSharp package)")
print("weaver package shape OK")
PY
python - "$(winpath "$CPKG")" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = set(z.namelist())
def need(cond, msg):
    if not cond:
        print("FAIL:", msg); sys.exit(1)
# The NumSharp package carries the compile-time analyzer, so referencing NumSharp ALONE gives every
# consumer the [NDScoped]-target gate + the NDW012 leak nudge (step 18 proves it end-to-end).
need("analyzers/dotnet/cs/NumSharp.Build.Analyzer.dll" in names,
     "NumSharp package is missing analyzers/dotnet/cs/NumSharp.Build.Analyzer.dll")
# The analyzer ships ALONE — Microsoft.CodeAnalysis is provided by the compiler host at run time.
need(not any(n.startswith("analyzers/") and "CodeAnalysis" in n for n in names),
     "Microsoft.CodeAnalysis must NOT ship under analyzers/ (the compiler host provides it)")
print("NumSharp package analyzer payload OK")
PY

# 2b — the analyzer payload is TOGGLE-INDEPENDENT: -p:EnableNDArrayLeakAnalyzer=false turns the
# analyzer off for Core's OWN compile (so no NDW012 there) but the packed nupkg still carries it —
# the conditional ProjectReference split in NumSharp.Core.csproj keeps the DLL building either way.
TOFF="$WORK/toggleoff"
mkdir -p "$TOFF"
dotnet pack "$ROOTW/src/NumSharp.Core/NumSharp.Core.csproj" -c Release -o "$(winpath "$TOFF")" \
  -p:GeneratePackageOnBuild=false -p:EnableNDArrayLeakAnalyzer=false -v q --nologo > "$TOFF/pack.log" 2>&1 \
  || { tail -20 "$TOFF/pack.log"; fail "2b: toggle-off pack of NumSharp.Core failed"; }
if grep -q "warning NDW012" "$TOFF/pack.log"; then
  fail "2b: NDW012 fired on Core's own compile despite EnableNDArrayLeakAnalyzer=false"
fi
python - "$(winpath "$TOFF")" <<'PY'
import sys, zipfile, glob
pkgs = glob.glob(sys.argv[1] + "/NumSharp.[0-9]*.nupkg")
if not pkgs:
    print("FAIL: 2b produced no NumSharp nupkg"); sys.exit(1)
names = zipfile.ZipFile(pkgs[0]).namelist()
if "analyzers/dotnet/cs/NumSharp.Build.Analyzer.dll" not in names:
    print("FAIL: toggle-off pack dropped the analyzer — the payload must not depend on EnableNDArrayLeakAnalyzer"); sys.exit(1)
print("2b: analyzer still packed under EnableNDArrayLeakAnalyzer=false (payload is toggle-independent)")
PY

step "3/18 scaffold the consumer (net8.0;net10.0, ProjectReference to NumSharp.Core)"
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
rm -rf "$HOME/.nuget/packages/numsharp.weaver" "$HOME/.nuget/packages/numsharp/$CVER"

step "4/18 dotnet add package → PrivateAssets=all written automatically (not-a-dependency UX)"
(cd "$CONSUMER" && dotnet add Consumer.csproj package NumSharp.Build --source "$FEEDW" > "$WORK/add.log" 2>&1) \
  || { tail -20 "$WORK/add.log"; fail "dotnet add package NumSharp.Build failed"; }
grep -qi "PrivateAssets" "$CONSUMER/Consumer.csproj" \
  || fail "dotnet add package did not write PrivateAssets — developmentDependency metadata missing?"
echo "PrivateAssets written by NuGet itself:"
grep -i -A2 "NumSharp.Build" "$CONSUMER/Consumer.csproj" | sed 's/^/    /'

step "5/18 build → the packaged NDScopeWeave target weaves per TFM"
(cd "$CONSUMER" && dotnet build -c Release -v n > "$WORK/build1.log" 2>&1) \
  || { tail -40 "$WORK/build1.log"; fail "consumer build failed"; }
WOVEN=$(grep -c "woven 3, already-scoped 0" "$WORK/build1.log" || true)
[ "$WOVEN" -eq 2 ] || { grep -i "NumSharp.Build" "$WORK/build1.log" || true; fail "expected 'woven 3' for both TFMs, saw $WOVEN"; }
echo "woven 3 methods on net8.0 AND net10.0"

step "6/18 run the woven consumer on both TFMs"
for tfm in net8.0 net10.0; do
  (cd "$CONSUMER" && dotnet run --no-build -c Release -f "$tfm" > "$WORK/run.$tfm.log" 2>&1) \
    || { cat "$WORK/run.$tfm.log"; fail "consumer run failed on $tfm"; }
  grep -q "CONSUMER-OK" "$WORK/run.$tfm.log" || { cat "$WORK/run.$tfm.log"; fail "no CONSUMER-OK on $tfm"; }
  echo "$tfm: CONSUMER-OK"
done

step "7/18 incremental rebuild → weave skipped via the per-TFM marker"
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

step "8/18 -p:SkipNDScopeWeave=true → no weave, and the consumer's own gate goes red (non-vacuous)"
(cd "$CONSUMER" && dotnet build -c Release -t:Rebuild -p:SkipNDScopeWeave=true -v n > "$WORK/build3.log" 2>&1) \
  || { tail -40 "$WORK/build3.log"; fail "SkipNDScopeWeave build failed"; }
if grep -qi "NumSharp.Build:" "$WORK/build3.log"; then
  fail "SkipNDScopeWeave=true did not skip the weave"
fi
if (cd "$CONSUMER" && dotnet run --no-build -c Release -f net8.0 > "$WORK/run.skip.log" 2>&1); then
  fail "UNWOVEN consumer passed its own weave check — the step-6 gate is vacuous"
fi
grep -q "UNWOVEN" "$WORK/run.skip.log" || { cat "$WORK/run.skip.log"; fail "expected UNWOVEN report from the skipped build"; }
echo "escape hatch works; step-6 gate proven non-vacuous"

step "9/18 pack the consumer → depends on NumSharp, NOT on NumSharp.Build"
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
if "NumSharp.Build" in text:
    print("FAIL: NumSharp.Build leaked into the consumer's dependencies"); sys.exit(1)
print("consumer nuspec OK: NumSharp present, NumSharp.Build absent")
PY

# ---------------------------------------------------------------- 9b-9d: the NOT-CONTAGIOUS guard (NDW018)
# Step 9 covers the reference `dotnet add package` writes (PrivateAssets="all", from developmentDependency).
# A HAND-WRITTEN reference has no such help: NuGet would list NumSharp.Build as a dependency of the
# author's package (exclude="Build,Analyzers" — inert for weaving, but every consumer would restore the
# weaver's payload). The packaged targets refuse that pack (NDW018) unless NumSharpBuildAllowAsDependency
# is set — and even when it IS set, weaving must not reach the consumer.
nuspec_deps() { # <nupkg> → prints the <dependency .../> lines
  python - "$(winpath "$1")" <<'PY'
import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
text = z.read(next(n for n in z.namelist() if n.endswith(".nuspec"))).decode("utf-8")
for d in re.findall(r'<dependency [^>]*/>', text): print(d)
PY
}
HW="$WORK/handwritten"
mkdir -p "$HW"
cp "$CONSUMER/nuget.config" "$HW/nuget.config"
cat > "$HW/LibHW.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework><Version>1.0.0</Version></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Build" Version="$WVER" />
  </ItemGroup>
</Project>
EOF
cat > "$HW/LibHW.cs" <<'EOF'
using NumSharp;
public static class LibHW
{
    [NDScoped]
    public static NDArray Twice(NDArray a) { var t = a + 0.0; return t * 2.0; }
}
EOF
echo "9b: a hand-written reference (no PrivateAssets) builds and weaves, but refuses to PACK"
(cd "$HW" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$HW/build.log"; fail "9b: hand-written-reference library did not build"; }
grep -q "NumSharp.Build: LibHW.dll - woven 1" "$HW/build.log" || fail "9b: hand-written-reference library was not woven"
if (cd "$HW" && dotnet pack -c Release --no-build -o "$HW/out" -v q --nologo > pack.log 2>&1); then
  fail "9b: packing a library that references NumSharp.Build WITHOUT PrivateAssets succeeded — the not-contagious guard is missing"
fi
grep -q "error NDW018" "$HW/pack.log" || { tail -20 "$HW/pack.log"; fail "9b: the refused pack did not report NDW018"; }
grep -q 'PrivateAssets="all"' "$HW/pack.log" || fail "9b: NDW018 does not name the fix (PrivateAssets=\"all\")"
[ -z "$(ls "$HW/out"/*.nupkg 2>/dev/null)" ] || fail "9b: a nupkg was produced despite NDW018"
echo "9b: NDW018 refused the pack; no nupkg written"

echo "9c: -p:NumSharpBuildAllowAsDependency=true packs it as a dependency on purpose — and its consumer is STILL not woven"
(cd "$HW" && dotnet pack -c Release --no-build -o "$HW/outc" -v q --nologo -p:NumSharpBuildAllowAsDependency=true > packc.log 2>&1) \
  || { tail -30 "$HW/packc.log"; fail "9c: the opt-out knob did not let the pack through"; }
nuspec_deps "$HW/outc/LibHW.1.0.0.nupkg" | grep -q 'id="NumSharp.Build"' || fail "9c: with the knob set, NumSharp.Build should be listed as a dependency"
FEED2="$WORK/feed2"
mkdir -p "$FEED2"
cp "$FEED"/*.nupkg "$HW/outc/LibHW.1.0.0.nupkg" "$FEED2/"
rm -rf "$GPF/libhw"
AC="$WORK/appc"
mkdir -p "$AC"
cat > "$AC/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="weaver-local2" value="$(winpath "$FEED2")" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
EOF
cat > "$AC/AppC.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup><PackageReference Include="LibHW" Version="1.0.0" /></ItemGroup>
</Project>
EOF
cat > "$AC/Program.cs" <<'EOF'
using System;
using System.Linq;
using NumSharp;
internal static class LocalC
{
    [NDScoped]  // inert here: AppC never installed the weaver, whatever LibHW's nuspec says
    public static NDArray Same(NDArray a) { var t = a + 1.0; return t - 1.0; }
}
internal static class Program
{
    private static int Main()
    {
        bool libWoven = typeof(LibHW).GetMethod("Twice").GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope));
        bool appWoven = typeof(LocalC).GetMethod("Same").GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope));
        Console.WriteLine($"lib-woven={libWoven} app-woven={appWoven}");
        return libWoven && !appWoven ? 0 : 1;
    }
}
EOF
(cd "$AC" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$AC/build.log"; fail "9c: consumer of the knob-packed library did not build"; }
grep -q '"NumSharp.Build/' "$AC/obj/project.assets.json" || fail "9c: the consumer's restore graph should contain the leaked NumSharp.Build (that is what the knob allows)"
if grep -q "NumSharp.Build: AppC.dll" "$AC/build.log"; then fail "9c: the weaver ran on the CONSUMER of a library that lists NumSharp.Build as a dependency"; fi
if grep -q "NumSharp.Build.targets" "$AC/obj/AppC.csproj.nuget.g.targets"; then fail "9c: NumSharp.Build.targets was imported transitively into the consumer"; fi
(cd "$AC" && dotnet run --no-build -c Release > run.log 2>&1) && grep -q "lib-woven=True app-woven=False" "$AC/run.log" \
  || { cat "$AC/run.log" 2>/dev/null; fail "9c: consumer weave-isolation assertion failed"; }
echo "9c: dependency listed on purpose, consumer restores it, consumer is NOT woven and imports no weaver targets"

echo "9d: the fix the error names — PrivateAssets=\"all\" by hand — packs clean; the Central Package Management spelling without it draws NDW018"
sed -i 's|<PackageReference Include="NumSharp.Build" Version="'"$WVER"'" />|<PackageReference Include="NumSharp.Build" Version="'"$WVER"'" PrivateAssets="all" />|' "$HW/LibHW.csproj"
grep -q 'PrivateAssets="all"' "$HW/LibHW.csproj" || fail "9d: could not rewrite the hand-written reference"
(cd "$HW" && dotnet pack -c Release -o "$HW/outd" -v q --nologo > packd.log 2>&1) || { tail -30 "$HW/packd.log"; fail "9d: pack with PrivateAssets=all failed"; }
if nuspec_deps "$HW/outd/LibHW.1.0.0.nupkg" | grep -q 'id="NumSharp.Build"'; then fail "9d: NumSharp.Build still listed after PrivateAssets=all"; fi
nuspec_deps "$HW/outd/LibHW.1.0.0.nupkg" | grep -q 'id="NumSharp"' || fail "9d: the NumSharp dependency went missing"
CPM="$WORK/cpm"
mkdir -p "$CPM"
cp "$CONSUMER/nuget.config" "$CPM/nuget.config"
cp "$HW/LibHW.cs" "$CPM/LibHW.cs"
cat > "$CPM/Directory.Packages.props" <<EOF
<Project>
  <PropertyGroup><ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally></PropertyGroup>
  <ItemGroup>
    <PackageVersion Include="NumSharp" Version="$CVER" />
    <PackageVersion Include="NumSharp.Build" Version="$WVER" />
  </ItemGroup>
</Project>
EOF
cat > "$CPM/LibCPM.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework><Version>1.0.0</Version></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" />
    <PackageReference Include="NumSharp.Build" />
  </ItemGroup>
</Project>
EOF
if (cd "$CPM" && dotnet pack -c Release -o "$CPM/out" -v q --nologo > pack.log 2>&1); then
  fail "9d: a Central-Package-Management reference without PrivateAssets packed — NDW018 missed the CPM assets-file shape"
fi
grep -q "error NDW018" "$CPM/pack.log" || { tail -20 "$CPM/pack.log"; fail "9d: the refused CPM pack did not report NDW018"; }
echo "9d: PrivateAssets=all packs clean (NumSharp.Build absent); CPM without it is refused with NDW018"

# ---------------------------------------------------------------- steps 10-14: adversarial shapes

# Scaffolds a single-TFM net8.0 consumer that takes BOTH NumSharp and the weaver as
# PackageReferences from the local feed — the real-world composition.
scaffold_pkgref() { # <dir> <extra-propertygroup-xml>
  mkdir -p "$1"
  cp "$CONSUMER/nuget.config" "$1/nuget.config"
  cat > "$1/Consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>Consumer</AssemblyName>
    $2
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Build" Version="$WVER">
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>
</Project>
EOF
}

step "10/18 PackageReference-NumSharp consumer (lib/ resolution, the real product path)"
PD="$WORK/pkgref"
scaffold_pkgref "$PD" ""
cp "$CONSUMER/Program.cs" "$PD/Program.cs"
(cd "$PD" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$PD/build.log"; fail "PackageReference consumer build failed"; }
grep -q "woven 3, already-scoped 0" "$PD/build.log" || fail "PackageReference consumer did not weave"
(cd "$PD" && dotnet run --no-build -c Release > run.log 2>&1) || { cat "$PD/run.log"; fail "PackageReference consumer run failed"; }
grep -q "CONSUMER-OK" "$PD/run.log" || { cat "$PD/run.log"; fail "no CONSUMER-OK from the PackageReference consumer"; }
echo "net8.0 PackageReference consumer woven and green"

# 10b — the SAME pure-nupkg consumer on net10.0: proves lib/net10.0 resolves from the package and
# the TFM-agnostic analyzers/ + build/ assets serve the second TFM too (the mixed P2P consumer of
# steps 3-9 covers net10.0 weaving, but not the everything-from-the-nupkg composition). The extra
# property lands AFTER the template's TargetFramework, so the later assignment overrides it — the
# same mechanism step 14 uses for SignAssembly.
PD10="$WORK/pkgref10"
scaffold_pkgref "$PD10" "<TargetFramework>net10.0</TargetFramework>"
cp "$CONSUMER/Program.cs" "$PD10/Program.cs"
(cd "$PD10" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$PD10/build.log"; fail "10b: net10.0 PackageReference consumer build failed"; }
grep -q "woven 3, already-scoped 0" "$PD10/build.log" || fail "10b: net10.0 PackageReference consumer did not weave"
(cd "$PD10" && dotnet run --no-build -c Release > run.log 2>&1) || { cat "$PD10/run.log"; fail "10b: net10.0 PackageReference consumer run failed"; }
grep -q "CONSUMER-OK" "$PD10/run.log" || { cat "$PD10/run.log"; fail "10b: no CONSUMER-OK from the net10.0 PackageReference consumer"; }
echo "net10.0 PackageReference consumer woven and green"

step "11/18 error shapes fail the build — the ANALYZER at compile-time, the weaver post-compile"
# The compile-time Roslyn analyzer arrives via the NumSharp PackageReference (analyzers/dotnet/cs/
# in THAT package); NumSharp.Build adds the post-compile weaver. The analyzer's ERROR diagnostics
# preempt the weaver — a failed CoreCompile never reaches the weave — so the SOURCE-detectable
# rejections (wrong attribute NDW009/010/011, ref egress NDW002, unsupported carrier NDW003)
# surface at COMPILE, while the IL-only NDW004 (an unrecognized state-machine shape) surfaces
# POST-COMPILE on a file the analyzer passes.

# 11a — analyzer (compile-time): every source-detectable rejection, all in ONE failed compile.
EA="$WORK/errors_analyzer"
scaffold_pkgref "$EA" ""
cat > "$EA/Program.cs" <<'EOF'
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

internal static class BadShapes
{
    [NDScoped] public static List<NDArray> Carrier(NDArray a) => new List<NDArray> { a + 1 };                                    // NDW003
    [NDScopedAsync] public static async Task<List<NDArray>> AsyncCarrier(NDArray a) { await Task.Yield(); return new List<NDArray> { a + 1 }; } // NDW003
    [NDScoped] public static void RefEgress(ref NDArray a) { a = a + 1; }                                                          // NDW002
    [NDScoped] public static async Task<NDArray> AsyncUnderSync(NDArray a) { await Task.Yield(); return a + 1; }                   // NDW009
    [NDScopedAsync] public static NDArray SyncUnderAsync(NDArray a) => a + 1;                                                      // NDW010
    [NDScoped] [NDScopedAsync] public static async Task<NDArray> HasBoth(NDArray a) { await Task.Yield(); return a + 1; }          // NDW011
}

internal static class Program { private static int Main() => 0; }
EOF
if (cd "$EA" && dotnet build -c Release -v n > build.log 2>&1); then
  fail "11a: the analyzer error-shape consumer BUILD SUCCEEDED — the analyzer must fail the build"
fi
for code in NDW002 NDW003 NDW009 NDW010 NDW011; do
  grep -q "error $code" "$EA/build.log" || { grep -o "error NDW[0-9]*" "$EA/build.log" | sort -u; fail "11a: missing $code (analyzer) in the failed build output"; }
done
grep -q "NumSharp.Build:.*woven" "$EA/build.log" && fail "11a: the weaver ran despite an analyzer compile error"
echo "11a: analyzer reported NDW002/003/009/010/011 at compile-time; the weave was preempted"

# 11b — weaver (post-compile): a hand-rolled bogus [AsyncStateMachine] the analyzer PASSES (a real,
#       if lying, Task-returning method) but the weaver cannot recognize → NDW004.
EW="$WORK/errors_weaver"
scaffold_pkgref "$EW" ""
cat > "$EW/Program.cs" <<'EOF'
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp;

internal static class BadShapes
{
    [NDScopedAsync]
    [AsyncStateMachine(typeof(BogusStateMachine))]
    public static Task NotReallyAsync() => Task.CompletedTask;   // NDW004 — weaver-only, analyzer passes it

    private struct BogusStateMachine { }
}

internal static class Program { private static int Main() => 0; }
EOF
if (cd "$EW" && dotnet build -c Release -v n > build.log 2>&1); then
  fail "11b: the weaver error-shape consumer BUILD SUCCEEDED — NDW004 must fail the build"
fi
grep -q "error NDW004" "$EW/build.log" || { grep -o "error NDW[0-9]*" "$EW/build.log" | sort -u; fail "11b: missing NDW004 (weaver) in the failed build output"; }
echo "11b: weaver reported NDW004 post-compile (the analyzer passed the shape)"

step "12/18 attribute-free consumer no-ops (assembly untouched)"
ND="$WORK/noattr"
scaffold_pkgref "$ND" ""
cat > "$ND/Program.cs" <<'EOF'
using System;
using NumSharp;
internal static class Program
{
    private static int Main()
    {
        Console.WriteLine("PLAIN-OK " + (np.arange(1.0, 4.0) + 1.0).GetDouble(2));
        return 0;
    }
}
EOF
(cd "$ND" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$ND/build.log"; fail "attribute-free consumer build failed"; }
# An attribute-free consumer that REFERENCES NumSharp still spawns the tool (it may hold an
# override INHERITING [NDScoped] from a referenced declaration — no attribute text of its own for
# the ASCII pre-scan to find), and the tool answers "no [NDScoped]/[NDScopedAsync]/[NDScopedExit]
# methods; nothing to do" without rewriting the file; the pre-scan's own "usage detected" skip is
# the wording for a project that does not reference NumSharp at all. Accept either — reject only
# silence — and prove the no-op did not rewrite (no "woven" summary for this assembly).
grep -Eq "no \[NDScoped\]/\[NDScopedAsync\] usage detected|no \[NDScoped\]/\[NDScopedAsync\](/\[NDScopedExit\])? methods; nothing to do" "$ND/build.log" \
  || fail "expected the no-op report (pre-scan skip or tool nothing-to-do)"
! grep -Eq "Consumer\.dll .*woven [1-9]" "$ND/build.log" || fail "attribute-free consumer must not be rewritten"
(cd "$ND" && dotnet run --no-build -c Release > run.log 2>&1) && grep -q "PLAIN-OK 4" "$ND/run.log" \
  || { cat "$ND/run.log"; fail "attribute-free consumer run failed"; }
echo "no-op verified (assembly left unwritten)"

step "13/18 transitive isolation: AppB references woven LibA, is NOT woven itself"
TD="$WORK/transitive"
mkdir -p "$TD/liba" "$TD/appb"
cp "$CONSUMER/nuget.config" "$TD/nuget.config"
cat > "$TD/liba/LibA.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Build" Version="$WVER"><PrivateAssets>all</PrivateAssets></PackageReference>
  </ItemGroup>
</Project>
EOF
cat > "$TD/liba/LibA.cs" <<'EOF'
using NumSharp;
public static class LibA
{
    [NDScoped]
    public static NDArray Twice(NDArray a) { var t = a + 0.0; return t * 2.0; }
}
EOF
cat > "$TD/appb/AppB.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <ProjectReference Include="../liba/LibA.csproj" />
  </ItemGroup>
</Project>
EOF
cat > "$TD/appb/Program.cs" <<'EOF'
using System;
using System.Linq;
using System.Reflection;
using NumSharp;

internal static class LocalB
{
    [NDScoped]  // inert here: AppB does NOT install the weaver
    public static NDArray Same(NDArray a) { var t = a + 1.0; return t - 1.0; }
}

internal static class Program
{
    private static int Main()
    {
        bool aWoven = typeof(LibA).GetMethod("Twice").GetMethodBody()
            .LocalVariables.Any(v => v.LocalType == typeof(NDScope));
        bool bWoven = typeof(LocalB).GetMethod("Same").GetMethodBody()
            .LocalVariables.Any(v => v.LocalType == typeof(NDScope));
        bool values = Math.Abs(LibA.Twice(np.arange(1.0, 4.0)).GetDouble(2) - 6.0) < 1e-12
                      && Math.Abs(LocalB.Same(np.arange(1.0, 4.0)).GetDouble(0) - 1.0) < 1e-12;
        Console.WriteLine($"A-woven={aWoven} B-woven={bWoven} values={values}");
        return aWoven && !bWoven && values ? 0 : 1;
    }
}
EOF
(cd "$TD/appb" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$TD/appb/build.log"; fail "transitive build failed"; }
if grep "NumSharp.Build:" "$TD/appb/build.log" | grep -q "AppB.dll"; then
  fail "the weaver ran on AppB — build/ assets leaked transitively"
fi
(cd "$TD/appb" && dotnet run --no-build -c Release > run.log 2>&1) \
  && grep -q "A-woven=True B-woven=False values=True" "$TD/appb/run.log" \
  || { cat "$TD/appb/run.log" 2>/dev/null; fail "transitive isolation assertion failed"; }
echo "LibA woven, AppB untouched (attribute inert without the package)"

step "14/18 strong-named consumer is re-signed with its own key and runs"
SD="$WORK/signed"
scaffold_pkgref "$SD" "<SignAssembly>true</SignAssembly><AssemblyOriginatorKeyFile>consumer.snk</AssemblyOriginatorKeyFile>"
cp "$ROOT/Open.snk" "$SD/consumer.snk"
cp "$CONSUMER/Program.cs" "$SD/Program.cs"
(cd "$SD" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$SD/build.log"; fail "signed consumer build failed"; }
grep -q "woven 3, already-scoped 0, re-signed" "$SD/build.log" || fail "signed consumer was not re-signed"
(cd "$SD" && dotnet run --no-build -c Release > run.log 2>&1) && grep -q "CONSUMER-OK" "$SD/run.log" \
  || { cat "$SD/run.log"; fail "signed consumer run failed"; }
python - "$(winpath "$SD/bin/Release/net8.0/Consumer.dll")" <<'PY'
import sys
# The strong-name evidence readable without .NET: the CorFlags StrongNameSigned bit (0x8) in the
# CLI header — a delay-signed image carries the key but leaves this bit clear.
data = open(sys.argv[1], "rb").read()
pe = int.from_bytes(data[0x3C:0x40], "little")
opt = pe + 24
magic = int.from_bytes(data[opt:opt + 2], "little")
ddoff = opt + (96 if magic == 0x10B else 112)
cli_rva = int.from_bytes(data[ddoff + 14 * 8:ddoff + 14 * 8 + 4], "little")
nsec = int.from_bytes(data[pe + 6:pe + 8], "little")
shoff = opt + int.from_bytes(data[pe + 20:pe + 22], "little")
cli = None
for i in range(nsec):
    s = shoff + 40 * i
    va = int.from_bytes(data[s + 12:s + 16], "little")
    vs = int.from_bytes(data[s + 8:s + 12], "little")
    raw = int.from_bytes(data[s + 20:s + 24], "little")
    if va <= cli_rva < va + vs:
        cli = raw + (cli_rva - va)
        break
flags = int.from_bytes(data[cli + 16:cli + 20], "little")
ok = (flags & 0x8) != 0
print("CorFlags.StrongNameSigned =", ok)
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail "re-signed consumer lacks the StrongNameSigned flag"
echo "re-signed with the consumer's key; signature flag set; runs green"

step "15/18 state-machine weave: async / iterator / non-async-Task methods woven and behaving"
AD="$WORK/asyncsm"
scaffold_pkgref "$AD" ""
cat > "$AD/Program.cs" <<'EOF'
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;

internal static class Subjects
{
    [NDScopedAsync]
    public static async Task<NDArray> AsyncCompute(NDArray input, List<NDArray> temps)
    {
        var t1 = input + 1.0;                        // hoisted across the await
        temps.Add(t1);
        var t2 = t1 * 2.0;                           // held by the awaited callee IN FLIGHT
        temps.Add(t2);
        var v = await SlowReadAsync(t2);             // real suspension; resumes on the pool
        var t3 = t1 + v;
        temps.Add(t3);
        await Task.Yield();
        return t3 + 0.5;
    }

    private static async Task<double> SlowReadAsync(NDArray operand)
    {
        await Task.Delay(25).ConfigureAwait(false);
        return operand.GetDouble(0);                 // UAF probe: reads after the caller suspended
    }

    [NDScoped]
    public static IEnumerable<NDArray> Yields(int n, List<NDArray> temps)
    {
        var acc = np.zeros(3);
        temps.Add(acc);
        for (int i = 0; i < n; i++)
        {
            var step = acc + i;
            temps.Add(step);
            yield return step * 10.0;
        }
    }

    [NDScopedAsync]
    public static Task<NDArray> Forwarding(SemaphoreSlim release, List<NDArray> temps)
    {
        var t = np.arange(6.0) + 1.0;
        temps.Add(t);
        return UseLaterAsync(release, t);
    }

    private static async Task<NDArray> UseLaterAsync(SemaphoreSlim release, NDArray operand)
    {
        await release.WaitAsync().ConfigureAwait(false);
        return operand + 10.0;                       // reads the tracked temp AFTER the caller returned
    }
}

internal static class Program
{
    private static int _failures;
    private static void Check(bool ok, string what) { if (!ok) { _failures++; Console.Error.WriteLine("FAIL: " + what); } }

    private static int Main()
    {
        // structure: state machines carry the slot field + a woven MoveNext
        foreach (var name in new[] { "AsyncCompute", "Yields" })
        {
            var m = typeof(Subjects).GetMethod(name, BindingFlags.Public | BindingFlags.Static);
            var sm = m.GetCustomAttributes<StateMachineAttribute>().First().StateMachineType;
            Check(sm.GetField("<>ndscope", BindingFlags.NonPublic | BindingFlags.Instance) != null, name + ": no <>ndscope slot");
            var mn = sm.GetMethod("MoveNext", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            Check(mn.GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope)), name + ": MoveNext not woven");
        }

        // async: temps reclaimed AT completion (deterministically — before the awaiter observes it)
        {
            var temps = new List<NDArray>();
            var r = Subjects.AsyncCompute(np.arange(4.0), temps).GetAwaiter().GetResult();
            Check(Math.Abs(r.GetDouble(1) - 4.5) < 1e-9, "async: wrong value");
            Check(!r.IsDisposed, "async: result reclaimed");
            Check(temps.All(t => t.IsDisposed), "async: temps not reclaimed at completion");
        }

        // iterator: yielded elements owned by the consumer; early break reclaims via Dispose
        {
            var temps = new List<NDArray>();
            var yielded = Subjects.Yields(3, temps).ToList();
            Check(yielded.All(y => !y.IsDisposed) && yielded[2].GetDouble(0) == 20.0, "iterator: yields wrong/reclaimed");
            Check(temps.All(t => t.IsDisposed), "iterator: temps not reclaimed at end");

            temps.Clear();
            NDArray first = null;
            foreach (var y in Subjects.Yields(5, temps)) { first = y; break; }
            Check(first is not null && !first.IsDisposed, "iterator(abandon): yield reclaimed");
            Check(temps.All(t => t.IsDisposed), "iterator(abandon): temps not reclaimed by Dispose");
        }

        // non-async incomplete Task: deferral protects the in-flight operand, sweeps at completion
        {
            var temps = new List<NDArray>();
            using var release = new SemaphoreSlim(0);
            var task = Subjects.Forwarding(release, temps);
            Thread.Sleep(30);
            Check(temps.All(t => !t.IsDisposed), "deferral: temp reclaimed while in flight");
            release.Release();
            var r = task.GetAwaiter().GetResult();
            Check(r.GetDouble(0) == 11.0, "deferral: operand corrupted");
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (temps.Any(t => !t.IsDisposed) && DateTime.UtcNow < deadline) Thread.Sleep(10);
            Check(temps.All(t => t.IsDisposed), "deferral: temps never reclaimed after completion");
        }

        Console.WriteLine(_failures == 0 ? "ASYNC-SM-OK" : "ASYNC-SM-FAILURES " + _failures);
        return _failures == 0 ? 0 : 1;
    }
}
EOF
(cd "$AD" && dotnet build -c Release -v n > build.log 2>&1)   || { tail -30 "$AD/build.log"; fail "async state-machine consumer build failed"; }
grep -q "woven 3, already-scoped 0" "$AD/build.log" || fail "async consumer did not weave all 3 subjects"
(cd "$AD" && dotnet run --no-build -c Release > run.log 2>&1) || { cat "$AD/run.log"; fail "async consumer run failed"; }
grep -q "ASYNC-SM-OK" "$AD/run.log" || { cat "$AD/run.log"; fail "async state-machine behaviors failed"; }
echo "async/iterator/deferred-task consumers woven and green"

step "16/18 leak analyzer (NDW012): a dropped NDArray warns; a [NDScoped] method with the same body does not"
# The NumSharp package's analyzers/dotnet/cs/ carries the leak analyzer. A method that is NOT scoped
# and drops an NDArray it never returns/disposes must draw NDW012; a [NDScoped] method is exempt.
LK="$WORK/leak"
scaffold_pkgref "$LK" ""
cat > "$LK/Program.cs" <<'EOF'
using NumSharp;
internal static class Leaks
{
    public static double Bad(NDArray a, NDArray b) { var t = a + b; return 0.0; }        // NDW012 — 't' leaks
    [NDScoped] public static NDArray Good(NDArray a, NDArray b) { var t = a + b; return t.copy(); } // exempt
}
internal static class Program { private static int Main() => 0; }
EOF
(cd "$LK" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$LK/build.log"; fail "leak-analyzer consumer build failed"; }
LKN=$(grep -oE "Program\.cs\([0-9]+,[0-9]+\): warning NDW012" "$LK/build.log" | sort -u | wc -l)
[ "$LKN" -eq 1 ] || { grep -i ndw012 "$LK/build.log" | head; fail "16: expected exactly one NDW012 site (the non-scoped leak), saw $LKN"; }
echo "16: NDW012 warns on the dropped NDArray and exempts the [NDScoped] method"

# 16b — the type-level ownership pass rides the same analyzer: a type that STORES an NDArray without
# being disposable draws NDW016, a disposable that never disposes its holder draws NDW017, and a
# [NDBorrowed] member/type draws neither.
OW="$WORK/ownership"
scaffold_pkgref "$OW" ""
cat > "$OW/Program.cs" <<'EOF'
using NumSharp;
internal sealed class Stores { private NDArray _a; public Stores(NDArray a) { _a = a; } }                       // NDW016
internal sealed class Forgets : System.IDisposable { private NDArray _a; public void Dispose() { } }              // NDW017
internal sealed class Owns : System.IDisposable { private NDArray _a; public void Dispose() => _a?.Dispose(); }   // clean
[NDBorrowed] internal sealed class Borrows { private NDArray _a; public Borrows(NDArray a) { _a = a; } }          // clean
internal sealed class BorrowsOne { [NDBorrowed] private NDArray _a; public BorrowsOne(NDArray a) { _a = a; } }   // clean
internal static class Program { private static int Main() => 0; }
EOF
(cd "$OW" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$OW/build.log"; fail "ownership-analyzer consumer build failed"; }
OW16=$(grep -oE "Program\.cs\([0-9]+,[0-9]+\): warning NDW016" "$OW/build.log" | sort -u | wc -l)
OW17=$(grep -oE "Program\.cs\([0-9]+,[0-9]+\): warning NDW017" "$OW/build.log" | sort -u | wc -l)
[ "$OW16" -eq 1 ] || { grep -i ndw016 "$OW/build.log" | head; fail "16b: expected exactly one NDW016 site (the non-disposable holder), saw $OW16"; }
[ "$OW17" -eq 1 ] || { grep -i ndw017 "$OW/build.log" | head; fail "16b: expected exactly one NDW017 site (the forgetful disposable), saw $OW17"; }
grep -q "Program.cs(2," "$OW/build.log" || fail "16b: NDW016 did not land on the storing type"
grep -q "Program.cs(3," "$OW/build.log" || fail "16b: NDW017 did not land on the forgotten member"
echo "16b: NDW016/NDW017 warn on the storing type and the forgotten member; the disposing and [NDBorrowed] types are clean"

step "17/18 weaver-missing guard (NDW013): [NDScoped] without the weaver warns; installing it silences"
# NDW013 ships in the NumSharp package: its analyzer (analyzers/dotnet/cs/, auto-applied to the consumer's
# compile) reports it PER MEMBER with a source location, and its build/NumSharp.targets carries a
# project-wide text-scan FALLBACK that yields whenever that analyzer is applied. So a consumer that
# references NumSharp but NOT NumSharp.Build gets the "your [NDScoped] is inert" warning exactly once
# per attributed member — from the analyzer — never a second, project-wide copy from the scan.
NW="$WORK/noweaver"
mkdir -p "$NW"
cp "$CONSUMER/nuget.config" "$NW/nuget.config"
cat > "$NW/Consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Library</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup><PackageReference Include="NumSharp" Version="$CVER" /></ItemGroup>
</Project>
EOF
cat > "$NW/S.cs" <<'EOF'
using NumSharp;
public static class S { [NDScoped] public static NDArray F(NDArray a) => a + 1.0; }
EOF
(cd "$NW" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$NW/build.log"; fail "no-weaver consumer build failed"; }
grep -q "warning NDW013" "$NW/build.log" || { grep -i ndw013 "$NW/build.log"; fail "17: NDW013 did not fire without the weaver"; }
# the ANALYZER is the reporter: the warning carries the attributed member's source location and sentence ...
grep -qE 'S\.cs\([0-9]+,[0-9]+\): warning NDW013' "$NW/build.log" \
  || { grep -i ndw013 "$NW/build.log" | head -5; fail "17: NDW013 did not come from the analyzer (no source location on the warning)"; }
grep -q "\[NDScoped\] on 'S.F(NumSharp.NDArray)'" "$NW/build.log" \
  || { grep -i ndw013 "$NW/build.log" | head -5; fail "17: the analyzer's per-member NDW013 sentence is missing"; }
# ... and the MSBuild fallback scan stayed silent (its sentence is project-wide: "<project> uses [NDScoped]...")
if grep -q "Consumer uses \[NDScoped\]" "$NW/build.log"; then
  fail "17: the MSBuild fallback scan fired although the analyzer is applied — NDW013 reported twice"
fi
(cd "$NW" && dotnet add Consumer.csproj package NumSharp.Build --source "$FEEDW" > add.log 2>&1) || { tail -20 "$NW/add.log"; fail "17: dotnet add NumSharp.Build failed"; }
(cd "$NW" && dotnet build -c Release -t:Rebuild -v n > build2.log 2>&1) || { tail -30 "$NW/build2.log"; fail "17: with-weaver build failed"; }
if grep -q "warning NDW013" "$NW/build2.log"; then fail "17: NDW013 still fired after installing the weaver"; fi
echo "17: NDW013 warns (from the analyzer, per member, scan silent) without the weaver and is silent once it is installed"

# 17b — the FALLBACK: a compile the analyzer does not reach has nothing to report NDW013 per member, so the
# MSBuild text scan in the package's build/NumSharp.targets must fire (project-wide) instead. The
# analyzer-less compile is a bare <Reference HintPath> to the PACKAGED NumSharp.dll (no PackageReference,
# so NuGet applies nothing) that imports the PACKAGED build/NumSharp.targets by hand — both taken from the
# copy step 10 restored into the global packages folder.
#   NOT the way to get here: `ExcludeAssets="analyzers"` on the PackageReference. NuGet does record it
#   (project.assets.json lists the include set WITHOUT Analyzers), but the SDK's ResolvePackageAssets
#   (observed on 10.0.101) still hands analyzers/dotnet/cs/*.dll to csc — so such a consumer KEEPS the
#   analyzer, keeps its per-member NDW013, and the scan correctly stays silent for it.
NWX="$WORK/noweaver_noanalyzer"
mkdir -p "$NWX"
NPKG="$GPF/numsharp/$CVER"
[ -f "$NPKG/lib/net8.0/NumSharp.dll" ] && [ -f "$NPKG/build/net8.0/NumSharp.targets" ] \
  || fail "17b: the restored NumSharp package (lib/net8.0/NumSharp.dll + build/net8.0/NumSharp.targets) not found under $NPKG"
# The targets sit under build/net8.0/ (NOT the TFM-agnostic build/ root) so that a netstandard2.0/net6.0
# consumer gets NU1202 instead of a "compatible" restore with no lib/ — asserted here so the layout cannot
# silently regress to the root.
[ ! -f "$NPKG/build/NumSharp.targets" ] || fail "17b: build/NumSharp.targets must NOT sit at the TFM-agnostic build/ root (NU1202 regression for unsupported TFMs)"
NPKGW="$(winpath "$NPKG")"
cat > "$NWX/Consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Library</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup>
    <Reference Include="NumSharp"><HintPath>$NPKGW/lib/net8.0/NumSharp.dll</HintPath></Reference>
  </ItemGroup>
  <Import Project="$NPKGW/build/net8.0/NumSharp.targets" />
</Project>
EOF
cp "$NW/S.cs" "$NWX/S.cs"
(cd "$NWX" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$NWX/build.log"; fail "17b: analyzer-less consumer build failed"; }
if grep -q "NumSharp.Build.Analyzer" "$NWX/build.log"; then fail "17b: the analyzer reached an analyzer-less (bare Reference) compile"; fi
grep -q "Consumer uses \[NDScoped\]" "$NWX/build.log" \
  || { grep -i ndw013 "$NWX/build.log" | head -5; fail "17b: the MSBuild fallback scan did not fire for the analyzer-less consumer"; }
if grep -qE 'S\.cs\([0-9]+,[0-9]+\): warning NDW013' "$NWX/build.log"; then
  fail "17b: a per-member (analyzer) NDW013 appeared in an analyzer-less compile"
fi
echo "17b: without the analyzer (bare Reference to the packaged DLL), the packaged MSBuild fallback scan reports NDW013 project-wide"

step "18/18 analyzer ships in NumSharp itself: diagnostics WITHOUT NumSharp.Build installed"
# The whole point of staging the analyzer into the NumSharp package: a consumer referencing NumSharp
# ALONE (no weaver) already gets the compile-time [NDScoped]-target gate and the NDW012 leak nudge.
# 18a — a bad [NDScoped] target FAILS the build (NDW003) with only the NumSharp PackageReference.
AA="$WORK/analyzer_alone"
mkdir -p "$AA"
cp "$CONSUMER/nuget.config" "$AA/nuget.config"
cat > "$AA/Consumer.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Library</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup><PackageReference Include="NumSharp" Version="$CVER" /></ItemGroup>
</Project>
EOF
cat > "$AA/Bad.cs" <<'EOF'
using System.Collections.Generic;
using NumSharp;
public static class Bad
{
    [NDScoped] public static List<NDArray> Carrier(NDArray a) => new List<NDArray> { a + 1 };   // NDW003
}
EOF
if (cd "$AA" && dotnet build -c Release -v n > build.log 2>&1); then
  fail "18a: bad [NDScoped] target BUILD SUCCEEDED without NumSharp.Build — the analyzer must ride the NumSharp package"
fi
grep -q "error NDW003" "$AA/build.log" || { grep -o "error NDW[0-9]*" "$AA/build.log" | sort -u; fail "18a: missing NDW003 from the NumSharp-only consumer"; }
echo "18a: NDW003 failed the build with NumSharp alone (no weaver installed)"

# 18b — the NDW012 leak warning also fires with only the NumSharp PackageReference.
AB="$WORK/analyzer_alone_leak"
mkdir -p "$AB"
cp "$CONSUMER/nuget.config" "$AB/nuget.config"
cp "$AA/Consumer.csproj" "$AB/Consumer.csproj"
cat > "$AB/Leak.cs" <<'EOF'
using NumSharp;
public static class Leaks
{
    public static double Bad(NDArray a, NDArray b) { var t = a + b; return 0.0; }   // NDW012 — 't' leaks
}
EOF
(cd "$AB" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$AB/build.log"; fail "18b: NumSharp-only leak consumer build failed"; }
grep -q "warning NDW012" "$AB/build.log" || { grep -i ndw012 "$AB/build.log"; fail "18b: NDW012 did not fire from the NumSharp-only consumer"; }
echo "18b: NDW012 warned with NumSharp alone (no weaver installed)"


echo
echo "verify_build_package: ALL 18 STEPS OK"
if [ "${KEEP:-0}" != "1" ]; then rm -rf "$WORK"; else echo "(KEEP=1: work dir kept at $WORK)"; fi
