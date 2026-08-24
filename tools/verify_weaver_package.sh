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
#  10  the REAL product path: a consumer taking NumSharp itself as a PackageReference (lib/
#      resolution, not the P2P ref-assembly path) weaves and runs
#  11  the NDW002/NDW003/NDW004 shapes surface as MSBuild ERRORS and FAIL the build
#  12  a consumer with NumSharp but no [NDScoped] methods no-ops — assembly left untouched
#  13  transitive isolation: AppB references woven LibA but does NOT install the weaver —
#      LibA stays woven, AppB's own [NDScoped] method stays inert (never woven, never an error)
#  14  a strong-named consumer is re-signed with its own key and still runs
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

step "1/14 pack NumSharp.Weaver the way CI does (build -t:Rebuild, then pack --no-build)"
dotnet build "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release -t:Rebuild -v q --nologo
dotnet pack  "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release --no-build -o "$FEEDW" -v q --nologo
PKG="$(ls "$FEED"/NumSharp.Weaver.*.nupkg 2>/dev/null | head -1)"
[ -n "$PKG" ] && [ -f "$PKG" ] || fail "pack produced no NumSharp.Weaver nupkg in $FEED"
WVER="$(basename "$PKG" .nupkg | sed 's/^NumSharp\.Weaver\.//')"
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

step "2/14 package shape: developmentDependency, no dependencies, no lib/, tools+targets payload"
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

step "3/14 scaffold the consumer (net8.0;net10.0, ProjectReference to NumSharp.Core)"
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

step "4/14 dotnet add package → PrivateAssets=all written automatically (not-a-dependency UX)"
(cd "$CONSUMER" && dotnet add Consumer.csproj package NumSharp.Weaver --source "$FEEDW" > "$WORK/add.log" 2>&1) \
  || { tail -20 "$WORK/add.log"; fail "dotnet add package NumSharp.Weaver failed"; }
grep -qi "PrivateAssets" "$CONSUMER/Consumer.csproj" \
  || fail "dotnet add package did not write PrivateAssets — developmentDependency metadata missing?"
echo "PrivateAssets written by NuGet itself:"
grep -i -A2 "NumSharp.Weaver" "$CONSUMER/Consumer.csproj" | sed 's/^/    /'

step "5/14 build → the packaged NDScopeWeave target weaves per TFM"
(cd "$CONSUMER" && dotnet build -c Release -v n > "$WORK/build1.log" 2>&1) \
  || { tail -40 "$WORK/build1.log"; fail "consumer build failed"; }
WOVEN=$(grep -c "woven 3, already-scoped 0" "$WORK/build1.log" || true)
[ "$WOVEN" -eq 2 ] || { grep -i "NumSharp.Weaver" "$WORK/build1.log" || true; fail "expected 'woven 3' for both TFMs, saw $WOVEN"; }
echo "woven 3 methods on net8.0 AND net10.0"

step "6/14 run the woven consumer on both TFMs"
for tfm in net8.0 net10.0; do
  (cd "$CONSUMER" && dotnet run --no-build -c Release -f "$tfm" > "$WORK/run.$tfm.log" 2>&1) \
    || { cat "$WORK/run.$tfm.log"; fail "consumer run failed on $tfm"; }
  grep -q "CONSUMER-OK" "$WORK/run.$tfm.log" || { cat "$WORK/run.$tfm.log"; fail "no CONSUMER-OK on $tfm"; }
  echo "$tfm: CONSUMER-OK"
done

step "7/14 incremental rebuild → weave skipped via the per-TFM marker"
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

step "8/14 -p:SkipNDScopeWeave=true → no weave, and the consumer's own gate goes red (non-vacuous)"
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

step "9/14 pack the consumer → depends on NumSharp, NOT on NumSharp.Weaver"
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
    <PackageReference Include="NumSharp.Weaver" Version="$WVER">
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>
</Project>
EOF
}

step "10/14 PackageReference-NumSharp consumer (lib/ resolution, the real product path)"
PD="$WORK/pkgref"
scaffold_pkgref "$PD" ""
cp "$CONSUMER/Program.cs" "$PD/Program.cs"
(cd "$PD" && dotnet build -c Release -v n > build.log 2>&1) \
  || { tail -30 "$PD/build.log"; fail "PackageReference consumer build failed"; }
grep -q "woven 3, already-scoped 0" "$PD/build.log" || fail "PackageReference consumer did not weave"
(cd "$PD" && dotnet run --no-build -c Release > run.log 2>&1) || { cat "$PD/run.log"; fail "PackageReference consumer run failed"; }
grep -q "CONSUMER-OK" "$PD/run.log" || { cat "$PD/run.log"; fail "no CONSUMER-OK from the PackageReference consumer"; }
echo "PackageReference consumer woven and green"

step "11/14 NDW002/NDW003/NDW004 surface as MSBuild ERRORS and fail the build"
ED="$WORK/errors"
scaffold_pkgref "$ED" ""
cat > "$ED/Program.cs" <<'EOF'
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

internal static class BadShapes
{
    [NDScoped]
    public static async Task<NDArray> Async(NDArray a) { await Task.Yield(); return a + 1; }

    [NDScoped]
    public static List<NDArray> Carrier(NDArray a) => new List<NDArray> { a + 1 };

    [NDScoped]
    public static void RefEgress(ref NDArray a) { a = a + 1; }
}

internal static class Program { private static int Main() => 0; }
EOF
if (cd "$ED" && dotnet build -c Release -v n > build.log 2>&1); then
  fail "the error-shape consumer BUILD SUCCEEDED — NDW00x must fail the build"
fi
for code in NDW002 NDW003 NDW004; do
  grep -q "error $code" "$ED/build.log" || { grep -o "error NDW[0-9]*" "$ED/build.log" | sort -u; fail "missing $code in the failed build output"; }
done
echo "all three rejection codes reported; build failed as required"

step "12/14 attribute-free consumer no-ops (assembly untouched)"
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
grep -q "no \[NDScoped\] methods; nothing to do" "$ND/build.log" || fail "expected the nothing-to-do no-op report"
(cd "$ND" && dotnet run --no-build -c Release > run.log 2>&1) && grep -q "PLAIN-OK 4" "$ND/run.log" \
  || { cat "$ND/run.log"; fail "attribute-free consumer run failed"; }
echo "no-op verified (assembly left unwritten)"

step "13/14 transitive isolation: AppB references woven LibA, is NOT woven itself"
TD="$WORK/transitive"
mkdir -p "$TD/liba" "$TD/appb"
cp "$CONSUMER/nuget.config" "$TD/nuget.config"
cat > "$TD/liba/LibA.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Weaver" Version="$WVER"><PrivateAssets>all</PrivateAssets></PackageReference>
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
if grep "NumSharp.Weaver:" "$TD/appb/build.log" | grep -q "AppB.dll"; then
  fail "the weaver ran on AppB — build/ assets leaked transitively"
fi
(cd "$TD/appb" && dotnet run --no-build -c Release > run.log 2>&1) \
  && grep -q "A-woven=True B-woven=False values=True" "$TD/appb/run.log" \
  || { cat "$TD/appb/run.log" 2>/dev/null; fail "transitive isolation assertion failed"; }
echo "LibA woven, AppB untouched (attribute inert without the package)"

step "14/14 strong-named consumer is re-signed with its own key and runs"
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

echo
echo "verify_weaver_package: ALL 14 STEPS OK"
if [ "${KEEP:-0}" != "1" ]; then rm -rf "$WORK"; else echo "(KEEP=1: work dir kept at $WORK)"; fi
