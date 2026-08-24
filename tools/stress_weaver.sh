#!/usr/bin/env bash
# =================================================================================================
# stress_weaver.sh — scale/concurrency stress harness for the NumSharp.Weaver package.
#
# Complements tools/verify_weaver_package.sh (the correctness gate): where the gate proves each
# claim once, this generates ~1,000 [NDScoped] methods across every weave shape and pushes them
# through BOTH consumption modes, then hammers the woven output at runtime.
#
#   A  SOURCE MODE ("referencing as a project"): ProjectReference to NumSharp.Core + a
#      ProjectReference to the weaver csproj (ReferenceOutputAssembly=false, builds the tool) +
#      a direct <Import> of build/NumSharp.Weaver.targets with $(NumSharpWeaverToolDll) pointed
#      at the tool's bin output — the in-repo consumption recipe. Multi-TFM (net8.0;net10.0).
#   B  NUPKG MODE: NumSharp AND NumSharp.Weaver restored from a local feed as PackageReferences.
#      Multi-TFM, plus a weave-cost measurement (Rebuild with vs without SkipNDScopeWeave).
#   P  PARALLEL MODE: 8 independent libs (120 [NDScoped] methods each, nupkg mode) built with
#      -m:8 so up to 8 weaver processes run CONCURRENTLY, plus an app referencing all 8 that
#      validates every lib assembly (and is itself weaver-free — transitive isolation at scale).
#
# Every mode's runner (RunnerMain.cs) does two things per assembly:
#   1. COVERAGE: reflects every [NDScoped] method AND property, asserts an NDScope local exists
#      in the shipped IL (the NDScopeWeaveTests invariant, at scale);
#   2. VALUES UNDER GC PRESSURE: invokes every generated method 25 times with forced full GCs
#      between sweeps — a wrong scope (double dispose / missed Returns) surfaces as a pooled
#      buffer reused under a live result, i.e. corrupted values, which the per-shape expected
#      values catch.
#
# Generated shapes (cycled): S1 bare NDArray; S2 ValueTuple; S3 NDArray[]; S4 INDArrayCarrier
# struct; S5 [NDScoped] PROPERTY; S6 bool+out NDArray; S7 void+out NDArray; S8 return inside
# try/finally; S9 scalar return (scope-only); S10 hand-scoped + attributed (idempotence at
# scale); S11 multi-return switch; specials: SB 40-statement bodies, SD 4-deep nested
# try/finally.
#
# Usage:  tools/stress_weaver.sh [workdir]     (workdir kept on failure, or with KEEP=1)
# =================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/nsstress.XXXXXX")}"
FEED="$WORK/feed"
mkdir -p "$FEED"

winpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
ROOTW="$(winpath "$ROOT")"
FEEDW="$(winpath "$FEED")"

step() { echo; echo "===== $*"; }
fail() { echo "FAIL: $*" >&2; echo "(work dir kept for inspection: $WORK)" >&2; exit 1; }

command -v dotnet >/dev/null 2>&1 || fail "dotnet not on PATH"
command -v python >/dev/null 2>&1 || fail "python not on PATH"

# ------------------------------------------------------------------ the source generator
cat > "$WORK/gen.py" <<'PYEOF'
import os, sys

outdir, n_classes, per_class, ns, anchor, specials = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6] == "1")
os.makedirs(outdir, exist_ok=True)

HEAD = "using System;\nusing NumSharp;\n\nnamespace %s;\n\n" % ns

def method(shape, mid):
    if shape == 1:
        return ("    [NDScoped]\n    public static NDArray S1_%s(NDArray a)\n"
                "    { var t = a + 1.0; var u = t * 2.0; return u - 1.0; }\n" % mid)
    if shape == 2:
        return ("    [NDScoped]\n    public static (NDArray, NDArray) S2_%s(NDArray a)\n"
                "    { var t = a + 2.0; return (t - 1.0, a * 3.0); }\n" % mid)
    if shape == 3:
        return ("    [NDScoped]\n    public static NDArray[] S3_%s(NDArray a)\n"
                "    { var t = a * 2.0; return new NDArray[] { t + 1.0, t - 1.0 }; }\n" % mid)
    if shape == 4:
        return ("    [NDScoped]\n    public static PairResult S4_%s(NDArray a)\n"
                "    { var sq = a * a; return new PairResult(a + 1.0, (sq - sq) + (a - 1.0)); }\n" % mid)
    if shape == 5:
        return ("    [NDScoped]\n    public static NDArray S5_%s\n"
                "    { get { var b = np.arange(1.0, 4.0); var t = b + 1.0; return t * 1.0; } }\n" % mid)
    if shape == 6:
        return ("    [NDScoped]\n    public static bool S6_%s(NDArray a, out NDArray r)\n"
                "    { var t = a + 5.0; r = t - 4.0; return true; }\n" % mid)
    if shape == 7:
        return ("    [NDScoped]\n    public static void S7_%s(NDArray a, out NDArray r)\n"
                "    { var t = a * 4.0; r = t / 2.0; }\n" % mid)
    if shape == 8:
        return ("    [NDScoped]\n    public static NDArray S8_%s(NDArray a)\n"
                "    { var t = a + 1.0; try { var u = t + 0.0; return u; } finally { GC.KeepAlive(t); } }\n" % mid)
    if shape == 9:
        return ("    [NDScoped]\n    public static double S9_%s(NDArray a)\n"
                "    { var t = a + 1.0; return t.GetDouble(0); }\n" % mid)
    if shape == 10:
        return ("    [NDScoped]\n    public static NDArray S10_%s(NDArray a)\n"
                "    { using var scope = NDScope.Open(); var t = a + 1.0; return scope.Returns(t * 1.0); }\n" % mid)
    # shape 11: multi-return switch (t.GetDouble(0)=2 for a=[1..5] -> case 2), every branch == 2
    return ("    [NDScoped]\n    public static NDArray S11_%s(NDArray a)\n"
            "    {\n        var t = a + 1.0;\n        switch ((int)t.GetDouble(0) %% 4)\n        {\n"
            "            case 0: { var u = t * 1.0; return u; }\n"
            "            case 1: { var v = t + 0.0; return v; }\n"
            "            case 2: return t - 0.0;\n"
            "            default: { var w = t * 2.0; return w - t; }\n        }\n    }\n" % mid)

with open(os.path.join(outdir, "Carrier.cs"), "w") as f:
    f.write(HEAD +
        "public readonly struct PairResult : INDArrayCarrier\n{\n"
        "    public readonly NDArray P;\n    public readonly NDArray M;\n"
        "    public PairResult(NDArray p, NDArray m) { P = p; M = m; }\n"
        "    void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(P); scope.Returns(M); }\n}\n\n"
        "public static class %s { }\n" % anchor)

total = hand = 0
idx = 0
for c in range(n_classes):
    body = [HEAD, "public static class GenC%d\n{\n" % c]
    for k in range(per_class):
        shape = idx % 11 + 1
        idx += 1
        total += 1
        if shape == 10:
            hand += 1
        body.append(method(shape, "%d_%d" % (c, k)))
    body.append("}\n")
    with open(os.path.join(outdir, "Gen%d.cs" % c), "w") as f:
        f.write("".join(body))

if specials:
    body = [HEAD, "public static class GenX\n{\n"]
    for k in range(5):  # SB: 40 sequential rebinds -> long body, many transients
        total += 1
        lines = "".join("        t = t + 1.0;\n" for _ in range(40))
        body.append("    [NDScoped]\n    public static NDArray SB_%d(NDArray a)\n    {\n"
                    "        var t = a + 1.0;\n%s        return t - 40.0;\n    }\n" % (k, lines))
    for k in range(10):  # SD: 4-deep nested try/finally, return from the innermost
        total += 1
        body.append(
            "    [NDScoped]\n    public static NDArray SD_%d(NDArray a)\n    {\n"
            "        var t = a + 1.0;\n"
            "        try { try { try { try { var u = t * 1.0; return u; }\n"
            "        finally { GC.KeepAlive(a); } } finally { GC.KeepAlive(t); } }\n"
            "        finally { GC.KeepAlive(a); } } finally { GC.KeepAlive(t); }\n    }\n" % k)
    body.append("}\n")
    with open(os.path.join(outdir, "GenX.cs"), "w") as f:
        f.write("".join(body))

print("TOTAL=%d;HAND=%d;WOVEN=%d" % (total, hand, total - hand))
PYEOF

# ------------------------------------------------------------------ the shared runtime validator
cat > "$WORK/RunnerMain.cs" <<'CSEOF'
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp;

public static class StressRunner
{
    private static int _failures;
    private static void Bad(string what) { _failures++; if (_failures <= 20) Console.Error.WriteLine("FAIL " + what); }
    private static void Eq(double v, double e, string what) { if (Math.Abs(v - e) > 1e-9) Bad(what + ": " + v + " != " + e); }

    public static int Run(Assembly[] assemblies)
    {
        var a = np.arange(1.0, 6.0);
        var checks = new List<Action>();
        int attributed = 0, woven = 0;

        foreach (var asm in assemblies)
        foreach (var type in asm.GetTypes())
        {
            foreach (var mm in type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                if (mm.GetCustomAttribute<NDScopedAttribute>() == null) continue;
                attributed++;
                if (mm.GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope))) woven++;
                else Bad("UNWOVEN " + type.Name + "." + mm.Name);
                var mi = mm;
                switch (mm.Name.Split('_')[0])
                {
                    case "S1":
                        checks.Add(() => Eq(((NDArray)mi.Invoke(null, new object[] { a })).GetDouble(0), 3, mi.Name)); break;
                    case "S2":
                        checks.Add(() =>
                        {
                            var t = (ITuple)mi.Invoke(null, new object[] { a });
                            Eq(((NDArray)t[0]).GetDouble(0), 2, mi.Name);
                            Eq(((NDArray)t[1]).GetDouble(0), 3, mi.Name);
                        });
                        break;
                    case "S3":
                        checks.Add(() =>
                        {
                            var r = (NDArray[])mi.Invoke(null, new object[] { a });
                            Eq(r[0].GetDouble(0), 3, mi.Name);
                            Eq(r[1].GetDouble(0), 1, mi.Name);
                        });
                        break;
                    case "S4":
                        checks.Add(() =>
                        {
                            var r = mi.Invoke(null, new object[] { a });
                            var rt = r.GetType();
                            Eq(((NDArray)rt.GetField("P").GetValue(r)).GetDouble(0), 2, mi.Name);
                            Eq(((NDArray)rt.GetField("M").GetValue(r)).GetDouble(0), 0, mi.Name);
                        });
                        break;
                    case "S6":
                        checks.Add(() =>
                        {
                            var args = new object[] { a, null };
                            if (!(bool)mi.Invoke(null, args)) Bad(mi.Name + " returned false");
                            Eq(((NDArray)args[1]).GetDouble(0), 2, mi.Name);
                        });
                        break;
                    case "S7":
                        checks.Add(() =>
                        {
                            var args = new object[] { a, null };
                            mi.Invoke(null, args);
                            Eq(((NDArray)args[1]).GetDouble(0), 2, mi.Name);
                        });
                        break;
                    case "S9":
                        checks.Add(() => Eq((double)mi.Invoke(null, new object[] { a }), 2, mi.Name)); break;
                    case "S8":
                    case "S10":
                    case "S11":
                    case "SB":
                    case "SD":
                        checks.Add(() => Eq(((NDArray)mi.Invoke(null, new object[] { a })).GetDouble(0), 2, mi.Name)); break;
                }
            }

            foreach (var pp in type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                if (pp.GetCustomAttribute<NDScopedAttribute>() == null) continue;
                attributed++;
                if (pp.GetMethod.GetMethodBody().LocalVariables.Any(v => v.LocalType == typeof(NDScope))) woven++;
                else Bad("UNWOVEN property " + pp.Name);
                var pi = pp;
                checks.Add(() => Eq(((NDArray)pi.GetValue(null)).GetDouble(0), 2, pi.Name));
            }
        }

        // 25 sweeps over EVERY generated member with forced full GCs in between: a mis-scoped
        // result whose buffer was reclaimed shows up as pool-reuse corruption in later sweeps.
        const int Sweeps = 25;
        for (int i = 0; i < Sweeps; i++)
        {
            foreach (var c in checks) c();
            if (i % 5 == 4) { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }
        }

        Console.WriteLine("attributed=" + attributed + " woven=" + woven +
                          " checks=" + checks.Count + " sweeps=" + Sweeps + " failures=" + _failures);
        bool ok = _failures == 0 && attributed == woven && attributed > 0;
        Console.WriteLine(ok ? "STRESS-OK" : "STRESS-FAIL");
        return ok ? 0 : 1;
    }
}
CSEOF

nugetcfg() { cat > "$1/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$FEEDW" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
EOF
}

# ------------------------------------------------------------------ stage 0: tool + feed
step "0  build the weaver, pack weaver + NumSharp into the local feed"
dotnet build "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release -v q --nologo
dotnet pack  "$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj" -c Release --no-build -o "$FEEDW" -v q --nologo
dotnet pack  "$ROOTW/src/NumSharp.Core/NumSharp.Core.csproj" -c Release -o "$FEEDW" \
  -p:GeneratePackageOnBuild=false -v q --nologo > "$WORK/corepack.log" 2>&1 \
  || { tail -20 "$WORK/corepack.log"; fail "NumSharp.Core pack failed"; }
WVER="$(basename "$(ls "$FEED"/NumSharp.Weaver.*.nupkg | head -1)" .nupkg | sed 's/^NumSharp\.Weaver\.//')"
CVER="$(basename "$(ls "$FEED"/NumSharp.[0-9]*.nupkg | head -1)" .nupkg | sed 's/^NumSharp\.//')"
rm -rf "$HOME/.nuget/packages/numsharp.weaver" "$HOME/.nuget/packages/numsharp/$CVER"
echo "feed: NumSharp.Weaver $WVER + NumSharp $CVER"

# ------------------------------------------------------------------ stage A: source mode
step "A  SOURCE MODE — 20x50+15 [NDScoped] members, ProjectReference + imported targets, net8.0;net10.0"
A="$WORK/srcmode"; mkdir -p "$A"
eval "$(python "$WORK/gen.py" "$A" 20 50 StressGen AnchorMain 1 | tail -1)"
echo "generated: TOTAL=$TOTAL HAND=$HAND WOVEN=$WOVEN"
cp "$WORK/RunnerMain.cs" "$A/"
cat > "$A/Program.cs" <<'EOF'
internal static class Program
{
    private static int Main() => StressRunner.Run(new[] { typeof(Program).Assembly });
}
EOF
cat > "$A/StressA.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFrameworks>net8.0;net10.0</TargetFrameworks>
    <AssemblyName>StressA</AssemblyName>
    <!-- Source-mode consumption: the packaged targets imported straight from the repo, the tool
         taken from its build output (built by the ReferenceOutputAssembly=false P2P below). -->
    <NumSharpWeaverToolDll>$ROOTW/tools/NumSharp.Weaver/bin/Release/net8.0/NumSharp.Weaver.dll</NumSharpWeaverToolDll>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="$ROOTW/src/NumSharp.Core/NumSharp.Core.csproj" />
    <ProjectReference Include="$ROOTW/tools/NumSharp.Weaver/NumSharp.Weaver.csproj"
                      ReferenceOutputAssembly="false" Private="false" />
  </ItemGroup>
  <Import Project="$ROOTW/tools/NumSharp.Weaver/build/NumSharp.Weaver.targets" />
</Project>
EOF
(cd "$A" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$A/build.log"; fail "A: build failed"; }
CNT=$(grep -c "woven $WOVEN, already-scoped $HAND" "$A/build.log" || true)
[ "$CNT" -eq 2 ] || { grep -i "weaver" "$A/build.log" | head -5; fail "A: expected 'woven $WOVEN, already-scoped $HAND' on 2 TFMs, saw $CNT"; }
for tfm in net8.0 net10.0; do
  (cd "$A" && dotnet run --no-build -c Release -f "$tfm" > "run.$tfm.log" 2>&1) || { cat "$A/run.$tfm.log"; fail "A: run $tfm failed"; }
  grep -q "STRESS-OK" "$A/run.$tfm.log" || { cat "$A/run.$tfm.log"; fail "A: no STRESS-OK on $tfm"; }
  echo "A $tfm: $(grep 'attributed=' "$A/run.$tfm.log")"
done
(cd "$A" && dotnet build -c Release -v n > build2.log 2>&1) || fail "A: incremental build failed"
grep -q "woven $WOVEN" "$A/build2.log" && fail "A: weave reran on an up-to-date build"
echo "A incremental: weave skipped"
rm "$A"/obj/Release/net8.0/NDScopeWeave.marker "$A"/obj/Release/net10.0/NDScopeWeave.marker
(cd "$A" && dotnet build -c Release -v n > build3.log 2>&1) || fail "A: marker-delete rebuild failed"
CNT=$(grep -c "woven 0, already-scoped $TOTAL, assembly unchanged" "$A/build3.log" || true)
[ "$CNT" -eq 2 ] || { grep -i "weaver" "$A/build3.log" | head -5; fail "A: expected idempotent no-write on 2 TFMs, saw $CNT"; }
echo "A idempotence at scale: woven dlls re-scanned, 0 rewoven, files unwritten"

# ------------------------------------------------------------------ stage B: nupkg mode
step "B  NUPKG MODE — same $TOTAL members, NumSharp + NumSharp.Weaver from the local feed"
B="$WORK/pkgmode"; mkdir -p "$B"; nugetcfg "$B"
python "$WORK/gen.py" "$B" 20 50 StressGen AnchorMain 1 > /dev/null
cp "$WORK/RunnerMain.cs" "$A/Program.cs" "$B/"
cat > "$B/StressB.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFrameworks>net8.0;net10.0</TargetFrameworks>
    <AssemblyName>StressB</AssemblyName>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Weaver" Version="$WVER">
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>
</Project>
EOF
(cd "$B" && dotnet build -c Release -v n > build.log 2>&1) || { tail -30 "$B/build.log"; fail "B: build failed"; }
CNT=$(grep -c "woven $WOVEN, already-scoped $HAND" "$B/build.log" || true)
[ "$CNT" -eq 2 ] || { grep -i "weaver" "$B/build.log" | head -5; fail "B: expected 'woven $WOVEN, already-scoped $HAND' on 2 TFMs, saw $CNT"; }
for tfm in net8.0 net10.0; do
  (cd "$B" && dotnet run --no-build -c Release -f "$tfm" > "run.$tfm.log" 2>&1) || { cat "$B/run.$tfm.log"; fail "B: run $tfm failed"; }
  grep -q "STRESS-OK" "$B/run.$tfm.log" || { cat "$B/run.$tfm.log"; fail "B: no STRESS-OK on $tfm"; }
  echo "B $tfm: $(grep 'attributed=' "$B/run.$tfm.log")"
done

# Weave cost: consumer-only Rebuild with vs without the weave (both TFMs, $TOTAL members each).
T0=$SECONDS
(cd "$B" && dotnet build -c Release -t:Rebuild -p:SkipNDScopeWeave=true -v q --nologo > /dev/null 2>&1) || fail "B: skip rebuild failed"
T_SKIP=$((SECONDS - T0))
T0=$SECONDS
(cd "$B" && dotnet build -c Release -t:Rebuild -v q --nologo > /dev/null 2>&1) || fail "B: weave rebuild failed"
T_WEAVE=$((SECONDS - T0))
echo "B weave cost: rebuild ${T_WEAVE}s with weave vs ${T_SKIP}s skipped (delta $((T_WEAVE - T_SKIP))s for 2 TFMs x $TOTAL members)"

# ------------------------------------------------------------------ stage P: parallel mode
step "P  PARALLEL MODE — 8 libs x 120 members, -m:8 (concurrent weaver processes), app validates all"
P="$WORK/par"; mkdir -p "$P/app"; nugetcfg "$P"
LTOTAL=""; LHAND=""; LWOVEN=""
REFS=""; ANCHORS=""
for i in 0 1 2 3 4 5 6 7; do
  L="$P/libp$i"; mkdir -p "$L"
  eval "$(python "$WORK/gen.py" "$L" 3 40 "StressP$i" "AnchorP$i" 0 | tail -1)"
  LTOTAL=$TOTAL; LHAND=$HAND; LWOVEN=$WOVEN
  cat > "$L/LibP$i.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    <PackageReference Include="NumSharp.Weaver" Version="$WVER"><PrivateAssets>all</PrivateAssets></PackageReference>
  </ItemGroup>
</Project>
EOF
  REFS="$REFS<ProjectReference Include=\"../libp$i/LibP$i.csproj\" />"
  ANCHORS="$ANCHORS typeof(StressP$i.AnchorP$i).Assembly,"
done
cp "$WORK/RunnerMain.cs" "$P/app/"
cat > "$P/app/Program.cs" <<EOF
internal static class Program
{
    private static int Main() => StressRunner.Run(new[] {$ANCHORS });
}
EOF
cat > "$P/app/StressPar.csproj" <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>StressPar</AssemblyName>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NumSharp" Version="$CVER" />
    $REFS
  </ItemGroup>
</Project>
EOF
(cd "$P/app" && dotnet build -c Release -m:8 -v n > build.log 2>&1) || { tail -30 "$P/app/build.log"; fail "P: parallel build failed"; }
CNT=$(grep -c "woven $LWOVEN, already-scoped $LHAND" "$P/app/build.log" || true)
[ "$CNT" -eq 8 ] || { grep -i "weaver" "$P/app/build.log" | head -10; fail "P: expected 8 concurrent lib weaves of 'woven $LWOVEN', saw $CNT"; }
if grep "NumSharp.Weaver:" "$P/app/build.log" | grep -q "StressPar.dll"; then
  fail "P: the weaver ran on the app — it does not install the package"
fi
(cd "$P/app" && dotnet run --no-build -c Release > run.log 2>&1) || { cat "$P/app/run.log"; fail "P: app run failed"; }
grep -q "STRESS-OK" "$P/app/run.log" || { cat "$P/app/run.log"; fail "P: no STRESS-OK from the parallel app"; }
echo "P: 8 libs woven concurrently ($LWOVEN each), app clean; $(grep 'attributed=' "$P/app/run.log")"

echo
echo "stress_weaver: ALL MODES OK  (A source-mode, B nupkg-mode, P parallel x8)"
if [ "${KEEP:-0}" != "1" ]; then rm -rf "$WORK"; else echo "(KEEP=1: work dir kept at $WORK)"; fi
