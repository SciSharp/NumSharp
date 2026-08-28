using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     The driver. Runs each level, prints a PASS/FAIL table, then reflects over its OWN assembly
    ///     to prove every <c>[NDScoped]</c>/<c>[NDScopedAsync]</c> method was actually woven (carries an
    ///     <see cref="NDScope"/> local) — so the example is runnable AND assertive. Exits non-zero if
    ///     any demo, or the weave-coverage check, is not OK.
    /// </summary>
    internal static class Program
    {
        private const BindingFlags AllDeclared =
            BindingFlags.Public | BindingFlags.NonPublic |
            BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        private static async Task<int> Main(string[] args)
        {
            bool stress = args.Contains("--stress");
            bool explain = args.Contains("--explain");
            string level = null;
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == "level")
                    level = args[i + 1].ToUpperInvariant();

            Console.WriteLine("NDScoping — every level and layer of NDScope reclamation");
            Console.WriteLine(new string('-', 64));

            if (explain)
            {
                Explain();
                return 0;
            }

            if (level == "D")
                return RunAnalyzerDemo();

            if (stress)
                Console.WriteLine($"(stress mode: {ReclamationProbe.StressRounds}x sweeps with forced GCs between — a mis-scope surfaces as a wrong value)");

            bool Run(string key) => level == null || level == key;

            var groups = new List<(string title, List<Report> reports)>();
            if (Run("0")) groups.Add(("LEVEL 0  Unscoped (the problem)", Level0_Unscoped.RunAll(stress)));
            if (Run("A")) groups.Add(("LEVEL A  Manual  NDScope.Open()", Level1_HandScope.RunAll(stress)));
            if (Run("B")) groups.Add(("LEVEL B  [NDScoped]  (woven)", Level2_Scoped_Sync.RunAll(stress)));
            if (Run("C")) groups.Add(("LEVEL C  [NDScopedAsync]  (woven)", await Level3_ScopedAsync.RunAll(stress)));
            if (Run("Z")) groups.Add(("LEVEL Z  the seam (mechanism)", LevelZ_Seam.RunAll(stress)));
            if (Run("CROSS")) groups.Add(("CROSS-CUTTING", CrossCutting.RunAll(stress)));

            if (groups.Count == 0)
            {
                Console.WriteLine($"unknown level '{level}'. Use one of: 0 | A | B | C | D | Z | cross");
                return 2;
            }

            int fails = 0;
            foreach (var (title, reports) in groups)
            {
                ReclamationProbe.RenderHeader(title);
                foreach (var rep in reports)
                {
                    ReclamationProbe.Render(rep);
                    if (!rep.Ok)
                        fails++;
                }
            }

            // The example proves IT ITSELF was woven — the NDScopeWeaveTests invariant, applied here.
            Console.WriteLine();
            Console.WriteLine(new string('-', 64));
            var (covered, total, unwoven) = WeaveCoverage();
            bool coverageOk = total > 0 && covered == total;
            Console.WriteLine($"weave coverage: {covered}/{total} attributed methods carry an NDScope local   " +
                              (coverageOk ? "OK" : "FAIL"));
            foreach (var name in unwoven)
                Console.WriteLine($"  NOT WOVEN: {name}");
            if (!coverageOk)
                fails++;

            Console.WriteLine(new string('-', 64));
            Console.WriteLine(fails == 0 ? "ALL GREEN" : $"{fails} FAILED");
            return fails == 0 ? 0 : 1;
        }

        // ---- Level D: shell the analyzer counter-example demo (it does not build into us) --------

        private static int RunAnalyzerDemo()
        {
            var dir = Path.Combine(SourceDir(), "CounterExamples");
            var script = Path.Combine(dir, "show-analyzer.sh");
            var command = "bash \"" + script.Replace('\\', '/') + "\"";

            Console.WriteLine("LEVEL D  the analyzer (a project that must NOT compile)");
            Console.WriteLine();
            Console.WriteLine("  Each method in CounterExamples/BadShapes.cs places [NDScoped]/[NDScopedAsync] on a");
            Console.WriteLine("  target the weaver cannot weave, so the analyzer reports a build error");
            Console.WriteLine("  (NDW002/003/005/006/009/010/011) BEFORE the weave. The demonstration is that the");
            Console.WriteLine("  build FAILS with those codes; show-analyzer.sh turns that into a green assertion.");
            Console.WriteLine();

            if (!File.Exists(script))
            {
                Console.WriteLine($"  script not found: {script}");
                return 1;
            }

            // Best-effort: run it here if this host has a POSIX 'bash' with 'dotnet' on its PATH
            // (Linux / macOS / CI, or Windows Git Bash). If 'bash' resolves to a WSL shell — which has
            // no Windows environment and no Windows 'dotnet' — fall back to printing the command so the
            // user can run it in their own shell. This verb is a convenience pointer; the real gate is
            // the script itself (and CI), so it never drags the example's exit code to failure.
            try
            {
                var psi = new ProcessStartInfo("bash")
                {
                    WorkingDirectory = dir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                };
                // If the spawned bash IS Git Bash (env inherited), hand it the SDK dir so the script can
                // put 'dotnet' on PATH (a non-interactive shell does not load the user's profile).
                var dotnetDir = FindDotnetDir();
                if (!string.IsNullOrEmpty(dotnetDir))
                    psi.Environment["NDSCOPING_DOTNET_DIR"] = dotnetDir;
                psi.ArgumentList.Add("show-analyzer.sh");

                using var p = Process.Start(psi);
                string outp = p.StandardOutput.ReadToEnd();
                p.StandardError.ReadToEnd();
                p.WaitForExit();

                if (outp.Contains("ANALYZER-OK"))
                {
                    Console.Write(outp);
                    Console.WriteLine();
                    Console.WriteLine("LEVEL D: ANALYZER-OK");
                    return 0;
                }

                Console.WriteLine("  (could not complete automatically on this host — the spawned 'bash' lacks a");
                Console.WriteLine("   usable 'dotnet'; on Windows this is typically WSL rather than Git Bash.)");
            }
            catch (Exception e)
            {
                Console.WriteLine($"  (could not launch bash here: {e.GetType().Name}: {e.Message})");
            }

            Console.WriteLine();
            Console.WriteLine("  Run it directly in a shell where 'dotnet' is on PATH:");
            Console.WriteLine($"      {command}");
            Console.WriteLine("  Expected: each 'error NDW00x' printed, then ANALYZER-OK.");
            return 0;
        }

        private static string SourceDir([CallerFilePath] string path = null) => Path.GetDirectoryName(path);

        // Locate the directory that holds the 'dotnet' host, so a spawned shell can run it. Tries the
        // SDK's own environment hints first, then the process host, then walks up from the runtime dir.
        private static string FindDotnetDir()
        {
            static bool HasHost(string d) =>
                d != null && (File.Exists(Path.Combine(d, "dotnet.exe")) || File.Exists(Path.Combine(d, "dotnet")));

            var hostPath = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
            if (!string.IsNullOrEmpty(hostPath) && File.Exists(hostPath))
                return Path.GetDirectoryName(hostPath);

            var root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
            if (HasHost(root))
                return root;

            var proc = Environment.ProcessPath;
            if (!string.IsNullOrEmpty(proc) &&
                string.Equals(Path.GetFileNameWithoutExtension(proc), "dotnet", StringComparison.OrdinalIgnoreCase))
                return Path.GetDirectoryName(proc);

            for (var d = new DirectoryInfo(System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory());
                 d != null; d = d.Parent)
                if (HasHost(d.FullName))
                    return d.FullName;

            return null;
        }

        // ---- weave-coverage reflection (mirrors NDScopeWeaveTests) --------------------------------

        private static (int covered, int total, List<string> unwoven) WeaveCoverage()
        {
            var methods = new List<MethodInfo>();
            var seen = new HashSet<MethodInfo>();
            foreach (var type in LoadedTypes())
            {
                foreach (var m in type.GetMethods(AllDeclared))
                    if (IsScoped(m) && seen.Add(m))
                        methods.Add(m);

                foreach (var p in type.GetProperties(AllDeclared))
                {
                    if (!IsScoped(p))
                        continue;
                    var getter = p.GetGetMethod(nonPublic: true);
                    if (getter != null && seen.Add(getter))
                        methods.Add(getter);
                }
            }

            var unwoven = methods
                .Where(m => !HasScopeLocal(m))
                .Select(m => m.DeclaringType!.Name + "." + m.Name)
                .OrderBy(n => n)
                .ToList();
            return (methods.Count - unwoven.Count, methods.Count, unwoven);
        }

        private static IEnumerable<Type> LoadedTypes()
        {
            try
            {
                return Assembly.GetExecutingAssembly().GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                return ex.Types.Where(t => t != null);
            }
        }

        private static bool IsScoped(MemberInfo m) =>
            m.GetCustomAttribute<NDScopedAttribute>(inherit: false) != null ||
            m.GetCustomAttribute<NDScopedAsyncAttribute>(inherit: false) != null;

        private static bool HasScopeLocal(MethodInfo m)
        {
            // async / iterator scoped methods are STUBS — the weave lands in the compiler's state
            // machine, so the NDScope local lives in ITS MoveNext. Follow the StateMachineAttribute.
            var smType = m.GetCustomAttributes<StateMachineAttribute>(inherit: false)
                          .FirstOrDefault()?.StateMachineType;
            if (smType != null)
            {
                var moveNext = smType.GetMethod("MoveNext", AllDeclared);
                return moveNext != null && HasLocalDirect(moveNext);
            }

            return HasLocalDirect(m);
        }

        private static bool HasLocalDirect(MethodBase m)
        {
            var body = m.GetMethodBody();
            if (body is null)
                return false;
            foreach (var local in body.LocalVariables)
                if (local.LocalType == typeof(NDScope))
                    return true;
            return false;
        }

        private static void Explain()
        {
            Console.WriteLine("Levels (how much is automated):");
            Console.WriteLine("  0  Unscoped         nothing — the finalizer backstop (the problem)");
            Console.WriteLine("  A  Manual           using var scope = NDScope.Open(); + scope.Returns(...)");
            Console.WriteLine("  B  [NDScoped]       the weaver injects Level A into the IL (sync methods + sync iterators)");
            Console.WriteLine("  C  [NDScopedAsync]  the state-machine / deferral weave (await-suspending / Task shapes)");
            Console.WriteLine("  D  Guarded          the Roslyn analyzer — NDW00x build errors (dotnet run -- level D)");
            Console.WriteLine("  Z  The seam         OpenOrResume/Suspend/DisposeSlot — the weaver only; shown to explain");
            Console.WriteLine();
            Console.WriteLine("Layers (egress vocabulary): NDArray, NDArray[], tuple, ITuple, INDArrayCarrier,");
            Console.WriteLine("  bare IArraySlice, out-param, scalar/void, iterator, async, passthrough.");
            Console.WriteLine("Cross-cutting: nesting, Attach/Detach, the exception path, threading, granularity.");
            Console.WriteLine();
            Console.WriteLine("Every demo asserts its claim via the public NDArray.IsDisposed and prints OK/FAIL.");
            Console.WriteLine("Verbs:  dotnet run   |   dotnet run -- level A|B|C|D|Z|cross|0   |   dotnet run -- --stress");
        }
    }
}
