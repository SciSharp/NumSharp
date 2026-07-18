using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     One embedded CPython engine per test process (CPython + numpy cannot re-initialize after
    ///     <c>Py_Finalize</c>, so <c>[AssemblyInitialize]</c>/<c>[AssemblyCleanup]</c> own the lifecycle).
    ///     Python is discovered from <c>PYTHONNET_PYDLL</c> or by probing common interpreters for one
    ///     that imports numpy; when none is found every interop test reports Inconclusive instead of
    ///     failing, so the suite is safe on machines and CI images without Python.
    /// </summary>
    [TestClass]
    public sealed class PythonSession
    {
        public static bool Available { get; private set; }
        public static string Reason { get; private set; } = "not initialized";

        [AssemblyInitialize]
        public static void Start(TestContext _)
        {
            try
            {
                string dll = DiscoverPythonDll(out string reason);
                if (dll is null)
                {
                    Reason = reason;
                    return;
                }

                Runtime.PythonDLL = dll;
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
                using (Py.GIL())
                {
                    using var np = Py.Import("numpy");   // fail fast if the found Python lacks numpy
                }

                Available = true;
                Reason = null;
            }
            catch (Exception e)
            {
                Available = false;
                Reason = $"engine failed to start: {e.GetType().Name}: {e.Message}";
            }
        }

        /// <summary>
        ///     Full engine shutdown at the end of the run. This is itself part of the proof: the
        ///     interop's shutdown handler must drain every outstanding lease crash-free, and the test
        ///     host must exit cleanly afterwards (a post-shutdown finalizer crash would fail the run).
        ///
        ///     <para>When <see cref="ShutdownLeakTests"/> ran, this is also where its assertions live —
        ///     the only in-process observation point AFTER a real engine death. pythonnet's Shutdown
        ///     runs no Python atexit pass (probed), so the <c>weakref.finalize</c> callbacks of
        ///     still-referenced exports never fire here; the interop's own deferred sweep must release
        ///     those pins once the engine has fully finished dying, and the import force-drain must
        ///     make later <c>Dispose</c> calls harmless no-ops.</para>
        /// </summary>
        [AssemblyCleanup]
        public static void Stop()
        {
            if (!Available)
                return;

            try
            {
                // pythonnet 3.0.x stashes runtime state with BinaryFormatter, which .NET 8+ removed;
                // NoopFormatter is pythonnet's own opt-out.
                RuntimeData.FormatterType = typeof(NoopFormatter);
                PythonEngine.Shutdown();
            }
            catch (Exception e)
            {
                Trace.WriteLine($"PythonEngine.Shutdown reported: {e.Message}");
            }

            AssertShutdownLifetimeContract();
        }

        /// <summary>The post-shutdown half of <see cref="ShutdownLeakTests"/>.</summary>
        private static void AssertShutdownLifetimeContract()
        {
            // --- exports: the deferred sweep must release Python-held pins once the engine is gone.
            if (ShutdownLeakTests.OrphanExportSlice is not null)
            {
                bool swept = PollUntil(() => PythonConvert.LiveExports == 0 &&
                                             ShutdownLeakTests.OrphanExportSlice.IsReleased, 10_000);
                if (!swept)
                    throw new AssertFailedException(
                        $"exports leaked at engine shutdown: LiveExports={PythonConvert.LiveExports}, " +
                        $"orphan buffer released={ShutdownLeakTests.OrphanExportSlice.IsReleased}. " +
                        "pythonnet's Shutdown runs no atexit pass, so without the interop's own sweep " +
                        "these pins are permanent.");
            }

            // --- imports: the shutdown handler force-drains synchronously, and a later Dispose of the
            //     still-referenced NDArray must be a harmless no-op (the lease was already claimed).
            if (ShutdownLeakTests.OrphanImportView is not null)
            {
                if (!PollUntil(() => PythonConvert.LiveImports == 0, 5_000))
                    throw new AssertFailedException(
                        $"import leases survived engine shutdown: LiveImports={PythonConvert.LiveImports}.");

                ShutdownLeakTests.OrphanImportView.Dispose();   // must not throw, must not double-release
                if (!ShutdownLeakTests.OrphanImportSlice.IsReleased)
                    throw new AssertFailedException(
                        "disposing the last NDArray over a force-drained lease must free the NumSharp block.");
            }
        }

        private static bool PollUntil(Func<bool> condition, int timeoutMs)
        {
            var sw = Stopwatch.StartNew();
            while (!condition())
            {
                if (sw.ElapsedMilliseconds > timeoutMs)
                    return condition();
                System.Threading.Thread.Sleep(25);
            }

            return true;
        }

        public static void EnsureOrInconclusive()
        {
            if (!Available)
                Assert.Inconclusive($"Python with numpy is unavailable on this machine: {Reason}");
        }

        // ---- discovery -------------------------------------------------------------------------

        private static string DiscoverPythonDll(out string reason)
        {
            var attempts = new List<string>();

            string env = Environment.GetEnvironmentVariable("PYTHONNET_PYDLL");
            if (!string.IsNullOrEmpty(env))
            {
                if (File.Exists(env) || !Path.IsPathRooted(env))
                {
                    reason = null;
                    return env;
                }

                attempts.Add($"PYTHONNET_PYDLL='{env}' does not exist");
            }

            foreach (string exe in CandidateInterpreters())
            {
                if (!Probe(exe, out string dll, out string detail))
                {
                    attempts.Add($"{exe}: {detail}");
                    continue;
                }

                reason = null;
                return dll;
            }

            reason = attempts.Count == 0 ? "no python interpreter candidates" : string.Join(" | ", attempts);
            return null;
        }

        private static IEnumerable<string> CandidateInterpreters()
        {
            yield return "python";
            yield return "python3";

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string exeName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "python.exe" : "python3";

            string claudePython = Path.Combine(home, ".claude", "python", exeName);
            if (File.Exists(claudePython))
                yield return claudePython;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                string programs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python");
                if (Directory.Exists(programs))
                    foreach (string dir in Directory.GetDirectories(programs, "Python3*"))
                    {
                        string exe = Path.Combine(dir, "python.exe");
                        if (File.Exists(exe))
                            yield return exe;
                    }
            }
        }

        /// <summary>
        ///     Runs the candidate interpreter once to (a) prove numpy imports, (b) locate the shared
        ///     library pythonnet must load, (c) reject versions outside pythonnet 3.0.5's support.
        /// </summary>
        private static bool Probe(string exe, out string dll, out string detail)
        {
            dll = null;
            const string script =
                "import sys, sysconfig, numpy;" +
                "print(sys.base_prefix);" +
                "print(sys.version_info.major);" +
                "print(sys.version_info.minor);" +
                "print(sysconfig.get_config_var('INSTSONAME') or '');" +
                "print(sysconfig.get_config_var('LIBDIR') or '')";
            try
            {
                var psi = new ProcessStartInfo(exe, $"-c \"{script}\"")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var proc = Process.Start(psi);
                string stdout = proc.StandardOutput.ReadToEnd();
                string stderr = proc.StandardError.ReadToEnd();
                if (!proc.WaitForExit(10_000))
                {
                    try { proc.Kill(); } catch { }
                    detail = "probe timed out";
                    return false;
                }

                if (proc.ExitCode != 0)
                {
                    detail = $"probe failed ({FirstLine(stderr)})";
                    return false;
                }

                string[] lines = stdout.Replace("\r", "").Split('\n');
                string prefix = lines[0].Trim();
                int major = int.Parse(lines[1].Trim());
                int minor = int.Parse(lines[2].Trim());

                if (major != 3 || minor < 7 || minor > 13)
                {
                    detail = $"python {major}.{minor} outside pythonnet 3.0.5's supported range (3.7-3.13)";
                    return false;
                }

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    dll = Path.Combine(prefix, $"python{major}{minor}.dll");
                }
                else
                {
                    string soname = lines.Length > 3 ? lines[3].Trim() : "";
                    string libdir = lines.Length > 4 ? lines[4].Trim() : "";
                    dll = soname.Length > 0 && libdir.Length > 0 ? Path.Combine(libdir, soname) : null;
                }

                if (dll is null || !File.Exists(dll))
                {
                    detail = $"shared library not found at '{dll}'";
                    return false;
                }

                detail = null;
                return true;
            }
            catch (Exception e)
            {
                detail = e.Message;
                return false;
            }
        }

        private static string FirstLine(string s)
        {
            if (string.IsNullOrEmpty(s)) return "no stderr";
            int i = s.IndexOfAny(new[] { '\r', '\n' });
            return i < 0 ? s : s.Substring(0, i);
        }
    }
}
