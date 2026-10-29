using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

// One embedded interpreter serves the whole test run, and it has one global lock (the GIL). MSTest's
// project template turns parallel test methods ON. With them, every test would wait for the GIL of its
// neighbours and the leak check of PythonTest would count their conversions. Run the tests one by one.
[assembly: DoNotParallelize]

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     Starts the embedded CPython once for the test run and shuts it down at the end. The interpreter is
///     the one the pipeline installed and named in <see cref="InterpreterVariable"/>; nothing is searched for.
/// </summary>
/// <remarks>
///     <para>The engine lives for the whole assembly, not for one test class: CPython and NumPy cannot be
///     initialized a second time in a process.</para>
///     <para>On a build server a missing interpreter fails the run. A test suite that skips itself reports
///     green and proves nothing, and that failure mode stays invisible until somebody reads the log.</para>
/// </remarks>
[TestClass]
public sealed class PythonEnvironment
{
    /// <summary>
    ///     The environment variable that names the interpreter to embed. The workflow sets it from the
    ///     <c>python-path</c> output of <c>actions/setup-python</c>. The value is binding: no other Python is tried.
    /// </summary>
    public const string InterpreterVariable = "NUMSHARP_PYTHONNET_PYTHON";

    /// <summary>The import names of the six libraries the tests use. All of them must import, or the run fails at once.</summary>
    private static readonly string[] Libraries = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"];

    /// <summary>
    ///     Why no engine runs, or <c>null</c> when one does. Set only on a developer machine where
    ///     <see cref="InterpreterVariable"/> is not set; each test then reports Inconclusive with this text.
    /// </summary>
    public static string? Skipped { get; private set; }

    /// <summary>The Python the engine runs in; <c>null</c> when <see cref="Skipped"/> is set.</summary>
    public static PythonInstall? Install { get; private set; }

    /// <summary>
    ///     Takes the interpreter that the pipeline named, starts the engine inside it (a virtual environment
    ///     included), imports the six libraries, registers NumSharp's codec and writes the job summary.
    /// </summary>
    /// <param name="context">The MSTest context of the run. Not used: the configuration comes from the environment.</param>
    /// <exception cref="InvalidOperationException">
    ///     The variable is not set on a build server, the interpreter cannot be embedded, the engine started
    ///     in another environment, or a library cannot be imported. MSTest then fails every test with this message.
    /// </exception>
    [AssemblyInitialize]
    public static void Start(TestContext context)
    {
        string? interpreter = Environment.GetEnvironmentVariable(InterpreterVariable);
        if (string.IsNullOrWhiteSpace(interpreter))
        {
            // GitHub Actions (and most build servers) set CI=true. There, an unset variable is a defect of
            // the pipeline and must be red. On a developer machine it only means "not configured here".
            if (string.Equals(Environment.GetEnvironmentVariable("CI"), "true", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    $"{InterpreterVariable} is not set, so no Python would be tested. Pass the interpreter of the job to the test step: " +
                    $"{InterpreterVariable}: ${{{{ steps.python.outputs.python-path }}}}");

            Skipped = $"Set {InterpreterVariable} to a Python interpreter that has the libraries of requirements.txt, then run the tests again.";
            return;
        }

        PythonInstall install = PythonInstall.Probe(interpreter);

        // pythonnet loads this file. A virtual environment has no Python library of its own, so the
        // library always belongs to the base install the environment was created from.
        Runtime.PythonDLL = install.Library;

        // Tell CPython which interpreter this process embeds. CPython computes sys.prefix, sys.path and
        // sys.executable from that path, exactly as the interpreter does when it runs: beside the
        // interpreter of a virtual environment it finds pyvenv.cfg and adopts the environment.
        // Without the name, the library alone starts an engine in the BASE install: the packages of the
        // environment cannot be imported. On Windows sys.executable is then the test host, which a
        // library that starts worker processes would launch again. On Linux the engine takes its paths
        // from the first `python3` on PATH, which crashes the process when that is another Python.
        PythonEngine.ProgramName = install.Executable;

        PythonEngine.Initialize();

        // Initialize leaves this thread holding the GIL, and MSTest runs the tests on other threads.
        // Release it, so that each test takes it for the duration of a `using (Py.GIL())` block.
        PythonEngine.BeginAllowThreads();

        var versions = new List<(string Module, string Version)>();
        string embedded;
        using (Py.GIL())
        {
            using PyObject sys = Py.Import("sys");
            using PyObject prefix = sys.GetAttr("prefix");
            embedded = prefix.As<string>();

            // The adoption of a virtual environment fails without an error. Check it here, or the first
            // import of a library that only the environment has fails later with a misleading message.
            if (install.IsVirtualEnvironment && !PythonInstall.SamePath(embedded, install.Prefix))
                throw new InvalidOperationException(
                    $"The engine started in '{embedded}', not in the virtual environment '{install.Prefix}'.");

            // Import everything now: a library that is absent from the environment fails the run here,
            // with its name, and not in the middle of the first test that needs it.
            foreach (string module in Libraries)
            {
                try
                {
                    using PyObject imported = Py.Import(module);
                    using PyObject version = imported.GetAttr("__version__");
                    // ToString() is Python's str(). As<string>() accepts an exact str only, and
                    // torch.__version__ is an instance of a str subclass.
                    versions.Add((module, version.ToString()!));
                }
                catch (PythonException e)
                {
                    throw new InvalidOperationException(
                        $"'{install.Executable}' cannot import '{module}' ({e.Message}). " +
                        "Does the workflow install requirements.txt into the interpreter it passes to the tests?", e);
                }
            }
        }

        // One call turns on every conversion: NDArray <-> NumPy, PEP 3118 buffers, the PyTorch and pandas
        // adapters, and C# tuples <-> Python tuples. It must run before the first decode, because
        // pythonnet caches a decoder lookup that found nothing.
        NDArrayPythonInterop.RegisterCodec();

        Install = install;
        WriteJobSummary(install, versions);
    }

    /// <summary>
    ///     Shuts the engine down after the last test. No Python object may be used afterwards.
    /// </summary>
    /// <remarks>
    ///     The shutdown takes some seconds: pythonnet 3.0.5 runs 87 full collections of Python's heap, and
    ///     with six large libraries imported each one takes about 0.1 second. A test run does not need
    ///     those seconds back. NumSharpPythonnetLocalExample shows how an interactive program avoids them.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    ///     pythonnet refuses to shut down while a Python error is pending on the calling thread.
    /// </exception>
    [AssemblyCleanup]
    public static void Stop()
    {
        if (Install is null)
            return;

        // pythonnet saves its state with BinaryFormatter during shutdown, and .NET 8 and later do not
        // support that class any more. NoopFormatter is pythonnet's own switch to skip the save.
        RuntimeData.FormatterType = typeof(NoopFormatter);
        PythonEngine.Shutdown();
    }

    /// <summary>
    ///     Appends a table of what this job embedded to the file GitHub Actions shows on the run's summary
    ///     page (<c>GITHUB_STEP_SUMMARY</c>). Outside GitHub Actions the variable is not set and nothing is written.
    /// </summary>
    /// <param name="install">The Python the engine runs in.</param>
    /// <param name="versions">The imported libraries and their versions.</param>
    private static void WriteJobSummary(PythonInstall install, IReadOnlyList<(string Module, string Version)> versions)
    {
        string? summaryFile = Environment.GetEnvironmentVariable("GITHUB_STEP_SUMMARY");
        if (string.IsNullOrEmpty(summaryFile))
            return;

        var markdown = new StringBuilder();
        markdown.AppendLine($"### NumSharp with Python on {RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture})");
        markdown.AppendLine();
        markdown.AppendLine("| Component | Version |");
        markdown.AppendLine("|---|---|");
        markdown.AppendLine($"| .NET | {Environment.Version} |");
        markdown.AppendLine($"| pythonnet | {typeof(PythonEngine).Assembly.GetName().Version!.ToString(3)} |");
        markdown.AppendLine($"| Python | {install.Version}{(install.IsVirtualEnvironment ? " (virtual environment)" : "")} |");
        foreach ((string module, string version) in versions)
            markdown.AppendLine($"| {module} | {version} |");
        markdown.AppendLine();

        // Append: every step of the job writes to the same file.
        File.AppendAllText(summaryFile, markdown.ToString());
    }
}

/// <summary>
///     What one interpreter reports about itself: enough to load its shared library and to tell a virtual
///     environment from a base install.
/// </summary>
/// <param name="Executable">The interpreter's own path (<c>sys.executable</c>). For a virtual environment this is the environment's interpreter, which is the path CPython adopts the environment through.</param>
/// <param name="Version">The Python version, for example 3.12.12.</param>
/// <param name="Prefix"><c>sys.prefix</c>: the root of the virtual environment, or of the install when there is no environment.</param>
/// <param name="BasePrefix"><c>sys.base_prefix</c>: the root of the install that owns the shared library.</param>
/// <param name="Library">The full path of the shared library pythonnet loads (<c>python312.dll</c>, <c>libpython3.12.so.1.0</c>, or the macOS framework binary).</param>
public sealed record PythonInstall(string Executable, Version Version, string Prefix, string BasePrefix, string Library)
{
    /// <summary>
    ///     True when the interpreter runs inside a virtual environment (venv, virtualenv, uv). The engine must
    ///     then be told the interpreter's path, or it starts in the base install without the environment's packages.
    /// </summary>
    public bool IsVirtualEnvironment => !SamePath(Prefix, BasePrefix);

    /// <summary>
    ///     The program the interpreter runs to describe itself. It prints one JSON object. JSON escapes every
    ///     character outside ASCII, so a path with such characters survives whatever code page the runner uses.
    /// </summary>
    private const string Describe =
        "import json, sys, sysconfig\n" +
        "print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable,\n" +
        "                  'prefix': sys.prefix, 'base_prefix': sys.base_prefix,\n" +
        "                  'instsoname': sysconfig.get_config_var('INSTSONAME') or '',\n" +
        "                  'libdir': sysconfig.get_config_var('LIBDIR') or ''}))";

    /// <summary>Runs the named interpreter once and reads what it reports.</summary>
    /// <param name="interpreter">The path of the interpreter the pipeline installed.</param>
    /// <returns>The install: its version is one pythonnet supports and its shared library exists.</returns>
    /// <exception cref="InvalidOperationException">
    ///     The interpreter does not run, is not CPython, has a version outside the range pythonnet drives, or
    ///     has no shared library. The message names the interpreter and the cause.
    /// </exception>
    public static PythonInstall Probe(string interpreter)
    {
        string output, errors;
        int exitCode;
        try
        {
            var start = new ProcessStartInfo(interpreter)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            // ArgumentList quotes each argument for the platform, so the program text needs no escaping.
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add(Describe);

            using Process process = Process.Start(start)!;
            // Read the error stream concurrently: a child that fills one pipe while the parent waits on
            // the other one blocks both.
            Task<string> pendingErrors = process.StandardError.ReadToEndAsync();
            output = process.StandardOutput.ReadToEnd();
            errors = pendingErrors.GetAwaiter().GetResult();
            process.WaitForExit();
            exitCode = process.ExitCode;
        }
        catch (Win32Exception e)
        {
            throw new InvalidOperationException($"{PythonEnvironment.InterpreterVariable}='{interpreter}' cannot be started: {e.Message}", e);
        }

        // The last line is the report: a site customization can print lines of its own before it.
        string? report = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
        if (exitCode != 0 || report is null)
        {
            string firstError = errors.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "no output";
            throw new InvalidOperationException($"'{interpreter}' did not run (exit code {exitCode}: {firstError}).");
        }

        string executable, prefix, basePrefix, soName, libDir;
        Version version;
        try
        {
            using JsonDocument json = JsonDocument.Parse(report);
            JsonElement root = json.RootElement;
            JsonElement v = root.GetProperty("version");
            version = new Version(v[0].GetInt32(), v[1].GetInt32(), v[2].GetInt32());
            executable = root.GetProperty("executable").GetString() ?? "";
            prefix = root.GetProperty("prefix").GetString() ?? "";
            basePrefix = root.GetProperty("base_prefix").GetString() ?? "";
            soName = root.GetProperty("instsoname").GetString() ?? "";
            libDir = root.GetProperty("libdir").GetString() ?? "";
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            // Something ran and printed, but not the report: the path names a program that is not CPython.
            throw new InvalidOperationException($"'{interpreter}' is not a CPython interpreter (unexpected output: {report}).", e);
        }

        // Each pythonnet release drives a fixed range of Python versions and fails obscurely outside it,
        // so compare before the library is loaded.
        Version oldest = PythonEngine.MinSupportedVersion, newest = PythonEngine.MaxSupportedVersion;
        var minor = new Version(version.Major, version.Minor);
        if (minor < new Version(oldest.Major, oldest.Minor) || minor > new Version(newest.Major, newest.Minor))
        {
            string pythonnet = typeof(PythonEngine).Assembly.GetName().Version!.ToString(3);
            throw new InvalidOperationException(
                $"'{interpreter}' is Python {version}, and pythonnet {pythonnet} drives Python {oldest.Major}.{oldest.Minor} to {newest.Major}.{newest.Minor}. " +
                "Change python-version in the workflow, or reference a newer pythonnet in the project file.");
        }

        string library = FindLibrary(basePrefix, soName, libDir, version)
            ?? throw new InvalidOperationException(
                $"The shared library of '{interpreter}' was not found under '{basePrefix}'. A Python that was built without one cannot be embedded.");

        return new PythonInstall(executable, version, prefix, basePrefix, library);
    }

    /// <summary>Finds the shared library of an install from what its interpreter reports.</summary>
    /// <param name="basePrefix"><c>sys.base_prefix</c>. The library belongs to the base install, never to a virtual environment.</param>
    /// <param name="soName">The build variable <c>INSTSONAME</c>; empty on Windows.</param>
    /// <param name="libDir">The build variable <c>LIBDIR</c>; empty on Windows.</param>
    /// <param name="version">The Python version, which is part of the file name.</param>
    /// <returns>The path of the first candidate that exists, or <c>null</c>.</returns>
    private static string? FindLibrary(string basePrefix, string soName, string libDir, Version version)
    {
        if (OperatingSystem.IsWindows())
        {
            string dll = Path.Combine(basePrefix, $"python{version.Major}{version.Minor}.dll");
            return File.Exists(dll) ? dll : null;
        }

        string extension = OperatingSystem.IsMacOS() ? "dylib" : "so";
        string name = $"libpython{version.Major}.{version.Minor}.{extension}";
        string[] candidates =
        [
            // A shared build on Linux or macOS: LIBDIR/libpython3.12.so.1.0
            soName.Length > 0 && libDir.Length > 0 ? Path.Combine(libDir, soName) : "",
            // The macOS framework build that actions/setup-python installs reports INSTSONAME as
            // "Python.framework/Versions/3.12/Python", which is not under LIBDIR. The binary is in the prefix.
            soName.Length > 0 ? Path.Combine(basePrefix, Path.GetFileName(soName)) : "",
            libDir.Length > 0 ? Path.Combine(libDir, name) : "",
            Path.Combine(basePrefix, "lib", name),
        ];
        return candidates.FirstOrDefault(candidate => candidate.Length > 0 && File.Exists(candidate));
    }

    /// <summary>Compares two folder paths the way the file system of this platform does.</summary>
    /// <param name="a">The first path.</param>
    /// <param name="b">The second path.</param>
    /// <returns>True when both name the same folder, without regard to a trailing separator and, off Linux, to letter case.</returns>
    /// <remarks>The comparison is on the text only. Both paths come from one interpreter, so a difference is a real one, not an alias.</remarks>
    public static bool SamePath(string a, string b)
    {
        static string Normalize(string path) =>
            Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        StringComparison comparison = OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
        return string.Equals(Normalize(a), Normalize(b), comparison);
    }
}
