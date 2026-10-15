using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     Owns the embedded CPython of this process: finds a Python on the machine, starts the engine inside
///     it (a virtual environment included), registers NumSharp's codec, and shuts the engine down on dispose.
/// </summary>
/// <remarks>
///     <para>Start it once and keep it for the life of the process. CPython and NumPy cannot be initialized
///     a second time after a shutdown, so a second <see cref="Start"/> in one process is not supported.</para>
///     <para>NumSharp.Interop.pythonnet never starts an engine itself. It attaches to the one the application
///     started, which is why this class exists in the example and not in the package.</para>
/// </remarks>
sealed class PythonHost : IDisposable
{
    /// <summary>
    ///     The environment variable that names the interpreter to embed, for example
    ///     <c>C:\envs\science\Scripts\python.exe</c>. The value is binding: when it is set, no other
    ///     Python is tried, so a typing error cannot silently select the system Python.
    /// </summary>
    public const string InterpreterVariable = "NUMSHARP_PYTHONNET_PYTHON";

    /// <summary>The Python this process embeds: its interpreter, version, prefixes and shared library.</summary>
    public PythonInstall Install { get; }

    /// <summary>True after <see cref="Dispose"/> ran, so a second call does not shut the engine down twice.</summary>
    private bool _stopped;

    /// <summary>Wraps an engine that <see cref="Start"/> already initialized.</summary>
    /// <param name="install">The Python the running engine was started in.</param>
    private PythonHost(PythonInstall install) => Install = install;

    /// <summary>
    ///     Finds a Python (see <see cref="Locate"/> for the order), starts the embedded engine inside it and
    ///     registers NumSharp's codec, so that an <c>NDArray</c> crosses every pythonnet boundary without an
    ///     explicit conversion call.
    /// </summary>
    /// <returns>The host that owns the running engine. Dispose it to shut the engine down.</returns>
    /// <exception cref="PythonSetupException">
    ///     No usable Python was found, the named interpreter cannot be used, or the engine started in another
    ///     environment than the one that was selected. The message says what to do.
    /// </exception>
    public static PythonHost Start()
    {
        PythonInstall install = Locate();

        // pythonnet loads this file. A virtual environment has no Python library of its own, so the
        // library always belongs to the base install the environment was created from.
        Runtime.PythonDLL = install.Library;

        // Tell CPython which interpreter this process embeds. CPython computes sys.prefix, sys.path and
        // sys.executable from that path, exactly as the interpreter does when it runs: beside the
        // interpreter of a virtual environment it finds pyvenv.cfg and adopts the environment.
        // Without the name, the library alone starts an engine in the BASE install: the packages of the
        // environment cannot be imported. On Windows sys.executable is then this .NET program, which a
        // library that starts worker processes would launch again. On Linux the engine takes its paths
        // from the first `python3` on PATH, which crashes the process when that is another Python.
        PythonEngine.ProgramName = install.Executable;

        try
        {
            PythonEngine.Initialize();
        }
        catch (Exception e) when (e is TypeInitializationException or DllNotFoundException or BadImageFormatException or MissingMethodException)
        {
            // pythonnet binds every C-API function when its runtime type loads. A library that is missing,
            // that has another bitness than this process, or that pythonnet does not support fails there.
            throw new PythonSetupException(
                $"pythonnet could not load '{install.Library}' (Python {install.Version}): {(e.InnerException ?? e).Message}", e);
        }

        // Initialize leaves the calling thread holding the GIL. Release it, so that every thread,
        // this one included, takes it for the duration of a `using (Py.GIL())` block.
        PythonEngine.BeginAllowThreads();

        string embedded;
        using (Py.GIL())
        {
            using PyObject sys = Py.Import("sys");
            using PyObject prefix = sys.GetAttr("prefix");
            embedded = prefix.As<string>();
        }

        // The adoption above fails without an error: the engine then runs in the base install and the
        // first import of a library that only the environment has fails later, far from the cause.
        // (The check is outside the GIL block: nothing may release a GIL after the engine shut down.)
        if (install.IsVirtualEnvironment && !PythonInstall.SamePath(embedded, install.Prefix))
        {
            Shutdown();
            throw new PythonSetupException(
                $"The engine started in '{embedded}', not in the virtual environment '{install.Prefix}', " +
                "so the packages of that environment cannot be imported.");
        }

        // One call turns on every conversion: NDArray <-> NumPy, PEP 3118 buffers, the PyTorch and
        // pandas adapters, and C# tuples <-> Python tuples. It must run before the first decode, because
        // pythonnet caches a decoder lookup that found nothing.
        NDArrayPythonInterop.RegisterCodec();
        return new PythonHost(install);
    }

    /// <summary>
    ///     Imports each module in the embedded interpreter and reads its <c>__version__</c>. Call it once at
    ///     startup: it proves that the environment holds every library the application needs, and the first
    ///     import of a large library (PyTorch) is the slow one.
    /// </summary>
    /// <param name="modules">Import names, for example <c>sklearn</c> (not the package name scikit-learn).</param>
    /// <returns>One pair for each module, in the order given: the import name and its version.</returns>
    /// <exception cref="PythonSetupException">
    ///     A module cannot be imported. The message names the interpreter in use and the commands that
    ///     install the pinned set of libraries (into that interpreter when it is a virtual environment,
    ///     else into a new environment beside the project).
    /// </exception>
    public IReadOnlyList<(string Module, string Version)> Import(params string[] modules)
    {
        var versions = new List<(string, string)>(modules.Length);
        using (Py.GIL())
        {
            foreach (string module in modules)
            {
                try
                {
                    // Py.Import returns the cached module on later imports; the wrapper is ours to dispose.
                    using PyObject imported = Py.Import(module);
                    using PyObject version = imported.GetAttr("__version__");
                    // ToString() is Python's str(). As<string>() accepts an exact str only, and
                    // torch.__version__ is an instance of a str subclass (TorchVersion).
                    versions.Add((module, version.ToString()!));
                }
                catch (PythonException e)
                {
                    // A virtual environment is the place to install into. Any other Python (the system
                    // one, for example, which pip refuses to change on current Linux distributions) is
                    // better left alone: point the user to the environment of the example instead.
                    string remedy = Install.IsVirtualEnvironment
                        ? $"Install the libraries of this example into it:{Environment.NewLine}    \"{Install.Executable}\" -m pip install -r requirements.txt"
                        : $"It is not a virtual environment. Create the one of this example, then run again:{Environment.NewLine}{SetupCommands}";
                    throw new PythonSetupException(
                        $"Python cannot import '{module}' ({e.Message}). The interpreter in use is {Install.Executable}{Environment.NewLine}{remedy}", e);
                }
            }
        }

        return versions;
    }

    /// <summary>
    ///     Runs one example against a new Python namespace in which <c>import numpy as np</c> already ran, and
    ///     drops the namespace afterwards. Every Python name the example bound dies with it, and with those
    ///     names the NumPy views that kept NumSharp buffers pinned.
    /// </summary>
    /// <param name="example">
    ///     The example. It receives the namespace as <c>dynamic</c> and must do its Python work inside
    ///     <c>using (Py.GIL())</c> blocks: this method does not hold the GIL while the example runs.
    /// </param>
    /// <exception cref="PythonException">
    ///     A Python statement of the example raised an error. The namespace is dropped before the exception
    ///     leaves this method, and the engine stays usable.
    /// </exception>
    public void Run(Action<dynamic> example)
    {
        PyModule scope;
        using (Py.GIL())
        {
            scope = Py.CreateScope();
            scope.Exec("import numpy as np");
        }

        try
        {
            example(scope);
        }
        finally
        {
            // A PyObject is disposed under the GIL, like every other use of it.
            using (Py.GIL())
                scope.Dispose();
        }
    }

    /// <summary>
    ///     Shuts the engine down. After this call no Python object may be used, and no <c>Py.GIL()</c> block
    ///     may be opened. NumSharp arrays that viewed Python memory must not be read afterwards (disposing them
    ///     stays safe); arrays that NumSharp owns are not affected.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    ///     pythonnet refuses to shut down while a Python error is pending on the calling thread.
    /// </exception>
    public void Dispose()
    {
        if (_stopped)
            return;

        _stopped = true;
        Shutdown();
    }

    /// <summary>
    ///     Stops the engine: without the crash that pythonnet's default shutdown has on .NET 8 and later, and
    ///     without the seconds it otherwise spends on garbage collection.
    /// </summary>
    /// <remarks>
    ///     Call it only at the end of the process: the objects that <c>gc.freeze()</c> sets aside are never
    ///     collected again.
    /// </remarks>
    private static void Shutdown()
    {
        // pythonnet 3.0.5 runs 87 full collections of Python's heap during its shutdown. With six large
        // libraries imported, one collection takes about 0.1 second and finds nothing: almost every object
        // belongs to a module and stays alive. gc.freeze() moves the objects that exist now out of the
        // collector's reach, so those collections return at once (measured here: 7.3 s without it, 0.2 s with it).
        using (Py.GIL())
            PythonEngine.RunSimpleString("import gc\ngc.collect()\ngc.freeze()");

        // pythonnet saves its state with BinaryFormatter during shutdown, and .NET 8 and later do not
        // support that class any more. NoopFormatter is pythonnet's own switch to skip the save.
        RuntimeData.FormatterType = typeof(NoopFormatter);
        PythonEngine.Shutdown();
    }

    // ---- discovery ------------------------------------------------------------------------------------

    /// <summary>
    ///     Selects the Python to embed. The first match wins:
    ///     <list type="number">
    ///       <item>the interpreter that <see cref="InterpreterVariable"/> names (binding);</item>
    ///       <item>a <c>.venv</c> folder in the current directory or beside the project file, which is what the
    ///         two commands in README.md create;</item>
    ///       <item><c>python</c> or <c>python3</c> on <c>PATH</c> (an activated environment or the system Python),
    ///         then the <c>py</c> launcher on Windows.</item>
    ///     </list>
    /// </summary>
    /// <returns>The selected install, already checked for a version that pythonnet supports and for a loadable library.</returns>
    /// <exception cref="PythonSetupException">Nothing usable was found. The message lists what was tried and the setup commands.</exception>
    private static PythonInstall Locate()
    {
        string? named = Environment.GetEnvironmentVariable(InterpreterVariable);
        if (!string.IsNullOrWhiteSpace(named))
        {
            // Binding: a named interpreter that cannot be used is an error. To continue with another
            // Python would run the example against packages nobody selected.
            return PythonInstall.TryProbe(named, out PythonInstall? chosen, out string why)
                ? chosen!
                : throw new PythonSetupException($"{InterpreterVariable}='{named}' cannot be used: {why}");
        }

        string relative = OperatingSystem.IsWindows() ? Path.Combine("Scripts", "python.exe") : Path.Combine("bin", "python");
        foreach (string folder in ProjectFolders())
        {
            string interpreter = Path.Combine(folder, ".venv", relative);
            if (!File.Exists(interpreter))
                continue;

            // The same rule as for a named interpreter: this environment was made for the example, so a
            // problem with it must be reported, not hidden behind the next candidate.
            return PythonInstall.TryProbe(interpreter, out PythonInstall? local, out string why)
                ? local!
                : throw new PythonSetupException($"The virtual environment '{Path.GetDirectoryName(Path.GetDirectoryName(interpreter))}' cannot be used: {why}");
        }

        var tried = new List<string>();
        string[] commands = OperatingSystem.IsWindows() ? ["python", "python3", "py"] : ["python3", "python"];
        foreach (string command in commands)
        {
            if (PythonInstall.TryProbe(command, out PythonInstall? found, out string why))
                return found!;
            tried.Add($"  {command}: {why}");
        }

        throw new PythonSetupException(
            "No usable Python was found. Create the environment of this example (Python 3.11 to 3.13):" + Environment.NewLine +
            SetupCommands + Environment.NewLine +
            $"or set {InterpreterVariable} to the path of an interpreter that has the libraries." + Environment.NewLine +
            "Tried:" + Environment.NewLine + string.Join(Environment.NewLine, tried));
    }

    /// <summary>
    ///     The two commands of README.md that create the environment of this example beside the project,
    ///     spelled for the operating system the program runs on. Error messages quote them.
    /// </summary>
    private static string SetupCommands => OperatingSystem.IsWindows()
        ? "    python -m venv .venv" + Environment.NewLine + @"    .venv\Scripts\python -m pip install -r requirements.txt"
        : "    python3 -m venv .venv" + Environment.NewLine + "    .venv/bin/python -m pip install -r requirements.txt";

    /// <summary>
    ///     The folders in which a <c>.venv</c> counts as "the environment of this example": the current directory
    ///     (<c>dotnet run</c> from the project folder), then the folders from the build output up to the one
    ///     that holds the project file (<c>dotnet run --project ...</c> from elsewhere, or a start from an IDE).
    /// </summary>
    /// <returns>Candidate folders, nearest first. The walk stops at the project folder or after four levels.</returns>
    private static IEnumerable<string> ProjectFolders()
    {
        yield return Directory.GetCurrentDirectory();

        // bin/<configuration>/<framework>/ is three levels below the project folder. The limit keeps a
        // published copy of the application from walking to the root of the drive.
        var folder = new DirectoryInfo(AppContext.BaseDirectory);
        for (int level = 0; folder is not null && level <= 4; level++, folder = folder.Parent)
        {
            yield return folder.FullName;
            if (folder.EnumerateFiles("*.csproj").Any())
                yield break;
        }
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
sealed record PythonInstall(string Executable, Version Version, string Prefix, string BasePrefix, string Library)
{
    /// <summary>
    ///     True when the interpreter runs inside a virtual environment (venv, virtualenv, uv). The engine must
    ///     then be told the interpreter's path, or it starts in the base install without the environment's packages.
    /// </summary>
    public bool IsVirtualEnvironment => !SamePath(Prefix, BasePrefix);

    /// <summary>
    ///     The program the interpreter runs to describe itself. It prints one JSON object. JSON escapes every
    ///     character outside ASCII, so a path with such characters survives whatever code page the console uses.
    /// </summary>
    private const string Describe =
        "import json, sys, sysconfig\n" +
        "print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable,\n" +
        "                  'prefix': sys.prefix, 'base_prefix': sys.base_prefix,\n" +
        "                  'instsoname': sysconfig.get_config_var('INSTSONAME') or '',\n" +
        "                  'libdir': sysconfig.get_config_var('LIBDIR') or ''}))";

    /// <summary>
    ///     Runs an interpreter once and reads what it reports. A failure is an answer, not an exception:
    ///     discovery tries several candidates and must be able to continue with the next one.
    /// </summary>
    /// <param name="command">A path to an interpreter, or a command name that is resolved on <c>PATH</c>.</param>
    /// <param name="install">The install when the interpreter can be embedded; otherwise <c>null</c>.</param>
    /// <param name="reason">Why the interpreter cannot be embedded; empty when it can.</param>
    /// <returns>True when pythonnet can embed this interpreter.</returns>
    public static bool TryProbe(string command, out PythonInstall? install, out string reason)
    {
        install = null;
        string output, errors;
        int exitCode;
        try
        {
            var start = new ProcessStartInfo(command)
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
        catch (Win32Exception)
        {
            reason = "not found";      // no such file, or no such command on PATH
            return false;
        }

        // The last line is the report: a site customization can print lines of its own before it.
        string? report = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
        if (exitCode != 0 || report is null)
        {
            string firstError = errors.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "no output";
            reason = $"it did not run (exit code {exitCode}: {firstError})";
            return false;
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
            reason = $"it is not a CPython interpreter (unexpected output: {report})";
            return false;
        }

        // Each pythonnet release drives a fixed range of Python versions and fails obscurely outside it,
        // so compare before the library is loaded.
        Version oldest = PythonEngine.MinSupportedVersion, newest = PythonEngine.MaxSupportedVersion;
        var minor = new Version(version.Major, version.Minor);
        if (minor < new Version(oldest.Major, oldest.Minor) || minor > new Version(newest.Major, newest.Minor))
        {
            string pythonnet = typeof(PythonEngine).Assembly.GetName().Version!.ToString(3);
            reason = $"it is Python {version}, and pythonnet {pythonnet} drives Python {oldest.Major}.{oldest.Minor} to {newest.Major}.{newest.Minor} " +
                     "(a newer Python needs a newer pythonnet: add a PackageReference to it)";
            return false;
        }

        if (executable.Length == 0)
        {
            reason = "it does not report its own path (sys.executable is empty)";
            return false;
        }

        string? library = FindLibrary(basePrefix, soName, libDir, version);
        if (library is null)
        {
            reason = $"its shared library was not found under '{basePrefix}'. A Python that was built without one cannot be embedded " +
                     "(on Debian and Ubuntu install libpython3-dev; with pyenv build with --enable-shared)";
            return false;
        }

        install = new PythonInstall(executable, version, prefix, basePrefix, library);
        reason = "";
        return true;
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
            // A macOS framework build (python.org, GitHub's setup-python) reports INSTSONAME as
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

/// <summary>
///     The Python environment is not usable as it is. The message is written for the person at the keyboard:
///     it says what is wrong and which command corrects it, so the program prints it without a stack trace.
/// </summary>
sealed class PythonSetupException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong and what to do about it.</param>
    /// <param name="inner">The exception from pythonnet or Python that revealed the problem, when there is one.</param>
    public PythonSetupException(string message, Exception? inner = null) : base(message, inner) { }
}
