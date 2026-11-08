using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using Python.Runtime;

namespace Pythonic;

/// <summary>
///     What one interpreter reports about itself: enough to load its shared library and to tell a virtual
///     environment from a base install. <see cref="Locate"/> selects the interpreter that
///     <see cref="PythonRuntime"/> embeds.
/// </summary>
/// <param name="Executable">The interpreter's own path (<c>sys.executable</c>). For a virtual environment this is the environment's interpreter, which is the path CPython adopts the environment through.</param>
/// <param name="Version">The Python version, for example 3.12.12.</param>
/// <param name="Prefix"><c>sys.prefix</c>: the root of the virtual environment, or of the install when there is no environment.</param>
/// <param name="BasePrefix"><c>sys.base_prefix</c>: the root of the install that owns the shared library.</param>
/// <param name="Library">The full path of the shared library pythonnet loads (<c>python312.dll</c>, <c>libpython3.12.so.1.0</c>, or the macOS framework binary).</param>
sealed record PythonInstall(string Executable, Version Version, string Prefix, string BasePrefix, string Library)
{
    /// <summary>
    ///     The environment variable that names the interpreter to embed, for example
    ///     <c>C:\envs\science\Scripts\python.exe</c>. The value is binding: when it is set, no other
    ///     Python is tried, so a typing error cannot silently select the system Python.
    /// </summary>
    public const string InterpreterVariable = "NUMSHARP_PYTHONNET_PYTHON";

    /// <summary>
    ///     True when the interpreter runs inside a virtual environment (venv, virtualenv, uv). The engine must
    ///     then be told the interpreter's path, or it starts in the base install without the environment's packages.
    /// </summary>
    public bool IsVirtualEnvironment => !SamePath(Prefix, BasePrefix);

    /// <summary>
    ///     The two commands of README.md that create the environment of this project beside the project
    ///     file, spelled for the operating system the program runs on. Error messages quote them.
    /// </summary>
    public static string SetupCommands => OperatingSystem.IsWindows()
        ? "    python -m venv .venv" + Environment.NewLine + @"    .venv\Scripts\python -m pip install -r requirements.txt"
        : "    python3 -m venv .venv" + Environment.NewLine + "    .venv/bin/python -m pip install -r requirements.txt";

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
    /// <exception cref="PythonInteropException">Nothing usable was found. The message lists what was tried and the setup commands.</exception>
    public static PythonInstall Locate()
    {
        string? named = Environment.GetEnvironmentVariable(InterpreterVariable);
        if (!string.IsNullOrWhiteSpace(named))
        {
            // Binding: a named interpreter that cannot be used is an error. To continue with another
            // Python would run the programs against packages nobody selected.
            return TryProbe(named, out PythonInstall? chosen, out string why)
                ? chosen!
                : throw new PythonInteropException($"{InterpreterVariable}='{named}' cannot be used: {why}");
        }

        string relative = OperatingSystem.IsWindows() ? Path.Combine("Scripts", "python.exe") : Path.Combine("bin", "python");
        foreach (string folder in ProjectFolders())
        {
            string interpreter = Path.Combine(folder, ".venv", relative);
            if (!File.Exists(interpreter))
                continue;

            // The same rule as for a named interpreter: this environment was made for the project, so a
            // problem with it must be reported, not hidden behind the next candidate.
            return TryProbe(interpreter, out PythonInstall? local, out string why)
                ? local!
                : throw new PythonInteropException($"The virtual environment '{Path.GetDirectoryName(Path.GetDirectoryName(interpreter))}' cannot be used: {why}");
        }

        var tried = new List<string>();
        string[] commands = OperatingSystem.IsWindows() ? ["python", "python3", "py"] : ["python3", "python"];
        foreach (string command in commands)
        {
            if (TryProbe(command, out PythonInstall? found, out string why))
                return found!;
            tried.Add($"  {command}: {why}");
        }

        throw new PythonInteropException(
            "No usable Python was found. Create the environment of this project (Python 3.11 to 3.13):" + Environment.NewLine +
            SetupCommands + Environment.NewLine +
            $"or set {InterpreterVariable} to the path of an interpreter that has the libraries." + Environment.NewLine +
            "Tried:" + Environment.NewLine + string.Join(Environment.NewLine, tried));
    }

    /// <summary>
    ///     The folders in which a <c>.venv</c> counts as "the environment of this project": the current directory
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
