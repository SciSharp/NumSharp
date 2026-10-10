using System.Collections.Concurrent;
using Python.Runtime;

namespace Pythonic;

/// <summary>
///     The backend under the <see cref="Python"/> tree. It starts CPython one time for the process, it is
///     the one place that takes the GIL (<see cref="Run(Action)"/>), and it keeps each imported module
///     (<see cref="Import"/>). It knows no library: the tree adds those.
/// </summary>
/// <remarks>
///     <para><b>The GIL.</b> Every use of a <see cref="PyObject"/>, its <c>Dispose</c> included, must run
///     inside a <see cref="Run(Action)"/> body. Give a <c>Run</c> body .NET values to return (a number, an
///     <c>NDArray</c>), never a <see cref="PyObject"/>: outside the body nothing holds the GIL for it.</para>
///     <para><b>One engine for each process.</b> CPython and NumPy cannot be initialized a second time
///     after a shutdown. <see cref="Initialize"/> starts the engine when it is first needed, and
///     <see cref="Shutdown"/> is for the end of the process only.</para>
///     <para>NumSharp.Interop.pythonnet never starts an engine. It attaches to the one that this class
///     started, and its conversion calls take the GIL again inside a <c>Run</c> body without harm, because
///     the GIL is re-entrant.</para>
/// </remarks>
static class PythonRuntime
{
    /// <summary>The engine was not started yet.</summary>
    private const int Idle = 0;

    /// <summary>The engine runs and <see cref="Run(Action)"/> can be called.</summary>
    private const int Running = 1;

    /// <summary><see cref="Shutdown"/> ran. The engine cannot start again in this process.</summary>
    private const int Stopped = 2;

    /// <summary>Serializes the start and the shutdown of the engine.</summary>
    private static readonly object Gate = new();

    /// <summary>The modules that <see cref="Import"/> returned, by import name. They live until <see cref="Shutdown"/>.</summary>
    private static readonly ConcurrentDictionary<string, PyObject> Modules = new(StringComparer.Ordinal);

    /// <summary>One of <see cref="Idle"/>, <see cref="Running"/> and <see cref="Stopped"/>.</summary>
    private static int _state;

    /// <summary>True when this class started the engine, and so is the one that must shut it down.</summary>
    private static bool _ownsEngine;

    /// <summary>
    ///     The Python that this process embeds, or <c>null</c> before the first <see cref="Initialize"/> and
    ///     when other code of the process started the engine.
    /// </summary>
    public static PythonInstall? Install { get; private set; }

    /// <summary>
    ///     Makes sure that an embedded CPython runs. The first call selects an interpreter
    ///     (<see cref="PythonInstall.Locate"/>) and starts the engine inside it, a virtual environment
    ///     included. Later calls return at once. An engine that other code of the process already started
    ///     is used as it is.
    /// </summary>
    /// <exception cref="PythonInteropException">
    ///     No usable Python was found, pythonnet cannot load its library, or the engine started in another
    ///     environment than the selected one. The message says what to do.
    /// </exception>
    /// <exception cref="InvalidOperationException"><see cref="Shutdown"/> already ran in this process.</exception>
    public static void Initialize()
    {
        if (Volatile.Read(ref _state) == Running)
            return;

        lock (Gate)
        {
            if (_state == Running)
                return;
            if (_state == Stopped)
                throw new InvalidOperationException("The Python engine was shut down. CPython cannot start a second time in one process.");

            if (!PythonEngine.IsInitialized)
            {
                Start();
                _ownsEngine = true;
            }

            Volatile.Write(ref _state, Running);
        }
    }

    /// <summary>Runs an action under the GIL. The engine starts first when it does not run yet.</summary>
    /// <param name="body">The Python work. Dispose every <see cref="PyObject"/> it creates before it returns.</param>
    /// <exception cref="PythonInteropException">The engine cannot start (see <see cref="Initialize"/>), or a module that the body imports is missing.</exception>
    /// <exception cref="PythonException">A Python call of the body raised an error. The GIL is released before the exception leaves this method.</exception>
    /// <exception cref="InvalidOperationException"><see cref="Shutdown"/> already ran in this process.</exception>
    public static void Run(Action body)
    {
        Initialize();
        using (Py.GIL())
            body();
    }

    /// <summary>Runs a function under the GIL and returns its result. The engine starts first when it does not run yet.</summary>
    /// <typeparam name="T">A .NET value that needs no GIL afterwards: a number, a string, an <c>NDArray</c>, or a tuple of those. Never a <see cref="PyObject"/>.</typeparam>
    /// <param name="body">The Python work. Dispose every <see cref="PyObject"/> it creates before it returns.</param>
    /// <returns>What <paramref name="body"/> returned.</returns>
    /// <exception cref="PythonInteropException">The engine cannot start (see <see cref="Initialize"/>), or a module that the body imports is missing.</exception>
    /// <exception cref="PythonException">A Python call of the body raised an error. The GIL is released before the exception leaves this method.</exception>
    /// <exception cref="InvalidOperationException"><see cref="Shutdown"/> already ran in this process.</exception>
    public static T Run<T>(Func<T> body)
    {
        Initialize();
        using (Py.GIL())
            return body();
    }

    /// <summary>
    ///     Imports a module one time and returns the same handle on every later call. The handle belongs to
    ///     the process: use it as it is and <b>do not dispose it</b>. Only what you get from it
    ///     (<c>GetAttr</c>, <c>Invoke</c>) is yours to dispose.
    /// </summary>
    /// <param name="moduleName">The import name with its dots, for example <c>sklearn.model_selection</c>.</param>
    /// <returns>The module. Call this method, and use the handle, inside a <see cref="Run(Action)"/> body.</returns>
    /// <exception cref="PythonInteropException">
    ///     Python cannot import the module. The message names the interpreter in use and the command that
    ///     installs the libraries of this project.
    /// </exception>
    public static PyObject Import(string moduleName)
    {
        if (Modules.TryGetValue(moduleName, out PyObject? known))
            return known;

        PyObject module;
        try
        {
            module = Py.Import(moduleName);
        }
        catch (PythonException e)
        {
            // A virtual environment is the place to install into. Any other Python (the system one, for
            // example, which pip refuses to change on current Linux distributions) is better left alone.
            string remedy = Install is { IsVirtualEnvironment: true }
                ? $"Install the libraries of this project into it:{Environment.NewLine}    \"{Install.Executable}\" -m pip install -r requirements.txt"
                : $"Create the environment of this project, then run again:{Environment.NewLine}{PythonInstall.SetupCommands}";
            throw new PythonInteropException(
                $"Python cannot import '{moduleName}' ({e.Message}). The interpreter in use is {Install?.Executable ?? "the one the process started"}.{Environment.NewLine}{remedy}", e);
        }

        // Two threads can import the same module for the first time. The GIL lets one of them reach this
        // line first; the other one drops its handle and takes the cached one.
        if (Modules.TryAdd(moduleName, module))
            return module;

        module.Dispose();
        return Modules[moduleName];
    }

    /// <summary>
    ///     Shuts the engine down at the end of the process. After this call no Python object may be used and
    ///     no <see cref="Run(Action)"/> call may be made. NumSharp arrays that viewed Python memory must not
    ///     be read afterwards (disposing them stays safe); arrays that NumSharp owns are not affected.
    ///     The call does nothing when the engine never started, when other code started it, or when it was
    ///     shut down already.
    /// </summary>
    /// <exception cref="InvalidOperationException">pythonnet refuses to shut down while a Python error is pending on the calling thread.</exception>
    public static void Shutdown()
    {
        lock (Gate)
        {
            if (_state != Running)
                return;

            Volatile.Write(ref _state, Stopped);
            if (!_ownsEngine)
                return;

            using (Py.GIL())
            {
                // The cached modules are the only Python references this class owns.
                foreach (PyObject module in Modules.Values)
                    module.Dispose();
                Modules.Clear();

                // pythonnet 3.0.5 runs 87 full collections of Python's heap during its shutdown. With
                // large libraries imported, one collection takes about 0.1 second and finds nothing.
                // gc.freeze() moves the objects that exist now out of the collector's reach, so those
                // collections return at once. The frozen objects are never collected again, which is
                // right only at the end of the process.
                PythonEngine.RunSimpleString("import gc\ngc.collect()\ngc.freeze()");
            }

            // pythonnet saves its state with BinaryFormatter during shutdown, and .NET 8 and later do not
            // support that class any more. NoopFormatter is pythonnet's own switch to skip the save.
            RuntimeData.FormatterType = typeof(NoopFormatter);
            PythonEngine.Shutdown();
        }
    }

    /// <summary>Starts the engine inside the interpreter that <see cref="PythonInstall.Locate"/> selects. The caller holds <see cref="Gate"/>.</summary>
    /// <exception cref="PythonInteropException">No usable Python, a library that pythonnet cannot load, or an engine that started outside the selected virtual environment.</exception>
    private static void Start()
    {
        PythonInstall install = PythonInstall.Locate();

        // pythonnet loads this file. A virtual environment has no Python library of its own, so the
        // library always belongs to the base install the environment was created from.
        Runtime.PythonDLL = install.Library;

        // Tell CPython which interpreter this process embeds. CPython computes sys.prefix, sys.path and
        // sys.executable from that path, exactly as the interpreter does when it runs: beside the
        // interpreter of a virtual environment it finds pyvenv.cfg and adopts the environment.
        // Without the name, the library alone starts an engine in the BASE install: the packages of the
        // environment cannot be imported. On Windows sys.executable is then this .NET program. On Linux
        // the engine takes its paths from the first `python3` on PATH, which crashes the process when
        // that is another Python.
        PythonEngine.ProgramName = install.Executable;

        try
        {
            PythonEngine.Initialize();
        }
        catch (Exception e) when (e is TypeInitializationException or DllNotFoundException or BadImageFormatException or MissingMethodException)
        {
            // pythonnet binds every C-API function when its runtime type loads. A library that is missing,
            // that has another bitness than this process, or that pythonnet does not support fails there.
            throw new PythonInteropException(
                $"pythonnet could not load '{install.Library}' (Python {install.Version}): {(e.InnerException ?? e).Message}", e);
        }

        // Initialize leaves the calling thread holding the GIL. Release it, so that every thread,
        // this one included, takes it for the duration of a Run body.
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
            RuntimeData.FormatterType = typeof(NoopFormatter);
            PythonEngine.Shutdown();
            Volatile.Write(ref _state, Stopped);
            throw new PythonInteropException(
                $"The engine started in '{embedded}', not in the virtual environment '{install.Prefix}', " +
                "so the packages of that environment cannot be imported.");
        }

        Install = install;
    }
}

/// <summary>
///     The Python side cannot be used as it is: no interpreter, a library that does not load, or a module
///     that is not installed. The message is written for the person at the keyboard. It says what is wrong
///     and which command corrects it, so a program can print it without a stack trace.
/// </summary>
/// <remarks>
///     It is the base type for the errors of a bridge. A bridge for one library derives its own exception
///     from it when a caller must tell that library's failures apart.
/// </remarks>
class PythonInteropException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong and what to do about it.</param>
    /// <param name="inner">The exception from pythonnet or Python that revealed the problem, when there is one.</param>
    public PythonInteropException(string message, Exception? inner = null) : base(message, inner) { }
}
