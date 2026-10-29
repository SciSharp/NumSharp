// NumSharpPythonnetLocalExample
//
// Embeds the Python on this machine in a .NET process and runs six data science libraries over NumSharp
// arrays through NumSharp.Interop.pythonnet. Set it up once, from this folder:
//
//     python -m venv .venv
//     .venv\Scripts\python -m pip install -r requirements.txt        (Linux, macOS: .venv/bin/python)
//     dotnet run
//
// The program prints which Python it embedded and the versions of the libraries. The six examples in
// Libraries/ print nothing: they are written to be read, and the comment beside each line states what
// the line leaves behind. Their asserted twins are the tests of ../NumSharpPythonnetGhActionsCicd.

using NumSharpPythonnetLocalExample;
using Python.Runtime;

try
{
    // The interpreter that NUMSHARP_PYTHONNET_PYTHON names, else .venv beside this project, else the
    // python on PATH. The engine starts inside it and the NumSharp codec is registered. Disposing the
    // host at the end of this block shuts the engine down.
    using PythonHost python = PythonHost.Start();

    PythonInstall install = python.Install;
    Console.WriteLine($"Python      {install.Version}  {install.Executable}{(install.IsVirtualEnvironment ? "  (virtual environment)" : "")}");
    Console.WriteLine($"library     {install.Library}");
    Console.WriteLine($"pythonnet   {typeof(PythonEngine).Assembly.GetName().Version!.ToString(3)}");

    // Importing all six now proves that the environment is complete before any example runs.
    foreach ((string module, string version) in python.Import("numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"))
        Console.WriteLine($"{module,-11} {version}");

    // Each example gets a new Python namespace (with `import numpy as np` done) and gives it back.
    python.Run(NumPyExample.Run);
    python.Run(PandasExample.Run);
    python.Run(SciPyExample.Run);
    python.Run(ScikitLearnExample.Run);
    python.Run(MatplotlibExample.Run);
    python.Run(PyTorchExample.Run);

    Console.WriteLine($"figure      {MatplotlibExample.FigurePath}");
    return 0;
}
catch (PythonSetupException problem)
{
    // The environment is not usable as it is. The message names the cause and the command that
    // corrects it, so it is printed as is, without a stack trace.
    Console.Error.WriteLine(problem.Message);
    return 1;
}
