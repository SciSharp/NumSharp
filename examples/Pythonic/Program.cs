// Pythonic
//
// Python programs that use a data science library, and their C# ports. A port has the statements of its
// Python program: the Python.* tree of this project (Python.cs) gives it the functions and the classes of
// the library, its NumPy lines are NumSharp lines, and it prints what the Python program prints. Set it up
// once, from this folder:
//
//     python -m venv .venv
//     .venv\Scripts\python -m pip install -r requirements.txt        (Linux, macOS: .venv/bin/python)
//
//     dotnet run                          every port, each under the name of its Python program
//     dotnet run -- classification        one port, with the output of its Python program and no more
//     python verify.py                    runs each pair and compares the two outputs

using System.Globalization;
using NumSharp.Interop.PythonNet;
using Pythonic;

// Python prints a decimal point on every machine. Make C# do the same.
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// One entry for each Python program of this folder: its name on the command line, its file, and the port.
(string Name, string Reference, Action Port)[] examples =
[
    ("classification", "ScikitLearn/classification.py", Classification.Run),
];

string? only = args.Length > 0 ? args[0] : null;
if (only is not null && !examples.Any(example => example.Name == only))
{
    Console.Error.WriteLine($"There is no port named '{only}'. The ports are: {string.Join(", ", examples.Select(example => example.Name))}.");
    return 1;
}

try
{
    foreach ((string name, string reference, Action port) in examples)
    {
        if (only is null)
            Console.WriteLine($"# {reference}");     // a heading, so that several outputs can be told apart
        else if (name != only)
            continue;

        port();                                      // the engine starts with the first Python call of the first port
    }

    // The lifetime check of the pythonic approach. Each port closed its PythonScope, so no NumPy array
    // views NumSharp memory any more, and no NumSharp array holds memory that Python allocated.
    // The wait is for the second half: when a scope closes, NumSharp hands the Python memory back at once,
    // and Python frees it on a thread-pool thread a moment later.
    SpinWait.SpinUntil(
        () => NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports == 0,
        TimeSpan.FromSeconds(5));
    if (NDArrayPythonInterop.LiveExports != 0 || NDArrayPythonInterop.LiveImports != 0)
    {
        Console.Error.WriteLine($"A port left memory shared with Python: {NDArrayPythonInterop.LiveExports} NumSharp buffers, {NDArrayPythonInterop.LiveImports} Python buffers.");
        return 2;
    }

    return 0;
}
catch (PythonInteropException problem)
{
    // The Python side is not usable as it is. The message names the cause and the command that corrects
    // it, so it is printed as is, without a stack trace.
    Console.Error.WriteLine(problem.Message);
    return 1;
}
finally
{
    PythonRuntime.Shutdown();                        // the end of the process: the engine cannot start again
}
