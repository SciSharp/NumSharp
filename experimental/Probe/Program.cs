using System;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

static class Probe
{
    static int Main(string[] args)
    {
        Runtime.PythonDLL = @"C:\Users\ELI\.claude\python\python312.dll";
        PythonEngine.Initialize();
        PythonEngine.BeginAllowThreads();

        // Register the codec at startup (the documented rule to avoid decoder-cache poisoning).
        NDArrayPythonInterop.RegisterCodec();

        int rc = 0;
        try
        {
            rc = Run(args);
        }
        catch (Exception e)
        {
            Console.WriteLine($"[FATAL] {e.GetType().Name}: {e.Message}");
            Console.WriteLine(e.StackTrace);
            rc = 99;
        }
        finally
        {
            RuntimeData.FormatterType = typeof(NoopFormatter);
            try { PythonEngine.Shutdown(); } catch (Exception e) { Console.WriteLine($"[shutdown] {e.Message}"); }
        }
        return rc;
    }

    static int Run(string[] args)
    {
        // Smoke round-trip: NumSharp -> numpy (view) -> mutate in Python -> read back in C#.
        var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
        using (Py.GIL())
        {
            using var scope = Py.CreateScope();
            scope.Exec("import numpy as np");
            using var pv = nd.ToNumpy();          // zero-copy view
            scope.Set("x", pv);
            scope.Exec("x[0,0] = 111.0");         // mutate through the shared buffer
        }
        Console.WriteLine($"after python mutate nd[0,0] = {nd.GetDouble(0, 0)} (expect 111)");

        // Import round-trip: numpy -> NumSharp view -> mutate in C# -> read back in Python.
        using (Py.GIL())
        {
            using var scope = Py.CreateScope();
            scope.Exec("import numpy as np\ny = np.arange(4, dtype='f8')");
            using var y = scope.Eval("y");
            using var view = NDArrayPythonInterop.ToNDArrayView(y);
            view.SetDouble(999.0, 2);
            using var back = scope.Eval("float(y[2])");
            Console.WriteLine($"after csharp mutate y[2] = {back.As<double>()} (expect 999)");
        }

        Console.WriteLine($"LiveExports={NDArrayPythonInterop.LiveExports} LiveImports={NDArrayPythonInterop.LiveImports}");
        Console.WriteLine("SMOKE OK");
        return 0;
    }
}
