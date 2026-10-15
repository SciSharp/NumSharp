// NumSharpOpenBlasAdvancedExample: a pinned OpenBLAS build, and results that reproduce bit for bit.
//
// Three things decide the last bits of a float matrix product: the OpenBLAS build, the number of
// threads, and the set of kernels that OpenBLAS selects for the processor. The project file pins
// the first one (OpenBlasVersion). Backend.cs pins the other two when it loads the library.
//
//     dotnet run -c Release
using System.Globalization;
using NumSharpOpenBlasAdvancedExample;

// The numbers below print with a decimal point on every machine.
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// The package loads OpenBLAS by itself when the program first uses one of its types. This variable
// turns that load off, so that the first load is the call in Backend.Start, with its arguments.
// OpenBLAS selects its kernels one time, while it loads, so a later call cannot change them.
//
// The statement works here because this file names no type of the package, and so the runtime has
// not started the package yet. Do not add such a name to this file. In a service, set the variable
// in the environment of the process instead: that does not depend on the order of the code.
Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", "0");

if (!Backend.Start())
    return 1;

Backend.Report();
Reproducibility.Run();
return 0;
