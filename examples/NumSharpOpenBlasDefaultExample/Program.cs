// NumSharpOpenBlasDefaultExample: NumSharp on OpenBLAS with no configuration.
//
// The project file has one package reference and nothing else. The package brings the OpenBLAS
// library for this platform. NumSharp then computes its matrix products and its linear algebra
// with that library.
//
//     dotnet run -c Release
using System.Globalization;
using NumSharp.Interop.OpenBLAS;
using NumSharpOpenBlasDefaultExample;

// The numbers below print with a decimal point on every machine.
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// The package installs its backend when the program first uses one of its types, not when the
// project references it. So this call comes before every computation. It costs nothing when the
// backend is already installed, and it reports the reason when no library can be loaded.
// Without a statement like this one, the products below run on NumSharp's own kernels and
// np.linalg.cholesky has nothing to call (README.md, "Name the package before you compute").
try
{
    OpenBlasEngine.Enable();
}
catch (Exception problem) when (problem is DllNotFoundException or EntryPointNotFoundException)
{
    Console.Error.WriteLine(problem.Message);
    return 1;
}

Console.WriteLine($"OpenBLAS    {OpenBlasEngine.Info}");
Console.WriteLine($"bundled     {OpenBlasEngine.IsBundledLibrary}");
Console.WriteLine($"LAPACK      {OpenBlasEngine.LapackAvailable}");

Products.Run();

double residual = LinearAlgebra.Run();
Console.WriteLine($"residual    {residual:0.0e+00}   the largest error of the identities in LinearAlgebra.cs");

Speed.Run();
return 0;
