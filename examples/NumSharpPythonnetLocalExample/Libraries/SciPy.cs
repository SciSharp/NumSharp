using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     SciPy: algorithms NumSharp does not have (filter design, image filters, optimizers, statistics tests)
///     run over NumSharp arrays, and their results return as arrays, tuples of arrays and plain C# numbers.
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>SciPyTests.cs</c> in NumSharpPythonnetGhActionsCicd.
/// </remarks>
static class SciPyExample
{
    /// <summary>Runs the example.</summary>
    /// <param name="py">A Python namespace in which <c>import numpy as np</c> already ran. It is used only inside <c>using (Py.GIL())</c> blocks.</param>
    /// <exception cref="PythonException">
    ///     A Python statement raised an error, for example because the library is not installed in the
    ///     interpreter that the process embeds.
    /// </exception>
    public static void Run(dynamic py)
    {
        using var scope = NDScope.Open();                                    // every NDArray built below is released when the method returns
        using (Py.GIL())
            py.Exec("from scipy import signal, ndimage, stats, optimize");

        // ---- 1. scipy.signal designs a filter and runs it over NumSharp's samples ---------------------
        NDArray time = np.linspace(0.0, 1.0, 200, endpoint: false);          // one second at 200 samples per second
        NDArray clean = np.sin(2 * Math.PI * 3 * time);                      // a 3 Hz signal
        NDArray noisy = clean + 0.4 * np.sin(2 * Math.PI * 40 * time);       // with 40 Hz interference
        using (Py.GIL())
        {
            (NDArray b, NDArray a) lowPass = py.signal.butter(4, 10, fs: 200);          // order 4, cut-off 10 Hz: two arrays of 5 coefficients, as one C# tuple
            NDArray filtered = py.signal.filtfilt(lowPass.b, lowPass.a, noisy);         // (200,): SciPy filtered NumSharp's samples
            double before = Math.Sqrt(np.mean(np.square(noisy - clean)).GetDouble());   // 0.283: the RMS error with the interference
            double after = Math.Sqrt(np.mean(np.square(filtered - clean)).GetDouble()); // 0.036: NumSharp measures what the filter left
        }

        // ---- 2. scipy.ndimage writes its result INTO a NumSharp array ---------------------------------
        NDArray impulse = np.zeros(new Shape(9, 9));
        impulse[4, 4] = 1.0;                                                 // one bright pixel
        NDArray blurred = np.zeros(new Shape(9, 9));
        using (Py.GIL())
            py.ndimage.gaussian_filter(impulse, sigma: 1.0, output: blurred);   // output= is NumSharp's buffer: no result array is allocated, nothing is copied back
        double peak = blurred.GetDouble(4, 4);                               // 0.159; np.sum(blurred) == 1.0: a Gaussian blur keeps the total brightness

        // ---- 3. result objects: their fields read as C# numbers, booleans and arrays ------------------
        NDArray x = np.linspace(0.0, 1.0, 20);
        NDArray y = 2.5 * x + 1.0;
        NDArray start = np.array(new[] { -1.2, 1.0 });
        using (Py.GIL())
        {
            dynamic line = py.stats.linregress(x, y);
            double slope = line.slope, intercept = line.intercept;           // 2.5, 1.0
            dynamic minimum = py.optimize.minimize(py.optimize.rosen, start, method: "BFGS");   // a Python function is an argument like any other
            bool converged = minimum.success;                                // true
            NDArray at = minimum.x;                                          // [1, 1] within 1e-4: the minimum of the Rosenbrock function
        }
    }
}
