using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     NumPy: the same buffer on both sides. A NumSharp array crosses to NumPy as a view (no bytes move),
///     NumPy computes over it, and NumPy's results come back as NumSharp arrays.
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>NumPyTests.cs</c> in NumSharpPythonnetGhActionsCicd.
///     In this file <c>np</c> is NumSharp and <c>py.np</c> is NumPy.
/// </remarks>
static class NumPyExample
{
    /// <summary>Runs the example.</summary>
    /// <param name="py">
    ///     A Python namespace in which <c>import numpy as np</c> already ran: <c>py.x = array</c> binds a
    ///     NumSharp array as a NumPy view, <c>NDArray a = py.x</c> reads one back, <c>py.Exec</c> runs
    ///     statements and <c>py.Eval</c> reads a value. It is used only inside <c>using (Py.GIL())</c> blocks.
    /// </param>
    /// <exception cref="PythonException">
    ///     A Python statement raised an error, for example because the library is not installed in the
    ///     interpreter that the process embeds.
    /// </exception>
    public static void Run(dynamic py)
    {
        using var scope = NDScope.Open();                                    // every NDArray built below is released when the method returns

        // ---- 1. NumPy views NumSharp's memory, and its results come back ------------------------------
        NDArray samples = np.linspace(0.0, 1.0, 64, endpoint: false);
        NDArray tone = np.sin(2 * Math.PI * 5 * samples) + 0.5 * np.sin(2 * Math.PI * 12 * samples);   // 5 Hz, and 12 Hz at half the amplitude
        using (Py.GIL())                                                     // Python work runs under the GIL; NumSharp work needs none
        {
            py.tone = tone;                                                  // the codec ran tone.ToNumpy(): NumPy views NumSharp's buffer, no bytes moved
            bool numpyOwnsIt = py.tone.flags.owndata;                        // false
            py.Exec("tone[0] = 3.0");                                        // NumPy writes: tone[0] reads 3.0 in C#
            tone[0] = 0.0;                                                   // C# writes: tone[0] reads 0.0 in NumPy
            NDArray spectrum = py.np.abs(py.np.fft.rfft(tone));              // NumPy's FFT of the shared samples: a (33,) float64 array, a view of NumPy's result
            long loudest = np.argmax(spectrum);                              // 5: NumSharp reduces NumPy's result (spectrum[5] == 32, spectrum[12] == 16)
        }

        // ---- 2. a function that returns several arrays reads into one C# tuple ------------------------
        NDArray matrix = np.array(new double[,] { { 2, 0, 1 }, { 0, 3, 0 }, { 1, 0, 2 }, { 0, 1, 1 } });
        using (Py.GIL())
        {
            (NDArray u, NDArray s, NDArray vt) svd = py.np.linalg.svd(matrix, full_matrices: false);   // shapes (4,3), (3,), (3,3); a keyword argument is a C# named argument
            NDArray rebuilt = np.matmul(svd.u * svd.s, svd.vt);              // NumSharp multiplies NumPy's factors: the matrix again (np.allclose)
        }

        // ---- 3. the explicit conversions choose a view or a copy, in each direction -------------------
        NDArray source = np.arange(6).astype(np.float64);
        using (Py.GIL())
        {
            py.view = source.ToNumpy();                                      // a view: it follows NumSharp
            py.copy = source.ToNumpyCopy();                                  // an independent NumPy array
            source[0] = 42.0;                                                // view[0] reads 42.0, copy[0] still reads 0.0
            using PyObject doubled = py.Eval("view * 2.0");                  // a new array that NumPy owns: [84, 2, 4, 6, 8, 10]
            NDArray alias = doubled.AsNDArray();                             // a view of NumPy's buffer: alias[1] = -1.0 changes the NumPy array
            NDArray snapshot = doubled.ToNDArray();                          // an independent NumSharp array
        }
    }
}
