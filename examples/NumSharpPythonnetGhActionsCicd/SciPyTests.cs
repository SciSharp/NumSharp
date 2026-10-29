using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     SciPy: algorithms NumSharp does not have (filter design, image filters, optimizers, statistics tests)
///     run over NumSharp arrays, and their results return as arrays, tuples of arrays and plain C# numbers.
/// </summary>
[TestClass]
public class SciPyTests : PythonTest
{
    /// <summary>
    ///     <c>scipy.signal</c> designs a low-pass filter and applies it to NumSharp's samples. The filter
    ///     coefficients arrive as a C# tuple of two arrays; NumSharp measures how much noise is gone.
    /// </summary>
    [TestMethod]
    public void SignalFilterRunsOverNumSharpSamples()
    {
        using var scope = NDScope.Open();
        NDArray time = np.linspace(0.0, 1.0, 200, endpoint: false);          // one second at 200 samples per second
        NDArray clean = np.sin(2 * Math.PI * 3 * time);                      // a 3 Hz signal
        NDArray noisy = clean + 0.4 * np.sin(2 * Math.PI * 40 * time);       // with 40 Hz interference
        using (Py.GIL())
        {
            py.Exec("from scipy import signal");
            (NDArray b, NDArray a) lowPass = py.signal.butter(4, 10, fs: 200);    // order 4, cut-off 10 Hz
            Assert.AreEqual(5L, lowPass.b.size, "an order 4 filter has 5 numerator coefficients");
            Assert.AreEqual(5L, lowPass.a.size);

            NDArray filtered = py.signal.filtfilt(lowPass.b, lowPass.a, noisy);
            Assert.AreEqual(200L, filtered.size);

            double before = Math.Sqrt(np.mean(np.square(noisy - clean)).GetDouble());
            double after = Math.Sqrt(np.mean(np.square(filtered - clean)).GetDouble());
            Assert.AreEqual(0.4 / Math.Sqrt(2), before, 1e-9, "0.283: the RMS of a sine of amplitude 0.4");
            Assert.AreEqual(0.0364, after, 1e-3, "the filter removes most of the interference");
        }
    }

    /// <summary>
    ///     <c>scipy.ndimage</c> writes its result into a NumSharp array (<c>output=</c>): no result array is
    ///     allocated on either side, and nothing is copied back.
    /// </summary>
    [TestMethod]
    public void ImageFilterWritesIntoANumSharpArray()
    {
        using var scope = NDScope.Open();
        NDArray impulse = np.zeros(new Shape(9, 9));
        impulse[4, 4] = 1.0;                                                 // one bright pixel
        NDArray blurred = np.zeros(new Shape(9, 9));
        using (Py.GIL())
        {
            py.Exec("from scipy import ndimage");
            py.ndimage.gaussian_filter(impulse, sigma: 1.0, output: blurred);
        }

        Assert.AreEqual(1.0, np.sum(blurred).GetDouble(), 1e-12, "a Gaussian blur keeps the total brightness");
        Assert.AreEqual(0.15915589174187972, blurred.GetDouble(4, 4), 1e-12, "the peak of the kernel");
        Assert.AreEqual(blurred.GetDouble(3, 4), blurred.GetDouble(5, 4), "the kernel is symmetric");
        Assert.AreEqual(blurred.GetDouble(4, 3), blurred.GetDouble(4, 5));
    }

    /// <summary>
    ///     <c>scipy.stats</c> and <c>scipy.optimize</c> return result objects. Their fields read as C#
    ///     numbers, booleans and arrays.
    /// </summary>
    [TestMethod]
    public void StatisticsAndOptimizersReturnNumbersAndArrays()
    {
        using var scope = NDScope.Open();
        NDArray x = np.linspace(0.0, 1.0, 20);
        NDArray y = 2.5 * x + 1.0;
        using (Py.GIL())
        {
            py.Exec("from scipy import stats, optimize");
            dynamic line = py.stats.linregress(x, y);
            double slope = line.slope, intercept = line.intercept, r = line.rvalue;
            Assert.AreEqual(2.5, slope, 1e-12);
            Assert.AreEqual(1.0, intercept, 1e-12);
            Assert.AreEqual(1.0, r, 1e-12, "the points are on a line");

            NDArray start = np.array(new[] { -1.2, 1.0 });
            dynamic minimum = py.optimize.minimize(py.optimize.rosen, start, method: "BFGS");
            bool converged = minimum.success;
            NDArray at = minimum.x;
            Assert.IsTrue(converged);
            Assert.AreEqual(1.0, at.GetDouble(0), 1e-4, "the Rosenbrock function has its minimum at (1, 1)");
            Assert.AreEqual(1.0, at.GetDouble(1), 1e-4);
            Assert.AreEqual(-1.2, start.GetDouble(0), "the optimizer does not write into the start vector");
        }
    }
}
