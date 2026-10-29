using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     NumPy: the same buffer on both sides. A NumSharp array crosses to NumPy as a view (no bytes move),
///     NumPy computes over it, and NumPy's results come back as NumSharp arrays.
/// </summary>
[TestClass]
public class NumPyTests : PythonTest
{
    /// <summary>
    ///     NumPy views NumSharp's memory: a write on either side is visible on the other, and NumPy's FFT of
    ///     the shared samples returns as an <see cref="NDArray"/> that NumSharp reduces.
    /// </summary>
    [TestMethod]
    public void NumPyViewsNumSharpMemory_AndItsResultsComeBack()
    {
        using var scope = NDScope.Open();
        NDArray samples = np.linspace(0.0, 1.0, 64, endpoint: false);
        NDArray tone = np.sin(2 * Math.PI * 5 * samples) + 0.5 * np.sin(2 * Math.PI * 12 * samples);
        using (Py.GIL())
        {
            py.tone = tone;                                           // the codec ran tone.ToNumpy()
            bool numpyOwnsIt = py.tone.flags.owndata;
            Assert.IsFalse(numpyOwnsIt, "NumPy views NumSharp's buffer; it did not copy it");

            py.Exec("tone[0] = 3.0");
            Assert.AreEqual(3.0, tone.GetDouble(0), "a write from NumPy is visible in NumSharp");
            tone[0] = 0.0;
            double seenByNumPy = py.Eval("tone[0]");
            Assert.AreEqual(0.0, seenByNumPy, "a write from NumSharp is visible in NumPy");

            NDArray spectrum = py.np.abs(py.np.fft.rfft(tone));
            Assert.AreEqual(NPTypeCode.Double, spectrum.typecode);
            Assert.AreEqual(33L, spectrum.size, "rfft of 64 samples has 64 / 2 + 1 bins");
            Assert.AreEqual(5L, np.argmax(spectrum), "the 5 Hz component is the loudest");
            Assert.AreEqual(32.0, spectrum.GetDouble(5), 1e-9);
            Assert.AreEqual(16.0, spectrum.GetDouble(12), 1e-9, "the 12 Hz component has half the amplitude");
        }
    }

    /// <summary>
    ///     A NumPy function that returns several arrays (<c>numpy.linalg.svd</c>) reads into one C# tuple, and
    ///     NumSharp multiplies the factors back into the original matrix.
    /// </summary>
    [TestMethod]
    public void SeveralResultsComeBackAsOneTuple()
    {
        using var scope = NDScope.Open();
        NDArray matrix = np.array(new double[,] { { 2, 0, 1 }, { 0, 3, 0 }, { 1, 0, 2 }, { 0, 1, 1 } });
        using (Py.GIL())
        {
            (NDArray u, NDArray s, NDArray vt) svd = py.np.linalg.svd(matrix, full_matrices: false);
            CollectionAssert.AreEqual(new long[] { 4, 3 }, svd.u.shape);
            CollectionAssert.AreEqual(new long[] { 3 }, svd.s.shape);
            CollectionAssert.AreEqual(new long[] { 3, 3 }, svd.vt.shape);
            Assert.IsTrue(svd.s.GetDouble(0) >= svd.s.GetDouble(1) && svd.s.GetDouble(1) >= svd.s.GetDouble(2), "singular values are sorted");

            NDArray rebuilt = np.matmul(svd.u * svd.s, svd.vt);
            Assert.IsTrue(np.allclose(rebuilt, matrix), "U * S * Vt is the matrix again");
        }
    }

    /// <summary>
    ///     The explicit conversion calls choose between a view and a copy in each direction:
    ///     <c>ToNumpy</c> / <c>ToNumpyCopy</c> out, <c>AsNDArray</c> / <c>ToNDArray</c> back.
    /// </summary>
    [TestMethod]
    public void ExplicitConversions_ChooseViewOrCopy()
    {
        using var scope = NDScope.Open();
        NDArray source = np.arange(6).astype(np.float64);
        using (Py.GIL())
        {
            py.view = source.ToNumpy();
            py.copy = source.ToNumpyCopy();
            source[0] = 42.0;
            double inView = py.Eval("view[0]"), inCopy = py.Eval("copy[0]");
            Assert.AreEqual(42.0, inView, "the view follows NumSharp");
            Assert.AreEqual(0.0, inCopy, "the copy does not");

            using PyObject doubled = py.Eval("view * 2.0");           // a new array that NumPy owns
            NDArray alias = doubled.AsNDArray();
            NDArray snapshot = doubled.ToNDArray();
            alias[1] = -1.0;
            py.doubled = doubled;
            double inNumPy = py.Eval("doubled[1]");
            Assert.AreEqual(-1.0, inNumPy, "AsNDArray is a view of NumPy's buffer");
            Assert.AreEqual(2.0, snapshot.GetDouble(1), "ToNDArray is an independent copy");
        }
    }
}
