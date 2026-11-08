using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Backing live-oracle tests for the closing comments posted on the pythonnet interop issues
    ///     resolved by the NumSharp.Interop.pythonnet package in 0.70.0 (#628), and for the pythonnet
    ///     halves of the answers posted in the 2026-11-08 triage after 0.80.0 (#349, #498; their other
    ///     halves are in NumSharp.Tests' <c>Issues</c> folder). Each mirrors the example shown in the
    ///     issue's closing comment and proves the crossing against the real numpy embedded in the test
    ///     process (over NumSharp's own memory). The tests are not referenced by the comments.
    /// </summary>
    [TestClass]
    public class ClosedInteropIssuesTests : InteropTestBase
    {
        // Issue #383 - NumSharp NDArray -> numpy (ToNumpy), zero-copy both directions
        [TestMethod]
        public void Issue383_ToNumpy_SharesBuffer()
        {
            var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);

            ExportTo("a", nd);   // nd.ToNumpy() under the hood, zero-copy

            PyBool("np.array_equal(a, np.arange(6, dtype='f8').reshape(2, 3))").Should().BeTrue();

            PyExec("a[0, 0] = 100.0");
            ReadAt<double>(nd, 0, 0).Should().BeApproximately(100.0, 1e-12, "numpy's write lands in NumSharp's memory");
        }

        // Issue #411 - pull a numpy array out of a PyObject dict and bring it into NumSharp
        [TestMethod]
        public void Issue411_ExtractNumpyFromPyObjectDict()
        {
            PyExec("d = {'weights': np.arange(4, dtype='f8')}");

            var nd = ImportOf("d['weights']");   // AsNDArray/FromArrayLike crossing

            nd.shape.Should().Equal(new long[] { 4 });
            nd.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(nd, 3).Should().BeApproximately(3.0, 1e-12);
        }

        // Issue #416 - numpy ndarray -> NumSharp NDArray (AsNDArray), no manual UnmanagedMemoryBlock
        [TestMethod]
        public void Issue416_NumpyToNDArray()
        {
            var nd = ImportOf("np.arange(4, dtype='f8')");

            nd.shape.Should().Equal(new long[] { 4 });
            nd.typecode.Should().Be(NPTypeCode.Double);
            for (int i = 0; i < 4; i++)
                ReadAt<double>(nd, i).Should().BeApproximately(i, 1e-12);
        }

        // Issue #383 (implicit) - register the codec once, then an NDArray auto-encodes to numpy
        [TestMethod]
        public void Issue383_ImplicitEncode_ViaCodec()
        {
            CodecTests.EnsureCodec();   // NDArrayPythonInterop.RegisterCodec()
            var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            using (Gil())
            {
                dynamic numpy = Py.Import("numpy");
                double total = (double)numpy.sum(nd);   // nd auto-encoded to a numpy array
                total.Should().BeApproximately(15.0, 1e-9);
            }
        }

        // Issue #411 (implicit) - PyObject.As<NDArray>() after RegisterCodec()
        [TestMethod]
        public void Issue411_ImplicitDecode_FromDict()
        {
            CodecTests.EnsureCodec();
            using (Gil())
            {
                using PyObject arr = Scope.Eval("{'weights': np.arange(4, dtype='f8')}['weights']");
                using var nd = arr.As<NDArray>();   // implicit decode via the codec
                nd.shape.Should().Equal(new long[] { 4 });
                ReadAt<double>(nd, 3).Should().BeApproximately(3.0, 1e-12);
            }
        }

        // Issue #416 (implicit) - PyObject.As<NDArray>() after RegisterCodec()
        [TestMethod]
        public void Issue416_ImplicitDecode_AsNDArray()
        {
            CodecTests.EnsureCodec();
            using (Gil())
            {
                using PyObject npArr = Scope.Eval("np.arange(4, dtype='f8')");
                using var nd = npArr.As<NDArray>();   // implicit decode via the codec
                nd.shape.Should().Equal(new long[] { 4 });
                for (int i = 0; i < 4; i++)
                    ReadAt<double>(nd, i).Should().BeApproximately(i, 1e-12);
            }
        }

        /// <summary>
        ///     Issue #498 (IronPython, closed in the 0.80.0 triage): the answer points at NumSharp.Interop.pythonnet - real
        ///     CPython with numpy from C#, with an <c>NDArray</c> and numpy sharing memory. numpy computes over the exported
        ///     array without a copy, and a write on either side is seen by the other.
        /// </summary>
        [TestMethod]
        public void Issue498_RealNumpyFromCSharp_SharesMemoryBothWays()
        {
            var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);

            ExportTo("a", nd);   // zero-copy: numpy sees NumSharp's own buffer

            PyFloat("float(np.sum(a))").Should().Be(15.0);
            PyExec("a[1, 2] = -7.0");
            ReadAt<double>(nd, 1, 2).Should().Be(-7.0, "numpy's write lands in the NDArray");
            WriteAt(nd, 42.0, 0, 0);
            PyFloat("float(a[0, 0])").Should().Be(42.0, "the NDArray's write is seen by numpy");
        }

        /// <summary>
        ///     Issue #349 (scipy.signal, closed in the 0.80.0 triage): the answer says SciPy itself can be called through
        ///     NumSharp.Interop.pythonnet. <c>scipy.signal.spectrogram</c> over an exported <c>NDArray</c> (Hann window, 8-sample
        ///     segments, overlap 4, no detrending, power-spectrum scaling) agrees within 1e-12 with the spectrogram built from
        ///     NumSharp's own <c>np.hanning</c> and <c>np.fft.rfft</c>, scaled as SciPy scales a one-sided spectrum
        ///     (1/sum(w)^2, inner bins doubled).
        /// </summary>
        [TestMethod]
        [PythonEcosystem]
        public void Issue349_ScipySpectrogram_OverAnExportedNDArray_MatchesNumSharpsOwn()
        {
            SkipUnless("scipy");   // before any export: a skipped test must not leave an export pinned

            var t = np.arange(32) / 32.0;
            var sig = np.sin(2 * System.Math.PI * 5 * t) + 0.5 * np.cos(2 * System.Math.PI * 12 * t);
            var w = np.hanning(8);

            ExportTo("sig", sig);
            ExportTo("w", w);
            PyExec("import scipy.signal\n" +
                   "_, _, sxx = scipy.signal.spectrogram(sig, fs=1.0, window=w, nperseg=8, noverlap=4, detrend=False, scaling='spectrum', mode='psd')");
            var scipyResult = ImportOf("sxx");

            var frames = np.stack(Enumerable.Range(0, 7).Select(k => sig[$"{k * 4}:{k * 4 + 8}"]).ToArray());
            var power = np.power(np.abs(np.fft.rfft(frames * w, axis: 1)), 2) / System.Math.Pow(np.sum(w).GetDouble(), 2);
            power[":, 1:-1"] *= 2;
            var ours = power.T;

            scipyResult.shape.Should().Equal(5L, 7L);
            np.allclose(scipyResult, ours, rtol: 1e-12, atol: 0).Should().BeTrue("SciPy over the NDArray matches NumSharp's own spectrogram");
        }
    }
}
