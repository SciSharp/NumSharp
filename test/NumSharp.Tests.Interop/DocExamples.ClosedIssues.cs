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
    ///     resolved by the NumSharp.Interop.pythonnet package in 0.70.0 (#628). Each mirrors the
    ///     example shown in the issue's closing comment and proves the crossing against the real numpy
    ///     embedded in the test process (over NumSharp's own memory). The tests are not referenced by
    ///     the comments.
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
    }
}
