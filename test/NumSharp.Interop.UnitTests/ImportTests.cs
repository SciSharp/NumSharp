using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>Python → NumSharp: copies from any exporter, zero-copy views, both lease routes, error paths.</summary>
    [TestClass]
    public class ImportTests : InteropTestBase
    {
        /// <summary>Copy-import <paramref name="expr"/>, then re-export and let numpy prove equality.</summary>
        private void AssertCopyRoundTrip(string name, string expr, NPTypeCode expectedTc, string equalExpr = "np.array_equal(rt, srcobj)")
        {
            PyExec($"srcobj = {expr}");
            var nd = ImportOf("srcobj");
            nd.typecode.Should().Be(expectedTc, name);

            ExportTo("rt", nd);
            PyBool(equalExpr).Should().BeTrue($"{name}: values must round-trip exactly");
            PyBool("np.shares_memory(rt, srcobj)").Should().BeFalse($"{name}: ToNDArray must copy");
        }

        [TestMethod]
        public void Copy_PreservesValuesForEveryLayout()
        {
            AssertCopyRoundTrip("contiguous", "np.arange(12, dtype='i4').reshape(3, 4)", NPTypeCode.Int32);
            AssertCopyRoundTrip("strided", "np.arange(12, dtype='i4').reshape(3, 4)[:, ::2]", NPTypeCode.Int32);
            AssertCopyRoundTrip("fortran", "np.asfortranarray(np.arange(6, dtype='f8').reshape(2, 3))", NPTypeCode.Double);
            AssertCopyRoundTrip("reversed", "np.arange(10, dtype='i8')[::-1]", NPTypeCode.Int64);
        }

        [TestMethod]
        public void Copy_PreservesValuesForEveryDtype()
        {
            AssertCopyRoundTrip("bool", "np.array([True, False, True])", NPTypeCode.Boolean);
            AssertCopyRoundTrip("float16", "np.arange(4, dtype='f2')", NPTypeCode.Half);
            AssertCopyRoundTrip("uint64", "np.arange(3, dtype='u8')", NPTypeCode.UInt64);
            AssertCopyRoundTrip("complex128", "np.array([1+2j, 3+4j])", NPTypeCode.Complex);
        }

        [TestMethod]
        public void Copy_Complex64_WidensToComplex128()
        {
            AssertCopyRoundTrip("complex64", "np.array([1+2j, 3+4j], dtype='c8')", NPTypeCode.Complex,
                "np.array_equal(rt, srcobj.astype('c16'))");
        }

        [TestMethod]
        public void Copy_ZeroDim_BecomesScalar()
        {
            var nd = ImportOf("np.float64(3.25)");
            nd.ndim.Should().Be(0);
            ReadAt<double>(nd).Should().BeApproximately(3.25, 1e-12);
        }

        [TestMethod]
        public void Copy_Empty_PreservesShapeAndDtype()
        {
            var nd = ImportOf("np.empty((0, 3), dtype='f4')");
            nd.typecode.Should().Be(NPTypeCode.Single);
            nd.ndim.Should().Be(2);
            nd.shape[0].Should().Be(0);
            nd.shape[1].Should().Be(3);
        }

        [TestMethod]
        public void Copy_NonNumpyExporters()
        {
            var bytes = ImportOf("b'abcd'");
            bytes.typecode.Should().Be(NPTypeCode.Byte);
            ReadAt<byte>(bytes, 0).Should().Be(97);
            ReadAt<byte>(bytes, 3).Should().Be(100);

            var arr = ImportOf("array.array('d', [1.5, 2.5])");
            arr.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(arr, 1).Should().BeApproximately(2.5, 1e-12);

            var mv = ImportOf("memoryview(np.arange(5, dtype='i8'))");
            mv.typecode.Should().Be(NPTypeCode.Int64);
            ReadAt<long>(mv, 4).Should().Be(4);
        }

        [TestMethod]
        public void Copy_BigEndianSource_IsRejectedNotByteSwapped()
        {
            PyExec("be = np.arange(3).astype('>i4')");
            ((Action)(() => ImportOf("be")))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian*");
        }

        [TestMethod]
        public void NonBufferObject_FailsWithGuidance()
        {
            ((Action)(() => ImportOf("{'a': 1}")))
                .Should().Throw<NotSupportedException>().WithMessage("*buffer*");
            ((Action)(() => ViewOf("3.5")))
                .Should().Throw<NotSupportedException>("python floats don't export buffers");
        }

        [TestMethod]
        public void View_SharesMemoryBothWays()
        {
            PyExec("iv = np.arange(8, dtype='f8')");
            var nd = ViewOf("iv");

            PyExec("iv[3] = 99.5");
            ReadAt<double>(nd, 3).Should().BeApproximately(99.5, 1e-12, "python writes must be visible through the view");

            WriteAt(nd, 42.5, 0);
            PyFloat("float(iv[0])").Should().BeApproximately(42.5, 1e-12, "NumSharp writes must land in python's buffer");

            ExportTo("rex", nd);
            PyBool("np.shares_memory(rex, iv)").Should().BeTrue("import view -> export must alias the original");
        }

        [TestMethod]
        public void View_ReadOnlySource_RejectedUnlessOptedIn()
        {
            ((Action)(() => ViewOf("b'abcd'")))
                .Should().Throw<InvalidOperationException>().WithMessage("*read-only*", "writing through bytes would corrupt an immutable object");

            var ro = ViewOf("b'abcd'", allowReadonly: true);
            ro.typecode.Should().Be(NPTypeCode.Byte);
            ReadAt<byte>(ro, 0).Should().Be(97);
        }

        [TestMethod]
        public void View_Bytearray_IsWritableAndLockedAgainstResize()
        {
            PyExec("ba = bytearray(b'abcd')");
            var ban = ViewOf("ba");

            WriteAt(ban, (byte)'z', 0);
            PyStr("ba.decode()").Should().Be("zbcd");

            PyExec("try:\n    ba.append(1)\n    aerr = ''\nexcept BufferError as e:\n    aerr = 'BufferError'");
            PyStr("aerr").Should().Be("BufferError", "the Py_buffer lease must pin resizable exporters");
        }

        [TestMethod]
        public void View_NumpyRefcheck_SeesTheLease()
        {
            PyExec("nv = np.arange(8, dtype='f8')");
            var nd = ViewOf("nv");

            // numpy's only reallocation guard is refcheck; the lease's Py_buffer holds a reference.
            PyExec("try:\n    nv.resize(16)\n    rerr = ''\nexcept ValueError as e:\n    rerr = 'ValueError'");
            PyStr("rerr").Should().Be("ValueError");
            ReadAt<double>(nd, 7).Should().BeApproximately(7.0, 1e-12, "the refused resize must not have moved the data");
        }

        [TestMethod]
        public void View_ArrayInterface_StridedTransposedReversed()
        {
            PyExec("ai2 = np.arange(20, dtype='i8')");

            var even = ViewOf("ai2[::2]");
            even.ndim.Should().Be(1);
            even.shape[0].Should().Be(10);
            WriteAt(even, -5L, 1);
            PyLong("int(ai2[2])").Should().Be(-5, "strided view writes must hit the base array");

            var rev = ViewOf("ai2[::-1]");
            WriteAt(rev, -9L, 0);
            PyLong("int(ai2[19])").Should().Be(-9, "negative-stride view element 0 is the base's last element");
            ReadAt<long>(rev, 19).Should().Be(0, "negative-stride window math must stay in bounds");

            PyExec("tb = np.arange(6, dtype='i4').reshape(2, 3)");
            var tr = ViewOf("tb.T");
            tr.shape[0].Should().Be(3);
            tr.shape[1].Should().Be(2);
            WriteAt(tr, 77, 2, 1);
            PyLong("int(tb[1, 2])").Should().Be(77, "transposed (F-order) view writes must hit the base");
        }

        [TestMethod]
        public void View_ArrayInterface_Broadcast_StaysReadOnlyThroughRoundTrip()
        {
            PyExec("bc = np.broadcast_to(np.arange(3, dtype='i8'), (4, 3))");

            ((Action)(() => ViewOf("bc")))
                .Should().Throw<InvalidOperationException>().WithMessage("*read-only*");

            var bcNd = ViewOf("bc", allowReadonly: true);
            ExportTo("bcr", bcNd);
            PyStr("bcr.tolist()").Should().Be("[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]");
            PyBool("bcr.flags.writeable").Should().BeFalse("broadcast layout must survive the round trip as read-only");
            PyBool("np.shares_memory(bcr, bc)").Should().BeTrue("only the 3-element buffer exists anywhere");
        }

        [TestMethod]
        public void View_CtypesArray_SharesMemoryBothWays()
        {
            // ctypes used to be unreachable: pythonnet's obj.GetBuffer hard-crashes on a raw ctypes
            // array for EVERY flag (view AND copy). Acquiring through the memoryview wrapper — the
            // canonical, uniformly-behaved exporter — makes it a normal shared view.
            PyExec("import ctypes\ncar = (ctypes.c_int * 4)()\ncar[0]=10; car[1]=20; car[2]=30; car[3]=40");
            var v = ViewOf("car", allowReadonly: true);

            v.typecode.Should().Be(NPTypeCode.Int32);
            v.size.Should().Be(4);
            v.Shape.IsWriteable.Should().BeTrue();
            ReadAt<int>(v, 2).Should().Be(30);

            WriteAt(v, 999, 1);
            PyLong("car[1]").Should().Be(999, "NumSharp writes must land in the ctypes buffer");

            PyExec("car[3] = -7");
            ReadAt<int>(v, 3).Should().Be(-7, "ctypes writes must be visible through the view");

            v.Dispose();
        }

        [TestMethod]
        public void Copy_CtypesArray_Works()
        {
            // The copy path reads through the memoryview for the same reason the view path does:
            // obj.GetBuffer on a raw ctypes array hard-crashes pythonnet, and ToNDArray used to hit
            // it too — so DecodeMode=Copy / ToNDArray on ctypes took the process down.
            PyExec("import ctypes\ncc = (ctypes.c_int * 4)()\ncc[0]=1; cc[1]=2; cc[2]=3; cc[3]=4");
            var nd = ImportOf("cc");

            nd.typecode.Should().Be(NPTypeCode.Int32);
            nd.size.Should().Be(4);
            ReadAt<int>(nd, 3).Should().Be(4);

            WriteAt(nd, -1, 0);
            PyLong("cc[0]").Should().Be(1, "ToNDArray copies — the write must NOT reach the ctypes buffer");
        }

        [TestMethod]
        public void View_CtypesArray_CoversOtherElementTypes()
        {
            PyExec("import ctypes\ncd = (ctypes.c_double * 3)()\ncd[0]=1.5; cd[1]=2.5; cd[2]=3.5");
            var d = ViewOf("cd", allowReadonly: true);
            d.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(d, 1).Should().BeApproximately(2.5, 1e-12);
            WriteAt(d, -8.25, 2);
            PyFloat("cd[2]").Should().BeApproximately(-8.25, 1e-12);
            d.Dispose();

            PyExec("cb = (ctypes.c_ubyte * 4)()\ncb[0]=1; cb[1]=2; cb[2]=3; cb[3]=4");
            var b = ViewOf("cb", allowReadonly: true);
            b.typecode.Should().Be(NPTypeCode.Byte);
            ReadAt<byte>(b, 3).Should().Be(4);
            b.Dispose();
        }

        [TestMethod]
        public void View_SubItemStride_IsRejectedWithCopyGuidance()
        {
            // A 2-byte stride over int32 cannot be expressed by NumSharp's element-based strides.
            PyExec("weird = np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), (2,), (2,))");
            ((Action)(() => ViewOf("weird")))
                .Should().Throw<NotSupportedException>().WithMessage("*multiple*");
        }
    }
}
