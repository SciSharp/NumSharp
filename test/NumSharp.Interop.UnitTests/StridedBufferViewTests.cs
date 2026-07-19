using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Zero-copy views over NON-numpy, NON-contiguous buffer exporters (a sliced / offset /
    ///     reversed <c>memoryview</c>, a strided memoryview of an <c>array.array</c>). These used to be
    ///     REJECTED ("not C-contiguous and not a numpy array") and would only copy; the buffer-strides
    ///     route now reconstructs them as true views. Every case passes data BACK AND FORTH —
    ///     read → mutate via NumSharp → Python reads → mutate via Python → NumSharp reads → compare —
    ///     and the base-class leak gate proves the strided leases release.
    /// </summary>
    [TestClass]
    public class StridedBufferViewTests : InteropTestBase
    {
        // ---- writable strided memoryviews: full round trip -----------------------------------------

        [TestMethod]
        public void StridedEven_Writable_RoundTripsBothWays()
        {
            PyExec("ba = bytearray(range(16))");        // view = ba[0,2,4,6,8,10,12,14]
            var v = ViewOf("memoryview(ba)[::2]", allowReadonly: true);

            v.size.Should().Be(8);
            v.Shape.IsWriteable.Should().BeTrue("a strided view over a writable bytearray is writable");
            ReadAt<byte>(v, 0).Should().Be(0);
            ReadAt<byte>(v, 7).Should().Be(14, "logical index 7 is byte 14");

            // NumSharp → Python: logical index 3 maps to byte 6
            WriteAt(v, (byte)99, 3);
            PyLong("ba[6]").Should().Be(99, "NumSharp write through the strided view lands at the right byte");
            PyLong("ba[7]").Should().Be(7, "the untouched neighbor byte is unchanged (stride is respected)");

            // Python → NumSharp: byte 8 is logical index 4
            PyExec("ba[8] = 200");
            ReadAt<byte>(v, 4).Should().Be(200, "Python's write is visible through the view");

            v.Dispose();
        }

        [TestMethod]
        public void StridedOffset_Writable_RoundTrips()
        {
            PyExec("ba = bytearray(range(16))");        // [1::2] = [1,3,5,7,9,11,13,15]
            var v = ViewOf("memoryview(ba)[1::2]", allowReadonly: true);

            ReadAt<byte>(v, 0).Should().Be(1, "the buffer pointer already includes the +1 offset");
            ReadAt<byte>(v, 7).Should().Be(15);

            WriteAt(v, (byte)111, 0);
            PyLong("ba[1]").Should().Be(111);
            PyLong("ba[0]").Should().Be(0, "the offset element is untouched");

            v.Dispose();
        }

        [TestMethod]
        public void Reversed_NegativeStride_RoundTrips()
        {
            PyExec("ba = bytearray(range(16))");        // [::-1] = [15,14,...,0]
            var v = ViewOf("memoryview(ba)[::-1]", allowReadonly: true);

            v.size.Should().Be(16);
            ReadAt<byte>(v, 0).Should().Be(15, "reversed element 0 is the last byte");
            ReadAt<byte>(v, 15).Should().Be(0);

            WriteAt(v, (byte)77, 0);
            PyLong("ba[15]").Should().Be(77, "negative-stride element 0 maps to the base's last byte");

            PyExec("ba[0] = 88");
            ReadAt<byte>(v, 15).Should().Be(88, "the base's first byte is reversed element 15");

            v.Dispose();
        }

        [TestMethod]
        public void StridedArrayArray_Int32_RoundTrips()
        {
            PyExec("aa = array.array('i', range(10))");     // int32, [::2] = [0,2,4,6,8]
            var v = ViewOf("memoryview(aa)[::2]", allowReadonly: true);

            v.typecode.Should().Be(NPTypeCode.Int32, "dtype comes from the buffer format 'i'");
            v.size.Should().Be(5);
            ReadAt<int>(v, 2).Should().Be(4);

            WriteAt(v, 1234, 2);              // logical 2 → aa index 4
            PyLong("aa[4]").Should().Be(1234);
            PyLong("aa[3]").Should().Be(3, "the skipped element is untouched");

            PyExec("aa[0] = -9");
            ReadAt<int>(v, 0).Should().Be(-9);

            v.Dispose();
        }

        [TestMethod]
        public void StridedArrayArray_Double_RoundTrips()
        {
            PyExec("ad = array.array('d', [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])");   // [::2] = [0.5, 2.5, 4.5]
            var v = ViewOf("memoryview(ad)[::2]", allowReadonly: true);

            v.typecode.Should().Be(NPTypeCode.Double);
            v.size.Should().Be(3);
            ReadAt<double>(v, 1).Should().BeApproximately(2.5, 1e-12);

            WriteAt(v, -8.25, 1);            // logical 1 → ad index 2
            PyFloat("ad[2]").Should().BeApproximately(-8.25, 1e-12);

            v.Dispose();
        }

        // ---- read-only strided sources -------------------------------------------------------------

        [TestMethod]
        public void ReadOnlyStrided_IsNonWriteableView_GuardedWriteThrows()
        {
            var v = ViewOf("memoryview(bytes(range(16)))[::2]", allowReadonly: true);

            v.Shape.IsWriteable.Should().BeFalse("a strided view over read-only bytes is non-writeable");
            ReadAt<byte>(v, 0).Should().Be(0);
            ReadAt<byte>(v, 7).Should().Be(14);

            ((Action)(() => v[0] = (NDArray)(byte)5))
                .Should().Throw<Exception>("guarded writes through a non-writeable strided view must not corrupt immutable bytes");

            v.Dispose();
        }

        [TestMethod]
        public void ReadOnlyStrided_RejectedWithoutOptIn()
        {
            ((Action)(() => ViewOf("memoryview(bytes(range(8)))[::2]")))
                .Should().Throw<InvalidOperationException>().WithMessage("*read-only*",
                    "a read-only strided source must be refused by default, exactly like contiguous bytes");
        }

        // ---- lease semantics: it is a real view, and it locks the source ---------------------------

        [TestMethod]
        public void StridedView_ExportRoundTrip_SharesMemory_NotACopy()
        {
            PyExec("ba = bytearray(range(20))");
            var v = ViewOf("memoryview(ba)[::4]", allowReadonly: true);   // [0,4,8,12,16]

            ExportTo("rex", v);
            PyStr("rex.tolist()").Should().Be("[0, 4, 8, 12, 16]", "the strided values survive the view→export round trip");
            PyBool("np.shares_memory(rex, ba)").Should().BeTrue("a re-exported strided view must alias the original bytearray — proof it was a VIEW, not a copy");

            v.Dispose();
        }

        [TestMethod]
        public void StridedView_LocksSourceAgainstResize()
        {
            PyExec("ba = bytearray(range(16))");
            var v = ViewOf("memoryview(ba)[::2]", allowReadonly: true);

            PyExec("try:\n    ba.append(1)\n    aerr = ''\nexcept BufferError:\n    aerr = 'BufferError'");
            PyStr("aerr").Should().Be("BufferError", "the strided Py_buffer lease pins the resizable exporter, exactly like a contiguous view");

            v.Dispose();
        }

        [TestMethod]
        public void StridedView_DerivedNumSharpSlice_StillAliasesPython()
        {
            PyExec("ba = bytearray(range(16))");        // view = [0,2,4,6,8,10,12,14]
            var v = ViewOf("memoryview(ba)[::2]", allowReadonly: true);

            var sub = v["2:5"];    // logical [4,6,8] → bytes 4,6,8
            ReadAt<byte>(sub, 0).Should().Be(4);
            WriteAt(sub, (byte)42, 0);
            PyLong("ba[4]").Should().Be(42, "a NumSharp slice of the strided view still writes through to Python");

            sub.Dispose();
            v.Dispose();
        }

        // ---- correctness vs the copy path, and the unviewable boundary -----------------------------

        [TestMethod]
        public void StridedView_ValuesMatchTheCopyPath()
        {
            PyExec("ba = bytearray(range(24))");
            var view = ViewOf("memoryview(ba)[3::4]", allowReadonly: true);   // [3,7,11,15,19,23]
            var copy = ImportOf("memoryview(ba)[3::4]");

            view.size.Should().Be(copy.size);
            for (int i = 0; i < view.size; i++)
                ReadAt<byte>(view, i).Should().Be(ReadAt<byte>(copy, i), $"strided VIEW and COPY must agree at index {i}");

            view.Dispose();
            copy.Dispose();
        }

        [TestMethod]
        public void StridedComplex64Memoryview_IsUnviewable_ButCopies()
        {
            // complex64 has no zero-copy NumSharp dtype even contiguously, so a strided one is
            // genuinely unviewable — the view route declines, the copy route widens.
            PyExec("cm = memoryview(np.array([1+2j, 3+4j, 5+6j], dtype='c8'))[::2]");
            ((Action)(() => ViewOf("cm", allowReadonly: true)))
                .Should().Throw<NotSupportedException>("complex64 cannot be reinterpreted as 16-byte Complex");

            var copy = ImportOf("cm");
            copy.typecode.Should().Be(NPTypeCode.Complex);
            copy.size.Should().Be(2);
            copy.Dispose();
        }

        // ---- through the registered codec (Auto): strided memoryviews now decode as VIEWS ----------

        [TestMethod]
        public void CodecAuto_StridedMemoryview_DecodesAsAView()
        {
            CodecTests.EnsureCodec();
            PyExec("ba = bytearray(range(16))");

            NDArray decoded;
            using (Gil())
            {
                using var src = Scope.Eval("memoryview(ba)[::2]");
                decoded = src.As<NDArray>();     // Auto: strided source is viewable → shared view
            }

            WriteAt(decoded, (byte)55, 1);       // logical 1 → byte 2
            PyLong("ba[2]").Should().Be(55, "Auto now decodes a strided memoryview as a zero-copy VIEW, not a copy");

            decoded.Dispose();
        }
    }
}
