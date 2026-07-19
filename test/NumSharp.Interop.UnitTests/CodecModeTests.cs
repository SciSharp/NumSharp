using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The <see cref="NumpyCodecMode"/> matrix: Auto (view-first, copy only when a view is
    ///     impossible), View (share or decline), Copy (always a snapshot) — across both directions.
    ///
    ///     <para>Codecs are constructed directly and exercised through the public
    ///     <see cref="IPyObjectDecoder.TryDecode{T}"/> / <see cref="IPyObjectEncoder.TryEncode"/> so
    ///     all three modes run in one engine session (process-global <c>RegisterCodec</c> is sticky and
    ///     could only pin ONE mode). The base-class leak gate proves none of it leaks.</para>
    /// </summary>
    [TestClass]
    public class CodecModeTests : InteropTestBase
    {
        private static readonly NumpyCodec Auto = new(NumpyCodecOptions.Default);
        private static readonly NumpyCodec ViewOnly = new(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.View, DecodeMode = NumpyCodecMode.View });
        private static readonly NumpyCodec CopyOnly = new(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.Copy, DecodeMode = NumpyCodecMode.Copy });

        private NDArray Decode(NumpyCodec codec, string expr, out bool ok)
        {
            using (Gil())
            {
                using PyObject src = Scope.Eval(expr);
                ok = codec.TryDecode<NDArray>(src, out NDArray nd);
                return nd;
            }
        }

        // ---- the defaults --------------------------------------------------------------------------

        [TestMethod]
        public void Default_Modes_AreAutoBothDirections()
        {
            NumpyCodecOptions.Default.EncodeMode.Should().Be(NumpyCodecMode.Auto);
            NumpyCodecOptions.Default.DecodeMode.Should().Be(NumpyCodecMode.Auto);
            NumpyCodecOptions.Default.DecodeAnyBuffer.Should().BeTrue();
        }

        // ---- decode: Auto = view-first, copy-fallback ----------------------------------------------

        [TestMethod]
        public void Auto_Decode_ContiguousNumpy_IsAZeroCopyView()
        {
            PyExec("s_auto = np.arange(4, dtype='f8')");
            NDArray nd = Decode(Auto, "s_auto", out bool ok);
            ok.Should().BeTrue();

            WriteAt(nd, -7.0, 2);
            PyFloat("float(s_auto[2])").Should().BeApproximately(-7.0, 1e-12,
                "a contiguous source is viewable, so Auto shares memory — the C# write reaches Python");
            nd.Dispose();
        }

        [TestMethod]
        public void Auto_Decode_Complex64_FallsBackToCopy()
        {
            // complex64 has NO zero-copy NumSharp representation (Complex is 16 bytes, c8 is 8) —
            // Auto must transparently fall back to a widening copy instead of failing.
            PyExec("c_auto = np.array([1+2j, 3+4j], dtype='c8')");
            NDArray nd = Decode(Auto, "c_auto", out bool ok);

            ok.Should().BeTrue("Auto falls back to a copy when a view is impossible");
            nd.typecode.Should().Be(NPTypeCode.Complex, "complex64 widens to complex128 on the copy path");
            ReadAt<Complex>(nd, 0).Should().Be(new Complex(1, 2));

            // it's a COPY: mutating it must NOT reach Python (no shared memory)
            WriteAt(nd, new Complex(99, 99), 0);
            PyFloat("float(c_auto[0].real)").Should().BeApproximately(1.0, 1e-12, "the fallback is an independent copy");
            nd.Dispose();
        }

        [TestMethod]
        public void Auto_Decode_StridedMemoryview_IsNowAView()
        {
            // A sliced memoryview is non-contiguous with no __array_interface__, yet the buffer-strides
            // route makes it viewable — so Auto SHARES it (view-first) instead of copying.
            PyExec("ba_sm = bytearray(range(16))");
            NDArray nd = Decode(Auto, "memoryview(ba_sm)[::2]", out bool ok);

            ok.Should().BeTrue();
            nd.size.Should().Be(8);
            WriteAt(nd, (byte)77, 1);   // logical 1 -> byte 2
            PyLong("ba_sm[2]").Should().Be(77, "Auto now views a strided memoryview — the write reaches Python");
            nd.Dispose();
        }

        [TestMethod]
        public void Auto_Decode_StridedComplex64_FallsBackToCopy()
        {
            // Strided AND complex64 is genuinely unviewable (no 16-byte reinterpret), so even the
            // buffer-strides route declines and Auto falls back to a widening copy.
            PyExec("cm_auto = np.array([1+2j, 3+4j, 5+6j, 7+8j], dtype='c8')");
            NDArray nd = Decode(Auto, "memoryview(cm_auto)[::2]", out bool ok);

            ok.Should().BeTrue("Auto copies when a strided source is genuinely unviewable");
            nd.typecode.Should().Be(NPTypeCode.Complex, "complex64 widens to complex128 on the copy path");
            nd.size.Should().Be(2);
            ReadAt<Complex>(nd, 0).Should().Be(new Complex(1, 2));

            WriteAt(nd, new Complex(9, 9), 0);
            PyFloat("float(cm_auto[0].real)").Should().BeApproximately(1.0, 1e-12, "the fallback is an independent copy");
            nd.Dispose();
        }

        [TestMethod]
        public void Auto_Decode_ReadonlyBytes_IsANonWriteableView()
        {
            // bytes is read-only but contiguous → viewable as a NON-WRITEABLE view (that IS a view).
            // Auto prefers it over a copy; the copy path would instead own writable memory.
            NDArray nd = Decode(Auto, "b'\\x01\\x02\\x03\\x04'", out bool ok);
            ok.Should().BeTrue();
            nd.Shape.IsWriteable.Should().BeFalse("a view over read-only bytes is non-writeable — proof it's the view path, not a copy");
            ReadAt<byte>(nd, 1).Should().Be(2);
            nd.Dispose();
        }

        // ---- decode: View = share or decline -------------------------------------------------------

        [TestMethod]
        public void View_Decode_ContiguousNumpy_IsAView()
        {
            PyExec("s_view = np.arange(4, dtype='f8')");
            NDArray nd = Decode(ViewOnly, "s_view", out bool ok);
            ok.Should().BeTrue();

            WriteAt(nd, 3.5, 0);
            PyFloat("float(s_view[0])").Should().BeApproximately(3.5, 1e-12);
            nd.Dispose();
        }

        [TestMethod]
        public void View_Decode_Complex64_Declines_NoSilentCopy()
        {
            // The whole point of View mode: when a view is impossible it DECLINES (returns false)
            // rather than silently copying — a loud signal for callers who depend on shared memory.
            NDArray nd = Decode(ViewOnly, "np.array([1+2j, 3+4j], dtype='c8')", out bool ok);
            ok.Should().BeFalse("View mode never falls back to a copy");
            nd.Should().BeNull();
        }

        [TestMethod]
        public void View_Decode_StridedMemoryview_IsAView()
        {
            // The buffer-strides route makes a sliced memoryview viewable, so View mode shares it.
            PyExec("ba_sv = bytearray(range(16))");
            NDArray nd = Decode(ViewOnly, "memoryview(ba_sv)[::2]", out bool ok);

            ok.Should().BeTrue("View mode shares a strided memoryview via the buffer-strides route");
            WriteAt(nd, (byte)33, 0);
            PyLong("ba_sv[0]").Should().Be(33, "the strided view is writable and shared");
            nd.Dispose();
        }

        [TestMethod]
        public void View_Decode_StridedComplex64_Declines_NoSilentCopy()
        {
            // Genuinely unviewable even strided → View mode declines rather than silently copying.
            PyExec("cm_view = np.array([1+2j, 3+4j, 5+6j, 7+8j], dtype='c8')");
            NDArray nd = Decode(ViewOnly, "memoryview(cm_view)[::2]", out bool ok);
            ok.Should().BeFalse("a genuinely unviewable strided source is declined in View mode, never copied");
            nd.Should().BeNull();
        }

        // ---- decode: Copy = always a snapshot ------------------------------------------------------

        [TestMethod]
        public void Copy_Decode_ContiguousNumpy_IsASnapshot()
        {
            PyExec("s_copy = np.arange(4, dtype='f8')");
            NDArray nd = Decode(CopyOnly, "s_copy", out bool ok);
            ok.Should().BeTrue();

            WriteAt(nd, -1.0, 0);
            PyFloat("float(s_copy[0])").Should().BeApproximately(0.0, 1e-12, "Copy mode never shares memory, even for a viewable source");
            nd.Shape.IsWriteable.Should().BeTrue("a copy owns writable memory");
            nd.Dispose();
        }

        // ---- acceptance: capability check, not a name allowlist -------------------------------------

        [TestMethod]
        public void CanDecode_AcceptsAnyBufferExporter_AndRejectsNonBuffers()
        {
            // A ctypes array's tp_name is generated per element type AND length ("c_long_Array_4"), so
            // it can never be allowlisted — the PEP 688 __buffer__ capability check is what accepts it.
            using (Gil())
            {
                using var ct = Scope.Eval("__import__('ctypes').c_int * 4");   // the TYPE itself
                Auto.CanDecode(new PyType(ct), typeof(NDArray)).Should().BeTrue("any buffer exporter must decode");

                foreach (string nonBuffer in new[] { "dict", "int", "str", "list", "float" })
                {
                    using var t = Scope.Eval(nonBuffer);
                    Auto.CanDecode(new PyType(t), typeof(NDArray)).Should().BeFalse(
                        $"{nonBuffer} exports no buffer — acceptance must not widen past real exporters");
                }
            }
        }

        [TestMethod]
        public void Auto_Decode_CtypesArray_IsAView()
        {
            // The payoff of the capability check + memoryview-wrapper acquisition: a ctypes array now
            // flows through Auto and comes back as a zero-copy shared view.
            PyExec("import ctypes\ncar_auto = (ctypes.c_int * 4)()\ncar_auto[0]=5; car_auto[1]=6; car_auto[2]=7; car_auto[3]=8");
            NDArray nd = Decode(Auto, "car_auto", out bool ok);

            ok.Should().BeTrue("a ctypes array is a buffer exporter, so Auto decodes it");
            nd.typecode.Should().Be(NPTypeCode.Int32);
            ReadAt<int>(nd, 2).Should().Be(7);

            WriteAt(nd, 4242, 0);
            PyLong("car_auto[0]").Should().Be(4242, "Auto decoded the ctypes array as a VIEW — the write reaches Python");
            nd.Dispose();
        }

        // ---- encode: Auto/View share, Copy detaches ------------------------------------------------

        [TestMethod]
        public void Auto_Encode_IsAZeroCopyView()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                using PyObject p = Auto.TryEncode(nd);
                p.Should().NotBeNull();
                Scope.Set("e_auto", p);
            }

            PyExec("e_auto[1] = 55.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(55.0, 1e-12, "Auto encode is a shared view — Python's write reaches NumSharp");
        }

        [TestMethod]
        public void Copy_Encode_IsASnapshot()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                using PyObject p = CopyOnly.TryEncode(nd);
                p.Should().NotBeNull();
                Scope.Set("e_copy", p);
            }

            PyExec("e_copy[1] = 55.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(1.0, 1e-12, "Copy encode is independent — Python's write does not reach NumSharp");
        }

        [TestMethod]
        public void Auto_Encode_Decimal_FallsBackToClrWrap()
        {
            // Decimal has no numpy dtype — neither a view NOR a copy can express it, so even Auto
            // returns null (pythonnet then CLR-wraps the NDArray).
            var dec = np.arange(3).astype(NPTypeCode.Decimal);
            using (Gil())
            {
                Auto.TryEncode(dec).Should().BeNull("no numpy dtype exists for decimal in any mode");
            }
        }
    }
}
