using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Codec &amp; verb correctness proved BYTE-FOR-BYTE against real numpy through pythonnet.
    ///     Every crossing reads its bytes on BOTH sides (numpy via <c>ascontiguousarray().tobytes('C')</c>,
    ///     NumSharp via its own <c>ascontiguousarray</c> + a raw buffer copy — independent of the interop's
    ///     <c>ToNumpy</c>) and asserts they are identical. That is the strongest equality there is: it
    ///     catches dtype width, endianness, and layout linearization that a value compare accepts.
    ///     numpy is spoken in the clean Pythonic style (<c>Numpy.*</c> + the PyObject extensions in
    ///     <c>Pythonic.cs</c>); <c>np</c> stays NumSharp.
    /// </summary>
    [TestClass]
    public class CodecByteContractTests : InteropTestBase
    {
        // Every numpy-mappable NumSharp dtype: its native typestr and a value sample (finite — NaN payloads
        // are a separate, value-level concern, not a byte-contract one).
        private static readonly (string typestr, NPTypeCode tc, string sample)[] Dtypes =
        {
            ("|b1",  NPTypeCode.Boolean, "[True, False, True, True]"),
            ("|u1",  NPTypeCode.Byte,    "[0, 1, 200, 255]"),
            ("|i1",  NPTypeCode.SByte,   "[0, -1, 127, -128]"),
            ("<i2",  NPTypeCode.Int16,   "[0, -1, 32767, -32768]"),
            ("<u2",  NPTypeCode.UInt16,  "[0, 1, 65535, 40000]"),
            ("<i4",  NPTypeCode.Int32,   "[0, -1, 2147483647, -2147483648]"),
            ("<u4",  NPTypeCode.UInt32,  "[0, 1, 4000000000, 4294967295]"),
            ("<i8",  NPTypeCode.Int64,   "[0, -1, 9223372036854775807]"),
            ("<u8",  NPTypeCode.UInt64,  "[0, 1, 18446744073709551615]"),
            ("<f2",  NPTypeCode.Half,    "[0.0, -1.5, 3.25, 65504.0]"),
            ("<f4",  NPTypeCode.Single,  "[0.0, -1.5, 32500000000.0]"),
            ("<f8",  NPTypeCode.Double,  "[0.0, -1.5, 3.25e100]"),
            ("<c16", NPTypeCode.Complex, "[1+2j, -3-4j, 0+0j]"),
        };

        // ==============================  helpers  ===============================================

        /// <summary>Build <paramref name="npExpr"/>, copy-decode it, assert byte-identity with the source
        /// (or with <paramref name="expectedExpr"/> when the crossing CONVERTS — widen/narrow/byteswap).</summary>
        private static void DecodeCopyByteExact(string npExpr, NPTypeCode tc, string expectedExpr = null)
        {
            using (Py.GIL())
            {
                using PyObject a = Numpy.eval(npExpr);
                using NDArray nd = a.ToNDArray();
                nd.typecode.Should().Be(tc, npExpr);
                if (expectedExpr is null)
                {
                    ByteContract.AssertSameBytes(nd, a, npExpr);
                }
                else
                {
                    using PyObject expected = Numpy.eval(expectedExpr);
                    ByteContract.AssertSameBytes(nd, expected, $"{npExpr} decodes byte-equal to {expectedExpr}");
                }
            }
        }

        /// <summary>Build <paramref name="npExpr"/>, VIEW-decode it (allowReadonly), assert byte-identity.</summary>
        private static void DecodeViewByteExact(string npExpr, NPTypeCode tc)
        {
            using (Py.GIL())
            {
                using PyObject a = Numpy.eval(npExpr);
                using NDArray nd = NDArrayPythonInterop.ToNDArrayView(a, allowReadonly: true);
                nd.typecode.Should().Be(tc, npExpr);
                ByteContract.AssertSameBytes(nd, a, npExpr);
            }
        }

        /// <summary>Export <paramref name="ns"/> (view AND copy), assert numpy dtype/shape and byte-identity
        /// with <paramref name="expectedExpr"/>.</summary>
        private static void EncodeByteExact(NDArray ns, string expectedDtypeStr, string expectedExpr)
        {
            using (Py.GIL())
            {
                using PyObject expected = Numpy.eval(expectedExpr);
                using PyObject view = ns.ToNumpy();
                view.dtype_str().Should().Be(expectedDtypeStr, expectedExpr);
                ByteContract.AssertSameBytes(view, expected, $"ToNumpy(view) of {expectedExpr}");

                using PyObject copy = ns.ToNumpyCopy();
                copy.dtype_str().Should().Be(expectedDtypeStr, expectedExpr);
                ByteContract.AssertSameBytes(copy, expected, $"ToNumpyCopy of {expectedExpr}");
            }
        }

        // ==============================  decode: every dtype  ===================================

        // ==============================  the harness has teeth  =================================

        [TestMethod]
        public void ByteContract_IsNonVacuous_DetectsValueAndWidthDifferences()
        {
            using (Gil())
            {
                using PyObject a = Numpy.array("[1, 2, 3]", "<i4");

                // a genuine numpy↔NumSharp match passes (sanity — the comparison is not always-throwing).
                using NDArray nd = a.ToNDArray();
                ByteContract.AssertSameBytes(nd, a);

                // differing VALUES differ in bytes.
                using PyObject differentValue = Numpy.array("[1, 2, 4]", "<i4");
                ((Action)(() => ByteContract.AssertSameBytes(a, differentValue)))
                    .Should().Throw<Exception>("differing values must differ in bytes — else the whole suite is vacuous");

                // same values, different WIDTH (i4 vs i8) differ in bytes.
                using PyObject differentWidth = Numpy.array("[1, 2, 3]", "<i8");
                ((Action)(() => ByteContract.AssertSameBytes(a, differentWidth)))
                    .Should().Throw<Exception>("different dtype widths must differ in bytes");

                // same values, different ENDIANNESS differ in bytes (proving the big-endian tests are real).
                using PyObject bigEndian = Numpy.array("[1, 2, 3]", ">i4");
                ((Action)(() => ByteContract.AssertSameBytes(a, bigEndian)))
                    .Should().Throw<Exception>("opposite byte order must differ in bytes");
            }
        }

        [TestMethod]
        public void Decode_EveryDtype_CopyIsByteExact()
        {
            foreach (var (typestr, tc, sample) in Dtypes)
                DecodeCopyByteExact($"np.array({sample}, dtype='{typestr}')", tc);
        }

        [TestMethod]
        public void Decode_EveryDtype_ViewIsByteExact()
        {
            foreach (var (typestr, tc, sample) in Dtypes)
                DecodeViewByteExact($"np.array({sample}, dtype='{typestr}')", tc);
        }

        // ==============================  decode: every layout  ==================================

        [TestMethod]
        public void Decode_EveryLayout_CopyIsByteExact()
        {
            // The copy path linearizes every layout to C-order — so it must equal the source's own C-order bytes.
            DecodeCopyByteExact("np.arange(12, dtype='<f8')", NPTypeCode.Double);                       // contiguous
            DecodeCopyByteExact("np.arange(12, dtype='<f8')[2:]", NPTypeCode.Double);                   // offset
            DecodeCopyByteExact("np.arange(12, dtype='<f8')[::2]", NPTypeCode.Double);                  // strided
            DecodeCopyByteExact("np.arange(12, dtype='<f8')[::-1]", NPTypeCode.Double);                 // reversed
            DecodeCopyByteExact("np.arange(12, dtype='<i4').reshape(3,4).T", NPTypeCode.Int32);         // transposed
            DecodeCopyByteExact("np.asfortranarray(np.arange(12, dtype='<i4').reshape(3,4))", NPTypeCode.Int32); // F-order
            DecodeCopyByteExact("np.broadcast_to(np.arange(3, dtype='<i8'), (4,3))", NPTypeCode.Int64); // broadcast
            DecodeCopyByteExact("np.arange(24, dtype='<f8').reshape(4,6)[1:3]", NPTypeCode.Double);     // row window
            DecodeCopyByteExact("np.arange(24, dtype='<f8').reshape(4,6)[:, 1]", NPTypeCode.Double);    // column
            DecodeCopyByteExact("np.array(7.5, dtype='<f8')", NPTypeCode.Double);                       // 0-d
            DecodeCopyByteExact("np.empty((0,3), dtype='<f4')", NPTypeCode.Single);                     // empty
        }

        [TestMethod]
        public void Decode_EveryViewableLayout_ViewIsByteExact()
        {
            // The view path reconstructs the exact strided layout — a byte-exact read proves it walks the
            // right elements (negative strides, transposes, broadcasts included).
            DecodeViewByteExact("np.arange(12, dtype='<f8')", NPTypeCode.Double);
            DecodeViewByteExact("np.arange(12, dtype='<f8')[2:]", NPTypeCode.Double);
            DecodeViewByteExact("np.arange(12, dtype='<f8')[::2]", NPTypeCode.Double);
            DecodeViewByteExact("np.arange(12, dtype='<f8')[::-1]", NPTypeCode.Double);
            DecodeViewByteExact("np.arange(12, dtype='<i4').reshape(3,4).T", NPTypeCode.Int32);
            DecodeViewByteExact("np.asfortranarray(np.arange(12, dtype='<i4').reshape(3,4))", NPTypeCode.Int32);
            DecodeViewByteExact("np.broadcast_to(np.arange(3, dtype='<i8'), (4,3))", NPTypeCode.Int64);
        }

        // ==============================  decode: big-endian byteswap  ===========================

        [TestMethod]
        public void Decode_BigEndian_CopyByteSwapsToNativeTwin()
        {
            // The COPY path byte-reverses each element — so a big-endian source must be byte-identical to
            // its little-endian TWIN (same values, native bytes), NOT to its own big-endian bytes.
            foreach (string code in new[] { "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16" })
                DecodeCopyByteExact(
                    $"np.array([1, 2, 3, 100], dtype='>{code}')", NDArrayPythonInterop.FromNumpyDtypeStr($"<{code}"),
                    $"np.array([1, 2, 3, 100], dtype='<{code}')");

            // complex128 big-endian reverses each 8-byte half independently.
            DecodeCopyByteExact("np.array([1+2j, -3-4j], dtype='>c16')", NPTypeCode.Complex, "np.array([1+2j, -3-4j], dtype='<c16')");
            // complex64 big-endian: widen AND swap — equal to native complex128.
            DecodeCopyByteExact("np.array([1+2j, -3-4j], dtype='>c8')", NPTypeCode.Complex, "np.array([1+2j, -3-4j], dtype='<c16')");
            // strided / transposed big-endian linearizes and swaps.
            DecodeCopyByteExact("np.arange(12, dtype='>i4').reshape(3,4).T", NPTypeCode.Int32, "np.arange(12, dtype='<i4').reshape(3,4).T");
        }

        [TestMethod]
        public void Decode_BigEndian_ViewIsDeclined()
        {
            // No native-endian zero-copy view of big-endian memory is possible.
            using (Gil())
            {
                using PyObject be = Numpy.array("[1, 2, 3]", ">i4");
                ((Action)(() => NDArrayPythonInterop.ToNDArrayView(be, allowReadonly: true)))
                    .Should().Throw<NotSupportedException>().WithMessage("*big-endian*");
            }
        }

        // ==============================  decode: complex64 widen / UCS-4  =======================

        [TestMethod]
        public void Decode_Complex64_WidensToComplex128_ByteExact()
        {
            DecodeCopyByteExact("np.array([1+2j, 3+4j, -5-6j], dtype='<c8')", NPTypeCode.Complex, "np.array([1+2j, 3+4j, -5-6j], dtype='<c16')");
            DecodeCopyByteExact("np.zeros(8, dtype='<c8')[::2]", NPTypeCode.Complex, "np.zeros(4, dtype='<c16')");   // strided c8
        }

        [TestMethod]
        public void Decode_Ucs4Text_NarrowsToCharUint16_ByteExact()
        {
            // numpy '<U1' holds one UCS-4 code point per element; NumSharp narrows to a 2-byte Char, which is
            // byte-identical to the code points taken as uint16 (BMP).
            DecodeCopyByteExact("np.array(['a', 'Z', '\\u05d0'], dtype='<U1')", NPTypeCode.Char,
                "np.array([ord('a'), ord('Z'), 0x05d0], dtype='<u2')");
        }

        // ==============================  decode: buffer exporters (non-numpy)  ==================

        [TestMethod]
        public void Decode_NonNumpyExporters_AreByteExact()
        {
            using (Gil())
            {
                // bytes -> Byte
                using (PyObject b = Numpy.eval("b'\\x01\\x02\\xfe\\xff'"))
                using (NDArray nd = b.ToNDArray())
                    ByteContract.AssertSameBytes(nd, b, "bytes decode byte-exact");

                // array.array('i') -> Int32, both copy and view
                using (PyObject arr = Numpy.eval("__import__('array').array('i', [1, -2, 3, -4])"))
                {
                    using (NDArray c = arr.ToNDArray()) ByteContract.AssertSameBytes(c, arr, "array.array copy");
                    using (NDArray v = arr.AsNDArray()) ByteContract.AssertSameBytes(v, arr, "array.array view");
                }

                // a strided memoryview of a bytearray -> Byte view, byte-exact over its own strided bytes
                using (PyObject mv = Numpy.eval("memoryview(bytearray(range(16)))[::2]"))
                using (NDArray v = mv.AsNDArray())
                    ByteContract.AssertSameBytes(v, mv, "strided memoryview view");

                // a ctypes array -> Int32 view
                using (PyObject ct = Numpy.eval("(__import__('ctypes').c_int32 * 4)(10, 20, 30, 40)"))
                using (NDArray v = ct.AsNDArray())
                    ByteContract.AssertSameBytes(v, ct, "ctypes array view");
            }
        }

        // ==============================  encode: every dtype / layout  ==========================

        [TestMethod]
        public void Encode_EveryDtype_IsByteExact()
        {
            foreach (var (typestr, tc, _) in Dtypes)
            {
                if (tc == NPTypeCode.Boolean)
                    EncodeByteExact(np.arange(4).astype(tc), typestr, "np.arange(4).astype('?')");
                else
                    EncodeByteExact(np.arange(5).astype(tc), typestr, $"np.arange(5).astype('{typestr}')");
            }

            // Char exports as uint16 (UTF-16 code units).
            EncodeByteExact(np.arange(97, 100).astype(NPTypeCode.Char), "<u2", "np.arange(97, 100).astype('<u2')");
        }

        [TestMethod]
        public void Encode_EveryLayout_KeepsStridesAndIsByteExact()
        {
            using (Gil())
            {
                // sliced / transposed / reversed views export as the SAME strided window: numpy's own strides,
                // and byte-identical logical data.
                NDArray b = np.arange(24).reshape(4, 6);
                (NDArray ns, string expr)[] cases =
                {
                    (b["1:3, ::2"], "np.arange(24).reshape(4,6)[1:3, ::2]"),
                    (b.T,           "np.arange(24).reshape(4,6).T"),
                    (b["::-1"],     "np.arange(24).reshape(4,6)[::-1]"),
                };
                foreach (var (ns, expr) in cases)
                {
                    using PyObject exported = ns.ToNumpy();
                    using PyObject expected = Numpy.eval(expr);
                    exported.strides_bytes().Should().Equal(expected.strides_bytes(), expr);
                    ByteContract.AssertSameBytes(exported, expected, expr);
                }

                // a broadcast view is read-only on both sides, and byte-identical.
                using (PyObject bc = np.broadcast_to(np.arange(3), new Shape(2, 3)).ToNumpy())
                using (PyObject exp = Numpy.eval("np.broadcast_to(np.arange(3), (2,3))"))
                {
                    bc.writeable().Should().BeFalse("a broadcast export is read-only, like NumSharp");
                    ByteContract.AssertSameBytes(bc, exp, "broadcast export");
                }
            }
        }

        // ==============================  encode: Decimal -> float64  ============================

        [TestMethod]
        public void Encode_Decimal_ConvertsToFloat64_ByteExact()
        {
            NDArray dec = np.array(new decimal[] { 1.5m, -2.5m, 3.25m, 0m }).astype(NPTypeCode.Decimal);
            EncodeByteExact(dec, "<f8", "np.array([1.5, -2.5, 3.25, 0.0], dtype='<f8')");
        }

        // ==============================  codec: modes are byte-exact  ===========================

        [TestMethod]
        public void Codec_AllDecodeModes_AreByteExact()
        {
            var auto = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Auto });
            var view = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
            var copy = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });

            using (Gil())
            {
                // A contiguous numpy array: all three modes yield byte-identical data (Auto/View share, Copy detaches).
                using PyObject a = Numpy.array("[10, 20, 30, 40]", "<i8");
                foreach (var (codec, name) in new[] { (auto, "Auto"), (view, "View"), (copy, "Copy") })
                {
                    codec.TryDecode(a, out NDArray nd).Should().BeTrue(name);
                    ByteContract.AssertSameBytes(nd, a, $"{name} decode is byte-exact");
                    nd.Dispose();
                }

                // A complex64 source: Auto/Copy widen (byte-exact vs c16), View declines.
                using PyObject c8 = Numpy.array("[1+2j, 3+4j]", "<c8");
                using PyObject c16 = c8.astype("<c16");
                auto.TryDecode(c8, out NDArray aWide).Should().BeTrue("Auto copies when a view is impossible");
                ByteContract.AssertSameBytes(aWide, c16, "Auto widens complex64 byte-exact");
                aWide.Dispose();

                view.TryDecode(c8, out NDArray vNone).Should().BeFalse("View declines complex64");
                vNone.Should().BeNull();
            }
        }

        // ==============================  codec: array-like decode (default on)  =================

        [TestMethod]
        public void Codec_ArrayLikeDecode_IsByteExactVsAsarray()
        {
            var codec = new NumpyCodec();   // defaults: DecodeArrayLike == true
            using (Gil())
            {
                foreach (string literal in new[] { "[1, 2, 3]", "(1.0, 2.5, -3.0)", "[[1, 2], [3, 4]]", "7" })
                {
                    using PyObject lit = Numpy.eval(literal);
                    codec.TryDecode(lit, out NDArray nd).Should().BeTrue(literal);
                    using PyObject expected = Numpy.asarray(lit);
                    ByteContract.AssertSameBytes(nd, expected, $"{literal} decodes byte-equal to np.asarray");
                    nd.Dispose();
                }
            }
        }

        // ==============================  round-trips: byte identity  ============================

        [TestMethod]
        public void RoundTrip_NumpyToNumSharpToNumpy_IsByteIdentical()
        {
            using (Gil())
            {
                foreach (string expr in new[]
                {
                    "np.arange(12, dtype='<f8').reshape(3,4)",
                    "np.arange(12, dtype='<i4').reshape(3,4).T",
                    "np.array([1+2j, 3+4j, -5-6j], dtype='<c16')",
                    "np.array([True, False, True], dtype='|b1')",
                })
                {
                    using PyObject src = Numpy.eval(expr);
                    using NDArray nd = src.ToNDArray();       // numpy -> NumSharp (copy)
                    using PyObject back = nd.ToNumpy();       // NumSharp -> numpy (view)
                    ByteContract.AssertSameBytes(back, src, $"round-trip byte identity: {expr}");
                }
            }
        }

        [TestMethod]
        public void SharedView_NumpyWriteIsVisibleInNumSharpBytes()
        {
            // The zero-copy import view shares memory: a write on the numpy side changes the NumSharp bytes.
            using (Gil())
            {
                using PyObject src = Numpy.array("[0, 0, 0, 0]", "<i4");
                using NDArray nd = src.AsNDArray();
                using (PyObject _ = src.InvokeMethod("__setitem__", 1L.ToPython(), 12345L.ToPython())) { }
                using PyObject expected = Numpy.array("[0, 12345, 0, 0]", "<i4");
                ByteContract.AssertSameBytes(nd, expected, "a numpy write reaches the shared NumSharp view");
            }
        }

        [TestMethod]
        public void SharedExport_NumSharpWriteIsVisibleInNumpyBytes()
        {
            // The reverse: a zero-copy EXPORT shares memory, so a NumSharp write changes the numpy bytes.
            NDArray ns = np.arange(4).astype(NPTypeCode.Int32);   // [0, 1, 2, 3]
            using (Gil())
            {
                using PyObject exported = ns.ToNumpy();
                WriteAt(ns, 999, 2);                              // NumSharp write
                using PyObject expected = Numpy.array("[0, 1, 999, 3]", "<i4");
                ByteContract.AssertSameBytes(exported, expected, "a NumSharp write reaches the shared numpy export");
            }
        }

        // ==============================  numpy scalars (0-d) / read-only / writable  ============

        [TestMethod]
        public void Decode_NumpyScalar_ZeroD_IsByteExact()
        {
            using (Gil())
            {
                foreach (var (expr, tc) in new[]
                {
                    ("np.float64(3.25)", NPTypeCode.Double),
                    ("np.int32(-7)", NPTypeCode.Int32),
                    ("np.complex128(1+2j)", NPTypeCode.Complex),
                    ("np.uint8(200)", NPTypeCode.Byte),
                })
                {
                    using PyObject s = Numpy.eval(expr);
                    using NDArray nd = s.ToNDArray();
                    nd.ndim.Should().Be(0, expr);
                    nd.typecode.Should().Be(tc, expr);
                    ByteContract.AssertSameBytes(nd, s, expr);
                }
            }
        }

        [TestMethod]
        public void Decode_ReadonlySource_IsNonWriteableView_ByteExact()
        {
            using (Gil())
            {
                // read-only numpy array: a view needs allowReadonly and comes back non-writeable, byte-exact.
                using PyObject ro = Numpy.evalStmts("r = np.arange(5, dtype='<f8'); r.flags.writeable = False", "r");
                ((Action)(() => NDArrayPythonInterop.ToNDArrayView(ro)))
                    .Should().Throw<InvalidOperationException>("a writable view of a read-only source is refused by default");
                using NDArray v = NDArrayPythonInterop.ToNDArrayView(ro, allowReadonly: true);
                v.Shape.IsWriteable.Should().BeFalse("numpy writeable=False carries across as a non-writeable view");
                ByteContract.AssertSameBytes(v, ro, "read-only numpy view is byte-exact");

                // bytes is inherently read-only: same contract.
                using PyObject b = Numpy.eval("b'\\x01\\x02\\x03\\x04'");
                using NDArray vb = b.AsNDArray(allowReadonly: true);
                vb.Shape.IsWriteable.Should().BeFalse();
                ByteContract.AssertSameBytes(vb, b, "read-only bytes view is byte-exact");
            }
        }

        [TestMethod]
        public void Decode_WritableBytearray_ViewWriteBack_IsByteExact()
        {
            using (Gil())
            {
                using PyObject ba = Numpy.eval("bytearray(b'\\x00\\x00\\x00\\x00')");
                using NDArray v = ba.AsNDArray();               // bytearray is writable -> writable view
                v.Shape.IsWriteable.Should().BeTrue();
                WriteAt(v, (byte)0xAB, 1);                       // NumSharp write reaches the bytearray
                using PyObject expected = Numpy.eval("bytearray(b'\\x00\\xab\\x00\\x00')");
                ByteContract.AssertSameBytes(v, expected, "the write-through view is byte-exact after a NumSharp write");
            }
        }

        // ==============================  encode modes / 2-byte wchar  ===========================

        [TestMethod]
        public void Codec_AllEncodeModes_AreByteExact()
        {
            var auto = new NumpyCodec(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.Auto });
            var view = new NumpyCodec(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.View });
            var copy = new NumpyCodec(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.Copy });
            NDArray ns = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);

            using (Gil())
            {
                using PyObject expected = Numpy.eval("np.arange(6, dtype='<f8').reshape(2,3)");
                foreach (var (codec, name) in new[] { (auto, "Auto"), (view, "View"), (copy, "Copy") })
                {
                    using PyObject p = codec.TryEncode(ns);
                    p.Should().NotBeNull(name);
                    ByteContract.AssertSameBytes(p, expected, $"{name} encode is byte-exact");
                }
            }
        }

        [TestMethod]
        public void Decode_TwoByteWchar_ViewsAsChar_ByteExact()
        {
            using (Gil())
            {
                long unit = Numpy.eval("__import__('array').array('u').itemsize").As<long>();
                if (unit != 2)
                    Assert.Inconclusive("wchar_t is 4 bytes on this platform (linux/macOS) — no zero-copy Char view");

                using PyObject ua = Numpy.eval("__import__('array').array('u', 'hi!')");
                using NDArray v = ua.AsNDArray();
                v.typecode.Should().Be(NPTypeCode.Char, "a 2-byte wchar buffer IS a UTF-16 Char buffer");
                using PyObject expected = Numpy.array("[ord('h'), ord('i'), ord('!')]", "<u2");
                ByteContract.AssertSameBytes(v, expected, "2-byte wchar views as Char, byte-exact vs its uint16 code units");
            }
        }
    }
}
