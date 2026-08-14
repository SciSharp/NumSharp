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
    ///     Value-level fidelity for the inputs a finite byte-contract sample deliberately skips —
    ///     NaN (sign + payload), ±inf, signed zero, subnormals, and the exact dtype extremes. The
    ///     crossing is a pure byte copy, so the contract is <b>bit preservation</b>: whatever bits the
    ///     source holds must survive intact, and a numpy → NumSharp → numpy round-trip must be
    ///     byte-identical regardless of NaN convention. Encode from the CLR is checked only for the
    ///     convention-independent specials (±inf / ±0 / subnormal / min / max share bits across .NET and
    ///     numpy); CLR-origin NaN is proven via a round-trip + numpy's own <c>isnan</c>, never by
    ///     asserting a specific NaN encoding.
    ///
    ///     <para>numpy is spoken through the established <c>Python.np.*</c> builders + the <c>PyObject</c>
    ///     extensions in <c>Pythonic.cs</c>; <c>np</c> stays NumSharp. Every crossing is compared with
    ///     <see cref="ByteContract"/> (canonical C-order bytes on both sides).</para>
    /// </summary>
    [TestClass]
    public class SpecialValueFidelityTests : InteropTestBase
    {
        // ==============================  float specials: decode + round-trip  ===================

        // The full float ladder per width: nan, +inf, -inf, +0, -0, smallest normal (tiny), smallest
        // subnormal (nextafter(0,1)), largest finite (max), most-negative finite (min). Built on the
        // numpy side so the exact bits are numpy's own; the interop must preserve every one.
        private const string F8Ladder =
            "np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.finfo(np.float64).tiny, " +
            "np.nextafter(0.0, 1.0), np.finfo(np.float64).max, np.finfo(np.float64).min], dtype='<f8')";
        private const string F4Ladder =
            "np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.finfo(np.float32).tiny, " +
            "np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max, np.finfo(np.float32).min], dtype='<f4')";
        private const string F2Ladder =
            "np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.finfo(np.float16).tiny, " +
            "np.nextafter(np.float16(0), np.float16(1)), np.finfo(np.float16).max, np.finfo(np.float16).min], dtype='<f2')";

        private static void DecodeAndRoundTrip_BitExact(string npExpr, NPTypeCode tc)
        {
            using (Py.GIL())
            {
                using PyObject src = Python.np.eval(npExpr);

                // copy path — a fresh owning array that must hold the identical bits...
                using (NDArray copy = src.ToNDArray())
                {
                    copy.typecode.Should().Be(tc, npExpr);
                    ByteContract.AssertSameBytes(copy, src, $"copy-decode preserves bits: {npExpr}");

                    using PyObject back = copy.ToNumpy();          // ...and re-exporting them is byte-identical.
                    ByteContract.AssertSameBytes(back, src, $"round-trip is bit-identical: {npExpr}");
                }

                // view path — the shared-memory decode must also present the identical bits.
                using NDArray view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: true);
                view.typecode.Should().Be(tc, npExpr);
                ByteContract.AssertSameBytes(view, src, $"view-decode preserves bits: {npExpr}");
            }
        }

        [TestMethod]
        public void Float64_Specials_DecodeAndRoundTrip_BitExact() => DecodeAndRoundTrip_BitExact(F8Ladder, NPTypeCode.Double);

        [TestMethod]
        public void Float32_Specials_DecodeAndRoundTrip_BitExact() => DecodeAndRoundTrip_BitExact(F4Ladder, NPTypeCode.Single);

        [TestMethod]
        public void Float16_Specials_DecodeAndRoundTrip_BitExact() => DecodeAndRoundTrip_BitExact(F2Ladder, NPTypeCode.Half);

        [TestMethod]
        public void Complex128_Specials_DecodeAndRoundTrip_BitExact()
            => DecodeAndRoundTrip_BitExact(
                "np.array([complex(np.nan, 1.0), complex(np.inf, -np.inf), complex(-0.0, 0.0), " +
                "complex(0.0, -0.0), complex(1.5, -2.5)], dtype='<c16')", NPTypeCode.Complex);

        // ==============================  signed zero: distinguishable both ways  ================

        [TestMethod]
        public void SignedZero_SurvivesBothDirections_AndNumpySeesTheSignBit()
        {
            using (Py.GIL())
            {
                // decode: numpy's -0.0 keeps its sign bit as a NumSharp double, and re-export is signbit-true.
                using PyObject negZeroSrc = Python.np.eval("np.array([0.0, -0.0], dtype='<f8')");
                using NDArray nd = negZeroSrc.ToNDArray();
                ByteContract.AssertSameBytes(nd, negZeroSrc, "-0.0 sign bit survives decode");

                // encode from the CLR: the .NET negative zero must be numpy-signbit-true on the far side.
                using PyObject exported = np.array(new[] { 0.0, -0.0 }).ToNumpy();
                Python.np.truthy("np.signbit(a)[1] and not np.signbit(a)[0]", ("a", exported))
                     .Should().BeTrue("CLR -0.0 exports as a signbit-set zero, +0.0 does not");
            }
        }

        // ==============================  encode from the CLR: convention-independent specials  =

        // ±inf / ±0 / smallest subnormal / max / lowest have identical bits in .NET and numpy, so a
        // CLR-built array of them is byte-exact against the numpy literal. NaN is excluded here (its
        // encoding differs by sign) and covered by the round-trip test below.
        private static void EncodeSpecials_ByteExact(NDArray ns, string expectedDtypeStr, string numpyExpr)
        {
            using (Py.GIL())
            {
                using PyObject expected = Python.np.eval(numpyExpr);

                using PyObject view = ns.ToNumpy();
                view.dtype_str().Should().Be(expectedDtypeStr, numpyExpr);
                ByteContract.AssertSameBytes(view, expected, $"ToNumpy(view) specials: {numpyExpr}");

                using PyObject copy = ns.ToNumpyCopy();
                copy.dtype_str().Should().Be(expectedDtypeStr, numpyExpr);
                ByteContract.AssertSameBytes(copy, expected, $"ToNumpyCopy specials: {numpyExpr}");
            }
        }

        [TestMethod]
        public void Encode_Float64_ClrSpecials_MatchNumpyLiterals()
            => EncodeSpecials_ByteExact(
                np.array(new[]
                {
                    double.PositiveInfinity, double.NegativeInfinity, 0.0, -0.0,
                    double.Epsilon, double.MaxValue, double.MinValue,
                }), "<f8",
                "np.array([np.inf, -np.inf, 0.0, -0.0, np.nextafter(0.0, 1.0), " +
                "np.finfo(np.float64).max, np.finfo(np.float64).min], dtype='<f8')");

        [TestMethod]
        public void Encode_Float32_ClrSpecials_MatchNumpyLiterals()
            => EncodeSpecials_ByteExact(
                np.array(new[]
                {
                    float.PositiveInfinity, float.NegativeInfinity, 0.0f, -0.0f,
                    float.Epsilon, float.MaxValue, float.MinValue,
                }), "<f4",
                "np.array([np.inf, -np.inf, 0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)), " +
                "np.finfo(np.float32).max, np.finfo(np.float32).min], dtype='<f4')");

        [TestMethod]
        public void Encode_Float16_ClrSpecials_MatchNumpyLiterals()
            => EncodeSpecials_ByteExact(
                np.array(new[]
                {
                    Half.PositiveInfinity, Half.NegativeInfinity, (Half)0.0, BitConverter.UInt16BitsToHalf(0x8000),
                    Half.Epsilon, Half.MaxValue, Half.MinValue,
                }), "<f2",
                "np.array([np.inf, -np.inf, 0.0, -0.0, np.nextafter(np.float16(0), np.float16(1)), " +
                "np.finfo(np.float16).max, np.finfo(np.float16).min], dtype='<f2')");

        [TestMethod]
        public void Encode_Complex128_ClrSpecials_MatchNumpyLiterals()
            => EncodeSpecials_ByteExact(
                np.array(new[]
                {
                    new Complex(double.PositiveInfinity, double.NegativeInfinity),
                    new Complex(-0.0, 0.0), new Complex(0.0, -0.0), new Complex(1.5, -2.5),
                }), "<c16",
                "np.array([complex(np.inf, -np.inf), complex(-0.0, 0.0), complex(0.0, -0.0), " +
                "complex(1.5, -2.5)], dtype='<c16')");

        // ==============================  CLR-origin NaN: round-trip + numpy semantics  =========

        [TestMethod]
        public void Encode_ClrNaN_IsNaNToNumpy_AndRoundTripsBitIdentical()
        {
            using (Py.GIL())
            {
                // A .NET NaN has a different sign bit than numpy's canonical np.nan, so we do NOT assert a
                // specific encoding — we assert numpy recognizes it as NaN, and that the exact CLR bits
                // survive a full round-trip (the interop copies bits, it does not canonicalize them).
                NDArray src = np.array(new[] { double.NaN, 1.0, double.NaN });
                byte[] before = ByteContract.NsBytes(src);

                using PyObject exported = src.ToNumpy();
                Python.np.truthy("np.isnan(a[0]) and np.isnan(a[2]) and a[1] == 1.0", ("a", exported))
                     .Should().BeTrue("numpy sees the exported CLR NaNs as NaN");

                using NDArray back = exported.ToNDArray();
                ByteContract.NsBytes(back).Should().Equal(before, "CLR NaN bits survive NumSharp→numpy→NumSharp unchanged");
            }
        }

        [TestMethod]
        public void Decode_ComplexNaN_Component_NumpyAndNumSharpAgree()
        {
            using (Py.GIL())
            {
                using PyObject src = Python.np.eval("np.array([complex(1.0, np.nan), complex(np.nan, 2.0)], dtype='<c16')");
                using NDArray nd = src.ToNDArray();
                ByteContract.AssertSameBytes(nd, src, "complex NaN components preserved bit-exact");

                Complex c0 = nd.GetAtIndex<Complex>(0);
                Complex c1 = nd.GetAtIndex<Complex>(1);
                double.IsNaN(c0.Imaginary).Should().BeTrue();
                (c0.Real == 1.0).Should().BeTrue();
                double.IsNaN(c1.Real).Should().BeTrue();
                (c1.Imaginary == 2.0).Should().BeTrue();
            }
        }

        // ==============================  integer extremes: round-trip both ways  ================

        // Byte-exact DECODE of int extremes is already covered by CodecByteContractTests' dtype sweep; here
        // we pin the complementary property — a full numpy→NumSharp→numpy round-trip is byte-identical for
        // every signed/unsigned width at its exact min/max/zero/-1 boundaries.
        [TestMethod]
        public void IntegerExtremes_RoundTrip_BitIdentical()
        {
            (string typestr, string sample)[] cases =
            {
                ("|u1", "[0, 1, 255]"),
                ("|i1", "[0, -1, 127, -128]"),
                ("<u2", "[0, 1, 65535]"),
                ("<i2", "[0, -1, 32767, -32768]"),
                ("<u4", "[0, 1, 4294967295]"),
                ("<i4", "[0, -1, 2147483647, -2147483648]"),
                ("<u8", "[0, 1, 18446744073709551615]"),
                ("<i8", "[0, -1, 9223372036854775807, -9223372036854775808]"),
            };

            using (Py.GIL())
            {
                foreach (var (typestr, sample) in cases)
                {
                    using PyObject src = Python.np.array(sample, typestr);
                    using NDArray nd = src.ToNDArray();
                    using PyObject back = nd.ToNumpy();
                    ByteContract.AssertSameBytes(back, src, $"int extreme round-trip: {typestr} {sample}");
                }
            }
        }

        // ==============================  Char / UCS-4 BMP boundaries + non-BMP rejection  =======

        [TestMethod]
        public void Char_Uint16_BmpBoundaries_ByteExact()
        {
            // numpy has no char dtype; a C# char is a 2-byte UTF-16 code unit == uint16. Every BMP
            // boundary must decode byte-identically as Char and export back as <u2.
            using (Py.GIL())
            {
                using PyObject src = Python.np.eval("np.array([0x0000, 0x0041, 0x007F, 0x0080, 0x07FF, 0x0800, 0xD7FF, 0xE000, 0xFFFF], dtype='<u2')");
                using NDArray nd = src.ToNDArray();
                nd.typecode.Should().Be(NPTypeCode.UInt16, "raw <u2 decodes to UInt16 (Char is opt-in via the text path)");

                using NDArray asChar = nd.astype(NPTypeCode.Char);
                using PyObject exported = asChar.ToNumpy();
                exported.dtype_str().Should().Be("<u2", "Char exports as uint16");
                ByteContract.AssertSameBytes(asChar, src, "Char BMP code units are byte-identical to <u2");
            }
        }

        [TestMethod]
        public void Ucs4Text_BmpBoundaries_NarrowToChar_ByteExact()
        {
            using (Py.GIL())
            {
                // '<U1' holds one UCS-4 code point per element; the copy path narrows each to a UTF-16 Char.
                using PyObject src = Python.np.eval("np.array(['\\u0000', 'A', '\\u007f', '\\u0800', '\\ud7ff', '\\ue000', '\\uffff'], dtype='<U1')");
                using NDArray nd = src.ToNDArray();
                nd.typecode.Should().Be(NPTypeCode.Char, "UCS-4 text narrows to Char");

                using PyObject expected = Python.np.eval("np.array([0x0000, 0x0041, 0x007f, 0x0800, 0xd7ff, 0xe000, 0xffff], dtype='<u2')");
                ByteContract.AssertSameBytes(nd, expected, "BMP UCS-4 narrows to byte-identical uint16");
            }
        }

        [TestMethod]
        public void Ucs4Text_NonBmp_IsRejected_NotSilentlyMangled()
        {
            using (Py.GIL())
            {
                // U+10000 needs a surrogate pair — a single System.Char cannot hold it, so the narrow throws.
                using PyObject src = Python.np.eval("np.array(['\\U00010000'], dtype='<U1')");
                ((Action)(() => { using var _ = src.ToNDArray(); }))
                    .Should().Throw<NotSupportedException>().WithMessage("*non-BMP*");
            }
        }
    }
}
