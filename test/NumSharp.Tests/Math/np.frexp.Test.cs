using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.frexp / np.ldexp — the mantissa/exponent decomposition family (inverses of each other).
    /// All expected values verified against NumPy 2.4.2 (win-amd64). frexp is a portable IEEE
    /// bit-fiddle and ldexp is Math.ScaleB (== C ldexp), so results are bit-identical to NumPy on
    /// every platform — EXCEPT frexp's exponent for ±inf/NaN, which follows the scalar C-runtime
    /// (npy_frexp): -1 on win-amd64 (the corpus/oracle host), reproduced by NumSharp everywhere.
    /// </summary>
    [TestClass]
    public class FrexpLdexpTests
    {
        // ------------------------------------------------------------------ frexp values
        [TestMethod]
        public void Frexp_Double_Values()
        {
            var (m, e) = np.frexp(np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 8.0 }));
            m.typecode.Should().Be(NPTypeCode.Double);
            e.typecode.Should().Be(NPTypeCode.Int32);        // exponent is ALWAYS int32
            m.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, 0.5, 0.75, 0.5, 0.5 });
            e.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 2, 2, 3, 4 });
        }

        [TestMethod]
        public void Frexp_ReconstructsInput()
        {
            // x == mantissa * 2^exponent, element-wise.
            var x = np.array(new double[] { 0.75, 123456.789, -0.001, 8.0 });
            var (m, e) = np.frexp(x);
            for (int i = 0; i < 4; i++)
                (m.GetAtIndex<double>(i) * System.Math.Pow(2, e.GetAtIndex<int>(i)))
                    .Should().BeApproximately(x.GetAtIndex<double>(i), 1e-12);
        }

        [TestMethod]
        public void Frexp_Double_Specials()
        {
            var (m, e) = np.frexp(np.array(new double[]
                { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN }));
            // Mantissa is the input itself for the special values; sign of zero is preserved.
            m.GetAtIndex<double>(0).Should().Be(0.0);
            double.IsNegative(m.GetAtIndex<double>(1)).Should().BeTrue();     // -0.0 keeps its sign
            double.IsPositiveInfinity(m.GetAtIndex<double>(2)).Should().BeTrue();
            double.IsNegativeInfinity(m.GetAtIndex<double>(3)).Should().BeTrue();
            double.IsNaN(m.GetAtIndex<double>(4)).Should().BeTrue();
            // Exponent: 0 for ±0, but -1 for ±inf/NaN (the win-amd64 C-runtime value).
            e.ToArray<int>().Should().BeEquivalentTo(new[] { 0, 0, -1, -1, -1 });
        }

        [TestMethod]
        public void Frexp_SignalingNaN_IsQuieted()
        {
            // NumPy's scalar frexp quiets a signalling NaN (sets the mantissa MSB): 0x7FF4… -> 0x7FFC….
            double sNaN = BitConverter.Int64BitsToDouble(0x7FF4000000000001L);
            var (m, _) = np.frexp(np.array(new[] { sNaN }));
            ((ulong)BitConverter.DoubleToInt64Bits(m.GetAtIndex<double>(0))).Should().Be(0x7FFC000000000001UL);
        }

        [TestMethod]
        public void Frexp_Subnormal_Double()
        {
            var (m, e) = np.frexp(np.array(new[] { 5e-324, 2.2250738585072014e-308 }));
            m.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, 0.5 });
            e.ToArray<int>().Should().BeEquivalentTo(new[] { -1073, -1021 });
        }

        [TestMethod]
        public void Frexp_Single_And_Half_PreserveMantissaDtype()
        {
            var (mf, ef) = np.frexp(np.array(new float[] { 1.0f, 8.0f }));
            mf.typecode.Should().Be(NPTypeCode.Single);
            ef.typecode.Should().Be(NPTypeCode.Int32);
            mf.ToArray<float>().Should().BeEquivalentTo(new float[] { 0.5f, 0.5f });

            var (mh, eh) = np.frexp(np.array(new Half[] { (Half)1.0f, (Half)8.0f }));
            mh.typecode.Should().Be(NPTypeCode.Half);      // mantissa keeps the Half dtype
            eh.typecode.Should().Be(NPTypeCode.Int32);
            eh.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 4 });
        }

        [TestMethod]
        public void Frexp_IntegerTierPromotion()
        {
            // bool/int8/uint8 -> float16, int16/uint16 -> float32, int32+ -> float64 (exponent int32).
            np.frexp(np.array(new bool[] { true }).astype(np.@bool)).Mantissa.typecode.Should().Be(NPTypeCode.Half);
            np.frexp(np.array(new sbyte[] { 4 })).Mantissa.typecode.Should().Be(NPTypeCode.Half);
            np.frexp(np.array(new short[] { 4 })).Mantissa.typecode.Should().Be(NPTypeCode.Single);
            var (m, e) = np.frexp(np.array(new long[] { 1024 }));
            m.typecode.Should().Be(NPTypeCode.Double);
            m.GetAtIndex<double>(0).Should().Be(0.5);
            e.GetAtIndex<int>(0).Should().Be(11);
        }

        [TestMethod]
        public void Frexp_Scalar_Returns0d()
        {
            var (m, e) = np.frexp(NDArray.Scalar(12.0f));
            m.ndim.Should().Be(0);
            e.ndim.Should().Be(0);
            m.GetAtIndex<float>(0).Should().Be(0.75f);
            e.GetAtIndex<int>(0).Should().Be(4);
        }

        [TestMethod]
        public void Frexp_Complex_Throws()
        {
            Action act = () => np.frexp(np.array(new Complex[] { new(1, 2) }));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("not supported for the input types"));
        }

        // ------------------------------------------------------------------ ldexp values
        [TestMethod]
        public void Ldexp_Double_Values()
        {
            var r = np.ldexp(np.array(new double[] { 1.0, 1.0, 1.0 }), np.array(new[] { -1, 0, 1 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, 1.0, 2.0 });
        }

        [TestMethod]
        public void Ldexp_ScalarExponent()
        {
            // np.ldexp(array, k) — the common scalar-exponent shape.
            var r = np.ldexp(np.array(new double[] { 1.0, 2.0, 4.0 }), (NDArray)3);
            r.ToArray<double>().Should().BeEquivalentTo(new double[] { 8.0, 16.0, 32.0 });
        }

        [TestMethod]
        public void Ldexp_Specials()
        {
            var r = np.ldexp(
                np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0.0, -0.0 }),
                np.array(new[] { 1, 1, 1, 5, 5 }));
            double.IsPositiveInfinity(r.GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNegativeInfinity(r.GetAtIndex<double>(1)).Should().BeTrue();
            double.IsNaN(r.GetAtIndex<double>(2)).Should().BeTrue();
            r.GetAtIndex<double>(3).Should().Be(0.0);
            double.IsNegative(r.GetAtIndex<double>(4)).Should().BeTrue();   // -0.0 * 2^5 keeps its sign
        }

        [TestMethod]
        public void Ldexp_Int64ExponentClamps_ToInfAndZero()
        {
            var r = np.ldexp(np.array(new double[] { 1.0, 1.0 }), np.array(new long[] { 1L << 40, -(1L << 40) }));
            double.IsPositiveInfinity(r.GetAtIndex<double>(0)).Should().BeTrue();   // huge +exp overflows
            r.GetAtIndex<double>(1).Should().Be(0.0);                               // huge -exp underflows
        }

        [TestMethod]
        public void Ldexp_Broadcasts()
        {
            var r = np.ldexp(np.array(new double[] { 1.0, 2.0, 4.0 }).reshape(3, 1), np.array(new[] { 0, 1, 2, 3 }));
            r.shape.Should().BeEquivalentTo(new long[] { 3, 4 });
            r.GetAtIndex<double>(0).Should().Be(1.0);   // 1 * 2^0
            r.GetAtIndex<double>(7).Should().Be(16.0);  // 2 * 2^3 (row 1, col 3)
        }

        [TestMethod]
        public void Ldexp_XDtypePromotesLikeFrexp_ExponentDoesNotWiden()
        {
            np.ldexp(np.array(new Half[] { (Half)1f }), np.array(new[] { 2 })).typecode.Should().Be(NPTypeCode.Half);
            np.ldexp(np.array(new float[] { 1f }), np.array(new[] { 2 })).typecode.Should().Be(NPTypeCode.Single);
            np.ldexp(np.array(new int[] { 1 }), np.array(new[] { 2 })).typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Ldexp_FloatExponent_Throws()
        {
            Action act = () => np.ldexp(np.array(new double[] { 1.0 }), np.array(new double[] { 2.0 }));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("not supported for the input types"));
        }

        [TestMethod]
        public void Ldexp_UInt64Exponent_Throws()
        {
            Action act = () => np.ldexp(np.array(new double[] { 1.0 }), np.array(new ulong[] { 2 }));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("not supported for the input types"));
        }

        [TestMethod]
        public void Ldexp_Complex_Throws()
        {
            Action act = () => np.ldexp(np.array(new Complex[] { new(1, 2) }), np.array(new[] { 1 }));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("not supported for the input types"));
        }

        // ------------------------------------------------------------------ out= / where=
        [TestMethod]
        public void Frexp_Out_WritesBothOutputs_AndReturnsThem()
        {
            var a = np.array(new double[] { 1.0, 2.0, 4.0, 8.0 });
            var m = np.empty(new Shape(4));
            var e = np.empty(new Shape(4)).astype(np.int32);
            var (rm, re) = np.frexp(a, m, e);
            ReferenceEquals(rm, m).Should().BeTrue();     // the supplied out arrays are returned
            ReferenceEquals(re, e).Should().BeTrue();
            m.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, 0.5, 0.5, 0.5 });
            e.ToArray<int>().Should().BeEquivalentTo(new[] { 1, 2, 3, 4 });
        }

        [TestMethod]
        public void Frexp_Where_MasksBothOutputs_MaskedOffKeepPrior()
        {
            var a = np.array(new double[] { 1.0, 2.0, 4.0, 8.0 });
            var m = np.full(new Shape(4), -9.0);
            var e = np.full(new Shape(4), -9).astype(np.int32);
            np.frexp(a, m, e, np.array(new bool[] { true, false, true, false }));
            m.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, -9.0, 0.5, -9.0 });
            e.ToArray<int>().Should().BeEquivalentTo(new[] { 1, -9, 3, -9 });     // masked-off keep prior contents
        }

        [TestMethod]
        public void Frexp_Out_SameKindUpcast_MantF64_ExpI64()
        {
            var a = np.array(new float[] { 1.0f, 2.0f });
            var m = np.empty(new Shape(2));                 // float64 out from a float32 mantissa loop
            var e = np.empty(new Shape(2)).astype(np.int64); // int64 out from the int32 exponent loop
            np.frexp(a, m, e);
            m.ToArray<double>().Should().BeEquivalentTo(new double[] { 0.5, 0.5 });
            e.ToArray<long>().Should().BeEquivalentTo(new long[] { 1, 2 });
        }

        [TestMethod]
        public void Frexp_MantOutInt32_Throws_SameKind()
        {
            // A float mantissa cannot same_kind-cast into an int32 out (NumPy UFuncTypeError, verbatim shape).
            Action act = () => np.frexp(np.array(new double[] { 1, 2 }),
                np.empty(new Shape(2)).astype(np.int32), np.empty(new Shape(2)).astype(np.int32));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("Cannot cast ufunc 'frexp' output"));
        }

        [TestMethod]
        public void Frexp_WhereNonBool_Throws()
        {
            Action act = () => np.frexp(np.array(new double[] { 1, 2 }), np.empty(new Shape(2)),
                np.empty(new Shape(2)).astype(np.int32), np.array(new int[] { 1, 0 }));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("to dtype('bool') according to the rule 'safe'"));
        }

        [TestMethod]
        public void Ldexp_Out_WritesResult_AndReturnsIt()
        {
            var a = np.array(new double[] { 1.0, 2.0, 4.0, 8.0 });
            var o = np.empty(new Shape(4));
            var r = np.ldexp(a, np.array(new[] { 1, 1, 1, 1 }), o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.ToArray<double>().Should().BeEquivalentTo(new double[] { 2, 4, 8, 16 });
        }

        [TestMethod]
        public void Ldexp_Where_Masks_MaskedOffKeepPrior()
        {
            var o = np.full(new Shape(4), -9.0);
            np.ldexp(np.array(new double[] { 1.0, 2.0, 4.0, 8.0 }), np.array(new[] { 1, 1, 1, 1 }),
                o, np.array(new bool[] { true, false, true, false }));
            o.ToArray<double>().Should().BeEquivalentTo(new double[] { 2, -9, 8, -9 });
        }

        [TestMethod]
        public void Ldexp_OutInt32_Throws_SameKind()
        {
            Action act = () => np.ldexp(np.array(new double[] { 1, 2 }), np.array(new[] { 1, 1 }),
                np.empty(new Shape(2)).astype(np.int32));
            act.Should().Throw<Exception>().Where(ex => ex.Message.Contains("Cannot cast ufunc 'ldexp' output"));
        }
    }
}
