using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Maths
{
    /// <summary>
    /// np.rint — round-half-to-even, float-tier unary ufunc. All expectations verified
    /// against NumPy 2.4.2 (np.rint has only float/complex loops: e/f/d/g/F/D/G).
    /// </summary>
    [TestClass]
    public class np_rint_Test
    {
        // ---- dtype tiers (int/bool promote to float; floats/complex preserved) ----
        [TestMethod]
        public void Rint_DtypeTiers()
        {
            np.rint(np.array(new bool[] { true })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new sbyte[] { 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new byte[] { 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new short[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new ushort[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new int[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new uint[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new long[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new ulong[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new Half[] { (Half)1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new float[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new double[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new Complex[] { new(1, 2) })).typecode.Should().Be(NPTypeCode.Complex);
        }

        // ---- round half to even (banker's) ----
        [TestMethod]
        public void Rint_HalfToEven_Float64()
        {
            var r = np.rint(np.array(new double[] { 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.4, 2.6, -2.6 }));
            double[] exp = { 0, 2, 2, 4, 0, -2, -2, 2, 3, -3 };
            for (int i = 0; i < exp.Length; i++)
                r.GetDouble(i).Should().Be(exp[i], $"rint index {i}");
        }

        [TestMethod]
        public void Rint_HalfToEven_Float32()
        {
            var r = np.rint(np.array(new float[] { 0.5f, 1.5f, 2.5f, -2.5f, 2.6f }));
            r.typecode.Should().Be(NPTypeCode.Single);
            float[] exp = { 0, 2, 2, -2, 3 };
            for (int i = 0; i < exp.Length; i++)
                r.GetSingle(i).Should().Be(exp[i]);
        }

        // ---- NaN / Inf preserved ----
        [TestMethod]
        public void Rint_NanInf_Preserved()
        {
            var r = np.rint(np.array(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity }));
            double.IsNaN(r.GetDouble(0)).Should().BeTrue();
            double.IsPositiveInfinity(r.GetDouble(1)).Should().BeTrue();
            double.IsNegativeInfinity(r.GetDouble(2)).Should().BeTrue();
        }

        // ---- int input -> float tier, values are the integers as floats ----
        [TestMethod]
        public void Rint_Int8_ToFloat16_Values()
        {
            var r = np.rint(np.array(new sbyte[] { -3, 0, 5, 127 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetHalf(0)).Should().Be(-3);
            ((double)r.GetHalf(3)).Should().Be(127);
        }

        // ---- complex: rounds real & imag separately (half-to-even) ----
        [TestMethod]
        public void Rint_Complex_RoundsBothParts()
        {
            var r = np.rint(np.array(new Complex[] { new(1.5, 2.5), new(0.5, -1.5), new(-2.5, 0.5) }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetComplex(0).Should().Be(new Complex(2, 2));
            r.GetComplex(1).Should().Be(new Complex(0, -2));
            r.GetComplex(2).Should().Be(new Complex(-2, 0));
        }

        // ---- decimal (NumSharp extension): preserved, half-to-even ----
        [TestMethod]
        public void Rint_Decimal_Preserved()
        {
            var r = np.rint(np.array(new decimal[] { 0.5m, 1.5m, 2.5m, -2.5m, 2.6m }));
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.GetDecimal(0).Should().Be(0m);
            r.GetDecimal(1).Should().Be(2m);
            r.GetDecimal(2).Should().Be(2m);
            r.GetDecimal(4).Should().Be(3m);
        }

        // ---- layouts: strided / negative-stride / broadcast / transpose / empty / scalar ----
        [TestMethod]
        public void Rint_StridedView()
        {
            var b = np.array(new double[] { 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 });
            var r = np.rint(b["::2"]); // [0.5,2.5,4.5] -> [0,2,4]
            r.GetDouble(0).Should().Be(0);
            r.GetDouble(1).Should().Be(2);
            r.GetDouble(2).Should().Be(4);
        }

        [TestMethod]
        public void Rint_NegativeStrideView()
        {
            var b = np.array(new double[] { 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 });
            var r = np.rint(b["::-1"]); // [5.5,4.5,3.5,2.5,1.5,0.5] -> [6,4,4,2,2,0]
            r.GetDouble(0).Should().Be(6);
            r.GetDouble(5).Should().Be(0);
        }

        [TestMethod]
        public void Rint_Broadcast()
        {
            var bc = np.array(new double[] { 0.5, 1.5, 2.5 }).reshape(1, 3);
            var r = np.rint(np.broadcast_to(bc, new Shape(2, 3)));
            r.shape.Should().Equal(2, 3);
            r.GetDouble(0, 0).Should().Be(0);
            r.GetDouble(1, 1).Should().Be(2);
        }

        [TestMethod]
        public void Rint_Empty_And_Scalar()
        {
            var e = np.rint(np.array(new double[] { }));
            e.size.Should().Be(0);
            e.typecode.Should().Be(NPTypeCode.Double);

            np.rint(NDArray.Scalar(2.5)).GetDouble(0).Should().Be(2.0);
        }

        // ---- ufunc out= / where= / dtype= ----
        [TestMethod]
        public void Rint_Out_ReturnsSameInstanceAndFills()
        {
            var outArr = np.zeros(new Shape(3), NPTypeCode.Double);
            var r = np.rint(np.array(new double[] { 0.5, 1.5, 2.5 }), @out: outArr);
            ReferenceEquals(r, outArr).Should().BeTrue();
            outArr.GetDouble(0).Should().Be(0);
            outArr.GetDouble(1).Should().Be(2);
            outArr.GetDouble(2).Should().Be(2);
        }

        [TestMethod]
        public void Rint_DtypeOverride_RunsLoopInThatDtype()
        {
            var r = np.rint(np.array(new double[] { 2.5 }), dtype: NPTypeCode.Single);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.GetSingle(0).Should().Be(2.0f);
        }

        [TestMethod]
        public void Rint_IntegerDtype_RaisesNoLoop()
        {
            // rint has no integer loop; dtype=int must raise (NumPy: "No loop matching ...").
            Action act = () => np.rint(np.array(new double[] { 2.5 }), dtype: NPTypeCode.Int32);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Rint_Where_KeepsPriorOutContents()
        {
            var outW = np.full(new Shape(3), 9.0, NPTypeCode.Double);
            var mask = np.array(new bool[] { true, false, true });
            np.rint(np.array(new double[] { 0.5, 1.5, 2.5 }), @out: outW, where: mask);
            outW.GetDouble(0).Should().Be(0);
            outW.GetDouble(1).Should().Be(9.0, "masked-off slot keeps prior out content");
            outW.GetDouble(2).Should().Be(2);
        }

        // ---- np.around(complex) now also works (bonus from the shared Round kernel) ----
        [TestMethod]
        public void Around_Complex_NowSupported()
        {
            var r = np.around(np.array(new Complex[] { new(1.5, 2.5) }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetComplex(0).Should().Be(new Complex(2, 2));
        }

        // =====================================================================
        //  Extended parity coverage (all values verified against NumPy 2.4.2)
        // =====================================================================

        // ---- signed zero is preserved (np.rint(-0.4) -> -0.0, signbit True) ----
        [TestMethod]
        public void Rint_SignedZero_Preserved_Float64()
        {
            var r = np.rint(np.array(new double[] { -0.0, -0.4, -0.5, 0.4, 0.5 }));
            // values are all zero...
            for (int i = 0; i < 5; i++) r.GetDouble(i).Should().Be(0.0);
            // ...but the SIGN of zero must match NumPy exactly.
            double.IsNegative(r.GetDouble(0)).Should().BeTrue("rint(-0.0) keeps -0.0");
            double.IsNegative(r.GetDouble(1)).Should().BeTrue("rint(-0.4) = -0.0");
            double.IsNegative(r.GetDouble(2)).Should().BeTrue("rint(-0.5) = -0.0 (half to even)");
            double.IsNegative(r.GetDouble(3)).Should().BeFalse("rint(0.4) = +0.0");
            double.IsNegative(r.GetDouble(4)).Should().BeFalse("rint(0.5) = +0.0");
        }

        [TestMethod]
        public void Rint_SignedZero_Preserved_Float32()
        {
            var r = np.rint(np.array(new float[] { -0.4f, 0.4f }));
            float.IsNegative(r.GetSingle(0)).Should().BeTrue();
            float.IsNegative(r.GetSingle(1)).Should().BeFalse();
        }

        [TestMethod]
        public void Rint_Subnormal_UnderflowsToSignedZero()
        {
            var r = np.rint(np.array(new double[] { 1e-300, -1e-300 }));
            r.GetDouble(0).Should().Be(0.0);
            double.IsNegative(r.GetDouble(0)).Should().BeFalse();
            r.GetDouble(1).Should().Be(0.0);
            double.IsNegative(r.GetDouble(1)).Should().BeTrue("rint(-1e-300) = -0.0");
        }

        // ---- Half (float16) rounds half-to-even and stays float16 ----
        [TestMethod]
        public void Rint_Half_Values()
        {
            var r = np.rint(np.array(new Half[] { (Half)0.5f, (Half)1.5f, (Half)2.5f, (Half)3.5f, (Half)(-2.5f), (Half)2.6f }));
            r.typecode.Should().Be(NPTypeCode.Half);
            double[] exp = { 0, 2, 2, 4, -2, 3 };
            for (int i = 0; i < exp.Length; i++)
                ((double)r.GetHalf(i)).Should().Be(exp[i]);
        }

        // ---- Char (NumSharp-specific, uint16 proxy) -> float32 ----
        [TestMethod]
        public void Rint_Char_ToFloat32()
        {
            var r = np.rint(np.array(new char[] { (char)0, (char)65, (char)255, (char)1000 }));
            r.typecode.Should().Be(NPTypeCode.Single, "char rides the uint16 tier -> float32");
            r.GetSingle(0).Should().Be(0);
            r.GetSingle(1).Should().Be(65);
            r.GetSingle(3).Should().Be(1000);
        }

        // ---- every one of the 15 supported dtypes produces the NumPy-parity result dtype ----
        [TestMethod]
        public void Rint_All15Dtypes_ResultDtype()
        {
            np.rint(np.array(new bool[] { true })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new byte[] { 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new sbyte[] { 1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new short[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new ushort[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new int[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new uint[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new long[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new ulong[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new char[] { 'a' })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new Half[] { (Half)1 })).typecode.Should().Be(NPTypeCode.Half);
            np.rint(np.array(new float[] { 1 })).typecode.Should().Be(NPTypeCode.Single);
            np.rint(np.array(new double[] { 1 })).typecode.Should().Be(NPTypeCode.Double);
            np.rint(np.array(new decimal[] { 1 })).typecode.Should().Be(NPTypeCode.Decimal);
            np.rint(np.array(new Complex[] { new(1, 2) })).typecode.Should().Be(NPTypeCode.Complex);
        }

        // ---- higher-rank (2D) value parity ----
        [TestMethod]
        public void Rint_2D_Values()
        {
            var r = np.rint(np.array(new double[] { 0.5, 1.5, 2.5, -2.5 }).reshape(2, 2));
            r.shape.Should().Equal(2, 2);
            r.GetDouble(0, 0).Should().Be(0);
            r.GetDouble(0, 1).Should().Be(2);
            r.GetDouble(1, 0).Should().Be(2);
            r.GetDouble(1, 1).Should().Be(-2);
        }

        // ---- F-contiguous input ----
        [TestMethod]
        public void Rint_FContiguous()
        {
            var f = np.array(new double[] { 0.5, 1.5, 2.5, 3.5 }).reshape(2, 2).copy(order: 'F');
            var r = np.rint(f);
            r.GetDouble(0, 0).Should().Be(0);
            r.GetDouble(0, 1).Should().Be(2);
            r.GetDouble(1, 0).Should().Be(2);
            r.GetDouble(1, 1).Should().Be(4);
        }

        // ---- out= with a same_kind float narrowing cast (f64 loop -> f32 out) ----
        [TestMethod]
        public void Rint_Out_SameKindNarrowingCast()
        {
            var o = np.zeros(new Shape(3), NPTypeCode.Single);
            var r = np.rint(np.array(new double[] { 0.5, 1.5, 2.5 }), @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.typecode.Should().Be(NPTypeCode.Single);
            o.GetSingle(0).Should().Be(0);
            o.GetSingle(1).Should().Be(2);
            o.GetSingle(2).Should().Be(2);
        }

        // ---- out= that requires a non-same_kind cast (float loop -> int out) must raise ----
        [TestMethod]
        public void Rint_Out_FloatToInt_Raises()
        {
            var o = np.zeros(new Shape(2), NPTypeCode.Int32);
            Action act = () => np.rint(np.array(new double[] { 0.5, 1.5 }), @out: o);
            act.Should().Throw<Exception>("float rint result cannot same_kind-cast into an int out");
        }

        // ---- in-place: out= aliases the input ----
        [TestMethod]
        public void Rint_InPlace_OutAliasesInput()
        {
            var a = np.array(new double[] { 0.5, 1.5, 2.5, 3.5 });
            var r = np.rint(a, @out: a);
            ReferenceEquals(r, a).Should().BeTrue();
            a.GetDouble(0).Should().Be(0);
            a.GetDouble(1).Should().Be(2);
            a.GetDouble(2).Should().Be(2);
            a.GetDouble(3).Should().Be(4);
        }

        // ---- where= broadcasts (2D out, row mask) ----
        [TestMethod]
        public void Rint_Where_Broadcasts()
        {
            var outW = np.full(new Shape(2, 2), 9.0, NPTypeCode.Double);
            var mask = np.array(new bool[] { true, false });   // (2,) broadcasts across rows
            np.rint(np.array(new double[] { 0.5, 1.5, 2.5, 3.5 }).reshape(2, 2), @out: outW, where: mask);
            outW.GetDouble(0, 0).Should().Be(0);
            outW.GetDouble(0, 1).Should().Be(9.0, "column 1 masked off");
            outW.GetDouble(1, 0).Should().Be(2);
            outW.GetDouble(1, 1).Should().Be(9.0, "column 1 masked off");
        }

        // ---- metamorphic invariants (oracle-free; hold for round-half-to-even) ----
        [TestMethod]
        public void Rint_Idempotent()
        {
            var x = np.array(new double[] { 0.5, 1.4, 2.6, -3.5, 2.5, -2.5, 100.49 });
            var once = np.rint(x);
            var twice = np.rint(once);
            for (int i = 0; i < x.size; i++)
                twice.GetDouble(i).Should().Be(once.GetDouble(i), "rint is idempotent");
        }

        [TestMethod]
        public void Rint_OddFunction()
        {
            // Round-half-to-even is symmetric about 0: rint(-x) == -rint(x).
            var x = np.array(new double[] { 0.5, 1.4, 2.6, 3.5, 2.5, 0.1 });
            var negThenRint = np.rint(-x);
            var rintThenNeg = -np.rint(x);
            for (int i = 0; i < x.size; i++)
                negThenRint.GetDouble(i).Should().Be(rintThenNeg.GetDouble(i));
        }

        // ---- large already-integer magnitudes are unchanged ----
        [TestMethod]
        public void Rint_LargeIntegralMagnitudes_Unchanged()
        {
            var r = np.rint(np.array(new double[] { 9007199254740992.0 /*2^53*/, 1e16, -1e16 }));
            r.GetDouble(0).Should().Be(9007199254740992.0);
            r.GetDouble(1).Should().Be(1e16);
            r.GetDouble(2).Should().Be(-1e16);
        }
    }
}
