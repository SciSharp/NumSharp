using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.round_ / np.around with the <c>decimals</c> argument — a port of NumPy 2.4.2's
    /// <c>PyArray_Round</c> (<c>op2(rint(op1(x, 10^|decimals|)), 10^|decimals|)</c> at the input precision).
    /// Every expectation was probed against NumPy 2.4.2 (bit-exact — the same values the <c>rounding</c>
    /// differential-fuzz tier gates). These cover the three former bugs the port fixed: negative decimals
    /// (previously threw via <c>Math.Round</c>'s 0..15 digit limit), Complex with decimals!=0 (previously a
    /// silent no-op), and bool dtype at decimals==0 (previously float64, must be float16).
    /// </summary>
    [TestClass]
    public class np_round_decimals_Test
    {
        private static ushort HalfBits(NDArray a, int i) => BitConverter.HalfToUInt16Bits(a.GetAtIndex<Half>(i));

        // ---- dtype resolution: np.round PRESERVES the input dtype at every decimals (probed 2.4.2) ----
        [TestMethod]
        public void Round_PreservesDtype_AtEveryDecimals()
        {
            foreach (var dec in new[] { -2, -1, 0, 1, 2 })
            {
                np.round_(np.array(new sbyte[] { 1 }), dec).typecode.Should().Be(NPTypeCode.SByte, $"int8 dec={dec}");
                np.round_(np.array(new int[] { 1 }), dec).typecode.Should().Be(NPTypeCode.Int32, $"int32 dec={dec}");
                np.round_(np.array(new long[] { 1 }), dec).typecode.Should().Be(NPTypeCode.Int64, $"int64 dec={dec}");
                np.round_(np.array(new ushort[] { 1 }), dec).typecode.Should().Be(NPTypeCode.UInt16, $"uint16 dec={dec}");
                np.round_(np.array(new Half[] { (Half)1 }), dec).typecode.Should().Be(NPTypeCode.Half, $"float16 dec={dec}");
                np.round_(np.array(new float[] { 1 }), dec).typecode.Should().Be(NPTypeCode.Single, $"float32 dec={dec}");
                np.round_(np.array(new double[] { 1 }), dec).typecode.Should().Be(NPTypeCode.Double, $"float64 dec={dec}");
                np.round_(np.array(new Complex[] { new(1, 2) }), dec).typecode.Should().Be(NPTypeCode.Complex, $"complex dec={dec}");
            }
        }

        // ---- bool: dec==0 takes the rint float16 tier (np.round(bool)->float16); dec!=0 RAISES ----
        [TestMethod]
        public void Round_Bool_Decimals0_IsFloat16()
        {
            var r = np.round_(np.array(new bool[] { false, true }), 0);
            r.typecode.Should().Be(NPTypeCode.Half, "NumPy round(bool, 0) is the rint float-tier -> float16");
            r.GetAtIndex<Half>(0).Should().Be((Half)0);
            r.GetAtIndex<Half>(1).Should().Be((Half)1);
        }

        [TestMethod]
        public void Round_Bool_NonzeroDecimals_Raises()
        {
            Action pos = () => np.round_(np.array(new bool[] { false, true }), 1);
            pos.Should().Throw<NotSupportedException>()
               .WithMessage("Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool')*");
            Action neg = () => np.round_(np.array(new bool[] { false, true }), -1);
            neg.Should().Throw<NotSupportedException>()
               .WithMessage("Cannot cast ufunc 'divide' output from dtype('float64') to dtype('bool')*");
        }

        // ---- float64 positive/negative/zero decimals (banker's rounding, exact NumPy values) ----
        [TestMethod]
        public void Round_Float64_Values()
        {
            var a = new double[] { 0.5, 1.5, 2.5, -0.5, -1.5, 0.125, 0.135, 1.55, 2.45, 2.675, 125.0, 135.0 };
            // dec=1: 1.55->1.6 (15.5000..2->16), 2.45->2.4 (24.4999..->24), 2.675->2.7 (26.75->27)
            var r1 = np.round_(np.array(a), 1);
            double[] e1 = { 0.5, 1.5, 2.5, -0.5, -1.5, 0.1, 0.1, 1.6, 2.4, 2.7, 125.0, 135.0 };
            for (int i = 0; i < e1.Length; i++) r1.GetDouble(i).Should().Be(e1[i], $"dec=1 index {i}");
            // dec=2: 0.125->0.12 (banker's), 0.135->0.14, 2.675->2.68
            var r2 = np.round_(np.array(a), 2);
            double[] e2 = { 0.5, 1.5, 2.5, -0.5, -1.5, 0.12, 0.14, 1.55, 2.45, 2.68, 125.0, 135.0 };
            for (int i = 0; i < e2.Length; i++) r2.GetDouble(i).Should().Be(e2[i], $"dec=2 index {i}");
            // dec=-1: 125->120 (12.5->12 even), 135->140 (13.5->14 even); everything <5 -> 0
            var rm1 = np.round_(np.array(a), -1);
            double[] em1 = { 0, 0, 0, -0.0, -0.0, 0, 0, 0, 0, 0, 120.0, 140.0 };
            for (int i = 0; i < em1.Length; i++) rm1.GetDouble(i).Should().Be(em1[i], $"dec=-1 index {i}");
        }

        // ---- negative -0.0 sign preserved (multiply/divide of -0.0 stays -0.0) ----
        [TestMethod]
        public void Round_NegativeZero_SignPreserved()
        {
            var r = np.round_(np.array(new double[] { -0.0 }), -2);
            BitConverter.DoubleToInt64Bits(r.GetDouble(0)).Should().Be(unchecked((long)0x8000000000000000UL),
                "-0.0 survives the scale/rint/unscale with its sign bit");
        }

        // ---- float16 fractional rounding — was carved [OpenBugs], now bit-exact ----
        [TestMethod]
        public void Round_Float16_Fractional_BitExact()
        {
            var a = new Half[] { (Half)0.5, (Half)1.5, (Half)2.5, (Half)1.55, (Half)2.45, (Half)12.34 };
            var r = np.round_(np.array(a), 1);
            // NumPy 2.4.2 hex: 3800 3e00 4100 3e66 40cd 4a26
            ushort[] exp = { 0x3800, 0x3e00, 0x4100, 0x3e66, 0x40cd, 0x4a26 };
            for (int i = 0; i < exp.Length; i++) HalfBits(r, i).Should().Be(exp[i], $"float16 dec=1 index {i}");
        }

        // ---- Complex nonzero decimals — was carved [OpenBugs] (no-op), now rounds re+im ----
        [TestMethod]
        public void Round_Complex_NonzeroDecimals()
        {
            var r = np.round_(np.array(new Complex[] { new(1.55, 2.45), new(125, 135) }), 1);
            var v0 = r.GetAtIndex<Complex>(0);
            v0.Real.Should().Be(1.6, "real part: 1.55*10=15.5000..2 -> rint 16 -> 1.6");
            v0.Imaginary.Should().Be(2.4, "imag part: 2.45*10=24.4999.. -> rint 24 -> 2.4");
            // dec=-1 rounds each component to the tens place
            var rm = np.round_(np.array(new Complex[] { new(1.55, 2.45), new(125, 135) }), -1);
            rm.GetAtIndex<Complex>(1).Should().Be(new Complex(120, 140), "125->120 (even), 135->140 (even)");
        }

        // ---- integer negative decimals: compute in double, cast back to int with C-style WRAP ----
        [TestMethod]
        public void Round_Int_NegativeDecimals_Values()
        {
            var r = np.round_(np.array(new int[] { 125, 135, 150, -125, 49, 51, 999 }), -1);
            int[] exp = { 120, 140, 150, -120, 50, 50, 1000 };
            for (int i = 0; i < exp.Length; i++) r.GetInt32(i).Should().Be(exp[i], $"int32 dec=-1 index {i}");
            r.typecode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Round_Int8_NegativeDecimals_Wraps()
        {
            // 127/10=12.7 -> rint 13 -> 130 -> (int8)130 = -126 ; -128/10=-12.8 -> rint -13 -> -130 -> 126
            var r = np.round_(np.array(new sbyte[] { 127, 126, 100, -128, -100 }), -1);
            sbyte[] exp = { -126, -126, 100, 126, -100 };
            for (int i = 0; i < exp.Length; i++) ((sbyte)r.GetValue(i)).Should().Be(exp[i], $"int8 wrap index {i}");
        }

        // ---- integer positive decimals is an identity (dtype + values unchanged, huge int64 untouched) ----
        [TestMethod]
        public void Round_Int_PositiveDecimals_Identity()
        {
            var big = 9_223_372_036_854_775_807L; // int64 max survives (no float round-trip)
            var r = np.round_(np.array(new long[] { big, 5, -7 }), 3);
            r.GetInt64(0).Should().Be(big);
            r.GetInt64(1).Should().Be(5);
            r.GetInt64(2).Should().Be(-7);
            r.typecode.Should().Be(NPTypeCode.Int64);
        }

        // ---- around is an exact alias of round_ ----
        [TestMethod]
        public void Around_Alias()
        {
            np.around(np.array(new double[] { 2.675 }), 2).GetDouble(0).Should().Be(2.68);
            np.around(np.array(new double[] { 125.0 }), -1).GetDouble(0).Should().Be(120.0);
        }

        // ---- out= : cast-error names the first ufunc (multiply for dec>0, divide for dec<0) ----
        [TestMethod]
        public void Round_Out_CastError_NamesFirstUfunc()
        {
            Action pos = () => np.round_(np.array(new double[] { 1.55 }), 1, np.zeros(new Shape(1), np.int32));
            pos.Should().Throw<ArgumentException>()
               .WithMessage("Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int32')*");
            // round(int, -1, out=int): the negative-decimals result is a float64 that cannot cast same_kind to int
            Action neg = () => np.round_(np.array(new int[] { 125 }), -1, np.zeros(new Shape(1), np.int32));
            neg.Should().Throw<ArgumentException>()
               .WithMessage("Cannot cast ufunc 'divide' output from dtype('float64') to dtype('int32')*");
        }

        // ---- out= that succeeds: same instance returned, correct value ----
        [TestMethod]
        public void Round_Out_Succeeds()
        {
            var o = np.zeros(new Shape(1), np.float64);
            var r = np.round_(np.array(new int[] { 125 }), -1, o);
            ReferenceEquals(r, o).Should().BeTrue("out= returns the same instance");
            r.GetDouble(0).Should().Be(120.0, "int negative-decimals into a float64 out");

            var o2 = np.zeros(new Shape(1), np.float64);
            np.round_(np.array(new double[] { 1.55 }), 1, o2).GetDouble(0).Should().Be(1.6, "f8 into f8 out");
        }

        // ---- layouts: any input layout rounds correctly with NumPy's LOGICAL values (read via
        // GetAtIndex, i.e. C-order through strides). Like NumPy's empty(shape, ISFORTRAN(a)), a
        // transposed (F-contiguous) input yields an F-contiguous result, so the raw buffer order
        // differs from the logical order — the values below are the logical (C-order) sequence. ----
        [TestMethod]
        public void Round_Layouts()
        {
            var b = np.array(new double[] { 1.55, 2.45, 125.0, 135.0, -0.5, 2.675 }).reshape(2, 3);

            var t = np.round_(b.T, -1); // transposed (3,2), negative decimals
            t.shape.Should().Equal(3, 2);
            double[] et = { 0, 140, 0, -0.0, 120, 0 };
            for (int i = 0; i < et.Length; i++) t.GetAtIndex<double>(i).Should().Be(et[i], $"transposed dec=-1 index {i}");

            var rev = np.round_(b["...,::-1"], 1); // reversed columns
            double[] er = { 125, 2.4, 1.6, 2.7, -0.5, 135 };
            for (int i = 0; i < er.Length; i++) rev.GetAtIndex<double>(i).Should().Be(er[i], $"reversed dec=1 index {i}");
        }

        // ---- empty input rounds to an empty array of the same dtype ----
        [TestMethod]
        public void Round_Empty()
        {
            var r = np.round_(np.zeros(new Shape(0), np.float64), 2);
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        // ---- decimals magnitude past double's range -> f (=10^|d|) overflows to inf -> NaN, like NumPy ----
        [TestMethod]
        public void Round_HugeDecimals_YieldNaN()
        {
            np.round_(np.array(new double[] { 1.5, 123.456 }), 400).GetDouble(0).Should().Be(double.NaN);
            // INT_MIN is NumPy's sentinel for INT_MAX magnitude; f overflows -> NaN (and must not hang)
            np.round_(np.array(new double[] { 1.5 }), int.MinValue).GetDouble(0).Should().Be(double.NaN);
        }
    }
}
