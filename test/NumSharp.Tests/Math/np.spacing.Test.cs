using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.spacing — distance to the adjacent representable value away from zero (one ULP).
    /// All expected values verified against NumPy 2.4.2.
    ///
    /// spacing is a PORTABLE bit-fiddle (npy_spacing / npy_half_spacing), so the results are
    /// bit-identical to NumPy on every platform (no host-libm dependence, unlike arcsinh/tanh):
    /// float32/float64 are the SIGNED formula (carry the sign of x, +minsubnormal at ±0, ±inf/NaN→NaN,
    /// overflow→+inf); float16 is NumPy's SEPARATE always-non-negative routine that halves the ULP at a
    /// negative power-of-2 boundary. Complex has no loop (float-only ufunc) → TypeError.
    /// </summary>
    [TestClass]
    public class SpacingTests
    {
        private static ushort HB(Half h) => BitConverter.HalfToUInt16Bits(h);

        // ---------------------------------------------------------------- float64 (signed)
        [TestMethod]
        public void Spacing_Double_Values()
        {
            var r = np.spacing(np.array(new double[] { 1.0, -1.0, 2.0, 0.5 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(2.220446049250313e-16);    // +eps (spacing(1)==finfo.eps)
            r.GetAtIndex<double>(1).Should().Be(-2.220446049250313e-16);   // sign follows x
            r.GetAtIndex<double>(2).Should().Be(4.440892098500626e-16);
            r.GetAtIndex<double>(3).Should().Be(1.1102230246251565e-16);
        }

        [TestMethod]
        public void Spacing_Double_ZeroIsPositiveMinSubnormal()
        {
            // Both signs of zero map to +minsubnormal (NumPy's _next(0,1) steps toward +inf regardless).
            var r = np.spacing(np.array(new double[] { 0.0, -0.0 }));
            r.GetAtIndex<double>(0).Should().Be(double.Epsilon);   // 5e-324
            r.GetAtIndex<double>(1).Should().Be(double.Epsilon);   // NOT -5e-324
        }

        [TestMethod]
        public void Spacing_Double_Specials()
        {
            var r = np.spacing(np.array(new double[]
                { double.PositiveInfinity, double.NegativeInfinity, double.NaN, double.MaxValue }));
            double.IsNaN(r.GetAtIndex<double>(0)).Should().BeTrue();   // +inf → NaN
            double.IsNaN(r.GetAtIndex<double>(1)).Should().BeTrue();   // -inf → NaN
            double.IsNaN(r.GetAtIndex<double>(2)).Should().BeTrue();   // NaN → NaN
            double.IsPositiveInfinity(r.GetAtIndex<double>(3)).Should().BeTrue();  // MaxValue overflows → +inf
        }

        // ---------------------------------------------------------------- float32 (signed)
        [TestMethod]
        public void Spacing_Single_Values()
        {
            var r = np.spacing(np.array(new float[] { 1.0f, -1.0f, 0.0f }));
            r.typecode.Should().Be(NPTypeCode.Single);
            BitConverter.SingleToUInt32Bits(r.GetAtIndex<float>(0)).Should().Be(0x34000000u);  // +eps32 = 2^-23
            BitConverter.SingleToUInt32Bits(r.GetAtIndex<float>(1)).Should().Be(0xB4000000u);  // -eps32
            r.GetAtIndex<float>(2).Should().Be(float.Epsilon);                                  // +minsubnormal
            float.IsNaN(np.spacing(np.array(new float[] { float.PositiveInfinity }))
                .GetAtIndex<float>(0)).Should().BeTrue();
        }

        // ---------------------------------------------------------------- float16 (always non-negative)
        [TestMethod]
        public void Spacing_Half_IsAlwaysNonNegative_AndHalvesUlpAtNegBoundary()
        {
            var r = np.spacing(np.array(new Half[]
                { (Half)1.0, (Half)(-1.0), (Half)0.5, (Half)(-0.5), (Half)2.0 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            HB(r.GetAtIndex<Half>(0)).Should().Be((ushort)0x1400);   // spacing(1) = 2^-10 = eps16
            HB(r.GetAtIndex<Half>(1)).Should().Be((ushort)0x1000);   // spacing(-1) = 2^-11 POSITIVE (half ULP)
            HB(r.GetAtIndex<Half>(2)).Should().Be((ushort)0x1000);   // spacing(0.5) = 2^-11
            HB(r.GetAtIndex<Half>(3)).Should().Be((ushort)0x0C00);   // spacing(-0.5) = 2^-12 POSITIVE
            HB(r.GetAtIndex<Half>(4)).Should().Be((ushort)0x1800);   // spacing(2) = 2^-9
        }

        [TestMethod]
        public void Spacing_Half_Specials()
        {
            var r = np.spacing(np.array(new Half[]
                { (Half)0.0, Half.PositiveInfinity, Half.NaN, (Half)65504.0, (Half)(-65504.0) }));
            HB(r.GetAtIndex<Half>(0)).Should().Be((ushort)0x0001);   // ±0 → smallest subnormal half
            Half.IsNaN(r.GetAtIndex<Half>(1)).Should().BeTrue();     // +inf → NaN
            Half.IsNaN(r.GetAtIndex<Half>(2)).Should().BeTrue();     // NaN → NaN
            Half.IsPositiveInfinity(r.GetAtIndex<Half>(3)).Should().BeTrue();  // largest +half overflows → +inf
            HB(r.GetAtIndex<Half>(4)).Should().Be((ushort)0x5000);   // spacing(-65504) = 32.0 (finite, positive)
        }

        // ---------------------------------------------------------------- dtype promotion (NumPy width rule)
        [TestMethod]
        public void Spacing_IntegerAndBool_Promote()
        {
            np.spacing(np.array(new int[] { 1, 2, 3 })).dtype.Should().Be(np.float64);         // int32 → float64
            np.spacing(np.array(new long[] { 1 })).dtype.Should().Be(np.float64);              // int64 → float64
            np.spacing(np.array(new short[] { 1 })).dtype.Should().Be(np.float32);             // int16 → float32
            np.spacing(np.array(new byte[] { 1 })).dtype.Should().Be(np.float16);              // uint8 → float16
            var b = np.spacing(np.array(new bool[] { true, false }));
            b.dtype.Should().Be(np.float16);                                                    // bool → float16
            HB(b.GetAtIndex<Half>(0)).Should().Be((ushort)0x1400);   // spacing(1.0 in f16)
            HB(b.GetAtIndex<Half>(1)).Should().Be((ushort)0x0001);   // spacing(0.0 in f16)
            var i32 = np.spacing(np.array(new int[] { 1, 2 }));
            i32.GetAtIndex<double>(0).Should().Be(2.220446049250313e-16);
            i32.GetAtIndex<double>(1).Should().Be(4.440892098500626e-16);
        }

        // ---------------------------------------------------------------- ufunc parameters
        [TestMethod]
        public void Spacing_OutParameter_ReturnsSameInstanceAndWrites()
        {
            var a = np.array(new double[] { 1.0, 2.0 });
            var o = np.zeros(2, np.float64);
            var r = np.spacing(a, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.GetAtIndex<double>(0).Should().Be(2.220446049250313e-16);
        }

        [TestMethod]
        public void Spacing_WhereMask_LeavesMaskedOutSlotsUntouched()
        {
            var a = np.array(new double[] { 1.0, 2.0, 4.0 });
            var o = np.array(new double[] { 9.0, 9.0, 9.0 });
            np.spacing(a, @out: o, where: np.array(new bool[] { true, false, true }));
            o.GetAtIndex<double>(0).Should().Be(2.220446049250313e-16);
            o.GetAtIndex<double>(1).Should().Be(9.0);                       // masked off → prior contents
            o.GetAtIndex<double>(2).Should().Be(8.881784197001252e-16);
        }

        [TestMethod]
        public void Spacing_DtypeSelectsLoop()
        {
            var a = np.array(new double[] { 1.0 });
            np.spacing(a, dtype: np.float32).dtype.Should().Be(np.float32);
            np.spacing(a, dtype: np.float16).dtype.Should().Be(np.float16);
            BitConverter.SingleToUInt32Bits(np.spacing(a, dtype: np.float32).GetAtIndex<float>(0))
                .Should().Be(0x34000000u);
        }

        // ---------------------------------------------------------------- error taxonomy (verbatim messages)
        [TestMethod]
        public void Spacing_Complex_Rejected()
        {
            Action act = () => np.spacing(np.array(new Complex[] { new Complex(1, 2) }));
            act.Should().Throw<IncorrectTypeException>()
                .WithMessage("ufunc 'spacing' not supported for the input types*");
        }

        [TestMethod]
        public void Spacing_DtypeComplexOrInt_NoLoop()
        {
            var a = np.array(new double[] { 1.0 });
            ((Action)(() => np.spacing(a, dtype: np.int32))).Should().Throw<IncorrectTypeException>()
                .WithMessage("No loop matching the specified signature*");
            ((Action)(() => np.spacing(a, dtype: np.complex128))).Should().Throw<IncorrectTypeException>()
                .WithMessage("No loop matching the specified signature*");
        }

        // ---------------------------------------------------------------- layout (non-contiguous view)
        [TestMethod]
        public void Spacing_TransposedView_MatchesLogicalOrder()
        {
            // A transposed (non-contiguous) view is read through its strides; result is C-contiguous.
            // b = [[1,2,3],[4,5,6]]; b.T = [[1,4],[2,5],[3,6]] (3,2); spacing → C-order [sp1,sp4,sp2,sp5,sp3,sp6].
            var b = np.arange(1.0, 7.0).reshape(2, 3);
            var r = np.spacing(b.T);
            r.shape.Should().Equal(3, 2);
            double Sp(double x) => np.spacing(np.array(new[] { x })).GetAtIndex<double>(0);
            r.GetAtIndex<double>(0).Should().Be(Sp(1.0));   // (0,0)
            r.GetAtIndex<double>(1).Should().Be(Sp(4.0));   // (0,1)
            r.GetAtIndex<double>(5).Should().Be(Sp(6.0));   // (2,1)
        }
    }
}
