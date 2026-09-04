using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.logaddexp / np.logaddexp2 / np.nextafter. All expected values probed from NumPy 2.4.2.
    ///
    /// These are float-tier binary ufuncs with arctan2's promotion (ee->e, ff->f, dd->d, gg->g):
    /// bool/int8/uint8 -> float16, int16/uint16 -> float32, int32+/int64+/char -> float64.
    ///
    /// nextafter is BIT-EXACT with NumPy (Math.BitIncrement/BitDecrement compose to ucrtbase
    /// nextafter). logaddexp/logaddexp2 are &lt;=2 ULP (managed fdlibm log1p vs NumPy's closed
    /// ucrtbase log1p); Math.Exp/double.Exp2/MathF.Exp are already bit-identical to NumPy's exp/exp2.
    /// </summary>
    [TestClass]
    public class LogAddExpNextAfterTests
    {
        // ---------------------------------------------------------------- logaddexp

        [TestMethod]
        public void LogAddExp_Float64()
        {
            var r = np.logaddexp(np.array(new[] { 1.0, 2, 3 }), np.array(new[] { 2.0, 2, 0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(2.313261687518223, 1e-15);
            r.GetAtIndex<double>(1).Should().BeApproximately(2.6931471805599454, 1e-15);
            r.GetAtIndex<double>(2).Should().BeApproximately(3.048587351573742, 1e-15);
        }

        [TestMethod]
        public void LogAddExp_SmallValue_NotLostToZero()
        {
            // The catastrophic case a naive log(1+exp(-tmp)) loses: log1p(1.9e-22) must NOT round to 0.
            np.logaddexp((NDArray)0.0, (NDArray)(-50.0)).GetAtIndex<double>(0)
                .Should().BeApproximately(1.9287498479639178e-22, 1e-30);
        }

        [TestMethod]
        public void LogAddExp_Infinities_SameSign_NoNaN()
        {
            // x == y branch handles same-sign infinities without producing NaN.
            np.logaddexp((NDArray)double.PositiveInfinity, (NDArray)double.PositiveInfinity)
                .GetAtIndex<double>(0).Should().Be(double.PositiveInfinity);
            np.logaddexp((NDArray)double.NegativeInfinity, (NDArray)double.NegativeInfinity)
                .GetAtIndex<double>(0).Should().Be(double.NegativeInfinity);
        }

        [TestMethod]
        public void LogAddExp_NaN_Propagates()
        {
            double.IsNaN(np.logaddexp((NDArray)double.NaN, (NDArray)1.0).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void LogAddExp2_Float64()
        {
            var r = np.logaddexp2(np.array(new[] { 1.0, 2, 3 }), np.array(new[] { 2.0, 2, 0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(2.584962500721156, 1e-14);
            r.GetAtIndex<double>(1).Should().Be(3.0);                    // x==y -> x+1
            r.GetAtIndex<double>(2).Should().BeApproximately(3.169925001442312, 1e-14);
        }

        [TestMethod]
        public void LogAddExp2_EqualInputs_XPlusOne()
        {
            np.logaddexp2((NDArray)3.0, (NDArray)3.0).GetAtIndex<double>(0).Should().Be(4.0);
        }

        // ---------------------------------------------------------------- nextafter (bit-exact)

        [TestMethod]
        public void NextAfter_Float64_BitExact()
        {
            np.nextafter((NDArray)1.0, (NDArray)2.0).GetAtIndex<double>(0).Should().Be(1.0000000000000002);
            np.nextafter((NDArray)1.0, (NDArray)0.0).GetAtIndex<double>(0).Should().Be(0.9999999999999999);
            np.nextafter((NDArray)0.0, (NDArray)1.0).GetAtIndex<double>(0).Should().Be(double.Epsilon);      // 5e-324
            np.nextafter((NDArray)0.0, (NDArray)(-1.0)).GetAtIndex<double>(0).Should().Be(-double.Epsilon);
            np.nextafter((NDArray)double.MaxValue, (NDArray)double.PositiveInfinity).GetAtIndex<double>(0)
                .Should().Be(double.PositiveInfinity);                  // overflow -> +inf
        }

        [TestMethod]
        public void NextAfter_EqualInputs_ReturnsSecond()
        {
            np.nextafter((NDArray)2.0, (NDArray)2.0).GetAtIndex<double>(0).Should().Be(2.0);
        }

        [TestMethod]
        public void NextAfter_Float32_BitExact()
        {
            np.nextafter(np.array(new[] { 1.0f }), np.array(new[] { 2.0f })).GetAtIndex<float>(0)
                .Should().Be(1.0000001f);
        }

        [TestMethod]
        public void NextAfter_NaN_ReturnsNaN()
        {
            float.IsNaN(np.nextafter(np.array(new[] { float.NaN }), np.array(new[] { 1.0f })).GetAtIndex<float>(0))
                .Should().BeTrue();
        }

        // ---------------------------------------------------------------- copysign (bit-exact)

        [TestMethod]
        public void CopySign_Float64()
        {
            var r = np.copysign(np.array(new[] { 1.0, -1, 2, -2 }), np.array(new[] { -3.0, 3, -0.0, 0.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(-1.0);   // |1| with sign of -3
            r.GetAtIndex<double>(1).Should().Be(1.0);
            r.GetAtIndex<double>(2).Should().Be(-2.0);   // sign of -0.0 is negative
            r.GetAtIndex<double>(3).Should().Be(2.0);
        }

        [TestMethod]
        public void CopySign_SignOfNegativeZero_IsNegative()
        {
            np.copysign((NDArray)3.0, (NDArray)(-0.0)).GetAtIndex<double>(0).Should().Be(-3.0);
        }

        [TestMethod]
        public void CopySign_Infinity()
        {
            np.copysign((NDArray)double.PositiveInfinity, (NDArray)(-1.0)).GetAtIndex<double>(0)
                .Should().Be(double.NegativeInfinity);
        }

        [TestMethod]
        public void CopySign_Float32_BitExact()
        {
            np.copysign(np.array(new[] { 5.0f }), np.array(new[] { -1.0f })).GetAtIndex<float>(0).Should().Be(-5.0f);
        }

        [TestMethod]
        public void CopySign_Promotion_FloatTier()
        {
            np.copysign(np.array(new short[] { 5 }), np.array(new short[] { -1 })).typecode.Should().Be(NPTypeCode.Single);
            np.copysign(np.array(new sbyte[] { 5 }), np.array(new sbyte[] { -1 })).typecode.Should().Be(NPTypeCode.Half);
            np.copysign(np.array(new[] { 5 }), np.array(new[] { -1 })).typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void CopySign_Half_ComputedInFloat32()
        {
            var r = np.copysign(np.array(new[] { (Half)5 }), np.array(new[] { (Half)(-1) }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(-5f);
        }

        [TestMethod]
        public void CopySign_OutAndWhere()
        {
            var o = np.array(new[] { -9.0, -9, -9 });
            var w = np.array(new[] { true, false, true });
            np.copysign(np.array(new[] { 1.0, 2, 3 }), np.array(new[] { -1.0, 1, -1 }), @out: o, where: w);
            o.GetAtIndex<double>(0).Should().Be(-1.0);
            o.GetAtIndex<double>(1).Should().Be(-9.0);   // masked off
            o.GetAtIndex<double>(2).Should().Be(-3.0);
        }

        // ---------------------------------------------------------------- dtype promotion

        [TestMethod]
        public void Promotion_FloatTier_MatchesArctan2()
        {
            np.logaddexp(np.array(new sbyte[] { 1 }), np.array(new sbyte[] { 2 })).typecode.Should().Be(NPTypeCode.Half);
            np.logaddexp(np.array(new short[] { 1 }), np.array(new short[] { 2 })).typecode.Should().Be(NPTypeCode.Single);
            np.logaddexp(np.array(new[] { 1 }), np.array(new[] { 2 })).typecode.Should().Be(NPTypeCode.Double);
            np.nextafter(np.array(new[] { 1.0f }), np.array(new[] { 2.0 })).typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Half_ComputedInFloat32()
        {
            // NumPy computes the ee->e half loop in float32; logaddexp(1,2) as half == 2.3125.
            var r = np.logaddexp(np.array(new[] { (Half)1 }), np.array(new[] { (Half)2 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(2.3125f);
        }

        // ---------------------------------------------------------------- ufunc out= / where=

        [TestMethod]
        public void OutParam_ReturnsSameInstance_AndWrites()
        {
            var o = np.zeros(3, typeof(double));
            var r = np.logaddexp(np.array(new[] { 1.0, 2, 3 }), np.array(new[] { 2.0, 2, 0 }), @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.GetAtIndex<double>(1).Should().BeApproximately(2.6931471805599454, 1e-15);
        }

        [TestMethod]
        public void WhereParam_LeavesMaskedSlotsUntouched()
        {
            var o = np.array(new[] { -1.0, -1, -1 });
            var w = np.array(new[] { true, false, true });
            np.logaddexp(np.array(new[] { 1.0, 2, 3 }), np.array(new[] { 2.0, 2, 0 }), @out: o, where: w);
            o.GetAtIndex<double>(1).Should().Be(-1.0);                  // masked off -> preserved
            o.GetAtIndex<double>(0).Should().BeApproximately(2.313261687518223, 1e-15);
        }

        // ---------------------------------------------------------------- scalar / empty

        [TestMethod]
        public void BothScalars_Returns0d()
        {
            np.logaddexp((NDArray)2.0, (NDArray)3.0).ndim.Should().Be(0);
            np.nextafter((NDArray)1.0, (NDArray)2.0).ndim.Should().Be(0);
        }

        [TestMethod]
        public void EmptyArrays_ReturnEmpty()
        {
            var r = np.logaddexp(np.zeros(0, dtype: typeof(double)), np.zeros(0, typeof(double)));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        // ---------------------------------------------------------------- broadcasting

        [TestMethod]
        public void Broadcasting_RowAgainstMatrix()
        {
            var a = np.arange(1.0, 7.0).reshape(2, 3);
            var b = np.array(new[] { 0.5, 1.5, 2.5 });
            var r = np.logaddexp2(a, b);
            r.shape.Should().Equal(2, 3);
            // logaddexp2(1, 0.5) == 1.7715533031636119 (probed)
            r.GetAtIndex<double>(0).Should().BeApproximately(1.7715533031636119, 1e-14);
        }

        // ---------------------------------------------------------------- errors

        [TestMethod]
        public void ComplexInput_RaisesNoLoop()
        {
            Action act = () => np.logaddexp(
                np.array(new[] { new Complex(1, 1) }), np.array(new[] { new Complex(2, 2) }));
            act.Should().Throw<IncorrectTypeException>().WithMessage("*not supported for the input types*");
        }

        [TestMethod]
        public void IntegerDtype_RaisesNoLoop()
        {
            Action act = () => np.nextafter(np.array(new[] { 1.0 }), np.array(new[] { 2.0 }), dtype: NPTypeCode.Int32);
            act.Should().Throw<IncorrectTypeException>().WithMessage("*No loop matching*nextafter*");
        }
    }
}
