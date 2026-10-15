using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Pins <see cref="np.float_power(NDArray,NDArray,NDArray,NDArray,DType)"/> against NumPy 2.4.2.
    ///     float_power is <c>power</c> restricted to the float64/complex128 loops, so these tests target
    ///     the behaviours that DISTINGUISH it from <c>np.power</c> (the always-float64 promotion, the
    ///     legal negative integer exponent, the dtype=/out= loop restriction) rather than re-proving the
    ///     shared power arithmetic (the differential-fuzz <c>divmod_power</c> tier covers that bit-exactly).
    /// </summary>
    [TestClass]
    public class FloatPowerTest
    {
        /// <summary>
        ///     Every real input promotes to float64 — the defining property of float_power vs power
        ///     (which keeps int/float16/float32/bool dtypes). A regression that preserved the input dtype
        ///     would silently change float_power's contract, so the dtype assertion is the point here.
        /// </summary>
        [TestMethod]
        public void IntegerInputs_PromoteToFloat64()
        {
            var r = np.float_power(np.array(new[] { 2, 3, 4 }), np.array(new[] { 3, 2, 1 }));
            r.dtype.Should().Be(typeof(double));
            Assert.AreEqual(8.0, r.GetDouble(0));
            Assert.AreEqual(9.0, r.GetDouble(1));
            Assert.AreEqual(4.0, r.GetDouble(2));
        }

        /// <summary>
        ///     The headline difference from power: a negative INTEGER exponent is legal (there is no
        ///     integer loop to trigger power's "Integers to negative integer powers are not allowed").
        ///     Guards the exact reason float_power exists.
        /// </summary>
        [TestMethod]
        public void NegativeIntegerExponent_IsLegal_WherePowerRaises()
        {
            var r = np.float_power(np.array(new[] { 2, 4 }), np.array(new[] { -1, -2 }));
            r.dtype.Should().Be(typeof(double));
            Assert.AreEqual(0.5, r.GetDouble(0));
            Assert.AreEqual(0.0625, r.GetDouble(1));

            // Sanity: np.power on the same integer operands DOES raise (the behaviour float_power avoids).
            Assert.ThrowsException<ArgumentException>(
                () => np.power(np.array(new[] { 2, 4 }), np.array(new[] { -1, -2 })));
        }

        /// <summary>
        ///     float16 and float32 inputs promote to float64 (minimum precision float64), unlike power
        ///     which keeps the narrower float width.
        /// </summary>
        [TestMethod]
        public void NarrowFloatInputs_PromoteToFloat64()
        {
            np.float_power(np.array(new[] { (Half)2, (Half)3 }), np.array(new[] { (Half)3, (Half)2 }))
                .dtype.Should().Be(typeof(double));
            np.float_power(np.array(new[] { 2f, 3f }), np.array(new[] { 3f, 2f }))
                .dtype.Should().Be(typeof(double));
        }

        /// <summary>
        ///     A complex operand promotes to complex128 (the DD-&gt;D loop), and the value matches the
        ///     complex power engine.
        /// </summary>
        [TestMethod]
        public void ComplexInput_PromotesToComplex128()
        {
            var r = np.float_power(np.array(new[] { new Complex(2, 0) }), np.array(new[] { new Complex(3, 0) }));
            r.dtype.Should().Be(typeof(Complex));
            Assert.AreEqual(new Complex(8, 0), r.GetAtIndex<Complex>(0));
        }

        /// <summary>
        ///     bool inputs promote to float64 (float_power has no bool loop); True**True == 1.0.
        /// </summary>
        [TestMethod]
        public void BoolInputs_PromoteToFloat64()
        {
            var r = np.float_power(np.array(new[] { true, false }), np.array(new[] { true, true }));
            r.dtype.Should().Be(typeof(double));
            Assert.AreEqual(1.0, r.GetDouble(0));
            Assert.AreEqual(0.0, r.GetDouble(1));
        }

        /// <summary>
        ///     Char and Decimal have no NumPy analog but must still promote to float64 (the general
        ///     "everything non-complex -&gt; float64" rule), never crash or stay in their own dtype.
        /// </summary>
        [TestMethod]
        public void CharAndDecimalInputs_PromoteToFloat64()
        {
            var rc = np.float_power(np.array(new[] { (char)2, (char)3 }), np.array(new[] { (char)3, (char)2 }));
            rc.dtype.Should().Be(typeof(double));
            Assert.AreEqual(8.0, rc.GetDouble(0));

            var rd = np.float_power(np.array(new[] { 2m, 3m }), np.array(new[] { 3m, 2m }));
            rd.dtype.Should().Be(typeof(double));
            Assert.AreEqual(9.0, rd.GetDouble(1));
        }

        /// <summary>
        ///     IEEE special-value edges, all float64 and all matching NumPy: 0**0=1, 0**-1=inf,
        ///     nan**0=1, (-1)**0.5=nan, (-8)**(1/3)=nan (a negative base with a non-integer exponent).
        /// </summary>
        [TestMethod]
        public void Specials_MatchNumpy()
        {
            var b = np.array(new[] { 0.0, 0.0, double.NaN, -1.0, -8.0 });
            var e = np.array(new[] { 0.0, -1.0, 0.0, 0.5, 1.0 / 3.0 });
            var r = np.float_power(b, e);
            Assert.AreEqual(1.0, r.GetDouble(0));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(1)));
            Assert.AreEqual(1.0, r.GetDouble(2));
            Assert.IsTrue(double.IsNaN(r.GetDouble(3)));
            Assert.IsTrue(double.IsNaN(r.GetDouble(4)));
        }

        /// <summary>
        ///     out= receives the result cast same_kind from the float64 loop (here float32) and the
        ///     SAME instance is returned — the NumPy ufunc out= contract.
        /// </summary>
        [TestMethod]
        public void Out_CastsSameKind_AndReturnsSameInstance()
        {
            var o = np.zeros(new Shape(3), NPTypeCode.Single);
            var r = np.float_power(np.array(new[] { 2.0, 3.0, 4.0 }), np.array(new[] { 2.0, 2.0, 2.0 }), @out: o);
            Assert.IsTrue(ReferenceEquals(o, r));
            o.dtype.Should().Be(typeof(float));
            Assert.AreEqual(4f, o.GetSingle(0));
            Assert.AreEqual(16f, o.GetSingle(2));
        }

        /// <summary>
        ///     where= computes only mask-true slots; false slots keep the prior out contents.
        /// </summary>
        [TestMethod]
        public void Where_LeavesFalseSlotsUntouched()
        {
            var o = np.full(new Shape(4), 7.0, NPTypeCode.Double);
            np.float_power(np.array(new[] { 2.0, 3.0, 4.0, 5.0 }), np.array(new[] { 2.0, 2.0, 2.0, 2.0 }),
                @out: o, where: np.array(new[] { true, false, true, false }));
            Assert.AreEqual(4.0, o.GetDouble(0));
            Assert.AreEqual(7.0, o.GetDouble(1));   // masked off — prior value kept
            Assert.AreEqual(16.0, o.GetDouble(2));
            Assert.AreEqual(7.0, o.GetDouble(3));
        }

        /// <summary>
        ///     dtype= selects the loop: float64 is legal (int inputs cast up), complex128 is legal, and
        ///     ANY other dtype has no loop — NumPy's "No loop matching..." TypeError, reproduced as the
        ///     house IncorrectTypeException.
        /// </summary>
        [TestMethod]
        public void Dtype_OnlyFloat64OrComplex128()
        {
            np.float_power(np.array(new[] { 2, 3 }), np.array(new[] { -1, -2 }), dtype: (DType)NPTypeCode.Double)
                .dtype.Should().Be(typeof(double));
            np.float_power(np.array(new[] { 2.0, 3.0 }), np.array(new[] { 2.0, 2.0 }), dtype: (DType)NPTypeCode.Complex)
                .dtype.Should().Be(typeof(Complex));

            var ex = Assert.ThrowsException<IncorrectTypeException>(
                () => np.float_power(np.array(new[] { 2.0 }), np.array(new[] { 3.0 }), dtype: (DType)NPTypeCode.Int32));
            StringAssert.Contains(ex.Message, "No loop matching the specified signature and casting was found for ufunc float_power");
        }

        /// <summary>
        ///     A complex input against an explicit float64 loop cannot same_kind-cast — NumPy's input-0
        ///     UFuncTypeError, reproduced verbatim (as ArgumentException) with the 'float_power' name.
        /// </summary>
        [TestMethod]
        public void ComplexInput_WithDtypeFloat64_RaisesInputCast()
        {
            var ex = Assert.ThrowsException<ArgumentException>(
                () => np.float_power(np.array(new[] { new Complex(1, 1) }), np.array(new[] { 2.0 }),
                    dtype: (DType)NPTypeCode.Double));
            StringAssert.Contains(ex.Message,
                "Cannot cast ufunc 'float_power' input 0 from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     Broadcasting a column against a row yields the full grid at float64, matching power's
        ///     broadcast semantics on the forced float loop.
        /// </summary>
        [TestMethod]
        public void Broadcast_ColumnByRow()
        {
            var r = np.float_power(np.array(new[] { 2.0, 3.0 }).reshape(2, 1),
                                   np.array(new[] { 1.0, 2.0, 3.0 }).reshape(1, 3));
            r.shape.Should().Equal(2, 3);
            Assert.AreEqual(8.0, r.GetDouble(0, 2));   // 2**3
            Assert.AreEqual(27.0, r.GetDouble(1, 2));  // 3**3
        }

        /// <summary>
        ///     The scalar/array-like object overload mirrors <c>np.power(x1, object)</c> — a plain scalar
        ///     exponent works and promotes to float64.
        /// </summary>
        [TestMethod]
        public void ObjectOverload_ScalarExponent()
        {
            var r = np.float_power(np.array(new[] { 2.0, 3.0, 4.0 }), 2);
            r.dtype.Should().Be(typeof(double));
            Assert.AreEqual(4.0, r.GetDouble(0));
            Assert.AreEqual(16.0, r.GetDouble(2));
        }
    }
}
