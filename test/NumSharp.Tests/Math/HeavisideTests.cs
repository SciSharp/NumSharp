using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.heaviside — the Heaviside step function <c>heaviside(x1, x2)</c>: 0 where x1&lt;0, 1 where x1&gt;0,
    /// x2 where x1==0, and the positive canonical NaN where x1 is NaN. All expected values probed from
    /// NumPy 2.4.2 (win-amd64).
    ///
    /// A float-tier binary ufunc with arctan2's promotion (ee->e, ff->f, dd->d, gg->g — no int/complex loop):
    /// bool/int8/uint8 -> float16, int16/uint16 -> float32, int32+/int64+/char -> float64. BIT-EXACT with
    /// NumPy at every dtype (the step is a port of npy_heaviside). Two NaN behaviours are load-bearing and
    /// pinned below: a NaN x1 yields the POSITIVE canonical NaN (its own sign discarded), while the x1==0
    /// case returns x2's EXACT bits (a NaN or -0.0 x2 keeps its sign). See NDHeavisideMath.
    /// </summary>
    [TestClass]
    public class HeavisideTests
    {
        // NumPy's positive canonical NaN bits (NPY_NAN / NPY_NANF), NOT .NET's negative NaN.
        private const ulong PosNanF64Bits = 0x7ff8000000000000UL;
        private const uint PosNanF32Bits = 0x7fc00000u;
        private const ushort PosNanF16Bits = 0x7e00;

        // ---------------------------------------------------------------- basic values (docstring examples)

        [TestMethod]
        public void Heaviside_Docstring_Half()
        {
            // np.heaviside([-1.5, 0, 2.0], 0.5) -> [0. , 0.5, 1. ]
            var r = np.heaviside(np.array(new[] { -1.5, 0.0, 2.0 }), (NDArray)0.5);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(0.0);
            r.GetAtIndex<double>(1).Should().Be(0.5);
            r.GetAtIndex<double>(2).Should().Be(1.0);
        }

        [TestMethod]
        public void Heaviside_Docstring_One()
        {
            // np.heaviside([-1.5, 0, 2.0], 1) -> [0., 1., 1.]
            var r = np.heaviside(np.array(new[] { -1.5, 0.0, 2.0 }), (NDArray)1.0);
            r.GetAtIndex<double>(0).Should().Be(0.0);
            r.GetAtIndex<double>(1).Should().Be(1.0);
            r.GetAtIndex<double>(2).Should().Be(1.0);
        }

        // ---------------------------------------------------------------- dtype promotion (== arctan2)

        [TestMethod]
        public void Heaviside_Int32_PromotesToFloat64()
        {
            var r = np.heaviside(np.array(new[] { -2, 0, 3 }), np.array(new[] { 7, 7, 7 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            // x1==0 -> h0 (the integer 7 cast to float64)
            r.GetAtIndex<double>(0).Should().Be(0.0);
            r.GetAtIndex<double>(1).Should().Be(7.0);
            r.GetAtIndex<double>(2).Should().Be(1.0);
        }

        [TestMethod]
        public void Heaviside_Float32_Preserved()
        {
            var r = np.heaviside(np.array(new[] { -1f, 0f, 2f }), (NDArray)0.5f);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.GetAtIndex<float>(1).Should().Be(0.5f);
        }

        [TestMethod]
        public void Heaviside_UInt8_Int8_PromotesToFloat16()
        {
            var r = np.heaviside(np.array(new byte[] { 0, 1, 2 }), np.array(new sbyte[] { 1, 1, 1 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(1f);   // x1==0 -> h0=1
            ((float)r.GetAtIndex<Half>(1)).Should().Be(1f);   // x1>0  -> 1
        }

        [TestMethod]
        public void Heaviside_Int16_PromotesToFloat32()
        {
            np.heaviside(np.array(new short[] { 1 }), np.array(new short[] { 0 }))
                .typecode.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Heaviside_Bool_PromotesToFloat16()
        {
            // bool x1 (True->1>0, False->0==0) with a bool fill; both bool -> float16 loop.
            var r = np.heaviside(np.array(new[] { false, true }), np.array(new[] { true, true }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(1f);   // x1=False==0 -> h0=True->1
            ((float)r.GetAtIndex<Half>(1)).Should().Be(1f);   // x1=True>0  -> 1
        }

        [TestMethod]
        public void Heaviside_Char_PromotesToFloat64()
        {
            var r = np.heaviside(np.array(new[] { (char)0, (char)1, (char)2 }), (NDArray)0.5);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(0.5);         // char 0 == 0 -> h0
            r.GetAtIndex<double>(1).Should().Be(1.0);
        }

        [TestMethod]
        public void Heaviside_Decimal_PreservedNoNaN()
        {
            var r = np.heaviside(np.array(new[] { -2m, 0m, 3m }), (NDArray)0.5m);
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(0m);
            r.GetAtIndex<decimal>(1).Should().Be(0.5m);
            r.GetAtIndex<decimal>(2).Should().Be(1m);
        }

        // ---------------------------------------------------------------- IEEE specials

        [TestMethod]
        public void Heaviside_Infinities()
        {
            var r = np.heaviside(np.array(new[] { double.NegativeInfinity, double.PositiveInfinity }), (NDArray)0.5);
            r.GetAtIndex<double>(0).Should().Be(0.0);   // -inf < 0
            r.GetAtIndex<double>(1).Should().Be(1.0);   // +inf > 0
        }

        [TestMethod]
        public void Heaviside_NegativeZeroX_ReturnsFill()
        {
            // IEEE -0.0 == 0.0, so a -0.0 step argument returns the fill (not 0).
            np.heaviside((NDArray)(-0.0), (NDArray)0.5).GetAtIndex<double>(0).Should().Be(0.5);
        }

        [TestMethod]
        public void Heaviside_NaNX_ReturnsPositiveCanonicalNaN_Float64()
        {
            // A NaN x1 yields NumPy's POSITIVE canonical NaN, regardless of x1's own NaN sign.
            var posNan = np.heaviside((NDArray)double.NaN, (NDArray)0.5).GetAtIndex<double>(0);
            BitConverter.DoubleToUInt64Bits(posNan).Should().Be(PosNanF64Bits);

            // .NET's double.NaN is NEGATIVE (0xfff8...); heaviside must still return the POSITIVE one even
            // when x1 is a negative NaN.
            double negNan = BitConverter.UInt64BitsToDouble(0xfff8000000000000UL);
            var fromNeg = np.heaviside((NDArray)negNan, (NDArray)0.5).GetAtIndex<double>(0);
            BitConverter.DoubleToUInt64Bits(fromNeg).Should().Be(PosNanF64Bits);
        }

        [TestMethod]
        public void Heaviside_NaNX_ReturnsPositiveCanonicalNaN_Float32()
        {
            var r = np.heaviside(np.array(new[] { float.NaN }), (NDArray)0.5f).GetAtIndex<float>(0);
            BitConverter.SingleToUInt32Bits(r).Should().Be(PosNanF32Bits);
        }

        [TestMethod]
        public void Heaviside_FillNaN_KeepsSign_AtZero()
        {
            // When x1==0 the result is x2's EXACT bits: a NEGATIVE-NaN fill stays negative (the OPPOSITE of
            // the x1==NaN case, which canonicalises to positive).
            double negNan = BitConverter.UInt64BitsToDouble(0xfff8000000000000UL);
            var r = np.heaviside((NDArray)0.0, (NDArray)negNan).GetAtIndex<double>(0);
            BitConverter.DoubleToUInt64Bits(r).Should().Be(0xfff8000000000000UL);
        }

        [TestMethod]
        public void Heaviside_FillNegativeZero_KeepsSign_AtZero()
        {
            var r = np.heaviside((NDArray)0.0, (NDArray)(-0.0)).GetAtIndex<double>(0);
            BitConverter.DoubleToUInt64Bits(r).Should().Be(0x8000000000000000UL);   // -0.0
        }

        [TestMethod]
        public void Heaviside_FillIgnored_WhenX1NonZero()
        {
            // h0 is consulted ONLY at x1==0, so a NaN fill is irrelevant when x1>0.
            np.heaviside((NDArray)5.0, (NDArray)double.NaN).GetAtIndex<double>(0).Should().Be(1.0);
            np.heaviside((NDArray)(-5.0), (NDArray)double.NaN).GetAtIndex<double>(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Heaviside_Float16_NaNX_CanonicalHalfNaN()
        {
            var r = np.heaviside(np.array(new[] { (Half)0, (Half)1 }), (Half)0.5).GetAtIndex<Half>(0);
            // x1=0 -> fill 0.5
            ((float)r).Should().Be(0.5f);
            var nanR = np.heaviside(np.array(new[] { Half.NaN }), (Half)0.5).GetAtIndex<Half>(0);
            BitConverter.HalfToUInt16Bits(nanR).Should().Be(PosNanF16Bits);
        }

        // ---------------------------------------------------------------- broadcasting / layouts

        [TestMethod]
        public void Heaviside_ScalarFill_Broadcast()
        {
            var r = np.heaviside(np.array(new[] { -1.0, 0.0, 1.0, -2.0, 0.0, 3.0 }), (NDArray)0.25);
            r.GetData<double>().ToArray().Should().Equal(0.0, 0.25, 1.0, 0.0, 0.25, 1.0);
        }

        [TestMethod]
        public void Heaviside_Broadcast_ColumnFill()
        {
            // (6,) step vs (6,1) fill -> (6,6) via broadcasting, matches a contiguous copy of the same values.
            var x = np.arange(6).astype(np.float64) - 2.5;      // spans neg / pos, no exact zero
            var h = np.array(new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }).reshape(6, 1);
            var got = np.heaviside(x, h);
            var truth = np.heaviside(x.copy(), h.copy());
            np.array_equal(got, truth).Should().BeTrue();
            got.shape.Should().Equal(6, 6);
        }

        [TestMethod]
        public void Heaviside_Transposed_And_NegativeStride()
        {
            var baseArr = (np.arange(36).astype(np.float64) - 15.0).reshape(6, 6);
            var h = np.full(new Shape(6, 6), 0.5, np.float64);
            // Transposed view
            np.array_equal(np.heaviside(baseArr.T, h.T),
                           np.heaviside(baseArr.T.copy(), h.T.copy())).Should().BeTrue();
            // Negative-stride view
            np.array_equal(np.heaviside(baseArr["::-1"], h["::-1"]),
                           np.heaviside(baseArr["::-1"].copy(), h["::-1"].copy())).Should().BeTrue();
        }

        // ---------------------------------------------------------------- ufunc parameters

        [TestMethod]
        public void Heaviside_DtypeFloat32_ComputesAtSinglePrecision()
        {
            var r = np.heaviside(np.array(new[] { -1.0, 0.0, 1.0 }), (NDArray)0.5, dtype: np.float32);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.GetAtIndex<float>(1).Should().Be(0.5f);
        }

        [TestMethod]
        public void Heaviside_Out_ReturnsSameInstance()
        {
            var x = np.array(new[] { -1.0, 0.0, 2.0, -3.0 });
            var o = np.zeros(new Shape(4), np.float64);
            var ret = np.heaviside(x, (NDArray)0.5, @out: o);
            ReferenceEquals(ret, o).Should().BeTrue();
            o.GetData<double>().ToArray().Should().Equal(0.0, 0.5, 1.0, 0.0);
        }

        [TestMethod]
        public void Heaviside_Where_LeavesMaskedOffUntouched()
        {
            var x = np.array(new[] { -1.0, 0.0, 2.0, -3.0 });
            var o = np.full(new Shape(4), 9.0, np.float64);
            var mask = np.array(new[] { true, false, true, false });
            np.heaviside(x, (NDArray)0.5, @out: o, where: mask);
            o.GetData<double>().ToArray().Should().Equal(0.0, 9.0, 1.0, 9.0);   // idx 1,3 keep 9
        }

        // ---------------------------------------------------------------- error taxonomy (no int/complex loop)

        [TestMethod]
        public void Heaviside_DtypeInteger_Throws_NoLoop()
        {
            Action act = () => np.heaviside(np.array(new[] { 1.0, 0.0, -1.0 }), (NDArray)0.5, dtype: np.int32);
            act.Should().Throw<IncorrectTypeException>()
               .WithMessage("*No loop matching the specified signature*heaviside*");
        }

        [TestMethod]
        public void Heaviside_DtypeComplex_Throws_NoLoop()
        {
            Action act = () => np.heaviside(np.array(new[] { 1.0, 0.0, -1.0 }), (NDArray)0.5, dtype: np.complex128);
            act.Should().Throw<IncorrectTypeException>()
               .WithMessage("*No loop matching the specified signature*heaviside*");
        }

        [TestMethod]
        public void Heaviside_ComplexInput_Throws()
        {
            Action act = () => np.heaviside(np.array(new[] { new Complex(1, 2) }), (NDArray)0.5);
            act.Should().Throw<IncorrectTypeException>()
               .WithMessage("*ufunc 'heaviside' not supported for the input types*");
        }

        // ---------------------------------------------------------------- scalar / empty

        [TestMethod]
        public void Heaviside_ScalarScalar()
        {
            np.heaviside((NDArray)0.0, (NDArray)0.5).GetAtIndex<double>(0).Should().Be(0.5);
            np.heaviside((NDArray)(-3.0), (NDArray)0.5).GetAtIndex<double>(0).Should().Be(0.0);
            np.heaviside((NDArray)3.0, (NDArray)0.5).GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void Heaviside_Empty_ReturnsEmpty()
        {
            var e = np.heaviside(np.zeros(new Shape(0), np.float64), (NDArray)0.5);
            e.size.Should().Be(0);
            e.typecode.Should().Be(NPTypeCode.Double);
        }
    }
}
