using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.linspace / np.logspace / np.geomspace at the edges the snapshot pools and the older tests did not reach,
    ///     found when the np-layer dtype plan's P3 batch 9 made them engine members and replayed them against NumPy 2.4.2.
    ///     Every expectation is NumPy 2.4.2's output (values as raw float64 bytes where a NaN's bits or a zero's sign is
    ///     the point) or its error text.
    /// </summary>
    [TestClass]
    public class np_spacing_numpy_edges_tests
    {
        // ------------------------------------------------------------------------------------------- the sample count

        /// <summary>
        ///     NumPy's linspace starts with <c>if num &lt; 0: raise ValueError("Number of samples, ...")</c>, before it
        ///     reads the device (the ramp's np.arange does) and before anything is allocated - NumSharp reported the
        ///     allocation's <c>negative dimensions are not allowed</c>, or the device error when both were bad.
        /// </summary>
        [TestMethod]
        public void Linspace_NegativeCount_IsNumPysValueError_First()
        {
            // np.linspace(0, 1, -5) -> ValueError: Number of samples, -5, must be non-negative.
            new Action(() => np.linspace(0.0, 1.0, -5))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -5, must be non-negative.");
            new Action(() => np.linspace(0.0, 1.0, -1L, endpoint: false))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
            new Action(() => np.linspace(0f, 1f, -2))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -2, must be non-negative.");
            // np.linspace(0, 1, -1, device='cuda') -> the count's ValueError, not the device's
            new Action(() => np.linspace(0.0, 1.0, -1, device: "cuda"))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
            // np.linspace(0, 1, -1, dtype=np.int32) -> the count's ValueError
            new Action(() => np.linspace(0.0, 1.0, -1, dtype: np.int32))
                .Should().Throw<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");
        }

        /// <summary>With a valid count, a device other than "cpu" is refused (NumPy's ValueError text).</summary>
        [TestMethod]
        public void Linspace_BadDevice_AfterAValidCount()
        {
            // np.linspace(0, 1, 3, device='cuda') -> ValueError: Device not understood. Only "cpu" is allowed, but received: cuda
            new Action(() => np.linspace(0.0, 1.0, 3, device: "cuda"))
                .Should().Throw<ArgumentException>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: cuda*");
            np.linspace(0.0, 1.0, 3, device: "cpu").Should().BeShaped(3);
        }

        // --------------------------------------------------------------------------------------------- one sample

        /// <summary>The float64 samples of an array, as raw bits (a NaN's bits and a zero's sign are the point).</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>Its elements' bit patterns in C order.</returns>
        private static long[] Bits(NDArray a)
        {
            a.typecode.Should().Be(NPTypeCode.Double);
            var values = a.ToArray<double>();
            var bits = new long[values.Length];
            for (int i = 0; i < values.Length; i++)
                bits[i] = BitConverter.DoubleToInt64Bits(values[i]);
            return bits;
        }

        /// <summary>
        ///     One sample is NumPy's ramp value <c>0 * (stop - start) + start</c>, not <c>start</c>: a NaN when an end is
        ///     infinite or NaN, and +0.0 for a start of -0.0 (NumSharp wrote <c>start</c>).
        /// </summary>
        [TestMethod]
        public void Linspace_OneSample_IsTheRampTimesDeltaPlusStart()
        {
            double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);   // Python's float('nan')
            // np.linspace(1, inf, 1) -> [nan]: 0 * inf is the CPU's default NaN (its sign is the host's, so only NaN-ness
            // is asserted); with endpoint=False too (the step is delta / 1), where the float64 result is NumPy's view.
            double.IsNaN(np.linspace(1.0, double.PositiveInfinity, 1).GetDouble(0)).Should().BeTrue();
            double.IsNaN(np.linspace(double.PositiveInfinity, 1.0, 1).GetDouble(0)).Should().BeTrue();
            double.IsNaN(np.linspace(double.NegativeInfinity, double.PositiveInfinity, 1).GetDouble(0)).Should().BeTrue();
            var viewed = np.linspace(1.0, double.PositiveInfinity, 1, endpoint: false);
            double.IsNaN(viewed.GetDouble(0)).Should().BeTrue();
            viewed.flags.owndata.Should().BeFalse();
            // np.linspace(1, nan, 1).view(np.int64) -> [0x7ff8000000000000]: 0 * nan keeps the NaN's bits
            Bits(np.linspace(1.0, nan, 1)).Should().Equal(0x7FF8000000000000);
            // np.linspace(-0.0, 1, 1) -> [0.0] (+0.0: 0 * 1 + -0.0), with and without the endpoint
            Bits(np.linspace(-0.0, 1.0, 1)).Should().Equal(0L);
            Bits(np.linspace(-0.0, 1.0, 1, endpoint: false)).Should().Equal(0L);
            Bits(np.linspace(-0.0, -0.0, 1)).Should().Equal(0L);
            // the integer dtypes floor NumPy's value, then cast: NaN -> int32's minimum (x86's cvttsd2si), as NumPy's astype
            np.linspace(1.0, double.PositiveInfinity, 1, dtype: np.int32).Should().BeOfValues(int.MinValue);
            np.linspace(-1.5, 3.0, 1, dtype: np.int16).Should().BeOfValues((short)-2);
            np.linspace(1.0, double.PositiveInfinity, 1, dtype: np.@bool).Should().BeOfValues(true);
        }

        /// <summary>logspace's one exponent is linspace's, so an infinite or NaN stop gives NaN (NumSharp gave base ** start).</summary>
        [TestMethod]
        public void Logspace_OneSample_IsPowerOfTheLinspaceValue()
        {
            double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            // np.logspace(0, inf, 1) -> [nan]; np.logspace(0, nan, 1).view(np.int64) -> [0x7ff8000000000000]
            double.IsNaN(np.logspace(0.0, double.PositiveInfinity, 1).GetDouble(0)).Should().BeTrue();
            double.IsNaN(np.logspace(0.0, double.PositiveInfinity, 1, endpoint: false).GetDouble(0)).Should().BeTrue();
            Bits(np.logspace(0.0, nan, 1)).Should().Equal(0x7FF8000000000000);
            // finite ends: base ** start, as before
            Bits(np.logspace(2.0, 3.0, 1)).Should().Equal(BitConverter.DoubleToInt64Bits(100.0));
        }

        // ------------------------------------------------------------------------------- a step that underflows to 0

        /// <summary>
        ///     NumPy's gh-5437 branch: when <c>delta / div</c> underflows to zero, NumPy computes <c>(i / div) * delta +
        ///     start</c>, which keeps a subnormal range spread out; NumSharp computed <c>start + i * 0</c>, a constant run.
        /// </summary>
        [TestMethod]
        public void Linspace_UnderflowingStep_SpreadsTheSubnormalRange()
        {
            // np.linspace(0, 1e-322, 50).view(np.int64)
            Bits(np.linspace(0.0, 1e-322, 50)).Should().Equal(
                0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13,
                14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 19, 19, 20, 20);
            // np.linspace(0, 1e-322, 50, endpoint=False).view(np.int64)
            Bits(np.linspace(0.0, 1e-322, 50, endpoint: false)).Should().Equal(
                0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13,
                14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 18, 19, 19, 20);
            // np.linspace(5e-324, 0, 7).view(np.int64): a negative step that rounds to -0.0 takes the branch too
            Bits(np.linspace(5e-324, 0.0, 7)).Should().Equal(1, 1, 1, 1, 0, 0, 0);
            // a subnormal step that does not underflow keeps i * step + start: np.linspace(1e-320, 2e-320, 49)[:6]
            Bits(np.linspace(1e-320, 2e-320, 49)).Should().StartWith(new long[] { 2024, 2066, 2108, 2150, 2192, 2234 });
        }

        // ------------------------------------------------------------------------------------------------ text dtypes

        /// <summary>
        ///     A text dtype is NumPy's final <c>astype</c> of the float64 samples (no floor: it is not an integer dtype);
        ///     NumSharp refused it (<c>NotSupportedException</c>: a dtype asked for by storage lane).
        /// </summary>
        [TestMethod]
        public void Linspace_TextDtype_FormatsTheFloat64Samples()
        {
            // repr(np.linspace(0.0, 1.0, 5, dtype=...)) and friends, NumPy 2.4.2
            var u = np.linspace(0.0, 1.0, 5, dtype: "U5");
            u.ToString(true).Should().Be("array(['0.0', '0.25', '0.5', '0.75', '1.0'], dtype='<U5')");
            u.flags.owndata.Should().BeTrue();
            np.linspace(0.0, 1.0, 5, dtype: "S5").ToString(true)
                .Should().Be("array([b'0.0', b'0.25', b'0.5', b'0.75', b'1.0'], dtype='|S5')");
            np.linspace(0.0, 1.0, 5, dtype: np.dtype("T")).ToString(true)
                .Should().Be("array(['0.0', '0.25', '0.5', '0.75', '1.0'], dtype=StringDType())");
            // a fixed width cuts the text: '10.0' -> '10.'
            np.linspace(1.0, 10.0, 3, dtype: "U3").ToString(true).Should().Be("array(['1.0', '5.5', '10.'], dtype='<U3')");
            // the unsized 'U' takes the width of the float64 text (32)
            np.linspace(-1e300, 1e300, 3, dtype: "U").ToString(true)
                .Should().Be("array(['-1e+300', '0.0', '1e+300'], dtype='<U32')");
            np.linspace(0.0, 1.0, 4, endpoint: false, dtype: "U5").ToString(true)
                .Should().Be("array(['0.0', '0.25', '0.5', '0.75'], dtype='<U5')");
            np.linspace(0.0, 1.0, 1, dtype: "U5").ToString(true).Should().Be("array(['0.0'], dtype='<U5')");
            np.linspace(0.0, 1.0, 0, dtype: "U5").ToString(true).Should().Be("array([], dtype='<U5')");
        }

        /// <summary>
        ///     geomspace's text dtypes fail NumPy's statements after the zero check: <c>result_type(start, stop,
        ///     float(num), zeros((), dtype))</c> has no common DType for StringDType; S and U promote to text and
        ///     <c>sign(start)</c> has no string loop. Both come before the sample count and the axis. NumSharp computed the
        ///     float64 progression and formatted it.
        /// </summary>
        [TestMethod]
        public void Geomspace_TextDtype_FailsNumPysStatements_BeforeTheCountAndTheAxis()
        {
            const string stringDTypeFloat =
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Float64DType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single " +
                "array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Float64DType'>, " +
                "<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes._PyFloatDType'>, <class 'numpy.dtypes.StringDType'>)";
            const string stringDTypeComplex =
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Complex128DType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single " +
                "array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Complex128DType'>, " +
                "<class 'numpy.dtypes.Complex128DType'>, <class 'numpy.dtypes._PyFloatDType'>, <class 'numpy.dtypes.StringDType'>)";
            const string signStr = "ufunc 'sign' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> None";
            const string signBytes = "ufunc 'sign' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None";
            var T = np.dtype("T");
            var c1 = new System.Numerics.Complex(1, 0);
            var c1000 = new System.Numerics.Complex(1000, 0);

            // np.geomspace(1.0, 1000.0, 4, dtype=StringDType()) -> DTypePromotionError
            new Action(() => np.geomspace(1.0, 1000.0, 4, dtype: T)).Should().Throw<DTypePromotionError>().WithMessage(stringDTypeFloat);
            new Action(() => np.geomspace(c1, c1000, 4, dtype: T)).Should().Throw<DTypePromotionError>().WithMessage(stringDTypeComplex);
            // np.geomspace(1.0, 1000.0, 4, dtype='U5' / 'S5') -> UFuncTypeError (NumSharp's TypeError), complex ends too
            new Action(() => np.geomspace(1.0, 1000.0, 4, dtype: "U5")).Should().ThrowExactly<TypeError>().WithMessage(signStr);
            new Action(() => np.geomspace(1.0, 1000.0, 4, dtype: "S5")).Should().ThrowExactly<TypeError>().WithMessage(signBytes);
            new Action(() => np.geomspace(c1, c1000, 4, dtype: "U5")).Should().ThrowExactly<TypeError>().WithMessage(signStr);
            // before the count and the axis: np.geomspace(1.0, 1000.0, -1, dtype=StringDType()), (..., 4, dtype='U5', axis=3),
            // (..., 0, dtype='U5')
            new Action(() => np.geomspace(1.0, 1000.0, -1, dtype: T)).Should().Throw<DTypePromotionError>().WithMessage(stringDTypeFloat);
            new Action(() => np.geomspace(1.0, 1000.0, -1, dtype: "U5")).Should().ThrowExactly<TypeError>().WithMessage(signStr);
            new Action(() => np.geomspace(1.0, 1000.0, 4, dtype: "U5", axis: 3)).Should().ThrowExactly<TypeError>().WithMessage(signStr);
            new Action(() => np.geomspace(1.0, 1000.0, 0, dtype: "U5")).Should().ThrowExactly<TypeError>().WithMessage(signStr);
            // after the zero check: np.geomspace(0.0, 1000.0, 4, dtype=StringDType()) -> the zero's ValueError
            new Action(() => np.geomspace(0.0, 1000.0, 4, dtype: T))
                .Should().Throw<ValueError>().WithMessage("Geometric sequence cannot include zero");
        }

        // ------------------------------------------------------------------------------------- the axis' moveaxis view

        /// <summary>
        ///     geomspace ends with <c>if axis != 0: result = moveaxis(result, 0, axis)</c> and <c>result.astype(dtype,
        ///     copy=False)</c>: axis -1 returns a view that does not own its data unless the dtype cast copies it. NumSharp
        ///     always returned an owner.
        /// </summary>
        [TestMethod]
        public void Geomspace_AxisMinusOne_IsNumPysMoveaxisView()
        {
            var c1 = new System.Numerics.Complex(1, 0);
            var c1000 = new System.Numerics.Complex(1000, 0);
            // np.geomspace(1.0, 1000.0, 4, axis=-1).flags.owndata -> False (values [1, 10, 100, 1000], as axis 0)
            var a = np.geomspace(1.0, 1000.0, 4, axis: -1);
            a.flags.owndata.Should().BeFalse();
            a.Should().BeOfValues(1.0, 10.0, 100.0, 1000.0);
            np.geomspace(1.0, 1000.0, 4, axis: 0).flags.owndata.Should().BeTrue();
            // the dtype already the result's: no copy (float64 for real ends, complex128 for complex ones)
            np.geomspace(1.0, 1000.0, 4, dtype: np.float64, axis: -1).flags.owndata.Should().BeFalse();
            np.geomspace(1.0, 1000.0, 4, dtype: np.complex128, axis: -1).flags.owndata.Should().BeFalse();
            np.geomspace(c1, c1000, 4, axis: -1).flags.owndata.Should().BeFalse();
            np.geomspace(1.0, 1000.0, 0, axis: -1).flags.owndata.Should().BeFalse();
            // a cast copies: np.geomspace(1.0, 1000.0, 4, dtype=np.float32, axis=-1).flags.owndata -> True
            np.geomspace(1.0, 1000.0, 4, dtype: np.float32, axis: -1).flags.owndata.Should().BeTrue();
            np.geomspace(c1, c1000, 4, dtype: np.float64, axis: -1).flags.owndata.Should().BeTrue();
        }
    }
}
