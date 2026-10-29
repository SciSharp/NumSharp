using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.arange</c> as NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>): the length from Python's arithmetic on the
    ///     bounds, <c>start</c> and <c>start + step</c> stored through the dtype's <c>setitem</c>, the rest filled by its
    ///     <c>fill</c> in the dtype's own arithmetic. Every expected value is NumPy 2.4.2's (win-amd64), floats as raw bits.
    ///     The oracle tier <c>arange_full.jsonl</c> holds the wide matrix; these pin the classes of difference the old ramp
    ///     (<c>start + i * step</c> computed in double) got wrong.
    /// </summary>
    [TestClass]
    public class ArangeNumPyPortTests
    {
        /// <summary>The raw bits of a float64 array's elements.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>The bits.</returns>
        private static ulong[] Bits64(NDArray a) => a.ToArray<double>().Select(BitConverter.DoubleToUInt64Bits).ToArray();

        /// <summary>
        ///     A float64 ramp is <c>start + i * delta</c> with <c>delta = (start + step) - start</c>, not
        ///     <c>start + i * step</c>: <c>np.arange(0.1, 2, 0.3)</c> gives <c>1.0000000000000002</c> at index 3 (the old ramp
        ///     gave <c>0.9999999999999999</c>), and a start past 2**53 whose step rounds away is a constant ramp.
        /// </summary>
        [TestMethod]
        public void Float64_RampIsStartPlusIDelta()
        {
            np.arange(0.1, 2, 0.3).ToArray<double>().Should().Equal(
                0.1, 0.4, 0.7000000000000001, 1.0000000000000002, 1.3000000000000003, 1.6000000000000003, 1.9000000000000004);
            np.arange(1e16, 1e16 + 4, 1.0).ToArray<double>().Should().Equal(1e16, 1e16, 1e16, 1e16);
            Bits64(np.arange(0.1, 1.0, 0.1)).Should().Equal(
                4591870180066957722, 4596373779694328218, 4599075939470750516, 4600877379321698714, 4602678819172646912,
                4603579539098121011, 4604480259023595111, 4605380978949069210, 4606281698874543309);
        }

        /// <summary>
        ///     float32 fills in float32 (<c>FLOAT_fill</c>) and float16 in float32 rounded per element (<c>HALF_fill</c>,
        ///     whose third element of <c>np.arange(0.1, 0.55, 0.1, dtype=float16)</c> is the tie-to-even 0.2998).
        /// </summary>
        [TestMethod]
        public void Float32AndFloat16_FillInTheirOwnArithmetic()
        {
            np.arange(0.1, 2, 0.3, np.float32).ToArray<float>().Select(BitConverter.SingleToUInt32Bits).Should().Equal(
                1036831949u, 1053609165u, 1060320052u, 1065353216u, 1067869799u, 1070386381u, 1072902964u);
            np.arange(0.1, 0.55, 0.1, np.float16).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).Should().Equal(
                (ushort)11878, (ushort)12902, (ushort)13516, (ushort)13926, (ushort)14336);
            // a C# float overload is the double overload with dtype float32
            np.arange(0.1f, 2f, 0.3f).dtype.Should().Be(np.float32);
        }

        /// <summary>
        ///     An integer ramp stores <c>start</c> and <c>start + step</c> truncated (a fractional step can make delta 0),
        ///     then wraps in the dtype past them (C's fill); <c>start</c> or <c>start + step</c> outside the dtype is NumPy's
        ///     OverflowError, raised by the store.
        /// </summary>
        [TestMethod]
        public void Integer_StoresTruncateAndRampsWrap()
        {
            np.arange(0, 5, 0.5, np.int32).ToArray<int>().Should().Equal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            np.arange(5, 0, -0.5, np.int32).ToArray<int>().Should().Equal(5, 4, 3, 2, 1, 0, -1, -2, -3, -4);
            np.arange(100, 150, 10, np.int8).ToArray<sbyte>().Should().Equal((sbyte)100, (sbyte)110, (sbyte)120, (sbyte)-126, (sbyte)-116);
            np.arange(5, -5, -2, np.uint8).ToArray<byte>().Should().Equal((byte)5, (byte)3, (byte)1, (byte)255, (byte)253);

            new Action(() => np.arange(300, 310, 1, np.int8)).Should().Throw<OverflowException>()
                .WithMessage("Python integer 300 out of bounds for int8");
            new Action(() => np.arange(120, 140, 10, np.int8)).Should().Throw<OverflowException>()
                .WithMessage("Python integer 130 out of bounds for int8");
            new Action(() => np.arange(-1, 3, 1, np.uint8)).Should().Throw<OverflowException>()
                .WithMessage("Python integer -1 out of bounds for uint8");
        }

        /// <summary>
        ///     Python-int bounds are exact at every magnitude (the old double path made <c>np.arange(2**60, 2**60 + 3)</c>
        ///     empty), and a quotient whose ceiling is exactly 2**63 is NumPy's x86 empty array.
        /// </summary>
        [TestMethod]
        public void PythonInts_AreExact()
        {
            long b = 1L << 60;
            np.arange(b, b + 3).ToArray<long>().Should().Equal(b, b + 1, b + 2);
            np.arange((1L << 53) + 1, (1L << 53) + 10, 2).ToArray<long>().Should().Equal(
                9007199254740993, 9007199254740995, 9007199254740997, 9007199254740999, 9007199254741001);
            np.arange(long.MinValue, 0).size.Should().Be(0);
            np.arange(0.0, 9223372036854775808.0, 1.0).size.Should().Be(0);
            np.arange(0, long.MaxValue, 1L << 62).ToArray<long>().Should().Equal(0, 1L << 62);
        }

        /// <summary>
        ///     The length's own cases: a quotient that underflows to zero counts one element for +0 and none for -0, and so
        ///     does a step of +-inf; a step far below 1 is legal (the old <c>|step| &lt; 1e-15</c> refusal is gone).
        /// </summary>
        [TestMethod]
        public void Length_UnderflowAndDivideByInf()
        {
            np.arange(0.0, 1e-320, 1e300).ToArray<double>().Should().Equal(0.0);
            np.arange(0.0, -1e-320, 1e300).size.Should().Be(0);
            np.arange(0.0, 5.0, double.PositiveInfinity).ToArray<double>().Should().Equal(0.0);
            np.arange(0.0, 5.0, double.NegativeInfinity).size.Should().Be(0);
            np.arange(0.0, 1e-14, 1e-15).size.Should().Be(10);
        }

        /// <summary>
        ///     NumPy's errors, in NumPy's order: a dtype without a fill function first (even with a zero step), then the
        ///     length (Python's ZeroDivisionError, a NaN, past npy_intp), then the allocation, then bool's fill.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPys()
        {
            new Action(() => np.arange(0, 5, 0)).Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            new Action(() => np.arange(0.0, 5.0, 0.0)).Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
            new Action(() => np.arange(0.0, double.NaN, 1.0)).Should().Throw<ValueError>().WithMessage("arange: cannot compute length");
            new Action(() => np.arange(0.0, double.PositiveInfinity, 1.0)).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed size exceeded");
            new Action(() => np.arange(0.0, 18446744073709551616.0, 1.0)).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed size exceeded");
            new Action(() => np.arange(0.0, 4611686018427387904.0, 1.0)).Should().Throw<ValueError>()
                .WithMessage("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
            new Action(() => np.arange(0, 5, 1, np.dtype("U3"))).Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.StrDType'>.");
            new Action(() => np.arange(0, 5, 0, np.dtype("S3"))).Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.BytesDType'>.");
            new Action(() => np.arange(0, 5, 1, np.dtypes.StringDType())).Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.StringDType'>.");
            new Action(() => np.arange(0, 3, 1, np.@bool)).Should().Throw<TypeError>()
                .WithMessage("arange() is only supported for booleans when the result has at most length 2.");
            np.arange(0.5, 2, 1, np.@bool).ToArray<bool>().Should().Equal(true, true);
        }

        /// <summary>
        ///     The result is a fresh owning C-contiguous array of the requested class: a lane-sharing C-type class keeps its
        ///     descriptor (<c>np.intc</c>), an omitted dtype is NumPy's discovery (float64 for floats, int64 for ints).
        /// </summary>
        [TestMethod]
        public void Result_IsAFreshArrayOfTheRequestedClass()
        {
            var a = np.arange(0, 4, 1, np.intc);
            a.dtype.Should().Be(np.intc);
            a.ToArray<int>().Should().Equal(0, 1, 2, 3);
            a.Shape.IsContiguous.Should().BeTrue();
            a.Storage.IsView.Should().BeFalse();
            np.arange(0.0, 3.0, 1.0, null).dtype.Should().Be(np.float64);
            np.arange(3L).dtype.Should().Be(np.int64);
            np.arange(0, 3, 1, np.complex128).ToArray<Complex>().Should().Equal(Complex.Zero, Complex.One, new Complex(2, 0));
        }
    }
}
