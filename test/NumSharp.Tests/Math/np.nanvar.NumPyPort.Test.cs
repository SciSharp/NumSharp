using System;
using System.Collections.Generic;
using System.Numerics;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.nanvar</c> / <c>np.nanstd</c> as NumPy's <c>nanvar</c> / <c>nanstd</c>
    ///     (<c>numpy/lib/_nanfunctions_impl.py</c>), pinned to the bit: a K-order copy with its NaNs zeroed, the non-NaN
    ///     count, the mean, the deviations rounded into the input's dtype and zeroed at the NaNs, their squares (the real part
    ///     of the complex product for complex), the sum, <c>/ (count - ddof)</c>, and NaN wherever <c>count - ddof &lt;= 0</c>.
    ///     Every expected string is NumPy 2.4.2's output for the same call, in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format (<c>dtype (shape) bits…</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The first test was <c>[OpenBugs]</c> group 4 of <c>OpenBugs.Reductions.cs</c> until the port (plan
    ///     <c>docs/plans/api-parameter-parity.md</c> item A3, 2026-10-05). The old implementation was a two-pass moment loop
    ///     in double with its own shortcuts: a 1-D axis reduction came back <c>(1,)</c>, keepdims was dropped for one
    ///     element and for every integer input, <c>ddof</c> at or past the count gave 0 instead of NaN, a ±inf element gave
    ///     0, float16 / float32 were summed in double, and there was no <c>dtype=</c>. The other tests pin the rules the
    ///     port reproduces.
    ///     </para>
    ///     <para>
    ///     The oracle tier <c>nanvarstd.jsonl</c> (13,382 cases, zero excuses) covers the same ground across every layout
    ///     and dtype.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_nanvar_nanstd_numpy_port_tests
    {
        /// <summary>
        ///     <c>np.nanstd</c> and <c>np.nanvar</c> of a 1-D array along <c>axis=0</c> return a 0-d result (shape
        ///     <c>(1,)</c> with <c>keepdims</c>), and a one-element input keeps <c>keepdims</c>: <c>(1,)</c>. (Formerly
        ///     <c>OpenBugsReductionTests.NanStdVar_1DAxis_ReturnNumPysShape</c>.)
        /// </summary>
        [TestMethod]
        public void NanStdVar_1DAxis_ReturnNumPysShape()
        {
            var fails = new List<string>();
            var v = np.array(new[] { 1.0, 2.0, 3.0 });
            Check(fails, "nanstd f64 [1,2,3] axis0", () => np.nanstd(v, 0), "float64 () 3fea20bd700c2c3e");
            Check(fails, "nanvar f64 [1,2,3] axis0 keepdims", () => np.nanvar(v, 0, true), "float64 (1,) 3fe5555555555555");
            Check(fails, "nanstd f64 [5] keepdims", () => np.nanstd(np.array(new[] { 5.0 }), keepdims: true), "float64 (1,) 0000000000000000");
            Check(fails, "nanvar f64 [5] axis0 keepdims", () => np.nanvar(np.array(new[] { 5.0 }), 0, true), "float64 (1,) 0000000000000000");
            AssertNoMismatch(fails, "NumPy reduces the only axis of a 1-D array to a 0-d result");
        }

        /// <summary>
        ///     A slice with no degrees of freedom left — all NaN, or <c>ddof</c> at or past its non-NaN count — is NaN,
        ///     whatever the division made of it. np.var of the same values gives +inf at <c>ddof == n</c>; nanvar replaces it.
        /// </summary>
        [TestMethod]
        public void NanVar_NoDegreesOfFreedom_IsNaN_NotInfinity()
        {
            var fails = new List<string>();
            Check(fails, "nanvar f64 [1,2,nan] ddof2", () => np.nanvar(Of(1, 2, double.NaN), ddof: 2), "float64 () nan");
            Check(fails, "var f64 [1,2] ddof2 (np.var: +inf)", () => np.var(Of(1, 2), ddof: 2), "float64 () 7ff0000000000000");
            Check(fails, "nanvar f64 [[1,nan],[3,4]] axis1 ddof1",
                  () => np.nanvar(np.array(new[,] { { 1.0, double.NaN }, { 3.0, 4.0 } }), 1, ddof: 1), "float64 (2,) nan 3fe0000000000000");
            Check(fails, "nanstd f32 [[1,nan],[3,4]] axis1 ddof1",
                  () => np.nanstd(np.array(new[,] { { 1f, float.NaN }, { 3f, 4f } }), 1, ddof: 1), "float32 (2,) nan 3f3504f3");
            Check(fails, "nanvar f64 all-nan", () => np.nanvar(Of(double.NaN, double.NaN)), "float64 () nan");
            Check(fails, "nanvar f32 all-nan slice axis0",
                  () => np.nanvar(np.array(new[,] { { float.NaN, 1f }, { float.NaN, 3f } }), 0), "float32 (2,) nan 3f800000");
            Check(fails, "nanvar f64 [5] ddof1", () => np.nanvar(Of(5), ddof: 1), "float64 () nan");
            Check(fails, "nanvar f64 [1,2,3] ddof5 (a negative divisor)", () => np.nanvar(Of(1, 2, 3), ddof: 5), "float64 () nan");
            AssertNoMismatch(fails, "nanvar overwrites every slice whose count - ddof <= 0 with NaN");
        }

        /// <summary>
        ///     The NaN of a slice without degrees of freedom is NumPy's <c>np.nan</c>, the POSITIVE quiet NaN, in the
        ///     result's dtype (.NET's <c>double.NaN</c> is the negative one).
        /// </summary>
        [TestMethod]
        public void NanVar_NoDegreesOfFreedom_IsThePositiveQuietNaN()
        {
            BitConverter.DoubleToInt64Bits(np.nanvar(Of(1.0), ddof: 1).GetDouble()).Should().Be(0x7FF8000000000000L);
            BitConverter.SingleToInt32Bits(np.nanvar(np.array(new[] { 1f }), ddof: 1).GetSingle()).Should().Be(0x7FC00000);
            BitConverter.HalfToInt16Bits(np.nanvar(np.array(new[] { (Half)1 }), ddof: 1).GetHalf()).Should().Be((short)0x7E00);
        }

        /// <summary>
        ///     ±inf is not NaN: it stays in the slice, its deviation is <c>inf - inf</c>, and the variance is NaN — the old
        ///     implementation answered 0. A complex element with an infinite part behaves the same way.
        /// </summary>
        [TestMethod]
        public void NanVar_InfiniteElement_IsNaN()
        {
            var fails = new List<string>();
            Check(fails, "nanvar f64 [1,inf,2]", () => np.nanvar(Of(1, double.PositiveInfinity, 2)), "float64 () nan");
            Check(fails, "nanvar f16 [1,inf,nan]", () => np.nanvar(OfHalf(1, double.PositiveInfinity, double.NaN)), "float16 () nan");
            Check(fails, "nanstd c128 [1, 2+infj]", () => np.nanstd(np.array(new[] { new Complex(1, 0), new Complex(2, double.PositiveInfinity) })), "float64 () nan");
            AssertNoMismatch(fails, "inf - inf is NaN, and nanvar only skips NaN");
        }

        /// <summary>
        ///     <c>dtype=</c> is the loop of both sums, but the deviations are rounded into the INPUT's dtype before they are
        ///     squared (NumPy's <c>subtract(arr, avg, out=arr, casting='unsafe')</c>): a float32 input with
        ///     <c>dtype=float64</c> squares float32 deviations, so its nanvar differs from np.var of the same values (which
        ///     squares float64 ones). A complex <c>dtype=</c> over a real input sums complex; a real <c>dtype=</c> over a
        ///     complex input keeps the real part of the mean.
        /// </summary>
        [TestMethod]
        public void NanVar_DtypeArgument_RoundsTheDeviationsIntoTheInputDtype()
        {
            var fails = new List<string>();
            var x32 = np.array(new[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, float.NaN, 7.7f, 1e-3f });
            var x32NoNaN = np.array(new[] { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 7.7f, 1e-3f });
            Check(fails, "nanvar f32 dtype=f64", () => np.nanvar(x32, dtype: np.float64), "float64 () 401818e6e1e49249");
            Check(fails, "var f32 (no nan) dtype=f64 (np.var squares float64)", () => np.var(x32NoNaN, dtype: np.float64), "float64 () 401818e6e25874a7");
            Check(fails, "nanvar f32 (2,4) axis1 dtype=f64", () => np.nanvar(x32.reshape(2, 4), 1, dtype: np.float64), "float64 (2,) 3ff8333338000000 4024f7ac02aaaaab");
            Check(fails, "nanstd f64 dtype=f32", () => np.nanstd(x32.astype(np.float64), dtype: np.float32), "float32 () 401d15b0");
            Check(fails, "nanvar f64 dtype=c128", () => np.nanvar(x32.astype(np.float64), dtype: np.complex128), "complex128 () 401818e6e25874a6 0000000000000000");
            Check(fails, "nanvar c128 dtype=f64",
                  () => np.nanvar(np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(double.NaN, 0), new Complex(2.5, 0.5) }), dtype: np.float64),
                  "float64 () 4003c71c71c71c72");
            Check(fails, "nanvar f16 dtype=f32", () => np.nanvar(OfHalf(1, 2, 4, double.NaN, 8), dtype: np.float32), "float32 () 40e60000");
            AssertNoMismatch(fails, "nanvar's deviations live in the input's dtype (out=arr)");
        }

        /// <summary>
        ///     A complex square is the real part of NumPy's complex product <c>x * conj(x)</c> (one fused rounding of
        ///     <c>re*re + round(im*im)</c> in <c>simd_cmul</c>), not np.var's <c>re² + im²</c>: on these values the two
        ///     differ in the last bits, and nanvar and var of the same NaN-free array differ with them.
        /// </summary>
        [TestMethod]
        public void NanVar_Complex_SquaresWithTheComplexProduct()
        {
            ulong[] bits =
            {
                0xc0786cd0cd813aec, 0xbfa9de22534ad5fc, 0x3f3f95b3ae9dc170, 0xbfc767447c2a6cd4, 0xbf2f4487e08cf678,
                0x3f41b640e1bc1a85, 0x408dee11d2db7c85, 0x3f93d0b798da79bf, 0xc068f9ab0a93843a, 0xc00591d30595a06c,
            };
            var values = new Complex[5];
            for (int i = 0; i < 5; i++)
                values[i] = new Complex(BitConverter.UInt64BitsToDouble(bits[2 * i]), BitConverter.UInt64BitsToDouble(bits[2 * i + 1]));
            var c = np.array(values);
            var fails = new List<string>();
            Check(fails, "nanvar c128", () => np.nanvar(c), "float64 () 410a70b4bba6f57b");
            Check(fails, "var c128 (re² + im²)", () => np.var(c), "float64 () 410a70b4bba6f57d");
            Check(fails, "nanstd c128 ddof1", () => np.nanstd(c, ddof: 1), "float64 () 408042ad8b193a49");
            AssertNoMismatch(fails, "nanvar squares with the complex multiply");
        }

        /// <summary>
        ///     A float or complex input with a non-inexact <c>dtype=</c> raises NumPy's TypeError, before the axis is checked.
        /// </summary>
        [TestMethod]
        public void NanVar_InexactInputWithIntegerDtype_RaisesTypeError_BeforeTheAxis()
        {
            ExpectMessage(() => np.nanvar(Of(1, 2), dtype: np.int32), "If a is inexact, then dtype must be inexact");
            ExpectMessage(() => np.nanvar(Of(1, 2), 5, dtype: np.int32), "If a is inexact, then dtype must be inexact");
            ExpectMessage(() => np.nanstd(Of(1, 2), dtype: np.@bool), "If a is inexact, then dtype must be inexact");
        }

        /// <summary>
        ///     On a 0-d float input, axis 0 and -1 reduce flat: nanvar's first statement that reads the axis is np.sum, and
        ///     ufunc.reduce reads them as "every axis". Any other axis is AxisError. An integer 0-d input is np.var, which
        ///     raises for axis 0 too.
        /// </summary>
        [TestMethod]
        public void NanVar_ZeroD_AxisZeroOrMinusOne_ReducesFlat()
        {
            var fails = new List<string>();
            Check(fails, "nanvar 0-d f64 axis0", () => np.nanvar(NDArray.Scalar(2.5), 0), "float64 () 0000000000000000");
            Check(fails, "nanvar 0-d f64 axis-1 keepdims", () => np.nanvar(NDArray.Scalar(2.5), -1, true), "float64 () 0000000000000000");
            Check(fails, "nanstd 0-d f32 nan axis0", () => np.nanstd(NDArray.Scalar(float.NaN), 0), "float32 () nan");
            AssertNoMismatch(fails, "ufunc.reduce reads axis 0 / -1 of a 0-d array as every axis");
            ExpectMessage(() => np.nanvar(NDArray.Scalar(2.5), 1), "axis 1 is out of bounds for array of dimension 0");
            ExpectMessage(() => np.nanvar(NDArray.Scalar(3), 0), "axis 0 is out of bounds for array of dimension 0");
        }

        /// <summary>
        ///     An integer or bool input has no NaN: nanvar is np.var with the same keywords (keepdims included — the old
        ///     implementation dropped it), and nanstd is np.std (an integer <c>dtype=</c> raises for an array result and
        ///     truncates a scalar one).
        /// </summary>
        [TestMethod]
        public void NanVar_IntegerInput_IsNpVar_KeywordsIncluded()
        {
            var i = np.array(new[,] { { 1, 2, 3 }, { 4, 6, 9 } });
            var fails = new List<string>();
            Check(fails, "nanvar i32 keepdims", () => np.nanvar(i, keepdims: true), "float64 (1,1) 401c8e38e38e38e3");
            Check(fails, "nanvar i32 axis1 keepdims", () => np.nanvar(i, 1, true), "float64 (2,1) 3fe5555555555555 4010e38e38e38e39");
            Check(fails, "nanvar i32 dtype=f32", () => np.nanvar(i, dtype: np.float32), "float32 () 40e471c8");
            Check(fails, "nanstd i32 dtype=i32 (scalar: truncated)", () => np.nanstd(i, dtype: np.int32), "int32 () 2");
            Check(fails, "nanvar bool", () => np.nanvar(np.array(new[] { true, false, true })), "float64 () 3fcc71c71c71c71d");
            AssertNoMismatch(fails, "nanvar of an array without NaN is np.var");
            ExpectMessage(() => np.nanstd(i, 0, dtype: np.int32),
                "Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     float16 sums in float16 (NumPy's HALF loops, through the eager SIMD half kernels), not in double: the old
        ///     implementation accumulated in double and was one ULP off here.
        /// </summary>
        [TestMethod]
        public void NanVar_Float16_SumsInFloat16()
        {
            var fails = new List<string>();
            Check(fails, "nanvar f16 0.1+0.2i", () => np.nanvar(OfHalf(0.1, 0.3, 0.5, 0.7, 0.9)), "float16 () 2d1e");
            Check(fails, "nanvar f16 with nan", () => np.nanvar(OfHalf(0.1, 0.3, double.NaN, 0.7, 0.9, 1.1, 1.3)), "float16 () 31b8");
            Check(fails, "nanstd f16 (2,3) axis1",
                  () => np.nanstd(OfHalf(1, 2, double.NaN, 4, 6, 9.5).reshape(2, 3) * NDArray.Scalar((Half)1.7), 1), "float16 (2,) 3acd 43bb");
            AssertNoMismatch(fails, "float16 nanvar is computed in float16");
        }

        /// <summary>
        ///     An empty input has a count of 0: the result is NaN in the input's dtype (float64 for complex) with the
        ///     reduction's shape, and an empty result where the kept axes are empty.
        /// </summary>
        [TestMethod]
        public void NanVar_EmptyInput_IsNaNOfTheInputDtype()
        {
            var fails = new List<string>();
            Check(fails, "nanvar f16 (0,)", () => np.nanvar(np.zeros(new Shape(0), np.float16)), "float16 () nan");
            Check(fails, "nanvar f32 (0,3) axis0", () => np.nanvar(np.zeros(new Shape(0, 3), np.float32), 0), "float32 (3,) nan nan nan");
            Check(fails, "nanvar f64 (3,0) axis0", () => np.nanvar(np.zeros(new Shape(3, 0)), 0), "float64 (0,) ");
            Check(fails, "nanstd c128 (0,) keepdims", () => np.nanstd(np.zeros(new Shape(0), np.complex128), keepdims: true), "float64 (1,) nan");
            AssertNoMismatch(fails, "an empty slice has no degrees of freedom");
        }

        /// <summary>
        ///     Both sums run over NumPy's K-order copy of the input, so on a permuted (and reversed) 3-D view the schedule is
        ///     that copy's: the digests are NumPy's bytes.
        /// </summary>
        [TestMethod]
        public void NanVar_PermutedInput_SumsInTheKOrderCopy()
        {
            var x = (np.arange(24 * 30 * 40).astype(np.float64) % 997) * 1.000123 + 1000.0;
            x["::13"] = NDArray.Scalar(double.NaN);
            var p = x.reshape(24, 30, 40).transpose(new[] { 2, 0, 1 });
            Sha256(np.nanvar(p, 2)).Should().Be("5fe5c996359d01f055bfdd345953776f6cb1370680bc0a21453e199e056cebbb");
            Sha256(np.nanstd(p, 0, ddof: 1)).Should().Be("77e2ab9d559f41d61a81b59f6620f582d89c0d576dfca6c032e771a6fd17b77b");
            Describe(np.nanvar(p)).Should().Be("float64 () 40f41e25a6e5a4f1");
            Sha256(np.nanvar(p[":, ::-1, :"], 1)).Should().Be("e0ecfbea9144f5c46f0a8c6b63b0fe55c9b4c79610701ce13840814aedd1afe7");
        }

        // ======================================================================================================
        //  Helpers
        // ======================================================================================================

        /// <summary>Asserts that <paramref name="call"/> throws with NumPy's message (the .NET type is the house's).</summary>
        /// <param name="call">The call that must raise.</param>
        /// <param name="message">NumPy's verbatim message.</param>
        private static void ExpectMessage(Action call, string message)
        {
            var e = call.Should().Throw<Exception>().Which;
            string text = e.Message;
            int p = text.IndexOf(" (Parameter '", StringComparison.Ordinal);
            if (p >= 0)
                text = text.Substring(0, p);
            text.Should().Be(message);
        }

        /// <summary>The SHA-256 of an array's values in logical C order, as lowercase hex.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The digest.</returns>
        private static unsafe string Sha256(NDArray a)
        {
            var c = np.ascontiguousarray(a);
            var bytes = new ReadOnlySpan<byte>((void*)c.Address, checked((int)(c.size * c.dtypesize)));
            return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant();
        }

        /// <summary>Builds a 1-D float64 array from literal values (an integer literal widens to its exact double).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous float64 array.</returns>
        private static NDArray Of(params double[] values) => np.array(values);

        /// <summary>
        ///     Builds a 1-D float16 array from literal values, each rounded to the nearest float16 (NumPy's
        ///     <c>np.array([...], np.float16)</c> rounds the same way).
        /// </summary>
        /// <param name="values">The elements, in order, as doubles.</param>
        /// <returns>A fresh C-contiguous float16 array.</returns>
        private static NDArray OfHalf(params double[] values) => np.array(Array.ConvertAll(values, v => (Half)v));
    }
}
