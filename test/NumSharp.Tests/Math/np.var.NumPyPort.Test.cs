using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.var</c> / <c>np.std</c> as NumPy's <c>_var</c> / <c>_std</c> (<c>numpy/_core/_methods.py</c>), pinned to the
    ///     bit: the mean by <c>umr_sum</c> and the count, <c>x = arr - mean</c>, <c>x * x</c> (or <c>real² + imag²</c> of a
    ///     complex <c>x</c>), <c>umr_sum</c> again, then <c>/ max(n - ddof, 0)</c>; <c>_std</c> takes the square root. Every
    ///     expected string is NumPy 2.4.2's output for the same call, in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format (<c>dtype (shape) bits…</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The first seven tests were <c>[OpenBugs]</c> group 5 of <c>OpenBugs.Reductions.cs</c> (2026-10-04) until the port
    ///     (plan <c>docs/plans/api-parameter-parity.md</c> item A2, 2026-10-05). The old engine answered one element per
    ///     output without reading it, kept float64 for float16 / float32 at a degenerate size, divided by a negative
    ///     <c>n - ddof</c>, and ignored <c>dtype=</c> at every size. The other tests pin what the port changed beyond those
    ///     cells: <c>dtype=</c> as the loop of both sums, NumPy's errors, the layout of <c>x</c> on a permuted input, the
    ///     float16 / decimal / char routes, and the chunked deviation sums of large inputs.
    ///     </para>
    ///     <para>
    ///     The oracle tier <c>varstd.jsonl</c> (16,108 cases, zero excuses) covers the same ground across every layout and
    ///     dtype; <c>FuzzCorpusTests.VarStdChunked</c> replays it with the chunked routes forced.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_var_std_numpy_port_tests
    {
        // ======================================================================================================
        //  Degenerate sizes (formerly OpenBugs group 5): NumPy's arithmetic runs at every size.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of ONE element (a one-element 1-D array or a 0-d array, flat, over its one axis,
        ///     or with <c>keepdims</c>) is NaN when the element is NaN or ±inf: NumPy computes <c>x - mean</c>
        ///     (<c>nan - nan</c>, <c>inf - inf</c>). A finite element gives 0. A missing-value marker therefore never turns
        ///     into "no spread".
        /// </summary>
        [TestMethod]
        public void VarStd_OneElement_NonFiniteValue_IsNaN()
        {
            var fails = new List<string>();
            Check(fails, "var f64 [nan]", () => np.var(Of(double.NaN)), "float64 () nan");
            Check(fails, "var f64 [inf]", () => np.var(Of(double.PositiveInfinity)), "float64 () nan");
            Check(fails, "var f64 [-inf]", () => np.var(Of(double.NegativeInfinity)), "float64 () nan");
            Check(fails, "std f64 [inf]", () => np.std(Of(double.PositiveInfinity)), "float64 () nan");
            Check(fails, "var f64 0-d nan", () => np.var(NDArray.Scalar(double.NaN)), "float64 () nan");
            Check(fails, "std f64 0-d -inf", () => np.std(NDArray.Scalar(double.NegativeInfinity)), "float64 () nan");
            Check(fails, "var f64 [nan] axis0", () => np.var(Of(double.NaN), 0), "float64 () nan");
            Check(fails, "var f64 [inf] keepdims", () => np.var(Of(double.PositiveInfinity), keepdims: true), "float64 (1,) nan");
            Check(fails, "var c128 [nan+0j]", () => np.var(OfComplex(new Complex(double.NaN, 0))), "float64 () nan");
            Check(fails, "std c128 [0+infj]", () => np.std(OfComplex(new Complex(0, double.PositiveInfinity))), "float64 () nan");
            Check(fails, "var f64 [5] (finite)", () => np.var(Of(5.0)), "float64 () 0000000000000000");
            AssertNoMismatch(fails, "NumPy's _var computes x - mean even for one element, so NaN and inf propagate");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> over an axis of length 1 is NaN at every output whose element is NaN or ±inf and
        ///     0 elsewhere, for every float dtype, complex, both axis positions and <c>keepdims</c>: the same arithmetic as
        ///     any other axis.
        /// </summary>
        [TestMethod]
        public void VarStd_LengthOneAxis_NonFiniteValue_IsNaN()
        {
            var fails = new List<string>();
            const string nanThenZeros64 = "nan 0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "var f64 (1,4) axis0 nan", () => np.var(Of(double.NaN, 2, 3, -4).reshape(1, 4), 0),
                  "float64 (4,) " + nanThenZeros64);
            Check(fails, "std f64 (4,1) axis1 inf", () => np.std(Of(double.PositiveInfinity, 2, 3, -4).reshape(4, 1), 1),
                  "float64 (4,) " + nanThenZeros64);
            Check(fails, "var f64 (2,1,2) axis1 -inf", () => np.var(Of(double.NegativeInfinity, 2, 3, -4).reshape(2, 1, 2), 1),
                  "float64 (2,2) " + nanThenZeros64);
            Check(fails, "var f64 (1,4) axis0 nan keepdims", () => np.var(Of(double.NaN, 2, 3, -4).reshape(1, 4), 0, keepdims: true),
                  "float64 (1,4) " + nanThenZeros64);
            Check(fails, "var f32 (1,4) axis0 nan", () => np.var(OfSingle(float.NaN, 2, 3, -4).reshape(1, 4), 0),
                  "float32 (4,) nan 00000000 00000000 00000000");
            Check(fails, "std f16 (1,4) axis0 inf", () => np.std(OfHalf(double.PositiveInfinity, 2, 3, -4).reshape(1, 4), 0),
                  "float16 (4,) nan 0000 0000 0000");
            Check(fails, "var c128 (1,4) axis0 nanj",
                  () => np.var(OfComplex(new Complex(0, double.NaN), new Complex(2, -1), new Complex(3, 0.5), new Complex(-4, 4)).reshape(1, 4), 0),
                  "float64 (4,) " + nanThenZeros64);
            AssertNoMismatch(fails, "NumPy reduces a length-1 axis with the same arithmetic as any other axis");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> over an axis of length 1 with <c>ddof &gt;= 1</c> is NaN at EVERY output: the zero
        ///     sum of squares divided by <c>max(1 - ddof, 0) = 0</c>. A sample variance (<c>ddof=1</c>) of single observations
        ///     is "undefined", not "no spread" — for every dtype, the integer and bool inputs (float64 results) included.
        /// </summary>
        [TestMethod]
        public void VarStd_LengthOneAxis_DdofOfOneOrMore_IsNaN()
        {
            var fails = new List<string>();
            Check(fails, "var f64 (1,4) axis0 ddof1", () => np.var(Of(5, 2, 3, -4).reshape(1, 4), 0, ddof: 1), "float64 (4,) nan nan nan nan");
            Check(fails, "std f64 (1,4) axis0 ddof2", () => np.std(Of(5, 2, 3, -4).reshape(1, 4), 0, ddof: 2), "float64 (4,) nan nan nan nan");
            Check(fails, "std f32 (4,1) axis1 ddof1", () => np.std(OfSingle(5, 2, 3, -4).reshape(4, 1), 1, ddof: 1), "float32 (4,) nan nan nan nan");
            Check(fails, "var f16 (2,1,2) axis1 ddof1", () => np.var(OfHalf(5, 2, 3, -4).reshape(2, 1, 2), 1, ddof: 1), "float16 (2,2) nan nan nan nan");
            Check(fails, "var i32 (1,4) axis0 ddof1", () => np.var(OfInt32(5, 2, 3, -4).reshape(1, 4), 0, ddof: 1), "float64 (4,) nan nan nan nan");
            Check(fails, "var bool (1,3) axis0 ddof1", () => np.var(OfBool(true, false, true).reshape(1, 3), 0, ddof: 1), "float64 (3,) nan nan nan");
            Check(fails, "var c128 (1,2) axis0 ddof1",
                  () => np.var(OfComplex(new Complex(1, 2), new Complex(2, -1)).reshape(1, 2), 0, ddof: 1), "float64 (2,) nan nan");
            Check(fails, "var f64 (1,1) axis1 ddof1 keepdims", () => np.var(Of(5.0).reshape(1, 1), 1, keepdims: true, ddof: 1), "float64 (1,1) nan");
            AssertNoMismatch(fails, "NumPy divides by max(n - ddof, 0), which is 0 for one element and ddof >= 1");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of one element keeps a float16/float32 input's dtype and computes in the
        ///     <c>dtype=</c> argument when one is given, so the result type of a float32 pipeline does not change when a slice
        ///     happens to hold one element.
        /// </summary>
        [TestMethod]
        public void VarStd_OneElement_KeepsFloat16Float32_AndHonoursTheDtypeArgument()
        {
            var fails = new List<string>();
            Check(fails, "var f32 [5]", () => np.var(OfSingle(5)), "float32 () 00000000");
            Check(fails, "var f16 [5]", () => np.var(OfHalf(5)), "float16 () 0000");
            Check(fails, "std f32 0-d 5", () => np.std(NDArray.Scalar(5f)), "float32 () 00000000");
            Check(fails, "var f32 [5] axis0", () => np.var(OfSingle(5), 0), "float32 () 00000000");
            Check(fails, "var f32 [5] ddof1", () => np.var(OfSingle(5), ddof: 1), "float32 () nan");
            Check(fails, "std f16 [nan]", () => np.std(OfHalf(double.NaN)), "float16 () nan");
            Check(fails, "var f64 [5] dtype=f32", () => np.var(Of(5.0), dtype: np.float32), "float32 () 00000000");
            Check(fails, "std f64 [5] axis0 dtype=f32", () => np.std(Of(5.0), 0, dtype: np.float32), "float32 () 00000000");
            Check(fails, "var i32 [5] dtype=f16", () => np.var(OfInt32(5), dtype: np.float16), "float16 () 0000");
            AssertNoMismatch(fails, "NumPy's _var computes in the input's float dtype, or in dtype= when given");
        }

        /// <summary>
        ///     A flat <c>np.var</c> / <c>np.std</c> of a contiguous float32, float64 or integer array with
        ///     <c>ddof &gt;= n</c> is <c>+inf</c>: the positive sum of squares divided by <c>max(n - ddof, 0) = 0</c>. Only a
        ///     slice whose squares sum to zero gives <c>0/0 = NaN</c>.
        /// </summary>
        [TestMethod]
        public void VarStd_DdofAtLeastTheCount_ContiguousRoute_IsInfinity()
        {
            var fails = new List<string>();
            const string inf64 = "float64 () 7ff0000000000000";
            Check(fails, "var f64 [1,2] ddof2", () => np.var(Of(1, 2), ddof: 2), inf64);
            Check(fails, "std f64 [1,2] ddof2", () => np.std(Of(1, 2), ddof: 2), inf64);
            Check(fails, "var f64 [1,2,4] ddof3", () => np.var(Of(1, 2, 4), ddof: 3), inf64);
            Check(fails, "var f64 [1,2] ddof3", () => np.var(Of(1, 2), ddof: 3), inf64);
            Check(fails, "std f64 [1,2] ddof3", () => np.std(Of(1, 2), ddof: 3), inf64);
            Check(fails, "var f64 (2,3) ddof7", () => np.var(Of(1, 2, 4, 8, 16, 32).reshape(2, 3), ddof: 7), inf64);
            Check(fails, "var f32 [1,2,4] ddof4", () => np.var(OfSingle(1, 2, 4), ddof: 4), "float32 () 7f800000");
            Check(fails, "std f32 [1,2] ddof2", () => np.std(OfSingle(1, 2), ddof: 2), "float32 () 7f800000");
            Check(fails, "var i32 [1,2] ddof2", () => np.var(OfInt32(1, 2), ddof: 2), inf64);
            Check(fails, "var i32 [1,2] ddof3", () => np.var(OfInt32(1, 2), ddof: 3), inf64);
            Check(fails, "var f64 [3,3] ddof2 (constant slice: 0/0)", () => np.var(Of(3, 3), ddof: 2), "float64 () nan");
            Check(fails, "var f64 (1,1) ddof2 (one element: 0/0)", () => np.var(Of(5.0).reshape(1, 1), ddof: 2), "float64 () nan");
            AssertNoMismatch(fails, "NumPy divides a positive sum of squares by max(n - ddof, 0) = 0, which is +inf");
        }

        /// <summary>
        ///     A flat <c>np.var</c> / <c>np.std</c> with <c>ddof &gt; n</c> over float16, complex, bool or a strided view is
        ///     <c>+inf</c> (NaN when the squares sum to zero), never the negative variance a divisor of <c>n - ddof</c> would
        ///     give: NumPy clamps the divisor at 0.
        /// </summary>
        [TestMethod]
        public void VarStd_DdofPastTheCount_GeneralRoute_IsInfinityNotNegative()
        {
            var fails = new List<string>();
            const string inf64 = "float64 () 7ff0000000000000";
            Check(fails, "var f64 [1,2,4] strided ddof4", () => np.var(Of(1, 0, 2, 0, 4)["::2"], ddof: 4), inf64);
            Check(fails, "std f64 [1,2,4] strided ddof4", () => np.std(Of(1, 0, 2, 0, 4)["::2"], ddof: 4), inf64);
            Check(fails, "var f16 [1,2] ddof3", () => np.var(OfHalf(1, 2), ddof: 3), "float16 () 7c00");
            Check(fails, "std f16 [1,2,4] ddof4", () => np.std(OfHalf(1, 2, 4), ddof: 4), "float16 () 7c00");
            Check(fails, "var c128 [1,2] ddof3", () => np.var(OfComplex(1, 2), ddof: 3), inf64);
            Check(fails, "std bool [T,F] ddof3", () => np.std(OfBool(true, false), ddof: 3), inf64);
            Check(fails, "var f16 (1,1) ddof2", () => np.var(OfHalf(5).reshape(1, 1), ddof: 2), "float16 () nan");
            Check(fails, "var f64 [1,2,4] strided ddof3 (ddof == n)", () => np.var(Of(1, 0, 2, 0, 4)["::2"], ddof: 3), inf64);
            AssertNoMismatch(fails, "NumPy clamps the divisor at 0, so a variance is never negative");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of an EMPTY float16/float32 array is NaN (or an empty result) in the input's
        ///     dtype, flat and per axis, and in the <c>dtype=</c> argument when one is given: NumPy runs its arithmetic on the
        ///     empty slice, and the empty sum divided by the zero count is a NaN of that dtype.
        /// </summary>
        [TestMethod]
        public void VarStd_EmptyFloat16Float32_KeepTheirDtype()
        {
            var fails = new List<string>();
            Check(fails, "var f16 (0,3)", () => np.var(np.zeros(new Shape(0, 3), np.float16)), "float16 () nan");
            Check(fails, "std f32 (0,3)", () => np.std(np.zeros(new Shape(0, 3), np.float32)), "float32 () nan");
            Check(fails, "var f16 (0,3) axis0", () => np.var(np.zeros(new Shape(0, 3), np.float16), 0), "float16 (3,) nan nan nan");
            Check(fails, "std f32 (0,3) axis0", () => np.std(np.zeros(new Shape(0, 3), np.float32), 0), "float32 (3,) nan nan nan");
            Check(fails, "var f32 (0,3) axis1", () => np.var(np.zeros(new Shape(0, 3), np.float32), 1), "float32 (0,) ");
            Check(fails, "var f16 (0,3) axis0 keepdims", () => np.var(np.zeros(new Shape(0, 3), np.float16), 0, keepdims: true),
                  "float16 (1,3) nan nan nan");
            Check(fails, "var f64 (0,) dtype=f32", () => np.var(np.zeros(new Shape(0), np.float64), dtype: np.float32), "float32 () nan");
            Check(fails, "var f64 (0,3) axis0", () => np.var(np.zeros(new Shape(0, 3), np.float64), 0), "float64 (3,) nan nan nan");
            Check(fails, "var c128 (0,3) axis0", () => np.var(np.zeros(new Shape(0, 3), np.complex128), 0), "float64 (3,) nan nan nan");
            Check(fails, "var i32 (0,3)", () => np.var(np.zeros(new Shape(0, 3), np.int32)), "float64 () nan");
            Check(fails, "var f16 (0,3) dtype=f16 (was heap corruption)", () => np.var(np.zeros(new Shape(0, 3), np.float16), 0, dtype: np.float16),
                  "float16 (3,) nan nan nan");
            AssertNoMismatch(fails, "NumPy's empty variance is NaN in the input's float dtype");
        }

        // ======================================================================================================
        //  dtype= is the loop of BOTH sums, at every size.
        // ======================================================================================================

        /// <summary>
        ///     <c>dtype=</c> is the loop of the mean's sum and of the deviation sum, so it decides the result dtype and the
        ///     arithmetic: an integer loop truncates the mean (<c>np.var([1, 2, 3, 4], dtype=np.int32)</c> is 1, not 1.25), a
        ///     float64 loop over a complex input keeps only the real part of the mean (NumPy's ComplexWarning), a complex loop
        ///     returns a complex variance, and a bool input with an integer loop squares <c>x * conj(x)</c> in that loop.
        /// </summary>
        [TestMethod]
        public void VarStd_DtypeArgument_IsTheLoopOfBothSums()
        {
            var fails = new List<string>();
            var i4 = np.array(new[] { 1, 2, 3, 4 });
            Check(fails, "var i32 dtype=i32", () => np.var(i4, dtype: np.int32), "int32 () 1");
            Check(fails, "std i32 dtype=i32 (scalar: truncated root)", () => np.std(i4, dtype: np.int32), "int32 () 1");
            Check(fails, "var i32 (2,2) axis0 dtype=i32", () => np.var(i4.reshape(2, 2), 0, dtype: np.int32), "int32 (2,) 1 1");
            Check(fails, "var i64 dtype=i32 keepdims", () => np.var(np.array(new long[] { 1, 2, 3, 4 }), keepdims: true, dtype: np.int32), "int32 (1,) 1");
            Check(fails, "var f64 dtype=i64", () => np.var(np.array(new[] { 1.5, 2.5, 3.5, 4.5 }), dtype: np.int64), "int64 () 2");
            Check(fails, "std f64 dtype=u8 (scalar)", () => np.std(np.array(new[] { 1.0, 2.0, 4.0, 8.0 }), dtype: np.uint8), "uint8 () 2");
            Check(fails, "std f64 dtype=i8 (scalar)", () => np.std(np.array(new[] { 1.0, 2.0, 4.0, 8.0 }), dtype: np.int8), "int8 () 2");
            Check(fails, "std f64 dtype=bool (scalar)", () => np.std(np.array(new[] { 1.0, 2.0, 4.0, 8.0 }), dtype: np.@bool), "bool () True");
            Check(fails, "std f64 [1,2,nan] dtype=i32 (NaN casts to INT_MIN)", () => np.std(np.array(new[] { 1.0, 2.0, double.NaN }), dtype: np.int32),
                  "int32 () -2147483648");
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, -1), new Complex(-2, 0.5), new Complex(4, 4) });
            Check(fails, "var c128 dtype=f64 (real part of the mean)", () => np.var(c, dtype: np.float64), "float64 () 4025200000000000");
            Check(fails, "var c128 dtype=c128", () => np.var(c, dtype: np.complex128), "complex128 () 4021580000000000 0000000000000000");
            Check(fails, "std c128 dtype=c128", () => np.std(c, dtype: np.complex128), "complex128 () 40078ef5ca97b3c8 0000000000000000");
            Check(fails, "var c128 (2,2) axis1 dtype=f32", () => np.var(c.reshape(2, 2), 1, dtype: np.float32), "float32 (2,) 40600000 41890000");
            var b = np.array(new[] { true, false, true, true });
            Check(fails, "var bool", () => np.var(b), "float64 () 3fc8000000000000");
            Check(fails, "var bool dtype=c128", () => np.var(b, dtype: np.complex128), "complex128 () 3fc8000000000000 0000000000000000");
            Check(fails, "var bool dtype=i8", () => np.var(b, dtype: np.int8), "int8 () 0");
            Check(fails, "var u64 big (float64 loop)", () => np.var(np.array(new ulong[] { 1UL << 63, (1UL << 63) + (1UL << 12), 3, ulong.MaxValue })),
                  "float64 () 47c0000000000000");
            Check(fails, "var i64 dtype=i64 (wrapping squares)", () => np.var(np.array(new long[] { 1L << 62, -(1L << 62), 7, 1L << 40 }), dtype: np.int64),
                  "int64 () -962072674294");
            Check(fails, "var f32 overflow", () => np.var(np.array(new[] { 1e30f, -1e30f, 3e30f, 0f })), "float32 () 7f800000");
            AssertNoMismatch(fails, "dtype= is the loop of both of _var's sums");
        }

        /// <summary>
        ///     <c>np.std</c> with an integer or bool <c>dtype=</c> RAISES when the result has dimensions — NumPy's
        ///     <c>sqrt(ret, out=ret)</c> refuses to cast the float root into the integer out — while a 0-d result is
        ///     <c>ret.dtype.type(sqrt(ret))</c>, a truncating cast (see <see cref="VarStd_DtypeArgument_IsTheLoopOfBothSums"/>).
        ///     The text names the sqrt loop the integer width selects.
        /// </summary>
        [TestMethod]
        public void Std_IntegerDtype_RaisesForAnArrayResult()
        {
            var i4 = np.array(new[] { 1, 2, 3, 4 }).reshape(2, 2);
            ExpectMessage(() => np.std(i4, 0, dtype: np.int32),
                "Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            ExpectMessage(() => np.std(i4, 0, dtype: np.int8),
                "Cannot cast ufunc 'sqrt' output from dtype('float16') to dtype('int8') with casting rule 'same_kind'");
            ExpectMessage(() => np.std(np.array(new[,] { { 1.0, 2.0 }, { 4.0, 8.0 } }), 0, dtype: np.int16),
                "Cannot cast ufunc 'sqrt' output from dtype('float32') to dtype('int16') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     A bool input with <c>dtype=bool</c> raises NumPy's boolean-subtract error: the mean is a bool (logical-OR sum,
        ///     unsafely divided), and <c>arr - mean</c> is a bool subtract.
        /// </summary>
        [TestMethod]
        public void VarStd_BoolLoopOnBoolInput_RaisesTheBooleanSubtractError()
        {
            ExpectMessage(() => np.var(np.array(new[] { true, false, true, true }), dtype: np.@bool),
                "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
        }

        /// <summary>
        ///     The axis is normalized BEFORE any arithmetic (<c>_count_reduce_items</c>), so an axis outside the operand is
        ///     NumPy's AxisError — for a 0-d operand too, where <c>ufunc.reduce</c> alone would accept axis 0 / -1.
        /// </summary>
        [TestMethod]
        public void VarStd_AxisOutsideTheOperand_IsAxisError_EvenForZeroD()
        {
            ExpectMessage(() => np.var(NDArray.Scalar(3.0), 0), "axis 0 is out of bounds for array of dimension 0");
            ExpectMessage(() => np.std(NDArray.Scalar(1.0), -1), "axis -1 is out of bounds for array of dimension 0");
            ExpectMessage(() => np.var(np.ones(new Shape(2, 3)), 2), "axis 2 is out of bounds for array of dimension 2");
            ExpectMessage(() => np.var(np.ones(new Shape(2, 3)), -3), "axis -3 is out of bounds for array of dimension 2");
        }

        // ======================================================================================================
        //  Layouts and dtype routes.
        // ======================================================================================================

        /// <summary>
        ///     The deviations are laid out as NumPy lays out <c>x = arr - arrmean</c>: NpyIter's K order of the operand and of
        ///     the keepdims mean, which NumPy allocates in the operand's own order. On a permuted input the deviation sum
        ///     then runs in the operand's memory order; NumSharp's own keepdims result is C-ordered, and taking ITS layout as
        ///     the mean's vote put <c>x</c> in C order (one ULP off at the first probed case).
        /// </summary>
        [TestMethod]
        public void VarStd_PermutedInput_SumsTheDeviationsInNumPysLayout()
        {
            var x = (np.arange(24 * 30 * 40).astype(np.float64) % 997) * 1.000123 + 1000.0;
            var p = x.reshape(24, 30, 40).transpose(new[] { 2, 0, 1 });
            Sha256(np.var(p, 2)).Should().Be("9e20d561b67c0d5fe4399815b72ecdf059c136eab2cb8216bdd0d06ee660aa2d");
            Sha256(np.std(p, 2, ddof: 1)).Should().Be("e0544cd8f2c80cafd8d397115e57278590972c5f90b61c0a4dc78917d7e702ca");
            Describe(np.var(p)).Should().Be("float64 () 40f41e58fd05eb75");
        }

        /// <summary>
        ///     float16 takes NumPy's HALF loops (each subtract and multiply in float32, rounded to half) through the eager SIMD
        ///     kernels, and a char input is an unsigned integer (float64 result), as NumSharp's uint16 proxy of it.
        /// </summary>
        [TestMethod]
        public void VarStd_Float16AndChar_FollowTheirLoops()
        {
            var fails = new List<string>();
            Check(fails, "var f16 [1,2,4] ddof1", () => np.var(OfHalf(1, 2, 4), ddof: 1), "float16 () 40ab");
            Check(fails, "std f16 (2,3) axis1", () => np.std(np.arange(6).astype(np.float16).reshape(2, 3) * NDArray.Scalar((Half)1.7), 1),
                  "float16 (2,) 3d8d 3d8d");
            Check(fails, "var f16 overflow", () => np.var(OfHalf(60000, -60000)), "float16 () 7c00");
            var ch = np.array(new[] { 'a', 'b', 'd', 'x' });
            Check(fails, "var char", () => np.var(ch), "float64 () 40564c0000000000");
            Check(fails, "std char (2,2) axis0", () => np.std(ch.reshape(2, 2), 0), "float64 (2,) 3ff8000000000000 4026000000000000");
            AssertNoMismatch(fails, "float16 runs NumPy's HALF loops; char is an unsigned integer");
        }

        /// <summary>
        ///     Decimal (no NumPy dtype) runs the same statements in exact decimal arithmetic — the mean, the deviations and
        ///     the square root keep 28 digits — and, having no NaN or infinity, an empty input or <c>ddof &gt;= n</c> raises
        ///     <see cref="DivideByZeroException"/> from the decimal division (the old engine returned a float64 NaN for an
        ///     empty input and a NEGATIVE decimal variance for <c>ddof &gt; n</c>).
        /// </summary>
        [TestMethod]
        public void VarStd_Decimal_IsExactDecimal_AndRaisesOnAZeroDivisor()
        {
            var d = np.array(new[] { 1m, 2m, 4m });
            np.var(d).GetAtIndex<decimal>(0).Should().Be(1.5555555555555555555555555556m);
            np.var(d, ddof: 1).GetAtIndex<decimal>(0).Should().Be(2.3333333333333333333333333334m);
            np.std(d).GetAtIndex<decimal>(0).Should().Be(1.2472191289246471285279162441m);
            np.var(d).dtype.Should().Be(np.@decimal);
            np.var(np.array(new[] { 5m })).GetAtIndex<decimal>(0).Should().Be(0m);
            // Per output: a sequential decimal sum for the mean and for the squared deviations, as np.sum adds decimal.
            var m = np.array(new[] { 1m, 2m, 4m, 8m, 16m, 32m }).reshape(2, 3);
            decimal mean1 = (8m + 16m + 32m) / 3;
            decimal var1 = ((8m - mean1) * (8m - mean1) + (16m - mean1) * (16m - mean1) + (32m - mean1) * (32m - mean1)) / 3;
            np.var(m, 1).GetAtIndex<decimal>(1).Should().Be(var1);
            new Action(() => np.var(d, ddof: 3)).Should().Throw<DivideByZeroException>();
            new Action(() => np.var(d, ddof: 4)).Should().Throw<DivideByZeroException>();
            new Action(() => np.var(np.zeros(new Shape(0), np.@decimal))).Should().Throw<DivideByZeroException>();
        }

        /// <summary>
        ///     Large inputs compute the deviations in L2-sized chunks — whole outputs (<c>ChunkedDeviationSum</c>), row blocks
        ///     of a reduced axis that is outermost in memory (<c>RowBlockDeviationSum</c>), whole buffer fills of a flat cast
        ///     loop (<c>FlatFillDeviationSum</c>) — instead of one array of squared deviations. Each must give the one-buffer
        ///     route's bytes exactly; the test hook forces the routes on small arrays (64-byte chunks) and the counter proves
        ///     they ran.
        /// </summary>
        [TestMethod]
        public void VarStd_ChunkedRoutes_AreByteIdenticalToTheOneBufferRoute()
        {
            var rng = np.random.default_rng(11);
            var cases = new List<(string, Func<NDArray>)>();
            var c2 = rng.normal(1000, 3, 600L).reshape(20, 30);
            var c3 = rng.normal(1000, 3, 5 * 7 * 60L).reshape(5, 7, 60);
            var wide = rng.normal(1000, 3, 9 * 700L).reshape(9, 700);
            var f2 = np.asfortranarray(wide.T);           // (700, 9) F: axis 1 is outermost in memory
            var i32 = rng.integers(-500, 500, 9 * 700L).astype(np.int32).reshape(9, 700);
            var h = rng.normal(8, 1.5, 9 * 700L).astype(np.float16).reshape(9, 700);
            var cx = rng.normal(1000, 3, 9 * 700L).astype(np.complex128)
                     + rng.normal(-5, 2, 9 * 700L).astype(np.complex128) * NDArray.Scalar(new Complex(0, 1));
            var flat = rng.normal(1000, 3, 300L);
            cases.Add(("C axis1 (pinned rows)", () => np.var(c2, 1)));
            cases.Add(("3-D axis1 (outer kept axis)", () => np.std(c3, 1, keepdims: true)));
            cases.Add(("wide axis0 (row blocks)", () => np.var(wide, 0)));
            cases.Add(("wide axis0 ddof1 keepdims (row blocks)", () => np.var(wide, 0, keepdims: true, ddof: 1)));
            cases.Add(("F axis1 (row blocks in memory order)", () => np.var(f2, 1)));
            cases.Add(("int32 axis0 (cast first)", () => np.var(i32, 0)));
            cases.Add(("float16 axis0 (eager)", () => np.std(h, 0)));
            cases.Add(("complex axis0 (modulus)", () => np.var(cx.reshape(9, 700), 0)));
            cases.Add(("cast axis0 f64->f32 (row blocks)", () => np.var(wide, 0, dtype: np.float32)));
            cases.Add(("cast axis1 f64->f32 rows fit the buffer", () => np.var(c2, 1, dtype: np.float32)));
            cases.Add(("flat cast f64->f32 (fills)", () => np.var(flat, dtype: np.float32)));
            cases.Add(("flat cast c128 modulus -> complex loop (fills)", () => np.std(cx, dtype: np.complex128)));

            long previousBufsize = np.setbufsize(32);   // small fills, so the flat cast cases split into several
            try
            {
                foreach (var (name, call) in cases)
                {
                    var oneBuffer = Describe(call());
                    int before = DefaultEngine.VarStdChunkedRuns;
                    DefaultEngine.VarStdChunkBytesTestOverride = 64;
                    string chunked;
                    try
                    {
                        chunked = Describe(call());
                    }
                    finally
                    {
                        DefaultEngine.VarStdChunkBytesTestOverride = 0;
                    }
                    chunked.Should().Be(oneBuffer, name);
                    DefaultEngine.VarStdChunkedRuns.Should().BeGreaterThan(before, $"{name}: the chunked route must have run");
                }
            }
            finally
            {
                np.setbufsize(previousBufsize);
            }
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

        /// <summary>Builds a 1-D float32 array from literal values (NumPy's <c>np.array([...], np.float32)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous float32 array.</returns>
        private static NDArray OfSingle(params float[] values) => np.array(values);

        /// <summary>
        ///     Builds a 1-D float16 array from literal values (NumPy's <c>np.array([...], np.float16)</c>). Every value the
        ///     cases use (small integers, NaN, ±inf, 60000) is exact in float16.
        /// </summary>
        /// <param name="values">The elements, in order, as doubles.</param>
        /// <returns>A fresh C-contiguous float16 array.</returns>
        private static NDArray OfHalf(params double[] values) => np.array(Array.ConvertAll(values, v => (Half)v));

        /// <summary>Builds a 1-D complex128 array from literal values (an integer literal is <c>value + 0j</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous complex128 array.</returns>
        private static NDArray OfComplex(params Complex[] values) => np.array(values);

        /// <summary>Builds a 1-D int32 array from literal values (NumPy's <c>np.array([...], np.int32)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous int32 array.</returns>
        private static NDArray OfInt32(params int[] values) => np.array(values);

        /// <summary>Builds a 1-D bool array from literal values (NumPy's <c>np.array([...], np.bool_)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous bool array.</returns>
        private static NDArray OfBool(params bool[] values) => np.array(values);
    }
}
