using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The six comparisons (==, !=, &lt;, &lt;=, &gt;, &gt;=) over every dtype pair and operand layout, through the
    ///     comparison run (<see cref="ComparisonRun"/>), the two-level route that needs no iterator, and the kernels
    ///     added with them (complex128, int64 against uint64, the strided move).
    /// </summary>
    /// <remarks>
    ///     <para>Every expected value comes from <see cref="Reference"/>, a per-element comparison written from NumPy's
    ///     rules and computed without NumSharp's kernels: integers exactly, floats in NumPy's comparison dtype
    ///     (<c>result_type</c> of the pair), complex by NumPy's <c>CLT</c> / <c>CEQ</c> macros, decimal by value.</para>
    ///     <para>The tests that drive <see cref="ComparisonRun.Run"/> and <see cref="ComparisonRun.RunRows"/> on raw
    ///     memory surround every buffer with sentinel bytes, so a store outside the result slots fails the test.</para>
    ///     <para>The two forms of a whole rows iteration have tests of their own: an operand whose rows are adjacent
    ///     elements (an F-ordered matrix, staged tile by tile) and a matrix against one short row (compared several
    ///     rows per kernel call). The counters <see cref="ComparisonRun.TileRuns"/> and
    ///     <see cref="ComparisonRun.RepeatedRowRuns"/> prove that those routes ran.</para>
    /// </remarks>
    [TestClass]
    public unsafe class ComparisonRunTests : TestClass
    {
        /// <summary>Every NumSharp dtype.</summary>
        private static readonly NPTypeCode[] AllTypes =
        {
            NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        /// <summary>The six comparisons.</summary>
        private static readonly ComparisonOp[] Ops =
        {
            ComparisonOp.Equal, ComparisonOp.NotEqual, ComparisonOp.Less, ComparisonOp.LessEqual, ComparisonOp.Greater,
            ComparisonOp.GreaterEqual,
        };

        /// <summary>The byte every margin and skipped slot holds before a call.</summary>
        private const byte Sentinel = 0x5A;

        // =====================================================================================================
        //  Reference and values
        // =====================================================================================================

        /// <summary>Bool, char and the eight integer dtypes: values that compare exactly as integers.</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>True for an integer-valued dtype.</returns>
        private static bool IsIntegerValued(NPTypeCode t) => t is NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte
            or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char or NPTypeCode.Int32 or NPTypeCode.UInt32
            or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>An integer-valued element as an exact integer (128 bits hold every int64 and uint64 value).</summary>
        /// <param name="v">The boxed element.</param>
        /// <returns>Its value.</returns>
        private static Int128 AsInteger(object v) => v switch
        {
            bool b => b ? 1 : 0, sbyte x => x, byte x => x, short x => x, ushort x => x, char x => x, int x => x, uint x => x,
            long x => x, ulong x => x,
            _ => throw new InvalidOperationException($"not an integer: {v.GetType()}"),
        };

        /// <summary>A real element as a double (the conversion NumPy's cast to float64 makes).</summary>
        /// <param name="v">The boxed element.</param>
        /// <returns>Its value.</returns>
        private static double AsDouble(object v) => v switch
        {
            bool b => b ? 1 : 0, sbyte x => x, byte x => x, short x => x, ushort x => x, char x => x, int x => x, uint x => x,
            long x => x, ulong x => x, Half h => (double)h, float f => f, double d => d, decimal m => (double)m,
            _ => throw new InvalidOperationException($"not real: {v.GetType()}"),
        };

        /// <summary>A real element as a decimal (NumSharp's own dtype; no NumPy analog).</summary>
        /// <param name="v">The boxed element.</param>
        /// <returns>Its value.</returns>
        private static decimal AsDecimal(object v) => v switch
        {
            bool b => b ? 1 : 0, sbyte x => x, byte x => x, short x => x, ushort x => x, char x => x, int x => x, uint x => x,
            long x => x, ulong x => x, Half h => (decimal)(double)h, float f => (decimal)f, double d => (decimal)d, decimal m => m,
            _ => throw new InvalidOperationException($"not real: {v.GetType()}"),
        };

        /// <summary>The comparison of two ordered values.</summary>
        /// <typeparam name="T">A type with the comparison operators.</typeparam>
        /// <param name="a">Left value.</param>
        /// <param name="b">Right value.</param>
        /// <param name="op">The comparison.</param>
        /// <returns><c>a op b</c> with the operators of <typeparamref name="T"/> (IEEE for floats).</returns>
        private static bool Cmp<T>(T a, T b, ComparisonOp op) where T : IComparisonOperators<T, T, bool>, IEqualityOperators<T, T, bool> => op switch
        {
            ComparisonOp.Equal => a == b,
            ComparisonOp.NotEqual => a != b,
            ComparisonOp.Less => a < b,
            ComparisonOp.LessEqual => a <= b,
            ComparisonOp.Greater => a > b,
            _ => a >= b,
        };

        /// <summary>
        ///     NumPy's comparison of one element of dtype <paramref name="ta"/> with one of <paramref name="tb"/>.
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="a">Left element.</param>
        /// <param name="ta">Its dtype.</param>
        /// <param name="b">Right element.</param>
        /// <param name="tb">Its dtype.</param>
        /// <returns>The expected result.</returns>
        /// <remarks>
        ///     Two integer-valued operands compare exactly whatever their widths and signs (NumPy's resolver never
        ///     compares two integers in float64). Otherwise both are converted to the pair's promoted dtype and
        ///     compared there: float16, float32 or float64 with IEEE rules, decimal by value, complex128 by NumPy's
        ///     macros (a NaN in any part makes the four ordered operators False).
        /// </remarks>
        private static bool Reference(ComparisonOp op, object a, NPTypeCode ta, object b, NPTypeCode tb)
        {
            if (ta == NPTypeCode.Complex || tb == NPTypeCode.Complex)
            {
                Complex x = a is Complex ca ? ca : new Complex(AsDouble(a), 0);
                Complex y = b is Complex cb ? cb : new Complex(AsDouble(b), 0);
                double ar = x.Real, ai = x.Imaginary, br = y.Real, bi = y.Imaginary;
                bool numbers = !double.IsNaN(ai) && !double.IsNaN(bi);
                return op switch
                {
                    ComparisonOp.Equal => ar == br && ai == bi,
                    ComparisonOp.NotEqual => ar != br || ai != bi,
                    ComparisonOp.Less => (ar < br && numbers) || (ar == br && ai < bi),
                    ComparisonOp.LessEqual => (ar < br && numbers) || (ar == br && ai <= bi),
                    ComparisonOp.Greater => (ar > br && numbers) || (ar == br && ai > bi),
                    _ => (ar > br && numbers) || (ar == br && ai >= bi),
                };
            }
            if (IsIntegerValued(ta) && IsIntegerValued(tb))
                return Cmp(AsInteger(a), AsInteger(b), op);

            NPTypeCode t = ta == tb ? ta : np._FindCommonScalarType(ta, tb);
            switch (t)
            {
                case NPTypeCode.Decimal: return Cmp(AsDecimal(a), AsDecimal(b), op);
                case NPTypeCode.Half: return Cmp((Half)AsDouble(a), (Half)AsDouble(b), op);
                case NPTypeCode.Single: return Cmp((float)AsDouble(a), (float)AsDouble(b), op);
                case NPTypeCode.Double: return Cmp(AsDouble(a), AsDouble(b), op);
                default: throw new InvalidOperationException($"no reference for {ta} with {tb} ({t})");
            }
        }

        /// <summary>
        ///     The values a dtype is tested with: extremes, the neighbours of other dtypes' limits, signed zeros, NaN
        ///     and infinities, and for complex a NaN in either part.
        /// </summary>
        /// <param name="t">The dtype.</param>
        /// <param name="finiteOnly">Leave NaN, infinities and values beyond decimal's range out (a decimal partner
        ///     cannot be converted from them).</param>
        /// <returns>The boxed values.</returns>
        private static object[] Pool(NPTypeCode t, bool finiteOnly = false)
        {
            switch (t)
            {
                case NPTypeCode.Boolean: return new object[] { false, true };
                case NPTypeCode.SByte: return new object[] { (sbyte)0, (sbyte)1, (sbyte)-1, sbyte.MinValue, sbyte.MaxValue, (sbyte)50, (sbyte)-50 };
                case NPTypeCode.Byte: return new object[] { (byte)0, (byte)1, (byte)255, (byte)127, (byte)128, (byte)50, (byte)254 };
                case NPTypeCode.Int16: return new object[] { (short)0, (short)1, (short)-1, short.MinValue, short.MaxValue, (short)255, (short)256, (short)-129 };
                case NPTypeCode.UInt16: return new object[] { (ushort)0, (ushort)1, ushort.MaxValue, (ushort)32767, (ushort)32768, (ushort)255, (ushort)256 };
                case NPTypeCode.Char: return new object[] { (char)0, (char)1, char.MaxValue, (char)32767, (char)32768, (char)255, 'a' };
                case NPTypeCode.Int32: return new object[] { 0, 1, -1, int.MinValue, int.MaxValue, 65535, 65536, -32769, 16777217 };
                case NPTypeCode.UInt32: return new object[] { 0u, 1u, uint.MaxValue, 2147483647u, 2147483648u, 65536u, 16777217u };
                case NPTypeCode.Int64: return new object[] { 0L, 1L, -1L, long.MinValue, long.MaxValue, 4294967296L, -2147483649L, 9007199254740993L, 9007199254740992L };
                case NPTypeCode.UInt64: return new object[] { 0UL, 1UL, ulong.MaxValue, 9223372036854775807UL, 9223372036854775808UL, 9007199254740993UL, 9007199254740992UL };
                case NPTypeCode.Half:
                    return finiteOnly
                        ? new object[] { (Half)0.0, (Half)1.0, (Half)(-1.0), (Half)50.5, (Half)255.0 }
                        : new object[] { (Half)0.0, (Half)(-0.0), (Half)1.0, (Half)(-1.0), Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity, Half.MaxValue, Half.Epsilon, (Half)50.5, (Half)255.0, (Half)256.0 };
                case NPTypeCode.Single:
                    return finiteOnly
                        ? new object[] { 0f, 1f, -1f, 50.5f, 255f, 16777216f }
                        : new object[] { 0f, -0f, 1f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue, float.Epsilon, 50.5f, 0.1f, 16777216f, 255f, 2147483648f, 65504f };
                case NPTypeCode.Double:
                    return finiteOnly
                        ? new object[] { 0.0, 1.0, -1.0, 50.5, 255.0, 9007199254740992.0 }
                        : new object[] { 0.0, -0.0, 1.0, -1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.MaxValue, double.Epsilon, 50.5, 0.1, 9007199254740992.0, 9223372036854775808.0, 18446744073709551616.0, 1e39, -1e39, 65520.0, 127.5, -128.5, 255.5 };
                case NPTypeCode.Decimal: return new object[] { 0m, 1m, -1m, 50.5m, 0.1m, 255m, 9007199254740993m };
                default:
                    return new object[]
                    {
                        new Complex(0, 0), new Complex(1, 0), new Complex(1, 1), new Complex(1, -1), new Complex(-1, 2),
                        new Complex(double.NaN, 0), new Complex(0, double.NaN), new Complex(1, double.NaN),
                        new Complex(double.PositiveInfinity, 1), new Complex(2, 0), new Complex(-0.0, 0.0),
                    };
            }
        }

        /// <summary>The pool of <paramref name="t"/> for a pair with <paramref name="other"/> (finite when a decimal is involved).</summary>
        /// <param name="t">The dtype.</param>
        /// <param name="other">The other operand's dtype.</param>
        /// <returns>The values.</returns>
        private static object[] PoolFor(NPTypeCode t, NPTypeCode other)
            => Pool(t, finiteOnly: other == NPTypeCode.Decimal && t is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double);

        /// <summary>
        ///     The pool entry for position <paramref name="i"/> of a sequence: a fixed function of the position that
        ///     reaches every entry of a pool of any length.
        /// </summary>
        /// <param name="i">Position in the sequence.</param>
        /// <param name="phase">Selects one of several unrelated sequences (the two operands of a test use two).</param>
        /// <param name="length">Number of pool entries.</param>
        /// <returns>An index in <c>[0, length)</c>.</returns>
        /// <remarks>
        ///     A multiplicative hash, not <c>(i * step + phase) % length</c>: a fixed step visits ONE entry when the
        ///     pool's length divides it (a step of 7 over the seven uint8 values gave arrays of one repeated value, in
        ///     which no stride, reversal or block-order error can be seen).
        /// </remarks>
        private static int PoolIndex(long i, int phase, int length)
        {
            ulong h = ((ulong)i + 1) * 0x9E3779B97F4A7C15UL + ((ulong)phase + 1) * 0xC2B2AE3D27D4EB4FUL;
            h ^= h >> 29;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 32;
            return (int)(h % (ulong)length);
        }

        /// <summary>An array of <paramref name="n"/> elements of <paramref name="t"/> drawn from a pool.</summary>
        /// <param name="t">The dtype.</param>
        /// <param name="other">The other operand's dtype (selects the pool).</param>
        /// <param name="n">Element count.</param>
        /// <param name="phase">Selects the sequence of pool entries (see <see cref="PoolIndex"/>).</param>
        /// <returns>The array (1-D, contiguous, owning).</returns>
        private static NDArray Make(NPTypeCode t, NPTypeCode other, long n, int phase)
        {
            var pool = PoolFor(t, other);
            var a = np.empty(new Shape(n), t);
            for (long i = 0; i < n; i++)
                a.SetAtIndex(pool[PoolIndex(i, phase, pool.Length)], i);
            return a;
        }

        /// <summary>Calls the comparison ufunc for <paramref name="op"/>.</summary>
        /// <param name="op">The comparison.</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="o">Optional <c>out=</c>.</param>
        /// <returns>The ufunc's result.</returns>
        private static NDArray Compare(ComparisonOp op, NDArray a, NDArray b, NDArray o = null) => op switch
        {
            ComparisonOp.Equal => np.equal(a, b, o),
            ComparisonOp.NotEqual => np.not_equal(a, b, o),
            ComparisonOp.Less => np.less(a, b, o),
            ComparisonOp.LessEqual => np.less_equal(a, b, o),
            ComparisonOp.Greater => np.greater(a, b, o),
            _ => np.greater_equal(a, b, o),
        };

        /// <summary>
        ///     Asserts that <paramref name="got"/> is the element-wise comparison of the two operands, broadcast
        ///     against each other, in logical order.
        /// </summary>
        /// <param name="what">Label for the failure message.</param>
        /// <param name="got">The result (bool, or a numeric <c>out=</c> holding 0 / 1).</param>
        /// <param name="a">Left operand.</param>
        /// <param name="ta">Its dtype.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="tb">Its dtype.</param>
        /// <param name="op">The comparison.</param>
        private static void AssertCompared(string what, NDArray got, NDArray a, NPTypeCode ta, NDArray b, NPTypeCode tb, ComparisonOp op)
        {
            // Operands of at most one dimension (the pair matrix: about 100,000 calls) are read in place, element by
            // element in logical order; GetAtIndex follows a view's strides. Everything else is broadcast and copied.
            if (a.ndim <= 1 && b.ndim <= 1 && got.ndim <= 1 && (a.size == b.size || a.size == 1 || b.size == 1))
            {
                long n = a.size == 1 ? b.size : a.size;
                got.size.Should().Be(n, what);
                for (long i = 0; i < n; i++)
                    AssertElement(what, i, got.GetAtIndex(i), a.GetAtIndex(a.size == 1 ? 0 : i), ta, b.GetAtIndex(b.size == 1 ? 0 : i), tb, op);
                return;
            }

            var (ba, bb) = np.broadcast_arrays(a, b);
            using var fa = ba.flatten();
            using var fb = bb.flatten();
            using var fg = got.flatten();
            fg.size.Should().Be(fa.size, what);
            for (long i = 0; i < fa.size; i++)
                AssertElement(what, i, fg.GetAtIndex(i), fa.GetAtIndex(i), ta, fb.GetAtIndex(i), tb, op);
        }

        /// <summary>Asserts one result element against the reference.</summary>
        /// <param name="what">Label for the failure message.</param>
        /// <param name="i">Logical index of the element.</param>
        /// <param name="g">The result element (a bool, or a number holding 0 / 1).</param>
        /// <param name="x">Left element.</param>
        /// <param name="ta">Its dtype.</param>
        /// <param name="y">Right element.</param>
        /// <param name="tb">Its dtype.</param>
        /// <param name="op">The comparison.</param>
        private static void AssertElement(string what, long i, object g, object x, NPTypeCode ta, object y, NPTypeCode tb, ComparisonOp op)
        {
            bool want = Reference(op, x, ta, y, tb);
            bool have = g is bool gb ? gb : Convert.ToDouble(g) != 0;
            if (have != want)
                Assert.Fail($"{what} [{i}]: {x} ({ta}) {op} {y} ({tb}) should be {want}");
        }

        // =====================================================================================================
        //  Through the public API
        // =====================================================================================================

        /// <summary>
        ///     Every dtype pair, every comparison, 1-D operands in every layout a view can have (contiguous, a step of
        ///     two, a step of three, reversed, a negative step, one value on either side), at lengths on both sides of
        ///     the staging threshold. Runs longer than a staging block are the subject of the long-run, out= and raw
        ///     tests below (a pair matrix at those lengths would take most of the class's time).
        /// </summary>
        /// <remarks>
        ///     The stepped views come in two positions: starting at the buffer's first element, and ENDING at its last
        ///     one (<c>[1::2]</c>, <c>[2::3]</c>). Under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c> every buffer ends at an
        ///     inaccessible page, so a kernel that reads one element past such a view ends the test host.
        /// </remarks>
        [TestMethod]
        public void EveryDtypePair_OneDimensionalLayouts_MatchAnElementReference()
        {
            foreach (var ta in AllTypes)
            foreach (var tb in AllTypes)
            foreach (long n in new long[] { 1, 7, 63, 65 })
            {
                using var a = Make(ta, tb, n, 0);
                using var b = Make(tb, ta, n, 3);
                using var a2 = Make(ta, tb, 2 * n, 1);
                using var b2 = Make(tb, ta, 2 * n, 5);
                using var a3 = Make(ta, tb, 3 * n, 2);
                NDArray aStep = a2["::2"], bStep = b2["::2"], aRev = a["::-1"], bRev = b["::-1"], aThree = a3["::3"], bBack = b2["::-2"];
                NDArray aStepEnd = a2["1::2"], bStepEnd = b2["1::2"], aThreeEnd = a3["2::3"];
                using var value = NDArray.Scalar(PoolFor(tb, ta)[(int)(n % PoolFor(tb, ta).Length)], tb);
                using var valueLeft = NDArray.Scalar(PoolFor(ta, tb)[(int)(n % PoolFor(ta, tb).Length)], ta);
                foreach (var op in Ops)
                {
                    string tag = $"{op} {ta} x {tb} n={n}";
                    using (var g = Compare(op, a, b)) AssertCompared(tag + " contiguous", g, a, ta, b, tb, op);
                    using (var g = Compare(op, aStep, bStep)) AssertCompared(tag + " step 2", g, aStep, ta, bStep, tb, op);
                    using (var g = Compare(op, aRev, bRev)) AssertCompared(tag + " reversed", g, aRev, ta, bRev, tb, op);
                    using (var g = Compare(op, aThree, b)) AssertCompared(tag + " step 3 / contiguous", g, aThree, ta, b, tb, op);
                    using (var g = Compare(op, a, bBack)) AssertCompared(tag + " contiguous / step -2", g, a, ta, bBack, tb, op);
                    using (var g = Compare(op, aRev, bStep)) AssertCompared(tag + " reversed / step 2", g, aRev, ta, bStep, tb, op);
                    using (var g = Compare(op, a, value)) AssertCompared(tag + " array / value", g, a, ta, value, tb, op);
                    using (var g = Compare(op, aRev, value)) AssertCompared(tag + " reversed / value", g, aRev, ta, value, tb, op);
                    using (var g = Compare(op, valueLeft, bStep)) AssertCompared(tag + " value / step 2", g, valueLeft, ta, bStep, tb, op);
                    using (var g = Compare(op, aStepEnd, bStepEnd)) AssertCompared(tag + " step 2 to the buffer end", g, aStepEnd, ta, bStepEnd, tb, op);
                    using (var g = Compare(op, aThreeEnd, bRev)) AssertCompared(tag + " step 3 to the buffer end / reversed", g, aThreeEnd, ta, bRev, tb, op);
                    using (var g = Compare(op, aStepEnd, value)) AssertCompared(tag + " step 2 to the buffer end / value", g, aStepEnd, ta, value, tb, op);
                }
            }
        }

        /// <summary>
        ///     Two- and three-dimensional operands whose iteration is rows of runs: a row, a column, an outer product,
        ///     C against F order, reversed rows and columns, stepped columns, a column of a matrix, and partial
        ///     broadcasts of a 3-D operand. Row lengths cover the one-byte kernels' partial vectors (40, 47, 63).
        /// </summary>
        [TestMethod]
        public void TwoLevelLayouts_MatchAnElementReference()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Int32, NPTypeCode.Int32),
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Int16, NPTypeCode.Int16),
                (NPTypeCode.Half, NPTypeCode.Half), (NPTypeCode.Complex, NPTypeCode.Complex), (NPTypeCode.Decimal, NPTypeCode.Decimal),
                (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.UInt64, NPTypeCode.Int64), (NPTypeCode.Int32, NPTypeCode.Double),
                (NPTypeCode.Single, NPTypeCode.Double), (NPTypeCode.Byte, NPTypeCode.Int32), (NPTypeCode.Boolean, NPTypeCode.Single),
                (NPTypeCode.Double, NPTypeCode.Complex), (NPTypeCode.Char, NPTypeCode.Int16), (NPTypeCode.SByte, NPTypeCode.Byte),
            };
            foreach (var (ta, tb) in pairs)
            {
                foreach (var (rows, cols) in new[] { (3, 5), (4, 70), (70, 4), (5, 40), (6, 47), (3, 63), (2, 300) })
                {
                    long n = (long)rows * cols;
                    using var aFlat = Make(ta, tb, n, 2);
                    using var bFlat = Make(tb, ta, n, 4);
                    NDArray a = aFlat.reshape(rows, cols), b = bFlat.reshape(rows, cols);
                    using var row = Make(tb, ta, cols, 6);
                    using var colFlat = Make(tb, ta, rows, 1);
                    using var colAFlat = Make(ta, tb, rows, 3);
                    NDArray col = colFlat.reshape(rows, 1), colA = colAFlat.reshape(rows, 1);
                    using var bF = b.copy('F');
                    foreach (var op in Ops)
                    {
                        string tag = $"{op} {ta} x {tb} ({rows},{cols})";
                        using (var g = Compare(op, a, row)) AssertCompared(tag + " row", g, a, ta, row, tb, op);
                        using (var g = Compare(op, a, col)) AssertCompared(tag + " column", g, a, ta, col, tb, op);
                        using (var g = Compare(op, colA, row)) AssertCompared(tag + " outer", g, colA, ta, row, tb, op);
                        using (var g = Compare(op, a, bF)) AssertCompared(tag + " C / F", g, a, ta, bF, tb, op);
                        using (var g = Compare(op, a["::-1"], b["::-1"])) AssertCompared(tag + " reversed rows", g, a["::-1"], ta, b["::-1"], tb, op);
                        using (var g = Compare(op, a[":, ::-1"], b[":, ::-1"])) AssertCompared(tag + " reversed columns", g, a[":, ::-1"], ta, b[":, ::-1"], tb, op);
                        using (var g = Compare(op, a[":, ::-1"], col)) AssertCompared(tag + " reversed columns / column", g, a[":, ::-1"], ta, col, tb, op);
                        using (var g = Compare(op, a[":, ::2"], b[":, ::2"])) AssertCompared(tag + " stepped columns", g, a[":, ::2"], ta, b[":, ::2"], tb, op);
                        using (var g = Compare(op, a[":, 1"], b[":, 0"])) AssertCompared(tag + " column / column", g, a[":, 1"], ta, b[":, 0"], tb, op);
                        using (var g = Compare(op, a.T, b.T)) AssertCompared(tag + " transposed", g, a.T, ta, b.T, tb, op);
                    }
                }

                const int d0 = 3, d1 = 4, d2 = 70;
                using var a3Flat = Make(ta, tb, d0 * d1 * d2, 1);
                NDArray a3 = a3Flat.reshape(d0, d1, d2);
                using var p1 = Make(tb, ta, d1 * d2, 2);
                using var p2 = Make(tb, ta, d0 * d2, 4);
                using var p3 = Make(tb, ta, d0 * d1, 6);
                using var p4 = Make(tb, ta, d1, 0);
                using var p5 = Make(tb, ta, d0 * 2 * d1 * d2, 3);
                var partials = new (string Name, NDArray V)[]
                {
                    ("(d1,d2)", p1.reshape(d1, d2)), ("(d0,1,d2)", p2.reshape(d0, 1, d2)), ("(d0,d1,1)", p3.reshape(d0, d1, 1)),
                    ("(1,d1,1)", p4.reshape(1, d1, 1)), ("stepped middle axis", p5.reshape(d0, 2 * d1, d2)[":, ::2, :"]),
                };
                foreach (var op in Ops)
                foreach (var (name, v) in partials)
                {
                    using (var g = Compare(op, a3, v)) AssertCompared($"{op} {ta} x {tb} 3-D {name}", g, a3, ta, v, tb, op);
                    using (var g = Compare(op, v, a3)) AssertCompared($"{op} {tb} x {ta} 3-D {name} swapped", g, v, tb, a3, ta, op);
                }
            }
        }

        /// <summary>
        ///     <c>out=</c> in the four forms the run stores into (contiguous bool, stepped bool, reversed bool, float64
        ///     through the iterator's Boolean buffer): the values are right and a stepped out's skipped slots are not
        ///     written.
        /// </summary>
        [TestMethod]
        public void OutForms_HoldTheResults_AndSkippedSlotsAreNotWritten()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Int32, NPTypeCode.Double),
                (NPTypeCode.Complex, NPTypeCode.Complex), (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.Half, NPTypeCode.Single),
            };
            foreach (var (ta, tb) in pairs)
            foreach (long n in new long[] { 2, 65, 1025, 9000 })
            {
                using var a = Make(ta, tb, n, 0);
                using var b = Make(tb, ta, n, 3);
                using var a2 = Make(ta, tb, 2 * n, 1);
                NDArray aStep = a2["::2"], aRev = a["::-1"], bRev = b["::-1"];
                foreach (var op in Ops)
                {
                    string tag = $"{op} {ta} x {tb} n={n}";
                    using var o = np.empty(new Shape(n), np.bool_);
                    Compare(op, a, b, o);
                    AssertCompared(tag + " out", o, a, ta, b, tb, op);
                    Compare(op, aStep, bRev, o);
                    AssertCompared(tag + " out, stepped / reversed inputs", o, aStep, ta, bRev, tb, op);

                    using var wide = np.zeros(new Shape(2 * n), np.bool_);
                    NDArray stepped = wide["::2"];
                    Compare(op, a, b, stepped);
                    AssertCompared(tag + " out=stepped", stepped, a, ta, b, tb, op);
                    Compare(op, aRev, bRev, stepped);
                    AssertCompared(tag + " out=stepped, reversed inputs", stepped, aRev, ta, bRev, tb, op);
                    using (var skipped = wide["1::2"])
                        np.any(skipped).Should().BeFalse($"{tag}: a slot between the stepped out's slots was written");

                    using var backing = np.empty(new Shape(n), np.bool_);
                    NDArray reversed = backing["::-1"];
                    Compare(op, a, b, reversed);
                    AssertCompared(tag + " out=reversed", reversed, a, ta, b, tb, op);

                    using var asFloat = np.empty(new Shape(n), np.float64);
                    Compare(op, aStep, b, asFloat);
                    AssertCompared(tag + " out=float64", asFloat, aStep, ta, b, tb, op);
                }
            }
        }

        /// <summary>
        ///     Long contiguous runs that start off a vector boundary (a view one or three elements into its buffer):
        ///     the run compares a prefix separately to align the rest, and the values do not change.
        /// </summary>
        [TestMethod]
        public void LongRunsOffAVectorBoundary_MatchAnElementReference()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Int16, NPTypeCode.Int16),
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Complex, NPTypeCode.Complex),
                (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.Half, NPTypeCode.Half), (NPTypeCode.Single, NPTypeCode.Double),
            };
            foreach (var (ta, tb) in pairs)
            {
                const long n = 6000;
                using var aBase = Make(ta, tb, n + 8, 0);
                using var bBase = Make(tb, ta, n + 8, 3);
                using var value = NDArray.Scalar(PoolFor(tb, ta)[1], tb);
                foreach (int offset in new[] { 0, 1, 3 })
                {
                    NDArray a = aBase[$"{offset}:{offset + n}"], b = bBase[$"{offset}:{offset + n}"], bOther = bBase[$"{(offset + 1) % 4}:{(offset + 1) % 4 + n}"];
                    foreach (var op in Ops)
                    {
                        string tag = $"{op} {ta} x {tb} offset {offset}";
                        using (var g = Compare(op, a, b)) AssertCompared(tag, g, a, ta, b, tb, op);
                        using (var g = Compare(op, a, bOther)) AssertCompared(tag + " / other offset", g, a, ta, bOther, tb, op);
                        using (var g = Compare(op, a, value)) AssertCompared(tag + " / value", g, a, ta, value, tb, op);
                        using (var g = Compare(op, value, a)) AssertCompared(tag + " value /", g, value, tb, a, ta, op);
                    }
                }
            }
        }

        /// <summary>
        ///     An array against ONE value of another dtype is compared in the array's own dtype, and the result is that
        ///     of the comparison in the promoted dtype for every element: every uint8 and int8 value, every float16 bit
        ///     pattern, and float32 values around each threshold, against values at, between and beyond the array
        ///     dtype's values.
        /// </summary>
        [TestMethod]
        public void ValueOfAnotherDtype_ComparesAsInThePromotedDtype()
        {
            // Integer arrays: every value of the narrow dtypes, the extremes and neighbours of the wide ones.
            var integerArrays = new (NPTypeCode Type, object[] Values)[]
            {
                (NPTypeCode.Byte, Enumerable.Range(0, 256).Select(i => (object)(byte)i).ToArray()),
                (NPTypeCode.SByte, Enumerable.Range(-128, 256).Select(i => (object)(sbyte)i).ToArray()),
                (NPTypeCode.Boolean, Enumerable.Range(0, 70).Select(i => (object)(i % 3 == 0)).ToArray()),
                (NPTypeCode.Int16, new short[] { short.MinValue, -32767, -129, -128, -1, 0, 1, 127, 128, 255, 256, 32766, short.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 6)).ToArray()),
                (NPTypeCode.UInt16, new ushort[] { 0, 1, 127, 128, 255, 256, 32767, 32768, 65534, ushort.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 7)).ToArray()),
                (NPTypeCode.Int32, new[] { int.MinValue, -2147483647, -65537, -1, 0, 1, 65535, 65536, 16777216, 16777217, 2147483646, int.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 6)).ToArray()),
                (NPTypeCode.UInt32, new uint[] { 0, 1, 65535, 65536, 16777217, 2147483647, 2147483648, 4294967294, uint.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 8)).ToArray()),
                (NPTypeCode.Int64, new[] { long.MinValue, -4294967297L, -1L, 0L, 1L, 4294967295L, 4294967296L, 9007199254740993L, long.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 8)).ToArray()),
                (NPTypeCode.UInt64, new[] { 0UL, 1UL, 4294967296UL, 9007199254740993UL, 9223372036854775807UL, 9223372036854775808UL, ulong.MaxValue }.SelectMany(v => Enumerable.Repeat((object)v, 10)).ToArray()),
            };
            var values = new (NPTypeCode Type, object Value)[]
            {
                (NPTypeCode.Boolean, false), (NPTypeCode.Boolean, true),
                (NPTypeCode.SByte, (sbyte)-128), (NPTypeCode.SByte, (sbyte)-1), (NPTypeCode.SByte, (sbyte)0), (NPTypeCode.SByte, (sbyte)127),
                (NPTypeCode.Byte, (byte)0), (NPTypeCode.Byte, (byte)128), (NPTypeCode.Byte, (byte)255),
                (NPTypeCode.Int16, (short)-129), (NPTypeCode.Int16, (short)-128), (NPTypeCode.Int16, (short)255), (NPTypeCode.Int16, (short)256), (NPTypeCode.Int16, short.MinValue),
                (NPTypeCode.UInt16, (ushort)256), (NPTypeCode.UInt16, (ushort)32768), (NPTypeCode.UInt16, ushort.MaxValue),
                (NPTypeCode.Int32, -1), (NPTypeCode.Int32, 0), (NPTypeCode.Int32, 1), (NPTypeCode.Int32, 127), (NPTypeCode.Int32, 128), (NPTypeCode.Int32, 255), (NPTypeCode.Int32, 256),
                (NPTypeCode.Int32, -129), (NPTypeCode.Int32, 65535), (NPTypeCode.Int32, 65536), (NPTypeCode.Int32, int.MinValue), (NPTypeCode.Int32, int.MaxValue),
                (NPTypeCode.UInt32, 2147483648u), (NPTypeCode.UInt32, uint.MaxValue),
                (NPTypeCode.Int64, -1L), (NPTypeCode.Int64, 4294967296L), (NPTypeCode.Int64, long.MinValue), (NPTypeCode.Int64, long.MaxValue), (NPTypeCode.Int64, 16777217L),
                (NPTypeCode.UInt64, 9223372036854775808UL), (NPTypeCode.UInt64, ulong.MaxValue), (NPTypeCode.UInt64, 255UL),
                (NPTypeCode.Half, (Half)0.5), (NPTypeCode.Half, (Half)(-0.5)), (NPTypeCode.Half, (Half)127.0), (NPTypeCode.Half, Half.NaN), (NPTypeCode.Half, Half.PositiveInfinity),
                (NPTypeCode.Single, 0.5f), (NPTypeCode.Single, 127.5f), (NPTypeCode.Single, -128.5f), (NPTypeCode.Single, 255f), (NPTypeCode.Single, 65535.5f), (NPTypeCode.Single, float.NaN),
                (NPTypeCode.Single, float.NegativeInfinity), (NPTypeCode.Single, 2147483648f), (NPTypeCode.Single, -0f),
                (NPTypeCode.Double, 0.5), (NPTypeCode.Double, -0.5), (NPTypeCode.Double, 127.5), (NPTypeCode.Double, 128.0), (NPTypeCode.Double, 255.0), (NPTypeCode.Double, 255.5),
                (NPTypeCode.Double, -129.0), (NPTypeCode.Double, 65535.999), (NPTypeCode.Double, 2147483647.5), (NPTypeCode.Double, -2147483648.5), (NPTypeCode.Double, 4294967295.5),
                (NPTypeCode.Double, 1e300), (NPTypeCode.Double, -1e300), (NPTypeCode.Double, double.NaN), (NPTypeCode.Double, double.PositiveInfinity), (NPTypeCode.Double, double.NegativeInfinity),
                (NPTypeCode.Double, 0.0), (NPTypeCode.Double, -0.0), (NPTypeCode.Double, 1.0),
            };
            foreach (var (ta, elements) in integerArrays)
                AssertAdoption(ta, elements, values);

            // float16: every bit pattern. float32: specials and, for every value below, the float32 values around it.
            var halves = Enumerable.Range(0, 65536).Select(i => (object)BitConverter.UInt16BitsToHalf((ushort)i)).ToArray();
            var floatValues = new (NPTypeCode Type, object Value)[]
            {
                (NPTypeCode.Double, 0.1), (NPTypeCode.Double, -0.1), (NPTypeCode.Double, 50.5), (NPTypeCode.Double, 1.0 + 1e-10), (NPTypeCode.Double, 1.0 - 1e-10),
                (NPTypeCode.Double, 65504.0), (NPTypeCode.Double, 65519.999), (NPTypeCode.Double, 65520.0), (NPTypeCode.Double, 1e39), (NPTypeCode.Double, -1e39),
                (NPTypeCode.Double, 1e-50), (NPTypeCode.Double, -1e-50), (NPTypeCode.Double, 0.0), (NPTypeCode.Double, -0.0), (NPTypeCode.Double, double.Epsilon),
                (NPTypeCode.Double, double.NaN), (NPTypeCode.Double, double.PositiveInfinity), (NPTypeCode.Double, double.NegativeInfinity),
                (NPTypeCode.Double, (double)float.MaxValue), (NPTypeCode.Double, (double)float.MaxValue * 1.0000001), (NPTypeCode.Double, 3.4028235677973366e38),
                (NPTypeCode.Double, 1.401298464324817e-45), (NPTypeCode.Double, 7e-46), (NPTypeCode.Double, 6.0e-8), (NPTypeCode.Double, 2.98e-8),
                (NPTypeCode.Single, 0.1f), (NPTypeCode.Single, 65520f), (NPTypeCode.Single, 2049f), (NPTypeCode.Single, 1.0000001f), (NPTypeCode.Single, float.NaN),
                (NPTypeCode.Int32, 2049), (NPTypeCode.Int32, 16777217), (NPTypeCode.Int32, -70000), (NPTypeCode.Int32, 0), (NPTypeCode.Int32, int.MaxValue),
                (NPTypeCode.Int64, 9007199254740993L), (NPTypeCode.Int64, long.MinValue), (NPTypeCode.UInt64, ulong.MaxValue), (NPTypeCode.Int16, (short)2049), (NPTypeCode.UInt16, (ushort)65535),
            };
            AssertAdoption(NPTypeCode.Half, halves, floatValues);

            var singles = new List<object> { 0f, -0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue, -float.MaxValue, float.Epsilon, -float.Epsilon };
            foreach (var (type, value) in floatValues)
            {
                double d = type == NPTypeCode.Double ? (double)value : type == NPTypeCode.Single ? (float)value : AsDouble(value);
                if (double.IsNaN(d))
                    continue;
                float nearest = (float)d;
                // The nearest float32 and two neighbours on each side: the thresholds of all six operators lie here.
                float down = nearest, up = nearest;
                for (int k = 0; k < 2; k++)
                {
                    if (!float.IsNegativeInfinity(down)) down = MathF.BitDecrement(down);
                    if (!float.IsPositiveInfinity(up)) up = MathF.BitIncrement(up);
                    singles.Add(down);
                    singles.Add(up);
                }
                singles.Add(nearest);
            }
            // Repeated past the staging threshold, so the vector kernel of the array's dtype is what compares them.
            AssertAdoption(NPTypeCode.Single, Enumerable.Repeat(singles, 2).SelectMany(x => x).ToArray(), floatValues);
        }

        /// <summary>
        ///     Compares an array holding <paramref name="elements"/> with each value, on both sides and with all six
        ///     operators, against the reference.
        /// </summary>
        /// <param name="ta">The array's dtype.</param>
        /// <param name="elements">Its elements (repeated as needed to pass the staging threshold).</param>
        /// <param name="values">The one-element operands.</param>
        /// <remarks>
        ///     A float array is checked without boxing (the float16 case is 65,536 elements against 40 values, six
        ///     operators and two sides: 31 million checks). Both operands convert exactly to float64 once they are in
        ///     the comparison dtype, and two floats compare as their float64 values do, so the reference there is one
        ///     float64 comparison per element, of the element and of the value AS CONVERTED TO THE COMPARISON DTYPE
        ///     (the rounding <see cref="Reference"/> applies).
        /// </remarks>
        private static void AssertAdoption(NPTypeCode ta, object[] elements, (NPTypeCode Type, object Value)[] values)
        {
            using var a = np.empty(new Shape(elements.Length), ta);
            for (int i = 0; i < elements.Length; i++)
                a.SetAtIndex(elements[i], i);
            bool floatArray = ta is NPTypeCode.Half or NPTypeCode.Single;
            double[] asDouble = floatArray ? elements.Select(AsDouble).ToArray() : null;
            foreach (var (tv, value) in values)
            {
                if (tv == ta)
                    continue;
                using var s = NDArray.Scalar(value, tv);
                // The value as the comparison dtype holds it, as a float64 (float arrays only).
                double t = 0;
                if (floatArray)
                {
                    t = np._FindCommonScalarType(ta, tv) switch
                    {
                        NPTypeCode.Half => (double)(Half)AsDouble(value),
                        NPTypeCode.Single => (float)AsDouble(value),
                        NPTypeCode.Double => AsDouble(value),
                        var other => throw new InvalidOperationException($"{ta} with {tv} compares in {other}"),
                    };
                }
                foreach (var op in Ops)
                {
                    using var right = Compare(op, a, s);
                    using var left = Compare(op, s, a);
                    var r = right.ToArray<bool>();
                    var l = left.ToArray<bool>();
                    for (int i = 0; i < elements.Length; i++)
                    {
                        bool wantRight = floatArray ? Cmp(asDouble[i], t, op) : Reference(op, elements[i], ta, value, tv);
                        bool wantLeft = floatArray ? Cmp(t, asDouble[i], op) : Reference(op, value, tv, elements[i], ta);
                        if (r[i] != wantRight)
                            Assert.Fail($"{ta} array {op} {tv} value: {elements[i]} {op} {value} should be {wantRight}");
                        if (l[i] != wantLeft)
                            Assert.Fail($"{tv} value {op} {ta} array: {value} {op} {elements[i]} should be {wantLeft}");
                    }
                }
            }
        }

        /// <summary>
        ///     int64 against uint64 compares the two integers exactly on every route (NumPy's qQ loops): operators,
        ///     <c>out=</c>, <c>where=</c>, a stepped operand. float64 holds neither 2^63 - 1 nor 2^53 + 1.
        /// </summary>
        [TestMethod]
        public void Int64AgainstUInt64_IsExact_OnEveryRoute()
        {
            // NumPy 2.4.2: np.array([2**63 - 1], np.int64) < np.array([2**63], np.uint64) -> [ True];
            // np.equal(np.array([2**53 + 1], np.int64), np.array([2**53], np.uint64)) -> [False]
            long[] signed = { long.MaxValue, 9007199254740993L, -1L, 0L, long.MinValue, 5L };
            ulong[] unsigned = { 9223372036854775808UL, 9007199254740992UL, ulong.MaxValue, 0UL, 0UL, 5UL };
            bool[] less = { true, false, true, false, true, false };
            bool[] equal = { false, false, false, true, false, true };
            foreach (int repeat in new[] { 1, 40 })
            {
                using var s = np.array(Enumerable.Repeat(signed, repeat).SelectMany(x => x).ToArray());
                using var u = np.array(Enumerable.Repeat(unsigned, repeat).SelectMany(x => x).ToArray());
                bool[] wantLess = Enumerable.Repeat(less, repeat).SelectMany(x => x).ToArray();
                bool[] wantEqual = Enumerable.Repeat(equal, repeat).SelectMany(x => x).ToArray();

                (s < u).ToArray<bool>().Should().Equal(wantLess);
                (u > s).ToArray<bool>().Should().Equal(wantLess);
                (s == u).ToArray<bool>().Should().Equal(wantEqual);
                (u != s).ToArray<bool>().Should().Equal(wantEqual.Select(x => !x));

                using var o = np.empty(new Shape(s.size), np.bool_);
                np.less(s, u, o).ToArray<bool>().Should().Equal(wantLess);
                np.equal(u, s, o).ToArray<bool>().Should().Equal(wantEqual);

                using var mask = np.ones(new Shape(s.size), np.bool_);
                using var masked = np.zeros(new Shape(s.size), np.bool_);
                np.equal(s, u, masked, mask).ToArray<bool>().Should().Equal(wantEqual);

                using var s2 = np.array(Enumerable.Repeat(signed, repeat).SelectMany(x => x).SelectMany(x => new[] { x, 0L }).ToArray());
                (s2["::2"] < u).ToArray<bool>().Should().Equal(wantLess);
            }
        }

        /// <summary>
        ///     <c>out=</c> that is the same memory as an input, walked the same way (bool and uint8 share an item
        ///     size): the result is the comparison of the ORIGINAL values, in the contiguous, stepped and reversed
        ///     layouts.
        /// </summary>
        [TestMethod]
        public void OutThatIsAnInput_ComparesTheOriginalValues()
        {
            foreach (long n in new long[] { 5, 70, 300, 9000 })
            foreach (string layout in new[] { ":", "::2", "::-1" })
            foreach (var op in Ops)
            {
                using var aBase = Make(NPTypeCode.Byte, NPTypeCode.Byte, 2 * n, 0);
                using var bBase = Make(NPTypeCode.Byte, NPTypeCode.Byte, 2 * n, 3);
                NDArray a = aBase[layout], b = bBase[layout];
                using var expected = Compare(op, a.copy(), b.copy());
                bool[] want = expected.ToArray<bool>();

                NDArray o = aBase.view(np.bool_)[layout];
                Compare(op, a, b, o);
                o.ToArray<bool>().Should().Equal(want, $"{op} n={n} layout {layout}: out aliases the left operand");
            }
        }

        /// <summary>
        ///     Two one-element operands of different rank give NumPy's broadcast shape (NumPy 2.4.2:
        ///     <c>(np.ones(1) &lt; np.float64(2)).shape == (1,)</c>, <c>(np.ones((1,1)) &lt; np.ones(1)).shape == (1, 1)</c>).
        /// </summary>
        [TestMethod]
        public void OneElementOperandsOfDifferentRank_GiveTheBroadcastShape()
        {
            using var one = np.array(new[] { 1.5 });
            using var zeroD = NDArray.Scalar(2.5);
            using var oneOne = np.array(new[] { 1.5 }).reshape(1, 1);
            using var asInt = np.array(new[] { 1 });

            using (var r = one < zeroD) { r.shape.Should().Equal(1L); r.GetBoolean(0).Should().BeTrue(); }
            using (var r = zeroD < one) { r.shape.Should().Equal(1L); r.GetBoolean(0).Should().BeFalse(); }
            using (var r = oneOne < one) { r.shape.Should().Equal(1L, 1L); r.GetBoolean(0, 0).Should().BeFalse(); }
            using (var r = one <= oneOne) { r.shape.Should().Equal(1L, 1L); r.GetBoolean(0, 0).Should().BeTrue(); }
            using (var r = asInt < zeroD) { r.shape.Should().Equal(1L); r.GetBoolean(0).Should().BeTrue(); }
            using (var r = oneOne > asInt) { r.shape.Should().Equal(1L, 1L); r.GetBoolean(0, 0).Should().BeTrue(); }
        }

        /// <summary>
        ///     The result's memory order is NumPy's: F when every array operand is F-ordered, C as soon as one is
        ///     C-ordered (the route without an iterator declines the F cases).
        /// </summary>
        [TestMethod]
        public void ResultOrder_FollowsTheOperands()
        {
            using var c = np.arange(12).reshape(3, 4).astype(np.float64);
            using var f = c.copy('F');
            using var f2 = (c + 1).copy('F');
            using var row = np.arange(4).astype(np.float64);

            using (var r = f < f2) r.Shape.IsFContiguous.Should().BeTrue("F with F is F-ordered");
            using (var r = c < f) { r.Shape.IsContiguous.Should().BeTrue("C with F is C-ordered"); }
            using (var r = f < row) r.Shape.IsFContiguous.Should().BeTrue("F with a row is F-ordered");
            using (var r = c < row) r.Shape.IsContiguous.Should().BeTrue("C with a row is C-ordered");
            using (var r = f < row)
                AssertCompared("F / row", r, f, NPTypeCode.Double, row, NPTypeCode.Double, ComparisonOp.Less);
        }

        /// <summary>
        ///     With the run switched off every comparison takes its former route, and the two routes agree bit for
        ///     bit on every layout (the former route is the per-element kernel NumPy's results were validated on).
        /// </summary>
        [TestMethod]
        public void RunAndFormerRoutes_Agree()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Byte, NPTypeCode.Double), (NPTypeCode.Half, NPTypeCode.Int32),
                (NPTypeCode.Complex, NPTypeCode.Complex), (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.Boolean, NPTypeCode.Boolean),
                (NPTypeCode.Single, NPTypeCode.Double), (NPTypeCode.Int16, NPTypeCode.UInt16), (NPTypeCode.Char, NPTypeCode.Char),
            };
            foreach (var (ta, tb) in pairs)
            {
                using var aFlat = Make(ta, tb, 6 * 90, 0);
                using var bFlat = Make(tb, ta, 6 * 90, 3);
                NDArray a = aFlat.reshape(6, 90), b = bFlat.reshape(6, 90);
                using var row = Make(tb, ta, 90, 5);
                using var value = NDArray.Scalar(PoolFor(tb, ta)[2 % PoolFor(tb, ta).Length], tb);
                var layouts = new (string Name, NDArray L, NDArray R)[]
                {
                    ("contiguous", a, b), ("row", a, row), ("value", a, value), ("reversed columns", a[":, ::-1"], b[":, ::-1"]),
                    ("stepped columns", a[":, ::3"], b[":, ::3"]), ("transposed against contiguous", a.T, b.reshape(90, 6)),
                    ("stepped rows", a["::2"], b["::2"]),
                };
                foreach (var op in Ops)
                foreach (var (name, l, r) in layouts)
                {
                    bool[] with, without;
                    using (var g = Compare(op, l, r)) with = g.ToArray<bool>();
                    ComparisonRun.Disabled = true;
                    try
                    {
                        using var g = Compare(op, l, r);
                        without = g.ToArray<bool>();
                    }
                    finally
                    {
                        ComparisonRun.Disabled = false;
                    }
                    with.Should().Equal(without, $"{op} {ta} x {tb} {name}");
                }
            }
        }

        /// <summary>The run exists for every dtype pair and operator on a host with runtime IL generation.</summary>
        [TestMethod]
        public void Run_ExistsForEveryDtypePair()
        {
            if (!DirectILKernelGenerator.Enabled)
                Assert.Inconclusive("Runtime IL generation is off on this host.");
            foreach (var ta in AllTypes)
            foreach (var tb in AllTypes)
            foreach (var op in Ops)
                ComparisonRun.Get(op, ta, tb).Should().NotBeNull($"{op} {ta} x {tb}");
        }

        // =====================================================================================================
        //  ComparisonRun on raw memory
        // =====================================================================================================

        /// <summary>
        ///     <see cref="ComparisonRun.Run"/> on raw buffers with sentinel margins: every input stride (contiguous,
        ///     2, 3, -1, -2, one value) with every output stride (1, 2, 3, -1), at lengths on both sides of the staging
        ///     threshold and across block edges. The results are right, and no byte outside the output's slots changes.
        /// </summary>
        [TestMethod]
        public void Run_OnRawMemory_WritesExactlyItsSlots()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean),
                (NPTypeCode.Int16, NPTypeCode.Double), (NPTypeCode.Complex, NPTypeCode.Complex), (NPTypeCode.Int64, NPTypeCode.UInt64),
                (NPTypeCode.Byte, NPTypeCode.Int32), (NPTypeCode.Single, NPTypeCode.Double),
            };
            long[] strides = { 1, 2, 3, -1, -2, 0 };
            long[] outStrides = { 1, 2, 3, -1 };
            foreach (var (ta, tb) in pairs)
            foreach (long n in new long[] { 1, 63, 64, 1023, 1025, 8200 })
            foreach (long ls in strides)
            foreach (long rs in strides)
            foreach (long os in outStrides)
            {
                // The full stride matrix on the short lengths; the long ones (block edges) on the diagonal cases.
                if (n > 1100 && !(ls == rs || ls == 1 || rs == 0))
                    continue;
                // The long runs are there for the block edges of the staging pass: one operator is enough.
                foreach (var op in n > 1100 ? new[] { ComparisonOp.Less } : new[] { ComparisonOp.Less, ComparisonOp.Equal, ComparisonOp.GreaterEqual })
                {
                    using var left = new RawRun(ta, tb, n, ls, phase: 0);
                    using var right = new RawRun(tb, ta, n, rs, phase: 3);
                    using var result = new RawOutput(n, os);
                    var run = ComparisonRun.Get(op, ta, tb);
                    run.Should().NotBeNull();
                    run.Run(left.First, ls * left.ItemSize, right.First, rs * right.ItemSize, result.First, os, n);
                    for (long i = 0; i < n; i++)
                    {
                        bool want = Reference(op, left.Value(i), ta, right.Value(i), tb);
                        if ((result.First[i * os] != 0) != want || result.First[i * os] > 1)
                            Assert.Fail($"{op} {ta} x {tb} n={n} strides {ls}/{rs}/{os} [{i}]: {left.Value(i)} {op} {right.Value(i)} should be {want}, slot holds {result.First[i * os]}");
                    }
                    result.AssertOnlySlotsWritten($"{op} {ta} x {tb} n={n} strides {ls}/{rs}/{os}");
                }
            }
        }

        /// <summary>
        ///     <see cref="ComparisonRun.RunRows"/> on raw buffers: rows of every form (two runs, a run and a value per
        ///     row, backward runs, stepped runs), row lengths that leave every partial vector of the one-byte kernels,
        ///     and outer strides of either sign. The results are right and the margins are untouched.
        /// </summary>
        [TestMethod]
        public void RunRows_OnRawMemory_WritesExactlyItsSlots()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Double, NPTypeCode.Double),
                (NPTypeCode.Int32, NPTypeCode.Double), (NPTypeCode.SByte, NPTypeCode.SByte),
            };
            foreach (var (ta, tb) in pairs)
            foreach (long count in new long[] { 5, 16, 31, 32, 33, 40, 47, 63, 64, 100, 130 })
            foreach (long rows in new long[] { 1, 3, 7 })
            foreach (var (lInner, rInner) in new (long, long)[] { (1, 1), (1, 0), (0, 1), (-1, -1), (-1, 0), (2, 1), (1, -1) })
            foreach (bool backwardRows in new[] { false, true })
            {
                var op = ComparisonOp.Less;
                int lSize = DirectILKernelGenerator.GetTypeSize(ta), rSize = DirectILKernelGenerator.GetTypeSize(tb);
                // Each operand is rows x count elements (a value per row when its inner stride is 0), laid out as a
                // matrix whose rows are walked forwards or backwards.
                using var left = new RawMatrix(ta, tb, rows, count, lInner, backwardRows, phase: 1);
                using var right = new RawMatrix(tb, ta, rows, count, rInner, backwardRows, phase: 4);
                using var result = new RawOutput(rows * count, 1);
                var run = ComparisonRun.Get(op, ta, tb);
                run.RunRows(left.First, lInner * lSize, left.OuterStride, right.First, rInner * rSize, right.OuterStride,
                    result.First, 1, count, rows, count);
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < count; j++)
                {
                    bool want = Reference(op, left.Value(i, j), ta, right.Value(i, j), tb);
                    if ((result.First[i * count + j] != 0) != want)
                        Assert.Fail($"{ta} x {tb} rows={rows} count={count} inner {lInner}/{rInner} backwardRows={backwardRows} [{i},{j}]: {left.Value(i, j)} < {right.Value(i, j)} should be {want}");
                }
                result.AssertOnlySlotsWritten($"{ta} x {tb} rows={rows} count={count} inner {lInner}/{rInner}");
            }
        }

        /// <summary>
        ///     <see cref="ComparisonRun.Run"/> with every operand placed against an INACCESSIBLE page: the lowest byte
        ///     of each run directly above one, or its highest byte directly below one. A kernel, a staging copy or a
        ///     store that touches one byte outside its operand ends the test host with an access violation; the values
        ///     and the untouched bytes between the output slots are checked as in the heap test.
        /// </summary>
        /// <remarks>
        ///     Value tests cannot see a read outside an operand (the bytes read are discarded) and sentinel margins see
        ///     writes only. This is the test for reads: the vector tails of every kernel, the alignment prefix, the
        ///     house strided copies of the staging pass (a step of two reads two vectors per result vector, a reversed
        ///     run is read from its last element down), the strided move and the reversed store. Windows only (the
        ///     fences are reserved, uncommitted pages).
        /// </remarks>
        [TestMethod]
        public void Run_BetweenInaccessiblePages_TouchesNoByteOutsideItsOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            var pairs = new (NPTypeCode, NPTypeCode, ComparisonOp[])[]
            {
                (NPTypeCode.Double, NPTypeCode.Double, new[] { ComparisonOp.Less }),
                (NPTypeCode.Single, NPTypeCode.Single, new[] { ComparisonOp.GreaterEqual }),
                (NPTypeCode.Byte, NPTypeCode.Byte, new[] { ComparisonOp.Less }),
                (NPTypeCode.Boolean, NPTypeCode.Boolean, new[] { ComparisonOp.NotEqual }),
                (NPTypeCode.Int16, NPTypeCode.Int16, new[] { ComparisonOp.Greater }),
                (NPTypeCode.Half, NPTypeCode.Half, new[] { ComparisonOp.LessEqual }),
                (NPTypeCode.Complex, NPTypeCode.Complex, new[] { ComparisonOp.Less, ComparisonOp.Equal }),
                (NPTypeCode.Int64, NPTypeCode.UInt64, new[] { ComparisonOp.Less, ComparisonOp.Equal }),
                (NPTypeCode.UInt64, NPTypeCode.Int64, new[] { ComparisonOp.GreaterEqual }),
                (NPTypeCode.Int16, NPTypeCode.Double, new[] { ComparisonOp.Less }),
                (NPTypeCode.Byte, NPTypeCode.Int32, new[] { ComparisonOp.Less }),
                (NPTypeCode.Single, NPTypeCode.Double, new[] { ComparisonOp.Equal }),
                (NPTypeCode.Half, NPTypeCode.Int32, new[] { ComparisonOp.Less }),
                (NPTypeCode.Decimal, NPTypeCode.Decimal, new[] { ComparisonOp.Less }),
            };
            long[] strides = { 1, 2, 3, -1, -2, -3, 0 };
            long[] outStrides = { 1, 2, -1, -3 };
            // Where each operand sits: all at the low edge of their fences, all at the high edge, and the two mixes.
            // A case takes two of the four placements, chosen by its position in the matrix, so that every stride
            // form meets every placement without running the matrix four times.
            var edges = new (bool L, bool R, bool O)[] { (false, false, false), (true, true, true), (false, true, true), (true, false, false) };

            using var lFence = new FencedMemory(pages: 16);
            using var rFence = new FencedMemory(pages: 16);
            using var oFence = new FencedMemory(pages: 2);
            int position = 0;
            foreach (var (ta, tb, ops) in pairs)
            foreach (long n in new long[] { 1, 17, 63, 64, 65, 200, 1025, 5000 })
            foreach (long ls in strides)
            foreach (long rs in strides)
            foreach (long os in outStrides)
            {
                // The long runs (block edges of the staging pass, the alignment prefix: 5,000 one-byte elements) take
                // the stride forms that reach each mechanism once; the short ones take the whole matrix.
                if (n > 200 && !((ls == rs || ls == 1 || rs == 0) && (os == 1 || os == -1)))
                    continue;
                if (!RawRun.Fits(lFence, ta, n, ls) || !RawRun.Fits(rFence, tb, n, rs) || !RawOutput.Fits(oFence, n, os))
                    continue;
                position++;
                for (int e = 0; e < 2; e++)
                foreach (var op in ops)
                {
                    // The element count takes part, so one stride form meets all four placements over the lengths.
                    var (lEnd, rEnd, oEnd) = edges[(position + (int)(n & 3) + 2 * e) & 3];
                    using var left = new RawRun(ta, tb, n, ls, phase: 2, lFence, lEnd);
                    using var right = new RawRun(tb, ta, n, rs, phase: 5, rFence, rEnd);
                    using var result = new RawOutput(n, os, oFence, oEnd);
                    ComparisonRun.Get(op, ta, tb).Run(left.First, ls * left.ItemSize, right.First, rs * right.ItemSize, result.First, os, n);
                    for (long i = 0; i < n; i++)
                    {
                        bool want = Reference(op, left.Value(i), ta, right.Value(i), tb);
                        if (result.First[i * os] != (want ? 1 : 0))
                            Assert.Fail($"{op} {ta} x {tb} n={n} strides {ls}/{rs}/{os} edges {lEnd}/{rEnd}/{oEnd} [{i}]: {left.Value(i)} {op} {right.Value(i)} should be {want}, slot holds {result.First[i * os]}");
                    }
                    result.AssertOnlySlotsWritten($"{op} {ta} x {tb} n={n} strides {ls}/{rs}/{os} edges {lEnd}/{rEnd}/{oEnd}");
                }
            }
        }

        /// <summary>
        ///     <see cref="ComparisonRun.RunRows"/> with both operand matrices and the output placed against inaccessible
        ///     pages (see <see cref="Run_BetweenInaccessiblePages_TouchesNoByteOutsideItsOperands"/>): the per-row
        ///     kernel calls, the overlapped last vector of one-byte rows and the backward-row loop read and write
        ///     nothing outside the rows.
        /// </summary>
        [TestMethod]
        public void RunRows_BetweenInaccessiblePages_TouchesNoByteOutsideItsOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Double, NPTypeCode.Double),
                (NPTypeCode.Int32, NPTypeCode.Double), (NPTypeCode.Int16, NPTypeCode.Int16), (NPTypeCode.Complex, NPTypeCode.Complex),
                (NPTypeCode.Int64, NPTypeCode.UInt64),
            };
            using var lFence = new FencedMemory(pages: 16);
            using var rFence = new FencedMemory(pages: 16);
            using var oFence = new FencedMemory(pages: 2);
            foreach (var (ta, tb) in pairs)
            foreach (long count in new long[] { 5, 16, 31, 32, 33, 40, 47, 63, 64, 100, 130 })
            foreach (long rows in new long[] { 1, 3, 7 })
            foreach (var (lInner, rInner) in new (long, long)[] { (1, 1), (1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (2, 1), (1, -1), (-2, 3) })
            foreach (bool backwardRows in new[] { false, true })
            foreach (bool atEnd in new[] { false, true })
            {
                var op = ComparisonOp.LessEqual;
                int lSize = DirectILKernelGenerator.GetTypeSize(ta), rSize = DirectILKernelGenerator.GetTypeSize(tb);
                using var left = new RawMatrix(ta, tb, rows, count, lInner, backwardRows, phase: 1, lFence, atEnd);
                using var right = new RawMatrix(tb, ta, rows, count, rInner, backwardRows, phase: 4, rFence, atEnd);
                using var result = new RawOutput(rows * count, 1, oFence, atEnd);
                ComparisonRun.Get(op, ta, tb).RunRows(left.First, lInner * lSize, left.OuterStride, right.First, rInner * rSize, right.OuterStride,
                    result.First, 1, count, rows, count);
                string what = $"{ta} x {tb} rows={rows} count={count} inner {lInner}/{rInner} backwardRows={backwardRows} atEnd={atEnd}";
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < count; j++)
                {
                    bool want = Reference(op, left.Value(i, j), ta, right.Value(i, j), tb);
                    if (result.First[i * count + j] != (want ? 1 : 0))
                        Assert.Fail($"{what} [{i},{j}]: {left.Value(i, j)} <= {right.Value(i, j)} should be {want}");
                }
                result.AssertOnlySlotsWritten(what);
            }
        }

        // =====================================================================================================
        //  Tiles (an operand whose successive rows are adjacent elements) and a matrix against one row
        // =====================================================================================================

        /// <summary>The dtype pairs whose tiled layouts are held to the element reference at every shape.</summary>
        private static readonly (NPTypeCode, NPTypeCode)[] TileCorePairs =
        {
            (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Byte, NPTypeCode.Byte),
            (NPTypeCode.Int16, NPTypeCode.Int16), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Int64, NPTypeCode.UInt64),
            (NPTypeCode.Int32, NPTypeCode.Double),
        };

        /// <summary>More pairs, at the shorter shapes: other widths, conversions on either side, 16-byte elements.</summary>
        private static readonly (NPTypeCode, NPTypeCode)[] TileMorePairs =
        {
            (NPTypeCode.Int32, NPTypeCode.Int32), (NPTypeCode.Half, NPTypeCode.Half), (NPTypeCode.Complex, NPTypeCode.Complex),
            (NPTypeCode.Decimal, NPTypeCode.Decimal), (NPTypeCode.UInt64, NPTypeCode.Int64), (NPTypeCode.Double, NPTypeCode.Int32),
            (NPTypeCode.Byte, NPTypeCode.Single), (NPTypeCode.Boolean, NPTypeCode.Int32), (NPTypeCode.Single, NPTypeCode.Complex),
            (NPTypeCode.Char, NPTypeCode.Int16), (NPTypeCode.SByte, NPTypeCode.Byte), (NPTypeCode.Int16, NPTypeCode.Int64),
        };

        /// <summary>
        ///     A C-ordered operand against an F-ordered one, in every form the tile route serves: the F operand on
        ///     either side, with its rows or its columns reversed, two F-ordered slices, and a column (one value a row)
        ///     against an F operand. Shapes on both sides of the vector groups of the transposition, of the staging
        ///     threshold and of one tile's width.
        /// </summary>
        /// <remarks>
        ///     The expected values come from <see cref="Reference"/> through <see cref="AssertCompared"/>, which reads
        ///     both operands element by element in logical order: it does not depend on the route under test.
        /// </remarks>
        [TestMethod]
        public void TiledOperands_MatchAnElementReference()
        {
            var shortShapes = new[] { (2, 2), (3, 5), (17, 33), (40, 25), (300, 3) };
            var allShapes = new[] { (2, 2), (3, 5), (17, 33), (25, 40), (40, 25), (7, 150), (300, 3), (9, 600) };
            long tilesBefore = ComparisonRun.TileRuns;
            foreach (var (pairs, shapes) in new[] { (TileCorePairs, allShapes), (TileMorePairs, shortShapes) })
            foreach (var (ta, tb) in pairs)
            foreach (var (rows, cols) in shapes)
            {
                long n = (long)rows * cols;
                using var aFlat = Make(ta, tb, n, 2);
                using var bFlat = Make(tb, ta, n, 4);
                NDArray a = aFlat.reshape(rows, cols), b = bFlat.reshape(rows, cols);
                using var aF = a.copy('F');
                using var bF = b.copy('F');
                using var colFlat = Make(ta, tb, rows, 1);
                NDArray col = colFlat.reshape(rows, 1);
                // Two operators a shape: < always, and one that changes with the shape and the dtype.
                foreach (var op in new[] { ComparisonOp.Less, Ops[(rows + cols + (int)ta) % Ops.Length] })
                {
                    string tag = $"{op} {ta} x {tb} ({rows},{cols})";
                    using (var g = Compare(op, a, bF)) AssertCompared(tag + " C / F", g, a, ta, bF, tb, op);
                    using (var g = Compare(op, aF, b)) AssertCompared(tag + " F / C", g, aF, ta, b, tb, op);
                    using (var g = Compare(op, a, bF["::-1"])) AssertCompared(tag + " C / F with reversed rows", g, a, ta, bF["::-1"], tb, op);
                    using (var g = Compare(op, a, bF[":, ::-1"])) AssertCompared(tag + " C / F with reversed columns", g, a, ta, bF[":, ::-1"], tb, op);
                    using (var g = Compare(op, aF["1:, 1:"], bF[":-1, :-1"])) AssertCompared(tag + " two F slices", g, aF["1:, 1:"], ta, bF[":-1, :-1"], tb, op);
                    using (var g = Compare(op, col, bF)) AssertCompared(tag + " column / F", g, col, ta, bF, tb, op);
                }
            }
            if (DirectILKernelGenerator.Enabled)
                (ComparisonRun.TileRuns - tilesBefore).Should().BeGreaterThan(500, "most of these layouts run tile by tile");
        }

        /// <summary>
        ///     The tile route, the repeated-row route and the row-by-row loop they replace give the same bytes, which
        ///     are also the bytes of the comparison of two C-ordered copies; and the routes are taken for the layouts
        ///     they are meant for, and declined for the ones they are not.
        /// </summary>
        /// <remarks>
        ///     The large shapes reach what the reference test cannot afford: rows cut into several tiles, the overlapped
        ///     last tile of both directions, and the taller tiles of an operand whose rows span 4 MB or more.
        /// </remarks>
        [TestMethod]
        public void TilesAndRepeatedRows_AgreeWithTheRowByRowRoute()
        {
            if (!DirectILKernelGenerator.Enabled)
                Assert.Inconclusive("Runtime IL generation is off: there is no comparison run.");

            // (dtype, rows, cols, whether the tile route serves C / F of that shape)
            var cases = new (NPTypeCode Type, int Rows, int Cols)[]
            {
                (NPTypeCode.Double, 250, 400), (NPTypeCode.Byte, 250, 400), (NPTypeCode.Boolean, 251, 397), (NPTypeCode.Single, 64, 4096),
                (NPTypeCode.Int16, 33, 1000), (NPTypeCode.Int64, 400, 250), (NPTypeCode.UInt32, 129, 513),
                // Rows that span 4 MB or more: the taller tiles.
                (NPTypeCode.Double, 700, 800), (NPTypeCode.Byte, 2100, 2100), (NPTypeCode.Int16, 1100, 2000), (NPTypeCode.Single, 1100, 1000),
            };
            foreach (var (type, rows, cols) in cases)
            {
                long n = (long)rows * cols;
                using var aFlat = ((np.arange(n) * 7) % 251).astype(type);
                using var bFlat = ((np.arange(n) * 13) % 251).astype(type);
                NDArray a = aFlat.reshape(rows, cols), b = bFlat.reshape(rows, cols);
                using var bF = b.copy('F');
                using var aF = a.copy('F');
                using var row = ((np.arange(cols) * 5) % 251).astype(type);
                using var rowAsMatrix = np.broadcast_to(row, a.Shape).copy();
                string tag = $"{type} ({rows},{cols})";

                using var expected = a < b;
                long before = ComparisonRun.TileRuns;
                using (var g = a < bF) np.array_equal(g, expected).Should().BeTrue(tag + " C < F");
                using (var g = aF < b) np.array_equal(g, expected).Should().BeTrue(tag + " F < C");
                (ComparisonRun.TileRuns - before).Should().Be(2, tag + ": both calls run tile by tile");

                using var expectedRow = a <= rowAsMatrix;
                long rowsBefore = ComparisonRun.RepeatedRowRuns;
                using (var g = a <= row) np.array_equal(g, expectedRow).Should().BeTrue(tag + " matrix <= row");
                bool repeats = (long)cols * a.dtypesize <= 2048;
                (ComparisonRun.RepeatedRowRuns - rowsBefore).Should().Be(repeats ? 1 : 0, tag + ": a row of " + (long)cols * a.dtypesize + " bytes");

                ComparisonRun.TilesDisabled = true;
                ComparisonRun.RepeatedRowDisabled = true;
                try
                {
                    before = ComparisonRun.TileRuns;
                    rowsBefore = ComparisonRun.RepeatedRowRuns;
                    using (var g = a < bF) np.array_equal(g, expected).Should().BeTrue(tag + " C < F, row by row");
                    using (var g = aF < b) np.array_equal(g, expected).Should().BeTrue(tag + " F < C, row by row");
                    using (var g = a <= row) np.array_equal(g, expectedRow).Should().BeTrue(tag + " matrix <= row, a call a row");
                    (ComparisonRun.TileRuns - before).Should().Be(0);
                    (ComparisonRun.RepeatedRowRuns - rowsBefore).Should().Be(0);
                }
                finally
                {
                    ComparisonRun.TilesDisabled = false;
                    ComparisonRun.RepeatedRowDisabled = false;
                }
            }

            // Declined: long rows of 16-byte elements, a conversion to complex128, and two F operands (an F-ordered
            // result, which the iterator route computes).
            {
                using var c = (np.arange(1000).astype(NPTypeCode.Complex)).reshape(10, 100);
                using var cF = c.copy('F');
                using var f32 = (np.arange(1000) % 97).astype(NPTypeCode.Single).reshape(25, 40);
                using var c2540 = (np.arange(1000) % 89).astype(NPTypeCode.Complex).reshape(25, 40);
                using var c2540F = c2540.copy('F');
                using var f32F = f32.copy('F');
                long before = ComparisonRun.TileRuns;
                using (var g = c < cF) np.array_equal(g, c < c).Should().BeTrue();
                using (var g = f32 < c2540F) np.array_equal(g, f32 < c2540).Should().BeTrue();
                using (var g = f32F < c2540F) g.Shape.IsFContiguous.Should().BeTrue();
                (ComparisonRun.TileRuns - before).Should().Be(0, "none of the three is served by tiles");

                // Short rows of 16-byte elements are.
                using (var g = c2540 <= c2540F) np.array_equal(g, c2540 <= c2540).Should().BeTrue();
                (ComparisonRun.TileRuns - before).Should().Be(1);
            }
        }

        /// <summary>
        ///     A matrix against one row (and one row against a matrix) at row lengths from 2 elements up, with enough
        ///     rows for the repeated-row route: the values are the element reference's.
        /// </summary>
        [TestMethod]
        public void MatrixAgainstOneRow_MatchesAnElementReference()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Int16, NPTypeCode.Int16),
                (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Complex, NPTypeCode.Complex),
                (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.Half, NPTypeCode.Half),
            };
            long before = ComparisonRun.RepeatedRowRuns;
            foreach (var (ta, tb) in pairs)
            foreach (var (rows, cols) in new[] { (16, 2), (17, 3), (40, 33), (300, 3), (16, 100), (21, 40), (64, 16), (15, 40) })
            {
                long n = (long)rows * cols;
                using var aFlat = Make(ta, tb, n, 2);
                using var bFlat = Make(tb, ta, n, 5);
                NDArray a = aFlat.reshape(rows, cols), b = bFlat.reshape(rows, cols);
                using var rowB = Make(tb, ta, cols, 6);
                using var rowA = Make(ta, tb, cols, 3);
                foreach (var op in new[] { ComparisonOp.Less, Ops[(rows + cols) % Ops.Length] })
                {
                    string tag = $"{op} {ta} x {tb} ({rows},{cols})";
                    using (var g = Compare(op, a, rowB)) AssertCompared(tag + " matrix / row", g, a, ta, rowB, tb, op);
                    using (var g = Compare(op, rowA, b)) AssertCompared(tag + " row / matrix", g, rowA, ta, b, tb, op);
                    // Rows that do NOT follow each other in memory (every second row, the rows in reverse) must
                    // not be compared as one block.
                    using (var g = Compare(op, a["::2"], rowB)) AssertCompared(tag + " every second row / row", g, a["::2"], ta, rowB, tb, op);
                    using (var g = Compare(op, rowA, b["::-1"])) AssertCompared(tag + " row / reversed rows", g, rowA, ta, b["::-1"], tb, op);
                }
            }
            if (DirectILKernelGenerator.Enabled)
                (ComparisonRun.RepeatedRowRuns - before).Should().BeGreaterThan(150, "every shape of 16 rows or more takes the repeated-row route");
        }

        /// <summary>
        ///     <see cref="ComparisonRun.RunRows"/> on raw memory with a tiled operand on either side, in every form of
        ///     the operand beside it (rows that follow each other, rows that do not, one value a row, a second tiled
        ///     operand), with all operands and the output against inaccessible pages. The values are right, the bytes
        ///     around the output are untouched, and no byte outside an operand is read.
        /// </summary>
        /// <remarks>
        ///     The tiled operand's columns have a gap after each column piece in half of the cases, and its rows or
        ///     its columns run downwards in memory in turn, so the tile's edges are the buffer's edges in every
        ///     direction. On a host without <see cref="FencedMemory"/> the operands are heap blocks and the margins of
        ///     the output are checked only.
        /// </remarks>
        [TestMethod]
        public void RunRows_Tiles_OnRawMemory_TouchNoByteOutsideTheirOperands()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Int16, NPTypeCode.Int16),
                (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Int32, NPTypeCode.Double),
                (NPTypeCode.Double, NPTypeCode.Int16), (NPTypeCode.Int64, NPTypeCode.UInt64), (NPTypeCode.Complex, NPTypeCode.Complex),
            };
            bool fenced = FencedMemory.IsSupported;
            using var lFence = fenced ? new FencedMemory(pages: 80) : null;
            using var rFence = fenced ? new FencedMemory(pages: 80) : null;
            using var oFence = fenced ? new FencedMemory(pages: 8) : null;
            int position = 0;
            long tilesBefore = ComparisonRun.TileRuns;
            foreach (var (ta, tb) in pairs)
            foreach (long rows in new long[] { 2, 9, 16, 17, 33 })
            foreach (long count in new long[] { 4, 31, 33, 64, 70, 300, 600 })
            foreach (int form in new[] { 0, 1, 2, 3 })
            foreach (bool tiledOnLeft in new[] { true, false })
            {
                if (rows * count < ComparisonRun.StagedMinElements)
                    continue;
                position++;
                // The variations each case takes in turn: a gap after each column piece, downward rows, downward columns,
                // and the edge of the fences the operands lie against.
                long gap = (position & 1) == 0 ? 0 : 3;
                bool downRows = (position & 2) != 0, downColumns = (position & 4) != 0, atEnd = (position & 8) != 0;
                var op = Ops[position % Ops.Length];
                NPTypeCode tiledType = tiledOnLeft ? ta : tb, otherType = tiledOnLeft ? tb : ta;
                FencedMemory tiledFence = tiledOnLeft ? lFence : rFence, otherFence = tiledOnLeft ? rFence : lFence;
                // The operand beside the tiled one is checked at the largest gap any of its forms has.
                if (fenced && (!RawTiled.Fits(tiledFence, tiledType, rows, count, gap) || !RawTiled.Fits(otherFence, otherType, rows, count, 3)
                               || rows * count > oFence.Bytes))
                    continue;

                using var tiled = new RawTiled(tiledType, otherType, rows, count, gap, downRows, downColumns, phase: 1, tiledFence, atEnd);
                // The operand beside it: 0 rows that follow each other, 1 rows laid out downwards (a kernel call a
                // row), 2 one value a row, 3 a second tiled operand.
                using var otherRows = form <= 2
                    ? new RawMatrix(otherType, tiledType, rows, count, inner: form == 2 ? 0 : 1, backwardRows: form == 1, phase: 4, otherFence, !atEnd)
                    : null;
                using var otherTiled = form == 3
                    ? new RawTiled(otherType, tiledType, rows, count, gap == 0 ? 2 : 0, !downRows, downColumns, phase: 4, otherFence, !atEnd)
                    : null;
                int otherSize = DirectILKernelGenerator.GetTypeSize(otherType);
                byte* otherFirst = form == 3 ? otherTiled.First : otherRows.First;
                long otherInner = form == 3 ? otherTiled.InnerStride : (form == 2 ? 0 : otherSize);
                long otherOuter = form == 3 ? otherTiled.OuterStride : otherRows.OuterStride;

                using var result = new RawOutput(rows * count, 1, oFence, atEnd);
                var run = ComparisonRun.Get(op, ta, tb);
                if (tiledOnLeft)
                    run.RunRows(tiled.First, tiled.InnerStride, tiled.OuterStride, otherFirst, otherInner, otherOuter, result.First, 1, count, rows, count);
                else
                    run.RunRows(otherFirst, otherInner, otherOuter, tiled.First, tiled.InnerStride, tiled.OuterStride, result.First, 1, count, rows, count);

                string what = $"{op} {ta} x {tb} rows={rows} count={count} form={form} tiledOnLeft={tiledOnLeft} gap={gap} downRows={downRows} downColumns={downColumns} atEnd={atEnd}";
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < count; j++)
                {
                    object other = form == 3 ? otherTiled.Value(i, j) : otherRows.Value(i, j);
                    object x = tiledOnLeft ? tiled.Value(i, j) : other, y = tiledOnLeft ? other : tiled.Value(i, j);
                    bool want = Reference(op, x, ta, y, tb);
                    if (result.First[i * count + j] != (want ? 1 : 0))
                        Assert.Fail($"{what} [{i},{j}]: {x} {op} {y} should be {want}, slot holds {result.First[i * count + j]}");
                }
                result.AssertOnlySlotsWritten(what);
            }
            if (DirectILKernelGenerator.Enabled)
                (ComparisonRun.TileRuns - tilesBefore).Should().BeGreaterThan(position / 2, "most of these calls run tile by tile");
        }

        /// <summary>
        ///     <see cref="ComparisonRun.RunRows"/> on raw memory for a matrix against one row, the row on either side,
        ///     with the operands and the output against inaccessible pages: the repeated block is compared against
        ///     exactly the matrix's rows, and the last, shorter batch of rows reads and writes nothing beyond them.
        /// </summary>
        [TestMethod]
        public void RunRows_RepeatedRow_OnRawMemory_TouchesNoByteOutsideItsOperands()
        {
            var pairs = new (NPTypeCode, NPTypeCode)[]
            {
                (NPTypeCode.Byte, NPTypeCode.Byte), (NPTypeCode.Boolean, NPTypeCode.Boolean), (NPTypeCode.Int16, NPTypeCode.Int16),
                (NPTypeCode.Single, NPTypeCode.Single), (NPTypeCode.Double, NPTypeCode.Double), (NPTypeCode.Int64, NPTypeCode.UInt64),
                (NPTypeCode.Complex, NPTypeCode.Complex),
            };
            bool fenced = FencedMemory.IsSupported;
            using var matrixFence = fenced ? new FencedMemory(pages: 80) : null;
            using var rowFence = fenced ? new FencedMemory(pages: 2) : null;
            using var oFence = fenced ? new FencedMemory(pages: 8) : null;
            int position = 0;
            long before = ComparisonRun.RepeatedRowRuns, expectedRuns = 0;
            foreach (var (ta, tb) in pairs)
            foreach (long rows in new long[] { 15, 16, 17, 40, 257 })
            foreach (long count in new long[] { 1, 3, 16, 33, 40, 100, 130 })
            foreach (bool rowOnLeft in new[] { false, true })
            {
                position++;
                bool atEnd = (position & 1) != 0;
                var op = Ops[position % Ops.Length];
                NPTypeCode matrixType = rowOnLeft ? tb : ta, rowType = rowOnLeft ? ta : tb;
                int matrixSize = DirectILKernelGenerator.GetTypeSize(matrixType), rowSize = DirectILKernelGenerator.GetTypeSize(rowType);
                if (fenced && (rows * count * matrixSize > matrixFence.Bytes || rows * count > oFence.Bytes))
                    continue;
                using var matrix = new RawMatrix(matrixType, rowType, rows, count, inner: 1, backwardRows: false, phase: 2, matrixFence, atEnd);
                using var row = new RawRun(rowType, matrixType, count, 1, phase: 5, rowFence, !atEnd);
                using var result = new RawOutput(rows * count, 1, oFence, atEnd);
                var run = ComparisonRun.Get(op, ta, tb);
                if (rows >= 16 && count * System.Math.Max(matrixSize, rowSize) <= 2048)
                    expectedRuns++;
                if (rowOnLeft)
                    run.RunRows(row.First, rowSize, 0, matrix.First, matrixSize, matrix.OuterStride, result.First, 1, count, rows, count);
                else
                    run.RunRows(matrix.First, matrixSize, matrix.OuterStride, row.First, rowSize, 0, result.First, 1, count, rows, count);

                string what = $"{op} {ta} x {tb} rows={rows} count={count} rowOnLeft={rowOnLeft} atEnd={atEnd}";
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < count; j++)
                {
                    object x = rowOnLeft ? row.Value(j) : matrix.Value(i, j), y = rowOnLeft ? matrix.Value(i, j) : row.Value(j);
                    bool want = Reference(op, x, ta, y, tb);
                    if (result.First[i * count + j] != (want ? 1 : 0))
                        Assert.Fail($"{what} [{i},{j}]: {x} {op} {y} should be {want}, slot holds {result.First[i * count + j]}");
                }
                result.AssertOnlySlotsWritten(what);
            }
            if (DirectILKernelGenerator.Enabled)
                (ComparisonRun.RepeatedRowRuns - before).Should().Be(expectedRuns, "rows of at most 2,048 bytes, 16 of them or more, are repeated");
        }

        // =====================================================================================================
        //  Kernels
        // =====================================================================================================

        /// <summary>
        ///     The complex128 kernels agree with NumPy's scalar rule on a grid of parts that holds NaN, infinities and
        ///     signed zeros, for the three operand forms and every tail length, and write exactly their bytes.
        /// </summary>
        [TestMethod]
        public void ComplexKernels_MatchTheScalarRule_AndWriteExactlyTheirBytes()
        {
            double[] parts = { 0.0, -0.0, 1.0, -1.0, 2.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity };
            var grid = (from re in parts from im in parts select new Complex(re, im)).ToArray();
            var paths = new[] { ExecutionPath.SimdFull, ExecutionPath.SimdScalarRight, ExecutionPath.SimdScalarLeft };
            foreach (int n in Enumerable.Range(0, 20).Concat(new[] { 63, 64, 65, 200 }))
            {
                Complex* lhs = (Complex*)NativeMemory.Alloc((nuint)((n + 1) * sizeof(Complex)));
                Complex* rhs = (Complex*)NativeMemory.Alloc((nuint)((n + 1) * sizeof(Complex)));
                byte* res = (byte*)NativeMemory.Alloc((nuint)(n + 64));
                try
                {
                    for (int shift = 0; shift < grid.Length; shift += 7)
                    {
                        for (int i = 0; i <= n; i++)
                        {
                            lhs[i] = grid[(i * 5 + shift) % grid.Length];
                            rhs[i] = grid[(i * 11 + 3 * shift + 1) % grid.Length];
                        }
                        foreach (var path in paths)
                        foreach (var op in Ops)
                        {
                            new Span<byte>(res, n + 64).Fill(Sentinel);
                            var kernel = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(NPTypeCode.Complex, NPTypeCode.Complex, op, path));
                            kernel(lhs, rhs, (bool*)res, null, null, null, 1, n);
                            for (int i = 0; i < n; i++)
                            {
                                Complex a = path == ExecutionPath.SimdScalarLeft ? lhs[0] : lhs[i];
                                Complex b = path == ExecutionPath.SimdScalarRight ? rhs[0] : rhs[i];
                                bool want = Reference(op, a, NPTypeCode.Complex, b, NPTypeCode.Complex);
                                if (res[i] != (want ? 1 : 0))
                                    Assert.Fail($"complex {op} {path} n={n} [{i}]: {a} {op} {b} should be {want}, byte {res[i]}");
                            }
                            for (int k = n; k < n + 64; k++)
                                res[k].Should().Be(Sentinel, $"complex {op} {path} n={n} wrote result slot {k}");
                        }
                    }
                }
                finally
                {
                    NativeMemory.Free(lhs);
                    NativeMemory.Free(rhs);
                    NativeMemory.Free(res);
                }
            }
        }

        /// <summary>
        ///     The int64 / uint64 kernels compare exactly in both operand orders, for the three operand forms and
        ///     every tail length, and write exactly their bytes.
        /// </summary>
        [TestMethod]
        public void MixedSignKernels_CompareExactly_AndWriteExactlyTheirBytes()
        {
            long[] signedPool = { 0, 1, -1, long.MinValue, long.MaxValue, 5, -5, 9007199254740993L, long.MaxValue - 1 };
            ulong[] unsignedPool = { 0, 1, ulong.MaxValue, 9223372036854775807UL, 9223372036854775808UL, 5, 9007199254740993UL, 9223372036854775806UL };
            var paths = new[] { ExecutionPath.SimdFull, ExecutionPath.SimdScalarRight, ExecutionPath.SimdScalarLeft };
            foreach (int n in Enumerable.Range(0, 20).Concat(new[] { 63, 64, 65, 200 }))
            {
                long* s = (long*)NativeMemory.Alloc((nuint)((n + 1) * 8));
                ulong* u = (ulong*)NativeMemory.Alloc((nuint)((n + 1) * 8));
                byte* res = (byte*)NativeMemory.Alloc((nuint)(n + 64));
                try
                {
                    for (int shift = 0; shift < 9; shift++)
                    {
                        for (int i = 0; i <= n; i++)
                        {
                            s[i] = signedPool[(i * 5 + shift) % signedPool.Length];
                            u[i] = unsignedPool[(i * 3 + 2 * shift) % unsignedPool.Length];
                        }
                        foreach (bool signedFirst in new[] { true, false })
                        foreach (var path in paths)
                        foreach (var op in Ops)
                        {
                            new Span<byte>(res, n + 64).Fill(Sentinel);
                            NPTypeCode ta = signedFirst ? NPTypeCode.Int64 : NPTypeCode.UInt64, tb = signedFirst ? NPTypeCode.UInt64 : NPTypeCode.Int64;
                            var kernel = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(ta, tb, op, path));
                            if (signedFirst) kernel(s, u, (bool*)res, null, null, null, 1, n);
                            else kernel(u, s, (bool*)res, null, null, null, 1, n);
                            for (int i = 0; i < n; i++)
                            {
                                int li = path == ExecutionPath.SimdScalarLeft ? 0 : i, ri = path == ExecutionPath.SimdScalarRight ? 0 : i;
                                object a = signedFirst ? s[li] : u[li], b = signedFirst ? u[ri] : s[ri];
                                bool want = Reference(op, a, ta, b, tb);
                                if (res[i] != (want ? 1 : 0))
                                    Assert.Fail($"{ta} {op} {tb} {path} n={n} [{i}]: {a} {op} {b} should be {want}, byte {res[i]}");
                            }
                            for (int k = n; k < n + 64; k++)
                                res[k].Should().Be(Sentinel, $"{ta} {op} {tb} {path} n={n} wrote result slot {k}");
                        }
                    }
                }
                finally
                {
                    NativeMemory.Free(s);
                    NativeMemory.Free(u);
                    NativeMemory.Free(res);
                }
            }
        }

        /// <summary>
        ///     The strided move kernel copies every element width, at every count around its unroll step, for strides of
        ///     either sign and a repeated source element, and touches no other byte.
        /// </summary>
        [TestMethod]
        public void StridedMoveKernel_CopiesEveryWidthAndStride()
        {
            foreach (int size in new[] { 1, 2, 4, 8, 16 })
            {
                var move = DirectILKernelGenerator.GetStridedMoveKernel(size);
                move.Should().NotBeNull($"width {size}");
                foreach (int count in new[] { 0, 1, 2, 3, 4, 5, 7, 8, 9, 100 })
                foreach (long srcStride in new long[] { 1, 2, 3, -1, -2, 0 })
                foreach (long dstStride in new long[] { 1, 2, -1, -3 })
                {
                    long srcSpan = (System.Math.Abs(srcStride) * System.Math.Max(count - 1, 0) + 1) * size;
                    long dstSpan = (System.Math.Abs(dstStride) * System.Math.Max(count - 1, 0) + 1) * size;
                    byte* src = (byte*)NativeMemory.Alloc((nuint)(srcSpan + 64));
                    byte* dst = (byte*)NativeMemory.Alloc((nuint)(dstSpan + 128));
                    try
                    {
                        for (long k = 0; k < srcSpan + 64; k++)
                            src[k] = (byte)(k * 37 + 11);
                        new Span<byte>(dst, (int)(dstSpan + 128)).Fill(Sentinel);
                        byte* srcFirst = srcStride < 0 ? src + srcSpan - size : src;
                        byte* dstBase = dst + 64;
                        byte* dstFirst = dstStride < 0 ? dstBase + dstSpan - size : dstBase;
                        move(dstFirst, dstStride * size, srcFirst, srcStride * size, count);

                        var written = new bool[dstSpan + 128];
                        for (long i = 0; i < count; i++)
                        {
                            byte* d = dstFirst + i * dstStride * size, s = srcFirst + i * srcStride * size;
                            for (int k = 0; k < size; k++)
                            {
                                d[k].Should().Be(s[k], $"width {size} count {count} strides {srcStride}/{dstStride} element {i} byte {k}");
                                written[d + k - dst] = true;
                            }
                        }
                        for (long k = 0; k < dstSpan + 128; k++)
                        {
                            if (!written[k])
                                dst[k].Should().Be(Sentinel, $"width {size} count {count} strides {srcStride}/{dstStride}: byte {k - 64} outside the destination elements changed");
                        }
                    }
                    finally
                    {
                        NativeMemory.Free(src);
                        NativeMemory.Free(dst);
                    }
                }
            }
        }

        /// <summary>The bit helpers of the complex kernels, for every byte value, with and without BMI2.</summary>
        [TestMethod]
        public void BitHelpers_SpreadAndCompress()
        {
            for (uint bits = 0; bits < 256; bits++)
            {
                ulong expected = 0;
                for (int k = 0; k < 8; k++)
                    expected |= (ulong)((bits >> k) & 1) << (8 * k);
                DirectILKernelGenerator.SpreadBitsToBytesPortable(bits).Should().Be(expected, $"portable spread of {bits:X2}");
                DirectILKernelGenerator.SpreadBitsToBytes(bits).Should().Be(expected, $"spread of {bits:X2}");

                uint even = (bits & 1) | ((bits >> 1) & 2) | ((bits >> 2) & 4) | ((bits >> 3) & 8);
                DirectILKernelGenerator.CompressEvenBits(bits).Should().Be(even, $"even bits of {bits:X2}");
            }
        }

        // =====================================================================================================
        //  Raw buffers
        // =====================================================================================================

        /// <summary>Writes one boxed element of <paramref name="t"/> at <paramref name="p"/>.</summary>
        /// <param name="p">Destination.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="v">The boxed value (of the dtype's CLR type).</param>
        private static void WriteElement(byte* p, NPTypeCode t, object v)
        {
            switch (t)
            {
                case NPTypeCode.Boolean: *p = (bool)v ? (byte)1 : (byte)0; break;
                case NPTypeCode.SByte: *(sbyte*)p = (sbyte)v; break;
                case NPTypeCode.Byte: *p = (byte)v; break;
                case NPTypeCode.Int16: *(short*)p = (short)v; break;
                case NPTypeCode.UInt16: *(ushort*)p = (ushort)v; break;
                case NPTypeCode.Char: *(char*)p = (char)v; break;
                case NPTypeCode.Int32: *(int*)p = (int)v; break;
                case NPTypeCode.UInt32: *(uint*)p = (uint)v; break;
                case NPTypeCode.Int64: *(long*)p = (long)v; break;
                case NPTypeCode.UInt64: *(ulong*)p = (ulong)v; break;
                case NPTypeCode.Half: *(Half*)p = (Half)v; break;
                case NPTypeCode.Single: *(float*)p = (float)v; break;
                case NPTypeCode.Double: *(double*)p = (double)v; break;
                case NPTypeCode.Decimal: *(decimal*)p = (decimal)v; break;
                default: *(Complex*)p = (Complex)v; break;
            }
        }

        /// <summary>
        ///     One input run in native memory: <c>n</c> logical elements of a dtype at an element stride (0: one
        ///     value), with the values it holds kept for the reference. The memory is the run's own heap block, or a
        ///     place against a fence of a <see cref="FencedMemory"/>.
        /// </summary>
        private sealed class RawRun : IDisposable
        {
            /// <summary>The run's own heap block; null when the run lives in a fence.</summary>
            private readonly byte* _block;
            private readonly object[] _values;
            private readonly long _stride;

            /// <summary>Address of logical element 0.</summary>
            public readonly byte* First;

            /// <summary>Item size of the dtype.</summary>
            public readonly int ItemSize;

            /// <summary>Bytes from the run's lowest byte to its highest: the elements and the gaps between them.</summary>
            /// <param name="t">The dtype.</param>
            /// <param name="n">Logical element count.</param>
            /// <param name="stride">Element stride; 0 for one value.</param>
            /// <returns>The span in bytes.</returns>
            private static long Span(NPTypeCode t, long n, long stride)
                => (System.Math.Abs(stride) * (stride == 0 ? 0 : System.Math.Max(n - 1, 0)) + 1) * DirectILKernelGenerator.GetTypeSize(t);

            /// <summary>Whether a run of this form fits the accessible bytes of <paramref name="fence"/>.</summary>
            /// <param name="fence">The fenced memory.</param>
            /// <param name="t">The dtype.</param>
            /// <param name="n">Logical element count.</param>
            /// <param name="stride">Element stride; 0 for one value.</param>
            /// <returns>True when it fits.</returns>
            public static bool Fits(FencedMemory fence, NPTypeCode t, long n, long stride) => Span(t, n, stride) <= fence.Bytes;

            /// <summary>Allocates (or places) and fills the run.</summary>
            /// <param name="t">The dtype.</param>
            /// <param name="other">The other operand's dtype (selects the value pool).</param>
            /// <param name="n">Logical element count.</param>
            /// <param name="stride">Element stride; 0 for one value.</param>
            /// <param name="phase">Selects the sequence of pool entries.</param>
            /// <param name="fence">Fenced memory to place the run in; null for a heap block of its own.</param>
            /// <param name="atEnd">In a fence: the run's highest byte is the last accessible one (else its lowest
            ///     byte is the first accessible one).</param>
            public RawRun(NPTypeCode t, NPTypeCode other, long n, long stride, int phase, FencedMemory fence = null, bool atEnd = false)
            {
                ItemSize = DirectILKernelGenerator.GetTypeSize(t);
                _stride = stride;
                long span = Span(t, n, stride);
                byte* low;
                if (fence is null)
                    low = _block = (byte*)NativeMemory.AllocZeroed((nuint)(span + 64));
                else
                    low = fence.Place(span, atEnd);
                // A backward run starts at its highest element.
                First = stride < 0 ? low + span - ItemSize : low;
                var pool = PoolFor(t, other);
                _values = new object[stride == 0 ? 1 : n];
                for (long i = 0; i < _values.Length; i++)
                {
                    _values[i] = pool[PoolIndex(i, phase, pool.Length)];
                    WriteElement(First + i * stride * ItemSize, t, _values[i]);
                }
            }

            /// <summary>The value of logical element <paramref name="i"/>.</summary>
            /// <param name="i">The index.</param>
            /// <returns>The boxed value.</returns>
            public object Value(long i) => _values[_stride == 0 ? 0 : i];

            /// <summary>Frees the run's own block (a fenced run owns no memory).</summary>
            public void Dispose() => NativeMemory.Free(_block);
        }

        /// <summary>
        ///     One input operand of a rows-of-runs iteration in native memory: <c>rows</c> runs of <c>count</c>
        ///     elements at an inner element stride (0: one value per row), the rows laid out forwards or backwards. The
        ///     memory is the operand's own heap block, or a place against a fence of a <see cref="FencedMemory"/>.
        /// </summary>
        private sealed class RawMatrix : IDisposable
        {
            /// <summary>The operand's own heap block; null when it lives in a fence.</summary>
            private readonly byte* _block;
            private readonly object[,] _values;
            private readonly long _inner;

            /// <summary>Address of the first element of row 0.</summary>
            public readonly byte* First;

            /// <summary>Byte stride from the first element of one row to that of the next.</summary>
            public readonly long OuterStride;

            /// <summary>Allocates (or places) and fills the operand.</summary>
            /// <param name="t">The dtype.</param>
            /// <param name="other">The other operand's dtype (selects the value pool).</param>
            /// <param name="rows">Number of rows.</param>
            /// <param name="count">Elements per row.</param>
            /// <param name="inner">Inner element stride (0: one value per row).</param>
            /// <param name="backwardRows">Lay the rows out from the highest address down.</param>
            /// <param name="phase">Selects the sequence of pool entries.</param>
            /// <param name="fence">Fenced memory to place the operand in; null for a heap block of its own.</param>
            /// <param name="atEnd">In a fence: the operand's highest byte is the last accessible one (else its lowest
            ///     byte is the first accessible one).</param>
            public RawMatrix(NPTypeCode t, NPTypeCode other, long rows, long count, long inner, bool backwardRows, int phase,
                FencedMemory fence = null, bool atEnd = false)
            {
                int size = DirectILKernelGenerator.GetTypeSize(t);
                _inner = inner;
                long rowSpan = (System.Math.Abs(inner) * (inner == 0 ? 0 : count - 1) + 1) * size;      // bytes one row occupies
                byte* low;
                if (fence is null)
                    low = _block = (byte*)NativeMemory.AllocZeroed((nuint)(rowSpan * rows + 64));
                else
                    low = fence.Place(rowSpan * rows, atEnd);
                OuterStride = backwardRows ? -rowSpan : rowSpan;
                byte* rowZero = backwardRows ? low + rowSpan * (rows - 1) : low;
                First = inner < 0 ? rowZero + rowSpan - size : rowZero;
                var pool = PoolFor(t, other);
                _values = new object[rows, inner == 0 ? 1 : count];
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < _values.GetLength(1); j++)
                {
                    _values[i, j] = pool[PoolIndex(i * count + j, phase, pool.Length)];
                    WriteElement(First + i * OuterStride + j * inner * size, t, _values[i, j]);
                }
            }

            /// <summary>The value at row <paramref name="i"/>, position <paramref name="j"/>.</summary>
            /// <param name="i">Row.</param>
            /// <param name="j">Position in the row.</param>
            /// <returns>The boxed value.</returns>
            public object Value(long i, long j) => _values[i, _inner == 0 ? 0 : j];

            /// <summary>Frees the operand's own block (a fenced operand owns no memory).</summary>
            public void Dispose() => NativeMemory.Free(_block);
        }

        /// <summary>
        ///     One input operand of a rows-of-runs iteration stored COLUMN by column in native memory (the layout of an
        ///     F-ordered matrix): the <c>rows</c> elements of one column are adjacent, the columns are
        ///     <c>rows + gap</c> elements apart. The rows and the columns can each run downwards in memory. The memory
        ///     is the operand's own heap block, or a place against a fence of a <see cref="FencedMemory"/>.
        /// </summary>
        private sealed class RawTiled : IDisposable
        {
            /// <summary>The operand's own heap block; null when it lives in a fence.</summary>
            private readonly byte* _block;
            private readonly object[,] _values;

            /// <summary>Address of element (row 0, position 0).</summary>
            public readonly byte* First;

            /// <summary>Byte stride between two elements of one row: one column, of either sign.</summary>
            public readonly long InnerStride;

            /// <summary>Byte stride between two rows: one element, of either sign.</summary>
            public readonly long OuterStride;

            /// <summary>Bytes from the operand's lowest byte to its highest.</summary>
            /// <param name="t">The dtype.</param>
            /// <param name="rows">Number of rows.</param>
            /// <param name="count">Elements per row.</param>
            /// <param name="gap">Unused elements after each column piece.</param>
            /// <returns>The span in bytes.</returns>
            private static long Span(NPTypeCode t, long rows, long count, long gap)
                => ((count - 1) * (rows + gap) + rows) * DirectILKernelGenerator.GetTypeSize(t);

            /// <summary>Whether an operand of this form fits the accessible bytes of <paramref name="fence"/>.</summary>
            /// <param name="fence">The fenced memory.</param>
            /// <param name="t">The dtype.</param>
            /// <param name="rows">Number of rows.</param>
            /// <param name="count">Elements per row.</param>
            /// <param name="gap">Unused elements after each column piece.</param>
            /// <returns>True when it fits.</returns>
            public static bool Fits(FencedMemory fence, NPTypeCode t, long rows, long count, long gap) => Span(t, rows, count, gap) <= fence.Bytes;

            /// <summary>Allocates (or places) and fills the operand.</summary>
            /// <param name="t">The dtype.</param>
            /// <param name="other">The other operand's dtype (selects the value pool).</param>
            /// <param name="rows">Number of rows.</param>
            /// <param name="count">Elements per row.</param>
            /// <param name="gap">Unused elements after each column piece (0: the columns follow each other).</param>
            /// <param name="downRows">Row 0 is the highest element of each column piece.</param>
            /// <param name="downColumns">Position 0 is the column at the highest address.</param>
            /// <param name="phase">Selects the sequence of pool entries.</param>
            /// <param name="fence">Fenced memory to place the operand in; null for a heap block of its own.</param>
            /// <param name="atEnd">In a fence: the operand's highest byte is the last accessible one (else its lowest
            ///     byte is the first accessible one).</param>
            public RawTiled(NPTypeCode t, NPTypeCode other, long rows, long count, long gap, bool downRows, bool downColumns, int phase,
                FencedMemory fence = null, bool atEnd = false)
            {
                int size = DirectILKernelGenerator.GetTypeSize(t);
                long span = Span(t, rows, count, gap), column = (rows + gap) * size;
                byte* low;
                if (fence is null)
                    low = _block = (byte*)NativeMemory.AllocZeroed((nuint)(span + 64));
                else
                    low = fence.Place(span, atEnd);
                InnerStride = downColumns ? -column : column;
                OuterStride = downRows ? -size : size;
                First = low + (downColumns ? (count - 1) * column : 0) + (downRows ? (rows - 1) * size : 0);
                var pool = PoolFor(t, other);
                _values = new object[rows, count];
                for (long i = 0; i < rows; i++)
                for (long j = 0; j < count; j++)
                {
                    _values[i, j] = pool[PoolIndex(i * count + j, phase, pool.Length)];
                    WriteElement(First + i * OuterStride + j * InnerStride, t, _values[i, j]);
                }
            }

            /// <summary>The value at row <paramref name="i"/>, position <paramref name="j"/>.</summary>
            /// <param name="i">Row.</param>
            /// <param name="j">Position in the row.</param>
            /// <returns>The boxed value.</returns>
            public object Value(long i, long j) => _values[i, j];

            /// <summary>Frees the operand's own block (a fenced operand owns no memory).</summary>
            public void Dispose() => NativeMemory.Free(_block);
        }

        /// <summary>
        ///     A bool output in native memory with a byte stride. Every byte around and between its slots is preset to
        ///     the sentinel, so that a store outside the slots is seen. The memory is the output's own heap block with
        ///     64-byte margins, or a place against a fence of a <see cref="FencedMemory"/> (the whole fenced area is
        ///     then preset and checked).
        /// </summary>
        private sealed class RawOutput : IDisposable
        {
            /// <summary>The output's own heap block; null when it lives in a fence.</summary>
            private readonly byte* _block;

            /// <summary>The preset area that <see cref="AssertOnlySlotsWritten"/> scans.</summary>
            private readonly byte* _area;

            /// <summary>Bytes of <see cref="_area"/>.</summary>
            private readonly long _areaBytes;

            /// <summary>Lowest slot address.</summary>
            private readonly byte* _low;

            private readonly long _span, _stride;

            /// <summary>Address of output slot 0.</summary>
            public readonly byte* First;

            /// <summary>Bytes from the lowest slot to the highest.</summary>
            /// <param name="n">Slot count.</param>
            /// <param name="stride">Byte stride between slots.</param>
            /// <returns>The span in bytes.</returns>
            private static long Span(long n, long stride) => System.Math.Abs(stride) * System.Math.Max(n - 1, 0) + 1;

            /// <summary>Whether an output of this form fits the accessible bytes of <paramref name="fence"/>.</summary>
            /// <param name="fence">The fenced memory.</param>
            /// <param name="n">Slot count.</param>
            /// <param name="stride">Byte stride between slots.</param>
            /// <returns>True when it fits.</returns>
            public static bool Fits(FencedMemory fence, long n, long stride) => Span(n, stride) <= fence.Bytes;

            /// <summary>Allocates (or places) the output and presets the sentinel.</summary>
            /// <param name="n">Slot count.</param>
            /// <param name="stride">Byte stride between slots (not 0).</param>
            /// <param name="fence">Fenced memory to place the output in; null for a heap block of its own.</param>
            /// <param name="atEnd">In a fence: the highest slot is the last accessible byte (else the lowest slot is the
            ///     first accessible byte).</param>
            public RawOutput(long n, long stride, FencedMemory fence = null, bool atEnd = false)
            {
                _stride = stride;
                _span = Span(n, stride);
                if (fence is null)
                {
                    _areaBytes = _span + 128;
                    _area = _block = (byte*)NativeMemory.Alloc((nuint)_areaBytes);
                    _low = _area + 64;
                }
                else
                {
                    _area = fence.Start;
                    _areaBytes = fence.Bytes;
                    _low = fence.Place(_span, atEnd);
                }
                new Span<byte>(_area, checked((int)_areaBytes)).Fill(Sentinel);
                First = stride < 0 ? _low + _span - 1 : _low;
            }

            /// <summary>Asserts that every byte of the preset area that is not a slot still holds the sentinel.</summary>
            /// <param name="what">Label for the failure message.</param>
            public void AssertOnlySlotsWritten(string what)
            {
                long step = System.Math.Abs(_stride);
                long lowOffset = _low - _area;
                // Below the slots, between them, above them: three scans, so the common case is two vector searches.
                var below = new ReadOnlySpan<byte>(_area, checked((int)lowOffset));
                int bad = below.IndexOfAnyExcept(Sentinel);
                if (bad >= 0)
                    Assert.Fail($"{what}: byte {bad - lowOffset} outside the output slots was written ({below[bad]})");
                long aboveOffset = lowOffset + _span;
                var above = new ReadOnlySpan<byte>(_area + aboveOffset, checked((int)(_areaBytes - aboveOffset)));
                bad = above.IndexOfAnyExcept(Sentinel);
                if (bad >= 0)
                    Assert.Fail($"{what}: byte {_span + bad} outside the output slots was written ({above[bad]})");
                if (step > 1)
                {
                    for (long k = 0; k < _span; k++)
                    {
                        if (k % step != 0 && _low[k] != Sentinel)
                            Assert.Fail($"{what}: byte {k} between the output slots was written ({_low[k]})");
                    }
                }
            }

            /// <summary>Frees the output's own block (a fenced output owns no memory).</summary>
            public void Dispose() => NativeMemory.Free(_block);
        }
    }
}
