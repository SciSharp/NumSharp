using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.sum / np.mean over an operand with a REVERSED or a BROADCAST axis (a stride &lt;= 0 on an axis longer than 1):
    ///     NumPy reduces it through its reduction iterator, which keeps each axis in its logical order
    ///     (<c>NPY_ITER_DONT_NEGATE_STRIDES</c>) and lets a stride-0 operand abstain from the axis sort, so a reversed axis is
    ///     added from its first logical element and a broadcast row is one call of the add loop. NumSharp's axis kernels walk
    ///     memory order and its flat engine declined a broadcast operand; since the np-layer dtype plan's P2 (the np.average
    ///     family, D4) <c>DefaultEngine.ReduceAdd</c> runs the port of that iterator with NumPy's add loops
    ///     (<c>ReduceAddThroughIterator</c>, <c>NumPyAddLoops</c>) for these layouts in an inexact loop.
    /// </summary>
    /// <remarks>
    ///     The first four tests were the <c>[OpenBugs]</c> groups 10 to 13 of <c>OpenBugs.Reductions.cs</c> (plan
    ///     <c>docs/plans/api-parameter-parity.md</c> finding 5, finding 10, X12 and X14), moved here unchanged when they
    ///     passed. Every expected string is NumPy 2.4.2's output for the same call, in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format.
    /// </remarks>
    [TestClass]
    public class np_sum_ReversedBroadcast_Tests : TestClass
    {
        /// <summary>
        ///     <c>np.sum(b[::-1], axis=0)</c> of a float32 <c>(4, 3)</c> array whose first two columns cancel
        ///     (<c>±1e8</c> with <c>1.0</c>s) is <c>[0, 1, 8]</c> in NumPy: it adds the rows in the view's logical order
        ///     (row 3 first). NumSharp added them in memory order (row 0 first) and gave <c>[1, 0, 8]</c>; <c>np.mean</c>
        ///     inherited it through its sum (<c>[0.25, 0, 2]</c> for NumPy's <c>[0, 0.25, 2]</c>).
        /// </summary>
        /// <remarks>Was <c>[OpenBugs]</c> group 10 (plan api-parameter-parity finding 5).</remarks>
        [TestMethod]
        public void Sum_ReversedViewAxis_WalksTheLogicalOrder()
        {
            var fails = new List<string>();
            var b = np.array(new float[,] { { 1e8f, 1f, 2f }, { 1f, 1e8f, 2f }, { -1e8f, 3f, 2f }, { 1f, -1e8f, 2f } });
            var v = b["::-1"];
            Check(fails, "sum f32 b[::-1] axis0", () => np.sum(v, 0), "float32 (3,) 00000000 3f800000 41000000");
            Check(fails, "mean f32 b[::-1] axis0", () => np.mean(v, 0), "float32 (3,) 00000000 3e800000 40000000");
            AssertNoMismatch(fails, "NumPy adds the rows of a reversed view in its logical order");
        }

        /// <summary>
        ///     <c>np.sum(np.broadcast_to(r, (4, 5)))</c> of five float32 values is <c>0x40d3f48a</c> in NumPy (four calls
        ///     of the add loop, one per broadcast row, each adding the row's pairwise sum to the result); NumSharp's
        ///     elementwise fold gave <c>0x40d3f488</c>. A float64 scalar broadcast to <c>(3, 4)</c> was one ULP off the same
        ///     way, and <c>np.mean</c> inherited both through its sum.
        /// </summary>
        /// <remarks>Was <c>[OpenBugs]</c> group 11 (plan api-parameter-parity finding 10). np.mean's flat route now sends a
        ///     broadcast float or complex operand to <c>_mean</c>'s statements, whose sum is np.sum's.</remarks>
        [TestMethod]
        public void Sum_BroadcastOperand_Flat_IsNumPysCallSchedule()
        {
            var fails = new List<string>();
            var b = np.broadcast_to(np.array(new[] { -0.8627f, 1.9222f, 0.373f, -1.1762f, 1.3996f }), new Shape(4, 5));
            Check(fails, "sum f32 broadcast (4,5)", () => np.sum(b), "float32 () 40d3f48a");
            Check(fails, "mean f32 broadcast (4,5)", () => np.mean(b), "float32 () 3ea9906e");
            var s = np.broadcast_to(np.array(new[] { -0.8627 }), new Shape(3, 4));
            Check(fails, "sum f64 scalar broadcast (3,4)", () => np.sum(s), "float64 () c024b46dc5d63887");
            Check(fails, "mean f64 scalar broadcast (3,4)", () => np.mean(s), "float64 () bfeb9b3d07c84b5f");
            AssertNoMismatch(fails, "NumPy adds a broadcast operand one row (one add-loop call) at a time");
        }

        /// <summary>
        ///     <c>np.sum</c> along an axis of a broadcast operand adds in NumPy's order. A float32 column of nine values
        ///     broadcast to <c>(9, 3)</c> and summed over axis 0 is <c>0x40466664</c> in every output (NumPy adds the rows
        ///     sequentially; NumSharp gave <c>0x40466668</c>). A float16 column broadcast to <c>(3, 7)</c> and summed along
        ///     the stride-0 axis 1 rounds once (<c>0xcce7</c>; NumSharp rounded each addition, <c>0xcce6</c>). An int32 row
        ///     broadcast to <c>(4, 9)</c> and summed over axis 1 in a float32 loop is <c>0xce2eb2aa</c> (NumSharp gave
        ///     <c>0xce2eb2a8</c>).
        /// </summary>
        /// <remarks>Was <c>[OpenBugs]</c> group 12 (plan api-parameter-parity X12, found by the A6 work: 510 of 5,382 probe
        ///     cases differed along the stride-0 axis and along the others, in same-dtype and in cast loops).</remarks>
        [TestMethod]
        public void Sum_BroadcastOperand_Axis_IsNumPysOrder()
        {
            var fails = new List<string>();
            var col = np.broadcast_to(np.array(new[] { -4.6f, 3.6f, 5.2f, -1.6f, 4.7f, 3.5f, -4f, 3.3f, -7f }).reshape(9, 1), new Shape(9, 3));
            Check(fails, "sum f32 (9,1)->(9,3) axis0", () => np.sum(col, 0), "float32 (3,) 40466664 40466664 40466664");
            var h = np.broadcast_to(np.array(new[] { (Half)1.5, (Half)5.19921875, (Half)(-2.80078125) }).reshape(3, 1), new Shape(3, 7));
            Check(fails, "sum f16 (3,1)->(3,7) axis1", () => np.sum(h, 1), "float16 (3,) 4940 508d cce7");
            var row = np.broadcast_to(np.array(new[] { 257711264, 906851054, -438111839, -971706796, 1928501897, 1173551007, -1835420885, 201676561, -1955788270 })
                                        .reshape(1, 9), new Shape(4, 9));
            Check(fails, "sum i32 (1,9)->(4,9) axis1 dtype=f32", () => np.sum(row, 1, false, np.float32),
                  "float32 (4,) ce2eb2aa ce2eb2aa ce2eb2aa ce2eb2aa");
            AssertNoMismatch(fails, "NumPy adds a broadcast operand along an axis in its iterator's call schedule");
        }

        /// <summary>
        ///     <c>TensorEngine.Mean</c> of a broadcast float16 operand with <c>dtype=float32</c> is float32 (<c>1.05</c>,
        ///     <c>0x3f866666</c>: NumPy's <c>np.mean(b, dtype=np.float32)</c>; NumSharp returned the float16 <c>0x3c33</c>).
        ///     Of a broadcast complex128 operand with <c>dtype=float64</c> it is the float64 mean of the real parts
        ///     (<c>0.5</c>; NumSharp returned the complex mean <c>0.5+1.5j</c>).
        /// </summary>
        /// <remarks>Was <c>[OpenBugs]</c> group 13 (plan api-parameter-parity X14): the flat route's elementwise fold kept
        ///     the input dtype for these two; a broadcast float or complex operand now takes <c>_mean</c>'s statements.</remarks>
        [TestMethod]
        public void Mean_BroadcastOperand_Flat_HonorsDtype()
        {
            var fails = new List<string>();
            var h = np.broadcast_to(np.array(new[] { (Half)1.5, (Half)2.25, (Half)(-3.0), (Half)0.5, (Half)4.0 }), new Shape(4, 5));
            Check(fails, "mean f16 broadcast dtype=f32", () => h.TensorEngine.Mean(h, null, np.float32, false), "float32 () 3f866666");
            var c = np.broadcast_to(np.array(new[] { new Complex(1.5, 2), new Complex(-0.5, 1) }), new Shape(3, 2));
            Check(fails, "mean c128 broadcast dtype=f64", () => c.TensorEngine.Mean(c, null, np.float64, false), "float64 () 3fe0000000000000");
            AssertNoMismatch(fails, "NumPy's mean sums and returns in the dtype= loop");
        }

        /// <summary>A float64 quiet NaN with the given payload.</summary>
        /// <param name="payload">The low mantissa bits.</param>
        /// <returns>The NaN.</returns>
        private static double Nan(long payload) => BitConverter.Int64BitsToDouble(0x7FF8000000000000 | payload);

        /// <summary>A float32 quiet NaN with the given payload.</summary>
        /// <param name="payload">The low mantissa bits.</param>
        /// <returns>The NaN.</returns>
        private static float NanF(int payload) => BitConverter.Int32BitsToSingle(0x7FC00000 | payload);

        /// <summary>The expected rendering of <paramref name="n"/> float64 NaNs, the first <paramref name="body"/> with
        ///     payload <paramref name="bodyPayload"/> and the rest with <paramref name="tailPayload"/>.</summary>
        /// <param name="n">The element count.</param>
        /// <param name="body">How many leading elements carry <paramref name="bodyPayload"/>.</param>
        /// <param name="bodyPayload">The body's payload.</param>
        /// <param name="tailPayload">The tail's payload.</param>
        /// <returns>The <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/> string.</returns>
        private static string F64Regions(int n, int body, long bodyPayload, long tailPayload)
        {
            var sb = new System.Text.StringBuilder($"float64 ({n},)");
            for (int i = 0; i < n; i++)
                sb.Append(' ').Append((0x7FF8000000000000 | (i < body ? bodyPayload : tailPayload)).ToString("x16"));
            return sb.ToString();
        }

        /// <summary>The float32 form of <see cref="F64Regions"/>.</summary>
        /// <param name="n">The element count.</param>
        /// <param name="body">How many leading elements carry <paramref name="bodyPayload"/>.</param>
        /// <param name="bodyPayload">The body's payload.</param>
        /// <param name="tailPayload">The tail's payload.</param>
        /// <returns>The <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/> string.</returns>
        private static string F32Regions(int n, int body, int bodyPayload, int tailPayload)
        {
            var sb = new System.Text.StringBuilder($"float32 ({n},)");
            for (int i = 0; i < n; i++)
                sb.Append(' ').Append((0x7FC00000 | (i < body ? bodyPayload : tailPayload)).ToString("x8"));
            return sb.ToString();
        }

        /// <summary>
        ///     A column of NaNs with different payloads, broadcast to <c>(3, m)</c> and summed over axis 0: NumPy's add loop
        ///     runs <c>out = out + row</c> with the row a broadcast value, its scalar-broadcast loop. When both operands are
        ///     NaN that loop keeps the BROADCAST value's NaN in its vector body (the first <c>m - m % 8</c> float64 lanes, or
        ///     <c>m - m % 16</c> float32 lanes) and the ARRAY operand's (the running sum's) in the tail, for a call longer
        ///     than two vectors; a shorter call keeps the second operand's (the row's). NumPy 2.4.2's bits.
        /// </summary>
        [TestMethod]
        public void Sum_BroadcastColumn_AlongTheBroadcastAxis_KeepsNumPysNaNPerRegion()
        {
            var fails = new List<string>();
            var col = np.array(new[,] { { Nan(1) }, { Nan(2) }, { 3.0 } });
            Check(fails, "f64 m=20", () => np.sum(np.broadcast_to(col, new Shape(3, 20)), 0), F64Regions(20, 16, 2, 1), rawNaN: true);
            Check(fails, "f64 m=12", () => np.sum(np.broadcast_to(col, new Shape(3, 12)), 0), F64Regions(12, 8, 2, 1), rawNaN: true);
            Check(fails, "f64 m=8 (a short call)", () => np.sum(np.broadcast_to(col, new Shape(3, 8)), 0), F64Regions(8, 8, 2, 2), rawNaN: true);
            var colf = np.array(new[,] { { NanF(1) }, { NanF(2) }, { 3.0f } });
            Check(fails, "f32 m=40", () => np.sum(np.broadcast_to(colf, new Shape(3, 40)), 0), F32Regions(40, 32, 2, 1), rawNaN: true);
            Check(fails, "f32 m=16 (a short call)", () => np.sum(np.broadcast_to(colf, new Shape(3, 16)), 0), F32Regions(16, 16, 2, 2), rawNaN: true);
            AssertNoMismatch(fails, "NumPy's scalar-broadcast add keeps the broadcast NaN in its body and the array's in its tail");
        }

        /// <summary>
        ///     Reversed views summed over axis 0, NaNs in both operands: <c>out = out + row</c> of two arrays keeps the
        ///     SECOND operand's (the row's) NaN, whether the reversed axis is the reduced one (<c>a[::-1]</c>) or the kept
        ///     innermost one (<c>a[:, ::-1]</c>, NumSharp's lane-reversing vector loop). NumPy 2.4.2's bits.
        /// </summary>
        [TestMethod]
        public void Sum_ReversedViews_AlongAxis0_KeepTheSecondOperandsNaN()
        {
            var fails = new List<string>();
            var a = np.array(new[,] { { Nan(5), Nan(6), 1.0, 2.0, Nan(7) }, { Nan(8), 3.0, Nan(9), 4.0, Nan(10) } });
            Check(fails, "a[::-1] axis0", () => np.sum(a["::-1"], 0),
                  "float64 (5,) 7ff8000000000005 7ff8000000000006 7ff8000000000009 4018000000000000 7ff8000000000007", rawNaN: true);
            Check(fails, "a[:, ::-1] axis0", () => np.sum(a[":, ::-1"], 0),
                  "float64 (5,) 7ff800000000000a 4018000000000000 7ff8000000000009 7ff8000000000006 7ff8000000000008", rawNaN: true);
            AssertNoMismatch(fails, "NumPy's add of two arrays keeps the second operand's NaN");
        }

        /// <summary>The generator's value pool for float64: magnitudes that cancel and small values they absorb, so the
        ///     order of the additions shows in the result (every value is exact in float64).</summary>
        private static readonly double[] F64Pool = { 1e16, 1.0, -1e16, 3.0, 0.5, -2.0, 7.25 };

        /// <summary>The float32 pool (<see cref="F64Pool"/>'s idea at float32's precision).</summary>
        private static readonly float[] F32Pool = { 1e8f, 1f, -1e8f, 3f, 0.5f, -2f, 7.25f };

        /// <summary>The float16 pool (<see cref="F64Pool"/>'s idea at float16's precision: 2048 + 1 is 2048).</summary>
        private static readonly double[] F16Pool = { 2048.0, 1.0, -2048.0, 3.0, 0.5, -2.0, 7.25 };

        /// <summary>A <paramref name="rows"/> x <paramref name="cols"/> grid with <c>pool[(i * 5 + j * 3) % pool.Length]</c> at
        ///     <c>(i, j)</c>, the test generator's layout (each column runs through the whole pool).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="rows">The row count.</param>
        /// <param name="cols">The column count.</param>
        /// <param name="pool">The values.</param>
        /// <returns>The grid.</returns>
        private static T[,] Grid<T>(int rows, int cols, T[] pool)
        {
            var g = new T[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    g[i, j] = pool[(i * 5 + j * 3) % pool.Length];
            return g;
        }

        /// <summary>A float16 grid from <see cref="F16Pool"/> (every pool value is exact in float16).</summary>
        /// <param name="rows">The row count.</param>
        /// <param name="cols">The column count.</param>
        /// <returns>The grid.</returns>
        private static Half[,] GridF16(int rows, int cols)
        {
            var g = new Half[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    g[i, j] = (Half)F16Pool[(i * 5 + j * 3) % F16Pool.Length];
            return g;
        }

        /// <summary>A float16 from its raw bits (a NaN payload or a subnormal pattern, which must not pass a conversion).</summary>
        /// <param name="bits">The bit pattern.</param>
        /// <returns>The value.</returns>
        private static Half H(ushort bits) => BitConverter.UInt16BitsToHalf(bits);

        /// <summary>
        ///     The several-rows elementwise form (<c>NumPyAddLoops.DoubleBinaryRows</c> / <c>SingleBinaryRows</c>): a
        ///     reversed view summed over axis 0 runs the reduce double loop over rows that all fold into one out row, which
        ///     NumSharp takes in one call, blocks of lanes kept in registers across the rows. Each lane must still be NumPy's
        ///     sequence of per-row calls: the rows in the view's logical order (25 of the 37 float64 columns differ from the
        ///     memory-order sum), the second operand's NaN (the LAST NaN of a column wins), an <c>inf + -inf</c> default NaN
        ///     that a later NaN replaces or that stays, <c>+0.0</c> from a column of <c>-0.0</c> (the identity start), and an
        ///     overflow to <c>inf</c>; at every width the loop has (float64: two 16-lane blocks, a 4-lane block, a scalar
        ///     lane; float32: two 32-lane blocks, an 8-lane block, five scalar lanes). A broadcast row (each reduced row the
        ///     same memory) takes the same form with a row stride of 0. NumPy 2.4.2's bits.
        /// </summary>
        /// <remarks>The same values summed over axis 0 of the plain C-ordered array take NumSharp's own axis kernel, whose
        ///     slab fold keeps the FIRST NaN of a column (NumPy: the last): that is <c>[OpenBugs]</c> group 15 of
        ///     <c>OpenBugs.Reductions.cs</c>, not this route.</remarks>
        [TestMethod]
        public void Sum_RowsForm_ReversedAndBroadcastRows_AreNumPysPerRowCalls()
        {
            var fails = new List<string>();
            var g = Grid(6, 37, F64Pool);
            foreach (var (i, j, p) in new[] { (1, 0, 11L), (4, 0, 14L), (0, 5, 20L), (3, 17, 31L), (2, 33, 42L), (5, 33, 45L), (4, 36, 50L), (1, 36, 51L) })
                g[i, j] = Nan(p);
            g[5, 12] = double.PositiveInfinity; g[3, 12] = double.NegativeInfinity; g[1, 12] = Nan(70);
            g[5, 13] = double.PositiveInfinity; g[2, 13] = double.NegativeInfinity;
            for (int i = 0; i < 6; i++)
                g[i, 14] = -0.0;
            g[0, 15] = 1e308; g[1, 15] = 1e308;
            var a = np.array(g);
            Check(fails, "f64 a[::-1] axis0", () => np.sum(a["::-1"], 0),
                  "float64 (37,) 7ff800000000000b 4027800000000000 401f000000000000 4024000000000000 c341c37937e07ffb 7ff8000000000014 4004000000000000 4341c37937e08005" +
                  " 4027800000000000 401f000000000000 4024000000000000 c341c37937e07ffb 7ff8000000000046 fff8000000000000 0000000000000000 7ff0000000000000" +
                  " 401f000000000000 7ff800000000001f c341c37937e07ffb 401b000000000000 4004000000000000 4341c37937e08005 4027800000000000 401f000000000000" +
                  " 4024000000000000 c341c37937e07ffb 401b000000000000 4004000000000000 4341c37937e08005 4027800000000000 401f000000000000 4024000000000000" +
                  " c341c37937e07ffb 7ff800000000002a 4004000000000000 4341c37937e08005 7ff8000000000033", rawNaN: true);

            var gf = Grid(5, 77, F32Pool);
            foreach (var (i, j, p) in new[] { (1, 0, 11), (4, 0, 14), (0, 20, 20), (3, 40, 31), (2, 66, 42), (4, 66, 45), (3, 75, 50), (1, 75, 51) })
                gf[i, j] = NanF(p);
            gf[4, 12] = float.PositiveInfinity; gf[2, 12] = float.NegativeInfinity; gf[0, 12] = NanF(70);
            for (int i = 0; i < 5; i++)
                gf[i, 14] = -0.0f;
            var f = np.array(gf);
            Check(fails, "f32 f[::-1] axis0", () => np.sum(f["::-1"], 0),
                  "float32 (77,) 7fc0000b ccbebc1f 40f80000 00000000 411c0000 410c0000 3f000000 4cbebc21" +
                  " ccbebc1f 40f80000 00000000 411c0000 7fc00046 3f000000 00000000 ccbebc1f" +
                  " 40f80000 00000000 411c0000 410c0000 7fc00014 4cbebc21 ccbebc1f 40f80000" +
                  " 00000000 411c0000 410c0000 3f000000 4cbebc21 ccbebc1f 40f80000 00000000" +
                  " 411c0000 410c0000 3f000000 4cbebc21 ccbebc1f 40f80000 00000000 411c0000" +
                  " 7fc0001f 3f000000 4cbebc21 ccbebc1f 40f80000 00000000 411c0000 410c0000" +
                  " 3f000000 4cbebc21 ccbebc1f 40f80000 00000000 411c0000 410c0000 3f000000" +
                  " 4cbebc21 ccbebc1f 40f80000 00000000 411c0000 410c0000 3f000000 4cbebc21" +
                  " ccbebc1f 40f80000 7fc0002a 411c0000 410c0000 3f000000 4cbebc21 ccbebc1f" +
                  " 40f80000 00000000 411c0000 7fc00033 3f000000", rawNaN: true);

            var r = Grid(1, 37, F64Pool);
            r[0, 3] = Nan(5); r[0, 33] = Nan(7); r[0, 36] = Nan(6); r[0, 10] = double.PositiveInfinity; r[0, 11] = -0.0;
            var row = np.array(r);
            Check(fails, "f64 broadcast row (1,37)->(7,37) axis0", () => np.sum(np.broadcast_to(row, new Shape(7, 37)), 0),
                  "float64 (37,) 436f161421c8e000 4035000000000000 4049600000000000 7ff8000000000005 c02c000000000000 401c000000000000 400c000000000000 436f161421c8e000" +
                  " 4035000000000000 4049600000000000 7ff0000000000000 0000000000000000 401c000000000000 400c000000000000 436f161421c8e000 4035000000000000" +
                  " 4049600000000000 c36f161421c8e000 c02c000000000000 401c000000000000 400c000000000000 436f161421c8e000 4035000000000000 4049600000000000" +
                  " c36f161421c8e000 c02c000000000000 401c000000000000 400c000000000000 436f161421c8e000 4035000000000000 4049600000000000 c36f161421c8e000" +
                  " c02c000000000000 7ff8000000000007 400c000000000000 436f161421c8e000 7ff8000000000006", rawNaN: true);
            var rf = Grid(1, 77, F32Pool);
            rf[0, 3] = NanF(5); rf[0, 70] = NanF(7); rf[0, 76] = NanF(6);
            var rowf = np.array(rf);
            Check(fails, "f32 broadcast row (1,77)->(5,77) axis0", () => np.sum(np.broadcast_to(rowf, new Shape(5, 77)), 0),
                  "float32 (77,) 4dee6b28 41700000 42110000 7fc00005 c1200000 40a00000 40200000 4dee6b28" +
                  " 41700000 42110000 cdee6b28 c1200000 40a00000 40200000 4dee6b28 41700000" +
                  " 42110000 cdee6b28 c1200000 40a00000 40200000 4dee6b28 41700000 42110000" +
                  " cdee6b28 c1200000 40a00000 40200000 4dee6b28 41700000 42110000 cdee6b28" +
                  " c1200000 40a00000 40200000 4dee6b28 41700000 42110000 cdee6b28 c1200000" +
                  " 40a00000 40200000 4dee6b28 41700000 42110000 cdee6b28 c1200000 40a00000" +
                  " 40200000 4dee6b28 41700000 42110000 cdee6b28 c1200000 40a00000 40200000" +
                  " 4dee6b28 41700000 42110000 cdee6b28 c1200000 40a00000 40200000 4dee6b28" +
                  " 41700000 42110000 cdee6b28 c1200000 40a00000 40200000 7fc00007 41700000" +
                  " 42110000 cdee6b28 c1200000 40a00000 7fc00006", rawNaN: true);
            AssertNoMismatch(fails, "every lane of the several-rows form is NumPy's sequence of per-row add calls");
        }

        /// <summary>
        ///     float16 sums through the reduction iterator's elementwise form, which runs the house SIMD widen-compute-narrow
        ///     kernel in place on the out row (<c>DirectILKernelGenerator.HalfArithRun</c>): reversed rows (both operands
        ///     contiguous: two 8-lane vectors and a scalar tail), reversed columns and a stepped view (the input gathered
        ///     into a stack block), a broadcast column (the input one value per call) and a broadcast row. NumPy's HALF_add
        ///     per call: float32 arithmetic narrowed once, the second operand's NaN, a signalling NaN quieted
        ///     (<c>0x7c01</c> -> <c>0x7e01</c>), <c>inf + -inf</c> the default NaN <c>0xfe00</c>, <c>60000 + 60000</c>
        ///     <c>inf</c>, subnormals exact, <c>+0.0</c> from <c>-0.0</c>s, and <c>2048 + 1</c> rounding to 2048. NumPy 2.4.2's
        ///     bits.
        /// </summary>
        [TestMethod]
        public void Sum_Float16_ThroughTheIterator_IsNumPysHalfAddPerCall()
        {
            var fails = new List<string>();
            var g = GridF16(5, 21);
            foreach (var (i, j, p) in new[] { (1, 0, 11), (4, 0, 14), (0, 9, 20), (3, 19, 31) })
                g[i, j] = H((ushort)(0x7E00 | p));
            g[2, 3] = H(0x7C01);
            g[4, 5] = Half.PositiveInfinity; g[1, 5] = Half.NegativeInfinity;
            for (int i = 0; i < 5; i++)
                g[i, 6] = H(0x8000);
            g[0, 7] = (Half)60000.0; g[3, 7] = (Half)60000.0;
            g[1, 8] = H(0x0001); g[2, 8] = H(0x0003);
            var h = np.array(g);
            Check(fails, "f16 h[::-1] axis0", () => np.sum(h["::-1"], 0),
                  "float16 (21,) 7e0b e7f5 45c0 7e01 48e0 fe00 0000 7c00" +
                  " e7fd 7e14 4000 48e0 4860 3800 6805 e7f5" +
                  " 45c0 4000 48e0 7e1f 3800", rawNaN: true);
            Check(fails, "f16 h[:, ::-1] axis0", () => np.sum(h[":, ::-1"], 0),
                  "float16 (21,) 3c00 7e1f 48e0 4000 4600 e7f4 6804 3c00" +
                  " 4880 48e0 4000 7e14 e7fc 7c00 0000 fe00" +
                  " 48e0 7e01 4600 e7f4 7e0e", rawNaN: true);
            Check(fails, "f16 h[::-1, ::2] axis0", () => np.sum(h["::-1, ::2"], 0),
                  "float16 (11,) 7e0b 45c0 48e0 0000 e7fd 4000 4860 6805" +
                  " 45c0 48e0 3800", rawNaN: true);
            var col = np.array(new[,] { { (Half)1.5 }, { (Half)2048.0 }, { H(0x7E09) }, { (Half)(-2048.0) }, { (Half)1.0 } });
            Check(fails, "f16 broadcast column (5,1)->(5,21) axis0", () => np.sum(np.broadcast_to(col, new Shape(5, 21)), 0),
                  "float16 (21,) 7e09 7e09 7e09 7e09 7e09 7e09 7e09 7e09" +
                  " 7e09 7e09 7e09 7e09 7e09 7e09 7e09 7e09" +
                  " 7e09 7e09 7e09 7e09 7e09", rawNaN: true);
            var hr = GridF16(1, 21);
            hr[0, 4] = H(0x7E03);
            var hrow = np.array(hr);
            Check(fails, "f16 broadcast row (1,21)->(5,21) axis0", () => np.sum(np.broadcast_to(hrow, new Shape(5, 21)), 0),
                  "float16 (21,) 7100 4b80 5088 f100 7e03 4500 4100 7100" +
                  " 4b80 5088 f100 c900 4500 4100 7100 4b80" +
                  " 5088 f100 c900 4500 4100", rawNaN: true);
            AssertNoMismatch(fails, "the float16 elementwise form is NumPy's HALF_add per call");
        }

        /// <summary>
        ///     The reduce form of a stride-0 run (a broadcast axis innermost: <c>np.sum</c> of a broadcast column) computes
        ///     NumPy's pairwise sum of n copies of one value from the value itself. It must give the bits of NumPy's pairwise
        ///     sum over a buffer of those copies (<see cref="ILKernelGenerator.PairwiseFold{T}"/>, the scalar port of
        ///     <c>DOUBLE_pairwise_sum</c> / <c>FLOAT_pairwise_sum</c>) for every length: each branch of NumPy's recursion (a run
        ///     shorter than 8, a block of 8 to 128, a split) and the split points past the block.
        /// </summary>
        [TestMethod]
        public unsafe void PairwiseOfCopies_IsNumPysPairwiseSumOverTheCopies()
        {
            double[] values = { 1.1, -0.0, 0.1, 1e300, -7.25, double.PositiveInfinity, 3e-310, Nan(3), 123456.789 };
            var buffer = new double[2600];
            foreach (double v in values)
            {
                Array.Fill(buffer, v);
                fixed (double* p = buffer)
                    for (long n = 0; n <= buffer.Length; n++)
                    {
                        double expected = ILKernelGenerator.PairwiseFold(p, n, 1);
                        double actual = NumPyAddLoops.PairwiseOfCopies(v, n);
                        BitConverter.DoubleToInt64Bits(actual).Should().Be(BitConverter.DoubleToInt64Bits(expected), $"float64 {v} x {n}");
                    }
            }

            float[] fvalues = { 1.1f, -0.0f, 0.1f, 3e38f, -7.25f, float.NegativeInfinity, 1e-40f, NanF(3), 1234.567f };
            var fbuffer = new float[2600];
            foreach (float v in fvalues)
            {
                Array.Fill(fbuffer, v);
                fixed (float* p = fbuffer)
                    for (long n = 0; n <= fbuffer.Length; n++)
                    {
                        float expected = ILKernelGenerator.PairwiseFold(p, n, 1);
                        float actual = NumPyAddLoops.PairwiseOfCopies(v, n);
                        BitConverter.SingleToInt32Bits(actual).Should().Be(BitConverter.SingleToInt32Bits(expected), $"float32 {v} x {n}");
                    }
            }
        }
    }
}
