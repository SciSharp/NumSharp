using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.divmod's <c>out=</c> / <c>where=</c> / <c>dtype=</c> route, which runs NumPy's two-output ufunc call: ONE
    ///     iterator over <c>(x1, x2, out1, out2[, where])</c> whose inner loop reads each element pair once and writes both
    ///     results (parity QA-44: the route used to compose floor_divide and remainder, so a quotient out that was also an
    ///     input fed the remainder pass the QUOTIENTS). Every expected value, text and layout below was produced by NumPy
    ///     2.4.2 on win-amd64.
    /// </summary>
    [TestClass]
    public class NpDivmodOutRouteTests
    {
        // ---- aliasing: an out that is an input --------------------------------------------------------------------

        /// <summary>
        ///     <c>a = [7.5, -7.5, 3.0]; np.divmod(a, 2.0, out=(a, r))</c> leaves <c>a = [3, -4, 1]</c> and
        ///     <c>r = [1.5, 0.5, 1.0]</c>: the remainders of the ORIGINAL dividends (the composition gave <c>[1, 0, 1]</c>).
        /// </summary>
        [TestMethod]
        public void QuotientOutIsTheDividend_RemainderReadsTheOriginal()
        {
            using var a = np.array(new[] { 7.5, -7.5, 3.0 });
            using var r = np.zeros(new Shape(3));
            using var two = np.array(2.0);

            var (q, rem) = np.divmod(a, two, (a, r));

            ReferenceEquals(q, a).Should().BeTrue("a given out is returned itself");
            ReferenceEquals(rem, r).Should().BeTrue();
            a.ToArray<double>().Should().Equal(3.0, -4.0, 1.0);
            r.ToArray<double>().Should().Equal(1.5, 0.5, 1.0);
        }

        /// <summary>
        ///     <c>y = [7, -7, 3]; d = [2, 2, 2]; np.divmod(y, d, out=(d, r))</c> (int64) leaves <c>d = [3, -4, 1]</c> and
        ///     <c>r = [1, 1, 1]</c>: the quotient out aliasing the DIVISOR is read before it is written too.
        /// </summary>
        [TestMethod]
        public void QuotientOutIsTheDivisor_RemainderReadsTheOriginal()
        {
            using var y = np.array(new long[] { 7, -7, 3 });
            using var d = np.array(new long[] { 2, 2, 2 });
            using var r = np.zeros(new Shape(3), NPTypeCode.Int64);

            np.divmod(y, d, (d, r));

            d.ToArray<long>().Should().Equal(3L, -4L, 1L);
            r.ToArray<long>().Should().Equal(1L, 1L, 1L);
        }

        /// <summary>
        ///     A remainder out that is the divisor (<c>np.divmod(a, b, out=(q, b))</c>) leaves <c>b = [1.5, 0.5, 1.0]</c>,
        ///     and the quotient is computed from the original divisor too.
        /// </summary>
        [TestMethod]
        public void RemainderOutIsTheDivisor()
        {
            using var a = np.array(new[] { 7.5, -7.5, 3.0 });
            using var b = np.array(new[] { 2.0, 2.0, 2.0 });
            using var q = np.zeros(new Shape(3));

            np.divmod(a, b, (q, b));

            q.ToArray<double>().Should().Equal(3.0, -4.0, 1.0);
            b.ToArray<double>().Should().Equal(1.5, 0.5, 1.0);
        }

        /// <summary>
        ///     Overlap that is NOT elementwise is resolved by the iterator's COPY_IF_OVERLAP copy, as in NumPy:
        ///     <c>a = arange(10.); np.divmod(a[:-1], 3.0, out=(a[1:], empty(9)))</c> is <c>[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]</c>;
        ///     <c>a = arange(6.); np.divmod(a[::2], 2.0, out=(a[1::2], a[::2]))</c> is <c>[0, 0, 0, 1, 0, 2]</c>;
        ///     <c>a = [1..6]; np.divmod(a, a[::-1], out=(a, empty(6)))</c> is <c>[0, 0, 0, 1, 2, 6]</c>.
        /// </summary>
        [TestMethod]
        public void OverlappingViews_ComputeAsIfFromACopy()
        {
            using var three = np.array(3.0);
            using var a = np.arange(10.0);
            using var r9 = np.empty(new Shape(9));
            np.divmod(a["0:9"], three, (a["1:"], r9));
            a.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0);

            using var two = np.array(2.0);
            using var c = np.arange(6.0);
            np.divmod(c["::2"], two, (c["1::2"], c["::2"]));
            c.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 1.0, 0.0, 2.0);

            using var e = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
            using var r6 = np.empty(new Shape(6));
            np.divmod(e, e["::-1"], (e, r6));
            e.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 1.0, 2.0, 6.0);
        }

        /// <summary>
        ///     The same array for both outs keeps the value NumPy's loop stores LAST: the float loops write the remainder
        ///     inside <c>npy_divmod</c> and then the quotient (<c>o = [3.]</c> for 7.5 divmod 2, float16 too), the integer
        ///     loops the quotient and then the remainder (<c>o = [1]</c> for 7 divmod 2).
        /// </summary>
        [TestMethod]
        public void SameArrayForBothOuts_KeepsNumPysLastStore()
        {
            using var f = np.zeros(new Shape(1));
            using var x = np.array(new[] { 7.5 });
            using var two = np.array(2.0);
            np.divmod(x, two, (f, f));
            f.GetDouble(0).Should().Be(3.0, "the float loop stores the quotient last");

            using var i = np.zeros(new Shape(1), NPTypeCode.Int64);
            using var xi = np.array(new long[] { 7 });
            using var di = np.array(new long[] { 2 });
            np.divmod(xi, di, (i, i));
            i.GetInt64(0).Should().Be(1L, "the integer loop stores the remainder last");

            using var h = np.zeros(new Shape(1), NPTypeCode.Half);
            using var xh = np.array(new[] { (Half)7.5 });
            using var dh = np.array(new[] { (Half)2 });
            np.divmod(xh, dh, (h, h));
            ((double)h.GetHalf(0)).Should().Be(3.0, "float16 follows the float loops");
        }

        // ---- argument checks: NumPy's order and texts ------------------------------------------------------------

        /// <summary>
        ///     NumPy numbers a multi-output ufunc's outputs by OPERAND position: the quotient out is <c>output 2</c>, the
        ///     remainder out <c>output 3</c>; a <c>dtype=</c> input cast names <c>input 0</c> / <c>input 1</c> — all under
        ///     divmod's own name (the composition named floor_divide and remainder).
        /// </summary>
        [TestMethod]
        public void CastRefusals_NameDivmodAndTheOperandPosition()
        {
            using var x = np.array(new[] { 1.5 });
            using var one = np.array(new[] { 1.0 });
            using var i32 = np.empty(new Shape(1), np.int32);
            using var f64 = np.empty(new Shape(1));

            Action quotient = () => np.divmod(x, one, (i32, f64));
            quotient.Should().Throw<TypeError>().WithMessage(
                "Cannot cast ufunc 'divmod' output 2 from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            Action remainder = () => np.divmod(x, one, (f64, i32));
            remainder.Should().Throw<TypeError>().WithMessage(
                "Cannot cast ufunc 'divmod' output 3 from dtype('float64') to dtype('int32') with casting rule 'same_kind'");

            Action input0 = () => np.divmod(x, one, dtype: np.int32);
            input0.Should().Throw<TypeError>().WithMessage(
                "Cannot cast ufunc 'divmod' input 0 from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
            using var ione = np.array(new long[] { 1 });
            Action input1 = () => np.divmod(ione, one, dtype: np.int32);
            input1.Should().Throw<TypeError>().WithMessage(
                "Cannot cast ufunc 'divmod' input 1 from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     Both outputs' writeability is checked before anything else (NumPy's argument parsing), so a read-only
        ///     remainder out beats a bad quotient cast and NOTHING is written: the composition wrote the quotient before it
        ///     even looked at the remainder's out.
        /// </summary>
        [TestMethod]
        public void ReadOnlyOut_IsReportedFirst_AndNothingIsWritten()
        {
            using var x = np.ones(new Shape(3));
            using var two = np.array(2.0);
            using var ro = np.empty(new Shape(3));
            ro.setflags(write: false);
            using var q = np.full(new Shape(3), -1.0, np.float64);
            using var i32 = np.empty(new Shape(3), np.int32);

            Action readOnlyR = () => np.divmod(x, two, (q, ro));
            readOnlyR.Should().Throw<ValueError>().WithMessage("output array is read-only");
            q.ToArray<double>().Should().Equal(-1.0, -1.0, -1.0);

            Action readOnlyQ = () => np.divmod(x, two, (ro, q));
            readOnlyQ.Should().Throw<ValueError>().WithMessage("output array is read-only");

            Action beatsCast = () => np.divmod(x, two, (i32, ro));
            beatsCast.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        /// <summary>
        ///     The iterator's broadcast texts list EVERY operand (x1, x2, out1, out2, where), and an out is never stretched:
        ///     the stretch text names the out and the joint shape, checked for out1 before out2.
        /// </summary>
        [TestMethod]
        public void BroadcastRefusals_ListEveryOperand()
        {
            using var three = np.ones(new Shape(3));
            using var four = np.ones(new Shape(4));
            using var e1 = np.empty(new Shape(3));
            using var e2 = np.empty(new Shape(3));
            using var big = np.empty(new Shape(2, 3));
            using var mask4 = np.ones(new Shape(4), NPTypeCode.Boolean);

            Action inputs = () => np.divmod(three, four, (e1, e2));
            inputs.Should().Throw<IncorrectShapeException>().WithMessage(
                "operands could not be broadcast together with shapes (3,) (4,) (3,) (3,) ");
            Action mask = () => np.divmod(three, three, (e1, e2), mask4);
            mask.Should().Throw<ArgumentException>().WithMessage(
                "operands could not be broadcast together with shapes (3,) (3,) (3,) (3,) (4,) ");
            Action stretchR = () => np.divmod(three, three, (big, e2));
            stretchR.Should().Throw<ArgumentException>().WithMessage(
                "non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)");
            Action stretchQ = () => np.divmod(three, three, (e1, big));
            stretchQ.Should().Throw<ArgumentException>().WithMessage(
                "non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)");
        }

        /// <summary>
        ///     divmod has no bool and no complex loop: <c>dtype=bool</c> / <c>dtype=complex128</c> are NumPy's no-loop
        ///     TypeError (a bool PROMOTION runs the int8 loop instead); a non-bool mask is the mask's cast refusal; a C#
        ///     integer literal outside an int8 loop is NumPy's Python-int OverflowError.
        /// </summary>
        [TestMethod]
        public void LoopRefusals()
        {
            using var t = np.array(new[] { true });
            Action boolLoop = () => np.divmod(t, t, dtype: np.dtype("bool"));
            boolLoop.Should().Throw<TypeError>().WithMessage(
                "No loop matching the specified signature and casting was found for ufunc divmod");
            using var x = np.ones(new Shape(2));
            using var two = np.array(2.0);
            Action complexLoop = () => np.divmod(x, two, dtype: np.complex128);
            complexLoop.Should().Throw<TypeError>().WithMessage(
                "No loop matching the specified signature and casting was found for ufunc divmod");

            using var i64Mask = np.ones(new Shape(2), NPTypeCode.Int64);
            Action badMask = () => np.divmod(x, two, where: i64Mask);
            badMask.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");

            using var i8 = np.array(new sbyte[] { 1 });
            using var q8 = np.empty(new Shape(1), np.int8);
            using var r8 = np.empty(new Shape(1), np.int8);
            Action overflow = () => np.divmod(i8, 1000, (q8, r8));
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
        }

        // ---- values, partial outs, masks, casts --------------------------------------------------------------------

        /// <summary>
        ///     <c>out=(None, r)</c>: the omitted quotient is a fresh array, the given remainder is written and returned
        ///     (<c>np.divmod([7, -7], 2, out=(None, r))</c> is <c>([3, -4], [1, 1])</c>).
        /// </summary>
        [TestMethod]
        public void PartialOut_FreshQuotientGivenRemainder()
        {
            using var x = np.array(new long[] { 7, -7 });
            using var two = np.array(new long[] { 2 });
            using var r = np.zeros(new Shape(2), NPTypeCode.Int64);

            var (q, rem) = np.divmod(x, two, (null, r));

            using (q)
            {
                q.typecode.Should().Be(NPTypeCode.Int64);
                q.ToArray<long>().Should().Equal(3L, -4L);
            }
            ReferenceEquals(rem, r).Should().BeTrue();
            r.ToArray<long>().Should().Equal(1L, 1L);
        }

        /// <summary>
        ///     <c>where=[True, False, True]</c> into <c>q = full(-1)</c>, <c>r = full(-2)</c> leaves <c>q = [3, -1, 1]</c>,
        ///     <c>r = [1.5, -2, 1]</c>; a (2,1) row mask broadcast over a (2,3) iteration writes only the first row
        ///     (<c>r = [[0, 1, 2], [0, 0, 0]]</c>).
        /// </summary>
        [TestMethod]
        public void Where_MaskedOffPositionsKeepTheirValues()
        {
            using var x = np.array(new[] { 7.5, -7.5, 3.0 });
            using var two = np.array(2.0);
            using var q = np.full(new Shape(3), -1.0, np.float64);
            using var r = np.full(new Shape(3), -2.0, np.float64);
            using var mask = np.array(new[] { true, false, true });

            np.divmod(x, two, (q, r), mask);

            q.ToArray<double>().Should().Equal(3.0, -1.0, 1.0);
            r.ToArray<double>().Should().Equal(1.5, -2.0, 1.0);

            using var m = np.arange(6.0).reshape(2, 3);
            using var d = np.array(new[] { 2.0, 3.0, 4.0 });
            using var q2 = np.zeros(new Shape(2, 3));
            using var r2 = np.zeros(new Shape(2, 3));
            using var rows = np.array(new[,] { { true }, { false } });
            np.divmod(m, d, (q2, r2), rows);
            q2.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            r2.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        }

        /// <summary>
        ///     The loop is resolved from the inputs and cast into the outs: an int64 loop into float64 outs
        ///     (<c>[7, -7, 3] divmod [2, 2, 0]</c> → <c>[3, -4, 0]</c>, <c>[1, 1, 0]</c>, integer ÷0 being (0, 0)); a
        ///     <c>dtype=float32</c> loop into float64 outs rounds through float32 (<c>7.3 % 2.0</c> → <c>1.30000019…</c>);
        ///     integer inputs under <c>dtype=float64</c> / <c>dtype=float16</c> compute in that float loop; bool inputs run
        ///     the int8 loop.
        /// </summary>
        [TestMethod]
        public void LoopDtype_ComesFromTheInputsOrDtype_AndCastsIntoTheOuts()
        {
            using var x = np.array(new long[] { 7, -7, 3 });
            using var d = np.array(new long[] { 2, 2, 0 });
            using var qf = np.zeros(new Shape(3));
            using var rf = np.zeros(new Shape(3));
            np.divmod(x, d, (qf, rf));
            qf.ToArray<double>().Should().Equal(3.0, -4.0, 0.0);
            rf.ToArray<double>().Should().Equal(1.0, 1.0, 0.0);

            using var s = np.array(new[] { 7.3 });
            using var two = np.array(new[] { 2.0 });
            using var q1 = np.zeros(new Shape(1));
            using var r1 = np.zeros(new Shape(1));
            np.divmod(s, two, (q1, r1), dtype: np.float32);
            q1.GetDouble(0).Should().Be(3.0);
            r1.GetDouble(0).Should().Be((double)(7.3f - 6f), "the float32 loop's remainder, widened");

            using var xi = np.array(new long[] { 7, -7 });
            using var di = np.array(new long[] { 2, 2 });
            var (qd, rd) = np.divmod(xi, di, dtype: np.float64);
            using (qd)
            using (rd)
            {
                qd.typecode.Should().Be(NPTypeCode.Double);
                qd.ToArray<double>().Should().Equal(3.0, -4.0);
                rd.ToArray<double>().Should().Equal(1.0, 1.0);
            }

            using var x8 = np.array(new sbyte[] { 7, -7 });
            using var d8 = np.array(new sbyte[] { 2, 2 });
            var (qh, rh) = np.divmod(x8, d8, dtype: np.float16);
            using (qh)
            using (rh)
            {
                qh.typecode.Should().Be(NPTypeCode.Half);
                ((double)qh.GetHalf(0)).Should().Be(3.0);
                ((double)qh.GetHalf(1)).Should().Be(-4.0);
                ((double)rh.GetHalf(0)).Should().Be(1.0);
                ((double)rh.GetHalf(1)).Should().Be(1.0);
            }

            using var tb = np.array(new[] { true, false });
            using var tt = np.array(new[] { true, true });
            var (qb, rb) = np.divmod(tb, tt, dtype: np.int8);
            using (qb)
            using (rb)
            {
                qb.typecode.Should().Be(NPTypeCode.SByte);
                qb.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)0);
                rb.ToArray<sbyte>().Should().Equal((sbyte)0, (sbyte)0);
            }
        }

        /// <summary>
        ///     The special values match the fused no-out kernel (one Divmod core): int64 ÷0 → (0, 0) and MIN / -1 → (MIN, 0);
        ///     float ÷0 → (±inf / nan, nan).
        /// </summary>
        [TestMethod]
        public void SpecialValues_MatchTheNoOutKernel()
        {
            using var x = np.array(new[] { 7L, -7L, long.MinValue });
            using var d = np.array(new[] { 0L, 0L, -1L });
            using var q = np.zeros(new Shape(3), NPTypeCode.Int64);
            using var r = np.zeros(new Shape(3), NPTypeCode.Int64);
            np.divmod(x, d, (q, r));
            q.ToArray<long>().Should().Equal(0L, 0L, long.MinValue);
            r.ToArray<long>().Should().Equal(0L, 0L, 0L);

            using var f = np.array(new[] { 7.0, -7.0, 0.0, double.NaN, double.PositiveInfinity });
            using var zero = np.array(0.0);
            using var qf = np.zeros(new Shape(5));
            using var rf = np.zeros(new Shape(5));
            np.divmod(f, zero, (qf, rf));
            var qv = qf.ToArray<double>();
            qv[0].Should().Be(double.PositiveInfinity);
            qv[1].Should().Be(double.NegativeInfinity);
            double.IsNaN(qv[2]).Should().BeTrue();
            double.IsNaN(qv[3]).Should().BeTrue();
            qv[4].Should().Be(double.PositiveInfinity);
            foreach (var v in rf.ToArray<double>())
                double.IsNaN(v).Should().BeTrue();
        }

        /// <summary>
        ///     Inputs broadcast UP to the outs (<c>[7, -7] divmod [[2], [3]]</c> into (2,2) int64 outs is
        ///     <c>[[3, -4], [2, -3]]</c>, <c>[[1, 1], [1, 2]]</c>); a 0-d call returns its 0-d outs; an empty call returns
        ///     its empty outs untouched.
        /// </summary>
        [TestMethod]
        public void Shapes_BroadcastUp_ZeroD_Empty()
        {
            using var x = np.array(new long[] { 7, -7 });
            using var d = np.array(new long[,] { { 2 }, { 3 } });
            using var q = np.zeros(new Shape(2, 2), NPTypeCode.Int64);
            using var r = np.zeros(new Shape(2, 2), NPTypeCode.Int64);
            np.divmod(x, d, (q, r));
            q.ToArray<long>().Should().Equal(3L, -4L, 2L, -3L);
            r.ToArray<long>().Should().Equal(1L, 1L, 1L, 2L);

            using var x0 = np.array(7.5);
            using var d0 = np.array(2.0);
            using var q0 = np.array(0.0);
            using var r0 = np.array(0.0);
            var (qq, rr) = np.divmod(x0, d0, (q0, r0));
            ReferenceEquals(qq, q0).Should().BeTrue();
            ReferenceEquals(rr, r0).Should().BeTrue();
            q0.ndim.Should().Be(0);
            q0.GetAtIndex<double>(0).Should().Be(3.0);
            r0.GetAtIndex<double>(0).Should().Be(1.5);

            using var empty = np.zeros(new Shape(0, 3));
            using var qe = np.zeros(new Shape(0, 3));
            using var re = np.zeros(new Shape(0, 3));
            var (qx, rx) = np.divmod(empty, d0, (qe, re));
            ReferenceEquals(qx, qe).Should().BeTrue();
            ReferenceEquals(rx, re).Should().BeTrue();
            qe.shape.Should().Equal(0L, 3L);
        }

        /// <summary>
        ///     A fresh result takes the layout NumPy's iterator allocates, every passed operand voting: F inputs under an F
        ///     mask give F results; with <c>out=(None, F)</c> the fresh quotient is F; with <c>out=(None, C)</c> beside F
        ///     inputs the operands disagree and the quotient is C; F inputs under <c>dtype=</c> alone give F results.
        /// </summary>
        [TestMethod]
        public void FreshResults_TakeTheIteratorsLayout()
        {
            using var f = np.asfortranarray(np.arange(1.0, 7.0).reshape(2, 3));
            using var two = np.array(2.0);

            using var fmask = np.asfortranarray(np.ones(new Shape(2, 3), NPTypeCode.Boolean));
            var (q1, r1) = np.divmod(f, two, where: fmask);
            using (q1)
            using (r1)
            {
                q1.Shape.IsFContiguous.Should().BeTrue();
                r1.Shape.IsFContiguous.Should().BeTrue();
            }

            using var fout = np.empty(new Shape(2, 3), 'F', np.float64);
            var (q2, _) = np.divmod(f, two, (null, fout));
            using (q2)
                q2.Shape.IsFContiguous.Should().BeTrue();

            using var cout = np.empty(new Shape(2, 3));
            var (q3, _) = np.divmod(f, two, (null, cout));
            using (q3)
            {
                q3.Shape.IsContiguous.Should().BeTrue();
                q3.Shape.IsFContiguous.Should().BeFalse();
            }

            var (q4, r4) = np.divmod(f, two, dtype: np.float64);
            using (q4)
            using (r4)
            {
                q4.Shape.IsFContiguous.Should().BeTrue();
                r4.Shape.IsFContiguous.Should().BeTrue();
                q4.ToArray<double>().Should().Equal(0.0, 1.0, 1.0, 2.0, 2.0, 3.0);
            }
        }

        /// <summary>
        ///     NEP 50 in the out route: a C# integer literal that must round twice into a float32 loop is packed as NumPy
        ///     packs it (<c>float32([2**60]) divmod (2**60 + 2**36 + 1)</c> into float32 outs is <c>([1], [0])</c>); a
        ///     decimal loop (NumSharp's extension) floors like the float loops.
        /// </summary>
        [TestMethod]
        public void WeakIntPack_AndDecimalLoop()
        {
            using var big = np.array(new[] { (float)(1L << 60) });
            using var q = np.zeros(new Shape(1), NPTypeCode.Single);
            using var r = np.zeros(new Shape(1), NPTypeCode.Single);
            np.divmod(big, (1L << 60) + (1L << 36) + 1, (q, r));
            q.GetSingle(0).Should().Be(1f);
            r.GetSingle(0).Should().Be(0f);

            using var m = np.array(new[] { 7.5m, -7.5m });
            using var dm = np.array(new[] { 2m, 2m });
            using var qm = np.zeros(new Shape(2), NPTypeCode.Decimal);
            using var rm = np.zeros(new Shape(2), NPTypeCode.Decimal);
            np.divmod(m, dm, (qm, rm));
            qm.ToArray<decimal>().Should().Equal(3m, -4m);
            rm.ToArray<decimal>().Should().Equal(1.5m, 0.5m);
        }

        /// <summary>
        ///     Parity QA-27: divmod has no complex loop, so a complex operand is NumPy's generic no-loop TypeError, raised
        ///     for the LOOP before any element is read (a zero-size operand too), on the plain route and the out= / where=
        ///     route alike; a non-bool mask is refused first (NumPy's argument parsing), a real dtype= turns it into the
        ///     numbered input cast refusal, and dtype=complex is the no-loop text. Every row is NumPy 2.4.2's answer.
        /// </summary>
        [TestMethod]
        public void Complex_RaisesNumPysTypeErrorInNumPysOrder()
        {
            const string notSupported = "ufunc 'divmod' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''";
            using var c = np.array(new[] { new System.Numerics.Complex(1, 2), new System.Numerics.Complex(3, -1) });
            using var f = np.array(new[] { 2.0, 1.0 });
            using var emptyC = np.zeros(new Shape(0), NPTypeCode.Complex);
            using var one = NDArray.Scalar(1.0);
            using var two = NDArray.Scalar(2.0);
            using var q = np.empty(new Shape(2), NPTypeCode.Complex);
            using var r = np.empty(new Shape(2), NPTypeCode.Complex);
            using var boolMask = np.array(new[] { true, false });
            using var intMask = np.array(new[] { 1L, 0L });

            new Action(() => np.divmod(c, f)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(f, c)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(c, c)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(emptyC, one)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(c, two)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(c, f, @out: (q, r))).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(c, f, where: boolMask)).Should().ThrowExactly<TypeError>().WithMessage(notSupported);
            new Action(() => np.divmod(c, f, dtype: NPTypeCode.Double)).Should().ThrowExactly<TypeError>().WithMessage(
                "Cannot cast ufunc 'divmod' input 0 from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
            new Action(() => np.divmod(f, f, dtype: NPTypeCode.Complex)).Should().Throw<TypeError>().WithMessage(
                "No loop matching the specified signature and casting was found for ufunc divmod");
            // The mask is parsed first: its refusal (NumPy's TypeError text, the house ArgumentException of where=).
            new Action(() => np.divmod(c, f, where: intMask)).Should().ThrowExactly<ArgumentException>().WithMessage(
                "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'");
        }
    }
}
