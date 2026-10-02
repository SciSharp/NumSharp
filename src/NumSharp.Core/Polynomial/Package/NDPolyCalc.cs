using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyCalc.cs — numpy.polynomial's calculus family: {p}der and {p}int for every basis (plan U4)
// =============================================================================
//
// NumPy 2.4.2's twelve functions share one Python prologue and differ only in a 2-4 line recurrence. This
// file is the prologue and the orchestration, in NumPy's statement order — which is also its error order:
//
//   der(c, m=1, scl=1, axis=0)                  int(c, m=1, k=[], lbnd=0, scl=1, axis=0)
//     c = np.array(c, ndmin=1, copy=True)         c = np.array(c, ndmin=1, copy=True)
//     ints/bool -> float64                        ints/bool -> float64
//                                                 k = [k] unless iterable
//     cnt = _as_int(m); iaxis = _as_int(axis)     cnt, iaxis (C# ints: the TypeErrors are compile errors)
//     cnt < 0 -> ValueError                       cnt < 0 -> ValueError; len(k) > cnt -> ValueError;
//                                                 ndim(lbnd) != 0, ndim(scl) != 0 -> ValueError
//     normalize_axis_index -> AxisError           normalize_axis_index -> AxisError
//     cnt == 0 -> return c (the copy)             cnt == 0 -> return c (the copy)
//     moveaxis(c, iaxis, 0)                       k padded with 0; moveaxis(c, iaxis, 0)
//     cnt >= n -> c[:1]*0                         per order: c *= scl; n == 1 and all(c[0] == 0) -> c[0] += k[i]
//     per order: c *= scl; recurrence               else recurrence; tmp[0] += k[i] - {p}val(lbnd, tmp)
//     moveaxis(c, 0, iaxis) [hermeder: not for    moveaxis(c, 0, iaxis)
//       the cnt >= n branch]
//
// The recurrences and `c *= scl` run in ONE IL kernel per call (DirectILKernelGenerator.PolyCalculus.cs),
// in place in ONE buffer whatever the order. What stays here, because it is Python-level arithmetic on
// Python/NumPy scalars and small arrays, runs through the U1/U3 engines that already reproduce it:
//   * the integral's `tmp[0] += k[i] - {p}val(lbnd, tmp)`: {p}val is U3's NDPolyEval.Val (a Python-scalar or
//     0-d lbnd, the evaluation kernel's weak-x typing); the subtraction and the add are PolyNumber arithmetic
//     — scalarmath for a 1-D series (the result is stored back with setitem's cast: a complex value into a
//     real series keeps its real part, NumPy's ComplexWarning), in-place ufuncs for an N-D one (a complex
//     value into a real series raises NumPy's UFuncTypeError);
//   * the n == 1 branch (NumPy only takes it for a one-coefficient series, so it is one row of work);
//   * a `scl` the kernel cannot fold: an array (derivatives accept one — it broadcasts, per NumPy's in-place
//     rules; one `np.multiply(c, scl, out=c, dtype=loop)` pass per order) or a promotion the kernel has no widened
//     loop for (decimal). A NumPy scalar / 0-d array that promotes a float16 / float32 series (`c *= np.float64(x)`
//     computes in float64 and casts back) IS folded: the kernel widens, multiplies in that loop and casts back.
//
// LAYOUT — the result is the same view NumPy returns
// --------------------------------------------------
// NumPy's der/tmp are np.empty (C order in the MOVED axis order), returned through np.moveaxis: a view whose
// strides depend only on the shape and the axis. The buffer here is exactly that array plus the rows the
// in-place recurrence consumes, and the result is the same view of it (OWNDATA false, like NumPy's). Three
// results are NOT a fresh buffer in NumPy and are built to its layout instead: cnt == 0 returns the K-order
// copy, der's cnt >= n returns c[:1]*0 (a ufunc output laid out by NpyIter's KEEPORDER vote over c[:1], and
// hermeder skips the moveaxis back), and an integral whose every order took the n == 1 zero branch returns
// the moved copy itself.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy 2.4.2's <c>{p}der</c> / <c>{p}int</c> for the six <c>numpy.polynomial</c> bases (see the file header).
    ///     One shared implementation behind the basis facades; the basis is data (<see cref="PolyBasis"/>).
    /// </summary>
    internal static unsafe class NDPolyCalc
    {
        /// <summary>How <c>c *= scl</c> is carried out.</summary>
        private enum ScaleRoute
        {
            /// <summary>The kernel multiplies by the converted scalar as it loads/scales the series — in the series dtype, or
            ///     widened into a promoting strong scalar's loop (<see cref="PolyCalcKey.ScaleLoop"/>).</summary>
            Fused,
            /// <summary>An in-place ufunc per order (<see cref="ScaleInPlace"/>): an array scl, or a promotion the kernel has
            ///     no widened loop for.</summary>
            House,
        }

        /// <summary>The Python int 1 — NumPy's default <c>scl</c>.</summary>
        private static readonly PyScalar s_one = PyScalar.Int(1);

        /// <summary>The Python int 0 — NumPy's default <c>lbnd</c> and the padding of <c>k</c>.</summary>
        private static readonly object s_zero = 0L;

        // ---------------------------------------------------------------------------------------------
        //  Entry points
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>{p}der(c, m, scl, axis)</c>: the series differentiated <paramref name="m"/> times along
        ///     <paramref name="axis"/>, each time multiplied by <paramref name="scl"/> (NumPy's chain-rule factor).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The coefficients (anything <c>np.array</c> accepts; see <see cref="Coefficients"/>).</param>
        /// <param name="m">The order (≥ 0).</param>
        /// <param name="scl">The scale: null for NumPy's default Python int 1, a Python/NumPy scalar, or an array that
        ///     broadcasts in place against the series (a derivative accepts one).</param>
        /// <param name="axis">The series axis (negative counts from the end).</param>
        /// <returns>The derivative, of the series' dtype (integer and bool series become float64).</returns>
        /// <exception cref="ValueError"><c>The order of derivation must be non-negative</c>, or an array scl whose
        ///     broadcast would stretch the series (NumPy's non-broadcastable-output text).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="ArgumentException">A scl the series cannot absorb — NumPy's UFuncTypeError text
        ///     (<c>Cannot cast ufunc 'multiply' output from dtype('complex128') to dtype('float64') ...</c>).</exception>
        /// <exception cref="IncorrectShapeException">An array scl that does not broadcast with the series.</exception>
        /// <exception cref="NotSupportedException">A null or string series (object/str arrays in NumPy).</exception>
        internal static NDArray Der(PolyBasis basis, object c, int m, object scl, int axis)
        {
            NDArray src = Coefficients(c);
            NPTypeCode t = PolyTyping.IsIntLike(src.typecode) ? NPTypeCode.Double : src.typecode;
            if (m < 0)
                throw new ValueError("The order of derivation must be non-negative");
            int iaxis = NormalizeAxis(axis, src.ndim);
            if (m == 0)
                return KCopyAs(src, t);

            Moved(src.Shape, iaxis, out var md, out var ms);
            long n = md[0];
            if (m >= n)
                return FirstRowTimesZero(src, t, iaxis, moveBack: basis != PolyBasis.HermiteE);

            // NumPy raises a scl it cannot cast at the first `c *= scl`: before any work, so here.
            byte* sclRaw = stackalloc byte[16];
            var route = ClassifyScale(scl, t, out var sclNum, sclRaw, out var scaleLoop);

            bool oneD = md.Length == 1;
            long cols = Cols(md);
            int size = DirectILKernelGenerator.GetTypeSize(t);
            var buf = new NDArray(t, new Shape(BufferDims(md, n)), false);
            byte* b0 = Ptr(buf);
            long bufRow = cols * size;
            long block = Block(n, size, cols);

            if (route == ScaleRoute.Fused)
            {
                if (TryDirectSource(src, md, ms, out var sp, out long sRow, out long sCol))
                {
                    // One pass: load + convert + scale + all m orders, block by block.
                    var k = DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, false, t, src.typecode, oneD, true, scaleLoop));
                    k(sp, sRow, sCol, b0, bufRow, cols, n, m, block, sclRaw);
                }
                else
                {
                    // A layout the kernel cannot walk as (row, flat column): the house copy lays the series into the
                    // buffer first, then the kernel scales it in place.
                    using (var dst = MovedBackView(buf, 0, n, md, iaxis))
                        NDIter.Copy(dst, src);
                    var k = DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, false, t, t, oneD, true, scaleLoop));
                    k(null, 0, 0, b0, bufRow, cols, n, m, block, sclRaw);
                }
            }
            else
            {
                using (var dst = MovedBackView(buf, 0, n, md, iaxis))
                    NDIter.Copy(dst, src);
                var k = DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, false, t, t, oneD, false));
                for (int o = 0; o < m; o++)
                {
                    // c *= scl for this order's c (rows [o, n)), then one order of the recurrence.
                    using (var win = MovedView(buf, o, n - o, md))
                        ScaleInPlace(win, sclNum);
                    k(null, 0, 0, b0 + o * bufRow, bufRow, cols, n - o, 1, block, null);
                }
            }

            return MovedBackView(buf, m, n - m, md, iaxis);
        }

        /// <summary>
        ///     <c>{p}int(c, m, k, lbnd, scl, axis)</c>: the series integrated <paramref name="m"/> times along
        ///     <paramref name="axis"/>, each order multiplied by <paramref name="scl"/> and shifted so its value at
        ///     <paramref name="lbnd"/> is the matching integration constant.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The coefficients (anything <c>np.array</c> accepts).</param>
        /// <param name="m">The order (≥ 0).</param>
        /// <param name="k">The integration constants (null for NumPy's <c>[]</c>): a scalar, a Python tuple / list
        ///     (ValueTuple, object[], jagged / NDArray[] arrays, any IList — Python items, each converted only when its
        ///     order uses it, as NumPy does), a str (its characters), or an array (a typed C# array or NDArray —
        ///     NumPy-scalar elements, or rows of an N-D array); missing ones are 0.</param>
        /// <param name="lbnd">The lower bound (null for NumPy's Python int 0); must be a scalar.</param>
        /// <param name="scl">The scale (null for NumPy's Python int 1); must be a scalar.</param>
        /// <param name="axis">The series axis.</param>
        /// <returns>The integral, of the series' dtype (integer and bool series become float64).</returns>
        /// <exception cref="ValueError">NumPy's texts: <c>The order of integration must be non-negative</c>,
        ///     <c>Too many integration constants</c>, <c>lbnd must be a scalar.</c>, <c>scl must be a scalar.</c>,
        ///     <c>setting an array element with a sequence.</c> (an array constant for a 1-D series), a
        ///     non-broadcastable constant, or a ragged sequence (np.array's inhomogeneous-shape text — for lbnd/scl at the
        ///     ndim checks, for a constant when its order uses it).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="IndexError">An empty series: <c>index 0 is out of bounds for axis 0 with size 0</c>.</exception>
        /// <exception cref="TypeError">An array constant for a complex 1-D series (<c>complex()</c>'s text).</exception>
        /// <exception cref="ArgumentException">A scl, constant or lbnd value the series cannot absorb in place (NumPy's
        ///     UFuncTypeError text).</exception>
        /// <exception cref="IncorrectShapeException">Constants that do not broadcast with the series' columns.</exception>
        /// <exception cref="OverflowException">A Python-int scl / lbnd / constant too large for the series' float dtype
        ///     (<c>int too large to convert to float</c>).</exception>
        /// <exception cref="NotSupportedException">A null or string series; a str / null constant or a str lbnd when an
        ///     order USES it (NumPy fails there with a str-dtype error; a str lbnd is never read by an order that takes the
        ///     one-coefficient zero branch, nor at all when <paramref name="m"/> is 0).</exception>
        internal static NDArray Int(PolyBasis basis, object c, int m, object k, object lbnd, object scl, int axis)
        {
            NDArray src = Coefficients(c);
            NPTypeCode t = PolyTyping.IsIntLike(src.typecode) ? NPTypeCode.Double : src.typecode;
            var ks = IntegrationConstants(k);
            if (m < 0)
                throw new ValueError("The order of integration must be non-negative");
            if (ks.Count > m)
                throw new ValueError("Too many integration constants");
            // np.ndim: a scalar (a str too — NumPy only fails when it EVALUATES at a str lbnd, see Correct) is 0; a
            // ragged sequence raises np.asarray's inhomogeneous ValueError from here.
            if (NDim(lbnd) != 0)
                throw new ValueError("lbnd must be a scalar.");
            if (NDim(scl) != 0)
                throw new ValueError("scl must be a scalar.");
            int iaxis = NormalizeAxis(axis, src.ndim);
            if (m == 0)
                return KCopyAs(src, t);

            // k = list(k) + [0] * (cnt - len(k))
            while (ks.Count < m)
                ks.Add(s_zero);
            object lbndArg = lbnd ?? s_zero;

            Moved(src.Shape, iaxis, out var md, out var ms);
            long n = md[0];
            // `c *= scl` runs before NumPy reads c[0], so a scl it cannot cast is reported before an empty series.
            byte* sclRaw = stackalloc byte[16];
            var route = ClassifyScale(scl, t, out var sclNum, sclRaw, out var scaleLoop);
            if (n == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");

            if (n > 1)
                return Grow(basis, src, null, t, md, ms, iaxis, 0, m, ks, lbndArg, route, sclNum, sclRaw, scaleLoop);

            // One coefficient: NumPy's `n == 1 and np.all(c[0] == 0)` branch keeps the series one coefficient long
            // for as long as it holds. It works on NumPy's own copy (whose layout the result then keeps).
            var cK = KCopyAs(src, t);
            var cm = MoveAxisView(cK, iaxis, 0);
            bool oneD = md.Length == 1;
            for (int i = 0; i < m; i++)
            {
                InPlace(BinaryOp.Multiply, cm, sclNum, "multiply");
                if (!FirstRowIsZero(cm, oneD))
                    return Grow(basis, null, cm, t, md, ms, iaxis, i, m, ks, lbndArg, route, sclNum, sclRaw, scaleLoop);
                AddToFirstRow(cm, ks[i], oneD);
            }
            // NumPy returns np.moveaxis(c, 0, iaxis) of its moved copy: a VIEW (OWNDATA false) in the copy's layout.
            return MoveAxisView(cm, 0, iaxis);
        }

        // ---------------------------------------------------------------------------------------------
        //  The integral's growing orders
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Orders <paramref name="first"/>..m-1 of an integral, each growing the series by one coefficient, in ONE
        ///     buffer: the series sits <c>m - first</c> rows down and every order writes its tmp one row higher (see the
        ///     kernel file's in-place layout). After each order the lbnd/constant correction runs on tmp[0].
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="src">The caller's series (null when <paramref name="scaled"/> carries it).</param>
        /// <param name="scaled">A one-coefficient series (moved order) already multiplied by scl for order
        ///     <paramref name="first"/> — the n == 1 branch found it nonzero — or null.</param>
        /// <param name="t">The coefficient dtype.</param>
        /// <param name="md">The moved dims (series axis first) of the caller's series.</param>
        /// <param name="ms">The moved element strides of the caller's series.</param>
        /// <param name="iaxis">The normalized series axis.</param>
        /// <param name="first">The first order to run.</param>
        /// <param name="m">The order count.</param>
        /// <param name="ks">The padded constants (see <see cref="IntegrationConstants"/>: converted where an order uses them).</param>
        /// <param name="lbndArg">The lower bound as {p}val takes it.</param>
        /// <param name="route">How scl is applied.</param>
        /// <param name="sclNum">The scale (House route).</param>
        /// <param name="sclRaw">The converted scale (Fused route).</param>
        /// <param name="scaleLoop">The Fused route's multiply loop dtype (<paramref name="t"/> or the wider loop).</param>
        /// <returns>The integral (a moveaxis view of the buffer).</returns>
        private static NDArray Grow(PolyBasis basis, NDArray src, NDArray scaled, NPTypeCode t, long[] md, long[] ms, int iaxis,
            int first, int m, List<object> ks, object lbndArg, ScaleRoute route, in PolyNumber sclNum, byte* sclRaw, NPTypeCode scaleLoop)
        {
            bool oneD = md.Length == 1;
            long len = scaled is null ? md[0] : 1;
            long cols = Cols(md);
            int remaining = m - first;
            int size = DirectILKernelGenerator.GetTypeSize(t);
            var buf = new NDArray(t, new Shape(BufferDims(md, len + remaining)), false);
            byte* b0 = Ptr(buf);
            long bufRow = cols * size;
            long block = Block(len + remaining, size, cols);
            long ws = remaining;   // the row c[0] sits at
            var scaleKernel = route == ScaleRoute.Fused
                ? DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, true, t, t, oneD, true, scaleLoop))
                : null;
            var plainKernel = DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, true, t, t, oneD, false));

            for (int i = first; i < m; i++)
            {
                byte* tmp0 = b0 + (ws - 1) * bufRow;   // the row tmp[0] goes to
                if (i == first && scaled is not null)
                {
                    using (var dst = MovedView(buf, ws, 1, md))
                        NDIter.Copy(dst, scaled);
                    plainKernel(null, 0, 0, tmp0, bufRow, cols, len, 1, block, null);
                }
                else if (i == first)
                {
                    if (route == ScaleRoute.Fused && TryDirectSource(src, md, ms, out var sp, out long sRow, out long sCol))
                    {
                        var k = DirectILKernelGenerator.GetPolyCalcKernel(new PolyCalcKey(basis, true, t, src.typecode, oneD, true, scaleLoop));
                        k(sp, sRow, sCol, tmp0, bufRow, cols, len, 1, block, sclRaw);
                    }
                    else
                    {
                        using (var dst = MovedBackView(buf, ws, len, md, iaxis))
                            NDIter.Copy(dst, src);
                        ScaledOrder(buf, ws, len, md, route, scaleKernel, plainKernel, sclNum, sclRaw, tmp0, bufRow, cols, block);
                    }
                }
                else
                    ScaledOrder(buf, ws, len, md, route, scaleKernel, plainKernel, sclNum, sclRaw, tmp0, bufRow, cols, block);

                ws -= 1;
                len += 1;
                Correct(basis, buf, ws, len, md, oneD, t, ks[i], lbndArg);
            }
            return MovedBackView(buf, ws, len, md, iaxis);
        }

        /// <summary>One integral order over a series already in the buffer: <c>c *= scl</c> (fused, or an in-place
        ///     ufunc), then the recurrence.</summary>
        /// <param name="buf">The buffer.</param><param name="ws">The row c[0] sits at.</param><param name="len">len(c).</param>
        /// <param name="md">Moved dims.</param><param name="route">How scl is applied.</param>
        /// <param name="scaleKernel">The fused-scale kernel (Fused route).</param><param name="plainKernel">The non-scaling kernel.</param>
        /// <param name="sclNum">The scale (House route).</param><param name="sclRaw">The converted scale (Fused route).</param>
        /// <param name="tmp0">tmp[0]'s row.</param><param name="bufRow">Row stride in bytes.</param>
        /// <param name="cols">Columns.</param><param name="block">Columns per kernel block.</param>
        private static void ScaledOrder(NDArray buf, long ws, long len, long[] md, ScaleRoute route, PolyCalcKernel scaleKernel,
            PolyCalcKernel plainKernel, in PolyNumber sclNum, byte* sclRaw, byte* tmp0, long bufRow, long cols, long block)
        {
            if (route == ScaleRoute.Fused)
            {
                scaleKernel(null, 0, 0, tmp0, bufRow, cols, len, 1, block, sclRaw);
                return;
            }
            using (var win = MovedView(buf, ws, len, md))
                ScaleInPlace(win, sclNum);
            plainKernel(null, 0, 0, tmp0, bufRow, cols, len, 1, block, null);
        }

        /// <summary>
        ///     <c>tmp[0] += k[i] - {p}val(lbnd, tmp)</c> for the tmp at buffer rows [ws, ws + len): scalarmath and a
        ///     setitem for a 1-D series, in-place ufunc arithmetic on tmp[0]'s row for an N-D one.
        /// </summary>
        /// <param name="basis">The basis ({p}val's).</param><param name="buf">The buffer.</param>
        /// <param name="ws">tmp[0]'s row.</param><param name="len">len(tmp).</param><param name="md">Moved dims.</param>
        /// <param name="oneD">A 1-D series.</param><param name="t">The coefficient dtype.</param>
        /// <param name="kRaw">The order's constant as <see cref="IntegrationConstants"/> holds it (converted here, AFTER
        ///     {p}val ran: Python evaluates <c>k[i] - chebval(lbnd, tmp)</c> left to right and only the subtraction
        ///     converts <c>k[i]</c>, so an lbnd error is reported before a constant's).</param>
        /// <param name="lbndArg">The lower bound.</param>
        /// <exception cref="ValueError">An array constant for a real 1-D series (NumPy's setitem text), a constant whose
        ///     broadcast would stretch tmp[0], or a ragged constant (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="TypeError">An array constant for a complex 1-D series (<c>complex()</c>'s text).</exception>
        /// <exception cref="ArgumentException">A result tmp[0] cannot hold in place (N-D) — named complex64 where NumPy's
        ///     value is complex64.</exception>
        /// <exception cref="IncorrectShapeException">A constant that does not broadcast with tmp[0] (N-D).</exception>
        /// <exception cref="OverflowException">A Python-int lbnd or constant too large for the series' dtype.</exception>
        /// <exception cref="NotSupportedException">A str lbnd (NumPy evaluates the series AT it — a str-dtype error), or a str /
        ///     None constant.</exception>
        private static void Correct(PolyBasis basis, NDArray buf, long ws, long len, long[] md, bool oneD, NPTypeCode t,
            object kRaw, object lbndArg)
        {
            // np.ndim('a') is 0, so a str lbnd passes the scalar check and NumPy only fails here, at {p}val(lbnd, tmp).
            if (lbndArg is string)
                throw new NotSupportedException("a str lbnd makes NumPy evaluate the series at a str, which NumSharp has no dtype for");
            // A Python complex lbnd against a float16/float32 series is NumPy's complex64 loop: {p}val's value is a
            // complex64 (a 1-D series) or complex64 array (N-D), which NumSharp's single complex width carries as
            // complex128 — so the 1-D value is interpreted in NumPy's complex64 scalarmath here (PyVal1D), and the N-D
            // case, which can only end in NumPy's cast error, names complex64 in it.
            bool valC64 = lbndArg is System.Numerics.Complex && t is NPTypeCode.Half or NPTypeCode.Single;
            if (oneD)
            {
                // tmp[0] is a NumPy scalar and so is {p}val's result: scalarmath, then setitem's cast.
                byte* p = Ptr(buf) + ws * DirectILKernelGenerator.GetTypeSize(t);
                PolyNumber v;
                if (valC64)
                    v = PyVal1D(basis, PyScalar.Cplx((System.Numerics.Complex)lbndArg), p, len, t);
                else
                {
                    using (var tmp = MovedView(buf, ws, len, md))
                    using (var val = NDPolyEval.Val(basis, lbndArg, tmp, tensor: true))
                        v = PolyNumber.ScalarOf(val);
                }
                var r = PolyNumber.Binary(BinaryOp.Subtract, Constant(kRaw), v);
                StoreFirst(p, t, PolyNumber.Binary(BinaryOp.Add, PolyNumber.FromScalar(t, p), r));
                return;
            }

            NDArray arr;
            using (var tmp = MovedView(buf, ws, len, md))
                arr = NDPolyEval.Val(basis, lbndArg, tmp, tensor: true);
            try
            {
                var ki = Constant(kRaw);
                var r = PolyNumber.Binary(BinaryOp.Subtract, ki, PolyNumber.FromArray(arr));
                bool rC64 = PolyNumber.IsComplex64Loop(ki, valC64 ? NPTypeCode.Complex : arr.typecode, valC64);
                using (var row0 = RowView(buf, ws, md))
                    InPlace(BinaryOp.Add, row0, r, "add", rC64);
                if (r.IsNdArray && !ReferenceEquals(r.Array, arr)) r.Array.Dispose();
            }
            finally
            {
                arr.Dispose();
            }
        }

        /// <summary>
        ///     <c>{p}val(x, c)</c> of a 1-D series at a Python scalar x run as NumPy runs it: the U3 step trees for a weak x
        ///     (<see cref="PolySteps.RewriteForWeakX"/> — x-only subexpressions already folded with CPython's arithmetic)
        ///     interpreted with <see cref="PolyNumber"/> arithmetic, i.e. NumPy scalarmath on every coefficient. Used where the
        ///     evaluation kernel cannot reproduce NumPy: a Python complex x against a float16/float32 series, which NumPy
        ///     computes in COMPLEX64 (PolyNumber emulates complex64 scalars in float32; the kernel's complex is complex128).
        /// </summary>
        /// <param name="basis">The basis.</param><param name="x">The Python x.</param>
        /// <param name="c">Element 0 of the contiguous 1-D series.</param><param name="n">len(c) (≥ 1).</param>
        /// <param name="t">The series dtype.</param>
        /// <returns>NumPy's value (a NumPy scalar, complex64 for the case this serves).</returns>
        private static PolyNumber PyVal1D(PolyBasis basis, in PyScalar x, byte* c, long n, NPTypeCode t)
        {
            var prog = PolySteps.RewriteForWeakX(basis, x.Kind);
            int size = DirectILKernelGenerator.GetTypeSize(t);
            var xv = x;
            PolyNumber C(long i) => PolyNumber.FromScalar(t, c + i * size);
            PolyNumber Eval(PolyExpr e, PolyNumber ck, PolyNumber c0, PolyNumber c1, PolyNumber tmp, long row) => e switch
            {
                PolyLeaf l => l.S switch
                {
                    PolySym.Ck => ck,
                    PolySym.C0 => c0,
                    PolySym.C1 => c1,
                    PolySym.Tmp => tmp,
                    _ => throw new InvalidOperationException($"symbol {l.S} survives the weak-x rewrite"),
                },
                PolyWeak w => PolyNumber.FromPython(w.W.Value(row, xv)),
                PolyBin b => PolyNumber.Binary(b.Op, Eval(b.A, ck, c0, c1, tmp, row), Eval(b.B, ck, c0, c1, tmp, row)),
                _ => throw new InvalidOperationException("unknown step-tree node"),
            };

            if (prog.Horner)
            {
                // polyval: c0 = c[-1] + x*0; for i in range(2, len(c) + 1): c0 = c[-i] + c0*x
                var h = Eval(prog.HornerInit, C(n - 1), default, default, default, 0);
                for (long i = 2; i <= n; i++)
                    h = Eval(prog.HornerStep, C(n - i), h, default, default, 0);
                return h;
            }
            if (n == 1)
                return Eval(prog.FinalLen1, default, C(0), default, default, 0);
            if (n == 2)
                return Eval(prog.Final, default, C(0), C(1), default, 0);
            // Clenshaw: c0 = c[-2]; c1 = c[-1]; for i: tmp = c0; nd -= 1; c0 = <StepC0>; c1 = <StepC1 of the OLD c1>
            PolyNumber a0 = C(n - 2), a1 = C(n - 1);
            long nd = n;
            for (long i = 3; i <= n; i++)
            {
                var tmp = a0;
                nd--;
                a0 = Eval(prog.StepC0, C(n - i), default, a1, default, nd);
                a1 = Eval(prog.StepC1, default, default, a1, tmp, nd);
            }
            return Eval(prog.Final, default, a0, a1, default, 0);
        }

        /// <summary>
        ///     Stores a scalarmath result into tmp[0] as NumPy's <c>arr[0] = value</c> does: cast to the series dtype (a
        ///     complex into a real series keeps the real part — NumPy's ComplexWarning, which NumSharp does not emit); an
        ///     array result has no single value to store, and NumPy words that per target dtype: a complex element converts
        ///     through <c>complex(value)</c>, which refuses a non-0-d array with a TypeError.
        /// </summary>
        /// <param name="p">tmp[0]'s element.</param><param name="t">The series dtype.</param><param name="value">The value.</param>
        /// <exception cref="ValueError"><c>setting an array element with a sequence.</c> — an array of rank ≥ 1 into a real series.</exception>
        /// <exception cref="TypeError"><c>only 0-dimensional arrays can be converted to Python scalars</c> — into a complex series.</exception>
        private static void StoreFirst(byte* p, NPTypeCode t, in PolyNumber value)
        {
            if (value.IsNdArray)
            {
                if (t == NPTypeCode.Complex)
                    throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
                throw new ValueError("setting an array element with a sequence.");
            }
            value.WriteAs(t, p);
        }

        // ---------------------------------------------------------------------------------------------
        //  The n == 1 branch
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>np.all(c[0] == 0)</c> (NaN is nonzero, -0.0 is zero, a complex needs both parts zero).</summary>
        /// <param name="cm">The moved one-coefficient series.</param><param name="oneD">A 1-D series (c[0] is a scalar).</param>
        /// <returns>Whether every element of c[0] is zero (true for an empty row).</returns>
        private static bool FirstRowIsZero(NDArray cm, bool oneD)
        {
            if (oneD)
                return !PolyNumber.FromScalar(cm.typecode, Ptr(cm)).NotZero();
            using var row0 = cm[0];
            return np.count_nonzero(row0) == 0;
        }

        /// <summary>NumPy's <c>c[0] += k[i]</c>: scalarmath + setitem for a 1-D series, an in-place ufunc on the row otherwise.</summary>
        /// <param name="cm">The moved one-coefficient series.</param>
        /// <param name="kRaw">The constant as <see cref="IntegrationConstants"/> holds it (converted here, where NumPy uses it).</param>
        /// <param name="oneD">A 1-D series.</param>
        /// <exception cref="ValueError">An array constant into a real 1-D series, a stretching broadcast, or a ragged constant.</exception>
        /// <exception cref="TypeError">An array constant into a complex 1-D series.</exception>
        /// <exception cref="ArgumentException">A constant the series cannot absorb in place (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="IncorrectShapeException">A constant that does not broadcast with the row.</exception>
        /// <exception cref="NotSupportedException">A str / None constant.</exception>
        private static void AddToFirstRow(NDArray cm, object kRaw, bool oneD)
        {
            var ki = Constant(kRaw);
            if (oneD)
            {
                byte* p = Ptr(cm);
                StoreFirst(p, cm.typecode, PolyNumber.Binary(BinaryOp.Add, PolyNumber.FromScalar(cm.typecode, p), ki));
                return;
            }
            using var row0 = cm[0];
            InPlace(BinaryOp.Add, row0, ki, "add");
        }

        // ---------------------------------------------------------------------------------------------
        //  In-place arithmetic with NumPy's type and broadcast rules
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's in-place <c>target op= operand</c> (<c>np.op(target, operand, out=target)</c>): the loop dtype by
        ///     NEP 50 (a Python operand is weak), the same_kind cast back into the target checked FIRST, then the
        ///     broadcast with the target as the output — which may absorb the operand but never be stretched — then the
        ///     op in the loop dtype and the cast back.
        /// </summary>
        /// <param name="op">The operator.</param><param name="target">The array updated in place (rank ≥ 1).</param>
        /// <param name="operand">The other operand.</param><param name="ufunc">The ufunc name for the error text.</param>
        /// <param name="operandIsComplex64">The operand is an ARRAY NumPy holds as complex64 (NumSharp carries it as
        ///     complex128, #569) — e.g. <c>k - {p}val(1j, tmp)</c> of a float32 series. It only renames the dtype in the
        ///     cast error: every such loop meets a float16/float32 target, which cannot hold a complex, so it always raises.</param>
        /// <exception cref="ArgumentException">NumPy's UFuncTypeError: <c>Cannot cast ufunc '{name}' output from dtype(...)
        ///     to dtype(...) with casting rule 'same_kind'</c>.</exception>
        /// <exception cref="IncorrectShapeException">Shapes that do not broadcast (NumPy lists all three operands).</exception>
        /// <exception cref="ValueError">A broadcast that would stretch the target.</exception>
        private static void InPlace(BinaryOp op, NDArray target, in PolyNumber operand, string ufunc, bool operandIsComplex64 = false)
        {
            var t = target.typecode;
            var loop = operand.IsPython ? PolyNumber.WeakPromote(t, operand.Py) : PolyTyping.Promote(t, operand.Dtype);
            if (loop != t && !NDIterCasting.CanCast(loop, t, NPY_CASTING.NPY_SAME_KIND_CASTING))
            {
                // A Python complex meeting a float16/float32 array, or a complex64 array meeting a dtype complex64
                // absorbs, is NumPy's complex64 loop (NumSharp computes it as complex128, #569); the text names NumPy's
                // dtype. The array test asks "target op complex64-array" with the target as the scalar-sized side —
                // NEP 50 promotion is symmetric, so the side does not matter.
                bool c64 = PolyNumber.IsComplex64Loop(operand, t, bIsComplex64: false)
                           || (operandIsComplex64 && PolyNumber.IsComplex64Loop(PolyNumber.FromArray(target), NPTypeCode.Complex, bIsComplex64: true));
                string loopName = loop == NPTypeCode.Complex && c64 ? "complex64" : loop.AsNumpyDtypeName();
                throw new ArgumentException($"Cannot cast ufunc '{ufunc}' output from dtype('{loopName}') to " +
                                            $"dtype('{t.AsNumpyDtypeName()}') with casting rule 'same_kind'");
            }
            if (operand.IsNdArray)
                CheckInPlaceBroadcast(target.Shape, operand.Array.Shape);

            var r = PolyNumber.Binary(op, PolyNumber.FromArray(target), operand);
            NDIter.Copy(target, r.Array);
            if (!ReferenceEquals(r.Array, target)) r.Array.Dispose();
        }

        /// <summary>
        ///     The House route's <c>c *= scl</c> over a buffer window: NumPy's in-place ufunc. For a real operand — an array
        ///     scl (one factor per column / row / coefficient, derivatives only), or a scalar-sized promotion the kernel has
        ///     no widened loop for — the house <c>np.multiply(win, scl, out: win, dtype: loop)</c> runs it as one pass in
        ///     NumPy's loop dtype, exactly NumPy's <c>multiply(c, scl, out=c)</c>; the general <see cref="InPlace"/> would
        ///     first materialize the whole window in the loop dtype (a float32 window's float64 copy, then the float64
        ///     product) — two series-sized temporaries per order, 8x the in-place pass's cost on a (11, 100K) float32 series.
        ///     The loop is PINNED with <c>dtype:</c> rather than left to the house ufunc's own resolution, which computes a
        ///     float32 window times a 0-d float64 operand in float32 (one ULP off NumPy's float64 loop in the corpus). A
        ///     complex, decimal or char operand or series keeps <see cref="InPlace"/> (NumPy's complex product forms and the
        ///     NumSharp-only dtypes are pinned there).
        /// </summary>
        /// <param name="win">The window (a C-contiguous view of the calculus buffer, the series' dtype).</param>
        /// <param name="scl">The scale: a NumPy scalar, a 0-d array, or an array (ClassifyScale has already refused a scale
        ///     the series cannot absorb with NumPy's cast text).</param>
        /// <exception cref="IncorrectShapeException">An array scl that does not broadcast with the window (NumPy's text).</exception>
        /// <exception cref="ValueError">An array scl whose broadcast would stretch the window.</exception>
        private static void ScaleInPlace(NDArray win, in PolyNumber scl)
        {
            var t = win.typecode;
            bool realSeries = t is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;
            bool realOperand = !scl.IsPython && !scl.IsComplex64 && scl.Dtype is not (NPTypeCode.Complex or NPTypeCode.Decimal or NPTypeCode.Char);
            if (!realSeries || !realOperand)
            {
                InPlace(BinaryOp.Multiply, win, scl, "multiply");
                return;
            }
            // NumPy validates the broadcast (target as output) before the loop runs; same texts as InPlace.
            if (scl.IsNdArray)
                CheckInPlaceBroadcast(win.Shape, scl.Array.Shape);
            // A NumPy scalar enters the ufunc as a strong 0-d operand of its own dtype, built here and released after the
            // pass (an array scl is the caller's and is used as it is); the loop is NumPy's promotion of the two (both
            // strong), which ClassifyScale already proved casts back into the series same_kind.
            using NDArray built = scl.Kind == PolyNumberKind.Array ? null : scl.AsArrayOf(scl.Dtype);
            NPTypeCode loop = PolyTyping.Promote(t, scl.Dtype);
            np.multiply(win, built ?? scl.Array, @out: win, dtype: loop);
        }

        /// <summary>
        ///     NpyIter's broadcast of <c>(target, operand, out=target)</c>: incompatible dims raise NumPy's text listing all
        ///     three shapes; a broadcast shape other than the target's raises the non-broadcastable-output text.
        /// </summary>
        /// <param name="target">The output (and first input).</param><param name="operand">The other input.</param>
        /// <exception cref="IncorrectShapeException"><c>operands could not be broadcast together with shapes (a) (b) (a) </c>.</exception>
        /// <exception cref="ValueError"><c>non-broadcastable output operand with shape (a) doesn't match the broadcast shape (b)</c>.</exception>
        private static void CheckInPlaceBroadcast(Shape target, Shape operand)
        {
            var a = target.dimensions ?? Array.Empty<long>();
            var b = operand.dimensions ?? Array.Empty<long>();
            int nd = Math.Max(a.Length, b.Length);
            var r = new long[nd];
            bool same = nd == a.Length;
            for (int d = 0; d < nd; d++)
            {
                long da = d < nd - a.Length ? 1 : a[d - (nd - a.Length)];
                long db = d < nd - b.Length ? 1 : b[d - (nd - b.Length)];
                if (da != db && da != 1 && db != 1)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with shapes {target.ToPythonTuple()} {operand.ToPythonTuple()} {target.ToPythonTuple()} ");
                r[d] = da == 1 ? db : da;
                if (d >= nd - a.Length && r[d] != da) same = false;
            }
            if (!same)
                throw new ValueError($"non-broadcastable output operand with shape {target.ToPythonTuple()} " +
                                     $"doesn't match the broadcast shape {new Shape(r).ToPythonTuple()}");
        }

        // ---------------------------------------------------------------------------------------------
        //  Argument conversion
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array(c, ndmin=1)</c> without the copy (the kernel only reads the source): an NDArray is used as is (a
        ///     0-d one as a one-element view), anything else is converted by the U1 rules (C# primitives are Python
        ///     scalars, a typed C# array is an ndarray, object[]/IList a Python list).
        /// </summary>
        /// <param name="c">The argument.</param>
        /// <returns>An array of rank ≥ 1.</returns>
        /// <exception cref="NotSupportedException">null (an object array in NumPy) or a string (a str array).</exception>
        private static NDArray Coefficients(object c)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            // A str / object series is refused as soon as it exists, as before the conversion learned to describe one
            // (U4's argument checks follow the series in this order); the conversion's refusal names the item that made it.
            if (v.NonNumeric)
                throw v.Refusal ?? new NotSupportedException("a str series makes NumPy build a str array, a dtype NumSharp does not have");
            var a = v.Source;
            return a.ndim == 0 ? np.expand_dims(a, 0) : a;
        }

        /// <summary>
        ///     NumPy's <c>if not np.iterable(k): k = [k]</c> then <c>list(k)</c>. An array (an NDArray, a typed C# array,
        ///     a Memory&lt;T&gt;) yields NumPy scalars (1-D) or row views (N-D) as <see cref="PolyNumber"/>s — a 0-d array
        ///     is not iterable and is one constant; a Python tuple / list (<see cref="PolySequence.IsSequence"/>) yields its
        ///     items and a str its characters, RAW: Python converts an item only when its order uses it
        ///     (<see cref="Constant"/>), so a bad item is reported after the argument checks and after the lbnd
        ///     evaluation of its order, as in NumPy; anything else is one raw constant.
        /// </summary>
        /// <param name="k">The argument (null: NumPy's default empty list).</param>
        /// <returns>The constants: boxed <see cref="PolyNumber"/>s (array items) or the raw Python items.</returns>
        private static List<object> IntegrationConstants(object k)
        {
            var list = new List<object>();
            switch (k)
            {
                case null:
                    return list;
                case string s:
                    // A str is iterable: its one-character strs are the constants — refused only when an order uses one.
                    foreach (char ch in s)
                        list.Add(ch.ToString());
                    return list;
                case NDArray nd:
                    AddArrayItems(list, nd);
                    return list;
            }
            if (PolySequence.IsArrayLike(k))
                AddArrayItems(list, np.asanyarray(k));
            else if (PolySequence.IsSequence(k))
                list.AddRange(PolySequence.Items(k));
            else
                list.Add(k);   // not iterable: k = [k]
            return list;
        }

        /// <summary>
        ///     One constant as the operand of its order's arithmetic: an array item as collected, a raw Python item
        ///     converted by the house mapping (a nested tuple / list by np.array's coercion).
        /// </summary>
        /// <param name="kRaw">An entry of <see cref="IntegrationConstants"/>.</param>
        /// <returns>The operand.</returns>
        /// <exception cref="ValueError">A ragged tuple / list.</exception>
        /// <exception cref="NotSupportedException">A str / None item (NumPy's str / object operand).</exception>
        private static PolyNumber Constant(object kRaw) => kRaw is PolyNumber p ? p : PolyNumber.FromObject(kRaw);

        /// <summary><c>list(nd)</c>: a 0-d array is not iterable (one constant); rank 1 yields NumPy scalars, rank ≥ 2 row views.</summary>
        /// <param name="list">The destination (items boxed as <see cref="PolyNumber"/>s).</param><param name="nd">The array.</param>
        private static void AddArrayItems(List<object> list, NDArray nd)
        {
            if (nd.ndim == 0)
            {
                list.Add(PolyNumber.FromArray(nd));
                return;
            }
            var s = nd.Shape;
            long len = s.dimensions[0];
            int size = nd.dtypesize;
            for (long i = 0; i < len; i++)
            {
                if (nd.ndim == 1)
                {
                    list.Add(PolyNumber.FromScalar(nd.typecode, (byte*)nd.Storage.Address + (s.offset + i * s.strides[0]) * size));
                    continue;
                }
                var dims = s.dimensions.AsSpan(1).ToArray();
                var strides = s.strides.AsSpan(1).ToArray();
                list.Add(PolyNumber.FromArray(View(nd, dims, strides, s.offset + i * s.strides[0])));
            }
        }

        /// <summary>
        ///     <c>np.ndim(x)</c> for the lbnd/scl checks: 0 for null (the default), a str and every scalar; an NDArray's or
        ///     typed C# array's rank; 1 for a Memory&lt;T&gt;; a Python tuple / list's <c>asarray(x).ndim</c> — which raises
        ///     np.array's inhomogeneous ValueError for a ragged one, exactly where NumPy's check does.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns>The rank (only "nonzero" matters).</returns>
        /// <exception cref="ValueError">A ragged tuple / list.</exception>
        private static int NDim(object x)
        {
            switch (x)
            {
                case null:
                case string:
                    return 0;
                case NDArray nd:
                    return nd.ndim;
            }
            if (PolySequence.IsSequence(x))
                return PolySequence.NDim(x);
            if (x is Array arr)
                return arr.Rank;   // a typed C# array (IsSequence took every other array)
            return PolySequence.IsArrayLike(x) ? 1 : 0;
        }

        /// <summary><c>normalize_axis_index(axis, ndim)</c>.</summary>
        /// <param name="axis">The axis.</param><param name="ndim">The rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><c>axis {axis} is out of bounds for array of dimension {ndim}</c>.</exception>
        private static int NormalizeAxis(int axis, int ndim)
        {
            if (axis < -ndim || axis >= ndim)
                throw new AxisError(axis, ndim);
            return axis < 0 ? axis + ndim : axis;
        }

        /// <summary>
        ///     How <c>c *= scl</c> runs, raising NumPy's cast error up front. The loop dtype is NEP 50's (a Python scl is
        ///     weak and adopts the series dtype; a strong one — a NumPy scalar, a 0-d array — may promote it). A scalar-sized
        ///     scl the kernel folds in: in the series dtype, or — for a promoting strong scalar on a float16 / float32 series
        ///     (<see cref="PolyCalcKey.HasWidenedScale"/>) — in the wider loop, the series widened, multiplied and cast back per
        ///     element; either way scl is converted exactly as NumPy converts the operand entering its loop. An array scl
        ///     (derivatives only) and the remaining promotions (decimal) run as NumPy's in-place ufunc per order.
        /// </summary>
        /// <param name="scl">The argument (null: NumPy's Python int 1).</param>
        /// <param name="t">The series dtype.</param>
        /// <param name="num">The scale as a <see cref="PolyNumber"/>.</param>
        /// <param name="raw">16 bytes that receive the converted scalar (Fused route).</param>
        /// <param name="scaleLoop">The loop dtype the Fused kernel multiplies in (<paramref name="t"/>, or the wider loop);
        ///     <see cref="NPTypeCode.Empty"/> for the House route.</param>
        /// <returns>The route.</returns>
        /// <exception cref="ArgumentException">A scl whose loop dtype cannot be cast back into the series (NumPy's
        ///     UFuncTypeError text).</exception>
        /// <exception cref="NotSupportedException">A string scl.</exception>
        /// <exception cref="ValueError">A ragged list / tuple scl.</exception>
        /// <exception cref="OverflowException">A Python int scl too large for the series' dtype.</exception>
        private static ScaleRoute ClassifyScale(object scl, NPTypeCode t, out PolyNumber num, byte* raw, out NPTypeCode scaleLoop)
        {
            num = scl is null ? PolyNumber.FromPython(s_one) : PolyNumber.FromObject(scl);
            var loop = num.IsPython ? PolyNumber.WeakPromote(t, num.Py) : PolyTyping.Promote(t, num.Dtype);
            if (loop != t && !NDIterCasting.CanCast(loop, t, NPY_CASTING.NPY_SAME_KIND_CASTING))
            {
                string loopName = loop == NPTypeCode.Complex && PolyNumber.IsComplex64Loop(num, t, bIsComplex64: false)
                    ? "complex64" : loop.AsNumpyDtypeName();
                throw new ArgumentException($"Cannot cast ufunc 'multiply' output from dtype('{loopName}') to " +
                                            $"dtype('{t.AsNumpyDtypeName()}') with casting rule 'same_kind'");
            }
            if (!num.IsNdArray && (loop == t || PolyCalcKey.HasWidenedScale(t, loop)))
            {
                num.WriteAs(loop, raw);
                scaleLoop = loop;
                return ScaleRoute.Fused;
            }
            scaleLoop = NPTypeCode.Empty;
            return ScaleRoute.House;
        }

        // ---------------------------------------------------------------------------------------------
        //  Layout
        // ---------------------------------------------------------------------------------------------

        /// <summary><c>np.array(c, copy=True)</c> in dtype <paramref name="t"/>: NumPy's order='K' copy (the conversion of an
        ///     integer series — <c>c + 0.0</c> / <c>astype(np.double)</c> — keeps that layout too).</summary>
        /// <param name="src">The series.</param><param name="t">The dtype.</param>
        /// <returns>A fresh array.</returns>
        private static NDArray KCopyAs(NDArray src, NPTypeCode t)
        {
            var r = new NDArray(t, NDPolyEval.LikeKeepOrder(src.Shape), false);
            NDIter.Copy(r, src);
            return r;
        }

        /// <summary>
        ///     Der's <c>cnt >= n</c> result, <c>c[:1] * 0</c> of NumPy's moved copy: the converted first coefficients times the
        ///     Python int 0 (an array op, so NaN/inf give NaN and a complex takes simd_cmul), laid out as NumPy's ufunc
        ///     allocates it — NpyIter's KEEPORDER vote over the <c>c[:1]</c> view of the K-order copy — and moved back,
        ///     except by hermeder, which returns it in the moved order.
        /// </summary>
        /// <param name="src">The series.</param><param name="t">The coefficient dtype.</param>
        /// <param name="iaxis">The series axis.</param><param name="moveBack">Apply NumPy's final moveaxis.</param>
        /// <returns>The result.</returns>
        private static NDArray FirstRowTimesZero(NDArray src, NPTypeCode t, int iaxis, bool moveBack)
        {
            var copyShape = NDPolyEval.LikeKeepOrder(src.Shape);
            Moved(copyShape, iaxis, out var cd, out var cs);
            Moved(src.Shape, iaxis, out var sd, out var ss);
            long first = Math.Min(1, cd[0]);
            cd[0] = first;
            int[] perm = Shape.MultiSortedStridePerm(new[] { new Shape(cd, cs) }, cd.Length);
            var r = new NDArray(t, new Shape((long[])cd.Clone(), Shape.StridesForPerm(cd, perm)), false);
            if (r.size > 0)
            {
                sd[0] = first;
                using (var head = View(src, sd, ss, src.Shape.offset))
                    NDIter.Copy(r, head);
                InPlace(BinaryOp.Multiply, r, PolyNumber.FromPython(PyScalar.Int(0)), "multiply");
            }
            // np.moveaxis returns a VIEW even for axis 0 (OWNDATA false); hermeder returns the ufunc output itself.
            return moveBack ? MoveAxisView(r, 0, iaxis) : r;
        }

        /// <summary>
        ///     NumPy's <c>np.moveaxis(a, from, to)</c>: always a NEW view (even when the axes do not move — NumPy never hands
        ///     back the same object, so the result's OWNDATA is false), with the axis permutation applied to dims and strides.
        /// </summary>
        /// <param name="a">The array.</param><param name="from">The axis to move.</param><param name="to">Its destination.</param>
        /// <returns>The view.</returns>
        private static NDArray MoveAxisView(NDArray a, int from, int to)
        {
            var s = a.Shape;
            int nd = s.NDim;
            var order = new List<int>(nd);
            for (int d = 0; d < nd; d++)
                if (d != from) order.Add(d);
            order.Insert(to, from);
            var dims = new long[nd];
            var strides = new long[nd];
            for (int d = 0; d < nd; d++)
            {
                dims[d] = s.dimensions[order[d]];
                strides[d] = s.strides[order[d]];
            }
            return View(a, dims, strides, s.offset);
        }

        /// <summary>The dims and element strides of <paramref name="s"/> with axis <paramref name="iaxis"/> moved first
        ///     (NumPy's <c>np.moveaxis(c, iaxis, 0)</c>).</summary>
        /// <param name="s">The shape.</param><param name="iaxis">The axis.</param>
        /// <param name="dims">The moved dims.</param><param name="strides">The moved element strides.</param>
        private static void Moved(Shape s, int iaxis, out long[] dims, out long[] strides)
        {
            int nd = s.NDim;
            dims = new long[nd];
            strides = new long[nd];
            dims[0] = s.dimensions[iaxis];
            strides[0] = s.strides[iaxis];
            for (int d = 0, k = 1; d < nd; d++)
            {
                if (d == iaxis) continue;
                dims[k] = s.dimensions[d];
                strides[k] = s.strides[d];
                k++;
            }
        }

        /// <summary>The column count: the product of the moved dims after the series axis.</summary>
        /// <param name="md">Moved dims.</param><returns>The count.</returns>
        private static long Cols(long[] md)
        {
            long c = 1;
            for (int d = 1; d < md.Length; d++) c *= md[d];
            return c;
        }

        /// <summary>The buffer dims: <paramref name="rows"/> then the other axes (moved order).</summary>
        /// <param name="md">Moved dims.</param><param name="rows">The row count.</param><returns>The dims.</returns>
        private static long[] BufferDims(long[] md, long rows)
        {
            var d = (long[])md.Clone();
            d[0] = rows;
            return d;
        }

        /// <summary>The C-order element strides of the moved buffer dims (row stride first).</summary>
        /// <param name="md">Moved dims (any row count).</param><returns>The strides.</returns>
        private static long[] BufferStrides(long[] md)
        {
            var s = new long[md.Length];
            long acc = 1;
            for (int d = md.Length - 1; d >= 0; d--)
            {
                s[d] = acc;
                acc *= Math.Max(md[d], 1);
            }
            return s;
        }

        /// <summary>
        ///     Columns per kernel block: enough that the block's rows stay cache-resident (~512 KB for the whole block)
        ///     through the recurrence, never fewer than 16 (vector loop efficiency) nor more than 4096.
        /// </summary>
        /// <param name="rows">Rows the block spans.</param><param name="size">Element size.</param><param name="cols">Columns.</param>
        /// <returns>The block width (≥ 1).</returns>
        private static long Block(long rows, int size, long cols)
        {
            if (cols <= 1) return 1;
            long b = (512L * 1024) / Math.Max(1L, rows * size);
            b = Math.Clamp(b, 16, 4096) & ~7L;
            return Math.Min(b, cols);
        }

        /// <summary>
        ///     Whether the kernel can load the caller's series directly: every row's columns (C order of the other axes)
        ///     must sit at ONE constant stride — true for every 1-D series, every 2-D series, and N-D series whose other
        ///     axes are C-contiguous among themselves.
        /// </summary>
        /// <param name="src">The series.</param><param name="md">Moved dims.</param><param name="ms">Moved element strides.</param>
        /// <param name="p">Element (row 0, column 0).</param>
        /// <param name="rowBytes">Byte stride along the series axis.</param><param name="colBytes">Byte stride between columns.</param>
        /// <returns>True when the (row, column) walk is exact.</returns>
        private static bool TryDirectSource(NDArray src, long[] md, long[] ms, out byte* p, out long rowBytes, out long colBytes)
        {
            int size = src.dtypesize;
            p = Ptr(src);
            rowBytes = ms[0] * size;
            long col = 1, expect = 0;
            bool have = false;
            for (int d = md.Length - 1; d >= 1; d--)
            {
                if (md[d] == 1) continue;
                if (!have)
                {
                    col = ms[d];
                    have = true;
                }
                else if (ms[d] != expect)
                {
                    colBytes = 0;
                    return false;
                }
                expect = ms[d] * md[d];
            }
            colBytes = col * size;
            return true;
        }

        /// <summary>The address of an array's element 0.</summary>
        /// <param name="a">The array.</param><returns>The pointer.</returns>
        private static byte* Ptr(NDArray a) => (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;

        /// <summary>A view of <paramref name="a"/>'s storage with explicit dims, element strides and offset.</summary>
        /// <param name="a">The array.</param><param name="dims">Dims.</param><param name="strides">Element strides.</param>
        /// <param name="offset">Element offset into the storage.</param>
        /// <returns>The view (writes go through). Internal: the Vandermonde family (<see cref="NDPolyVander"/>) builds NumPy's
        ///     moveaxis / reshape result views with it too.</returns>
        internal static NDArray View(NDArray a, long[] dims, long[] strides, long offset)
        {
            var shape = new Shape(dims, strides, offset, a.Shape.bufferSize);
            return new NDArray(a.Storage.Alias(ref shape), a.TensorEngine, skipEngineResolve: true);
        }

        /// <summary>Buffer rows [start, start + rows) in the moved order (series axis first).</summary>
        /// <param name="buf">The buffer.</param><param name="start">First row.</param><param name="rows">Row count.</param>
        /// <param name="md">Moved dims (the row count is replaced).</param>
        /// <returns>The view.</returns>
        private static NDArray MovedView(NDArray buf, long start, long rows, long[] md)
        {
            var dims = BufferDims(md, rows);
            var strides = BufferStrides(buf.Shape.dimensions);
            return View(buf, dims, strides, buf.Shape.offset + start * strides[0]);
        }

        /// <summary>Buffer rows [start, start + rows) with the series axis moved back to <paramref name="iaxis"/> —
        ///     NumPy's <c>np.moveaxis(c, 0, iaxis)</c> of its C-order result.</summary>
        /// <param name="buf">The buffer.</param><param name="start">First row.</param><param name="rows">Row count.</param>
        /// <param name="md">Moved dims.</param><param name="iaxis">The series axis.</param>
        /// <returns>The view.</returns>
        private static NDArray MovedBackView(NDArray buf, long start, long rows, long[] md, int iaxis)
        {
            var bs = BufferStrides(buf.Shape.dimensions);
            int nd = md.Length;
            var dims = new long[nd];
            var strides = new long[nd];
            dims[iaxis] = rows;
            strides[iaxis] = bs[0];
            for (int d = 0, k = 1; d < nd; d++)
            {
                if (d == iaxis) continue;
                dims[d] = md[k];
                strides[d] = bs[k];
                k++;
            }
            return View(buf, dims, strides, buf.Shape.offset + start * bs[0]);
        }

        /// <summary>Buffer row <paramref name="row"/> as an array of the other axes (N-D series only).</summary>
        /// <param name="buf">The buffer.</param><param name="row">The row.</param><param name="md">Moved dims.</param>
        /// <returns>The view.</returns>
        private static NDArray RowView(NDArray buf, long row, long[] md)
        {
            var bs = BufferStrides(buf.Shape.dimensions);
            return View(buf, md.AsSpan(1).ToArray(), bs.AsSpan(1).ToArray(), buf.Shape.offset + row * bs[0]);
        }
    }
}
