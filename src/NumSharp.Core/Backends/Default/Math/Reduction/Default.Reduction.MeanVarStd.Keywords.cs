using System;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

// =============================================================================
// Default.Reduction.MeanVarStd.Keywords.cs — np.mean with NumPy's where=, np.var / np.std with where=, mean= and
// correction= (np-layer dtype plan, P10 item 1, decision D10; plan docs/plans/api-parameter-parity.md, items B3 and B4)
// =============================================================================
//
// numpy/_core/_methods.py, statement for statement, each statement through the engine member NumPy's runs through:
//
//   _count_reduce_items(arr, axis, keepdims, where): with a mask, umr_sum(broadcast_to(where, arr.shape), axis, intp, None,
//       keepdims) — so broadcast_to's errors come first, then ufunc.reduce's axis rule (a 0-d input takes axis 0 / -1), and
//       a non-bool mask is counted by its VALUES here (its dtype error is raised by the next sum's prologue);
//   _mean: ret = umr_sum(arr, axis, dtype, out, keepdims, where=where) (float64 for an integer / bool input, float32 for
//       float16, whose result is float16), then true_divide(ret, rcount, out=ret, casting='unsafe') for an array (one
//       count per output) or ret.dtype.type(ret / rcount) for a scalar;
//   _var: arrmean = mean= as given, or umr_sum(arr, axis, dtype, keepdims=True, where=where) / rcount; x = subtract(arr,
//       arrmean); x squared in NumPy's three forms (square for a float / integer input; real² + imag² into x.real for a
//       complex x; multiply(x, conjugate(x)).real otherwise); ret = umr_sum(x, axis, dtype, out, keepdims, where=where);
//       ret / max(rcount - ddof, 0);
//   _std: the square root of _var's result, in place (sqrt(out=ret) for an array, ret.dtype.type(sqrt(ret)) for a scalar).
//
// np.var / np.std's correction= is fromnumeric's own statement, run first: with correction given, a nonzero ddof raises
// "ddof and correction can't be provided simultaneously.", else ddof = correction.
//
// A call without where= / mean= keeps the engine members' own routes (MeanLikeNumPy, VarStdNumPy: the fused deviation
// passes and the exact sums), whose results these statements reproduce; these routes favour exactness over speed.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Whether <paramref name="where"/> is the C# <c>true</c> literal, NumPy's <c>where=True</c>: a 0-d bool array the
        ///     implicit conversion marked (<see cref="NDArray.WasPythonBool"/>). <c>_mean</c> / <c>_var</c> test
        ///     <c>where is True</c>, so <c>np.True_</c> (an explicit 0-d true array) takes the masked statements instead,
        ///     which give the same values but read the axis by ufunc.reduce's rule.
        /// </summary>
        /// <param name="where">The caller's mask, or null.</param>
        /// <returns>True for the marked literal.</returns>
        internal static bool IsPythonTrueMask(NDArray where)
            => where is not null && where.WasPythonBool && where.ndim == 0 && where.GetTypeCode == NPTypeCode.Boolean
               && where.GetAtIndex<bool>(0);

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c>'s <c>correction=</c> rule (<c>numpy/_core/fromnumeric.py</c>), run before anything
        ///     else: with <paramref name="correction"/> given, a nonzero <paramref name="ddof"/> is refused, otherwise it IS the
        ///     delta degrees of freedom.
        /// </summary>
        /// <param name="ddof">The caller's <c>ddof</c> (null is 0).</param>
        /// <param name="correction">The caller's <c>correction</c>, or null for none.</param>
        /// <returns>The delta degrees of freedom.</returns>
        /// <exception cref="ValueError">Both given and <paramref name="ddof"/> is not 0: <c>ddof and correction can't be
        ///     provided simultaneously.</c></exception>
        internal static int DdofWithCorrection(int? ddof, int? correction)
        {
            if (correction is null)
                return ddof ?? 0;
            if ((ddof ?? 0) != 0)
                throw new ValueError("ddof and correction can't be provided simultaneously.");
            return correction.Value;
        }

        /// <summary>
        ///     NumPy's <c>_count_reduce_items(arr, axis, keepdims, where)</c> for a mask: the mask broadcast to the array's
        ///     shape and summed in the int64 loop, one count per output (a 0-d count for a reduction over every axis without
        ///     keepdims).
        /// </summary>
        /// <param name="arr">The reduced array.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Whether the counts keep the reduced axes with length 1.</param>
        /// <param name="where">The mask (not null).</param>
        /// <returns>A fresh int64 array of the counts.</returns>
        /// <exception cref="IncorrectShapeException">The mask does not broadcast to the array's shape (np.broadcast_to's
        ///     text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds (ufunc.reduce's rule).</exception>
        private NDArray CountReduceItems(NDArray arr, int? axis, bool keepdims, NDArray where)
        {
            using var stretched = NDArray.broadcast_to(where, arr.Shape);
            // umr_sum(..., nt.intp, None, keepdims): a non-bool mask is summed by value, as NumPy's is.
            return ReduceAdd(stretched, axis, keepdims, np.int64);
        }

        /// <summary>
        ///     <c>np.mean</c> with a <c>where=</c> mask: <c>_methods._mean</c>'s statements (see the file header).
        /// </summary>
        /// <param name="arr">The array to average.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="dtype">The sum's loop (<c>dtype=</c>), or null for <c>_mean</c>'s default.</param>
        /// <param name="out">The out, or null.</param>
        /// <param name="where">The mask (not null; a <c>where=True</c> literal never reaches here).</param>
        /// <returns><paramref name="out"/> itself, or the fresh mean (NumPy's read-only scalar for a reduction over every axis
        ///     without keepdims).</returns>
        /// <exception cref="IncorrectShapeException">The mask does not broadcast to the array (raised first), or the out does
        ///     not broadcast.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">The mask is not bool (raised by the sum, after the count).</exception>
        /// <exception cref="ValueError">A read-only or misshapen out; a <c>StringDType</c> input (its add has no identity, so
        ///     the masked sum needs an initial value).</exception>
        /// <exception cref="TypeError">NumPy's refusal for a fixed-width text input.</exception>
        /// <exception cref="DivideByZeroException">A decimal input (NumSharp's dtype, no NaN) with an output whose mask
        ///     selects nothing.</exception>
        [NDScoped]
        private NDArray MeanWhereNumPy(NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out, NDArray @where)
        {
            // rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where): broadcast_to's errors, then the axis.
            var rcount = CountReduceItems(arr, axis, keepdims, @where);

            // `if dtype is None`: float64 for an integer / bool (NumSharp's char) input, float32 for float16 with a float16
            // result (is_float16_result); every other input sums in its own loop.
            NPTypeCode inT = arr.GetTypeCode;
            bool float16Result = dtype is null && inT == NPTypeCode.Half;
            DType loop = dtype ?? (inT.IsInteger() || inT == NPTypeCode.Boolean || inT == NPTypeCode.Char ? np.float64
                : float16Result ? np.float32 : null);

            // ret = umr_sum(arr, axis, dtype, out, keepdims, where=where): the keyword sum (its prologue raises the mask's
            // dtype error, a read-only out, and the text refusals).
            var ret = AddMulNumPy(arr, axis, keepdims, loop, @out, null, @where, multiply: false);
            if (@out is not null)
            {
                // isinstance(ret, ndarray): true_divide(out, rcount, out=out, casting='unsafe') — the division runs on the
                // sum as out's dtype holds it, one count per output.
                DivideByCountLikeNumPy(@out, rcount);
                return @out;
            }

            if (ret.ndim == 0)
            {
                // The scalar branch: ret.dtype.type(ret / rcount), or arr.dtype.type(ret / rcount) for a float16 result;
                // rcount is one np.intp here (every axis reduced without keepdims). A decimal output with no selected
                // element divides 0 by 0 in decimal, which raises (decimal has no NaN).
                return FinishMeanLikeNumPy(ret, rcount.GetAtIndex<long>(0), ret.GetTypeCode, float16Result);
            }

            // true_divide(ret, rcount, out=ret, casting='unsafe'), then arr.dtype.type(ret) for a float16 result.
            DivideByCountLikeNumPy(ret, rcount);
            return float16Result ? ret.astype(np.float16, copy: true) : ret;
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> with <c>where=</c> or <c>mean=</c>: <c>_methods._var</c> / <c>_std</c>'s statements
        ///     (see the file header).
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="ddof">The delta degrees of freedom (after the <c>correction=</c> rule).</param>
        /// <param name="dtype">The loop of both sums (<c>dtype=</c>), or null: float64 for an integer / bool input, else the
        ///     resolution of each sum.</param>
        /// <param name="out">The out (NumPy hands it to the last sum only), or null.</param>
        /// <param name="where">The mask, or null (a <c>where=True</c> literal never reaches here).</param>
        /// <param name="mean">The <c>mean=</c> to subtract as given (a weak C# literal or an array), or null to compute it.</param>
        /// <param name="takeSqrt">True for <c>np.std</c>.</param>
        /// <returns><paramref name="out"/> itself, or the fresh result (a read-only 0-d scalar when it has no
        ///     dimensions).</returns>
        /// <exception cref="IncorrectShapeException">The mask does not broadcast to the array, or the mean does not broadcast
        ///     against it (the subtraction's text), or the out does not broadcast.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">The mask is not bool; std into an integer or bool result with dimensions
        ///     (NumPy's sqrt output-cast error).</exception>
        /// <exception cref="TypeError">A boolean subtraction (a bool input in a bool loop, or a bool mean of a bool input), or
        ///     NumPy's text refusals.</exception>
        /// <exception cref="ValueError">A read-only or misshapen out.</exception>
        [NDScoped]
        private NDArray VarStdKeywordsNumPy(NDArray arr, int? axis, bool keepdims, int ddof, DType dtype, NDArray @out,
            NDArray @where, NDArray mean, bool takeSqrt)
        {
            // rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where): without a mask the axis is normalized
            // against the operand (a 0-d one refuses every axis); with one, the mask's counts.
            NDArray rcounts = null;
            long rcount = 0;
            if (@where is null)
                rcount = ReduceItemCount(arr, axis);
            else
                rcounts = CountReduceItems(arr, axis, keepdims, @where);

            // `if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)): dtype = f8`.
            NPTypeCode inT = arr.GetTypeCode;
            DType loop = dtype ?? (inT.IsInteger() || inT == NPTypeCode.Boolean || inT == NPTypeCode.Char ? np.float64 : null);

            NDArray arrmean = mean;
            if (arrmean is null)
            {
                // arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where), divided by the counts in place (rcount
                // reshaped to the kept-dims mean when it is an array). A 0-d sum is NumPy's scalar: divided on a copy.
                arrmean = ReduceAdd(arr, axis, keepdims: true, loop, null, null, @where);
                if (!arrmean.Shape.IsWriteable)
                    arrmean = arrmean.copy();
                if (rcounts is null)
                    DivideByCountLikeNumPy(arrmean, rcount);
                else
                    DivideByCountLikeNumPy(arrmean, rcounts.ndim == 0 ? rcounts : rcounts.reshape(arrmean.Shape));
            }

            // x = subtract(arr, arrmean, out=...): NumPy's ufunc result, laid out over both operands as its iterator lays it
            // out (the next sum walks x in that order, so a different layout would add in a different order).
            var x = Subtract(arr, arrmean);
            x = InUfuncLayout(x, arr, arrmean);

            // The square in NumPy's three forms, by the INPUT's kind.
            NDArray squared;
            if (inT.IsInteger() || inT == NPTypeCode.Char || inT.IsFloatingPoint())
            {
                // `issubclass(arr.dtype.type, (nt.floating, nt.integer))`: um.square(x, out=x).
                Square(x, null, x);
                squared = x;
            }
            else if (x.GetTypeCode == NPTypeCode.Complex)
            {
                // _complex_to_float: xv = x.view((float64, (2,))); square(xv, out=xv); x = add(xv[..., 0], xv[..., 1],
                // out=x.real) — the parts squared in place, their sum written over the real parts; the next sum reads that
                // stride-2 float64 view, as NumPy's does.
                var re = x.real;
                var im = x.imag;
                Square(re, null, re);
                Square(im, null, im);
                Add(re, im, null, re);
                squared = re;
            }
            else
            {
                // The general form: multiply(x, conjugate(x), out=x).real — x² for a real x (a bool input).
                Multiply(x, x, null, x);
                squared = x;
            }

            // ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where): out joins this sum only.
            var ret = ReduceAdd(squared, axis, keepdims, loop, @out, null, @where);
            bool scalarResult = @out is null && ret.ndim == 0;
            if (scalarResult && !ret.Shape.IsWriteable)
                ret = ret.copy();

            // rcount = maximum(rcount - ddof, 0); ret / rcount (true_divide into ret, or the scalar's own division).
            if (rcounts is null)
                DivideByCountLikeNumPy(ret, Math.Max(rcount - ddof, 0));
            else
                DivideByCountLikeNumPy(ret, CountsMinusDdof(rcounts, ddof));

            if (takeSqrt)
            {
                // _std: sqrt(ret, out=ret) for an array (an out included: the ufunc's same_kind refusal of an integer out),
                // ret.dtype.type(sqrt(ret)) for a scalar.
                if (@out is not null)
                    Sqrt(@out, null, @out);
                else
                    SqrtLikeNumPy(ret, scalarResult);
            }

            if (@out is not null)
                return @out;
            return scalarResult ? ret.MarkReductionScalar() : ret;
        }

        /// <summary>
        ///     <paramref name="x"/> in the layout NumPy's ufunc gives <c>subtract(arr, mean)</c>'s result
        ///     (<see cref="Shape.NpyIterOutputShape"/> over both operands): <paramref name="x"/> itself when its strides
        ///     already are that layout, otherwise a copy in it (the values are elementwise, so only the order the next sum
        ///     walks changes).
        /// </summary>
        /// <param name="x">The difference.</param>
        /// <param name="arr">The first operand.</param>
        /// <param name="mean">The second operand.</param>
        /// <returns><paramref name="x"/>, or a fresh copy in NumPy's layout (the caller's scope releases the original).</returns>
        private NDArray InUfuncLayout(NDArray x, NDArray arr, NDArray mean)
        {
            if (x.ndim < 2)
                return x;
            var layout = Shape.NpyIterOutputShape((long[])x.Shape.dimensions.Clone(), new[] { arr.Shape, mean.Shape });
            var have = x.Shape.strides;
            var want = layout.strides;
            bool same = true;
            for (int d = 0; d < have.Length && same; d++)
                same = x.Shape.dimensions[d] <= 1 || have[d] == want[d];
            if (same)
                return x;
            var laid = NewArray(x.GetTypeCode, layout, false);
            NDIter.Copy(laid, x);
            return laid;
        }

        /// <summary>
        ///     NumPy's <c>um.maximum(rcount - ddof, 0)</c> for per-output counts: a fresh contiguous int64 array of
        ///     <c>max(count - ddof, 0)</c> in <paramref name="counts"/>' shape.
        /// </summary>
        /// <param name="counts">The int64 counts (any layout; a read-only 0-d scalar included).</param>
        /// <param name="ddof">The delta degrees of freedom.</param>
        /// <returns>The divisors.</returns>
        private unsafe NDArray CountsMinusDdof(NDArray counts, int ddof)
        {
            var divisors = NewArray(NPTypeCode.Int64, counts.ndim == 0 ? Shape.Scalar : Shape.Adopt((long[])counts.Shape.dimensions.Clone()), false);
            NDIter.Copy(divisors, counts);
            // The fresh array is dense C, offset 0: one pass over its buffer.
            long* p = (long*)divisors.Address;
            for (long i = 0, n = divisors.size; i < n; i++)
                p[i] = Math.Max(p[i] - ddof, 0);
            return divisors;
        }
    }
}
