using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================================================
        // NumPy's out= forms of the reductions: ufunc.reduce with an out operand (add / multiply), _mean, _var and _std
        // with out=. Moved from numpy.matrix's members (Matrixlib/NDMatrix.Reductions.cs) by the np-layer dtype plan's P1d,
        // statement for statement, with the matrix's keepdims=True made a parameter.
        //
        // THE LOOP WITH out=. A reduction without dtype= resolves its loop from out AND the operand
        // (ufunc_object.c reducelike_promote_and_resolve: the operand DTypes it promotes are (out, arr, out)), and it does
        // not widen small integers when out is given. So the loop is promote_types(out.dtype, arr.dtype): an int8 input
        // summed into an int8 out wraps in int8, into a bool out it sums in int8 and stores the truth value, into a
        // float64 out a float32 input is summed in float64. The result reaches out with an UNSAFE cast.
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>add.reduce</c> / <c>multiply.reduce(arr, axis, dtype, out, keepdims)</c> with an <c>out=</c> argument
        ///     (<c>PyUFunc_Reduce</c> with an out operand): the text loop resolution first, then the reduce prologue on
        ///     (out, arr), then the reduction in <c>dtype ?? promote_types(out.dtype, arr.dtype)</c> stored into
        ///     <paramref name="out"/> with an unsafe cast.
        /// </summary>
        /// <param name="op">The NumPy function: <c>"sum"</c> (add) or <c>"prod"</c> (multiply). It names the function in the
        ///     text routes' messages.</param>
        /// <param name="arr">The reduced array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1: <paramref name="out"/> must have
        ///     that shape.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null for the out= loop rule.</param>
        /// <param name="out">The destination (not null).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only, has the wrong rank or a length that cannot hold
        ///     the result (NumPy's reduce prologue texts), or text that a numeric loop must parse does not parse.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input (NumPy's
        ///     "remapped shapes" text, under the house broadcast-error type).</exception>
        /// <exception cref="TypeError">NumPy's loop refusal for text (a text input, a text out or a string request without a
        ///     loop), or a detailed / new-style <c>dtype=</c> request.</exception>
        /// <exception cref="OverflowException">A string repeat past <c>2^63 - 1</c> characters, or an integer text outside an
        ///     integer loop or out.</exception>
        /// <remarks>
        ///     <para>The result is computed fresh in the loop dtype and copied into <paramref name="out"/> (NDIter's unsafe
        ///     cast copy), so an out of another dtype receives NumPy's unsafe cast of the loop's result. NumPy differs in one
        ///     place: past one buffer fill (<c>np.getbufsize()</c> elements) its iterator carries the partial result through
        ///     out's dtype between fills, where this rounds into out once.</para>
        ///     <para>The fresh result is this method's intermediate and is disposed on every path.</para>
        /// </remarks>
        private NDArray ReduceIntoOut(string op, NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out)
        {
            // Text (the input, the out, or a string request): add / multiply.reduce's text loop resolution answers — it
            // computes into out (the StringDType concatenation, the string repeat, a numeric loop with the text cast around
            // it) or raises NumPy's refusal.
            if (ReduceTextIntoOut(op, arr, axis, keepdims, dtype, @out))
                return @out;

            // NumPy's reduce prologue runs before an element is read; then the out= loop rule (file header).
            string ufunc = op == "sum" ? "add" : "multiply";
            ReductionArguments.Validate(arr, axis, keepdims, @out, null, ufunc);
            // A char loop (a char or bool out over char data: promote_types(bool, char) is char) is NumSharp's uint16 proxy,
            // which ReduceAdd / ReduceProduct run themselves (ReduceInCharLoop).
            DType loop = dtype ?? DTypePromotion.PromoteTypes(@out.dtype, arr.dtype);
            NDArray result = op == "sum" ? ReduceAdd(arr, axis, keepdims, loop) : ReduceProduct(arr, axis, keepdims, loop);
            try
            {
                // The prologue proved out has the result's exact shape, so this is a plain element copy with NumPy's unsafe
                // cast (the same copy np.copyto(out, result, casting='unsafe') runs).
                NDIter.Copy(@out, result);
            }
            finally
            {
                if (!ReferenceEquals(result, arr))
                    result.Dispose();
            }
            return @out;
        }

        /// <summary>
        ///     The text form of <see cref="ReduceIntoOut"/>: when <paramref name="arr"/> or <paramref name="out"/> holds text
        ///     (<c>S</c>, <c>U</c> or <c>StringDType</c>), or <paramref name="dtype"/> names a text class, NumPy resolves the
        ///     reduction's loop on (out, arr) under the request (<see cref="Strings.NDReduceLikeResolution"/>) and computes or
        ///     refuses: the <c>StringDType</c> concatenation of a <c>StringDType</c> or <c>U</c> input into a <c>StringDType</c>
        ///     out, the string repeat of integer counts (into a text out, or through the loop's text into a numeric out), or a
        ///     NUMERIC loop with the text cast around it (a text input parsed, the result's <c>str()</c> written into a text
        ///     out).
        /// </summary>
        /// <param name="op">The NumPy function whose reduction this is (<c>"sum"</c> or <c>"prod"</c>).</param>
        /// <param name="arr">The reduced array.</param>
        /// <param name="axis">The axis as the caller passed it, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="dtype">The <c>dtype=</c> request (for <c>_mean</c>'s sum, its implicit float loop), or null.</param>
        /// <param name="out">The destination.</param>
        /// <returns>True when a loop computed into <paramref name="out"/>; false when no text is involved (the caller's numeric
        ///     path continues).</returns>
        /// <exception cref="ValueError">A read-only out, an out of the wrong shape, two or more axes for a loop that is not
        ///     reorderable, an empty reduction without identity, or a text that does not parse into the numeric loop (or into
        ///     a numeric out, from the repeat's text).</exception>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="TypeError">NumPy's loop refusal (its resolved-dtypes, no-loop and mixed-string texts), or a
        ///     detailed or new-style <c>dtype=</c>.</exception>
        /// <exception cref="OverflowException">A repeat past <c>2^63 - 1</c> characters, or an integer text outside an
        ///     integer loop or out.</exception>
        private bool ReduceTextIntoOut(string op, NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out)
        {
            // A StringDType input: its loop resolution (a dtype= request, an out of another class or multiply have no
            // loop: refused there), then the concatenation into out.
            if (Strings.NDStringUfuncs.VStringReduces(op, arr, axis, dtype, @out, loopOnly: true))
            {
                Strings.NDVStringReduction.SumNumPy(op, arr, ReduceOutAxis(arr, axis), keepdims, @out);
                return true;
            }
            switch (Strings.NDStringUfuncs.ReduceIntoOutRoute(op, arr, axis, @out, dtype, out var resolution))
            {
                case Strings.NDStringUfuncs.TextOutReduce.Concatenate:
                    Strings.NDVStringReduction.SumNumPyOfCast(arr, ReduceOutAxis(arr, axis), keepdims, @out);
                    return true;
                case Strings.NDStringUfuncs.TextOutReduce.Repeat:
                    // The repeat runs in the loop's text descriptor: the out's own when it is that text, else the out's dtype
                    // cast to the loop's string class (a numeric out reads as U21 / U32 / U5, a bytes out of the unicode loop
                    // as U), which NumPy's buffered iterator round-trips through.
                    if (resolution is not null && !resolution.Descriptors[0].Equals(@out.dtype))
                        Strings.NDStringRepeatReduction.ReduceNumPyThroughText(arr, axis, keepdims, @out, resolution.Descriptors[0]);
                    else
                        Strings.NDStringRepeatReduction.ReduceNumPy(arr, axis, keepdims, @out);
                    return true;
                case Strings.NDStringUfuncs.TextOutReduce.NumericCast:
                    // A numeric loop over text: the identity first, then the buffered casts of the input and the out.
                    Strings.NDTextNumericReduction.Reduce(op == "sum" ? "add" : "multiply", arr,
                        ReductionArguments.AxisFlags(arr.ndim, axis), keepdims, @out, resolution.Descriptors[0]);
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     The axis a text route takes: counted from the front once the caller's checks validated it, or null for every
        ///     axis — and for a 0-d input, whose axis 0 / -1 <c>ufunc.reduce</c> reads as every axis.
        /// </summary>
        /// <param name="arr">The reduced array.</param>
        /// <param name="axis">The validated axis as the caller passed it, or null.</param>
        /// <returns>The non-negative axis, or null.</returns>
        private static int? ReduceOutAxis(NDArray arr, int? axis)
            => axis is { } ax && arr.ndim > 0 ? (ax < 0 ? ax + arr.ndim : ax) : null;

        /// <summary>
        ///     NumPy's <c>_count_reduce_items(arr, axis)</c> without a mask, the count <c>_mean</c> and <c>_var</c> divide by:
        ///     the size for every axis, else the extent of the normalized axis.
        /// </summary>
        /// <param name="arr">The reduced array.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <returns>The number of elements each result element combines.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds — for a 0-d input every axis is
        ///     (<c>normalize_axis_index</c>, unlike <c>ufunc.reduce</c>'s rule).</exception>
        private static long ReduceItemCount(NDArray arr, int? axis)
            => axis is null ? arr.size : arr.Shape.dimensions[NormalizeAxis(axis.Value, arr.ndim)];

        /// <summary>
        ///     NumPy's <c>_mean(arr, axis, dtype, out, keepdims)</c> with an <c>out=</c> argument: <c>umr_sum(arr, axis, dtype,
        ///     out, keepdims)</c> — the sum STORED into out — then <c>true_divide(out, rcount, out=out, casting='unsafe')</c>, so
        ///     the division runs on the sum as out's dtype holds it (an integer out holds truncated values).
        /// </summary>
        /// <param name="arr">The array to average.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1 (<paramref name="out"/> has that shape).</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null: float64 for a bool or integer input, float32 for float16
        ///     (<c>_mean</c>'s implicit loop), else the out= loop rule.</param>
        /// <param name="out">The destination (not null).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or misshapen (the reduce prologue's texts), or text
        ///     a numeric loop must parse does not parse.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input.</exception>
        /// <exception cref="TypeError">NumPy's refusal for text: no add loop, or no division loop for a text out (raised AFTER
        ///     the sum was stored, as NumPy's division meets the stored sum).</exception>
        /// <remarks>
        ///     NumPy's order: <c>_count_reduce_items</c> normalizes the axis first (an axis out of bounds is the AxisError
        ///     even with a read-only out, probed 2.4.2), then <c>umr_sum</c> parses out (the reduce prologue, or the text
        ///     routes' own checks).
        /// </remarks>
        private NDArray MeanIntoOut(NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out)
        {
            // _mean: ret = umr_sum(arr, axis, dtype, out, keepdims); ret = true_divide(ret, rcount, out=ret,
            // casting='unsafe'). The sum's loop is dtype=, else float64 for integers and bool, float32 for float16, else the
            // reduce rule for a missing dtype with an out (file header).
            DType loop = dtype;
            if (loop is null)
            {
                NPTypeCode code = arr.GetTypeCode;
                if (code == NPTypeCode.Boolean || code.IsInteger() || code == NPTypeCode.Char)
                    loop = np.float64;
                else if (code == NPTypeCode.Half)
                    loop = np.float32;
            }

            // Text (a text input, a text out, or a text loop request): _count_reduce_items validates the axis, then
            // umr_sum(arr, axis, loop, out, keepdims) is add.reduce under the loop as its request — the implicit float64 /
            // float32 loop of an integer, bool or float16 input passes as a request too — whose ported loop resolution
            // computes into out or refuses. A sum it computed meets true_divide(ret, rcount, out=ret, casting='unsafe'):
            // a numeric out divides, a text out has no divide loop (NumPy's division refusal, after the sum was stored).
            if (Strings.NDStringUfuncs.IsText(arr) || Strings.NDStringUfuncs.IsText(@out) || Strings.NDStringUfuncs.IsTextRequest(loop))
            {
                long rcount = ReduceItemCount(arr, axis);
                if (ReduceTextIntoOut("sum", arr, axis, keepdims, loop, @out))
                {
                    DivideByCountLikeNumPy(@out, rcount);
                    return @out;
                }
            }

            // _count_reduce_items before umr_sum parses out=: the axis first.
            long count = ReduceItemCount(arr, axis);
            ReductionArguments.Validate(arr, axis, keepdims, @out, null, "add");
            NDArray total = ReduceAdd(arr, axis, keepdims, loop ?? DTypePromotion.PromoteTypes(@out.dtype, arr.dtype));
            try
            {
                // The prologue proved out has the sum's exact shape: an element copy with NumPy's unsafe cast.
                NDIter.Copy(@out, total);
            }
            finally
            {
                if (!ReferenceEquals(total, arr))
                    total.Dispose();
            }
            // The division runs ON out: the sum as out's dtype holds it, divided by the strong int64 count.
            DivideByCountLikeNumPy(@out, count);
            return @out;
        }

        /// <summary>
        ///     NumPy's <c>_var(arr, axis, dtype, out, ddof, keepdims)</c> with an <c>out=</c> argument (<c>_std</c>'s first
        ///     statement too): NumPy hands <paramref name="out"/> to its LAST sum only — the mean and the deviations are
        ///     computed as without it, the sum of the squared deviations runs in <c>promote_types(out.dtype, deviation
        ///     dtype)</c> unless <paramref name="dtype"/> is given, is stored into <paramref name="out"/> with an unsafe cast,
        ///     and <paramref name="out"/> is then divided in place by <c>max(N - ddof, 0)</c>.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1 (<paramref name="out"/> has that shape).</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The destination (not null).</param>
        /// <returns><paramref name="out"/> itself, holding the variance.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or misshapen (the reduce prologue's texts).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input.</exception>
        /// <exception cref="TypeError">A text out for a float or complex input without <paramref name="dtype"/> (the last
        ///     sum's add loop on (out, x) has none), the refusals the mean's own reduction raises for a text input, or the
        ///     boolean-subtract refusal of a bool input in a bool loop.</exception>
        /// <remarks>
        ///     NumPy's order, which an error makes visible: the axis (<c>_count_reduce_items</c>), the mean's errors (a text
        ///     input), <c>x = arr - arrmean</c>'s (two bools), and only then out's (its read-only check, its rank and shape)
        ///     in the last <c>umr_sum</c> (probed 2.4.2). <see cref="VarStdNumPy"/> parses out at that point itself.
        /// </remarks>
        private NDArray VarianceIntoOut(NDArray arr, int? axis, bool keepdims, int ddof, DType dtype, NDArray @out)
        {
            // _var's last statement, ret = umr_sum(x, axis, dtype, out, keepdims): without dtype= a float or complex input
            // keeps its deviations' float dtype (a complex one's (x * conj(x)).real is float64), and a text out meets
            // add.reduce's loop resolution on (out, x), which has no loop. NumPy validated the axis first
            // (_count_reduce_items), the mean and the deviations cannot fail, and umr_sum parses out= (its read-only check)
            // before it resolves the loop. (An integer or bool input sums in float64 and the division refuses the text out;
            // a text input is refused by the mean's own reduction; both run below.)
            NPTypeCode code = arr.GetTypeCode;
            if (dtype is null && Strings.NDStringUfuncs.IsText(@out) && (code.IsFloatingPoint() || code == NPTypeCode.Complex))
            {
                ReduceItemCount(arr, axis);
                if (!@out.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
                DType deviations = code == NPTypeCode.Complex ? np.float64 : arr.dtype;
                throw Strings.NDStringUfuncs.IsVString(@out)
                    ? Strings.NDStringUfuncs.VStringNoLoop("add", new[] { Strings.NDStringUfuncs.DispatchClass(@out), deviations.Meta.ToString() })
                    : Strings.NDStringUfuncs.ReduceOutError("add", deviations, @out.dtype, accumulate: false);
            }

            // _var: ret = umr_sum(x, axis, dtype, out, keepdims) — the sum of the squared deviations, whose loop out joins,
            // stored into out — then ret = true_divide(ret, max(rcount - ddof, 0), out=ret, casting='unsafe'). VarStdNumPy
            // runs those statements with out, parsing out where NumPy's last umr_sum does (after the axis, the mean and the
            // deviations). The uninitialized-shape sentinel (an NDArray without a shape) comes back as the operand itself,
            // out untouched.
            return VarStdNumPy(arr, axis, keepdims, ddof, dtype, takeSqrt: false, @out: @out);
        }

        /// <summary>
        ///     NumPy's <c>_std(arr, axis, dtype, out, ddof, keepdims)</c> with an <c>out=</c> argument: <c>_var</c> into
        ///     <paramref name="out"/> (<see cref="VarianceIntoOut"/>), then <c>sqrt(out, out=out)</c> under the ufunc's
        ///     <c>same_kind</c> rule.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The destination (not null): a float or complex array of the result's shape.</param>
        /// <returns><paramref name="out"/> itself, holding the standard deviation.</returns>
        /// <exception cref="ValueError">See <see cref="VarianceIntoOut"/>.</exception>
        /// <exception cref="AxisError">See <see cref="VarianceIntoOut"/>.</exception>
        /// <exception cref="IncorrectShapeException">See <see cref="VarianceIntoOut"/>.</exception>
        /// <exception cref="TypeError">See <see cref="VarianceIntoOut"/>.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="out"/>: the square root's float loop cannot
        ///     be cast back under <c>same_kind</c> (<c>Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64')
        ///     with casting rule 'same_kind'</c>), raised AFTER the variance was stored, so out holds the variance then.</exception>
        private NDArray StdIntoOut(NDArray arr, int? axis, bool keepdims, int ddof, DType dtype, NDArray @out)
        {
            VarianceIntoOut(arr, axis, keepdims, ddof, dtype, @out);
            // _std: ret = um.sqrt(ret, out=ret). The ufunc refuses an integer or bool out itself (its loop is a float one
            // that same_kind cannot cast back), after the variance is in out.
            Sqrt(@out, @out: @out);
            return @out;
        }
    }
}
