using System;
using System.Collections.Frozen;
using System.Text;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The ORDER in which NumPy 2.4.2 reports the problems of a ufunc or reduction call that involves text, and its
    ///     answer for an unsized string <c>dtype=</c> request (<c>'U'</c>, <c>'S'</c>, <c>np.str_</c>, <c>np.bytes_</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Ufunc calls</b> (probed against 2.4.2, <c>ufdt*.py</c>): the <c>dtype=</c> class check
    ///     (<see cref="DefaultEngine.ThrowIfNotGeneralDType"/>) comes first, then a read-only <c>out=</c> ("output array is
    ///     read-only"), then a non-bool <c>where=</c>, then the loop resolution (the refusal texts of this class), then the
    ///     cast of the loop's output descriptor into <c>out=</c>, and only then broadcasting. A call NumPy goes on to
    ///     compute raises <see cref="StringComputationNotSupportedException"/> only after all of those passed, so every
    ///     refusal NumPy makes is reported with NumPy's text even on a call NumSharp cannot compute yet.
    ///     </para>
    ///     <para>
    ///     <b>Reductions</b> validate their <c>axis</c> before they resolve a loop, with three rules that differ only for a
    ///     0-d input (<see cref="ReduceAxisRule"/>), and a direct <c>ufunc.reduce</c> reads <c>dtype=</c> before the axis
    ///     where <c>np.mean</c>'s family reads it after.
    ///     </para>
    ///     <para>
    ///     <b>A string <c>dtype=</c></b> selects the string loops (<c>add</c> concatenates numbers as text, after casting
    ///     each input to the requested class) or names a missing signature: the promoter ufuncs name the operand CLASSES
    ///     and the requested class (<c>(&lt;class 'numpy.dtypes.Float64DType'&gt;, …) -&gt; &lt;class
    ///     'numpy.dtypes.StrDType'&gt;</c>), every other ufunc gives the legacy resolver's <c>No loop matching the
    ///     specified signature and casting was found for ufunc X</c>.
    ///     </para>
    /// </remarks>
    internal static partial class NDStringUfuncs
    {
        /// <summary>
        ///     The binary ufuncs whose promoter, under a string <c>dtype=</c> request, names the operand CLASSES and the
        ///     requested class (probed: <c>subtract(f8, f8, dtype='U')</c>). Every other binary ufunc gives the legacy
        ///     resolver's "No loop matching the specified signature" text.
        /// </summary>
        private static readonly FrozenSet<string> s_requestClassForm = FrozenSet.ToFrozenSet(new[]
        {
            "subtract", "multiply", "maximum", "minimum", "fmax", "fmin", "gcd", "lcm", "matmul", "vecdot", "matvec",
            "vecmat",
        }, StringComparer.Ordinal);

        /// <summary>
        ///     How a reduction validates its <c>axis</c> argument. The three rules agree for an input of one or more
        ///     dimensions and differ only for a 0-d input — which is exactly where a string input's refusal and an
        ///     AxisError compete.
        /// </summary>
        internal enum ReduceAxisRule
        {
            /// <summary>
            ///     <c>ufunc.reduce</c> (<c>PyUFunc_GenericReduction</c>): a 0-d input accepts axis 0 and -1 as "every axis"
            ///     (NumPy's backward-compatibility rule for <c>np.sum(scalar, axis=0)</c>); a tuple is checked item by item
            ///     and a repeat is <c>duplicate value in 'axis'</c>.
            /// </summary>
            UfuncReduce,

            /// <summary>
            ///     <c>normalize_axis_index</c> / <c>normalize_axis_tuple</c> against the input's rank (<c>np.mean</c>'s
            ///     <c>_count_reduce_items</c>, <c>_ureduce</c> of the median / quantile family, <c>ndarray.sort</c>): a 0-d
            ///     input refuses every axis; a tuple repeat is <c>repeated axis</c>.
            /// </summary>
            CountItems,

            /// <summary>
            ///     The functions that ravel a 0-d input to one dimension first (the accumulations, the arg-reductions,
            ///     <c>argsort</c>, <c>argpartition</c>): the axis is checked against <c>max(ndim, 1)</c>.
            /// </summary>
            Raveled,
        }

        /// <summary>True when <paramref name="dtype"/> names a fixed-width string class (sized or not).</summary>
        /// <param name="dtype">A <c>dtype=</c> request, or null.</param>
        /// <returns>True for an <c>S</c> or <c>U</c> descriptor.</returns>
        internal static bool IsStringRequest(DType dtype) => dtype?.Meta is StringDTypeMeta;

        /// <summary>
        ///     The part of a ufunc call's argument parsing NumPy finishes before any loop resolution: a read-only
        ///     <c>out=</c>, then a <c>where=</c> mask that is not bool.
        /// </summary>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not bool (NumPy's TypeError text; the house
        ///     type for the mask check).</exception>
        internal static void ThrowIfOutWhereInvalid(NDArray @out, NDArray where)
        {
            var e = OutWhereError(@out, where);
            if (e is not null)
                throw e;
        }

        /// <summary>
        ///     <see cref="ThrowIfOutWhereInvalid"/> as a value: the exception NumPy's argument parsing raises for
        ///     <paramref name="out"/> / <paramref name="where"/>, or null when both are acceptable. The guards that RETURN
        ///     their answer (<see cref="BinaryError"/>, <see cref="UnaryError"/>) put it ahead of the loop answer.
        /// </summary>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The <see cref="ValueError"/> for a read-only out, the <see cref="ArgumentException"/> for a non-bool
        ///     mask (the house type of <see cref="DefaultEngine.ValidateWhereMask"/>), or null.</returns>
        internal static Exception OutWhereError(NDArray @out, NDArray where)
        {
            if (@out is not null && !@out.Shape.IsWriteable)
                return new ValueError("output array is read-only");
            if (where is not null && where.typecode != NPTypeCode.Boolean)
            {
                try
                {
                    DefaultEngine.ValidateWhereMask(where);
                }
                catch (ArgumentException e)
                {
                    // ValidateWhereMask owns the text (array vs scalar wording); it only ever throws here.
                    return e;
                }
            }

            return null;
        }

        /// <summary>
        ///     The string part of a ufunc call's prologue, run by <see cref="DefaultEngine.UfuncLoopCode(DType, string,
        ///     NDArray, NDArray, NDArray, NDArray)"/> right after the <c>dtype=</c> class check. Does nothing for numeric
        ///     operands under a numeric (or no) request. With a string operand it raises the <c>out=</c> / <c>where=</c>
        ///     problems NumPy reports before any loop resolution, and under a string request it raises NumPy's complete
        ///     answer (<see cref="DTypeLoopError"/>).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null for a unary ufunc.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or an input / output cast the string
        ///     loop refuses (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="TypeError">NumPy has no loop for the request.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (a computing call without out / where).</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text.</exception>
        /// <remarks>
        ///     A string operand under a numeric (or no) request returns after the <c>out=</c> / <c>where=</c> checks: the
        ///     caller's own guard (<see cref="ThrowIfUnary"/>, <see cref="ThrowIfBinary"/>) then gives the loop answer, after
        ///     the ufunc-specific checks NumPy makes first (a bool-only ufunc's no-loop text for a numeric request).
        /// </remarks>
        internal static void ThrowIfStringCall(string ufunc, DType dtype, NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            bool request = IsStringRequest(dtype);
            if (!request && !IsString(x1) && !IsString(x2))
                return;
            ThrowIfOutWhereInvalid(@out, where);
            if (request)
                throw DTypeLoopError(ufunc, dtype, x1, x2, @out, where);
        }

        /// <summary>
        ///     NumPy's answer for a ufunc called with a string <c>dtype=</c> request (the operands may be numbers or text).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="requested">The string descriptor requested (its CLASS decides; a sized one never gets here, the
        ///     class check refused it).</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null for a unary ufunc.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     <list type="bullet">
        ///       <item><description><c>negative</c> / <c>positive</c> / <c>absolute</c> / <c>sign</c> name the input class
        ///       and the requested class; <c>absolute</c> of a complex input keeps its complex→real loop, which a
        ///       signature matches by number, so it gets the no-loop text (probed).</description></item>
        ///       <item><description><c>add</c> runs NumPy's string add promoter: two strings of one class concatenate in
        ///       their own class whatever the request; two classes refuse; otherwise each input is cast to the requested
        ///       class (inputs in order, same_kind: <c>U → S</c> fails) and the call concatenates their text.</description></item>
        ///       <item><description><c>multiply</c> of a string and an integer (not bool) is the string repeat, which
        ///       refuses to run without <c>out=</c>.</description></item>
        ///       <item><description>The promoter ufuncs name both operand classes; every other ufunc is the legacy
        ///       resolver's no-loop text.</description></item>
        ///     </list>
        /// </remarks>
        internal static Exception DTypeLoopError(string ufunc, DType requested, NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            var req = requested.Meta;
            if (x2 is null)
            {
                if (s_classNoLoop.Contains(ufunc) && !(ufunc == "absolute" && x1.typecode == NPTypeCode.Complex))
                    return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {x1.dtype.Meta} -> {req}");
                return NoLoopForSignature(ufunc);
            }

            DType d1 = x1.dtype, d2 = x2.dtype;
            bool s1 = d1.Meta is StringDTypeMeta, s2 = d2.Meta is StringDTypeMeta;
            if (ufunc == "add")
            {
                if (s1 && s2)
                {
                    // The string add promoter keeps the operands' own class: add(S2, S2, dtype='U') is S4 (probed).
                    if (d1.Meta != d2.Meta)
                        return new TypeError($"ufunc 'add' cannot use operands with types {d1.ToString(true)} and {d2.ToString(true)}");
                    return ComputeAnswer("add", x1, x2, @out, where, Concatenation(d1, d2));
                }

                // A number (or the other string class) reaches the requested class by NumPy's input cast, inputs in
                // order; text never narrows from U to S by a same_kind cast.
                var c1 = s1 && d1.Meta == req ? d1 : DTypePromotion.CastDescrToDType(d1, req);
                if (!DTypeCasting.CanCastTypeTo(d1, c1, NPY_CASTING.NPY_SAME_KIND_CASTING))
                    return InputCastText("add", 0, d1, c1);
                var c2 = s2 && d2.Meta == req ? d2 : DTypePromotion.CastDescrToDType(d2, req);
                if (!DTypeCasting.CanCastTypeTo(d2, c2, NPY_CASTING.NPY_SAME_KIND_CASTING))
                    return InputCastText("add", 1, d2, c2);
                return ComputeAnswer("add", x1, x2, @out, where, Concatenation(c1, c2));
            }

            // The string repeat promoter wins over the request: any integer width (not bool) times a string.
            if (ufunc == "multiply" && (s1 && d2.typecode.IsInteger() || s2 && d1.typecode.IsInteger()))
                return MultiplyRepeatAnswer(x1, x2, @out, where);

            if (s_requestClassForm.Contains(ufunc))
                return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types ({d1.Meta}, {d2.Meta}) -> {req}");
            return NoLoopForSignature(ufunc);
        }

        /// <summary>
        ///     NumPy's answer for <c>np.clip</c> under a string <c>dtype=</c> request. <c>np.clip</c> routes by the bounds it
        ///     was given (<c>_clip</c>): none is <c>positive</c>, one is <c>minimum</c> / <c>maximum</c>, two is the
        ///     <c>clip</c> ufunc, and each names the operand classes and the requested class.
        /// </summary>
        /// <param name="requested">The string descriptor requested.</param>
        /// <param name="a">The clipped array.</param>
        /// <param name="min">The lower bound, or null.</param>
        /// <param name="max">The upper bound, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception ClipDTypeLoopError(DType requested, NDArray a, NDArray min, NDArray max)
        {
            var req = requested.Meta;
            if (min is null && max is null)
                return new TypeError($"ufunc 'positive' did not contain a loop with signature matching types {a.dtype.Meta} -> {req}");
            if (min is null)
                return new TypeError($"ufunc 'minimum' did not contain a loop with signature matching types ({a.dtype.Meta}, {max.dtype.Meta}) -> {req}");
            if (max is null)
                return new TypeError($"ufunc 'maximum' did not contain a loop with signature matching types ({a.dtype.Meta}, {min.dtype.Meta}) -> {req}");
            return new TypeError(
                $"ufunc 'clip' did not contain a loop with signature matching types ({a.dtype.Meta}, {min.dtype.Meta}, {max.dtype.Meta}) -> {req}");
        }

        /// <summary>
        ///     NumPy's answer for a gufunc (<c>matmul</c>, <c>vecdot</c>, <c>matvec</c>, <c>vecmat</c>) under a string
        ///     <c>dtype=</c> request: the loop is resolved before the core dimensions are checked, and the promoter names the
        ///     operand classes and the requested class.
        /// </summary>
        /// <param name="gufunc">NumPy's gufunc name.</param>
        /// <param name="requested">The string descriptor requested.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static TypeError GufuncDTypeLoopError(string gufunc, DType requested, NDArray x1, NDArray x2)
            => new TypeError(
                $"ufunc '{gufunc}' did not contain a loop with signature matching types ({x1.dtype.Meta}, {x2.dtype.Meta}) -> {requested.Meta}");

        /// <summary>
        ///     The legacy type resolver's refusal of a <c>dtype=</c> signature: <c>No loop matching the specified signature
        ///     and casting was found for ufunc X</c> (TypeError; the house type for this text is
        ///     <see cref="IncorrectTypeException"/>).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static IncorrectTypeException NoLoopForSignature(string ufunc)
            => new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {ufunc}");

        /// <summary>
        ///     NumPy's input-cast refusal of a resolved loop: <c>Cannot cast ufunc 'add' input 0 from dtype('&lt;U2') to
        ///     dtype('S2') with casting rule 'same_kind'</c> (UFuncTypeError; the house type for a cast refusal is
        ///     <see cref="ArgumentException"/>).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="index">The input's position.</param>
        /// <param name="from">The input's descriptor.</param>
        /// <param name="to">The loop's descriptor for that input.</param>
        /// <returns>The exception (not thrown).</returns>
        private static ArgumentException InputCastText(string ufunc, int index, DType from, DType to)
            => new ArgumentException(
                $"Cannot cast ufunc '{ufunc}' input {index} from {from.ToString(true)} to {to.ToString(true)} with casting rule 'same_kind'");

        /// <summary>
        ///     The descriptor of a string concatenation: the class of the two (equal-class) inputs and the sum of their
        ///     lengths — <c>add(U2, U3)</c> is <c>U5</c>, and NumPy's numbers cast to text count their
        ///     <c>REQUIRED_STR_LEN</c> (<c>float64</c> 32, <c>int64</c> 21).
        /// </summary>
        /// <param name="d1">The first input's descriptor in the loop's class.</param>
        /// <param name="d2">The second input's descriptor in the loop's class.</param>
        /// <returns>The output descriptor, its length clamped to the class limit.</returns>
        internal static DType Concatenation(DType d1, DType d2)
        {
            var meta = (StringDTypeMeta)d1.Meta;
            long chars = d1.itemsize / meta.UnitSize + d2.itemsize / meta.UnitSize;
            return meta.Descr(Math.Min(chars, meta.MaxChars));
        }

        /// <summary>
        ///     NumPy's string repeat (<c>multiply</c> of a string and an integer): without <c>out=</c> the legacy loop cannot
        ///     size its result and refuses; with one, the loop's output is <c>out=</c>'s descriptor cast to the string
        ///     operand's class (<c>out=float64</c> reads as <c>&lt;U32</c>), which must reach <c>out=</c> by a same_kind cast.
        /// </summary>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception MultiplyRepeatAnswer(NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            if (@out is null)
                return new TypeError("The 'out' kwarg is necessary when using the string multiply ufunc directly. " +
                                     "Use numpy.strings.multiply to multiply strings without specifying 'out'.");
            var text = (IsString(x1) ? x1 : x2).dtype.Meta;
            var result = @out.dtype.Meta == text ? @out.dtype : DTypePromotion.CastDescrToDType(@out.dtype, text);
            return ComputeAnswer("multiply", x1, x2, @out, where, result);
        }

        /// <summary>
        ///     The checks NumPy still makes after it resolved a string loop it computes: the cast of the loop's output
        ///     descriptor into <c>out=</c> (same_kind), then the broadcast of every operand. Only a call that passes both is
        ///     the next phase.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null for a unary ufunc.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="result">The loop's output descriptor.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception ComputeAnswer(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where, DType result)
        {
            if (@out is not null && !DTypeCasting.CanCastTypeTo(result, @out.dtype, NPY_CASTING.NPY_SAME_KIND_CASTING))
                return new ArgumentException(
                    $"Cannot cast ufunc '{ufunc}' output from {result.ToString(true)} to {@out.dtype.ToString(true)} with casting rule 'same_kind'");
            var broadcast = BroadcastError(x1, x2, @out, where);
            if (broadcast is not null)
                return broadcast;
            return NextPhase(ufunc, result.Meta is StringDTypeMeta ? result : (IsString(x1) ? x1 : x2 ?? x1).dtype);
        }

        /// <summary>
        ///     NumPy's broadcast refusal for a ufunc call, or null when the operands broadcast: NpyIter joins every operand
        ///     (inputs, then <c>out=</c>, then the <c>where=</c> mask) and lists all of their shapes when they do not
        ///     broadcast together; an <c>out=</c> that would have to be stretched is <c>non-broadcastable output operand with
        ///     shape … doesn't match the broadcast shape …</c>.
        /// </summary>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The refusal, or null.</returns>
        /// <remarks>
        ///     The house types follow the numeric ufunc paths: two inputs alone raise
        ///     <see cref="IncorrectShapeException"/> (the elementwise broadcast wrapper), a call with <c>out=</c> or
        ///     <c>where=</c> raises <see cref="ArgumentException"/> (the out / where join); NumPy raises ValueError for both,
        ///     with these texts.
        /// </remarks>
        internal static Exception BroadcastError(NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            var operands = new[] { x1, x2, @out, where };
            // NpyIter's broadcast: right-aligned, every dimension equal or 1. Computed here rather than through the shared
            // pairwise shape helper, whose (1,)-shortcut answers () for (1,) against () — NumPy's join of those is (1,),
            // which decides whether a 0-d out= is "non-broadcastable".
            int nd = 0;
            foreach (var op in operands)
                if (op is not null)
                    nd = Math.Max(nd, op.ndim);
            var full = new long[nd];
            for (int i = 0; i < nd; i++)
                full[i] = 1;
            bool ok = true;
            foreach (var op in operands)
            {
                if (op is null)
                    continue;
                // A 0-d shape may carry no dimension array at all.
                var dims = op.Shape.dimensions ?? Array.Empty<long>();
                int lead = nd - dims.Length;
                for (int i = 0; i < dims.Length; i++)
                {
                    long d = dims[i], f = full[lead + i];
                    if (d == f || d == 1)
                        continue;
                    if (f == 1)
                        full[lead + i] = d;
                    else
                        ok = false;
                }
            }

            if (!ok)
            {
                var sb = new StringBuilder("operands could not be broadcast together with shapes ");
                foreach (var op in operands)
                    if (op is not null)
                        sb.Append(op.Shape.ToPythonTuple()).Append(' ');
                return @out is null && where is null ? new IncorrectShapeException(sb.ToString()) : new ArgumentException(sb.ToString());
            }

            // An output operand may absorb the inputs' broadcast but is never stretched itself.
            if (@out is not null && !SameDimensions(full, @out.Shape.dimensions ?? Array.Empty<long>()))
                return new ArgumentException(
                    $"non-broadcastable output operand with shape {@out.Shape.ToPythonTuple()} doesn't match the broadcast shape " +
                    $"{Shape.Adopt(full).ToPythonTuple()}");
            return null;
        }

        /// <summary>True when two dimension lists are equal (same rank, same extents).</summary>
        /// <param name="a">The first dimensions.</param>
        /// <param name="b">The second dimensions.</param>
        /// <returns>True for equal dimensions.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     The axis rule a reduction-style NumPy function applies (see <see cref="ReduceAxisRule"/>), by function name.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <returns>The rule.</returns>
        internal static ReduceAxisRule AxisRuleOf(string op) => op switch
        {
            "mean" or "var" or "std" or "nanmean" or "nanvar" or "nanstd" or "median" or "nanmedian" or "percentile"
                or "quantile" or "nanpercentile" or "nanquantile" or "sort" or "partition" or "average" => ReduceAxisRule.CountItems,
            "cumsum" or "cumprod" or "nancumsum" or "nancumprod" or "cumulative_sum" or "cumulative_prod" or "argmax"
                or "argmin" or "nanargmax" or "nanargmin" or "argsort" or "argpartition" => ReduceAxisRule.Raveled,
            _ => ReduceAxisRule.UfuncReduce,
        };

        /// <summary>
        ///     Validates an integer <c>axis</c> the way NumPy's <paramref name="op"/> does, before it resolves any loop.
        /// </summary>
        /// <param name="op">The NumPy function name (its <see cref="AxisRuleOf"/> rule applies; <c>average</c> prefixes its
        ///     argument name, <c>axis: axis 9 is out of bounds …</c>).</param>
        /// <param name="ndim">The input's rank.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <exception cref="AxisError">The axis is out of bounds under the rule.</exception>
        internal static void ValidateReduceAxis(string op, int ndim, int? axis)
        {
            if (axis is not { } ax)
                return;
            switch (AxisRuleOf(op))
            {
                case ReduceAxisRule.UfuncReduce:
                    if (ndim == 0 && (ax == 0 || ax == -1))
                        return;
                    CheckAxis(op, ax, ndim);
                    return;
                case ReduceAxisRule.Raveled:
                    CheckAxis(op, ax, Math.Max(ndim, 1));
                    return;
                default:
                    CheckAxis(op, ax, ndim);
                    return;
            }
        }

        /// <summary>
        ///     Validates a tuple <c>axis</c> the way NumPy's <paramref name="op"/> does: every item is checked first, then
        ///     repeats — <c>ufunc.reduce</c>'s <c>duplicate value in 'axis'</c>, <c>normalize_axis_tuple</c>'s
        ///     <c>repeated axis</c> (with <c>average</c>'s argument name: <c>repeated axis in `axis` argument</c>).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <param name="axes">The axes, or null for every axis.</param>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">An axis appears twice.</exception>
        internal static void ValidateReduceAxes(string op, int ndim, int[] axes)
        {
            if (axes is null)
                return;
            var rule = AxisRuleOf(op);
            int bound = rule == ReduceAxisRule.Raveled ? Math.Max(ndim, 1) : ndim;
            var seen = new bool[Math.Max(bound, 1)];
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                normalized[i] = CheckAxis(op, axes[i], bound);
            foreach (int ax in normalized)
            {
                if (seen[ax])
                    throw new ValueError(rule == ReduceAxisRule.UfuncReduce ? "duplicate value in 'axis'"
                        : op == "average" ? "repeated axis in `axis` argument" : "repeated axis");
                seen[ax] = true;
            }
        }

        /// <summary>
        ///     <c>normalize_axis_index</c>: the non-negative axis, or NumPy's AxisError (with <c>average</c>'s argument-name
        ///     prefix).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="ndim">The rank the axis must address.</param>
        /// <returns>The axis in <c>[0, ndim)</c>.</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        private static int CheckAxis(string op, int axis, int ndim)
        {
            int ax = axis < 0 ? axis + ndim : axis;
            if (ax >= 0 && ax < ndim)
                return ax;
            if (op == "average")
                throw new AxisError($"axis: axis {axis} is out of bounds for array of dimension {ndim}");
            throw new AxisError(axis, ndim);
        }

        /// <summary>
        ///     The guard of a reduction-style function: does nothing unless the input holds text or <c>dtype=</c> names a
        ///     string class; otherwise reports the problems in NumPy's order — a direct <c>ufunc.reduce</c>'s <c>dtype=</c>
        ///     class check, the axis, <c>np.mean</c>'s family's later class check — and then NumPy's answer
        ///     (<see cref="ReduceDTypeLoopError"/> for a string request, <see cref="ReduceError"/> for a string input).
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"sum"</c>, <c>"cumprod"</c>, <c>"median"</c>, ...).</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <exception cref="TypeError">A sized or byte-swapped <c>dtype=</c> (the class-only text), or NumPy's refusal.</exception>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">An arg reduction (<c>argmax</c>, <c>argmin</c>, <c>nanargmax</c>,
        ///     <c>nanargmin</c>) over an empty reduced length.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text, or casts the text to
        ///     a numeric <paramref name="dtype"/> (a parse).</exception>
        internal static void ThrowIfReduce(string op, NDArray a, int? axis, DType dtype = null)
        {
            bool request = IsStringRequest(dtype);
            if (!request && !IsString(a))
                return;
            PreReduceDTypeCheck(op, dtype);
            ValidateReduceAxis(op, a.ndim, axis);
            PostReduceDTypeCheck(op, dtype);
            // The arg reductions take no dtype=, so only a string INPUT reaches here: an empty reduced length is
            // PyArray_ArgMax's ValueError before any comparison, whatever the other axes hold (probed: argmax of a
            // (2, 0) array along axis 1 raises, of a (0, 2) array along axis 1 returns an empty int64 array).
            if (!request && EmptyArgReductionError(op, a, axis) is { } empty)
                throw empty;
            throw ReduceAnswer(op, a, dtype, request);
        }

        /// <summary>
        ///     The answer of a reduction-style function once its checks passed: a string <c>dtype=</c> request is
        ///     <see cref="ReduceDTypeLoopError"/>; a text input is <see cref="ReduceError"/> — except that a NUMERIC request
        ///     over text makes a multiply / maximum / minimum reduction cast the text to that number type first
        ///     (<c>np.prod(['1', '2'], dtype='f8')</c> is 2.0, <c>'a'</c> is <c>could not convert string to float</c>, probed),
        ///     which is text-to-number conversion, the next phase. The add family ignores a numeric request: the string add
        ///     loop resolves from the two text operands alone (<c>np.add(u, u, dtype='f8')</c> concatenates, probed), so
        ///     <c>np.sum(u, dtype='f8')</c> is the same <c>add.reduce</c> refusal as without one.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="request">Whether <paramref name="dtype"/> names a string class.</param>
        /// <returns>The exception to throw (never null).</returns>
        private static Exception ReduceAnswer(string op, NDArray a, DType dtype, bool request)
        {
            if (request)
                return ReduceDTypeLoopError(op, dtype, a);
            if (dtype is not null && !ReducesWithAdd(op))
                return NextPhase(op, a);
            return ReduceError(op, a);
        }

        /// <summary>
        ///     Whether NumPy's <paramref name="op"/> reduces with the <c>add</c> ufunc (directly, or through
        ///     <c>np.mean</c> / <c>np.sum</c>), whose string loop ignores a numeric <c>dtype=</c> request.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <returns>True for the add family.</returns>
        private static bool ReducesWithAdd(string op) => op switch
        {
            "sum" or "mean" or "var" or "std" or "nansum" or "nanmean" or "nanvar" or "nanstd" or "cumsum" or "nancumsum"
                or "cumulative_sum" or "trace" or "average" or "median" => true,
            _ => false,
        };

        /// <summary>
        ///     <c>PyArray_ArgMax</c> / <c>PyArray_ArgMin</c>'s refusal of an empty reduced length:
        ///     <c>attempt to get argmax of an empty sequence</c> (ValueError). NumPy moves the reduced axis last (a 0-d
        ///     input and <c>axis=None</c> ravel first) and refuses when that axis is 0 long, before it compares anything.
        ///     <c>np.nanargmax</c> / <c>np.nanargmin</c> of text are <c>np.argmax</c> / <c>np.argmin</c> (their
        ///     <c>_replace_nan</c> leaves a non-inexact array alone), so they name <c>argmax</c> / <c>argmin</c> too.
        /// </summary>
        /// <param name="op">The NumPy function name; only <c>argmax</c>, <c>argmin</c>, <c>nanargmax</c> and
        ///     <c>nanargmin</c> have this refusal.</param>
        /// <param name="a">The string array (its axis already validated).</param>
        /// <param name="axis">The <c>axis</c> argument, or null for the flattened array.</param>
        /// <returns>The ValueError to throw, or null when <paramref name="op"/> is not an arg reduction or the reduced
        ///     length is not 0.</returns>
        private static ValueError EmptyArgReductionError(string op, NDArray a, int? axis)
        {
            string name = op switch
            {
                "argmax" or "nanargmax" => "argmax",
                "argmin" or "nanargmin" => "argmin",
                _ => null,
            };
            if (name is null)
                return null;
            // A 0-d input is ravelled to one element (never empty); axis=None reduces the whole array.
            long reduced = axis is not { } ax || a.ndim == 0
                ? a.size
                : a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
            return reduced == 0 ? new ValueError($"attempt to get {name} of an empty sequence") : null;
        }

        /// <summary>
        ///     <see cref="ThrowIfReduce(string, NDArray, int?, DType)"/> for a tuple <c>axis</c>.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axes">The axes, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <exception cref="TypeError">A detailed <c>dtype=</c>, or NumPy's refusal.</exception>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">An axis appears twice.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text, or casts the text to
        ///     a numeric <paramref name="dtype"/> (a parse).</exception>
        internal static void ThrowIfReduce(string op, NDArray a, int[] axes, DType dtype = null)
        {
            bool request = IsStringRequest(dtype);
            if (!request && !IsString(a))
                return;
            PreReduceDTypeCheck(op, dtype);
            ValidateReduceAxes(op, a.ndim, axes);
            PostReduceDTypeCheck(op, dtype);
            throw ReduceAnswer(op, a, dtype, request);
        }

        /// <summary>
        ///     The <c>dtype=</c> class check a direct <c>ufunc.reduce</c> makes while parsing its arguments, before the axis
        ///     (<c>np.sum(a, axis=9, dtype='U3')</c> is the class-only TypeError, probed). The accumulations take the
        ///     descriptor as it is, and so does <c>np.trace</c>, whose C reduction receives a type NUMBER
        ///     (<c>np.trace(a, dtype='&gt;f8')</c> computes, probed).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="dtype">The request, or null.</param>
        /// <exception cref="TypeError">A detailed descriptor.</exception>
        private static void PreReduceDTypeCheck(string op, DType dtype)
        {
            if (dtype is not null && op != "trace" && AxisRuleOf(op) == ReduceAxisRule.UfuncReduce)
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
        }

        /// <summary>
        ///     The <c>dtype=</c> class check of <c>np.mean</c>'s family, whose Python layer validates the axis
        ///     (<c>_count_reduce_items</c>) before its <c>umr_sum</c> reads the request.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="dtype">The request, or null.</param>
        /// <exception cref="TypeError">A detailed descriptor.</exception>
        private static void PostReduceDTypeCheck(string op, DType dtype)
        {
            if (dtype is not null && op is "mean" or "var" or "std" or "nanmean" or "nanvar" or "nanstd")
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
        }

        /// <summary>
        ///     NumPy's answer for <c>np.percentile</c> / <c>np.quantile</c> / <c>np.nanpercentile</c> / <c>np.nanquantile</c>
        ///     of a string array, once <c>q</c> passed its range check (the caller's), in NumPy's order: the axis
        ///     (<c>_ureduce</c>), then — for the NaN forms — <c>np.isnan</c>, or <c>np.nanmean</c> for an empty input; then
        ///     the method; then the interpolation itself.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The string array.</param>
        /// <param name="method">The <c>method=</c> argument.</param>
        /// <param name="axes">The reduction axes, or null for every axis.</param>
        /// <param name="qIsScalar">True when <c>q</c> is a scalar (no leading <c>q</c> axis in the result).</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <exception cref="AxisError">An axis is out of bounds (thrown, as NumPy raises it first).</exception>
        /// <exception cref="ValueError">An axis appears twice (thrown).</exception>
        /// <exception cref="ArgumentException">An unknown method (thrown, the numeric path's text).</exception>
        /// <remarks>
        ///     <para>
        ///     An empty reduced length makes NumPy index an empty axis: a selection method's <c>take</c> raises
        ///     <c>cannot do a non-empty take from an empty axes.</c>, an interpolating method's <c>arr[0]</c> raises
        ///     <c>index 0 is out of bounds for axis 0 with size 0</c> (both IndexError, probed).
        ///     </para>
        ///     <para>
        ///     Otherwise the selection methods (<c>lower</c>, <c>higher</c>, <c>nearest</c>, <c>inverted_cdf</c>,
        ///     <c>closest_observation</c>) only partition and pick (the next phase), and the interpolating ones subtract the
        ///     neighbours: NumPy scalars when the result is a scalar (Python's <c>unsupported operand type(s) for -</c>),
        ///     arrays otherwise (<c>subtract</c>'s no-loop text).
        ///     </para>
        /// </remarks>
        internal static Exception QuantileAnswer(string op, NDArray a, string method, int[] axes, bool qIsScalar)
        {
            ValidateReduceAxes(op, a.ndim, axes);
            if (op is "nanpercentile" or "nanquantile")
                // _nanquantile_unchecked sends an empty input to np.nanmean (add.reduce); otherwise NaN removal tests isnan.
                return a.size == 0 ? AddNotCompatible(a.dtype, "reduce") : NotSupportedForInputTypes("isnan");

            var parsed = NumSharp.Statistics.QuantileEngine.ParseMethod(method);
            bool select = NumSharp.Statistics.QuantileEngine.IsDiscrete(parsed);

            long reduced = 1;
            if (axes is null)
                reduced = a.size;
            else
                foreach (int ax in axes)
                    reduced *= a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
            if (reduced == 0)
                return new IndexError(select ? "cannot do a non-empty take from an empty axes."
                    : "index 0 is out of bounds for axis 0 with size 0");

            if (select)
                return NextPhase(op, a);
            int remaining = axes is null ? 0 : a.ndim - axes.Length;
            return QuantileError(op, a, method, scalarResult: qIsScalar && remaining == 0);
        }

        /// <summary>
        ///     NumPy's answer for <c>np.average</c> of a string array, in NumPy's order: the axis tuple
        ///     (<c>normalize_axis_tuple</c> with the argument name <c>axis</c>); without weights, <c>a.mean(axis)</c> (the
        ///     add.reduce text); with weights, their shape checks, then <c>wgt.sum(dtype=result_type(a, wgt))</c> — a SIZED
        ///     string descriptor (<c>&lt;U32</c> for float weights), which a ufunc's <c>dtype=</c> refuses by class.
        /// </summary>
        /// <param name="a">The string array.</param>
        /// <param name="axes">The axes, or null.</param>
        /// <param name="weights">The weights, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <exception cref="AxisError">An axis is out of bounds (thrown; <c>axis: axis 9 is out of bounds …</c>).</exception>
        /// <exception cref="ValueError">An axis appears twice (thrown; <c>repeated axis in `axis` argument</c>).</exception>
        /// <remarks>
        ///     The weight-shape refusals keep the numeric path's house type (<see cref="ArgumentException"/>) and NumPy's texts.
        /// </remarks>
        internal static Exception AverageAnswer(NDArray a, int[] axes, NDArray weights)
        {
            ValidateReduceAxes("average", a.ndim, axes);
            if (weights is null)
                return AddNotCompatible(a.dtype, "reduce");

            if (!SameDimensions(a.Shape.dimensions ?? Array.Empty<long>(), weights.Shape.dimensions ?? Array.Empty<long>()))
            {
                if (axes is null)
                    return new ArgumentException("Axis must be specified when shapes of a and weights differ.");
                var expected = new long[axes.Length];
                for (int i = 0; i < axes.Length; i++)
                    expected[i] = a.Shape.dimensions[axes[i] < 0 ? axes[i] + a.ndim : axes[i]];
                if (!SameDimensions(expected, weights.Shape.dimensions ?? Array.Empty<long>()))
                    return new ArgumentException("Shape of weights must be consistent with shape of a along specified axis.");
            }

            // result_type(text, weights) is text of a fixed length; the weight sum's dtype= then refuses it by class.
            return new TypeError(DefaultEngine.GeneralDTypeOnlyText);
        }

        /// <summary>
        ///     NumPy's answer for a reduction called with a string <c>dtype=</c> request. A numeric input is cast to the
        ///     requested CLASS (<c>float64</c> reads as <c>&lt;U32</c>, NumPy's <c>REQUIRED_STR_LEN</c>); a string input
        ///     keeps its own descriptor, as NumPy's string add loop does.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="requested">The string descriptor requested.</param>
        /// <param name="a">The reduced array.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     The add family refuses with <c>the resolved dtypes are not compatible with add.reduce</c> (its loop output is
        ///     the concatenation, twice the input length); the multiply family names <c>(requested, input) -&gt;
        ///     requested</c>, except an <c>int64</c> input, which NumPy's string repeat promoter takes and refuses for a
        ///     missing <c>out=</c>; maximum / minimum name the same signature (probed against 2.4.2).
        /// </remarks>
        internal static Exception ReduceDTypeLoopError(string op, DType requested, NDArray a)
        {
            var req = requested.Meta;
            DType d = a.dtype;
            DType asText = d.Meta is StringDTypeMeta ? d : DTypePromotion.CastDescrToDType(d, req);
            switch (op)
            {
                case "cumsum":
                case "nancumsum":
                case "cumulative_sum":
                    return AddNotCompatible(asText, "accumulate");
                case "prod":
                case "nanprod":
                case "cumprod":
                case "nancumprod":
                case "cumulative_prod":
                    return a.typecode == NPTypeCode.Int64
                        ? MultiplyRepeatAnswer(a, a, null, null)
                        : new TypeError($"ufunc 'multiply' did not contain a loop with signature matching types ({req}, {d.Meta}) -> {req}");
                case "max":
                case "amax":
                    return new TypeError($"ufunc 'maximum' did not contain a loop with signature matching types ({req}, {d.Meta}) -> {req}");
                case "min":
                case "amin":
                    return new TypeError($"ufunc 'minimum' did not contain a loop with signature matching types ({req}, {d.Meta}) -> {req}");
                case "nanmax":
                    return new TypeError($"ufunc 'fmax' did not contain a loop with signature matching types ({req}, {d.Meta}) -> {req}");
                case "nanmin":
                    return new TypeError($"ufunc 'fmin' did not contain a loop with signature matching types ({req}, {d.Meta}) -> {req}");
                default:
                    // sum, mean, var, std, the nan forms and trace: add.reduce over the input as text.
                    return AddNotCompatible(asText, "reduce");
            }
        }
    }
}
