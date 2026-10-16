using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     NumPy 2.4.2's ufunc layer for NEP 55 <c>StringDType</c> operands (strings phase 1, P1.7d): which calls COMPUTE on
    ///     the text (phase 2 — a <see cref="StringComputationNotSupportedException"/> once NumPy's checks passed) and the
    ///     exact refusal of every other call, raised in NumPy's order. Every guard of this class (<see cref="ThrowIfUnary"/>,
    ///     <see cref="ThrowIfBinary"/>, <see cref="BinaryGuard"/>, the reduction guards) tests <see cref="IsVString"/> FIRST,
    ///     because the fixed-width answers behind them read the text as fixed-width units.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>One refusal text.</b> <c>StringDType</c> is not a legacy dtype, so NumPy's ufunc dispatcher
    ///     (<c>promote_and_get_ufuncimpl</c>) finds no loop and no legacy type resolver to fall back to: every refusal is
    ///     <c>ufunc 'X' did not contain a loop with signature matching types IN -&gt; OUT</c> naming the operand DTYPE CLASSES
    ///     (<c>&lt;class 'numpy.dtypes.StringDType'&gt;</c>, <c>&lt;class 'numpy.dtypes.Float64DType'&gt;</c>; a weak Python
    ///     scalar as <c>_PyLongDType</c> / <c>_PyFloatDType</c> / <c>_PyComplexDType</c>; a Python str as <c>StrDType</c>).
    ///     IN is the lone class of a unary ufunc and a tuple otherwise; OUT is <c>None</c>, or the requested class under
    ///     <c>dtype=</c>, and <c>(None, None)</c> for the two-output ufuncs. A reduction's signature puts the <c>out=</c> /
    ///     <c>dtype=</c> class (else <c>None</c>) before the input: <c>(None, &lt;class 'numpy.dtypes.StringDType'&gt;)</c>.
    ///     The fixed-width classes keep their own three texts (<see cref="NoLoop"/>, <see cref="NotSupportedForInputTypes"/>).
    ///     </para>
    ///     <para>
    ///     <b>The loops StringDType has</b> (probed against 2.4.2): the six comparisons over <c>(T, T)</c> / <c>(T, U)</c> /
    ///     <c>(U, T)</c> (a Python str is <c>U</c>); <c>add</c> over the same pairs; <c>multiply</c> of a <c>T</c> by any
    ///     integer (not bool); <c>maximum</c> / <c>minimum</c> over <c>(T, T)</c> only; <c>logical_and</c> / <c>or</c> /
    ///     <c>xor</c> over any operands (the truth of each string); <c>isnan</c> (the NA test). Their reductions:
    ///     <c>add</c>, <c>maximum</c> and <c>minimum</c> reduce and accumulate, but their loops are not reorderable (one axis
    ///     at most) and have no identity (an empty reduction refuses). Everything else is the refusal above.
    ///     </para>
    /// </remarks>
    internal static partial class NDStringUfuncs
    {
        /// <summary>
        ///     True when <paramref name="a"/> holds NEP 55 <c>StringDType</c> text (a <see cref="VStringArraySlice"/>). A null
        ///     operand (an omitted optional argument) is not.
        /// </summary>
        /// <param name="a">The operand, or null.</param>
        /// <returns>True for a <c>StringDType</c> array.</returns>
        /// <remarks>Test this BEFORE <see cref="IsString"/>, which refuses a <c>StringDType</c> operand loudly (its callers
        ///     are fixed-width routes).</remarks>
        internal static bool IsVString(NDArray a) => a is not null && a.typecode == NPTypeCode.String && a.Storage.IsVString;

        /// <summary>
        ///     The class NumPy's ufunc dispatcher names for an operand in a refusal: the DTypeMeta of the operand's descriptor,
        ///     or — for a C# number literal (<see cref="NDArray.WasPythonLiteral"/>), NumPy's Python scalar — the abstract
        ///     NEP 50 class of its category (<c>_PyLongDType</c>, <c>_PyFloatDType</c>, <c>_PyComplexDType</c>).
        /// </summary>
        /// <param name="x">The operand, or null for an absent slot (a reduction without <c>out=</c>).</param>
        /// <returns>NumPy's repr of the class (<c>&lt;class 'numpy.dtypes.StringDType'&gt;</c>), or <c>None</c>.</returns>
        /// <remarks>
        ///     A Python bool is NOT weak (<c>BoolDType</c>), and a 0-d ARRAY is strong whatever it holds: only the literal mark
        ///     picks the abstract class, exactly as NumPy's <c>NPY_ARRAY_WAS_PYTHON_LITERAL</c> flag does. (The numeric ufuncs
        ///     keep the house rule that reads every 0-d operand as weak for their RESULT dtype; this name is for the text.)
        /// </remarks>
        internal static string DispatchClass(NDArray x)
        {
            if (x is null)
                return "None";
            if (x.WasPythonLiteral)
                return x.Storage.GetAtIndex(0) switch
                {
                    Complex => DTypeRegistry.PyComplex.ToString(),
                    float or double => DTypeRegistry.PyFloat.ToString(),
                    _ => DTypeRegistry.PyLong.ToString(),
                };
            return x.dtype.Meta.ToString();
        }

        /// <summary>The number of outputs of NumPy's ufunc <paramref name="ufunc"/> (2 for <c>frexp</c>, <c>modf</c>, <c>divmod</c>).</summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <returns>1 or 2.</returns>
        internal static int UfuncOutputs(string ufunc) => ufunc is "frexp" or "modf" or "divmod" ? 2 : 1;

        /// <summary>True when <paramref name="loop"/> is a numeric <c>dtype=</c> request (not absent, not a string lane).</summary>
        /// <param name="loop">The requested loop, or null.</param>
        /// <returns>True for a numeric request.</returns>
        private static bool IsNumericLoop(NPTypeCode? loop) => loop is { } l && l != NPTypeCode.String && l != NPTypeCode.Empty;

        /// <summary>
        ///     The OUT part of a <c>StringDType</c> refusal under a <c>dtype=</c> request: the requested class, or
        ///     <c>None</c> without a numeric request.
        /// </summary>
        /// <param name="loop">The requested loop, or null.</param>
        /// <returns>The class repr, or <c>None</c>.</returns>
        private static string LoopOutputClass(NPTypeCode? loop)
            => IsNumericLoop(loop) ? DTypeRegistry.FromTypeCode(loop.Value).ToString() : "None";

        /// <summary>
        ///     NumPy's dispatcher refusal for <c>StringDType</c> operands (the class remarks): <c>ufunc 'X' did not contain a
        ///     loop with signature matching types IN -&gt; OUT</c>.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="inputs">The input classes (<see cref="DispatchClass"/>), in call order; one input prints bare.</param>
        /// <param name="output">The output class, or <c>None</c>.</param>
        /// <param name="nout">The ufunc's output count: two print as a tuple of <paramref name="output"/>.</param>
        /// <returns>The exception (not thrown); the house type of NumPy's UFuncTypeError.</returns>
        internal static TypeError VStringNoLoop(string ufunc, string[] inputs, string output = "None", int nout = 1)
        {
            string ins = inputs.Length == 1 ? inputs[0] : "(" + string.Join(", ", inputs) + ")";
            string outs = nout == 1 ? output : "(" + output + ", " + output + ")";
            return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {ins} -> {outs}");
        }

        /// <summary>
        ///     The output descriptor of the <c>StringDType</c> loop NumPy runs for <paramref name="ufunc"/> over these operands,
        ///     or null when it has none (the class remarks list the loops).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand (at least one operand holds <c>StringDType</c> text).</param>
        /// <param name="x2">The second operand, or null for a unary ufunc.</param>
        /// <returns><see cref="DType.Boolean"/> for the comparisons, the logical ufuncs and <c>isnan</c>; the
        ///     <c>StringDType</c> result of <c>add</c> / <c>multiply</c> / <c>maximum</c> / <c>minimum</c>; null without a
        ///     loop.</returns>
        internal static DType VStringLoopOutput(string ufunc, NDArray x1, NDArray x2)
        {
            bool t1 = IsVString(x1), t2 = IsVString(x2);
            // A Python str (a C# string literal, a 0-d U array) and a U array are StrDType, the class StringDType's
            // promoters pair with; S is BytesDType, which no StringDType loop takes.
            bool u1 = x1?.dtype.Meta == DTypeRegistry.Str, u2 = x2?.dtype.Meta == DTypeRegistry.Str;
            switch (ufunc)
            {
                case "isnan":
                    return x2 is null && t1 ? DType.Boolean : null;
                case "logical_and":
                case "logical_or":
                case "logical_xor":
                    return DType.Boolean;
                case "equal":
                case "not_equal":
                case "less":
                case "less_equal":
                case "greater":
                case "greater_equal":
                    return (t1 || u1) && (t2 || u2) ? DType.Boolean : null;
                case "add":
                    return (t1 || u1) && (t2 || u2) ? TextLoopResult(x1, x2) : null;
                case "multiply":
                    return t1 && IsRepeatOperand(x2) ? x1.dtype : t2 && IsRepeatOperand(x1) ? x2.dtype : null;
                case "maximum":
                case "minimum":
                    return t1 && t2 ? TextLoopResult(x1, x2) : null;
                default:
                    return null;
            }
        }

        /// <summary>
        ///     The <c>StringDType</c> descriptor a text loop over two operands produces — NumPy's
        ///     <c>binary_resolve_descriptors</c> after the promoter cast a <c>U</c> operand to a default <c>StringDType()</c>:
        ///     the two NA objects' compatible one (the present one, the first when both are) and the FIRST descriptor's
        ///     <c>coerce</c> (so <c>'z' + t</c> of a <c>coerce=False</c> <c>t</c> gives a coercing result, probed).
        /// </summary>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <returns>The result descriptor (used to name the loop in an output-cast refusal and to type an empty result).</returns>
        private static DType TextLoopResult(NDArray x1, NDArray x2)
        {
            var d1 = IsVString(x1) ? x1.dtype : DTypeRegistry.VString.default_descr();
            var d2 = IsVString(x2) ? x2.dtype : DTypeRegistry.VString.default_descr();
            try
            {
                return NDVStringArithmetic.BinaryOutputDescr(d1, d2);
            }
            catch (TypeError)
            {
                // Two different NA objects have no compatible one: the loop's resolution raises that next
                // (VStringNAConflict), so the first operand's descriptor names the loop here.
                return d1;
            }
        }

        /// <summary>
        ///     Whether an operand is a repeat count of <c>StringDType</c>'s <c>multiply</c> loop: an integer array of any
        ///     width (NumSharp's <c>char</c> is NumPy's uint16) or a C# integer literal — never bool.
        /// </summary>
        /// <param name="x">The operand, or null.</param>
        /// <returns>True for an integer count.</returns>
        private static bool IsRepeatOperand(NDArray x) => x is not null && x.typecode is NPTypeCode.SByte or NPTypeCode.Byte
            or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64
            or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>
        ///     NumPy's answer for a unary ufunc over a <c>StringDType</c> operand, in NumPy's order: the <c>out=</c> /
        ///     <c>where=</c> parse, then the loop resolution — <c>isnan</c> computes (the NA test, phase 2), every other
        ///     ufunc is the dispatcher's refusal.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x">The <c>StringDType</c> operand.</param>
        /// <param name="loop">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The exception to throw: <see cref="ValueError"/> for a read-only out, <see cref="ArgumentException"/>
        ///     for a non-bool mask or an output cast the <c>isnan</c> loop refuses, <see cref="TypeError"/> for the refusal;
        ///     null when <c>isnan</c> computes (<see cref="VStringIsNaN"/>).</returns>
        internal static Exception VStringUnaryAnswer(string ufunc, NDArray x, NPTypeCode? loop, NDArray @out, NDArray where)
        {
            var parse = OutWhereError(@out, where);
            if (parse is not null)
                return parse;
            // A numeric dtype= fixes the loop's output: only isnan's bool loop survives dtype=bool (probed: isnan(t,
            // dtype='f8') names Float64DType as the output).
            if (VStringLoopOutput(ufunc, x, null) is { } output && (!IsNumericLoop(loop) || loop == NPTypeCode.Boolean))
                return VStringComputeAnswer(ufunc, x, null, @out, where, output);
            return VStringNoLoop(ufunc, new[] { DispatchClass(x) }, LoopOutputClass(loop), UfuncOutputs(ufunc));
        }

        /// <summary>
        ///     NumPy's <c>isnan</c> over a <c>StringDType</c> array (<c>string_isnan_strided_loop</c>): <c>True</c> exactly for
        ///     an NA cell of a descriptor whose NA object is NaN-like, <c>False</c> everywhere else (an NA of a <c>None</c>-like
        ///     or string descriptor included), after NumPy's checks in its order.
        /// </summary>
        /// <param name="x">The <c>StringDType</c> array.</param>
        /// <param name="loop">The <c>dtype=</c> request, or null (only bool keeps the loop).</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> itself, or a fresh bool array in NumPy's output layout.</returns>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, an <paramref name="out"/> the bool output
        ///     cannot be cast to (same_kind), or operands that do not broadcast.</exception>
        /// <exception cref="TypeError">A numeric <paramref name="loop"/> other than bool (the dispatcher's refusal).</exception>
        internal static NDArray VStringIsNaN(NDArray x, NPTypeCode? loop, NDArray @out, NDArray where)
        {
            var answer = VStringUnaryAnswer("isnan", x, loop, @out, where);
            if (answer is not null)
                throw answer;
            // A fresh result takes NumPy's ufunc output layout (the input and the mask vote); with out= the flags are a
            // temporary of the input's dimensions, cast (and broadcast) into out= under where= (NumPy's output cast).
            var dims = @out is null ? BroadcastDims(x, null, null, where) : (long[])x.Shape.dimensions.Clone();
            var shape = dims.Length == 0 ? Shape.NewScalar()
                : Shape.NpyIterOutputShape(dims, where is null || @out is not null ? new[] { x.Shape } : new[] { x.Shape, where.Shape });
            var flags = new NDArray(NPTypeCode.Boolean, shape, true);
            try
            {
                // Zero-filled: a descriptor without a NaN-like NA has no True cell. Masked-off elements of a fresh result are
                // unspecified in NumPy (uninitialized); here they hold the flag of their own element.
                if (flags.size != 0 && x.dtype.VString.has_nan_na)
                    NDVStringArithmetic.FillIsNaN(x, flags);
                if (@out is null)
                    return flags;
                np.copyto(@out, flags, "unsafe", where);
                return @out;
            }
            finally
            {
                if (@out is not null)
                    flags.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's answer for a binary ufunc with a <c>StringDType</c> operand, in NumPy's order: the <c>out=</c> /
        ///     <c>where=</c> parse, the loop resolution (a numeric <c>dtype=</c> keeps only a bool loop under
        ///     <c>dtype=bool</c>), then for a loop that exists the output cast into <c>out=</c> and the broadcast before the
        ///     computation (phase 2).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="loop">The <c>dtype=</c> request, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="computed">The computing loop's output descriptor when every check passed (the answer is then null and
        ///     the caller computes, or builds an empty result from it when the broadcast is empty); null for a refusal.</param>
        /// <returns>The exception to throw, or null when the call computes (<paramref name="computed"/> is then set).</returns>
        internal static Exception VStringBinaryAnswer(string ufunc, NDArray x1, NDArray x2, NDArray @out, NPTypeCode? loop,
            NDArray where, out DType computed)
        {
            computed = null;
            var parse = OutWhereError(@out, where);
            if (parse is not null)
                return parse;
            if (VStringLoopOutput(ufunc, x1, x2) is { } output
                && (!IsNumericLoop(loop) || loop == NPTypeCode.Boolean && output.typecode == NPTypeCode.Boolean))
            {
                var answer = VStringComputeAnswer(ufunc, x1, x2, @out, where, output);
                if (answer is null)
                    computed = output;
                return answer;
            }
            return VStringNoLoop(ufunc, new[] { DispatchClass(x1), DispatchClass(x2) }, LoopOutputClass(loop), UfuncOutputs(ufunc));
        }

        /// <summary>
        ///     Runs the <c>StringDType</c> loop of a binary ufunc whose checks all passed (<see cref="VStringBinaryAnswer"/>
        ///     returned null) over a non-empty broadcast: the six comparisons (<see cref="NDVStringComparison"/>), the logical
        ///     ufuncs over the operands' truth (the bool cast of each text operand, then the bool loop), the concatenation,
        ///     the repetition and the elementwise <c>maximum</c> / <c>minimum</c> (<see cref="NDVStringArithmetic"/>).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="output">The loop's output descriptor (bool, or the <c>StringDType</c> result).</param>
        /// <returns><paramref name="out"/> itself, or the fresh result.</returns>
        /// <exception cref="ValueError">An NA the loop's rule refuses (an ordering or a concatenation of a <c>None</c>-like
        ///     NA, NumPy's texts).</exception>
        /// <exception cref="TypeError">A repetition of a <c>None</c>-like NA (<c>Cannot multiply null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OverflowException">A repetition past <c>2^63 - 1</c> bytes (<c>Overflow encountered in string
        ///     multiply</c>).</exception>
        /// <exception cref="InvalidOperationException">A ufunc without a computing <c>StringDType</c> loop (a routing bug).</exception>
        private static NDArray VStringCompute(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where, DType output)
        {
            // A bool loop into a TEXT out= (S / U / T): NumPy's output cast formats the bools ('True' / 'False'); the bools
            // are computed into a temporary and cast into out= under where= (the guard checked the cast and the broadcast).
            bool boolLoop = output.typecode == NPTypeCode.Boolean;
            bool textOut = boolLoop && @out is not null && @out.typecode == NPTypeCode.String;
            if (NDStringComparison.IsComparison(ufunc))
                return textOut
                    ? NDStringArithmetic.CastInto(@out, where, () => NDVStringComparison.Execute(ufunc, x1, x2, null, null))
                    : NDVStringComparison.Execute(ufunc, x1, x2, @out, where);
            switch (ufunc)
            {
                case "logical_and":
                case "logical_or":
                case "logical_xor":
                    return textOut
                        ? NDStringArithmetic.CastInto(@out, where, () => LogicalOverTruth(ufunc, x1, x2, null, null))
                        : LogicalOverTruth(ufunc, x1, x2, @out, where);
                case "add":
                    return NDVStringArithmetic.Add(x1, x2, @out, where, output);
                case "multiply":
                    return NDVStringArithmetic.Multiply(x1, x2, @out, where, output);
                case "maximum":
                case "minimum":
                    return NDVStringArithmetic.MaximumMinimum(ufunc == "maximum", x1, x2, @out, where, output);
                default:
                    throw ComputingLoopMisrouted(ufunc);
            }
        }

        /// <summary>
        ///     The checks NumPy makes between finding a <c>StringDType</c> loop and running it, in its order: the loop's
        ///     descriptor resolution (two NA objects must be compatible — the comparison, <c>add</c>, <c>maximum</c> and
        ///     <c>minimum</c> loops refuse two different ones), the cast of the loop's output into <c>out=</c> (same_kind), then
        ///     the broadcast of every operand.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="output">The loop's output descriptor.</param>
        /// <returns>The NA refusal (<see cref="TypeError"/>, <c>Cannot find a compatible null string value for null strings
        ///     'None' and 'nan'</c>), the output-cast refusal (<see cref="ArgumentException"/>, NumPy's UFuncTypeError text),
        ///     the broadcast refusal, or null when the loop computes.</returns>
        /// <remarks>
        ///     The output descriptor of a <c>StringDType</c> loop (<c>add</c>, <c>multiply</c>, <c>maximum</c>,
        ///     <c>minimum</c>) depends on <paramref name="out"/>: NumPy's <c>resolve_descriptors</c> hands every operand to the
        ///     loop through <c>PyArray_CastDescrToDType(descr, StringDType)</c>, so a <c>StringDType</c> out= IS the loop's
        ///     output (its own NA object and coerce, no cast to check), while any other out= becomes the cast's default
        ///     <c>StringDType()</c> — which is what the refusal names whatever NA object the inputs carry (probed:
        ///     <c>Cannot cast ufunc 'add' output from StringDType() to dtype('float64') with casting rule 'same_kind'</c>
        ///     for <c>na_object=nan</c> inputs).
        /// </remarks>
        private static Exception VStringComputeAnswer(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where, DType output)
        {
            if (VStringNAConflict(ufunc, x1, x2) is { } conflict)
                return conflict;
            if (@out is not null)
            {
                bool textLoop = output.Meta == DTypeRegistry.VString;
                // A StringDType out= is the text loop's own output descriptor: nothing is cast.
                if (!(textLoop && IsVString(@out)))
                {
                    var produced = textLoop ? DTypeRegistry.VString.default_descr() : output;
                    if (!DTypeCasting.CanCastTypeTo(produced, @out.dtype, NPY_CASTING.NPY_SAME_KIND_CASTING))
                        return new ArgumentException(
                            $"Cannot cast ufunc '{ufunc}' output from {produced.ToString(true)} to {@out.dtype.ToString(true)} with casting rule 'same_kind'");
                }
            }
            return BroadcastError(x1, x2, @out, where);
        }

        /// <summary>
        ///     NumPy's <c>stringdtype_compatible_na</c> check in the resolve_descriptors of the comparison, <c>add</c>,
        ///     <c>maximum</c> and <c>minimum</c> loops: two <c>StringDType</c> operands whose NA objects are both present and
        ///     unequal (by NumPy's <c>na_eq_cmp</c>) have no loop to run. A <c>U</c> operand is cast to a default descriptor
        ///     (no NA object) first, so it never conflicts; the other loops (<c>multiply</c>, the logical ufuncs, <c>isnan</c>)
        ///     have one <c>StringDType</c> operand at most.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <returns>NumPy's <see cref="TypeError"/> (<c>Cannot find a compatible null string value for null strings 'None'
        ///     and 'nan'</c>, each NA object as its <c>repr()</c>), or null.</returns>
        private static TypeError VStringNAConflict(string ufunc, NDArray x1, NDArray x2)
        {
            if (!IsVString(x1) || !IsVString(x2))
                return null;
            if (!(NDStringComparison.IsComparison(ufunc) || ufunc is "add" or "maximum" or "minimum"))
                return null;
            var p1 = x1.dtype.VString;
            var p2 = x2.dtype.VString;
            if (VStringParameters.CompatibleNA(p1, p2, out _, out _))
                return null;
            return new TypeError(
                $"Cannot find a compatible null string value for null strings '{NDPyObject.Repr(p1.na_object)}' and " +
                $"'{NDPyObject.Repr(p2.na_object)}'");
        }

        /// <summary>
        ///     NumPy's answer for a reduction-style function over a <c>StringDType</c> array once the axis was validated: the
        ///     reduction ufunc's loop resolution (<c>multiply</c>, <c>fmax</c>, <c>fmin</c> have no <c>StringDType</c> loop: the
        ///     dispatcher's refusal with <c>None</c> — or the <c>dtype=</c> class — in the output slot; a <c>dtype=</c> request
        ///     refuses every loop, the <c>StringDType</c> loops producing only <c>StringDType</c>; a <c>StringDType</c>
        ///     <c>out=</c> must have an NA object compatible with the input's), then for <c>add</c> / <c>maximum</c> /
        ///     <c>minimum</c> the reduction's own checks (one axis at most: the loop is not reorderable; an empty reduction with
        ///     a non-empty result has no identity). Null when the reduction computes.
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"sum"</c>, <c>"prod"</c>, <c>"max"</c>, <c>"cumsum"</c>, ...).</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The reduced axes, normalized (null for every axis).</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> array, or null: its class takes the reduction signature's first slot.</param>
        /// <returns>The exception to throw, or null when NumPy computes the call (<see cref="NDVStringReduction"/>: the sum /
        ///     max / min / accumulation and the arg-reductions return; <c>np.mean</c>'s family, <c>np.var</c>'s and
        ///     <c>np.ptp</c> compute their sum / extremes and then meet NumPy's later refusal). <c>np.median</c> sorts first
        ///     (strings plan P2.2 over <c>StringDType</c>): an empty input reaches its mean's refusal, anything else is still
        ///     deferred.</returns>
        internal static Exception VStringReduceAnswer(string op, NDArray a, int[] axes, DType dtype, NDArray @out = null)
        {
            if (VStringReduceLoopAnswer(op, a, dtype, @out) is { } refusal)
                return refusal;
            string ufunc = ReduceUfuncOf(op, out bool accumulate);
            if (ufunc is null)
                return null;
            if (op == "median")
                // np.median sorts (computes) before its mean; only an empty input reaches the mean's add.reduce refusal.
                return a.size == 0 && axes is null
                    ? new ValueError($"zero-size array to reduction operation {ufunc} which has no identity")
                    : NextPhaseVString(op, a.dtype);
            if (accumulate)
                return null;
            return (Exception)VStringNotReorderable(ufunc, a, axes) ?? VStringNoIdentity(ufunc, a, axes);
        }

        /// <summary>
        ///     The loop resolution part of <see cref="VStringReduceAnswer"/> (<c>reducelike_promote_and_resolve</c>, which
        ///     NumPy runs before <c>PyUFunc_ReduceWrapper</c>'s own checks): <c>multiply</c>, <c>fmax</c> and <c>fmin</c> have
        ///     no <c>StringDType</c> loop, a <c>dtype=</c> request has none either, an <c>out=</c> of another class has none
        ///     (<c>add</c> pairs <c>StringDType</c> with <c>StrDType</c>, <c>maximum</c> / <c>minimum</c> with
        ///     <c>StringDType</c> only), and a <c>StringDType</c> <c>out=</c> must have an NA object compatible with the
        ///     input's.
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"sum"</c>, <c>"max"</c>, <c>"cumsum"</c>, ...).</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> array, or null: its class takes the reduction signature's first slot.</param>
        /// <returns>The refusal, or null when a <c>StringDType</c> loop resolves (also for the arg-reductions, which compute
        ///     without a ufunc). A function that sorts first (the median family, the sorts and searches) is P2.2's
        ///     <c>StringDType</c> part: its deferral.</returns>
        /// <remarks>
        ///     The keyword route of <c>np.max</c> / <c>np.min</c> (<see cref="NDVStringReduction.MaxMinNumPy"/>) and
        ///     <c>np.trace</c>'s reduction (<see cref="NDVStringReduction.SumNumPy"/>) run NumPy's <c>out=</c> /
        ///     <c>where=</c> / <c>initial=</c> checks after this and <see cref="VStringNotReorderable"/>, and before
        ///     <see cref="VStringNoIdentity"/> — exactly where <c>PyUFunc_ReduceWrapper</c> runs them.
        /// </remarks>
        internal static Exception VStringReduceLoopAnswer(string op, NDArray a, DType dtype, NDArray @out)
        {
            string ufunc = ReduceUfuncOf(op, out bool accumulate);
            if (ufunc is null)
                // The arg-reductions compute (the dtype's ArgFunc); the sorts and searches are P2.2's StringDType part.
                return op is "argmax" or "argmin" or "nanargmax" or "nanargmin" ? null : NextPhaseVString(op, a.dtype);
            string input = a.dtype.Meta.ToString();
            // The reduction signature's first slot: dtype= wins, then the out's class, else None.
            string first = dtype is not null ? dtype.Meta.ToString() : @out is not null ? DispatchClass(@out) : "None";
            bool loopExists = ufunc is "add" or "maximum" or "minimum";
            if (!loopExists)
                return VStringNoLoop(ufunc, new[] { first, input }, dtype is not null ? first : "None");
            if (dtype is not null)
                return VStringNoLoop(ufunc, new[] { first, input }, first);
            // An out= of another class than the loop's operands has no loop either: add pairs StringDType with StrDType,
            // maximum / minimum with StringDType only.
            if (@out is not null && !IsVString(@out) && !(ufunc == "add" && @out.dtype.Meta == DTypeRegistry.Str))
                return VStringNoLoop(ufunc, new[] { first, input });
            // The loop's descriptor resolution over (out, input, out): two NA objects must be compatible (the out first).
            if (IsVString(@out) && !VStringParameters.CompatibleNA(@out.dtype.VString, a.dtype.VString, out _, out _))
                return new TypeError(
                    $"Cannot find a compatible null string value for null strings '{NDPyObject.Repr(@out.dtype.VString.na_object)}' " +
                    $"and '{NDPyObject.Repr(a.dtype.VString.na_object)}'");
            // add.accumulate's enforce_uniform_args: the accumulator descriptor (the out's) must be equivalent to the input's,
            // and NumPy's string-to-string cast from a descriptor WITH an NA object to one without is unsafe (lossy), so an
            // out with an NA object over an input without one is refused (the reverse is a no-cast). Any other out is the
            // cast's default StringDType(), which is equivalent to every input.
            if (accumulate && IsVString(@out) && @out.dtype.VString.HasNAObject && !a.dtype.VString.HasNAObject)
            {
                string o = @out.dtype.ToString(true);
                return new TypeError(
                    $"the resolved dtypes are not compatible with add.accumulate. Resolved ({o}, {a.dtype.ToString(true)}, {o})");
            }
            return null;
        }

        /// <summary>
        ///     <c>PyUFunc_ReduceWrapper</c>'s first check: the <c>StringDType</c> <c>add</c> / <c>maximum</c> /
        ///     <c>minimum</c> loops are not reorderable (their result depends on the order the elements meet), so a reduction
        ///     over two or more axes is refused before anything is read.
        /// </summary>
        /// <param name="ufunc">The ufunc NumPy names.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axes">The reduced axes, normalized (null for every axis).</param>
        /// <returns>The <see cref="ValueError"/> (<c>reduction operation 'add' is not reorderable, so at most one axis may
        ///     be specified</c>), or null.</returns>
        internal static ValueError VStringNotReorderable(string ufunc, NDArray a, int[] axes)
            => (axes?.Length ?? a.ndim) > 1
                ? new ValueError($"reduction operation '{ufunc}' is not reorderable, so at most one axis may be specified")
                : null;

        /// <summary>
        ///     <c>PyArray_CopyInitialReduceValues</c>' refusal (a reduction without an identity and without <c>initial=</c>):
        ///     an empty reduced extent with a non-empty result has no first element to start from.
        /// </summary>
        /// <param name="ufunc">The ufunc NumPy names.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axes">The reduced axes, normalized (null for every axis).</param>
        /// <returns>The <see cref="ValueError"/> (<c>zero-size array to reduction operation add which has no
        ///     identity</c>), or null.</returns>
        internal static ValueError VStringNoIdentity(string ufunc, NDArray a, int[] axes)
        {
            int nd = a.ndim;
            long reduced = 1, result = 1;
            var dims = a.Shape.dimensions ?? Array.Empty<long>();
            for (int i = 0; i < nd; i++)
            {
                if (axes is null || Array.IndexOf(axes, i) >= 0)
                    reduced *= dims[i];
                else
                    result *= dims[i];
            }
            return reduced == 0 && result != 0
                ? new ValueError($"zero-size array to reduction operation {ufunc} which has no identity")
                : null;
        }

        /// <summary>
        ///     The guard of a <c>StringDType</c> reduction that COMPUTES: runs NumPy's checks in its order (a direct
        ///     <c>ufunc.reduce</c>'s <c>dtype=</c> class check, the axis, <c>np.mean</c>'s family's later class check, an empty
        ///     arg-reduction, then <see cref="VStringReduceAnswer"/>) and reports whether the caller computes.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> array, or null (its read-only check runs where NumPy runs it: before the axis
        ///     for a direct <c>ufunc.reduce</c>, after it otherwise).</param>
        /// <param name="loopOnly">True to stop after the loop resolution (<see cref="VStringReduceLoopAnswer"/>): the caller
        ///     runs <c>PyUFunc_ReduceWrapper</c>'s checks itself, with the out's shape check between the reorderable and the
        ///     identity checks (<see cref="NDVStringReduction.SumNumPy"/>).</param>
        /// <returns>False for an input that is not <c>StringDType</c> (the caller's own path continues); true when the
        ///     <c>StringDType</c> reduction computes.</returns>
        /// <exception cref="ValueError">A read-only out, an arg-reduction over an empty line, an N-D reduction over every
        ///     axis (not reorderable), or an empty reduction without identity.</exception>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="TypeError">NumPy's refusal (a <c>dtype=</c> request, an out of another class, two incompatible
        ///     NA objects).</exception>
        /// <exception cref="StringComputationNotSupportedException"><c>np.median</c> of a non-empty input (P2.2 over
        ///     <c>StringDType</c>).</exception>
        internal static bool VStringReduces(string op, NDArray a, int? axis, DType dtype = null, NDArray @out = null, bool loopOnly = false)
        {
            if (!IsVString(a))
                return false;
            // np.trace's C reduction reaches the ufunc after its diagonal's axis checks (the caller's), so its dtype= check
            // and its out's read-only check both come after the axis, in that order.
            bool outFirst = AxisRuleOf(op) == ReduceAxisRule.UfuncReduce && op != "trace";
            if (@out is not null && outFirst && !@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            PreReduceDTypeCheck(op, dtype);
            ValidateReduceAxis(op, a.ndim, axis);
            PostReduceDTypeCheck(op, dtype);
            if (@out is not null && !outFirst && !@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            if (EmptyArgReductionError(op, a, axis) is { } emptyArg)
                throw emptyArg;
            var refusal = loopOnly
                ? VStringReduceLoopAnswer(op, a, dtype, @out)
                : VStringReduceAnswer(op, a, ReducedAxes(a, axis), dtype, @out);
            if (refusal is not null)
                throw refusal;
            return true;
        }

        /// <summary>
        ///     <see cref="VStringReduces(string, NDArray, int?, DType, NDArray, bool)"/> for a tuple <c>axis</c>: the read-only
        ///     out (a direct <c>ufunc.reduce</c> parses it first), every axis checked and then repeats, the out's read-only
        ///     check of the functions that meet it later, then <see cref="VStringReduceAnswer"/> (two or more axes are not
        ///     reorderable; an empty tuple reduces nothing).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The reduced array.</param>
        /// <param name="axes">The tuple (not null).</param>
        /// <param name="out">The <c>out=</c> array, or null.</param>
        /// <returns>False for an input that is not <c>StringDType</c>; true when the <c>StringDType</c> reduction computes.</returns>
        /// <exception cref="ValueError">A read-only out, a repeated axis, two or more axes (not reorderable), or an empty
        ///     reduction without identity.</exception>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="TypeError">NumPy's loop refusal.</exception>
        internal static bool VStringReducesAxes(string op, NDArray a, int[] axes, NDArray @out = null)
        {
            if (!IsVString(a))
                return false;
            bool outFirst = AxisRuleOf(op) == ReduceAxisRule.UfuncReduce;
            if (@out is not null && outFirst && !@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            ValidateReduceAxes(op, a.ndim, axes);
            if (@out is not null && !outFirst && !@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            if (VStringReduceAnswer(op, a, ReducedAxes(a, axes), null, @out) is { } refusal)
                throw refusal;
            return true;
        }

        /// <summary>
        ///     The reduced axes of an integer <c>axis</c> argument, normalized: null for every axis (and for a 0-d input,
        ///     whose axis 0 / -1 NumPy's <c>ufunc.reduce</c> reads as every axis).
        /// </summary>
        /// <param name="a">The reduced array (its axis already validated).</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <returns>The axes, or null.</returns>
        internal static int[] ReducedAxes(NDArray a, int? axis)
        {
            if (axis is not { } ax || a.ndim == 0)
                return null;
            return new[] { ax < 0 ? ax + a.ndim : ax };
        }

        /// <summary>
        ///     The reduced axes of a tuple <c>axis</c> argument, normalized (null for every axis).
        /// </summary>
        /// <param name="a">The reduced array (its axes already validated).</param>
        /// <param name="axes">The axes, or null.</param>
        /// <returns>The axes, or null.</returns>
        internal static int[] ReducedAxes(NDArray a, int[] axes)
        {
            if (axes is null || a.ndim == 0)
                return null;
            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                normalized[i] = axes[i] < 0 ? axes[i] + a.ndim : axes[i];
            return normalized;
        }

        /// <summary>
        ///     The shape-decided guard of a function that computes on <c>StringDType</c> text (the truth tests, the counts, the
        ///     index searches, the sorts and arg-reductions): NumPy's checks first (the axis, the empty arg-reduction, the 0-d
        ///     <c>nonzero</c>), then the bool stand-in when the shape decides the answer (an empty input, a line of one
        ///     element), else the phase-2 deferral.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <returns>The bool stand-in the caller's numeric path runs on (owned by the caller).</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">An arg-reduction over an empty line, or <c>nonzero</c> of a 0-d array.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on the text (phase 2).</exception>
        private static NDArray VStringStandIn(string op, NDArray a, int? axis)
        {
            ValidateReduceAxis(op, a.ndim, axis);
            if (EmptyArgReductionError(op, a, axis) is { } empty)
                throw empty;
            if (op is "nonzero" && a.ndim == 0)
                throw new ValueError(NonZero0dMessage);
            if (ShapeDecides(op, a, axis))
                return BoolStandIn(a);
            throw NextPhaseVString(op, a.dtype);
        }

        /// <summary>
        ///     NumPy's <c>isclose</c> (behind <c>np.isclose</c> / <c>np.allclose</c>) with a <c>StringDType</c> operand: <c>y</c>
        ///     is promoted with a weak Python float first (a <c>StringDType</c> <c>y</c> raises NumPy's
        ///     <see cref="DTypePromotionError"/>), then <c>x - y</c> meets the dispatcher's refusal.
        /// </summary>
        /// <param name="a">The <c>a</c> argument (NumPy's <c>x</c>).</param>
        /// <param name="b">The <c>b</c> argument (NumPy's <c>y</c>).</param>
        /// <returns>The exception to throw.</returns>
        /// <exception cref="DTypePromotionError"><paramref name="b"/> holds <c>StringDType</c> text (raised by the
        ///     promotion itself, with NumPy's text).</exception>
        private static Exception VStringIsCloseAnswer(NDArray a, NDArray b)
        {
            var y = DTypePromotion.ResultType(b, 1.0);
            return VStringNoLoop("subtract", new[] { DispatchClass(a), y.Meta.ToString() });
        }

        /// <summary>
        ///     NumPy's refusal of a <c>StringDType</c> operand in the BLAS-style products (<c>dot</c>, <c>inner</c>,
        ///     <c>vdot</c>, <c>tensordot</c>, <c>convolve</c>, <c>correlate</c>): their C entry points accept only legacy
        ///     descriptors (<c>PyArray_DescrFromType</c> of the operands' common type number), and a <c>StringDType</c> operand
        ///     stops them before any shape check (probed: <c>np.dot</c> of misaligned <c>StringDType</c> vectors gives this
        ///     text, not the alignment error). The message is two lines, NumPy's typo (<c>arbitraryuser</c>) included.
        /// </summary>
        /// <param name="descr">The <c>StringDType</c> descriptor NumPy names.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError VStringProductError(DType descr)
            => new TypeError(
                "This function currently only supports native NumPy dtypes and old-style user dtypes, but the dtype was " +
                $"{descr}.\n(The function may need to be updated to support arbitraryuser dtypes.)");

        /// <summary>
        ///     The descriptor lookup NumPy's legacy products start with (<c>PyArray_ObjectType(op1, NPY_NOTYPE)</c>, then
        ///     <c>PyArray_ObjectType(op2, typenum)</c>) when one operand holds <c>StringDType</c> text: the refusal of a
        ///     descriptor that is not legacy, or null when NumPy goes on with an OBJECT array.
        /// </summary>
        /// <param name="a">The first operand (<c>op1</c>).</param>
        /// <param name="b">The second operand (<c>op2</c>; for <c>np.convolve</c> the reversed kernel, of the same dtype).</param>
        /// <returns>The legacy-only <see cref="TypeError"/>, or null for the OBJECT fallback.</returns>
        /// <remarks>
        ///     A <c>StringDType</c> first operand is refused at once. A <c>StringDType</c> second operand is discovered by
        ///     PROMOTING the first operand's descriptor with its own: a <c>U</c> first operand (<c>StrDType</c>) gives a
        ///     <c>StringDType</c>, refused the same way (named by the promoted descriptor); every other class has no common
        ///     DType with <c>StringDType</c>, and the legacy discovery swallows that failure and continues with OBJECT
        ///     (<c>handle_promotion</c>'s <c>PROMOTION_FAILED</c> branch), so the product runs over Python objects.
        /// </remarks>
        private static TypeError LegacyProductRefusal(NDArray a, NDArray b)
        {
            if (IsVString(a))
                return VStringProductError(a.dtype);
            DType promoted;
            try
            {
                promoted = DTypePromotion.PromoteTypes(a.dtype, b.dtype);
            }
            catch (Exception e) when (e is DTypePromotionError or TypeError)
            {
                // No common DType: NumPy's discovery falls back to OBJECT instead of raising.
                return null;
            }
            return VStringProductError(promoted);
        }

        /// <summary>
        ///     NumPy's answer for <c>np.dot</c> / <c>np.inner</c> with a <c>StringDType</c> operand, in the order of
        ///     <c>PyArray_MatrixProduct2</c> (which <c>np.inner</c> reaches after the same descriptor lookup, with
        ///     <paramref name="b"/> already transposed).
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand (for <c>np.inner</c>: its last two axes swapped, as NumPy passes it).</param>
        /// <param name="out">The <c>out=</c> array of <c>np.dot</c>, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     <para>The descriptor lookup first (<see cref="LegacyProductRefusal"/>): a <c>StringDType</c> first operand, or a
        ///     <c>U</c> one beside it, is the legacy-only refusal.</para>
        ///     <para>Otherwise NumPy multiplies Python objects. A 0-d operand makes the product the OBJECT <c>multiply</c> ufunc
        ///     (<c>out=</c> parsed and cast from <c>dtype('O')</c> first); otherwise the contracted lengths must align, an
        ///     <c>out=</c> must be an object array (<c>new_array_for_sum</c>), and <c>OBJECT_dot</c> multiplies
        ///     <c>a[0, ..., 0] * b[0, ..., 0]</c> first. That first product decides by the Python types alone: a float /
        ///     complex / bytes / None beside a str raises Python's <see cref="TypeError"/> (<c>can't multiply sequence by
        ///     non-int of type 'float'</c>, <c>unsupported operand type(s) for *: 'float' and 'NoneType'</c>). A product
        ///     that does not raise (an int repeating the text) builds an object array, which NumSharp does not have: a
        ///     <see cref="NotSupportedException"/>.</para>
        /// </remarks>
        internal static Exception VStringProductAnswer(NDArray a, NDArray b, NDArray @out)
        {
            if (LegacyProductRefusal(a, b) is { } refusal)
                return refusal;

            // `np.multiply(ap1, ap2, out)` over the object arrays: the out= parse, its cast from object, then the products.
            if (a.ndim == 0 || b.ndim == 0)
            {
                if (@out is not null)
                    return OutWhereError(@out, null) ?? new ArgumentException(
                        $"Cannot cast ufunc 'multiply' output from dtype('O') to {@out.dtype.ToString(true)} with casting rule 'same_kind'");
                // A 0-d operand broadcasts against anything; an empty other operand leaves no product to make.
                if (a.size == 0 || b.size == 0)
                    return ObjectArrayRefusal("dot", a, b);
                return (Exception)FirstProductError(a, 0, b, 0) ?? ObjectArrayRefusal("dot", a, b);
            }

            // dot_alignment_error: a's last axis against b's second-to-last (its only axis when 1-D).
            long l = a.Shape.dimensions[a.ndim - 1];
            int matchDim = b.ndim > 1 ? b.ndim - 2 : 0;
            if (b.Shape.dimensions[matchDim] != l)
                return new IncorrectShapeException(
                    $"shapes {a.Shape.ToPythonTuple()} and {b.Shape.ToPythonTuple()} not aligned: {l} (dim {a.ndim - 1}) " +
                    $"!= {b.Shape.dimensions[matchDim]} (dim {matchDim})");

            // new_array_for_sum: an out= must be an OBJECT C-array of the result's rank, which no NumSharp array is.
            if (@out is not null)
                return new ValueError(
                    "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");

            // OBJECT_dot runs once per result element, its first product being the first elements of both operands; an
            // empty result or an empty contraction multiplies nothing (NumPy then returns an object array of zeros / None).
            long resultSize = 1;
            for (int i = 0; i < a.ndim - 1; i++)
                resultSize *= a.Shape.dimensions[i];
            for (int i = 0; i < b.ndim; i++)
                if (i != matchDim)
                    resultSize *= b.Shape.dimensions[i];
            if (l > 0 && resultSize > 0 && FirstProductError(a, 0, b, 0) is { } first)
                return first;
            return ObjectArrayRefusal("dot", a, b);
        }

        /// <summary>
        ///     NumPy's answer for <c>np.convolve</c> / <c>np.correlate</c> with a <c>StringDType</c> operand once its earlier
        ///     checks passed (for <c>np.convolve</c>: the two emptiness checks, the swap by length and the mode; for
        ///     <c>np.correlate</c>: the mode), in the order of <c>PyArray_Correlate</c> / <c>PyArray_Correlate2</c>.
        /// </summary>
        /// <param name="a">The first operand (for <c>np.convolve</c>: the longer one, after NumPy's swap).</param>
        /// <param name="v">The second operand (for <c>np.convolve</c>: the kernel, which NumPy reverses).</param>
        /// <param name="mode">The parsed mode: 0 valid, 1 same, 2 full.</param>
        /// <param name="convolve">True for <c>np.convolve</c> (<c>multiarray.correlate(a, v[::-1], mode)</c>), false for
        ///     <c>np.correlate</c> (<c>multiarray.correlate2(a, v, mode)</c>).</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     The descriptor lookup first (<see cref="LegacyProductRefusal"/>), then for the OBJECT fallback
        ///     <c>PyArray_FromAny(op, OBJECT, 1, 1)</c> (one dimension each: <c>object of too small depth for desired
        ///     array</c> / <c>object too deep for desired array</c>, the first operand first), <c>_pyarray_correlate</c>'s
        ///     emptiness checks (correlate only: convolve made them in Python), the swap that makes <c>ap1</c> the longer
        ///     operand (correlate only), and <c>OBJECT_dot</c>'s first product <c>ap1[0] * ap2[n_left]</c> (<c>n_left</c> 0 /
        ///     <c>n // 2</c> / <c>n - 1</c> for valid / same / full; convolve's <c>ap2</c> is the reversed kernel). A product
        ///     that does not raise builds an object array: a <see cref="NotSupportedException"/>.
        /// </remarks>
        internal static Exception VStringSlidingAnswer(NDArray a, NDArray v, int mode, bool convolve)
        {
            if (LegacyProductRefusal(a, v) is { } refusal)
                return refusal;
            if (OneDimensionalError(a) is { } depthA)
                return depthA;
            if (OneDimensionalError(v) is { } depthV)
                return depthV;

            long n1 = a.size, n2 = v.size;
            if (!convolve)
            {
                if (n1 == 0)
                    return new ValueError("first array argument cannot be empty");
                if (n2 == 0)
                    return new ValueError("second array argument cannot be empty");
            }
            // _pyarray_correlate makes ap1 the longer operand (np.convolve swapped in Python already, so it never swaps).
            NDArray ap1 = a, ap2 = v;
            if (n1 < n2)
            {
                ap1 = v;
                ap2 = a;
                (n1, n2) = (n2, n1);
            }
            long nLeft = mode switch { 0 => 0, 1 => n2 / 2, _ => n2 - 1 };
            // np.convolve's ap2 is v[::-1]: its element n_left is v's element n2 - 1 - n_left.
            long kernelIndex = convolve ? n2 - 1 - nLeft : nLeft;
            string op = convolve ? "convolve" : "correlate";
            return (Exception)FirstProductError(ap1, 0, ap2, kernelIndex) ?? ObjectArrayRefusal(op, a, v);
        }

        /// <summary>
        ///     <c>PyArray_FromAny(op, typec, 1, 1, ...)</c>'s depth check of a sliding product's operand.
        /// </summary>
        /// <param name="operand">The operand.</param>
        /// <returns>The <see cref="ValueError"/> for a 0-d operand (<c>object of too small depth for desired array</c>) or
        ///     for two or more dimensions (<c>object too deep for desired array</c>), else null.</returns>
        private static ValueError OneDimensionalError(NDArray operand)
            => operand.ndim == 0 ? new ValueError("object of too small depth for desired array")
                : operand.ndim > 1 ? new ValueError("object too deep for desired array")
                : null;

        /// <summary>
        ///     The first product of an OBJECT product (<c>PyNumber_Multiply(x, y)</c>), decided by the Python types of the two
        ///     elements as NumPy's object conversion yields them.
        /// </summary>
        /// <param name="x">The array holding the left factor.</param>
        /// <param name="xi">The left factor's logical (C-order) index.</param>
        /// <param name="y">The array holding the right factor.</param>
        /// <param name="yi">The right factor's logical (C-order) index.</param>
        /// <returns>Python's <see cref="TypeError"/>, or null when the product is defined (NumPy then computes on).</returns>
        private static TypeError FirstProductError(NDArray x, long xi, NDArray y, long yi)
        {
            string message = PyMultiplyError(PyObjectTypeName(x, xi), PyObjectTypeName(y, yi));
            return message is null ? null : new TypeError(message);
        }

        /// <summary>
        ///     The Python type of an element of <paramref name="arr"/> after NumPy's cast to OBJECT: <c>bool</c>, <c>int</c>
        ///     (every integer width, NumSharp's <c>char</c> included), <c>float</c> (float16 / float32 / float64, and
        ///     NumSharp's <c>decimal</c>), <c>complex</c>, <c>str</c> (a <c>U</c> element, a <c>StringDType</c> text) or
        ///     <c>bytes</c> (an <c>S</c> element); a <c>StringDType</c> NA element is its NA object (<c>NoneType</c>,
        ///     <c>float</c> for a NaN, <c>str</c> for a string sentinel).
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="index">The element's logical (C-order) index.</param>
        /// <returns>The Python type name.</returns>
        private static string PyObjectTypeName(NDArray arr, long index)
        {
            if (IsVString(arr))
                return arr.GetAtIndex(index) switch
                {
                    string => "str",
                    null => "NoneType",
                    double or float => "float",
                    var na => Unmanaged.NDBuffer.PythonTypeName(na),
                };
            switch (arr.typecode)
            {
                case NPTypeCode.Boolean:
                    return "bool";
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                    return "float";
                case NPTypeCode.Complex:
                    return "complex";
                case NPTypeCode.String:
                    return arr.dtype.Meta == DTypeRegistry.Str ? "str" : "bytes";
                default:
                    return "int";
            }
        }

        /// <summary>
        ///     CPython's <c>PyNumber_Multiply(x, y)</c> over the types a product of NumPy objects meets: numbers multiply; a
        ///     sequence (<c>str</c>, <c>bytes</c>) repeats only by an index (<c>int</c>, <c>bool</c>), else <c>can't multiply
        ///     sequence by non-int of type 'T'</c> naming the other operand's type (the LEFT sequence repeats first); anything
        ///     else is <c>unsupported operand type(s) for *: 'X' and 'Y'</c>.
        /// </summary>
        /// <param name="x">The left operand's Python type name.</param>
        /// <param name="y">The right operand's Python type name.</param>
        /// <returns>Python's TypeError text, or null when the product is defined.</returns>
        private static string PyMultiplyError(string x, string y)
        {
            bool xNumber = x is "bool" or "int" or "float" or "complex";
            bool yNumber = y is "bool" or "int" or "float" or "complex";
            if (xNumber && yNumber)
                return null;
            // binary_op1 found no number slot; the sequence repeat of the left operand is tried first, then the right's.
            if (x is "str" or "bytes")
                return y is "bool" or "int" ? null : $"can't multiply sequence by non-int of type '{y}'";
            if (y is "str" or "bytes")
                return x is "bool" or "int" ? null : $"can't multiply sequence by non-int of type '{x}'";
            return $"unsupported operand type(s) for *: '{x}' and '{y}'";
        }

        /// <summary>
        ///     The refusal of a product NumPy computes over Python objects (its legacy products fall back to an OBJECT array
        ///     when <c>StringDType</c> has no common DType with the other operand, and the first product did not raise).
        /// </summary>
        /// <param name="op">The NumPy function (<c>dot</c>, <c>convolve</c>, <c>correlate</c>).</param>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The exception (not thrown).</returns>
        private static NotSupportedException ObjectArrayRefusal(string op, NDArray a, NDArray b)
            => new NotSupportedException(
                $"NumPy computes np.{op} of {a.dtype} and {b.dtype} as Python objects: the two descriptors have no common " +
                "DType, so NumPy's legacy product falls back to an object array, which NumSharp does not have.");
    }
}
