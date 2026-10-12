using System;
using System.Collections.Frozen;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     What NumPy 2.4.2's ufunc layer does when a fixed-width string array (<c>S</c> or <c>U</c>) reaches a numeric
    ///     ufunc, a reduction or a product, expressed as the exception NumSharp raises in its place. Every compute entry
    ///     point asks this class first when an operand holds strings, so no numeric kernel ever sees the
    ///     <see cref="NPTypeCode.String"/> lane.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Three kinds of answer.</b> Where NumPy itself refuses (no loop for the operand classes), the exception carries
    ///     NumPy's exact text: that is final behavior, not a gap. The comparisons and the <c>logical_*</c> truth tests
    ///     COMPUTE (strings phase 2, P2.1: <see cref="NDStringComparison"/>, <see cref="NDStringTruth"/>), after the same
    ///     checks. Where NumPy computes in a way a later subphase implements (concatenation by <c>add</c>, the repeat,
    ///     <c>argmax</c>, sorting and searching, a text <c>out=</c>), the answer is <see cref="NextPhase(string, NDArray)"/>: a
    ///     <see cref="NotSupportedException"/> that names the operation. The refusals are raised BEFORE broadcasting, as
    ///     NumPy resolves the loop before it broadcasts.
    ///     </para>
    ///     <para>
    ///     <b>Three NumPy texts, chosen by the ufunc's type resolver</b> (probed against 2.4.2, <c>probe_compute*.py</c>):
    ///     <list type="bullet">
    ///       <item><description>ufuncs with promoter-based dispatch (<c>add</c>, <c>subtract</c>, <c>multiply</c>,
    ///       <c>maximum</c>, <c>minimum</c>, <c>fmax</c>, <c>fmin</c>, <c>gcd</c>, <c>lcm</c>, <c>matmul</c>, <c>clip</c>)
    ///       name the operand DESCRIPTORS: <c>ufunc 'subtract' did not contain a loop with signature matching types
    ///       (dtype('&lt;U3'), dtype('&lt;U3')) -&gt; None</c>;</description></item>
    ///       <item><description>the comparisons and the unary <c>negative</c>/<c>positive</c>/<c>absolute</c>/<c>sign</c>
    ///       name the operand CLASSES: <c>(&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class
    ///       'numpy.dtypes.Int64DType'&gt;) -&gt; None</c>;</description></item>
    ///       <item><description>every other ufunc goes through NumPy's legacy type resolver: <c>ufunc 'sqrt' not supported
    ///       for the input types, and the inputs could not be safely coerced to any supported types according to the
    ///       casting rule ''safe''</c>.</description></item>
    ///     </list>
    ///     The house exception type for NumPy's <c>UFuncTypeError</c> and <c>TypeError</c> is <see cref="TypeError"/>, and
    ///     for an input-cast refusal under <c>dtype=</c> it is <see cref="ArgumentException"/>, as for the numeric dtypes.
    ///     </para>
    ///     <para>
    ///     <b>Operand dtypes in the texts</b> are the dtypes NumSharp holds. A C# integer literal is an <c>int32</c> 0-d
    ///     array where a Python int is NEP 50's weak <c>_PyLongDType</c>, so <c>np.subtract(u, 1)</c> names
    ///     <c>dtype('int32')</c> where NumPy names <c>dtype('int64')</c>, the language-level difference every NumSharp
    ///     message shares.
    ///     </para>
    /// </remarks>
    internal static partial class NDStringUfuncs
    {
        /// <summary>
        ///     The binary ufuncs whose NumPy promoter reports a missing string loop with the operand DESCRIPTORS
        ///     (<c>(dtype('&lt;U3'), dtype('int64')) -&gt; None</c>). Every other binary ufunc without a string loop uses the
        ///     legacy resolver's "not supported for the input types" text.
        /// </summary>
        private static readonly FrozenSet<string> s_descriptorNoLoop = FrozenSet.ToFrozenSet(new[]
        {
            "add", "subtract", "multiply", "maximum", "minimum", "fmax", "fmin", "gcd", "lcm", "matmul", "clip",
        }, StringComparer.Ordinal);

        /// <summary>
        ///     The unary ufuncs whose NumPy promoter reports a missing string loop with the operand CLASS
        ///     (<c>&lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>).
        /// </summary>
        private static readonly FrozenSet<string> s_classNoLoop = FrozenSet.ToFrozenSet(new[]
        {
            "negative", "positive", "absolute", "sign",
        }, StringComparer.Ordinal);

        /// <summary>The six comparison ufuncs: NumPy compares two strings of one class and names the operand CLASSES otherwise.</summary>
        private static readonly FrozenSet<string> s_comparisons = FrozenSet.ToFrozenSet(new[]
        {
            "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        }, StringComparer.Ordinal);

        /// <summary>The logical ufuncs: NumPy reads a string's truth value (non-empty is True) for any operand mix.</summary>
        private static readonly FrozenSet<string> s_logical = FrozenSet.ToFrozenSet(new[]
        {
            "logical_and", "logical_or", "logical_xor",
        }, StringComparer.Ordinal);

        /// <summary>
        ///     True when <paramref name="a"/> holds fixed-width strings. A null operand (an omitted optional argument) is
        ///     not a string array.
        /// </summary>
        /// <param name="a">The operand, or null.</param>
        /// <returns>True for an <c>S</c> or <c>U</c> array.</returns>
        internal static bool IsString(NDArray a) => a is not null && a.typecode == NPTypeCode.String;

        /// <summary>
        ///     The guard every binary ufunc entry point runs first: does nothing for two numeric operands (two field reads
        ///     on the hot path), and raises NumPy's answer for a string operand.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name (<c>"subtract"</c>, <c>"remainder"</c>, ...).</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null; it decides string multiplication's error.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null; a non-bool mask is reported before the loop answer, and a
        ///     computing call broadcasts it with the other operands.</param>
        /// <exception cref="TypeError">NumPy has no loop for the operand classes (its UFuncTypeError / TypeError text).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/> (reported first, as NumPy parses it first).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, a numeric <paramref name="loop"/> that a
        ///     string input cannot reach by a same_kind cast, a computing loop's output that <paramref name="out"/> cannot
        ///     receive, or operands that do not broadcast with <paramref name="out"/> / <paramref name="where"/>.</exception>
        /// <exception cref="IncorrectShapeException">The two inputs of a computing call do not broadcast.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text (<see cref="NextPhase(string, NDArray)"/>).</exception>
        internal static void ThrowIfBinary(string ufunc, NDArray x1, NDArray x2, NDArray @out = null, NPTypeCode? loop = null,
            NDArray where = null)
        {
            if (!IsString(x1) && !IsString(x2))
                return;
            throw BinaryError(ufunc, x1, x2, @out, loop, where);
        }

        /// <summary>
        ///     The guard of a binary ufunc whose loop can COMPUTE on text (the comparisons, the logical ufuncs, <c>add</c>):
        ///     like <see cref="ThrowIfBinary"/>, but it RETURNS NumPy's result where that result is decided by the shapes alone
        ///     — a computing call whose broadcast result is EMPTY compares / concatenates nothing, so NumPy returns an empty
        ///     array of the loop's dtype (<c>np.equal(np.zeros(0, 'U2'), np.zeros(0, 'U2'))</c> is an empty bool array,
        ///     <c>np.add</c> of two empty <c>&lt;U2</c> arrays an empty <c>&lt;U4</c> one), or <c>out=</c> itself, untouched.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>Null when neither operand is text (the numeric path continues); otherwise the result NumPy's string loop
        ///     computes — the comparison of two strings of one class (<see cref="NDStringComparison"/>), a logical ufunc over
        ///     the operands' truth values (<see cref="LogicalOverTruth"/>), the concatenation of two strings of one class or
        ///     the repeat of a string by an integer into <paramref name="out"/> (<see cref="NDStringArithmetic"/>), the empty
        ///     result (or <paramref name="out"/>) of a computing call whose broadcast result is empty.</returns>
        /// <exception cref="Exception">Every exception <see cref="ThrowIfBinary"/> documents, for a call this phase does not
        ///     compute;
        ///     <see cref="SystemError"/> for a logical ufunc over a lone <c>U</c> unit above U+10FFFF (NumPy's bool cast);
        ///     <see cref="TypeError"/> for a concatenation wider than 2^31 - 1 bytes; <see cref="OverflowException"/> for a
        ///     repeat whose <c>reps * len</c> exceeds 2^63 - 1; <see cref="UnicodeDecodeError"/> for an <c>S</c> result cast
        ///     into a <c>U</c> <paramref name="out"/> that meets a byte ≥ 0x80.</exception>
        /// <remarks>Every caller must use the returned array: a non-null return means "this is the result", and the numeric
        ///     kernels behind the guard have no text lane.</remarks>
        internal static NDArray BinaryGuard(string ufunc, NDArray x1, NDArray x2, NDArray @out = null, NPTypeCode? loop = null,
            NDArray where = null)
        {
            if (!IsString(x1) && !IsString(x2))
                return null;
            var error = BinaryAnswer(ufunc, x1, x2, @out, loop, where, out DType computed);
            // ComputeAnswer's refusal means every check passed and the loop would compute: an empty result computes nothing,
            // the comparisons and the logical ufuncs compute now, the concatenation / repeat loops are a later subphase.
            if (error is StringComputationNotSupportedException && computed is not null)
            {
                var dims = BroadcastDims(x1, x2, @out, where);
                if (ShapeSize(dims) == 0)
                    return @out ?? new NDArray(computed, dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims), true);
                // A text out= receives the bool results FORMATTED ('True' / 'False', truncated to its width, NumPy's
                // output cast): the bools are computed into a temporary and cast into out= under where= (the guard
                // already checked the bool → text cast and the broadcast).
                bool textOut = IsString(@out);
                if (s_comparisons.Contains(ufunc) && ComparesText(x1.dtype, x2.dtype))
                    return textOut
                        ? NDStringArithmetic.CastInto(@out, where, () => NDStringComparison.Execute(ufunc, x1, x2, null, null))
                        : NDStringComparison.Execute(ufunc, x1, x2, @out, where);
                if (s_logical.Contains(ufunc))
                    return textOut
                        ? NDStringArithmetic.CastInto(@out, where, () => LogicalOverTruth(ufunc, x1, x2, null, null))
                        : LogicalOverTruth(ufunc, x1, x2, @out, where);
                // The concatenation of two strings of one class, and the repeat of a string by an integer into its (text)
                // out= — the guard's output cast already refused every non-text out for both. A numeric operand that a
                // string dtype= request casts to text never gets here (AddUnderStringRequest computes that call first).
                if (ufunc == "add" && ComparesText(x1.dtype, x2.dtype))
                    return NDStringArithmetic.Add(x1, x2, @out, where);
                if (ufunc == "multiply" && textOut)
                    return NDStringArithmetic.Multiply(x1, x2, @out, where);
            }
            throw error;
        }

        /// <summary>
        ///     <c>logical_and</c> / <c>logical_or</c> / <c>logical_xor</c> with a string operand: NumPy's
        ///     <c>logical_ufunc_promoter</c> resolves the bool loop and casts every operand to bool, so each string operand
        ///     becomes its truth array (the cast route: a <c>U</c> unit above U+10FFFF raises) and the bool loop runs on the
        ///     result through the numeric ufunc, with the call's own <c>out=</c> and <c>where=</c>.
        /// </summary>
        /// <param name="ufunc"><c>"logical_and"</c>, <c>"logical_or"</c> or <c>"logical_xor"</c>.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null (its cast was checked by the guard).</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The ufunc's result.</returns>
        /// <exception cref="SystemError">A <c>U</c> operand holds a unit above U+10FFFF.</exception>
        private static NDArray LogicalOverTruth(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            NDArray t1 = null, t2 = null;
            try
            {
                // The truth arrays keep the operands' memory order ('K'), so the bool loop's output layout follows them as
                // NumPy's follows the cast operands.
                t1 = IsString(x1) ? NDStringTruth.Truth(x1, NDStringTruth.Flavor.Cast) : null;
                t2 = IsString(x2) ? NDStringTruth.Truth(x2, NDStringTruth.Flavor.Cast) : null;
                NDArray a = t1 ?? x1, b = t2 ?? x2;
                return ufunc switch
                {
                    "logical_and" => np.logical_and(a, b, @out, where),
                    "logical_or" => np.logical_or(a, b, @out, where),
                    _ => np.logical_xor(a, b, @out, where),
                };
            }
            finally
            {
                // The truth arrays are temporaries: the result never aliases them (a fresh array or out=).
                t1?.Dispose();
                t2?.Dispose();
            }
        }

        /// <summary>
        ///     The broadcast dimensions of a ufunc's operands (inputs, <c>out=</c>, <c>where=</c>; right-aligned, 1 stretches),
        ///     for operands <see cref="BroadcastError"/> already accepted.
        /// </summary>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The broadcast dimensions.</returns>
        internal static long[] BroadcastDims(NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            var operands = new[] { x1, x2, @out, where };
            int nd = 0;
            foreach (var op in operands)
                if (op is not null)
                    nd = Math.Max(nd, op.ndim);
            var full = new long[nd];
            for (int i = 0; i < nd; i++)
                full[i] = 1;
            foreach (var op in operands)
            {
                if (op is null)
                    continue;
                var dims = op.Shape.dimensions ?? Array.Empty<long>();
                int lead = nd - dims.Length;
                for (int i = 0; i < dims.Length; i++)
                    if (dims[i] != 1)
                        full[lead + i] = dims[i];
            }
            return full;
        }

        /// <summary>The element count of a shape's dimensions (1 for none).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>Their product.</returns>
        private static long ShapeSize(long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return n;
        }

        /// <summary>
        ///     NumPy's answer for a binary ufunc with at least one string operand, in NumPy's resolution order: a numeric
        ///     <c>dtype=</c> loop first checks each input's same_kind cast (inputs in order), then the ufunc's string loops
        ///     decide between computing (the next phase) and its no-loop text.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null. NumPy's string multiply loop refuses to run without one
        ///     (it cannot size the result), so a string times an integer is an error only when it is null.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null. A string-lane request is ignored: it cannot select
        ///     a numeric loop, and the classification by operand classes gives NumPy's answer.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     NumPy parses <c>out=</c> and <c>where=</c> before it resolves the loop, so a read-only out or a non-bool mask
        ///     wins over every loop answer; a loop NumPy computes then still casts its output into <c>out=</c> and broadcasts
        ///     the operands before it runs (<see cref="ComputeAnswer"/>).
        /// </remarks>
        internal static Exception BinaryError(string ufunc, NDArray x1, NDArray x2, NDArray @out, NPTypeCode? loop,
            NDArray where = null)
            => BinaryAnswer(ufunc, x1, x2, @out, loop, where, out _);

        /// <summary>
        ///     <see cref="BinaryError"/>, also reporting the descriptor of the loop NumPy would compute (the comparisons and the
        ///     logical ufuncs: bool; a concatenation: its <see cref="Concatenation"/>; a repeat: <c>out=</c>'s), so
        ///     <see cref="BinaryGuard"/> can build an empty result.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="computed">The computing loop's output descriptor; null when the answer is not a computation.</param>
        /// <returns>The exception to throw (never null).</returns>
        private static Exception BinaryAnswer(string ufunc, NDArray x1, NDArray x2, NDArray @out, NPTypeCode? loop,
            NDArray where, out DType computed)
        {
            computed = null;
            var parse = OutWhereError(@out, where);
            if (parse is not null)
                return parse;
            DType d1 = x1.dtype, d2 = x2.dtype;
            bool s1 = d1.Meta is StringDTypeMeta, s2 = d2.Meta is StringDTypeMeta;
            bool sameClassText = s1 && s2 && d1.Meta == d2.Meta;
            // A string-lane request cannot select a numeric loop; only a numeric dtype= changes NumPy's answer.
            NPTypeCode? numeric = loop is { } l && l != NPTypeCode.String && l != NPTypeCode.Empty ? l : null;

            // The comparisons and the logical ufuncs have bool-output loops only.
            bool boolLoopsOnly = s_comparisons.Contains(ufunc) || s_logical.Contains(ufunc);
            if (boolLoopsOnly)
            {
                // A non-bool dtype= names a loop that does not exist, whatever the operands (probed: equal(u, u, dtype='f8')).
                if (numeric is { } n && n != NPTypeCode.Boolean)
                    return new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
                // logical_and/or/xor read the truth value of every operand, strings included, for any operand mix.
                if (s_logical.Contains(ufunc))
                {
                    computed = DType.Boolean;
                    return ComputeAnswer(ufunc, x1, x2, @out, where, DType.Boolean);
                }
                // Two strings of one class compare their text.
                if (sameClassText)
                {
                    computed = DType.Boolean;
                    return ComputeAnswer(ufunc, x1, x2, @out, where, DType.Boolean);
                }
                // dtype=bool fixes the loop at bool, and the string input's cast to it fails (probed: less(u, i, dtype=bool)).
                if (numeric is not null)
                    return InputCastError(ufunc, 0, d1, NPTypeCode.Boolean) ?? InputCastError(ufunc, 1, d2, NPTypeCode.Boolean);
                return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types ({d1.Meta}, {d2.Meta}) -> None");
            }

            // NumPy's string add promoter wins over a dtype= request: add(U3, U3, dtype='f8') still concatenates (probed).
            if (ufunc == "add" && sameClassText)
            {
                computed = Concatenation(d1, d2);
                return ComputeAnswer(ufunc, x1, x2, @out, where, computed);
            }

            // dtype= with a numeric loop: the signature fixes the loop, and NumPy checks every input's cast to it in input
            // order. Text never reaches a number by a same_kind cast, so a string input always fails here.
            if (numeric is { } loopCode)
            {
                var c1 = InputCastError(ufunc, 0, d1, loopCode);
                if (c1 != null) return c1;
                var c2 = InputCastError(ufunc, 1, d2, loopCode);
                if (c2 != null) return c2;
            }

            // NumPy's string add promoter refuses a bytes/str mix with its own text (probed: add(U3, S2)).
            if (ufunc == "add" && s1 && s2)
                return new TypeError($"ufunc 'add' cannot use operands with types {d1.ToString(true)} and {d2.ToString(true)}");

            // NumPy's string multiply promoter takes any integer type number (PyTypeNum_ISINTEGER: the eight widths, not
            // bool) as the repeat count and converts it to int64.
            if (ufunc == "multiply" && (s1 && d2.typecode.IsInteger() || s2 && d1.typecode.IsInteger()))
            {
                // NumPy's legacy string multiply loop needs a preallocated result (its verbatim refusal without one); with
                // one, an empty broadcast writes nothing into it.
                computed = @out?.dtype;
                return MultiplyRepeatAnswer(x1, x2, @out, where);
            }

            if (s_descriptorNoLoop.Contains(ufunc))
                return NoLoop(ufunc, d1, d2);

            return NotSupportedForInputTypes(ufunc);
        }

        /// <summary>
        ///     The guard every unary ufunc entry point runs first; does nothing for a numeric operand.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name (<c>"sqrt"</c>, <c>"absolute"</c>, ...).</param>
        /// <param name="x">The operand.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null; a read-only one is reported first.</param>
        /// <param name="where">The <c>where=</c> mask, or null; a non-bool one is reported before the loop answer.</param>
        /// <exception cref="TypeError">NumPy has no loop for the operand class (its UFuncTypeError / TypeError text).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or a numeric <paramref name="loop"/> the
        ///     string input cannot reach by a same_kind cast.</exception>
        internal static void ThrowIfUnary(string ufunc, NDArray x, NPTypeCode? loop = null, NDArray @out = null,
            NDArray where = null)
        {
            if (!IsString(x))
                return;
            throw UnaryError(ufunc, x, loop, @out, where);
        }

        /// <summary>
        ///     NumPy's answer for a unary ufunc applied to a string array. No unary numeric ufunc has a string loop, so the
        ///     answer is always an error.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="x">The string operand.</param>
        /// <param name="loop">The <c>dtype=</c> loop request, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception UnaryError(string ufunc, NDArray x, NPTypeCode? loop, NDArray @out = null, NDArray where = null)
        {
            // NumPy parses out= and where= before it resolves the loop.
            var parse = OutWhereError(@out, where);
            if (parse is not null)
                return parse;
            var d = x.dtype;
            // positive has no bool loop, so dtype=bool names the missing signature (probed: positive(u, dtype=bool)).
            if (ufunc == "positive" && loop == NPTypeCode.Boolean)
                return new TypeError(
                    $"ufunc 'positive' did not contain a loop with signature matching types {d.Meta} -> {DTypeRegistry.FromTypeCode(NPTypeCode.Boolean)}");
            // A numeric dtype= fixes the loop, and the input's cast to it fails first. The unary text names no index.
            if (loop is { } numeric && numeric != NPTypeCode.String && numeric != NPTypeCode.Empty)
                return new ArgumentException(
                    $"Cannot cast ufunc '{ufunc}' input from {d.ToString(true)} to dtype('{numeric.AsNumpyDtypeName()}') " +
                    "with casting rule 'same_kind'");

            if (s_classNoLoop.Contains(ufunc))
                return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {d.Meta} -> None");

            // isnat checks its operand kind itself, before any loop search.
            if (ufunc == "isnat")
                return new TypeError("ufunc 'isnat' is only defined for np.datetime64 and np.timedelta64.");

            return NotSupportedForInputTypes(ufunc);
        }

        /// <summary>
        ///     The guard for the reductions, scans and statistics built on a ufunc's <c>reduce</c> / <c>accumulate</c>
        ///     (and the <c>argmax</c> / truth-test family NumPy implements on strings): does nothing for a numeric array.
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"sum"</c>, <c>"cumprod"</c>, <c>"nanmax"</c>, ...).</param>
        /// <param name="a">The reduced array.</param>
        /// <exception cref="TypeError">NumPy refuses the reduction for string operands (its verbatim text).</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes it on text (<see cref="NextPhase(string, NDArray)"/>).</exception>
        internal static void ThrowIfReduce(string op, NDArray a)
        {
            if (!IsString(a))
                return;
            throw ReduceError(op, a);
        }

        /// <summary>
        ///     NumPy's answer for a reduction-style function over a string array, keyed by the NumPy function name. The
        ///     answer does not depend on <c>axis</c>, <c>keepdims</c>, <c>dtype=</c>, the size or the dimension: NumPy
        ///     resolves the reduction loop from the descriptor alone (probed: <c>np.sum(u, dtype='f8')</c>, an empty and a
        ///     0-d array all give the same text).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The string array.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception ReduceError(string op, NDArray a)
        {
            var d = a.dtype;
            switch (op)
            {
                // add.reduce resolves the string concatenation loop, whose output is the two lengths added, and then
                // refuses because a reduction needs output == input.
                case "sum":
                case "mean":
                case "var":
                case "std":
                case "nansum":
                case "nanmean":
                case "nanvar":
                case "nanstd":
                case "median":
                case "average":
                case "trace":
                    return AddNotCompatible(d, "reduce");
                case "cumsum":
                case "nancumsum":
                case "cumulative_sum":
                    return AddNotCompatible(d, "accumulate");
                case "prod":
                case "nanprod":
                case "cumprod":
                case "nancumprod":
                case "cumulative_prod":
                    return NoLoop("multiply", d, d);
                case "max":
                case "amax":
                case "ptp":
                    return NoLoop("maximum", d, d);
                case "min":
                case "amin":
                    return NoLoop("minimum", d, d);
                case "nanmax":
                    return NoLoop("fmax", d, d);
                case "nanmin":
                    return NoLoop("fmin", d, d);
                // The nan-statistics test for NaN first — except for an empty input, which NumPy hands to np.nanmean
                // (whose add.reduce refuses text) before any NaN test.
                case "nanmedian":
                case "nanpercentile":
                case "nanquantile":
                    return a.size == 0 ? AddNotCompatible(d, "reduce") : NotSupportedForInputTypes("isnan");
                default:
                    // argmax/argmin, any/all, nonzero, count_nonzero and the sorting family compute on text.
                    return NextPhase(op, a);
            }
        }

        /// <summary>
        ///     The ufunc NumPy's <paramref name="op"/> reduces or accumulates with, or null when its text answer does not
        ///     come from one binary loop on (<c>out</c>, input).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="accumulate">Whether the op is an accumulation (<c>ufunc.accumulate</c>) rather than a reduction.</param>
        /// <returns><c>add</c>, <c>multiply</c>, <c>maximum</c>, <c>minimum</c>, <c>fmax</c>, <c>fmin</c>, or null.</returns>
        internal static string ReduceUfuncOf(string op, out bool accumulate)
        {
            accumulate = op is "cumsum" or "nancumsum" or "cumulative_sum" or "cumprod" or "nancumprod" or "cumulative_prod";
            return op switch
            {
                "sum" or "mean" or "var" or "std" or "nansum" or "nanmean" or "nanvar" or "nanstd" or "median" or "average"
                    or "trace" or "cumsum" or "nancumsum" or "cumulative_sum" => "add",
                "prod" or "nanprod" or "cumprod" or "nancumprod" or "cumulative_prod" => "multiply",
                "max" or "amax" or "ptp" => "maximum",
                "min" or "amin" => "minimum",
                "nanmax" => "fmax",
                "nanmin" => "fmin",
                _ => null,
            };
        }

        /// <summary>
        ///     NumPy's loop resolution for a reduction or accumulation given an <c>out=</c> array, when the input or the out
        ///     holds text: the out acts as the loop's FIRST operand, so NumPy resolves the binary ufunc on
        ///     <c>(out.dtype, input.dtype)</c> and refuses before it reads an element or checks the out's shape.
        /// </summary>
        /// <param name="ufunc">The ufunc (<see cref="ReduceUfuncOf"/>).</param>
        /// <param name="a">The input.</param>
        /// <param name="out">The out array (any dtype).</param>
        /// <param name="accumulate">Whether this is <c>ufunc.accumulate</c>.</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     <para>
        ///     add: two texts of one class resolve the concatenation loop (output = the two lengths added) and are then
        ///     refused (<c>the resolved dtypes are not compatible with add.reduce. Resolved (dtype('&lt;U4'), dtype('&lt;U2'),
        ///     dtype('&lt;U6'))</c>); texts of both classes are the add promoter's <c>cannot use operands with types</c>.
        ///     multiply: an INTEGER out and a text input resolve the string repeat (the count as int64, the result the out's
        ///     dtype cast to the text class: <c>int8</c> reads as <c>&lt;U4</c>) and are refused the same way; a text out and an
        ///     integer input refuse an accumulation and REPEAT the out's text in a reduction (NumPy computes:
        ///     <c>np.prod([1, 2, 3], out=np.zeros((), 'U4'))</c> is <c>'1111'</c>). Every other combination names both
        ///     descriptors: <c>ufunc 'maximum' did not contain a loop with signature matching types (dtype('float64'),
        ///     dtype('&lt;U2')) -&gt; None</c> (probed against 2.4.2).
        ///     </para>
        /// </remarks>
        internal static Exception ReduceOutError(string ufunc, NDArray a, NDArray @out, bool accumulate)
        {
            DType d = a.dtype, o = @out.dtype;
            bool textD = d.Meta is StringDTypeMeta, textO = o.Meta is StringDTypeMeta;
            string method = accumulate ? "accumulate" : "reduce";
            switch (ufunc)
            {
                case "add":
                    if (textD && textO)
                    {
                        if (o.Meta != d.Meta)
                            return new TypeError($"ufunc 'add' cannot use operands with types {o.ToString(true)} and {d.ToString(true)}");
                        return new TypeError(
                            $"the resolved dtypes are not compatible with add.{method}. Resolved ({o.ToString(true)}, " +
                            $"{d.ToString(true)}, {Concatenation(o, d).ToString(true)})");
                    }
                    return NoLoop("add", o, d);
                case "multiply":
                    if (textD && IsRepeatCount(o))
                        return new TypeError(
                            $"the resolved dtypes are not compatible with multiply.{method}. Resolved (dtype('int64'), " +
                            $"{d.ToString(true)}, {DTypePromotion.CastDescrToDType(o, d.Meta).ToString(true)})");
                    if (textO && IsRepeatCount(d))
                        return accumulate
                            ? new TypeError(
                                "the resolved dtypes are not compatible with multiply.accumulate. Resolved " +
                                $"({o.ToString(true)}, dtype('int64'), {o.ToString(true)})")
                            : NextPhase("multiply", @out);
                    return NoLoop("multiply", o, d);
                default:
                    return NoLoop(ufunc, o, d);
            }
        }

        /// <summary>
        ///     Whether a descriptor is one NumPy's string repeat promoter takes as the repeat count: the eight integer widths
        ///     (<c>PyTypeNum_ISINTEGER</c>) — not bool, and not NumSharp's char (no NumPy dtype).
        /// </summary>
        /// <param name="d">The descriptor.</param>
        /// <returns>True for an integer count.</returns>
        private static bool IsRepeatCount(DType d) => d.typecode is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16
            or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>
        ///     The guard of <c>np.isclose</c> / <c>np.allclose</c> (NumPy's <c>allclose</c> is <c>all(isclose(...))</c>).
        ///     NumPy's <c>isclose</c> first promotes <c>y</c> with a weak Python float (<c>result_type(y, 1.)</c>), which a
        ///     string <c>y</c> fails with <see cref="DTypePromotionError"/>; a string <c>x</c> then meets
        ///     <c>subtract(x, y)</c>, which has no loop. Does nothing for two numeric operands.
        /// </summary>
        /// <param name="a">The <c>a</c> argument (NumPy's <c>x</c>).</param>
        /// <param name="b">The <c>b</c> argument (NumPy's <c>y</c>).</param>
        /// <exception cref="DTypePromotionError"><paramref name="b"/> holds strings: <c>The DType &lt;class
        ///     'numpy.dtypes.StrDType'&gt; could not be promoted by &lt;class 'numpy.dtypes._PyFloatDType'&gt;. ...</c>.</exception>
        /// <exception cref="TypeError">Only <paramref name="a"/> holds strings: <c>ufunc 'subtract' did not contain a loop
        ///     with signature matching types (dtype('&lt;U3'), dtype('float64')) -&gt; None</c>.</exception>
        internal static void ThrowIfIsClose(NDArray a, NDArray b)
        {
            if (!IsString(a) && !IsString(b))
                return;
            // NEP 50's weak float has no common DType with text, so a string y raises here, with NumPy's text.
            var y = DTypePromotion.ResultType(b, 1.0);
            throw NoLoop("subtract", a.dtype, y);
        }

        /// <summary>
        ///     NumPy's refusal of a string operand in the BLAS-style products (<c>dot</c>, <c>inner</c>, <c>vdot</c>,
        ///     <c>tensordot</c>, <c>convolve</c>, <c>correlate</c>): their <c>dotfunc</c> lookup needs a fixed item size, and
        ///     NumPy's message is <c>data type must provide an itemsize</c> (ValueError).
        /// </summary>
        /// <returns>The exception (not thrown).</returns>
        internal static ValueError NoItemsize() => new ValueError("data type must provide an itemsize");

        /// <summary>
        ///     The guard of <c>np.dot</c> / <c>np.inner</c> for a string operand, in NumPy's <c>PyArray_MatrixProduct2</c>
        ///     order: a 0-d operand makes NumPy compute a plain <c>multiply</c> (its string loops decide); otherwise the
        ///     contracted lengths must align, then <c>new_array_for_sum</c> either refuses the caller's <c>out=</c> or, with
        ///     none, fails to allocate a result for the string type number (it carries no item size). Does nothing for two
        ///     numeric operands.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand (for <c>np.inner</c>: already transposed, as NumPy passes it).</param>
        /// <param name="out">The <c>out=</c> array of <c>np.dot</c>, or null.</param>
        /// <exception cref="TypeError">A 0-d operand: <c>multiply</c>'s answer for the operand classes.</exception>
        /// <exception cref="IncorrectShapeException">The contracted lengths differ: <c>shapes (3,) and (4,) not aligned: 3
        ///     (dim 0) != 4 (dim 0)</c> (NumPy's ValueError text; the house type of the alignment error).</exception>
        /// <exception cref="ValueError">With <paramref name="out"/>: <c>output array is not acceptable (must have the right
        ///     datatype, number of dimensions, and be a C-Array)</c>, <c>output array has wrong dimensions</c>, or — for an
        ///     acceptable out — <c>dot not available for this type</c>; without: <c>data type must provide an
        ///     itemsize</c>.</exception>
        internal static void ThrowIfProduct(NDArray a, NDArray b, NDArray @out = null)
        {
            if (!IsString(a) && !IsString(b))
                return;
            if (a.ndim == 0 || b.ndim == 0)
                throw ScalarProductError(a, b);

            // dot_alignment_error: a's last axis against b's second-to-last (its only axis when 1-D).
            long l = a.Shape.dimensions[a.ndim - 1];
            int matchDim = b.ndim > 1 ? b.ndim - 2 : 0;
            if (b.Shape.dimensions[matchDim] != l)
                throw new IncorrectShapeException(
                    $"shapes {a.Shape.ToPythonTuple()} and {b.Shape.ToPythonTuple()} not aligned: {l} (dim {a.ndim - 1}) " +
                    $"!= {b.Shape.dimensions[matchDim]} (dim {matchDim})");

            if (@out is null)
                throw NoItemsize();

            // new_array_for_sum: out must have the result's rank, the result's TYPE NUMBER (any width of the promoted
            // string class: str when either operand is str, else bytes) and be a C-array (C-contiguous, writeable).
            int nd = a.ndim + b.ndim - 2;
            bool unicode = a.dtype.Meta is StringDTypeMeta { IsUnicode: true } || b.dtype.Meta is StringDTypeMeta { IsUnicode: true };
            bool sameClass = @out.dtype.Meta is StringDTypeMeta outMeta && outMeta.IsUnicode == unicode;
            if (@out.ndim != nd || !sameClass || !@out.Shape.IsContiguous || !@out.Shape.IsWriteable)
                throw new ValueError(
                    "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");

            // The result's dimensions: a's leading axes, b's axes but its contracted one.
            int d = 0;
            for (int i = 0; i < a.ndim - 1; i++, d++)
                if (@out.Shape.dimensions[d] != a.Shape.dimensions[i])
                    throw new ValueError("output array has wrong dimensions");
            for (int i = 0; i < b.ndim - 2; i++, d++)
                if (@out.Shape.dimensions[d] != b.Shape.dimensions[i])
                    throw new ValueError("output array has wrong dimensions");
            if (b.ndim > 1 && @out.Shape.dimensions[d] != b.Shape.dimensions[b.ndim - 1])
                throw new ValueError("output array has wrong dimensions");

            // The string descriptor has no dot function.
            throw new ValueError("dot not available for this type");
        }

        /// <summary>
        ///     NumPy's answer for <c>np.percentile</c> / <c>np.quantile</c> of a string array. The interpolating methods
        ///     (<c>linear</c>, the default, and the other continuous ones) subtract two neighbouring elements as NumPy
        ///     scalars, which Python refuses; the selection methods only sort and pick an element, which NumPy computes on
        ///     text (the next phase).
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"percentile"</c> or <c>"quantile"</c>).</param>
        /// <param name="a">The string array.</param>
        /// <param name="method">The <c>method=</c> argument.</param>
        /// <param name="scalarResult">True when the interpolated result is a scalar (a scalar <c>q</c> and no axis left):
        ///     NumPy's <c>_lerp</c> then subtracts two NumPy scalars, which Python refuses with its own text; with any
        ///     array result it subtracts arrays, and <c>subtract</c>'s promoter names the descriptors.</param>
        /// <returns>The exception (not thrown).</returns>
        /// <remarks>
        ///     The full answer, with the axis, method and empty-axis checks NumPy makes first, is
        ///     <see cref="QuantileAnswer"/>; this is its last step.
        /// </remarks>
        internal static Exception QuantileError(string op, NDArray a, string method, bool scalarResult = true)
        {
            switch (method)
            {
                // The discontinuous methods select an element without arithmetic (NumPy's _get_indexes + take).
                case "lower":
                case "higher":
                case "nearest":
                case "inverted_cdf":
                case "closest_observation":
                    return NextPhase(op, a);
                default:
                    if (!scalarResult)
                        return NoLoop("subtract", a.dtype, a.dtype);
                    // Python's binary-operator refusal names the NumPy scalar type of each operand (np.str_ / np.bytes_).
                    var scalar = "numpy." + a.dtype.Meta.ScalarName;
                    return new TypeError($"unsupported operand type(s) for -: '{scalar}' and '{scalar}'");
            }
        }

        /// <summary>
        ///     The guard of <c>np.einsum</c>'s contraction (after the subscripts parsed and the single-operand view path
        ///     returned): does nothing unless an operand holds text or <c>dtype=</c> names a string class, and otherwise
        ///     raises NumPy's <c>c_einsum</c> answer (<see cref="EinsumError"/>).
        /// </summary>
        /// <param name="operands">The einsum operands.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="casting">The <c>casting=</c> rule (already validated).</param>
        /// <exception cref="TypeError">An operand cannot reach the loop descriptor under <paramref name="casting"/>, the
        ///     bool zero NumPy initialises its output with cannot reach the output descriptor by a 'safe' cast, or the loop
        ///     has no string sum-of-products (<c>invalid data type for einsum</c>).</exception>
        internal static void ThrowIfEinsum(NDArray[] operands, DType dtype, string casting)
        {
            bool any = IsStringRequest(dtype);
            foreach (var op in operands)
                any |= IsString(op);
            if (!any)
                return;
            throw EinsumError(operands, dtype, casting);
        }

        /// <summary>
        ///     NumPy's <c>c_einsum</c> over text, in its order (probed against 2.4.2): every operand must reach the loop
        ///     descriptor under <c>casting=</c> (the operands' result type, or the request — an unsized string request
        ///     adapted to each operand: <c>&lt;U2</c> under <c>'S'</c> reads as <c>S2</c>); then the output, initialised
        ///     from a <c>bool</c> zero by a 'safe' cast, must hold <c>False</c> (an unsized request allocates one character:
        ///     <c>&lt;U1</c>, which cannot); a string loop that passes both has no sum-of-products.
        /// </summary>
        /// <param name="operands">The einsum operands.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="casting">The <c>casting=</c> rule.</param>
        /// <returns>The exception to throw (never null).</returns>
        internal static Exception EinsumError(NDArray[] operands, DType dtype, string casting)
        {
            var rule = DTypeCasting.ParseCasting(casting);
            bool unsized = IsStringRequest(dtype) && dtype.itemsize == 0;
            DType loop = dtype ?? DTypePromotion.ResultType(operands);
            for (int i = 0; i < operands.Length; i++)
            {
                var from = operands[i].dtype;
                var target = !unsized ? loop : from.Meta == dtype.Meta ? from : DTypePromotion.CastDescrToDType(from, dtype.Meta);
                if (!DTypeCasting.CanCastTypeTo(from, target, rule))
                    return new TypeError(
                        $"Iterator operand {i} dtype could not be cast from {from.ToString(true)} to {target.ToString(true)} " +
                        $"according to the rule '{casting}'");
            }

            var output = unsized ? dtype.Meta.DefaultDescr() : loop;
            if (!DTypeCasting.CanCastTypeTo(DType.Boolean, output, NPY_CASTING.NPY_SAFE_CASTING))
                // NumPy's PyArray_AssignRawScalar of the bool zero into the output raises TypeError, like the operand
                // cast above (np.copyto keeps the house InvalidCastException for its own scalar casts).
                return new TypeError(
                    $"Cannot cast scalar from dtype('bool') to {output.ToString(true)} according to the rule 'safe'");
            return new TypeError("invalid data type for einsum");
        }

        /// <summary>
        ///     <c>PyArray_MatrixProduct2</c>'s scalar branch: the common type NUMBER of both operands is the string class
        ///     (<c>U</c> wins over <c>S</c>), each operand is converted to that class by <c>PyArray_FromAny</c> (a string of
        ///     the class keeps its width, a number or the other class discovers one: int64 → <c>&lt;U21</c>), and the
        ///     product is <c>multiply</c> of the converted pair, which has no string × string loop.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The exception (not thrown).</returns>
        private static Exception ScalarProductError(NDArray a, NDArray b)
        {
            DType d1 = a.dtype, d2 = b.dtype;
            // PyArray_ObjectType keeps the larger type number: StrDType (19) over BytesDType (18).
            var target = d1.Meta is StringDTypeMeta { IsUnicode: true } || d2.Meta is StringDTypeMeta { IsUnicode: true }
                ? DTypeRegistry.Str : DTypeRegistry.Bytes;
            var c1 = d1.Meta == target ? d1 : DTypePromotion.CastDescrToDType(d1, target);
            var c2 = d2.Meta == target ? d2 : DTypePromotion.CastDescrToDType(d2, target);
            return NoLoop("multiply", c1, c2);
        }

        /// <summary>
        ///     NumPy's legacy type-resolver refusal: <c>ufunc 'X' not supported for the input types, and the inputs could
        ///     not be safely coerced to any supported types according to the casting rule ''safe''</c> (the doubled quotes
        ///     are NumPy's).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError NotSupportedForInputTypes(string ufunc)
            => new TypeError(
                $"ufunc '{ufunc}' not supported for the input types, and the inputs could not be safely coerced to any " +
                "supported types according to the casting rule ''safe''");

        /// <summary>
        ///     NumPy's promoter refusal naming the operand descriptors: <c>ufunc 'X' did not contain a loop with signature
        ///     matching types (dtype('&lt;U3'), dtype('int64')) -&gt; None</c>.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="operands">The operand descriptors, in call order.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError NoLoop(string ufunc, params DType[] operands)
        {
            var reprs = new string[operands.Length];
            for (int i = 0; i < reprs.Length; i++)
                reprs[i] = operands[i].ToString(true);
            return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types ({string.Join(", ", reprs)}) -> None");
        }

        /// <summary>
        ///     NumPy's refusal to reduce or accumulate with the string <c>add</c> loop: <c>the resolved dtypes are not
        ///     compatible with add.reduce. Resolved (dtype('&lt;U3'), dtype('&lt;U3'), dtype('&lt;U6'))</c>. The third
        ///     descriptor is the concatenation loop's output, the input length doubled.
        /// </summary>
        /// <param name="descr">The reduced array's descriptor.</param>
        /// <param name="method">The ufunc method, <c>"reduce"</c> or <c>"accumulate"</c>.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError AddNotCompatible(DType descr, string method)
        {
            var meta = (StringDTypeMeta)descr.Meta;
            long chars = descr.itemsize / meta.UnitSize;
            // The doubled length cannot pass the class limit; NumPy's own concatenation would refuse an overflowing
            // length before this point, so clamping only guards the arithmetic.
            var doubled = meta.Descr(Math.Min(2 * chars, meta.MaxChars));
            return new TypeError(
                $"the resolved dtypes are not compatible with add.{method}. " +
                $"Resolved ({descr.ToString(true)}, {descr.ToString(true)}, {doubled.ToString(true)})");
        }

        /// <summary>
        ///     The refusal for a call NumPy answers by computing on text that NumSharp does not compute yet (sorting,
        ///     searching, conversion between text and numbers, the <c>np.strings</c> functions beyond <c>add</c> /
        ///     <c>multiply</c>): those are the next phase of NumSharp's string support.
        /// </summary>
        /// <param name="op">The NumPy function or ufunc name.</param>
        /// <param name="a">A string operand of the call; its descriptor is named in the message.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static StringComputationNotSupportedException NextPhase(string op, NDArray a) => NextPhase(op, a.dtype);

        /// <summary>
        ///     <see cref="NextPhase(string, NDArray)"/> naming a descriptor rather than an operand: the string loop's output
        ///     descriptor of a computing loop this phase does not run yet.
        /// </summary>
        /// <param name="op">The NumPy function or ufunc name.</param>
        /// <param name="descr">The descriptor named in the message.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static StringComputationNotSupportedException NextPhase(string op, DType descr)
            => new StringComputationNotSupportedException(
                $"{op} is not supported for fixed-width string arrays yet ({descr.ToString(true)}): NumSharp's string support " +
                "covers storage, element access, conversion to and from C#, copying, casts between string dtypes, conversion " +
                "between text and numbers, indexing, printing, comparisons, truth tests, concatenation and repetition. The rest " +
                "of string computation (sorting and searching, the np.strings functions) is the next phase.");

        /// <summary>
        ///     The comparison operators <c>==</c> / <c>!=</c> with a string operand and an operand of another class: NumPy's
        ///     <c>_failed_comparison_workaround</c>. The ufunc has no loop, so the operator answers "never equal" without
        ///     reading an element: an all-False (<c>==</c>) or all-True (<c>!=</c>) bool array of the broadcast shape.
        /// </summary>
        /// <param name="lhs">The left operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <param name="notEqual">True for <c>!=</c>.</param>
        /// <returns>The bool result, or null when both operands are strings of one class (NumPy compares the text; the
        ///     caller raises <see cref="NextPhase(string, NDArray)"/>).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError, raised by the same
        ///     workaround when it allocates the result).</exception>
        internal static NumSharp.Generic.NDArray<bool> FailedComparison(NDArray lhs, NDArray rhs, bool notEqual)
        {
            DType d1 = lhs.dtype, d2 = rhs.dtype;
            if (d1.Meta is StringDTypeMeta && d2.Meta is StringDTypeMeta && d1.Meta == d2.Meta)
                return null;
            // The workaround allocates its result through an iterator over both operands, so a broadcast failure is
            // still reported; the values are a constant that needs no element of either operand. The typed array is
            // built directly: a typed wrapper made afterwards would alias it, and NumPy's result owns its data.
            var shape = Shape.ResolveReturnShape(lhs.Shape, rhs.Shape);
            var result = new NumSharp.Generic.NDArray<bool>(shape.Clean(), fillZeros: !notEqual);
            if (notEqual)
                result.Storage.InternalArray.Fill(true);
            return result;
        }

        /// <summary>
        ///     NumPy's input-cast check under <c>dtype=</c>: input <paramref name="index"/> must reach the requested loop by
        ///     a same_kind cast. A string input never does; a numeric input follows the numeric casting table.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="index">The input's position (NumPy names it).</param>
        /// <param name="from">The input's descriptor.</param>
        /// <param name="loop">The requested numeric loop.</param>
        /// <returns>The exception, or null when the cast is allowed.</returns>
        private static ArgumentException InputCastError(string ufunc, int index, DType from, NPTypeCode loop)
        {
            if (from.Meta is not StringDTypeMeta)
            {
                var code = from.typecode;
                if (code == loop || NDIterCasting.CanCast(code, loop, NPY_CASTING.NPY_SAME_KIND_CASTING))
                    return null;
            }
            return new ArgumentException(
                $"Cannot cast ufunc '{ufunc}' input {index} from {from.ToString(true)} to dtype('{loop.AsNumpyDtypeName()}') " +
                "with casting rule 'same_kind'");
        }
    }
}
