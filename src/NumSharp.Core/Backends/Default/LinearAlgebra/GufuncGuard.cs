namespace NumSharp.Backends
{
    /// <summary>
    ///     The checks every generalized ufunc of the product family shares (<c>np.matmul</c>, <c>np.vecdot</c>,
    ///     <c>np.matvec</c>, <c>np.vecmat</c>, and the gufunc core checks of <c>np.linalg.solve</c>), with NumPy's texts
    ///     from <c>_parse_axes_arg</c> / <c>_validate_num_dims</c> / <c>_get_coredim_sizes</c> in
    ///     <c>numpy/_core/src/umath/ufunc_object.c</c>, plus the <c>order=</c> / <c>casting=</c> name checks
    ///     <c>np.matmul</c> and <c>np.einsum</c> share.
    /// </summary>
    /// <remarks>
    ///     A gufunc's core dimensions are matched EXACTLY — they do not broadcast, so a length-1 core
    ///     dim against a length-3 one is an error, not a stretch. Only the LEADING (loop) axes
    ///     broadcast, and those are left to the ordinary elementwise machinery downstream.
    ///     <para>Engine-layer since the np-layer dtype plan's P2 (it lived beside the public functions): the gufunc calls
    ///     are <see cref="DefaultEngine"/> members now, and <c>np.linalg.solve</c> / <c>np.einsum</c> call these checks
    ///     from the np layer.</para>
    /// </remarks>
    internal static class GufuncGuard
    {
        /// <summary>
        ///     The part of a gufunc call NumPy settles while it parses the arguments and resolves the loop, before any
        ///     core-dimension, <c>axes</c> / <c>axis</c> / <c>keepdims</c> semantics or shape check: the <c>dtype=</c> class
        ///     check, then — with text involved (a string operand or a string <c>dtype=</c> request) — a read-only
        ///     <c>out=</c>, then the loop refusal.
        /// </summary>
        /// <param name="name">NumPy's gufunc name (<c>matmul</c>, <c>vecdot</c>, <c>matvec</c>, <c>vecmat</c>).</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <exception cref="TypeError">A <paramref name="dtype"/> with details beyond its class (<c>'U3'</c>,
        ///     <c>'&gt;f8'</c>), or NumPy's no-loop text: the operand CLASSES and the requested class under a string request
        ///     (<c>(&lt;class 'numpy.dtypes.Float64DType'&gt;, …) -&gt; &lt;class 'numpy.dtypes.StrDType'&gt;</c>), the operand
        ///     DESCRIPTORS for a string operand (<c>(dtype('&lt;U2'), dtype('float64')) -&gt; None</c>).</exception>
        /// <exception cref="ValueError">Text with a read-only <paramref name="out"/>.</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        /// <remarks>
        ///     No gufunc has a string loop, so text never reaches <see cref="ToLoop"/> (the string lane has no element
        ///     size); a numeric request still passes only its class check here (probed: <c>vecdot(f, f, dtype='U3')</c>
        ///     is the class-only TypeError, and NumPy resolves the loop before <c>axis</c> / <c>keepdims</c> semantics:
        ///     <c>matvec(u2, u, axis=0)</c> names the missing loop).
        /// </remarks>
        internal static void ThrowIfText(string name, NDArray x1, NDArray x2, DType dtype, NDArray @out)
        {
            if (dtype is not null)
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
            bool request = Backends.Strings.NDStringUfuncs.IsStringRequest(dtype);
            bool vstring = Backends.Strings.NDStringUfuncs.IsVString(x1) || Backends.Strings.NDStringUfuncs.IsVString(x2);
            if (!request && !vstring && !Backends.Strings.NDStringUfuncs.IsString(x1) && !Backends.Strings.NDStringUfuncs.IsString(x2))
                return;
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            // A StringDType operand: NumPy's dispatcher (a new-style DType in the key) names every operand's class, and a
            // string request's class as the output.
            if (vstring)
                throw Backends.Strings.NDStringUfuncs.VStringNoLoop(name,
                    new[] { Backends.Strings.NDStringUfuncs.DispatchClass(x1), Backends.Strings.NDStringUfuncs.DispatchClass(x2) },
                    request ? dtype.Meta.ToString() : "None");
            throw request
                ? Backends.Strings.NDStringUfuncs.GufuncDTypeLoopError(name, dtype, x1, x2)
                : Backends.Strings.NDStringUfuncs.NoLoop(name, x1.dtype, x2.dtype);
        }

        /// <summary>
        ///     NumPy's minimum-rank check of one input (<c>_validate_num_dims</c>): an operand with fewer dimensions than
        ///     the gufunc's core needs raises "<c>{name}: Input operand {i} does not have enough dimensions (has {n},
        ///     gufunc core with signature {sig} requires {k})</c>".
        /// </summary>
        /// <param name="name">The gufunc's name as NumPy prints it (<c>vecdot</c>, <c>solve</c>, ...).</param>
        /// <param name="signature">The gufunc signature the message quotes (<c>(n),(n)-&gt;()</c>).</param>
        /// <param name="operandIndex">The input's position (0 or 1), as the message numbers it.</param>
        /// <param name="operand">The input.</param>
        /// <param name="required">How many core dimensions the signature gives that input.</param>
        /// <exception cref="ValueError"><paramref name="operand"/> has fewer than <paramref name="required"/>
        ///     dimensions.</exception>
        internal static void RequireRank(string name, string signature, int operandIndex, NDArray operand, int required)
        {
            if (operand.ndim >= required)
                return;

            throw new ValueError(
                $"{name}: Input operand {operandIndex} does not have enough dimensions " +
                $"(has {operand.ndim}, gufunc core with signature {signature} requires {required})");
        }

        /// <summary>
        ///     NumPy's core-size check (<c>_get_coredim_sizes</c>): a core dimension an earlier operand already fixed must
        ///     have the same size here — "<c>{name}: Input operand {i} has a mismatch in its core dimension {d}, with
        ///     gufunc signature {sig} (size {actual} is different from {expected})</c>". Core dimensions never broadcast.
        /// </summary>
        /// <param name="name">The gufunc's name as NumPy prints it.</param>
        /// <param name="signature">The gufunc signature the message quotes.</param>
        /// <param name="operandIndex">The input whose core dimension is checked, as the message numbers it.</param>
        /// <param name="coreDimension">The core dimension's position within that input's core, as the message numbers
        ///     it.</param>
        /// <param name="actual">The size this operand carries — reported FIRST, as NumPy does.</param>
        /// <param name="expected">The size an earlier operand already fixed for this core dimension.</param>
        /// <exception cref="ValueError">The two sizes differ.</exception>
        internal static void RequireCoreSize(string name, string signature, int operandIndex, int coreDimension,
            long actual, long expected)
        {
            if (actual == expected)
                return;

            throw new ValueError(
                $"{name}: Input operand {operandIndex} has a mismatch in its core dimension {coreDimension}, " +
                $"with gufunc signature {signature} (size {actual} is different from {expected})");
        }

        /// <summary>
        ///     Applies a gufunc's <c>dtype=</c> (the loop selector) and <c>out=</c> in NumPy's order:
        ///     the loop runs at the requested dtype, and only then is the answer delivered into
        ///     <paramref name="out"/>.
        /// </summary>
        /// <param name="result">The gufunc's answer, computed at the loop dtype.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <returns><paramref name="result"/> when <paramref name="out"/> is null, else <paramref name="out"/> itself
        ///     holding the answer.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only, or <paramref name="result"/> does not
        ///     broadcast into its shape (<c>np.copyto</c>'s texts).</exception>
        /// <exception cref="System.InvalidCastException">The answer's dtype cannot reach <paramref name="out"/>'s under
        ///     <c>same_kind</c> (<c>np.copyto</c>'s text, not the gufunc's UFuncTypeError).</exception>
        internal static NDArray Deliver(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;

            np.copyto(@out, result);
            return @out;
        }

        /// <summary>Casts an operand to the requested loop dtype (a descriptor; <see langword="null"/> = no request), or returns it unchanged.</summary>
        /// <param name="operand">The operand.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <returns><paramref name="operand"/> itself without a request; else the cast (no copy when it already has the
        ///     dtype).</returns>
        /// <exception cref="System.NotSupportedException"><paramref name="dtype"/> is a class without storage yet
        ///     (datetime64 before Stage C).</exception>
        /// <remarks>The cast is UNSAFE: no casting rule is checked here (NumPy checks <c>same_kind</c> before the
        ///     loop).</remarks>
        internal static NDArray ToLoop(NDArray operand, DType dtype)
            // GetTypeCode() raises the descriptive "no storage yet" for a descriptor-only class (datetime64 before Stage C).
            => dtype is null ? operand : operand.TensorEngine.Cast(operand, dtype.GetTypeCode().AsType(), copy: false);

        /// <summary>
        ///     Validates a gufunc <c>axes=</c> list and normalizes each INPUT entry against its
        ///     operand's rank.
        /// </summary>
        /// <param name="name">The gufunc's name as NumPy prints it.</param>
        /// <param name="axes">The <c>axes=</c> list: one entry per operand, each the axes carrying that operand's core
        ///     dimensions.</param>
        /// <param name="coreRanks">
        ///     How many core dimensions each of the three slots has — <c>{x1, x2, out}</c>. For
        ///     <c>vecdot</c>'s <c>(n),(n)-&gt;()</c> that is <c>{1, 1, 0}</c>.
        /// </param>
        /// <param name="x1">The first input (its rank normalizes entry 0).</param>
        /// <param name="x2">The second input (its rank normalizes entry 1).</param>
        /// <param name="outputCoreRankOverride">
        ///     The output's EFFECTIVE core rank when it differs from the signature's — namely under
        ///     <c>keepdims</c>, where the reduced core dimension is kept as size 1 and a PROVIDED
        ///     output entry must name it. It changes only the length a provided output entry is
        ///     checked against; whether the entry may be OMITTED still keys off the signature rank
        ///     (<c>coreRanks[2]</c>), because keepdims does not add a signature core axis. Only
        ///     <c>vecdot</c> passes it — <c>matvec</c>/<c>vecmat</c> reject keepdims before reaching
        ///     here — so their behaviour is unchanged (<c>null</c> ⇒ the signature rank).
        /// </param>
        /// <returns>
        ///     Three entries. The two input entries hold non-negative axis indices; the OUTPUT entry
        ///     is returned as given, because the result's rank is not known until the loop has run.
        /// </returns>
        /// <exception cref="ValueError">The list has neither three entries nor (for an output without core axes)
        ///     two.</exception>
        /// <exception cref="AxisError">An entry's length is not its operand's core rank, or an input axis is out of
        ///     range.</exception>
        /// <remarks>
        ///     <c>axes</c> names, per operand, which axes carry the core dimensions — so it is the
        ///     general form of <c>axis</c>, and NumPy rejects the two together. The output entry may
        ///     be omitted ONLY when the output has no core axes, which among this family is
        ///     <c>vecdot</c> alone.
        /// </remarks>
        internal static int[][] NormalizeAxes(string name, int[][] axes, int[] coreRanks, NDArray x1, NDArray x2,
            int? outputCoreRankOverride = null)
        {
            if (axes.Length != 3 && !(axes.Length == 2 && coreRanks[2] == 0))
                throw new ValueError(
                    "axes should be a list with an entry for all 3 inputs and outputs; entries for " +
                    "outputs can only be omitted if none of them has core axes.");

            var resolved = new int[3][];
            resolved[2] = System.Array.Empty<int>();

            for (int i = 0; i < axes.Length; i++)
            {
                var entry = axes[i] ?? System.Array.Empty<int>();
                int expected = i == 2 ? (outputCoreRankOverride ?? coreRanks[2]) : coreRanks[i];
                if (entry.Length != expected)
                    throw new AxisError(
                        $"{name}: operand {i} has {expected} core dimensions, " +
                        $"but {entry.Length} dimensions are specified by axes tuple.");

                if (i == 2)
                {
                    resolved[2] = (int[])entry.Clone();
                    continue;
                }

                var operand = i == 0 ? x1 : x2;
                var normalized = new int[entry.Length];
                for (int k = 0; k < entry.Length; k++)
                {
                    int axis = entry[k] < 0 ? entry[k] + operand.ndim : entry[k];
                    if (axis < 0 || axis >= operand.ndim)
                        throw new AxisError(entry[k], operand.ndim);
                    normalized[k] = axis;
                }

                resolved[i] = normalized;
            }

            return resolved;
        }

        /// <summary>NumPy rejects <c>axis</c> and <c>axes</c> together, whichever gufunc it is.</summary>
        /// <param name="axes">The <c>axes=</c> list, or null.</param>
        /// <param name="axis">The <c>axis=</c> value, or null.</param>
        /// <exception cref="TypeError">Both are given (<c>cannot specify both 'axis' and 'axes'</c>).</exception>
        internal static void RejectAxisWithAxes(int[][] axes, int? axis)
        {
            if (axes is not null && axis is not null)
                throw new TypeError("cannot specify both 'axis' and 'axes'");
        }

        /// <summary>
        ///     The two rejections a gufunc whose core dimensions are DISTINCT raises for
        ///     <c>axis</c> and <c>keepdims</c> — <c>matvec</c> and <c>vecmat</c>, never
        ///     <c>vecdot</c>.
        /// </summary>
        /// <param name="name">The gufunc's name as NumPy prints it.</param>
        /// <param name="signature">The gufunc signature the messages quote.</param>
        /// <param name="axis">The <c>axis=</c> value, or null.</param>
        /// <param name="keepdims">The <c>keepdims=</c> value.</param>
        /// <param name="secondOperandCoreRank">
        ///     How many core dimensions input 1 has — 1 for <c>matvec</c>, 2 for <c>vecmat</c>.
        ///     NumPy names that operand in the keepdims message.
        /// </param>
        /// <exception cref="TypeError"><paramref name="axis"/> is given, or <paramref name="keepdims"/> is true.</exception>
        internal static void RejectAxisAndKeepdims(string name, string signature, int? axis, bool keepdims,
            int secondOperandCoreRank)
        {
            if (axis is not null)
                throw new TypeError(
                    $"{name}: axis can only be used with a single shared core dimension, not with " +
                    $"the 2 distinct ones implied by signature {signature}.");

            if (keepdims)
                throw new TypeError(
                    $"{name} does not support keepdims: its signature {signature} requires input 1 " +
                    $"to have {secondOperandCoreRank} core dimensions, but keepdims can only be used " +
                    "when all inputs have the same number of core dimensions and all outputs have no " +
                    "core dimensions.");
        }

        /// <summary>
        ///     The <c>order=</c> name check <c>np.matmul</c> and <c>np.einsum</c> share: one of <c>'C'</c>, <c>'F'</c>,
        ///     <c>'A'</c>, <c>'K'</c>.
        /// </summary>
        /// <param name="order">The order letter the caller passed.</param>
        /// <exception cref="ValueError">Any other character (<c>order must be one of 'C', 'F', 'A', or 'K' (got
        ///     '…')</c>), a lowercase letter included.</exception>
        internal static void RequireOrder(char order)
        {
            if (order is not ('C' or 'F' or 'A' or 'K'))
                throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
        }

        /// <summary>
        ///     The <c>casting=</c> name check <c>np.matmul</c> and <c>np.einsum</c> share: one of NumPy's five casting
        ///     rules.
        /// </summary>
        /// <param name="casting">The rule name the caller passed.</param>
        /// <exception cref="ValueError">Any other string, null included (<c>casting must be one of 'no', 'equiv',
        ///     'safe', 'same_kind', 'unsafe' (got '…')</c>).</exception>
        internal static void RequireCasting(string casting)
        {
            if (casting is not ("no" or "equiv" or "safe" or "same_kind" or "unsafe"))
                throw new ValueError(
                    $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
        }
    }
}
