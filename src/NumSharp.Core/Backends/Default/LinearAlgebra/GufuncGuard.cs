using NumSharp.Backends.Iteration;

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
        ///     The first checks of NumPy's <c>ufunc_generic_fastcall</c> for a gufunc call, in its order: <c>axis</c> with
        ///     <c>axes</c> (the keyword parse), the <c>dtype=</c> class (<c>_get_fixed_signature</c>), then a read-only
        ///     <c>out=</c> (<c>convert_ufunc_arguments</c>' <c>_set_out_array</c>). The casting and order names come next,
        ///     in the caller (only <c>np.matmul</c> takes them), then the loop resolution (<see cref="ThrowIfText"/>).
        /// </summary>
        /// <param name="axes">The <c>axes=</c> list, or null.</param>
        /// <param name="axis">The <c>axis=</c> value, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>, or a <paramref name="dtype"/>
        ///     with details beyond its class (<c>'U3'</c>, <c>'&gt;f8'</c>, a StringDType instance).</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>).</exception>
        /// <remarks>NumPy checks the out operand while it converts the arguments, before it resolves a loop or looks at
        ///     a shape, so a read-only out is refused even when the call would fail later (probed 2.4.2:
        ///     <c>np.vecdot(a, b, out=ro)</c> with mismatched core sizes names the read-only out).</remarks>
        internal static void CheckParsedArguments(int[][] axes, int? axis, DType dtype, NDArray @out)
        {
            RejectAxisWithAxes(axes, axis);
            if (dtype is not null)
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
        }

        /// <summary>
        ///     The loop resolution of a gufunc call for text (NumPy's <c>promote_and_get_ufuncimpl</c>): no gufunc of this
        ///     family has a string loop, so a StringDType operand, a string <c>dtype=</c> request, or a fixed-width string
        ///     operand without a numeric request raises NumPy's no-loop TypeError. Runs after
        ///     <see cref="CheckParsedArguments"/> and the casting / order names, and before <see cref="ValidateCasting"/> and
        ///     any core-dimension, <c>axes</c> / <c>axis</c> / <c>keepdims</c> semantics or shape check.
        /// </summary>
        /// <param name="name">NumPy's gufunc name (<c>matmul</c>, <c>vecdot</c>, <c>matvec</c>, <c>vecmat</c>).</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="dtype">The <c>dtype=</c> request (its class already checked), or null.</param>
        /// <exception cref="TypeError">NumPy's no-loop text: the operand CLASSES and the requested class (or None) for a
        ///     StringDType operand (<c>(&lt;class 'numpy.dtypes.StringDType'&gt;, …) -&gt; &lt;class
        ///     'numpy.dtypes.Float64DType'&gt;</c>), the operand classes and the requested class under a string request
        ///     (<c>(&lt;class 'numpy.dtypes.Float64DType'&gt;, …) -&gt; &lt;class 'numpy.dtypes.StrDType'&gt;</c>), the
        ///     operand DESCRIPTORS for a fixed-width string operand without a request
        ///     (<c>(dtype('&lt;U2'), dtype('float64')) -&gt; None</c>).</exception>
        /// <exception cref="System.NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        /// <remarks>
        ///     A fixed-width string operand under a NUMERIC request is not refused here: NumPy's legacy dispatcher takes the
        ///     requested loop, and <see cref="ValidateCasting"/> refuses the string input under same_kind (probed 2.4.2:
        ///     <c>np.matmul(s, s, dtype=np.float64)</c> is "Cannot cast ufunc 'matmul' input 0 from dtype('S1') to
        ///     dtype('float64') with casting rule 'same_kind'"), while under <c>casting='unsafe'</c> the loop converts the
        ///     text (the numbers it holds, or "could not convert string to float"). StringDType is a new-style DType,
        ///     which the legacy dispatcher does not serve: no loop, whatever the request.
        /// </remarks>
        internal static void ThrowIfText(string name, NDArray x1, NDArray x2, DType dtype)
        {
            bool request = Backends.Strings.NDStringUfuncs.IsStringRequest(dtype);
            bool vstring = Backends.Strings.NDStringUfuncs.IsVString(x1) || Backends.Strings.NDStringUfuncs.IsVString(x2);
            if (!request && !vstring && !Backends.Strings.NDStringUfuncs.IsString(x1) && !Backends.Strings.NDStringUfuncs.IsString(x2))
                return;
            // A StringDType operand: NumPy's dispatcher (a new-style DType in the key) names every operand's class, and the
            // requested class (any request: a numeric one too) as the output.
            if (vstring)
                throw Backends.Strings.NDStringUfuncs.VStringNoLoop(name,
                    new[] { Backends.Strings.NDStringUfuncs.DispatchClass(x1), Backends.Strings.NDStringUfuncs.DispatchClass(x2) },
                    dtype is null ? "None" : dtype.Meta.ToString());
            if (request)
                throw Backends.Strings.NDStringUfuncs.GufuncDTypeLoopError(name, dtype, x1, x2);
            // Fixed-width text without a request has no loop; under a numeric request the loop exists and the casts decide.
            if (dtype is null)
                throw Backends.Strings.NDStringUfuncs.NoLoop(name, x1.dtype, x2.dtype);
        }

        /// <summary>
        ///     The loop descriptor of a gufunc call with numeric (or, under a numeric request, fixed-width text) operands:
        ///     the <c>dtype=</c> request, else the operands' common dtype (NumPy's promotion of the two inputs).
        /// </summary>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="dtype">The <c>dtype=</c> request (its class already checked), or null.</param>
        /// <returns>The descriptor every input is cast to and the product is computed in.</returns>
        /// <exception cref="System.NotSupportedException"><paramref name="dtype"/> is a class without storage yet
        ///     (datetime64 before Stage C): NumSharp cannot hold its loop.</exception>
        /// <exception cref="DTypePromotionError">The operands have no common dtype.</exception>
        internal static DType ResolveLoop(NDArray x1, NDArray x2, DType dtype)
        {
            if (dtype is null)
                return DTypePromotion.PromoteTypes(x1.dtype, x2.dtype);
            // A descriptor-only class raises its own descriptive "no storage yet".
            _ = dtype.GetTypeCode();
            return dtype;
        }

        /// <summary>
        ///     NumPy's <c>PyUFunc_ValidateCasting</c> for a gufunc call (<c>validate_casting</c>, the first statement of
        ///     <c>PyUFunc_GeneralizedFunctionInternal</c>): each input's cast to the loop, in order, then the loop's cast to
        ///     <paramref name="out"/>, each under <paramref name="casting"/>. Runs after the loop resolution and before the
        ///     keepdims / axis support checks, the ranks, <c>axes=</c> and the core sizes.
        /// </summary>
        /// <param name="name">NumPy's gufunc name.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="loop">The loop descriptor (<see cref="ResolveLoop"/>).</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="casting">The casting rule: <c>np.matmul</c>'s argument (already validated), <c>"same_kind"</c>
        ///     for the gufuncs without the keyword (NumPy's default).</param>
        /// <exception cref="System.ArgumentException">A cast the rule refuses, with NumPy's UFuncTypeError text: "Cannot
        ///     cast ufunc '{name}' input {i} from {from} to {loop} with casting rule '{casting}'", or "... output from {loop}
        ///     to {out} ...".</exception>
        /// <remarks>
        ///     Checked by descriptor, so a text <paramref name="out"/> (whose storage lane alone carries no width or class)
        ///     gets NumPy's answer: a number to fixed-width text cast is same_kind (not safe), a number to StringDType is
        ///     safe, and the copy formats each value as its scalar's <c>str()</c> (probed 2.4.2: <c>matmul</c> of float64
        ///     into <c>out=U5</c> gives <c>'7.0'</c>; of complex128, <c>'(7+0j'</c>, cut at five characters).
        /// </remarks>
        internal static void ValidateCasting(string name, NDArray x1, NDArray x2, DType loop, NDArray @out, string casting)
        {
            var rule = DTypeCasting.ParseCasting(casting);
            RequireCast(name, "input 0", x1.dtype, loop, casting, rule);
            RequireCast(name, "input 1", x2.dtype, loop, casting, rule);
            if (@out is not null)
                RequireCast(name, "output", loop, @out.dtype, casting, rule);
        }

        /// <summary>One cast of <see cref="ValidateCasting"/>.</summary>
        /// <param name="name">NumPy's gufunc name.</param>
        /// <param name="role"><c>"input 0"</c>, <c>"input 1"</c> or <c>"output"</c>, as NumPy names the operand.</param>
        /// <param name="from">The source descriptor.</param>
        /// <param name="to">The target descriptor.</param>
        /// <param name="casting">The rule's name, as the message quotes it.</param>
        /// <param name="rule">The rule.</param>
        /// <exception cref="System.ArgumentException">The rule refuses the cast (NumPy's UFuncTypeError text).</exception>
        private static void RequireCast(string name, string role, DType from, DType to, string casting, NPY_CASTING rule)
        {
            if (from == to || DTypeCasting.CanCastTypeTo(from, to, rule))
                return;
            throw new System.ArgumentException(
                $"Cannot cast ufunc '{name}' {role} from {from.ToString(true)} to {to.ToString(true)} " +
                $"with casting rule '{casting}'");
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
        /// <exception cref="ValueError"><paramref name="result"/> does not broadcast into <paramref name="out"/>'s shape
        ///     (<c>np.copyto</c>'s text).</exception>
        /// <remarks>The copy is unsafe: <see cref="ValidateCasting"/> has checked the loop's cast to <paramref name="out"/>
        ///     under the call's rule before the gufunc checks, as NumPy does, and a read-only out was refused while the
        ///     arguments were parsed.</remarks>
        internal static NDArray Deliver(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;

            np.copyto(@out, result, "unsafe");   // cast already validated
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
