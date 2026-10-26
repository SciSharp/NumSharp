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
        ///     <paramref name="out"/>. Without <c>out=</c>, the answer is handed back as NumPy's ufunc machinery does:
        ///     a 0-d answer becomes NumPy's scalar, whose stand-in here is the 0-d array with a scalar's flags.
        /// </summary>
        /// <param name="result">The gufunc's answer, computed at the loop dtype: a fresh array.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <returns><paramref name="result"/> when <paramref name="out"/> is null (a 0-d one marked as the scalar
        ///     stand-in: an owner, not writeable), else <paramref name="out"/> itself holding the answer.</returns>
        /// <exception cref="ValueError"><paramref name="result"/> does not broadcast into <paramref name="out"/>'s shape
        ///     (<c>np.copyto</c>'s text).</exception>
        /// <remarks>
        ///     <para>The copy is unsafe: <see cref="ValidateCasting"/> has checked the loop's cast to <paramref name="out"/>
        ///     under the call's rule before the gufunc checks, as NumPy does, and a read-only out was refused while the
        ///     arguments were parsed.</para>
        ///     <para>The scalar step is NumPy's <c>replace_with_wrapped_result_and_return</c> (ufunc_object.c): it calls
        ///     <c>npy_apply_wrap</c> with <c>return_scalar</c> set when the output is 0-d and no <c>out=</c> was given,
        ///     which is <c>PyArray_Return</c>. NumSharp has no scalar type, so the 0-d array takes the scalar's flags
        ///     (<see cref="NDArray.MarkReductionScalar"/>, the reductions' rule; probed 2.4.2:
        ///     <c>np.matmul(a, b).flags</c> of two vectors and <c>np.vecdot(a, b).flags</c> are C, F, OWNDATA and not
        ///     WRITEABLE). A given <c>out=</c> clears <c>return_scalar</c>, so a 0-d out comes back itself, writeable.</para>
        /// </remarks>
        internal static NDArray Deliver(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result.MarkReductionScalar();   // no-op unless 0-d

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
        ///     NumPy's gufunc checks of <c>PyUFunc_GeneralizedFunctionInternal</c>, in its order: keepdims support, axis
        ///     support, the operands' ranks with the flexible dimensions dropped (<c>_validate_num_dims</c>), the
        ///     <c>axes=</c> / <c>axis=</c> remap (<c>_parse_axes_arg</c> / <c>_parse_axis_arg</c>), the core sizes
        ///     (<c>_get_coredim_sizes</c>), and the loop broadcast of the iterator it builds next
        ///     (<see cref="CheckLoopBroadcast"/>), over the inputs and a given out.
        /// </summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <param name="op">The operands: the inputs, then the output (null when no out= was given).</param>
        /// <param name="axes">The <c>axes=</c> list, or null.</param>
        /// <param name="axis">The <c>axis=</c> value, or null (never both: the argument parse refused that).</param>
        /// <param name="keepdimsPassed">Whether the caller passed keepdims (NumPy's <c>keepdims != -1</c>): a gufunc that
        ///     cannot keep dimensions refuses it whatever its value.</param>
        /// <param name="keepdims">The keepdims value.</param>
        /// <returns>The call's core-dimension facts, for the iterator check and the computation.</returns>
        /// <exception cref="TypeError">keepdims passed to a gufunc whose inputs have different core ranks or whose
        ///     outputs have core dimensions; axis given to a gufunc with more than one distinct core dimension.</exception>
        /// <exception cref="ValueError">An operand with too few dimensions; an <c>axes=</c> list of the wrong length; a
        ///     repeated axis in an <c>axes=</c> entry; a core dimension whose size differs between operands; loop axes that
        ///     do not broadcast, or an output that would need a broadcast or a reduction.</exception>
        /// <exception cref="AxisError">An <c>axes=</c> entry of the wrong length, or an axis out of its operand's
        ///     range.</exception>
        /// <remarks>
        ///     Ported statement for statement so the first error is NumPy's: the checks of one stage run over every operand
        ///     before the next stage starts, and an output's rank (when no out= is given) is the loop rank plus its core
        ///     rank. A flexible dimension (<c>n?</c> / <c>m?</c>) that an operand cannot provide is dropped from EVERY
        ///     operand that names it, so a short out= changes how the inputs are read (probed 2.4.2:
        ///     <c>np.matmul(np.zeros((3,4)), np.zeros((4,5)), out=np.zeros(()))</c> drops both and reports input 1's
        ///     <c>k</c> as 5).
        /// </remarks>
        internal static GufuncLayout CheckCoreDimensions(GufuncSignature sig, NDArray[] op, int[][] axes, int? axis,
            bool keepdimsPassed, bool keepdims)
        {
            if (keepdimsPassed)
                CheckKeepdimsSupport(sig);
            if (axis.HasValue)
                CheckAxisSupport(sig);

            int nin = sig.Nin, nop = sig.Nop;
            var layout = new GufuncLayout
            {
                OpCoreNumDims = new int[nop],
                Missing = new bool[sig.CoreNumDimIx],
                Keepdims = keepdimsPassed && keepdims,
            };
            for (int i = 0; i < nop; i++)
                layout.OpCoreNumDims[i] = sig.CoreNumDims(i);
            // keepdims set and true: every output keeps the inputs' core rank (as length-1 axes).
            if (layout.Keepdims)
                for (int i = nin; i < nop; i++)
                    layout.OpCoreNumDims[i] = layout.OpCoreNumDims[0];

            // _initialize_variable_parts: the call works on its own copy of the flexible flags.
            ValidateNumDims(sig, op, (bool[])sig.CanIgnore.Clone(), layout);

            // broadcast_ndim: the loop rank, from the operands that were given.
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null)
                    continue;
                int n = op[i].ndim - layout.OpCoreNumDims[i];
                if (n > layout.BroadcastNdim)
                    layout.BroadcastNdim = n;
            }

            if (axes is not null)
                layout.RemapAxis = ParseAxesArg(sig, op, axes, layout);
            else if (axis.HasValue)
                layout.RemapAxis = ParseAxisArg(sig, op, axis.Value, layout);

            layout.CoreDimSizes = GetCoreDimSizes(sig, op, layout);
            // The iterator NumPy builds next broadcasts the loop axes (and refuses an output that would need a broadcast
            // or a reduction) before the loop runs.
            CheckLoopBroadcast(sig, op, layout);
            return layout;
        }

        /// <summary>NumPy's <c>_check_keepdims_support</c>: keepdims needs every input to have the first input's core
        ///     rank and every output none.</summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <exception cref="TypeError">The signature does not allow keepdims (NumPy's text names the first operand that
        ///     breaks the rule).</exception>
        private static void CheckKeepdimsSupport(GufuncSignature sig)
        {
            int inputCoreDims = sig.CoreNumDims(0);
            for (int i = 1; i < sig.Nop; i++)
            {
                if (sig.CoreNumDims(i) == (i < sig.Nin ? inputCoreDims : 0))
                    continue;
                throw new TypeError(
                    $"{sig.Name} does not support keepdims: its signature {sig.Text} requires " +
                    $"{(i < sig.Nin ? "input" : "output")} {(i < sig.Nin ? i : i - sig.Nin)} to have " +
                    $"{sig.CoreNumDims(i)} core dimensions, but keepdims can only be used when all inputs have the " +
                    "same number of core dimensions and all outputs have no core dimensions.");
            }
        }

        /// <summary>NumPy's <c>_check_axis_support</c>: axis= needs a single shared core dimension.</summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <exception cref="TypeError">The signature has more than one distinct core dimension.</exception>
        private static void CheckAxisSupport(GufuncSignature sig)
        {
            if (sig.CoreNumDimIx != 1)
                throw new TypeError(
                    $"{sig.Name}: axis can only be used with a single shared core dimension, not with the " +
                    $"{sig.CoreNumDimIx} distinct ones implied by signature {sig.Text}.");
        }

        /// <summary>
        ///     NumPy's <c>_validate_num_dims</c>: an operand with fewer dimensions than its core rank drops its flexible
        ///     core dimensions (from every operand that names them, marking them missing) until it has enough, and is
        ///     refused when it still has not.
        /// </summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <param name="op">The operands (null for an output not given).</param>
        /// <param name="canIgnore">The call's copy of the flexible flags; a dropped dimension stops being
        ///     flexible.</param>
        /// <param name="layout">The call's layout: its core ranks and missing flags are updated.</param>
        /// <exception cref="ValueError">"{name}: {Input|Output} operand {i} does not have enough dimensions (has {n},
        ///     gufunc core with signature {sig} requires {k})", k being the core rank after the drops.</exception>
        private static void ValidateNumDims(GufuncSignature sig, NDArray[] op, bool[] canIgnore, GufuncLayout layout)
        {
            int nin = sig.Nin, nop = sig.Nop;
            var opCore = layout.OpCoreNumDims;
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null)
                    continue;
                int ndim = op[i].ndim;
                if (ndim >= opCore[i])
                    continue;
                foreach (int ix in sig.CoreDimIxs[i])
                {
                    if (!canIgnore[ix])
                        continue;
                    // Drop the flexible dimension from every operand that uses it (ours included), then see whether
                    // this operand has enough dimensions now.
                    layout.Missing[ix] = true;
                    canIgnore[ix] = false;
                    for (int i1 = 0; i1 < nop; i1++)
                        foreach (int j1 in sig.CoreDimIxs[i1])
                            if (j1 == ix)
                                opCore[i1]--;
                    if (ndim == opCore[i])
                        break;
                }
                if (ndim < opCore[i])
                    throw new ValueError(
                        $"{sig.Name}: {(i < nin ? "Input" : "Output")} operand {(i < nin ? i : i - nin)} does not have " +
                        $"enough dimensions (has {ndim}, gufunc core with signature {sig.Text} requires {opCore[i]})");
            }
        }

        /// <summary>NumPy's <c>check_and_adjust_axis</c>.</summary>
        /// <param name="axis">The axis as given (negative counts from the end).</param>
        /// <param name="ndim">The rank it indexes.</param>
        /// <returns>The axis, normalized.</returns>
        /// <exception cref="AxisError">The axis is out of range (the message reports it as given).</exception>
        private static int CheckAndAdjustAxis(int axis, int ndim)
        {
            if (axis < -ndim || axis >= ndim)
                throw new AxisError(axis, ndim);
            return axis < 0 ? axis + ndim : axis;
        }

        /// <summary>
        ///     NumPy's <c>_parse_axis_arg</c>: <c>axis=</c> names the single shared core axis of every operand with core
        ///     dimensions; the remap moves it to the last default position.
        /// </summary>
        /// <param name="sig">The gufunc's core signature (one distinct core dimension).</param>
        /// <param name="op">The operands (null for an output not given).</param>
        /// <param name="axis">The axis as given.</param>
        /// <param name="layout">The call's layout (core ranks, loop rank).</param>
        /// <returns>Per operand its remap, or null for the identity.</returns>
        /// <exception cref="AxisError">The axis is out of an operand's range.</exception>
        private static int[][] ParseAxisArg(GufuncSignature sig, NDArray[] op, int axis, GufuncLayout layout)
        {
            var remap = new int[sig.Nop][];
            for (int iop = 0; iop < sig.Nop; iop++)
            {
                // _check_axis_support ensured the core rank is 0 or 1.
                if (layout.OpCoreNumDims[iop] == 0)
                    continue;
                int opNdim = op[iop] is not null ? op[iop].ndim : layout.BroadcastNdim + 1;
                int opAxis = CheckAndAdjustAxis(axis, opNdim);
                if (opAxis == opNdim - 1)
                    continue;
                var r = new int[opNdim];
                r[opNdim - 1] = opAxis;
                for (int a = 0; a < opAxis; a++)
                    r[a] = a;
                for (int a = opAxis; a < opNdim - 1; a++)
                    r[a] = a + 1;
                remap[iop] = r;
            }
            return remap;
        }

        /// <summary>
        ///     NumPy's <c>_parse_axes_arg</c>: one entry per operand (the outputs may be omitted only when none has core
        ///     dimensions), each naming the operand's core axes in core order; the remap places the loop axes first, in
        ///     their order, then the named core axes.
        /// </summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <param name="op">The operands (null for an output not given: its rank is the loop rank plus its core
        ///     rank).</param>
        /// <param name="axes">The <c>axes=</c> list; a null entry reads as an empty one.</param>
        /// <param name="layout">The call's layout (core ranks, loop rank).</param>
        /// <returns>Per operand its remap, or null for the identity (or an omitted output entry).</returns>
        /// <exception cref="ValueError">The list has the wrong length, or an entry repeats an axis.</exception>
        /// <exception cref="AxisError">An entry's length is not its operand's core rank, or an axis is out of
        ///     range.</exception>
        /// <remarks>Each operand's entry is checked completely (length, then each axis's range and repetition) before the
        ///     next operand's, so the first bad entry is the error, as in NumPy.</remarks>
        private static int[][] ParseAxesArg(GufuncSignature sig, NDArray[] op, int[][] axes, GufuncLayout layout)
        {
            int nin = sig.Nin, nop = sig.Nop;
            if (axes.Length != nop && (axes.Length != nin || sig.HasOutputCoreDims))
                throw new ValueError(
                    $"axes should be a list with an entry for all {nop} inputs and outputs; entries for " +
                    "outputs can only be omitted if none of them has core axes.");

            var remap = new int[nop][];
            for (int iop = 0; iop < axes.Length; iop++)
            {
                int ncore = layout.OpCoreNumDims[iop];
                int opNdim, nbroadcast;
                if (op[iop] is not null)
                {
                    opNdim = op[iop].ndim;
                    nbroadcast = opNdim - ncore;
                }
                else
                {
                    nbroadcast = layout.BroadcastNdim;
                    opNdim = layout.BroadcastNdim + ncore;
                }

                var entry = axes[iop] ?? System.Array.Empty<int>();
                if (entry.Length != ncore)
                    throw new AxisError(
                        $"{sig.Name}: operand {iop} has {ncore} core dimensions, " +
                        $"but {entry.Length} dimensions are specified by axes tuple.");

                // The core axes take the last default positions, in the entry's order.
                var seen = new bool[opNdim];
                var r = new int[opNdim];
                for (int a = nbroadcast; a < opNdim; a++)
                {
                    int opAxis = CheckAndAdjustAxis(entry[a - nbroadcast], opNdim);
                    if (seen[opAxis])
                        throw new ValueError($"axes item {iop} has value {opAxis} repeated");
                    seen[opAxis] = true;
                    r[a] = opAxis;
                }
                // The loop axes fill the first positions with the axes not named, in increasing order.
                for (int a = 0, next = 0; a < nbroadcast; a++)
                {
                    while (seen[next])
                        next++;
                    r[a] = next++;
                }

                bool identity = true;
                for (int a = 0; a < opNdim && identity; a++)
                    identity = r[a] == a;
                remap[iop] = identity ? null : r;
            }
            return remap;
        }

        /// <summary>
        ///     NumPy's <c>_get_coredim_sizes</c>: each core dimension takes its size from the first operand that has it, and
        ///     every later operand must match it exactly (a missing flexible dimension counts as 1); every output core
        ///     dimension must get a size.
        /// </summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <param name="op">The operands (null for an output not given).</param>
        /// <param name="layout">The call's layout (core ranks, missing flags, remap).</param>
        /// <returns>Per distinct core dimension, its size.</returns>
        /// <exception cref="ValueError">"{name}: {Input|Output} operand {i} has a mismatch in its core dimension {d}, with
        ///     gufunc signature {sig} (size {actual} is different from {expected})", or an output core dimension no operand
        ///     sizes.</exception>
        private static long[] GetCoreDimSizes(GufuncSignature sig, NDArray[] op, GufuncLayout layout)
        {
            int nin = sig.Nin, nop = sig.Nop;
            var sizes = new long[sig.CoreNumDimIx];
            System.Array.Fill(sizes, -1L);
            for (int i = 0; i < nop; i++)
            {
                if (op[i] is null)
                    continue;
                var dims = op[i].Shape.dimensions;
                int coreStart = dims.Length - layout.OpCoreNumDims[i];
                int delta = 0;
                var ixs = sig.CoreDimIxs[i];
                for (int idim = 0; idim < ixs.Length; idim++)
                {
                    int ix = ixs[idim];
                    long size;
                    if (layout.Missing[ix])
                    {
                        // A flexible dimension the operands dropped: its size is 1, and it takes no axis.
                        size = 1;
                        delta++;
                    }
                    else
                    {
                        int position = coreStart + idim - delta;
                        size = dims[layout.RemapAxis?[i] is { } remap ? remap[position] : position];
                    }

                    if (sizes[ix] < 0)
                        sizes[ix] = size;
                    else if (size != sizes[ix])
                        throw new ValueError(
                            $"{sig.Name}: {(i < nin ? "Input" : "Output")} operand {(i < nin ? i : i - nin)} has a " +
                            $"mismatch in its core dimension {idim - delta}, with gufunc signature {sig.Text} " +
                            $"(size {size} is different from {sizes[ix]})");
                }
            }

            for (int i = nin; i < nop; i++)
            {
                var ixs = sig.CoreDimIxs[i];
                for (int idim = 0; idim < ixs.Length; idim++)
                    if (sizes[ixs[idim]] < 0)
                        throw new ValueError(
                            $"{sig.Name}: Output operand {i - nin} has core dimension {idim} unspecified, with gufunc " +
                            $"signature {sig.Text}");
            }
            return sizes;
        }

        /// <summary>
        ///     The broadcast checks of the iterator NumPy's <c>PyUFunc_GeneralizedFunctionInternal</c> builds after the core
        ///     sizes (<c>NpyIter_AdvancedNew</c> with its <c>op_axes</c> and <c>iter_shape</c>, <c>npyiter_fill_axisdata</c>):
        ///     the loop axes of every given operand broadcast together, and an output may not need a broadcast or a
        ///     reduction.
        /// </summary>
        /// <param name="sig">The gufunc's core signature.</param>
        /// <param name="op">The operands (null for an output not given).</param>
        /// <param name="layout">The call's layout, complete through the core sizes.</param>
        /// <exception cref="ValueError">"operands could not be broadcast together with remapped shapes
        ///     [original-&gt;remapped]: ... and requested shape ..."; "non-broadcastable output operand with shape ...
        ///     [remapped to ...] doesn't match the broadcast shape ..."; "output operand requires a reduction along
        ///     dimension -1, but the reduction is not enabled. The dimension size of 1 does not match the expected output
        ///     shape."</exception>
        /// <remarks>
        ///     The iterator's dimensions are the loop axes, then every output core dimension; an operand maps each to one
        ///     of its axes, or to none (NumPy's -1, printed <c>newaxis</c>, or dropped while leading). The requested shape
        ///     is unknown (-1) on the loop axes and the core sizes after them. Two of NumPy's texts are kept with their
        ///     quirks: the remapped shapes end with a space before " and requested shape " (two spaces), and the
        ///     non-broadcastable output's remapped shape lists the iterator axes from the LAST one (probed 2.4.2:
        ///     <c>np.matmul(ones((2,3,4)), ones((2,4,5)), out=zeros((1,3,5)))</c> is "... (1,3,5) [remapped to (5,3,1)]
        ///     ... (2,3,5)"). Every gufunc of this family has one output, so the iterator axes past the loop axes are all
        ///     that output's core axes; NumPy marks another output's core axes as reduction axes, which only a second
        ///     output would meet.
        /// </remarks>
        private static void CheckLoopBroadcast(GufuncSignature sig, NDArray[] op, GufuncLayout layout)
        {
            int nin = sig.Nin, nop = sig.Nop;
            int bnd = layout.BroadcastNdim;
            int iterNdim = bnd;
            for (int i = nin; i < nop; i++)
                iterNdim += layout.OpCoreNumDims[i];

            // iter_shape: unknown on the loop axes, the core sizes (or 1 under keepdims) on each output's core axes.
            var itershape = new long[iterNdim];
            for (int idim = 0; idim < bnd; idim++)
                itershape[idim] = -1;

            // op_axes: an operand's axis per iterator dimension, or -1 (a missing loop axis, or another output's core
            // axis: NumPy's NPY_ITER_REDUCTION_AXIS(-1), which reads as -1 everywhere it matters here).
            var opAxes = new int[nop][];
            int j = bnd;
            for (int i = 0; i < nop; i++)
            {
                int n = op[i] is not null ? op[i].ndim - layout.OpCoreNumDims[i] : bnd;
                var axes = new int[iterNdim];
                for (int idim = 0; idim < bnd; idim++)
                    axes[idim] = idim >= bnd - n ? Remap(layout, i, idim - (bnd - n)) : -1;
                for (int idim = bnd; idim < iterNdim; idim++)
                    axes[idim] = -1;

                if (i >= nin)
                {
                    if (layout.Keepdims)
                    {
                        // keepdims: the kept axes are not core dimensions, they are added back with length 1.
                        for (int idim = 0; idim < layout.OpCoreNumDims[i]; idim++, j++)
                        {
                            itershape[j] = 1;
                            axes[j] = Remap(layout, i, n + idim);
                        }
                    }
                    else
                    {
                        int removed = 0;
                        var ixs = sig.CoreDimIxs[i];
                        for (int idim = 0; idim < ixs.Length; idim++)
                        {
                            // A missing flexible dimension takes no iterator axis.
                            if (layout.Missing[ixs[idim]])
                            {
                                removed++;
                                continue;
                            }
                            itershape[j] = layout.CoreDimSizes[ixs[idim]];
                            axes[j] = Remap(layout, i, n + idim - removed);
                            j++;
                        }
                    }
                }
                opAxes[i] = axes;
            }

            // npyiter_fill_axisdata: broadcast every given operand's mapped axes, starting from the requested shape.
            var bshape = new long[iterNdim];
            for (int idim = 0; idim < iterNdim; idim++)
                bshape[idim] = itershape[idim] < 0 ? 1 : itershape[idim];
            for (int iop = 0; iop < nop; iop++)
            {
                if (op[iop] is null)
                    continue;
                var dims = op[iop].Shape.dimensions;
                for (int idim = 0; idim < iterNdim; idim++)
                {
                    int i = opAxes[iop][idim];
                    if (i < 0 || i >= dims.Length)
                        continue;
                    if (bshape[idim] == 1)
                        bshape[idim] = dims[i];
                    else if (bshape[idim] != dims[i] && dims[i] != 1)
                        throw LoopBroadcastError(op, opAxes, itershape);
                }
            }
            // A requested length of 1 (a kept axis) may not have been broadcast up.
            for (int idim = 0; idim < iterNdim; idim++)
                if (itershape[idim] == 1 && bshape[idim] != 1)
                    throw LoopBroadcastError(op, opAxes, itershape);

            // The axis fill, from the last iterator axis: an output may neither broadcast (NO_BROADCAST) nor be reduced.
            for (int k = 0; k < iterNdim; k++)
            {
                int idim = iterNdim - k - 1;
                if (bshape[idim] == 1)
                    continue;
                for (int iop = nin; iop < nop; iop++)
                {
                    if (op[iop] is null)
                        continue;
                    int i = opAxes[iop][idim];
                    if (i < 0)
                        throw new ValueError(
                            $"output operand requires a reduction along dimension {i}, but the reduction is not " +
                            "enabled. The dimension size of 1 does not match the expected output shape.");
                    if (op[iop].Shape.dimensions[i] == 1)
                        throw NonBroadcastableOutputError(op[iop], opAxes[iop], bshape);
                }
            }
        }

        /// <summary>NumPy's <c>REMAP_AXIS(iop, axis)</c>: the actual axis behind a default position.</summary>
        /// <param name="layout">The call's layout.</param>
        /// <param name="i">The operand.</param>
        /// <param name="axis">The default position.</param>
        /// <returns>The remapped axis, or the position itself without a remap.</returns>
        private static int Remap(GufuncLayout layout, int i, int axis)
            => layout.RemapAxis?[i] is { } remap ? remap[axis] : axis;

        /// <summary>The iterator's broadcast error with remapped shapes (<c>npyiter_fill_axisdata</c>'s
        ///     <c>broadcast_error</c>, the op_axes and itershape form).</summary>
        /// <param name="op">The operands (null for an output not given: it is not printed).</param>
        /// <param name="opAxes">Each operand's axis per iterator dimension (-1 for none).</param>
        /// <param name="itershape">The requested shape (-1 for unknown).</param>
        /// <returns>The error to throw.</returns>
        private static ValueError LoopBroadcastError(NDArray[] op, int[][] opAxes, long[] itershape)
        {
            var sb = new System.Text.StringBuilder("operands could not be broadcast together with remapped shapes [original->remapped]: ");
            for (int iop = 0; iop < op.Length; iop++)
            {
                if (op[iop] is null)
                    continue;
                var dims = op[iop].Shape.dimensions;
                var remdims = new long[itershape.Length];
                for (int idim = 0; idim < remdims.Length; idim++)
                {
                    int i = opAxes[iop][idim];
                    remdims[idim] = i >= 0 && i < dims.Length ? dims[i] : -1;
                }
                sb.Append(Shape.ConvertShapeToString(dims)).Append("->").Append(Shape.ConvertShapeToString(remdims)).Append(' ');
            }
            sb.Append(" and requested shape ").Append(Shape.ConvertShapeToString(itershape));
            return new ValueError(sb.ToString());
        }

        /// <summary>The iterator's refusal to broadcast an output (<c>operand_different_than_broadcast</c>, the remapped
        ///     form), its remapped shape listed from the last iterator axis as NumPy prints it.</summary>
        /// <param name="out">The output operand.</param>
        /// <param name="axes">Its axis per iterator dimension (-1 for none).</param>
        /// <param name="bshape">The broadcast shape.</param>
        /// <returns>The error to throw.</returns>
        private static ValueError NonBroadcastableOutputError(NDArray @out, int[] axes, long[] bshape)
        {
            var dims = @out.Shape.dimensions;
            int ndim = axes.Length;
            var remdims = new long[ndim];
            for (int idim = 0; idim < ndim; idim++)
            {
                int i = axes[ndim - idim - 1];
                remdims[idim] = i >= 0 && i < dims.Length ? dims[i] : -1;
            }
            return new ValueError(
                $"non-broadcastable output operand with shape {Shape.ConvertShapeToString(dims)} [remapped to " +
                $"{Shape.ConvertShapeToString(remdims)}] doesn't match the broadcast shape {Shape.ConvertShapeToString(bshape)}");
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
        ///     The <c>order=</c> name check <c>np.matmul</c> and <c>np.einsum</c> share — NumPy's
        ///     <c>PyArray_OrderConverter</c> for one character: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>, in
        ///     either case (probed 2.4.2: <c>np.matmul(a, b, order='f')</c> is F-ordered, with no warning).
        /// </summary>
        /// <param name="order">The order letter the caller passed.</param>
        /// <returns>The letter in upper case, which is what the callers compare with.</returns>
        /// <exception cref="ValueError">Any other character (<c>order must be one of 'C', 'F', 'A', or 'K' (got
        ///     '…')</c>, quoting the character as given).</exception>
        internal static char RequireOrder(char order)
        {
            char upper = order switch { 'c' => 'C', 'f' => 'F', 'a' => 'A', 'k' => 'K', _ => order };
            if (upper is not ('C' or 'F' or 'A' or 'K'))
                throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            return upper;
        }

        /// <summary>
        ///     The <c>casting=</c> name check <c>np.matmul</c> and <c>np.einsum</c> share: one of NumPy's five casting
        ///     rules, spelled exactly (<c>'SAFE'</c> and <c>'safe '</c> are refused, probed 2.4.2).
        /// </summary>
        /// <param name="casting">The rule name the caller passed.</param>
        /// <exception cref="TypeError"><paramref name="casting"/> is null: NumPy's <c>casting must be str, not
        ///     NoneType</c> for <c>casting=None</c>.</exception>
        /// <exception cref="ValueError">Any other string (<c>casting must be one of 'no', 'equiv', 'safe', 'same_kind',
        ///     'unsafe' (got '…')</c>).</exception>
        internal static void RequireCasting(string casting)
        {
            if (casting is null)
                throw new TypeError("casting must be str, not NoneType");
            if (casting is not ("no" or "equiv" or "safe" or "same_kind" or "unsafe"))
                throw new ValueError(
                    $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
        }
    }
}
