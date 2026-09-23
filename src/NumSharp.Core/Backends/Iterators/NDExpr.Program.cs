using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Program.cs — the compiled program behind np.evaluate (ndexpr-evaluate.md Phase 3 + 6.1 + 3.3)
// =============================================================================
//
// np.evaluate used to redo, on EVERY call, everything that depends only on the tree and the
// operand dtypes: rebind the array leaves (a cloned tree), re-run the NEP50 typing pass (a
// reference-keyed Dictionary), rebuild the kernel's string cache key (a StringBuilder), and
// re-plan the vector emission — ~1.0 µs and ~3 KB of garbage per call at n = 8, three times the
// unfused chain's fixed cost, so every small-array call lost to the chain it replaces.
//
// A program is that dtype-dependent work done ONCE: the bound tree, the input dtype signature,
// the parameter mask (which 0-d inputs are hoisted into the kernel's aux block — NDExpr.Params.cs),
// the resolved result dtype and the compiled kernel (elementwise), or the reduction node with its
// lazily compiled flat / axis kernels. It holds NO NDArray — the inputs are resolved per root (the
// arrays the tree references, in binding order) and handed to the engine beside it — so one
// program serves every tree of the same structure over any arrays of the same dtypes and shape
// class.
//
// Two caches sit in front of Build:
//   • the per-root slot (NDExpr._resolved): a tree the caller hoisted or compiled is served with
//     no walk at all — the program and its input list come straight off the root (the parameter
//     mask is re-validated, since a 0-d array can be resized in place);
//   • the GLOBAL structural cache (NDExprProgramCache): the natural spelling
//     `np.evaluate((NDExpr)a * b + c)` inside a loop builds a NEW root every call, and used to
//     miss the per-root slot by construction. Its key is the tree's 64-bit structural hash
//     (NDExpr.Structure.cs) + the dtype signature + the parameter mask; a hit is VERIFIED node by
//     node against the cached program's bound tree before it is used, because a hash-only match
//     would run the wrong kernel. The natural spelling now costs two allocation-free walks over
//     the tree instead of a bind + typing + string key.
//
// Programs are immutable snapshots, so both caches are benign races: two threads racing on a
// fresh structure compile the same kernel (the kernel cache dedups it) and one of them wins.
//
// CompiledExpression (NDExpr.Compile) is the explicit handle over the same machinery — numexpr's
// NumExpr(expr) object: compile ahead of the hot loop, evaluate many times, and for the
// positional form pin ONE dtype signature (a call with other dtypes is an error, not a silent
// recompile).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// The dtype-dependent, array-independent half of an evaluation: bound tree, input signature,
    /// parameter mask, result dtype and kernel(s). Immutable after <see cref="Build"/> (the reduce
    /// kernels compile lazily but idempotently), shared across every root of the same structure,
    /// signature and mask.
    /// </summary>
    internal sealed class NDExprProgram
    {
        /// <summary>The tree with every array leaf rewritten to a positional <see cref="InputNode"/>.</summary>
        public readonly NDExpr Bound;

        /// <summary>Every input's dtype (iterator operands AND parameters) in input (binding) order.</summary>
        public readonly NPTypeCode[] InputTypes;

        /// <summary>
        /// Per input, whether it is a PARAMETER — a 0-d array the host copies into the kernel's aux
        /// block instead of streaming through the iterator (NDExpr.Params.cs). Null when no input is
        /// hoisted (the program then evaluates any shapes through the iterator).
        /// </summary>
        public readonly bool[] IsParam;

        /// <summary>Number of hoisted parameters (0 when <see cref="IsParam"/> is null).</summary>
        public readonly int ParamCount;

        /// <summary>True when compiled under the <see cref="NDExpr.ForceScalar"/> test hook (its own cache entry).</summary>
        public readonly bool ForcedScalar;

        /// <summary>The fused elementwise kernel over the ITERATOR operands; null when the root is a reduction.</summary>
        public readonly NDInnerLoopFunc Kernel;

        /// <summary>The tree's NumPy result dtype (the elementwise output, or the reduction's result).</summary>
        public readonly NPTypeCode ResultType;

        /// <summary>The root reduction, or null for an elementwise tree.</summary>
        public readonly ReduceNode Reduce;

        /// <summary>The reduction's accumulator dtype (see <see cref="ReduceNode.ResolveAccType"/>).</summary>
        public readonly NPTypeCode ReduceAccType;

        private NDInnerLoopFunc _flatReduce;
        private NDInnerLoopFunc _axisReduce;

        private NDExprProgram(NDExpr bound, NPTypeCode[] inputTypes, bool[] isParam, bool forcedScalar,
            NDInnerLoopFunc kernel, NPTypeCode resultType, ReduceNode reduce, NPTypeCode reduceAcc)
        {
            Bound = bound;
            InputTypes = inputTypes;
            IsParam = isParam;
            ParamCount = isParam is null ? 0 : CountTrue(isParam);
            ForcedScalar = forcedScalar;
            Kernel = kernel;
            ResultType = resultType;
            Reduce = reduce;
            ReduceAccType = reduceAcc;
        }

        private static int CountTrue(bool[] mask)
        {
            int n = 0;
            foreach (var b in mask)
                if (b) n++;
            return n;
        }

        /// <summary>The flat (axis=None) accumulating kernel, compiled on first use.</summary>
        public NDInnerLoopFunc FlatReduceKernel
            => _flatReduce ??= Reduce.CompileReduceKernel(InputTypes, IsParam, out _, out _);

        /// <summary>The axis-aware accumulating kernel (axis-independent), compiled on first use.</summary>
        public NDInnerLoopFunc AxisReduceKernel
            => _axisReduce ??= Reduce.CompileAxisReduceKernel(InputTypes, IsParam, out _, out _);

        /// <summary>
        /// Does this program serve these inputs: same count, same typecode per slot, and every input the
        /// program hoists as a parameter is still 0-d? (An operand-slot input may be any shape — the
        /// iterator handles a 0-d operand, just less cheaply — so a parameter-free program serves ANY
        /// shapes, which is what a positional <see cref="CompiledExpression"/> relies on.)
        /// </summary>
        /// <param name="inputs">The inputs of a call.</param>
        /// <returns>True when the program computes these inputs correctly.</returns>
        public bool Matches(NDArray[] inputs)
        {
            if (inputs.Length != InputTypes.Length)
                return false;
            for (int i = 0; i < inputs.Length; i++)
                if (inputs[i].typecode != InputTypes[i])
                    return false;
            return ParamsStillValid(inputs);
        }

        /// <summary>Is every input the program hoists as a parameter still a 0-d array? (A 0-d array can be resized in place.)</summary>
        /// <param name="inputs">The inputs the program was resolved for.</param>
        /// <returns>True when the hoisted reads are still correct.</returns>
        public bool ParamsStillValid(NDArray[] inputs)
        {
            var mask = IsParam;
            if (mask is null)
                return true;
            for (int i = 0; i < mask.Length; i++)
                if (mask[i] && inputs[i].ndim != 0)
                    return false;
            return true;
        }

        /// <summary>Does this program carry exactly <paramref name="types"/> and <paramref name="mask"/> (empty = no parameters) as its signature?</summary>
        /// <param name="types">A dtype signature.</param>
        /// <param name="mask">A parameter mask; empty when no input is a parameter.</param>
        /// <returns>True when both agree exactly.</returns>
        public bool SignatureEquals(ReadOnlySpan<NPTypeCode> types, ReadOnlySpan<bool> mask)
        {
            if (!types.SequenceEqual(InputTypes))
                return false;
            return IsParam is null ? mask.IsEmpty : mask.SequenceEqual(IsParam);
        }

        /// <summary>The inputs the iterator streams: <paramref name="inputs"/> itself without parameters, else the unflagged ones in order.</summary>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <returns>The iterator operand list.</returns>
        public NDArray[] IteratorOperands(NDArray[] inputs)
        {
            if (IsParam is null)
                return inputs;
            var ops = new NDArray[inputs.Length - ParamCount];
            for (int i = 0, k = 0; i < inputs.Length; i++)
                if (!IsParam[i])
                    ops[k++] = inputs[i];
            return ops;
        }

        /// <summary>
        /// Copy each parameter's single element (its own dtype, raw bytes) into the aux block at
        /// <paramref name="dst"/> + 16·j — the layout the kernel prologue reads. Reading happens BEFORE
        /// the pass, so a parameter that aliases the output keeps its original value.
        /// </summary>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="dst">Parameter slot 0 of the aux block (16 bytes per parameter).</param>
        public unsafe void PackParams(NDArray[] inputs, byte* dst)
        {
            var mask = IsParam;
            for (int i = 0, j = 0; i < mask.Length; i++)
            {
                if (!mask[i])
                    continue;
                var a = inputs[i];
                int size = a.typecode.SizeOf();
                // Logical element 0 of a 0-d view: base address + the view's element offset.
                byte* src = (byte*)a.Address + (long)a.Shape.offset * size;
                Buffer.MemoryCopy(src, dst + NDExprParamPlan.SlotBytes * j, NDExprParamPlan.SlotBytes, size);
                j++;
            }
        }

        /// <summary>
        /// Bind, type and compile <paramref name="root"/> for an input dtype signature and parameter mask.
        /// This is the cold path — bind clone, NEP50 typing pass, kernel JIT (~0.2–1 ms for a never-seen
        /// tree shape); every cache in front of it exists to keep it off the per-call path.
        /// </summary>
        /// <param name="root">The tree to compile.</param>
        /// <param name="positionalTypes">The positional dtype signature, or null for the embedded form (read off the embedded arrays in binding order).</param>
        /// <param name="isParam">Per input, whether it is hoisted as a parameter; null for none. Must be null or match the input count.</param>
        /// <returns>A fresh program.</returns>
        /// <exception cref="ArgumentException">
        /// The embedded form references no arrays, a positional signature is given for a tree that
        /// embeds arrays (mixed binding styles), or the mask flags every input.
        /// </exception>
        /// <exception cref="NotSupportedException">A reduction sits below the root, or nests another.</exception>
        public static NDExprProgram Build(NDExpr root, NPTypeCode[] positionalTypes, bool[] isParam = null)
        {
            var bind = new NDExprBindContext();
            var bound = root.BindArrays(bind);

            NPTypeCode[] inputTypes;
            if (positionalTypes is null)
            {
                if (bind.Operands.Count == 0)
                    throw new ArgumentException(NDExpr.NoArraysMessage, "expr");
                inputTypes = new NPTypeCode[bind.Operands.Count];
                for (int i = 0; i < inputTypes.Length; i++)
                    inputTypes[i] = bind.Operands[i].typecode;
            }
            else
            {
                if (bind.Operands.Count != 0)
                    throw new ArgumentException(NDExpr.MixedBindingMessage, "expr");
                inputTypes = positionalTypes;
            }

            if (isParam is not null && isParam.Length != inputTypes.Length)
                throw new ArgumentException("parameter mask length must equal the input count.", nameof(isParam));

            bool forcedScalar = NDExpr.ForceScalar;

            if (bound is ReduceNode reduce)
            {
                if (reduce.Child.ContainsReduce)
                    throw new NotSupportedException(
                        "nested reductions are not supported — a reduction must be the root of the expression.");

                var resolved = reduce.ResolveNumPyTypes(inputTypes, out _);
                var acc = ReduceNode.ResolveAccType(reduce.Kind, resolved);
                return new NDExprProgram(bound, inputTypes, isParam, forcedScalar, null, resolved, reduce, acc);
            }

            if (bound.ContainsReduce)
                throw new NotSupportedException(
                    "reduction nodes must be the root of the expression — " +
                    "elementwise use of a reduced value needs two np.evaluate calls.");

            var kernel = bound.CompileNumPy(inputTypes, isParam, out var resultType);
            return new NDExprProgram(bound, inputTypes, isParam, forcedScalar, kernel, resultType, null, NPTypeCode.Empty);
        }

        /// <summary>NumPy-style rendering of a dtype signature: <c>(float64, int32)</c>.</summary>
        /// <param name="types">The signature to render.</param>
        /// <returns>The parenthesized, comma-separated NumPy dtype names.</returns>
        public static string SignatureText(IReadOnlyList<NPTypeCode> types)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < types.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(types[i].AsNumpyDtypeName());
            }

            return sb.Append(')').ToString();
        }
    }

    /// <summary>
    /// What a root resolved to: its program plus, for the embedded-array form, the input list the
    /// program evaluates against (null for the positional form, whose inputs arrive per call).
    /// One immutable object so the root publishes both with a single reference write — a reader can
    /// never pair a new program with a stale input list.
    /// </summary>
    [NDBorrowed] // the inputs are the caller's arrays, referenced by the tree itself
    internal sealed class NDExprResolution
    {
        public readonly NDExprProgram Program;
        public readonly NDArray[] Inputs;

        public NDExprResolution(NDExprProgram program, NDArray[] inputs)
        {
            Program = program;
            Inputs = inputs;
        }
    }

    /// <summary>
    /// The process-wide program cache keyed by a tree's structural hash + dtype signature + parameter
    /// mask (NDExpr.Structure.cs). It is what makes a tree REBUILT per call as cheap as a hoisted one.
    /// Every hit is verified structurally before use; the hash only picks the candidate. One entry per
    /// hash: a genuine 64-bit collision between two live structures merely makes them evict each other
    /// (correct, slower), it can never pair a tree with a foreign program. Bounded by
    /// <see cref="Capacity"/> — when full it is cleared wholesale (the compiled kernels stay in the
    /// kernel cache, so a re-added entry costs a bind + typing pass, not a JIT).
    /// </summary>
    internal static class NDExprProgramCache
    {
        /// <summary>
        /// Entry cap before a wholesale clear. A loop that bakes a NEW literal value into the tree every
        /// call (<c>a * t</c> with a changing <c>t</c>) creates one program AND one kernel per value —
        /// spell such a value as a 0-d array instead (a hoisted parameter, NDExpr.Params.cs: one kernel
        /// for every value, no per-element cost); the cap keeps the literal footgun from growing the
        /// cache without bound. Settable for tests.
        /// </summary>
        internal static int Capacity = 4096;

        private static readonly ConcurrentDictionary<ulong, NDExprProgram> s_entries = new();
        private static long s_hits;
        private static long s_misses;

        /// <summary>Number of cached programs.</summary>
        internal static int Count => s_entries.Count;

        /// <summary>Verified hits since process start (diagnostics / tests).</summary>
        internal static long Hits => Interlocked.Read(ref s_hits);

        /// <summary>Lookups that found no verified program since process start (diagnostics / tests).</summary>
        internal static long Misses => Interlocked.Read(ref s_misses);

        /// <summary>
        /// The cached program for <paramref name="root"/> under <paramref name="hash"/>, or null. A candidate
        /// is accepted only when its scalar/vector mode, dtype signature, parameter mask AND node-by-node
        /// structure agree.
        /// </summary>
        /// <param name="hash">The structural hash computed over <paramref name="root"/> with the same signature, mask and mode.</param>
        /// <param name="root">The tree being evaluated (array leaves allowed).</param>
        /// <param name="operands">The input list collected over <paramref name="root"/> by the hash walk.</param>
        /// <param name="types">The dtype signature of the call.</param>
        /// <param name="mask">The parameter mask of the call (empty = none).</param>
        /// <param name="forceScalar">The <see cref="NDExpr.ForceScalar"/> mode of the call.</param>
        /// <returns>The verified program, or null.</returns>
        internal static NDExprProgram TryFind(ulong hash, NDExpr root, NDExprOperandScratch operands,
            ReadOnlySpan<NPTypeCode> types, ReadOnlySpan<bool> mask, bool forceScalar)
        {
            if (s_entries.TryGetValue(hash, out var program)
                && program.ForcedScalar == forceScalar
                && program.SignatureEquals(types, mask)
                && root.StructureEquals(program.Bound, operands))
            {
                Interlocked.Increment(ref s_hits);
                return program;
            }

            Interlocked.Increment(ref s_misses);
            return null;
        }

        /// <summary>Publish a freshly built program under its hash (replacing a colliding entry; clearing first when full).</summary>
        /// <param name="hash">The structural hash the program was built for.</param>
        /// <param name="program">The program.</param>
        internal static void Add(ulong hash, NDExprProgram program)
        {
            if (s_entries.Count >= Capacity)
                s_entries.Clear();
            s_entries[hash] = program;
        }

        /// <summary>Drop every entry and reset the counters (tests).</summary>
        internal static void Clear()
        {
            s_entries.Clear();
            Interlocked.Exchange(ref s_hits, 0);
            Interlocked.Exchange(ref s_misses, 0);
        }
    }

    [NDBorrowed] // a root's resolution references the caller's arrays through the tree that already holds them
    public abstract partial class NDExpr
    {
        internal const string NoArraysMessage =
            "expression references no arrays — embed NDArrays in the tree " +
            "(NDExpr.Arr / implicit conversion) or use the (expr, operands) overload with NDExpr.Input(i) leaves.";

        internal const string MixedBindingMessage =
            "expression mixes embedded array leaves with a positional operand list — use one binding style.";

        /// <summary>Input counts up to this many resolve their signature on the stack.</summary>
        private const int StackTypesLimit = 128;

        private NDExprResolution _resolved;

        // One scratch per thread: the walks run no user code, so a walk can never re-enter another on
        // the same thread, and Reset() runs before the call returns, so no caller array is retained.
        [ThreadStatic] private static NDExprOperandScratch t_scratch;

        private static NDExprOperandScratch Scratch
        {
            get
            {
                var s = t_scratch ??= new NDExprOperandScratch();
                s.Reset();                                  // a throwing walk may have left entries behind
                return s;
            }
        }

        /// <summary>
        /// The compiled program for this root (see <see cref="GetProgram(NDArray[], out NDArray[])"/>);
        /// the input list is discarded.
        /// </summary>
        /// <param name="positional">The positional operands, or null for the embedded-array form.</param>
        /// <returns>The program serving this root for these inputs' dtypes and shape class.</returns>
        internal NDExprProgram GetProgram(NDArray[] positional) => GetProgram(positional, out _);

        /// <summary>
        /// The compiled program for this root and the input list to evaluate it against: the per-root
        /// slot when it still applies (same binding form, same input dtypes, same scalar/vector mode,
        /// hoisted parameters still 0-d), else the global structural cache (two allocation-free walks),
        /// else a fresh build that both caches then hold. The embedded form's input list is the distinct
        /// arrays the tree references in binding order (0-d ones included — the program says which are
        /// parameters); the positional form returns <paramref name="positional"/> itself.
        /// </summary>
        /// <param name="positional">The positional operands, or null for the embedded-array form.</param>
        /// <param name="inputs">The input list <see cref="TensorEngine.Evaluate(NDExprProgram, NDArray[], NDArray)"/> takes.</param>
        /// <returns>The program serving this root for these inputs' dtypes and shape class.</returns>
        /// <exception cref="ArgumentException">The embedded form references no arrays, or the tree mixes binding styles.</exception>
        /// <exception cref="NotSupportedException">A reduction sits below the root, or nests another.</exception>
        internal NDExprProgram GetProgram(NDArray[] positional, out NDArray[] inputs)
        {
            bool forceScalar = ForceScalar;
            var r = _resolved;
            if (r is not null && r.Program.ForcedScalar == forceScalar)
            {
                if (positional is null)
                {
                    if (r.Inputs is not null && r.Program.ParamsStillValid(r.Inputs))
                    {
                        inputs = r.Inputs;
                        return r.Program;
                    }
                }
                else if (r.Inputs is null && r.Program.Matches(positional))
                {
                    inputs = positional;
                    return r.Program;
                }
            }

            return ResolveSlow(positional, forceScalar, out inputs);
        }

        /// <summary>
        /// The per-root slot missed (fresh root, other dtypes, mode toggled, a parameter resized): walk
        /// the tree once to hash its structure and collect its arrays, consult the global cache, build on
        /// a miss, and publish the resolution on the root.
        /// </summary>
        private NDExprProgram ResolveSlow(NDArray[] positional, bool forceScalar, out NDArray[] inputs)
        {
            var scratch = Scratch;
            try
            {
                var h = NDExprStructureHasher.Create();
                HashStructure(ref h, scratch);

                bool embedded = positional is null;
                if (embedded && scratch.Count == 0)
                    throw new ArgumentException(NoArraysMessage, "expr");
                if (!embedded && scratch.Count != 0)
                    throw new ArgumentException(MixedBindingMessage, "expr");

                int n = embedded ? scratch.Count : positional.Length;
                Span<NPTypeCode> types = n <= StackTypesLimit ? stackalloc NPTypeCode[n] : new NPTypeCode[n];
                Span<bool> mask = n <= StackTypesLimit ? stackalloc bool[n] : new bool[n];
                bool anyParam = false, allParam = true;
                for (int i = 0; i < n; i++)
                {
                    var a = embedded ? scratch[i] : positional[i];
                    types[i] = a.typecode;
                    bool p = a.ndim == 0;
                    mask[i] = p;
                    anyParam |= p;
                    allParam &= p;
                }

                // A 0-d input is hoisted as a parameter — unless EVERY input is 0-d, in which case the
                // result is 0-d and the iterator keeps computing it as before (a kernel needs at least one
                // streamed operand).
                if (!anyParam || allParam)
                    mask = default;

                ulong hash = FinishHash(ref h, types, mask, forceScalar);
                var program = NDExprProgramCache.TryFind(hash, this, scratch, types, mask, forceScalar);
                if (program is null)
                {
                    program = NDExprProgram.Build(this, embedded ? null : types.ToArray(), mask.IsEmpty ? null : mask.ToArray());
                    NDExprProgramCache.Add(hash, program);
                }

                if (embedded)
                {
                    inputs = scratch.ToArray();
                    _resolved = new NDExprResolution(program, inputs);
                }
                else
                {
                    inputs = positional;
                    _resolved = new NDExprResolution(program, null);
                }

                return program;
            }
            finally
            {
                scratch.Reset();
            }
        }

        /// <summary>
        /// Close the structural hash with what else selects a program: the dtype signature (same tree,
        /// other dtypes → other kernel), the parameter mask (a hoisted input changes the emitted loads)
        /// and the scalar/vector mode.
        /// </summary>
        private static ulong FinishHash(ref NDExprStructureHasher h, ReadOnlySpan<NPTypeCode> types, ReadOnlySpan<bool> mask, bool forceScalar)
        {
            h.Add((ulong)types.Length);
            for (int i = 0; i < types.Length; i++)
                h.Add((ulong)types[i] | (mask.IsEmpty || !mask[i] ? 0UL : 1UL << 32));
            h.Add(forceScalar ? 1UL : 0UL);
            return h.Value;
        }

        /// <summary>
        ///     Compile this tree ahead of its first evaluation (NumSharp extension; numexpr's
        ///     <c>NumExpr(expr)</c> object). The tree must reference its arrays directly
        ///     (<see cref="Arr"/> / implicit conversion): binding, the NumPy typing pass and the
        ///     kernel JIT (~1 ms for a new tree shape) run here, and every
        ///     <see cref="CompiledExpression.Evaluate(NDArray)"/> afterwards does only the per-call
        ///     work — the iteration shape, the result allocation and the iterator (~0.4 µs at n = 8).
        ///     <c>np.evaluate(expr)</c> reaches the same cached program implicitly — on this root's
        ///     second call, or on any root of the same structure through the global structural cache;
        ///     the handle makes the compile step explicit and hoistable. A 0-d array leaf is compiled
        ///     as a hoisted parameter (read per evaluation, never streamed).
        ///     A positional tree (<see cref="Input"/> leaves) compiles with
        ///     <see cref="Compile(NPTypeCode[])"/>, which pins its operand dtypes.
        /// </summary>
        /// <returns>The handle over this root's program and its input list.</returns>
        /// <exception cref="InvalidOperationException">The tree has no array leaves.</exception>
        public CompiledExpression Compile()
        {
            if (FirstArray() is null)
                throw new InvalidOperationException(
                    "expression references no arrays — a positional tree (NDExpr.Input leaves) is compiled " +
                    "with Compile(params NPTypeCode[] inputTypes), which pins the operand dtypes.");

            var program = GetProgram(null, out var inputs);
            return new CompiledExpression(this, program, inputs);
        }

        /// <summary>
        ///     Compile a positional tree (<see cref="Input"/> leaves) for one operand dtype signature —
        ///     <c>inputTypes[i]</c> is the dtype <c>Input(i)</c> will be evaluated against. The handle's
        ///     <see cref="CompiledExpression.Evaluate(NDArray[], NDArray)"/> requires exactly that signature
        ///     (a different one is a <see cref="TypeError"/>, never a silent recompile), so a hot loop
        ///     pays no per-call typing. Every operand streams through the iterator (a 0-d operand is a
        ///     stride-0 stream, not a hoisted parameter — the handle knows dtypes, not shapes).
        ///     Not the Tier-3C <see cref="Compile(NPTypeCode[], NPTypeCode, string)"/>,
        ///     which emits a raw inner-loop kernel computed entirely at ONE output dtype; this one types
        ///     every node with NumPy's result_type, exactly as <c>np.evaluate</c> does.
        /// </summary>
        /// <param name="inputTypes">The dtype of each positional operand.</param>
        /// <returns>The handle over the program compiled for that signature.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="inputTypes"/> is null.</exception>
        /// <exception cref="ArgumentException">
        /// The signature is empty or names a non-storage dtype, or the tree embeds arrays (mixed binding styles).
        /// </exception>
        /// <exception cref="NotSupportedException">A reduction sits below the root, or nests another.</exception>
        public CompiledExpression Compile(params NPTypeCode[] inputTypes)
        {
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));
            if (inputTypes.Length == 0)
                throw new ArgumentException("inputTypes must name at least one operand dtype.", nameof(inputTypes));
            for (int i = 0; i < inputTypes.Length; i++)
                if (inputTypes[i] == NPTypeCode.Empty)
                    throw new ArgumentException($"inputTypes[{i}] is not a storage dtype.", nameof(inputTypes));

            var types = (NPTypeCode[])inputTypes.Clone();
            bool forceScalar = ForceScalar;
            var scratch = Scratch;
            NDExprProgram program;
            try
            {
                var h = NDExprStructureHasher.Create();
                HashStructure(ref h, scratch);
                if (scratch.Count != 0)
                    throw new ArgumentException(MixedBindingMessage, "expr");
                ulong hash = FinishHash(ref h, types, default, forceScalar);

                program = NDExprProgramCache.TryFind(hash, this, scratch, types, default, forceScalar);
                if (program is null)
                {
                    program = NDExprProgram.Build(this, types);
                    NDExprProgramCache.Add(hash, program);
                }
            }
            finally
            {
                scratch.Reset();
            }

            _resolved = new NDExprResolution(program, null);
            return new CompiledExpression(this, program, null);
        }
    }

    /// <summary>
    ///     An <see cref="NDExpr"/> tree compiled ahead of evaluation — the handle
    ///     <see cref="NDExpr.Compile()"/> / <see cref="NDExpr.Compile(NPTypeCode[])"/> return. Holds the
    ///     bound tree, the resolved dtypes and the fused kernel; <see cref="Evaluate(NDArray)"/> runs the
    ///     per-call work only. Immutable and safe to share across threads (each evaluation builds its own
    ///     iterator). The arrays an embedded-form handle references are the caller's: it neither owns nor
    ///     disposes them, and it reads their CURRENT contents on every evaluation.
    /// </summary>
    [NDBorrowed] // the inputs are the caller's arrays, referenced by the tree itself
    public sealed class CompiledExpression
    {
        private readonly NDExpr _root;
        private readonly NDExprProgram _program;
        private readonly NDArray[] _inputs;   // embedded form; null for the positional form

        internal CompiledExpression(NDExpr root, NDExprProgram program, NDArray[] inputs)
        {
            _root = root;
            _program = program;
            _inputs = inputs;
        }

        /// <summary>The tree this handle was compiled from.</summary>
        public NDExpr Expression => _root;

        /// <summary>True for a positional tree (<see cref="NDExpr.Input"/> leaves, operands supplied per call).</summary>
        public bool IsPositional => _inputs is null;

        /// <summary>True when the root is a reduction (Sum / Prod / Min / Max / Mean).</summary>
        public bool IsReduction => _program.Reduce is not null;

        /// <summary>Number of distinct inputs the kernel reads (embedded arrays are deduplicated by reference; hoisted 0-d parameters count).</summary>
        public int OperandCount => _program.InputTypes.Length;

        /// <summary>The input dtype signature this handle is compiled for, in input order.</summary>
        public NPTypeCode[] InputTypes => (NPTypeCode[])_program.InputTypes.Clone();

        /// <summary>The NumPy result dtype of an evaluation (the elementwise output, or the reduction's result).</summary>
        public NPTypeCode ResultType => _program.ResultType;

        /// <summary>
        ///     Evaluate an embedded-array handle over the arrays the tree references (their current
        ///     contents). Same contract as <c>np.evaluate(expr, out)</c>: <paramref name="out"/> joins
        ///     the broadcast but is never stretched, takes a same_kind cast from
        ///     <see cref="ResultType"/>, and may alias an input.
        /// </summary>
        /// <param name="out">An optional pre-allocated, writeable result.</param>
        /// <returns>The evaluated array (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="InvalidOperationException">The handle is positional — pass its operands; or a hoisted 0-d array was resized in place.</exception>
        public NDArray Evaluate(NDArray @out = null)
        {
            if (IsPositional)
                throw new InvalidOperationException(
                    $"this expression was compiled for {OperandCount} positional operand(s) — pass them: Evaluate(operands, out).");

            var inputs = _inputs;
            if (!_program.ParamsStillValid(inputs))
                throw new InvalidOperationException(
                    "a 0-d array this handle was compiled to read as a scalar parameter is no longer 0-d " +
                    "(resized in place) — compile the expression again.");
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            return inputs[0].TensorEngine.Evaluate(_program, inputs, @out);
        }

        /// <summary>
        ///     Evaluate a positional handle against <paramref name="operands"/> (<c>Input(i)</c> reads
        ///     <c>operands[i]</c>), which must carry exactly the dtype signature the handle was compiled
        ///     for (<see cref="InputTypes"/>); otherwise a <see cref="TypeError"/>. Shapes broadcast per
        ///     call like a ufunc's.
        /// </summary>
        /// <param name="operands">The positional operands, one per <see cref="NDExpr.Input"/> index.</param>
        /// <param name="out">An optional pre-allocated, writeable result.</param>
        /// <returns>The evaluated array (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="operands"/> or one of its entries is null.</exception>
        /// <exception cref="InvalidOperationException">The handle is an embedded-array handle — evaluate it with <see cref="Evaluate(NDArray)"/>.</exception>
        /// <exception cref="TypeError">The operands' dtype signature differs from the compiled one.</exception>
        public NDArray Evaluate(NDArray[] operands, NDArray @out = null)
        {
            if (operands is null) throw new ArgumentNullException(nameof(operands));
            if (!IsPositional)
                throw new InvalidOperationException(
                    "this expression was compiled over embedded arrays — evaluate it with Evaluate(out).");
            foreach (var op in operands)
                if (op is null)
                    throw new ArgumentNullException(nameof(operands), "no operand may be null.");
            if (!_program.Matches(operands))
            {
                var got = new NPTypeCode[operands.Length];
                for (int i = 0; i < operands.Length; i++)
                    got[i] = operands[i].typecode;
                throw new TypeError(
                    $"compiled expression expects operand dtypes {NDExprProgram.SignatureText(_program.InputTypes)} " +
                    $"but was called with {NDExprProgram.SignatureText(got)} — compile a handle per signature " +
                    "(NDExpr.Compile(params NPTypeCode[])) or use np.evaluate, which recompiles per signature.");
            }

            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            return operands[0].TensorEngine.Evaluate(_program, operands, @out);
        }
    }
}
