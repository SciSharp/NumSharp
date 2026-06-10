using System;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Unary operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        // Call-invariant flag array for the unary NpyIter route (Wave 2.2).
        private static readonly NpyIterPerOpFlags[] s_unaryIterFlags =
        {
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        /// <summary>
        /// Execute a unary operation using IL-generated kernels.
        /// Handles type promotion, strided arrays, and kernel dispatch.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="op">Operation to perform</param>
        /// <param name="typeCode">Optional output type (null = same as input or float for trig/sqrt)</param>
        /// <param name="@out">Optional provided output (NumPy ufunc out=).</param>
        /// <param name="where">Optional bool write mask (NumPy ufunc where=).</param>
        /// <returns>Result array with specified or promoted type (or <paramref name="@out"/>)</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray ExecuteUnaryOp(NDArray nd, UnaryOp op, NPTypeCode? typeCode = null,
            NDArray @out = null, NDArray where = null)
        {
            if (nd.size == 0 && @out is null && where is null)
            {
                // For empty arrays, return empty array with correct output dtype
                // typeCode specifies the output type (e.g., Boolean for predicate ops)
                var emptyOutputType = typeCode ?? nd.GetTypeCode;
                return new NDArray(emptyOutputType, nd.Shape.Clean(), false);
            }

            var inputType = nd.GetTypeCode;

            // Determine output type:
            // - If explicit type provided, use it directly
            // - For trig/math functions (Sin, Cos, Exp, Log, Sqrt), use ResolveUnaryReturnType (promotes to float)
            // - For arithmetic functions (Negate, Abs), preserve input type
            NPTypeCode outputType;
            if (typeCode.HasValue)
            {
                outputType = typeCode.Value;
            }
            else if (op == UnaryOp.Negate || op == UnaryOp.Abs || op == UnaryOp.LogicalNot)
            {
                // Arithmetic/logical operations preserve type
                // LogicalNot on boolean returns boolean
                outputType = inputType;
            }
            else
            {
                // Math functions promote to computing type (typically float/double)
                outputType = ResolveUnaryReturnType(nd, (NPTypeCode?)null);
            }

            // ufunc out=/where= path (Wave 2.1): the output dtype above is the
            // final loop dtype; out only constrains the write-back cast.
            // NumPy disables the trivial loop for masked/provided-out execution
            // (ufunc_object.c:2213) — same here, straight to the iterator.
            if (@out is not null || where is not null)
            {
                return ExecuteUnaryUfuncInto(nd, op, inputType, outputType, @out, where);
            }

            // Handle scalar case
            if (nd.Shape.IsScalar)
            {
                return ExecuteScalarUnary(nd, op, outputType);
            }

            // -------- O(1) trivial-loop bypass -----------------------------
            // Same rationale as the binary/comparison routes (NumPy
            // check_for_trivial_loop): a single contiguous operand (C or F, not
            // broadcast) routes straight to the existing DirectIL whole-array unary
            // kernel, skipping NpyIter construction. The result takes the input's
            // layout so the linear read/write align; in/out dtypes may differ
            // (predicate ops -> bool, Abs(complex) -> double). Returns null for
            // strided/broadcast/unsupported -> NpyIter route below.
            {
                var trivial = TryTrivialContiguousUnaryOp(nd, op, inputType, outputType);
                if (trivial is not null) return trivial;
            }

            // -------- Fused strided-SIMD path for 1-D strided inputs -------
            // The fastest route for a non-contiguous 1-D *same-dtype* float/double
            // input: a single custom-IL kernel gathers each SIMD vector directly
            // from the strided source (lane-count scalar loads -> Vector.Create),
            // applies the op, and stores contiguously — one pass, no scratch tile,
            // no per-chunk delegate dispatch (cf. the gather-to-scratch buffered
            // path below). Same-width SIMD only; promoting (sqrt(int32)->double),
            // integer, and predicate cases fall through to the buffered/NpyIter
            // routes. Measured ~4x the buffered path and beats NumPy on strided sqrt.
            {
                var fused = TryStridedSimdUnaryOp(nd, op, inputType, outputType);
                if (fused is not null) return fused;
            }

            // -------- Buffered-SIMD path for 1-D strided inputs ------------
            // A non-contiguous (stepped/reversed/strided) 1-D input can't feed
            // the SIMD inner loop directly — the elementwise NpyIter route falls
            // to a scalar per-element walk. NumPy's answer is buffering: copy a
            // cache-sized chunk of the strided source into a contiguous tile,
            // run the SIMD kernel on the tile, repeat. We do exactly that here,
            // reusing the proven contiguous unary kernel, for ops whose contig
            // kernel is actually SIMD-accelerated (CanUseUnarySimd) — otherwise
            // the extra gather would just add cost over the scalar path. This
            // remains the route for the promoting (in != out) SIMD cases the
            // fused kernel above intentionally skips.
            {
                var buffered = TryBufferedStridedUnaryOp(nd, op, inputType, outputType);
                if (buffered is not null) return buffered;
            }

            // -------- NpyIter Tier 3B fast path (all unary ops) ------------
            // Funnels through the NpyIter inner-loop kernel factory for the
            // same architectural reasons as the binary route: unified driver,
            // coalesce + SIMD dispatch baked in, F-aware order handling.
            // Returns null only when EmitUnaryScalarOperation /
            // EmitUnaryVectorOperation can't lower (op+dtype combos those
            // emitters don't cover) — in which case we drop to the existing
            // direct UnaryKernel path below.
            {
                var routed = TryExecuteUnaryOpViaNpyIter(nd, op, inputType, outputType);
                if (routed is not null) return routed;
            }

            // Determine if array is contiguous
            bool isContiguous = nd.Shape.IsContiguous;

            // Allocate result (always contiguous)
            var result = new NDArray(outputType, nd.Shape.Clean(), false);

            // Get kernel key
            var key = new UnaryKernelKey(inputType, outputType, op, isContiguous);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.GetUnaryKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteUnaryKernel(kernel, nd, result);
            }
            else
            {
                // Fallback - should not happen for implemented operations
                throw new NotSupportedException(
                    $"IL kernel not available for {op}({inputType}) -> {outputType}. " +
                    "Please report this as a bug.");
            }

            // NumPy-aligned layout preservation: unary ops preserve F-contig.
            // The kernel writes in linear C-order; relay out when the input is strictly F-contig.
            if (ShouldProduceFContigOutput(nd, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        ///     Unary arm of the trivial-loop bypass (see the binary
        ///     <c>TryTrivialContiguousBinaryOp</c>). When the single operand is
        ///     contiguous (C or F, not broadcast), routes straight to the existing
        ///     DirectIL whole-array unary kernel (the contiguous <see cref="UnaryKernelKey"/>
        ///     variant walks input/output linearly), skipping NpyIter construction.
        ///
        ///     For an F-contig input we force the contiguous kernel AND allocate an
        ///     F result: both buffers are then walked in the same physical-linear =
        ///     F-logical order, so element k of input maps to element k of output.
        ///     This matches — and pre-empts — the post-kernel
        ///     <see cref="ShouldProduceFContigOutput(NDArray, Shape)"/> copy('F').
        ///     In/out dtypes may differ (predicate ops -> bool, Abs(complex) ->
        ///     double); the kernel emits the conversion. Returns null (→ NpyIter)
        ///     for strided/broadcast inputs or unsupported emit.
        /// </summary>
        private unsafe NDArray? TryTrivialContiguousUnaryOp(
            NDArray nd, UnaryOp op, NPTypeCode inputType, NPTypeCode outputType)
        {
            var s = nd.Shape;
            if (s.IsBroadcasted)
                return null;

            bool isC = s.IsContiguous;
            bool isF = !isC && s.IsFContiguous;
            if (!isC && !isF)
                return null;   // strided/transposed → NpyIter

            // Contiguous kernel variant (4th arg = isContiguous): linear input[i] -> output[i].
            var key = new UnaryKernelKey(inputType, outputType, op, true);
            UnaryKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetUnaryKernel(key);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            if (kernel == null)
                return null;

            // Reuse the input's canonical shape (offset 0, owns its buffer, right order)
            // rather than cloning dims + recomputing strides/flags. isF implies a strictly
            // column-major input (ndim > 1).
            Shape resultShape = CanonicalResultShape(s, isF);
            var result = new NDArray(outputType, resultShape, false);
            if (result.size == 0)
                return result;

            ExecuteUnaryKernel(kernel, nd, result);
            return result;
        }

        /// <summary>
        ///     Fused strided-SIMD execution for a 1-D non-contiguous (strided / reversed /
        ///     offset) unary input — the fastest unary-over-strided route. A single
        ///     custom-IL <see cref="StridedUnaryKernel"/> assembles each SIMD vector
        ///     directly from <c>lanes</c> strided scalar loads (Vector.Create), applies the
        ///     op, and stores contiguously — no scratch tile and no per-chunk delegate
        ///     dispatch (contrast <see cref="TryBufferedStridedUnaryOp"/>, which gathers into
        ///     a stack tile then calls the contiguous kernel per chunk).
        ///
        ///     Gated to <b>same-width</b> SIMD (input == output dtype) on Single / Double:
        ///     the kernel builds <c>Vector{W}&lt;T&gt;</c> in place, so the op must have a
        ///     vector body and there is no lane-width conversion. The float/double restriction
        ///     captures the measured win (expensive vector ops like Sqrt) and keeps the
        ///     per-lane <c>Vector.Create</c> cheap (4–8 lanes). Returns null (→ buffered /
        ///     NpyIter routes) for non-1-D, contiguous, broadcast, promoting (in != out),
        ///     integer, or non-SIMD-capable inputs — including the promoting cases (e.g.
        ///     <c>sqrt(int32) → double</c>) that <see cref="TryBufferedStridedUnaryOp"/> still
        ///     handles. The result is a fresh contiguous 1-D array — the same shape/layout
        ///     NumPy returns for a unary over a strided 1-D view.
        /// </summary>
        private unsafe NDArray? TryStridedSimdUnaryOp(
            NDArray nd, UnaryOp op, NPTypeCode inputType, NPTypeCode outputType)
        {
            var s = nd.Shape;
            // Only genuinely strided 1-D. (Contiguous — including a stride-1 offset slice —
            // reports IsContiguous and is handled by the trivial bypass.)
            if (s.NDim != 1 || s.IsContiguous || s.IsBroadcasted)
                return null;

            // Same-width SIMD only: the fused kernel assembles Vector<T> in place, so the
            // input and output dtype must match. Restrict to float/double — the proven win
            // and where Vector.Create over a few lanes is cheap. Promoting/integer cases
            // fall through to the buffered path / NpyIter.
            if (inputType != outputType)
                return null;
            if (inputType != NPTypeCode.Single && inputType != NPTypeCode.Double)
                return null;

            // Reuse the contiguous SIMD key so the (op, in, out) triple is shared; the op
            // must have a Vector body (CanUseUnarySimd) or the fused gather has no kernel.
            var key = new UnaryKernelKey(inputType, outputType, op, IsContiguous: true);
            if (!DirectILKernelGenerator.CanUseUnarySimd(key))
                return null;

            StridedUnaryKernel kernel;
            try { kernel = DirectILKernelGenerator.GetStridedUnaryKernel(key); }
            catch (NotSupportedException) { return null; }
            if (kernel == null)
                return null;

            long n = s.size;
            var result = new NDArray(outputType, new Shape(n), false);
            if (n == 0)
                return result;

            // In-memory element size via NPTypeCode.SizeOf() — NOT dtypesize (Marshal-based).
            // strides[0] is an element stride; the kernel works in BYTE strides.
            int inElem = inputType.SizeOf();
            long inByteStride = s.strides[0] * (long)inElem;
            byte* inBase = (byte*)nd.Address + s.offset * (long)inElem;

            kernel((void*)inBase, inByteStride, (void*)result.Address, n);
            return result;
        }

        /// <summary>
        ///     Buffered-SIMD execution for a 1-D non-contiguous (strided / reversed)
        ///     unary input. Gathers the strided source into a contiguous stack tile
        ///     and runs the SIMD contiguous unary kernel on the tile, chunk by chunk —
        ///     NumPy's buffering strategy. Returns null (→ NpyIter scalar route) when
        ///     the input isn't 1-D strided, is broadcast, or the op's contiguous kernel
        ///     isn't SIMD-accelerated (in which case the gather would be pure overhead
        ///     over the scalar walk). The result is a fresh contiguous 1-D array — the
        ///     same shape/layout NumPy returns for a unary over a strided 1-D view.
        ///
        ///     Now reached only for the promoting (in != out) SIMD-capable cases — e.g.
        ///     <c>sqrt(int32) → double</c>, <c>Abs(complex) → double</c> — since the
        ///     same-width float/double cases are taken by the faster fused
        ///     <see cref="TryStridedSimdUnaryOp"/> above.
        /// </summary>
        private unsafe NDArray? TryBufferedStridedUnaryOp(
            NDArray nd, UnaryOp op, NPTypeCode inputType, NPTypeCode outputType)
        {
            var s = nd.Shape;
            // Only genuinely strided 1-D. (Contiguous — including an offset slice with
            // stride 1 — reports IsContiguous and is handled by the trivial bypass.)
            if (s.NDim != 1 || s.IsContiguous || s.IsBroadcasted)
                return null;

            // Gate to ops whose contiguous kernel vectorizes: the point is to convert a
            // scalar strided walk into a SIMD contiguous one. For scalar-only ops
            // (exp/sin/... — no Vector intrinsic) the gather would only add cost.
            var key = new UnaryKernelKey(inputType, outputType, op, IsContiguous: true);
            if (!DirectILKernelGenerator.CanUseUnarySimd(key))
                return null;

            UnaryKernel kernel;
            try { kernel = DirectILKernelGenerator.GetUnaryKernel(key); }
            catch (NotSupportedException) { return null; }
            if (kernel == null)
                return null;

            long n = s.size;
            var result = new NDArray(outputType, new Shape(n), false);
            if (n == 0)
                return result;

            // In-memory element sizes via NPTypeCode.SizeOf() — NOT dtypesize, which is
            // Marshal-based and reports 4 for bool. strides[0] is an element stride.
            int inElem = inputType.SizeOf();
            int outElem = outputType.SizeOf();
            long inByteStride = s.strides[0] * inElem;
            byte* inBase = (byte*)nd.Address + s.offset * (long)inElem;
            byte* outBase = (byte*)result.Address;

            // Cache-resident tile reused across chunks (16B covers Complex/Decimal).
            const int TILE = 2048;
            byte* scratch = stackalloc byte[TILE * 16];
            long* dShape = stackalloc long[1];
            long* dStrides = stackalloc long[1];
            dStrides[0] = 1;

            for (long pos = 0; pos < n; pos += TILE)
            {
                int chunk = (int)Math.Min((long)TILE, n - pos);
                GatherStrided(inBase + pos * inByteStride, inByteStride, scratch, chunk, inElem);
                dShape[0] = chunk;
                // Contiguous kernel: ignores strides/shape, walks `chunk` linearly with SIMD.
                kernel((void*)scratch, (void*)(outBase + pos * outElem), dStrides, dShape, 1, chunk);
            }

            return result;
        }

        /// <summary>
        ///     Copy <paramref name="count"/> elements of <paramref name="elemSize"/> bytes
        ///     from a strided source (byte stride <paramref name="byteStride"/>, may be
        ///     negative for reversed views) into a contiguous destination. Typed per common
        ///     element width so the JIT emits a tight incremental load/store loop.
        /// </summary>
        private static unsafe void GatherStrided(byte* src, long byteStride, byte* dst, int count, int elemSize)
        {
            switch (elemSize)
            {
                case 1:
                    for (int k = 0; k < count; k++) { dst[k] = *src; src += byteStride; }
                    break;
                case 2:
                    { var d = (short*)dst; for (int k = 0; k < count; k++) { d[k] = *(short*)src; src += byteStride; } }
                    break;
                case 4:
                    { var d = (int*)dst; for (int k = 0; k < count; k++) { d[k] = *(int*)src; src += byteStride; } }
                    break;
                case 8:
                    { var d = (long*)dst; for (int k = 0; k < count; k++) { d[k] = *(long*)src; src += byteStride; } }
                    break;
                case 16:
                    { var d = (decimal*)dst; for (int k = 0; k < count; k++) { d[k] = *(decimal*)src; src += byteStride; } }
                    break;
                default:
                    for (int k = 0; k < count; k++) { Buffer.MemoryCopy(src, dst + (long)k * elemSize, elemSize, elemSize); src += byteStride; }
                    break;
            }
        }

        /// <summary>
        ///     Mirror of <c>DirectILKernelGenerator.IsPredicateOp</c> (private to
        ///     that partial) — the routing layer needs the same answer.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsUnaryPredicateOp(UnaryOp op)
            => op == UnaryOp.IsFinite || op == UnaryOp.IsNan || op == UnaryOp.IsInf;

        /// <summary>
        ///     <see cref="System.Numerics.Complex.Abs"/> — resolved once and
        ///     cached. Routes the Complex-magnitude special case in
        ///     <see cref="TryExecuteUnaryOpViaNpyIter"/> without depending on
        ///     <c>DirectILKernelGenerator.CachedMethods</c>, which is private to
        ///     the kernel partial.
        /// </summary>
        private static readonly MethodInfo s_complexAbs =
            typeof(System.Numerics.Complex).GetMethod(
                "Abs", BindingFlags.Public | BindingFlags.Static,
                new[] { typeof(System.Numerics.Complex) })
            ?? throw new MissingMethodException(typeof(System.Numerics.Complex).FullName, "Abs");

        /// <summary>
        ///     Try to execute a unary op via NpyIter Tier 3B. Returns the
        ///     result on success or null if the route is unsupported (size
        ///     overflow vs int.MaxValue, NotSupportedException from emit).
        ///
        ///     Layout: F-allocates output when input is strict-F (matches
        ///     the post-kernel <c>result.copy('F')</c> the direct path
        ///     applies via <see cref="ShouldProduceFContigOutput(NDArray, Shape)"/>);
        ///     picks NPY_FORTRANORDER in that case, NPY_CORDER otherwise.
        ///
        ///     Same-dtype path: scalar body =
        ///     <see cref="DirectILKernelGenerator.EmitUnaryScalarOperation"/>;
        ///     vector body when SIMD is viable.
        ///
        ///     Mixed-dtype path (Abs(complex)→double, IsNan(float)→bool,
        ///     Cast(int)→float, etc.): scalar body emits
        ///     EmitUnaryScalarOperation on the INPUT type (which produces
        ///     the output type for predicate ops like IsNan) or
        ///     EmitUnaryScalarOperation + EmitConvertTo (for math ops that
        ///     compute in input type then convert). Vector body is null
        ///     because cross-type unary SIMD isn't templated in Tier 3B.
        /// </summary>
        private unsafe NDArray? TryExecuteUnaryOpViaNpyIter(NDArray nd, UnaryOp op, NPTypeCode inputType, NPTypeCode outputType)
        {
            var cleanShape = nd.Shape.Clean();
            if (cleanShape.size < 0) return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue) return null;

            // Mirror the direct path's F-preservation: input strict-F →
            // allocate F-contig output AND iterate in F-order. Otherwise
            // C-contig output + C-order; the post-kernel copy step at the
            // end of this function handles the looser-F case.
            bool inputStrictF = nd.Shape.IsFContiguous && !nd.Shape.IsContiguous
                                 && cleanShape.NDim > 1 && cleanShape.size > 1
                                 && !nd.Shape.IsScalar;
            Shape resultShape = inputStrictF
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;
            var result = new NDArray(outputType, resultShape, false);

            var order = inputStrictF
                ? NPY_ORDER.NPY_FORTRANORDER
                : NPY_ORDER.NPY_CORDER;

            // SIMD viability: same-dtype only (existing rule from the
            // direct path's CanUseUnarySimd) + op-supported check.
            var key = new UnaryKernelKey(inputType, outputType, op, IsContiguous: true);
            bool simdViable = DirectILKernelGenerator.CanUseUnarySimd(key);

            // Promoting-unary buffered-cast route (NumPy ufunc parity): for
            // sqrt(i32)->f64-style ops, cast the input to the OUTPUT dtype in
            // buffer windows via the SIMD IL cast kernels and run the
            // same-dtype vector body over the buffer — NumPy's d->d loop with
            // a buffering iterator, replacing the per-element convert scalar
            // body. Predicate ops (IsNan/IsInf/...) operate on the INPUT type
            // and Complex Abs has its own scalar special case — both excluded.
            bool bufferedPromoting = inputType != outputType
                && !IsUnaryPredicateOp(op)
                && !(op == UnaryOp.Abs && inputType == NPTypeCode.Complex)
                && DirectILKernelGenerator.CanUseUnarySimd(
                       new UnaryKernelKey(outputType, outputType, op, IsContiguous: true))
                && DirectILKernelGenerator.TryGetCastKernel(inputType, outputType) != null;

            // Scalar body — mirrors the direct path's per-element sequence
            // in DirectILKernelGenerator.Unary's emit loops:
            //   • Predicate ops (IsNan / IsInf / IsFinite) operate on the
            //     INPUT type and the emitter itself produces bool — no
            //     convert before or after.
            //   • Complex Abs is a magnitude reduction: call ComplexAbs
            //     intrinsic, then convert from double to outputType if it
            //     differs (matches the direct path's special-case block).
            //   • Everything else: convert input → output type, then run
            //     the op on output type. This is required for promoting
            //     ops like Sqrt(int32) → double where EmitMathCall expects
            //     the value to already be in the floating-point domain.
            NPTypeCode capIn = inputType, capOut = outputType;
            UnaryOp capOp = op;
            Action<ILGenerator> scalarBody;
            Action<ILGenerator>? vectorBody;
            string cacheKey;
            if (bufferedPromoting)
            {
                // Kernel only ever sees the output dtype — the iterator's
                // buffered fill already performed the input cast.
                scalarBody = il => DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capOut);
                vectorBody = il => DirectILKernelGenerator.EmitUnaryVectorOperation(il, capOp, capOut);
                cacheKey = $"npy_unop_{op}_{outputType}_{outputType}";
            }
            else
            {
                scalarBody = il =>
                {
                    if (IsUnaryPredicateOp(capOp))
                    {
                        DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capIn);
                    }
                    else if (capOp == UnaryOp.Abs && capIn == NPTypeCode.Complex)
                    {
                        il.EmitCall(OpCodes.Call, s_complexAbs, null);
                        if (capOut != NPTypeCode.Double)
                            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, capOut);
                    }
                    else
                    {
                        if (capIn != capOut)
                            DirectILKernelGenerator.EmitConvertTo(il, capIn, capOut);
                        DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capOut);
                    }
                };
                vectorBody = simdViable
                    ? il => DirectILKernelGenerator.EmitUnaryVectorOperation(il, capOp, capIn)
                    : null;
                cacheKey = $"npy_unop_{op}_{inputType}_{outputType}";
            }

            try
            {
                // COPY_IF_OVERLAP + OVERLAP_ASSUME_ELEMENTWISE per NumPy's ufunc
                // iterator flags (ufunc_object.c:1070); cheap extent check here
                // since the result is freshly allocated.
                var globalFlags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
                NPTypeCode[]? opDtypes = null;
                var casting = NPY_CASTING.NPY_SAFE_CASTING;
                if (bufferedPromoting)
                {
                    // NumPy ufunc config (loop dtypes already resolved → unsafe).
                    globalFlags |= NpyIterGlobalFlags.BUFFERED
                                 | NpyIterGlobalFlags.GROWINNER
                                 | NpyIterGlobalFlags.DELAY_BUFALLOC;
                    opDtypes = new[] { outputType, outputType };
                    casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                }

                using var iter = NpyIterRef.MultiNew(
                    2, new[] { nd, result },
                    globalFlags,
                    order, casting,
                    s_unaryIterFlags,
                    opDtypes);

                if (bufferedPromoting)
                    iter.ExecuteElementWiseUnary(outputType, outputType, scalarBody, vectorBody, cacheKey);
                else
                    iter.ExecuteElementWiseUnary(inputType, outputType, scalarBody, vectorBody, cacheKey);
            }
            catch (NotSupportedException)
            {
                return null;
            }

            // Post-kernel "looser-F" copy step (mirrors the direct path's
            // tail branch): triggers when result is currently C-contig but
            // input is strict-F so the NumPy rule says output should be F.
            // strict-F-input cases skipped this by allocating F up front.
            if (!inputStrictF && ShouldProduceFContigOutput(nd, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        /// Execute scalar unary operation using IL-generated delegate.
        /// </summary>
        private NDArray ExecuteScalarUnary(NDArray nd, UnaryOp op, NPTypeCode outputType)
        {
            var inputType = nd.GetTypeCode;
            var key = new UnaryScalarKernelKey(inputType, outputType, op);
            var func = DirectILKernelGenerator.GetUnaryScalarDelegate(key);

            // Dispatch based on input type to avoid boxing
            return inputType switch
            {
                NPTypeCode.Boolean => InvokeUnaryScalar(func, nd.GetBoolean(Array.Empty<long>()), outputType),
                NPTypeCode.Byte => InvokeUnaryScalar(func, nd.GetByte(Array.Empty<long>()), outputType),
                NPTypeCode.SByte => InvokeUnaryScalar(func, nd.GetSByte(Array.Empty<long>()), outputType),
                NPTypeCode.Int16 => InvokeUnaryScalar(func, nd.GetInt16(Array.Empty<long>()), outputType),
                NPTypeCode.UInt16 => InvokeUnaryScalar(func, nd.GetUInt16(Array.Empty<long>()), outputType),
                NPTypeCode.Int32 => InvokeUnaryScalar(func, nd.GetInt32(Array.Empty<long>()), outputType),
                NPTypeCode.UInt32 => InvokeUnaryScalar(func, nd.GetUInt32(Array.Empty<long>()), outputType),
                NPTypeCode.Int64 => InvokeUnaryScalar(func, nd.GetInt64(Array.Empty<long>()), outputType),
                NPTypeCode.UInt64 => InvokeUnaryScalar(func, nd.GetUInt64(Array.Empty<long>()), outputType),
                NPTypeCode.Char => InvokeUnaryScalar(func, nd.GetChar(Array.Empty<long>()), outputType),
                NPTypeCode.Half => InvokeUnaryScalar(func, nd.GetHalf(Array.Empty<long>()), outputType),
                NPTypeCode.Single => InvokeUnaryScalar(func, nd.GetSingle(Array.Empty<long>()), outputType),
                NPTypeCode.Double => InvokeUnaryScalar(func, nd.GetDouble(Array.Empty<long>()), outputType),
                NPTypeCode.Decimal => InvokeUnaryScalar(func, nd.GetDecimal(Array.Empty<long>()), outputType),
                NPTypeCode.Complex => InvokeUnaryScalar(func, nd.GetComplex(Array.Empty<long>()), outputType),
                _ => throw new NotSupportedException($"Input type {inputType} not supported")
            };
        }

        /// <summary>
        /// Invoke a unary scalar delegate and create the result NDArray.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeUnaryScalar<TInput>(Delegate func, TInput input, NPTypeCode outputType)
        {
            // Dispatch based on output type to avoid boxing on result
            return outputType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TInput, bool>)func)(input)),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TInput, byte>)func)(input)),
                NPTypeCode.SByte => NDArray.Scalar(((Func<TInput, sbyte>)func)(input)),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TInput, short>)func)(input)),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TInput, ushort>)func)(input)),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TInput, int>)func)(input)),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TInput, uint>)func)(input)),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TInput, long>)func)(input)),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TInput, ulong>)func)(input)),
                NPTypeCode.Char => NDArray.Scalar(((Func<TInput, char>)func)(input)),
                NPTypeCode.Half => NDArray.Scalar(((Func<TInput, Half>)func)(input)),
                NPTypeCode.Single => NDArray.Scalar(((Func<TInput, float>)func)(input)),
                NPTypeCode.Double => NDArray.Scalar(((Func<TInput, double>)func)(input)),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TInput, decimal>)func)(input)),
                NPTypeCode.Complex => NDArray.Scalar(((Func<TInput, System.Numerics.Complex>)func)(input)),
                _ => throw new NotSupportedException($"Output type {outputType} not supported")
            };
        }

        /// <summary>
        /// Execute the IL-generated unary kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteUnaryKernel(
            UnaryKernel kernel,
            NDArray input, NDArray result)
        {
            int inputElemSize = input.dtypesize;
            var inputShape = input.Shape;

            // Calculate base address accounting for shape offset (for sliced views)
            byte* inputAddr = (byte*)input.Address + inputShape.offset * inputElemSize;

            fixed (long* strides = inputShape.strides)
            fixed (long* shape = result.shape)
            {
                kernel(
                    (void*)inputAddr,
                    (void*)result.Address,
                    strides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }
    }
}
