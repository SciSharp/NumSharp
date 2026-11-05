using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Comparison operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a comparison operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// Result is always NDArray&lt;bool&gt;.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Comparison operation to perform</param>
        /// <returns>Result array with bool type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray<bool> ExecuteComparisonOp(NDArray lhs, NDArray rhs, ComparisonOp op)
        {
            // Two strings of one class compare their text (the guard runs NDStringComparison and returns its result; an
            // empty result compares nothing); any other pair with a string operand has no comparison loop (NumPy's
            // class-named UFuncTypeError).
            if (Strings.NDStringUfuncs.BinaryGuard(UfuncName(op), lhs, rhs) is { } textResult)
                return textResult.AsGeneric<bool>();

            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteComparisonScalarScalar(lhs, rhs, op);
            }

            // -------- NumPy's result layout ----------------------------------
            // A fresh bool result takes the layout NumPy allocates (DefaultEngine.UfuncResultLayout.cs): F for an F pair,
            // a transposed operand's own axis order, C where NumPy's iterator votes C. The loop reads the comparison dtype,
            // so an input of another dtype is a cast (refusing NumPy's trivial loop for an N-D input) — except int64
            // against uint64, which NumPy compares with exact loops of their own (qQ / Qq). The routes below write C.
            // Two C-contiguous (or 0-d) operands are C on both of NumPy's routes, so the common call skips the
            // comparison-dtype lookup the cast flags need.
            if (!((lhs.Shape.IsContiguous || lhs.ndim == 0) && (rhs.Shape.IsContiguous || rhs.ndim == 0)))
            {
                var comparisonType = ComparisonKernelKey.GetComparisonType(lhsType, rhsType);
                bool exactPair = (lhsType == NPTypeCode.Int64 && rhsType == NPTypeCode.UInt64)
                                 || (lhsType == NPTypeCode.UInt64 && rhsType == NPTypeCode.Int64);
                if (TryBinaryNonCLayout(lhs, rhs, !exactPair && lhsType != comparisonType,
                                        !exactPair && rhsType != comparisonType, out var layout))
                    return ExecuteComparisonOpInLayout(lhs, rhs, op, lhsType, rhsType, layout);
            }
            return ExecuteComparisonOpCOrder(lhs, rhs, op, lhsType, rhsType);
        }

        /// <summary>
        ///     A fresh comparison result in a layout NumPy allocates that is not C order (see
        ///     <c>DefaultEngine.UfuncResultLayout.cs</c>).
        /// </summary>
        /// <param name="lhs">The first operand.</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">The first operand's dtype.</param>
        /// <param name="rhsType">The second operand's dtype.</param>
        /// <param name="layout">The layout to produce: dense, offset 0, not C-contiguous.</param>
        /// <returns>A fresh bool array laid out as <paramref name="layout"/>.</returns>
        /// <exception cref="NotSupportedException">No kernel exists for the dtypes (raised by the C routes).</exception>
        /// <remarks>
        ///     Operands stored exactly as the layout stores the result (an F pair, a dense transposed pair, either beside a
        ///     one-element operand) run the comparison run or the whole-array kernel straight into the layout, of any two
        ///     dtypes, as the C-ordered trivial loop does. Anything else runs the C routes over the operands viewed in the
        ///     layout's memory order and relabels the C result
        ///     (<see cref="ComputeInLayout{TState}(NDArray, NDArray, in Shape, TState, Func{NDArray, NDArray, TState, NDArray})"/>).
        /// </remarks>
        private unsafe NDArray<bool> ExecuteComparisonOpInLayout(NDArray lhs, NDArray rhs, ComparisonOp op,
                                                                 NPTypeCode lhsType, NPTypeCode rhsType, Shape layout)
        {
            int nd = layout.NDim;
            // A one-element operand of no higher rank than the result is read once; a non-C layout has at least two
            // elements, so at most one operand can be such a scalar.
            bool lhsScalar = lhs.Shape.size == 1 && lhs.ndim <= nd;
            bool rhsScalar = rhs.Shape.size == 1 && rhs.ndim <= nd;
            bool linear = rhsScalar ? IsLinearInLayout(lhs, layout)
                        : lhsScalar ? IsLinearInLayout(rhs, layout)
                        : IsLinearInLayout(lhs, layout) && IsLinearInLayout(rhs, layout);
            if (linear)
            {
                var path = rhsScalar ? ExecutionPath.SimdScalarRight
                         : lhsScalar ? ExecutionPath.SimdScalarLeft
                         : ExecutionPath.SimdFull;
                var ran = TryRunTrivialComparison(lhs, rhs, op, lhsType, rhsType, path, lhsScalar, rhsScalar, layout);
                if (ran is not null)
                    return ran;
            }

            var r = ComputeInLayout(lhs, rhs, layout, (Engine: this, op, lhsType, rhsType),
                static (a, b, s) => (NDArray)s.Engine.ExecuteComparisonOpCOrder(a, b, s.op, s.lhsType, s.rhsType));
            return AsBoolResult(r);
        }

        /// <summary>
        ///     Types a relabelled comparison result as <see cref="NDArray{T}"/> of bool: the relabel returns the kernel's
        ///     own <c>NDArray&lt;bool&gt;</c> instance; its copying fallback a plain array, wrapped here without orphaning a
        ///     reference to the buffer.
        /// </summary>
        /// <param name="r">The relabelled result (bool dtype).</param>
        /// <returns><paramref name="r"/> as an <c>NDArray&lt;bool&gt;</c>.</returns>
        /// <exception cref="InvalidOperationException">The result is not bool (a kernel bug).</exception>
        private static NDArray<bool> AsBoolResult(NDArray r)
        {
            if (r is NDArray<bool> typed)
                return typed;
            var wrapped = r.AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison expected a Boolean result.");
            // AsGeneric wraps r's storage in a NEW NDArray<bool>; r's own reference on the buffer is then dropped so the
            // caller's Dispose of the wrapper frees the buffer to the pool.
            if (!ReferenceEquals(wrapped, r))
                r.Dispose();
            return wrapped;
        }

        /// <summary>
        ///     The routes of a fresh comparison result in C order: the trivial contiguous loop, the two-level rows route,
        ///     the NDIter route and the direct kernel. Every one allocates a C-contiguous bool result;
        ///     <see cref="ExecuteComparisonOpInLayout"/> runs them over views in a non-C layout's memory order.
        /// </summary>
        /// <param name="lhs">The first operand (not both 0-d).</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">The first operand's dtype.</param>
        /// <param name="rhsType">The second operand's dtype.</param>
        /// <returns>A fresh C-contiguous bool result of the broadcast shape.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's text).</exception>
        /// <exception cref="NotSupportedException">No kernel exists for the dtypes.</exception>
        private unsafe NDArray<bool> ExecuteComparisonOpCOrder(NDArray lhs, NDArray rhs, ComparisonOp op,
                                                               NPTypeCode lhsType, NPTypeCode rhsType)
        {
            // -------- O(1) trivial-loop bypass -----------------------------
            // Same rationale as the binary route (NumPy check_for_trivial_loop):
            // equal-shape C-contiguous and array-vs-scalar cases route straight to
            // the DirectIL SimdFull / SimdScalar comparison kernel, skipping NDIter
            // construction. Unlike the binary bypass this allows mixed dtypes —
            // comparison promotes per element inside the kernel with no cast temp,
            // and the result is always bool. Returns null for broadcast/strided/
            // unsupported → the routes below. (An F pair is NumPy's F result:
            // ExecuteComparisonOpInLayout.)
            {
                var trivial = TryTrivialContiguousComparisonOp(lhs, rhs, op, lhsType, rhsType);
                if (trivial is not null) return trivial;
            }

            // -------- Two-level iterations, without an iterator ---------------
            // A matrix against a row or a column, an outer product, a stepped or
            // reversed view: R runs of N elements whose operands are a fixed
            // stride apart. DefaultEngine.CompareOp.Rows.cs finds that structure
            // in the strides and runs one vector-kernel call per run; the
            // iterator below costs 0.45 us to build and 15 ns per chunk it hands
            // out. Declines (null) for deeper iterations and narrow rows, which
            // the iterator route serves.
            {
                var rows = TryRowsComparisonOp(lhs, rhs, op, lhsType, rhsType);
                if (rows is not null) return rows;
            }

            // -------- NDIter Tier 3B fast path (all comparison ops) -----------
            // Mirrors the binary-op routing pattern. The iterator coalesces
            // dimensions, normalizes negative strides, and resolves stride=0
            // broadcast operands into per-outer-iter pointer advance — turning
            // the (M,N) op (M,1) broadcast from a per-element gather into a
            // contig inner-loop kernel of size N, repeated M times. Closes
            // 14-30× regressions on strided/broadcast comparison variants vs
            // the direct path. Contig same-dtype is at parity with the direct
            // path because the same SIMD body lives inside the inner kernel.
            {
                var routed = TryExecuteComparisonOpViaNDIter(lhs, rhs, op, lhsType, rhsType);
                if (routed is not null) return routed;
            }

            // Broadcast shapes
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var resultShape = leftShape.Clean();

            // Allocate result (always bool)
            var result = NewArray<bool>(resultShape, true);

            // Empty broadcast result: the kernels assume at least one element and walk a stride-0 axis as if it were
            // populated, so a zero-length sibling axis would read out of bounds (the binary route's same guard).
            if (result.size == 0)
                return result;

            // Classify execution path using strides
            ExecutionPath path;
            fixed (long* lhsStrides = leftShape.strides)
            fixed (long* rhsStrides = rightShape.strides)
            fixed (long* shape = resultShape.dimensions)
            {
                path = ClassifyPath(lhsStrides, rhsStrides, shape, resultShape.NDim, NPTypeCode.Boolean);
            }

            // Get kernel key
            var key = new ComparisonKernelKey(lhsType, rhsType, op, path);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.GetComparisonKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteComparisonKernel(kernel, lhs, rhs, result, leftShape, rightShape);
            }
            else
            {
                // Fallback - should not happen; the unwritten result goes back to the pool.
                result.Dispose();
                throw new NotSupportedException(
                    $"IL kernel not available for comparison {lhsType} {op} {rhsType}. " +
                    "Please report this as a bug.");
            }

            // The kernel writes C order. A non-C layout NumPy allocates reaches here only as operands viewed in that
            // layout's memory order (ExecuteComparisonOpInLayout), whose C result is the layout's buffer.
            return result;
        }

        /// <summary>
        ///     Comparison arm of the trivial-loop bypass (see the binary
        ///     <c>TryTrivialContiguousBinaryOp</c>). Handles two trivially
        ///     iterable shapes — equal-shape C-contiguous (one linear SimdFull walk)
        ///     and array-vs-scalar (scalar read once, a C-contiguous array walked
        ///     linearly via SimdScalarLeft/Right) — and routes to the existing
        ///     DirectIL comparison kernel, skipping NDIter construction. Dtypes may
        ///     differ (the kernel promotes per element; result is always bool). The
        ///     bool result is C-ordered, so the linear write matches the linear read;
        ///     an F or transposed pair is NumPy's F / K result and takes the same run
        ///     through <see cref="ExecuteComparisonOpInLayout"/>. Returns null
        ///     (→ the routes below) for broadcast, non-C, strided, or unsupported
        ///     emit.
        /// </summary>
        /// <param name="lhs">The first operand.</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">The first operand's dtype.</param>
        /// <param name="rhsType">The second operand's dtype.</param>
        /// <returns>A fresh C-contiguous bool result, or null when the call is not trivially iterable in C order.</returns>
        private unsafe NDArray<bool>? TryTrivialContiguousComparisonOp(
            NDArray lhs, NDArray rhs, ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            var ls = lhs.Shape;
            var rs = rhs.Shape;

            bool lhsScalarLike = ls.IsScalar || ls.size == 1;
            bool rhsScalarLike = rs.IsScalar || rs.size == 1;

            ExecutionPath path;
            Shape arrShape;   // operand whose shape+layout the result follows
            if (lhsScalarLike ^ rhsScalarLike)
            {
                // Scalar-broadcast: the array operand drives the result — only when the one-element operand adds no
                // dimension (a (1,1) operand against a (7,) array is a (1,7) result in NumPy; plan api-parameter-parity
                // X19). Otherwise the broadcasting route below computes the shape.
                if (rhsScalarLike ? rs.NDim > ls.NDim : ls.NDim > rs.NDim)
                    return null;
                var array = rhsScalarLike ? lhs : rhs;
                arrShape = array.Shape;
                if (arrShape.IsBroadcasted || !arrShape.IsContiguous)
                    return null;   // strided/transposed array operand → NDIter
                path = rhsScalarLike ? ExecutionPath.SimdScalarRight : ExecutionPath.SimdScalarLeft;
            }
            else if (lhsScalarLike && rhsScalarLike)
            {
                // Two one-element operands, of any two ranks ((1,) against a 0-d array, (1,1) against (1,)): one
                // compare, and the result has the dims of the higher-rank operand (all 1), as NumPy's broadcast gives.
                // Operands of DIFFERENT rank used to fall through to the iterator route, which built an iterator for
                // one element (1.0 us, against 0.47 us for the same pair of equal rank). A one-element array is
                // contiguous whatever its strides, so the kernel reads each operand at its first element.
                path = ExecutionPath.SimdFull;
                arrShape = ls.NDim >= rs.NDim ? ls : rs;
            }
            else
            {
                // Equal-shape (both arrays, or both size-1). Same dims, both C-contiguous.
                if (!ls.Equals(rs))
                    return null;
                if (ls.IsBroadcasted || rs.IsBroadcasted)
                    return null;
                if (!ls.IsContiguous || !rs.IsContiguous)
                    return null;
                path = ExecutionPath.SimdFull;
                arrShape = ls;
            }

            return TryRunTrivialComparison(lhs, rhs, op, lhsType, rhsType, path, lhsScalarLike, rhsScalarLike,
                                           CanonicalResultShape(arrShape, false));
        }

        /// <summary>
        ///     Runs a trivially iterable comparison — every array operand stored exactly as <paramref name="resultShape"/>
        ///     stores the result, at most one operand read once — straight into a fresh result of that shape: through the
        ///     comparison run when one exists for the dtypes, else the whole-array comparison kernel of
        ///     <paramref name="path"/>. Shared by the C-ordered trivial loop and the linear walk of a non-C layout.
        /// </summary>
        /// <param name="lhs">The first operand.</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">The first operand's dtype.</param>
        /// <param name="rhsType">The second operand's dtype.</param>
        /// <param name="path">SimdFull for two arrays (or two one-element operands), SimdScalarRight / SimdScalarLeft
        ///     when the second / first operand is read once.</param>
        /// <param name="lhsScalarLike">Whether the first operand is read once (one element).</param>
        /// <param name="rhsScalarLike">Whether the second operand is read once.</param>
        /// <param name="resultShape">The result's dense, offset-0 shape.</param>
        /// <returns>The fresh bool result, or null (nothing allocated) when neither the run nor the kernel exists.</returns>
        private unsafe NDArray<bool>? TryRunTrivialComparison(NDArray lhs, NDArray rhs, ComparisonOp op,
                                                              NPTypeCode lhsType, NPTypeCode rhsType, ExecutionPath path,
                                                              bool lhsScalarLike, bool rhsScalarLike, Shape resultShape)
        {
            var ls = lhs.Shape;
            var rs = rhs.Shape;

            // The comparison run (Backends/Kernels/ComparisonRun.cs) executes the pair, each operand read as one
            // linear run of its memory (or as one value):
            //   - operands of one dtype: the same vector kernel the key below names, started on a vector boundary
            //     (a buffer that begins in the middle of a 32-byte vector makes every second load cross a cache
            //     line: 100,000 float64 against a value, 12.5 us instead of 9.3);
            //   - operands of two dtypes: the operand that already has the comparison dtype is read in place, the
            //     other is converted 8 KB at a time with the SIMD cast kernels, and an array against ONE value of
            //     another dtype is compared in the array's own dtype. The kernel of the mixed key below converts
            //     both operands element by element instead (a float32 array against a float64 value: 70 us for
            //     100,000 elements where NumPy takes 7).
            // It is null only where runtime IL generation is unavailable or the run is switched off by its test hook
            // (ComparisonRun.Disabled); the kernel lookup below then decides.
            if (ComparisonRun.Get(op, lhsType, rhsType) is { } run)
            {
                var ran = NewArray<bool>(resultShape, false);
                if (ran.size != 0)
                {
                    int lhsItem = lhs.dtypesize, rhsItem = rhs.dtypesize;
                    run.Run(
                        (byte*)lhs.Address + ls.offset * lhsItem, lhsScalarLike ? 0 : lhsItem,
                        (byte*)rhs.Address + rs.offset * rhsItem, rhsScalarLike ? 0 : rhsItem,
                        (byte*)ran.Address, 1, ran.size);
                }
                return ran;
            }

            var key = new ComparisonKernelKey(lhsType, rhsType, op, path);
            ComparisonKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetComparisonKernel(key);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            if (kernel == null)
                return null;

            // The SimdFull / SimdScalar comparison kernels write every output byte
            // (4×-unrolled SIMD + remainder + scalar tail span [0, size)), so the
            // zero-fill is dead work for the bypass — allocate without it.
            var result = NewArray<bool>(resultShape, false);
            if (result.size == 0)
                return result;

            ExecuteComparisonKernel(kernel, lhs, rhs, result, lhs.Shape, rhs.Shape);
            return result;
        }

        /// <summary>
        /// Execute scalar × scalar comparison using IL-generated delegate.
        /// </summary>
        private NDArray<bool> ExecuteComparisonScalarScalar(NDArray lhs, NDArray rhs, ComparisonOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new ComparisonScalarKernelKey(lhsType, rhsType, op);
            var func = DirectILKernelGenerator.GetComparisonScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeComparisonScalarLhs(func, lhs.GetBoolean(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.SByte => InvokeComparisonScalarLhs(func, lhs.GetSByte(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Byte => InvokeComparisonScalarLhs(func, lhs.GetByte(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int16 => InvokeComparisonScalarLhs(func, lhs.GetInt16(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt16 => InvokeComparisonScalarLhs(func, lhs.GetUInt16(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int32 => InvokeComparisonScalarLhs(func, lhs.GetInt32(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt32 => InvokeComparisonScalarLhs(func, lhs.GetUInt32(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int64 => InvokeComparisonScalarLhs(func, lhs.GetInt64(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt64 => InvokeComparisonScalarLhs(func, lhs.GetUInt64(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Char => InvokeComparisonScalarLhs(func, lhs.GetChar(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Half => InvokeComparisonScalarLhs(func, lhs.GetHalf(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Single => InvokeComparisonScalarLhs(func, lhs.GetSingle(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Double => InvokeComparisonScalarLhs(func, lhs.GetDouble(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Decimal => InvokeComparisonScalarLhs(func, lhs.GetDecimal(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Complex => InvokeComparisonScalarLhs(func, lhs.GetComplex(Array.Empty<long>()), rhs, rhsType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue comparison scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private NDArray<bool> InvokeComparisonScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, NDArray rhs, NPTypeCode rhsType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => NewScalar(((Func<TLhs, bool, bool>)func)(lhsVal, rhs.GetBoolean(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.SByte => NewScalar(((Func<TLhs, sbyte, bool>)func)(lhsVal, rhs.GetSByte(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Byte => NewScalar(((Func<TLhs, byte, bool>)func)(lhsVal, rhs.GetByte(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Int16 => NewScalar(((Func<TLhs, short, bool>)func)(lhsVal, rhs.GetInt16(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.UInt16 => NewScalar(((Func<TLhs, ushort, bool>)func)(lhsVal, rhs.GetUInt16(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Int32 => NewScalar(((Func<TLhs, int, bool>)func)(lhsVal, rhs.GetInt32(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.UInt32 => NewScalar(((Func<TLhs, uint, bool>)func)(lhsVal, rhs.GetUInt32(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Int64 => NewScalar(((Func<TLhs, long, bool>)func)(lhsVal, rhs.GetInt64(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.UInt64 => NewScalar(((Func<TLhs, ulong, bool>)func)(lhsVal, rhs.GetUInt64(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Char => NewScalar(((Func<TLhs, char, bool>)func)(lhsVal, rhs.GetChar(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Half => NewScalar(((Func<TLhs, Half, bool>)func)(lhsVal, rhs.GetHalf(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Single => NewScalar(((Func<TLhs, float, bool>)func)(lhsVal, rhs.GetSingle(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Double => NewScalar(((Func<TLhs, double, bool>)func)(lhsVal, rhs.GetDouble(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Decimal => NewScalar(((Func<TLhs, decimal, bool>)func)(lhsVal, rhs.GetDecimal(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                NPTypeCode.Complex => NewScalar(((Func<TLhs, System.Numerics.Complex, bool>)func)(lhsVal, rhs.GetComplex(Array.Empty<long>()))).AsGeneric<bool>() ?? throw new InvalidOperationException("Comparison scalar dispatch expected a Boolean result."),
                _ => throw new NotSupportedException($"RHS type {rhsType} not supported")
            };
        }

        /// <summary>
        /// Execute the IL-generated comparison kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ExecuteComparisonKernel(
            ComparisonKernel kernel,
            NDArray lhs, NDArray rhs, NDArray<bool> result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (long* lhsStrides = lhsShape.strides)
            fixed (long* rhsStrides = rhsShape.strides)
            fixed (long* shape = result.Shape.dimensions)
            {
                kernel(
                    (void*)lhsAddr,
                    (void*)rhsAddr,
                    (bool*)result.Address,
                    lhsStrides,
                    rhsStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }

        #region Public API - Comparison Operations (TensorEngine overrides)

        // ---- ONE NumPy-shaped override per comparison ufunc ----------------
        // Merged bare + out=/where= forms (no duplicate overloads). The bare
        // path (out == null && where == null) keeps the existing trivial/SIMD
        // ladder via ExecuteComparisonOp and returns an NDArray<bool> instance
        // (TensorEngine contract — the C# operators cast it back for free).
        // The out=/where= path routes straight to the Into-path: NumPy's
        // masked execution never takes the trivial loop (ufunc_object.c:2213),
        // and a provided out needs reference identity + write-masking, so the
        // scalar×scalar / trivial-bypass arms of the ladder are skipped by
        // design (0-d EXLOOP works post-Wave-2.1). No F-layout post-step.
        // typeCode is validate-only: comparisons have bool-output loops ONLY —
        // NumPy raises the no-loop TypeError for any non-bool dtype= (probed
        // 2.4.2: equal(a, b, dtype=f64/i32) raises; dtype=bool is a no-op).

        public override NDArray Compare(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "equal", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "equal", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("equal", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.Equal, @out, where);
        }

        public override NDArray NotEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "not_equal", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "not_equal", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("not_equal", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.NotEqual, @out, where);
        }

        public override NDArray Less(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "less", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "less", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("less", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.Less, @out, where);
        }

        public override NDArray LessEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "less_equal", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "less_equal", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("less_equal", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.LessEqual, @out, where);
        }

        public override NDArray Greater(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "greater", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "greater", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("greater", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.Greater, @out, where);
        }

        public override NDArray GreaterEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "greater_equal", lhs, rhs, @out, where);
            ValidateBoolLoopDtype(typeCode, "greater_equal", lhs, rhs);
            // Two strings of one class compare their text (NDStringComparison's kernels, through the guard); any other pair
            // with a string has no loop.
            if (Strings.NDStringUfuncs.BinaryGuard("greater_equal", lhs, rhs, @out, typeCode, where) is { } textResult)
                return textResult;
            return ExecuteComparison(lhs, rhs, ComparisonOp.GreaterEqual, @out, where);
        }

        #endregion

        /// <summary>
        ///     Tier 3B NDIter routing for comparison ops. Output is always bool.
        ///
        ///     Returns the result NDArray&lt;bool&gt; on success, or null when the
        ///     iterator can't handle the shapes (e.g. > int.MaxValue dimension) so
        ///     the caller falls back to the direct path.
        ///
        ///     ─── Why this beats the direct path on broadcast/strided ───
        ///     The direct ComparisonKernel walks the output via coordinate math
        ///     (one ndim-loop per output element). For (M,N) op (M,1) it does M*N
        ///     full coord recomputes and gather loads. NDIter coalesces the
        ///     M*N output into the canonical iteration order, sets the broadcast
        ///     operand's stride to 0, and dispatches a contig inner kernel of
        ///     length N for each of M outer steps — the inner kernel becomes a
        ///     tight scalar/SIMD loop without coord math.
        ///
        ///     ─── Scalar vs vector body ───
        ///     For same-dtype with a SIMD-capable comparison type, the vector
        ///     body would produce the mask + PDEP-store fast path that the
        ///     direct kernel already does. The Tier 3B 4×-unrolled wrapper
        ///     however requires same-type-across-all-operands (CanSimdAllOperands
        ///     in the InnerLoop factory) which doesn't hold here because the
        ///     output is bool. We pass null for the vector body; the iterator
        ///     drops to a per-element scalar loop driven by the emitted body.
        ///     This trades some contig perf for huge wins on strided/broadcast.
        /// </summary>
        private unsafe NDArray<bool>? TryExecuteComparisonOpViaNDIter(
            NDArray lhs, NDArray rhs, ComparisonOp op,
            NPTypeCode lhsType, NPTypeCode rhsType)
        {
            // ─── Routing decision: NDIter wins on broadcast/strided, the
            // direct SIMD kernel wins on plain contig same-shape.
            //
            // The direct ComparisonKernel has a vector body that emits
            // Vector256.Compare + ExtractMostSignificantBits + PDEP-packed
            // store — the fastest contig path we have. The Tier 3B 4×-unrolled
            // wrapper can't host that because bool-output breaks its same-type
            // invariant, so routing contig through here costs us 3× on the
            // simple shape.
            //
            // Skip NDIter routing for the "easy" contig cases that the direct
            // kernel handles best. Anything with broadcast (different shapes)
            // or non-contig storage benefits from the iterator's coalesce +
            // stride normalization + per-outer-iter pointer advance.
            bool sameShape = lhs.Shape.NDim == rhs.Shape.NDim
                             && lhs.Shape.size == rhs.Shape.size;
            if (sameShape && lhs.Shape.IsContiguous && rhs.Shape.IsContiguous)
                return null;

            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // NDIter shape arithmetic is int-bounded.
            if (cleanShape.size < 0) return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue) return null;

            // A C-ordered result: a non-C layout NumPy allocates reaches this route only as operands viewed in that
            // layout's memory order (ExecuteComparisonOpInLayout), whose C result is the layout's buffer.
            var result = NewArray<bool>(cleanShape, true);

            // Empty broadcast result: nothing to compare. The iterator refuses a zero-size output against a size-1 input
            // axis ("output operand requires a reduction along dimension 0"), so (2,1,3) > (0,3) raised where NumPy
            // returns its (2,0,3) result; the binary route returns early for the same shapes.
            if (result.size == 0)
                return result;

            // Resolve common comparison type once (same rule as the kernel key — incl. the exact
            // int64/uint64 pair compared at Decimal, NumPy's qQ/Qq loops).
            var comparisonType = ComparisonKernelKey.GetComparisonType(lhsType, rhsType);

            // Per-element body. Stack on entry: [lhs (lhsType), rhs (rhsType)].
            // Stash rhs into a local so we can convert lhs (bottom of stack)
            // first, then reload rhs and convert it, matching the mixed-binary
            // pattern. Then EmitComparisonOperation pops both and pushes bool.
            // NOTE (Wave 4, measured): the buffered-cast route was tried here
            // and reverted — comparisons are add-class cheap ops where the
            // fused per-element convert beats the buffer round-trip (see the
            // A/B table in DefaultEngine.BinaryOp). Promoting unary math keeps
            // the buffered route (DefaultEngine.UnaryOp).
            NPTypeCode capLhs = lhsType, capRhs = rhsType, capCmp = comparisonType;
            ComparisonOp capOp = op;
            Action<ILGenerator> scalarBody = il =>
            {
                if (capLhs == capRhs && capLhs == capCmp)
                {
                    // Same-dtype fast path — no convert pass.
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
                else
                {
                    var locRhs = il.DeclareLocal(DirectILKernelGenerator.GetClrType(capRhs));
                    il.Emit(OpCodes.Stloc, locRhs);
                    if (capLhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capLhs, capCmp);
                    il.Emit(OpCodes.Ldloc, locRhs);
                    if (capRhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capRhs, capCmp);
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
            };

            // Vector body intentionally null: the Tier 3B 4×-unrolled wrapper
            // requires same-dtype-across-all-operands and bool output breaks
            // that invariant. Inner-loop factory falls to scalar-strided body.
            // Packed key (no per-call string): npy_cmp_{op}_{lhsType}_{rhsType}.
            var cacheKey = InnerLoopKernelKey.Comparison(op, lhsType, rhsType);

            try
            {
                // COPY_IF_OVERLAP + OVERLAP_ASSUME_ELEMENTWISE per NumPy's ufunc
                // iterator flags (ufunc_object.c:1070); cheap extent check here
                // since the bool result is freshly allocated.
                using var iter = NDIterRef.MultiNew(
                    3, new[] { lhs, rhs, result },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP,
                    NPY_ORDER.NPY_CORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    new[]
                    {
                        NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                        NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                        NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                    });

                // Chunks long enough for a vector loop go through the comparison run: it reads contiguous and
                // one-value operands in place and stages stepped, reversed or other-dtype ones 8 KB at a time
                // (Backends/Kernels/ComparisonRun.cs). The per-element kernel below ran every one of these layouts
                // at about 1 ns an element (a float64 (250,400) array against a column: 108 us, NumPy 41). Narrow
                // rows and tiny arrays are declined by the run and keep that kernel (and its 2-D block form).
                if (!iter.TryExecuteComparisonRun(ComparisonRun.Get(op, lhsType, rhsType)))
                    iter.ExecuteElementWiseBinary(lhsType, rhsType, NPTypeCode.Boolean, scalarBody, null, cacheKey);
            }
            catch (NotSupportedException)
            {
                // The direct kernel takes over; the unwritten result goes back to the pool rather than to the finalizer.
                result.Dispose();
                return null;
            }

            return result;
        }
    }
}
