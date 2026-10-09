using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray<bool> All(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: true);

        public NDArray<bool> Any(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: false);

        public NDArray<bool> All(NDArray nd, int[] axis, bool keepdims)
            => ReduceLogicalMultiAxis(nd, axis, keepdims, reduceAll: true);

        public NDArray<bool> Any(NDArray nd, int[] axis, bool keepdims)
            => ReduceLogicalMultiAxis(nd, axis, keepdims, reduceAll: false);

        [NDScoped]
        private NDArray<bool> ReduceLogicalAxis(NDArray nd, int axis, bool keepdims, bool reduceAll)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            // Text: NumPy's logical_and / logical_or.reduce validates the axis (a 0-d input accepts 0 and -1), then reads
            // the strings' truth values through the bool cast: the stand-in is the truth array, so no string reaches the
            // bool kernels below.
            if (Strings.NDStringUfuncs.ReduceStandIn(reduceAll ? "all" : "any", nd, (int?)axis) is { } standIn)
            {
                using (standIn)
                    return ReduceLogicalAxis(standIn, axis, keepdims, reduceAll);
            }

            if (nd.ndim == 0)
            {
                if (axis == 0 || axis == -1)
                    return NDArray.Scalar(reduceAll ? All(nd) : Any(nd)).AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");

                throw new AxisError(axis, 0);
            }

            axis = NormalizeAxis(axis, nd.ndim);

            if (TryStreamLogicalAxis(nd, axis, keepdims, reduceAll) is { } streamed)
                return streamed;

            // Allocate the result in the *reduced* shape (axis dropped), laid out as NumPy's reduce iterator allocates
            // it (LogicalReductionShape: the input's memory order for an F or permuted input, plan api-parameter-parity
            // X10). keepdims re-inserts the axis at the end with ExpandDimension, which keeps that order — a Reshape
            // to C dims would copy an F result into C order. The axis kernels assume one fewer output dim than input.
            var reducedDims = Shape.GetAxis(nd.Shape, axis);
            Shape reducedShape = LogicalReductionShape(reducedDims, nd.Shape, axis);
            NDArray<bool> result = CreateLogicalResult(reducedShape, reduceAll && nd.Shape.dimensions[axis] == 0);

            if (result.size == 0 || nd.Shape.dimensions[axis] == 0)
            {
                if (keepdims)
                    result.Storage.ExpandDimension(axis);
                return result;
            }

            // Fast path: IL-emitted axis kernel. Inner-axis (stride==1) routes through the
            // SIMD all/any helpers; non-inner uses AVX2 gather (float/double) or a scalar
            // early-exit loop. Returns null for unsupported dtypes (Half / Complex / Decimal),
            // which fall through to the NDAxisIter scalar kernel below.
            ReductionOp op = reduceAll ? ReductionOp.All : ReductionOp.Any;
            var key = new AxisReductionKernelKey(nd.GetTypeCode, NPTypeCode.Boolean, op, InnerAxisContiguous: axis == nd.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetBooleanAxisReductionKernel(key);
            if (kernel != null && nd.Shape.IsContiguous)
            {
                unsafe
                {
                    fixed (long* inStrides = nd.Shape.strides)
                    fixed (long* inDims = nd.Shape.dimensions)
                    fixed (long* outStrides = result.Shape.strides)
                    {
                        byte* inBase = (byte*)nd.Storage.Address + nd.Shape.offset * nd.dtypesize;
                        long outSize = result.size > 0 ? result.size : 1;
                        kernel(inBase, (void*)result.Address, inStrides, inDims, outStrides,
                               axis, nd.Shape.dimensions[axis], nd.ndim, outSize);
                    }
                }
            }
            else
            {
                // NDAxisIter.ReduceBool writes each output through the result's own strides, so the K-ordered result of
                // an F or permuted input is filled in place (no C buffer relabelled or copied afterwards).
                NpFunc.Invoke(nd.GetTypeCode, ExecuteLogicalAxis<int>, nd, result, axis, reduceAll);
            }

            if (keepdims)
                result.Storage.ExpandDimension(axis);

            return result;
        }

        /// <summary>
        ///     The smallest bool operand <see cref="TryStreamLogicalAxis"/> sends to np.evaluate's axis stream: 16K elements.
        ///     Measured (NPY/NS, two pinned P-cores, (n/100, 100) operands, all-true / sparse / half-true data, C and F,
        ///     both axes): at 1K the engine's per-row kernels win (1.5-9x NumPy, np.evaluate's fixed cost puts some cells
        ///     at 1.0-1.2x), from 100K the stream wins every cell where the engine walked a strided row per output (an F
        ///     operand along axis 1 at 1M: 806 us -> 31 us; a C operand along axis 0: 498 -> 87 us) and ties elsewhere.
        /// </summary>
        private const long LogicalAxisStreamMinElements = 16384;

        /// <summary>
        ///     The reduced-axis length from which a reduction along the operand's CONTIGUOUS axis stays on the engine: each
        ///     output is then one long contiguous row the engine scans with one SIMD helper call, already at the read-bandwidth
        ///     floor (measured: 100 rows of 10K bools, all true, <c>all</c> — engine 20 us, stream 23 us, NumPy 26 us), while
        ///     short rows pay the engine's per-row overhead (10K rows of 100: engine 70 us, stream 37 us).
        /// </summary>
        private const long LogicalAxisLongRowElements = 4096;

        /// <summary>
        ///     <c>np.all</c> / <c>np.any</c> along one axis of a bool operand in one memory order (C or F), through np.evaluate's
        ///     Any / All axis stream: a contiguous reduced axis folds whole rows with vectorized span scans, a strided one
        ///     combines each output's slab into an accumulator row a vector at a time, both stopping once every output of
        ///     the current block is decided. Exact by construction (logical OR / AND are order-free), and the result comes
        ///     back in the operand's order (C for C, F for F) — NumPy's reduction layout.
        /// </summary>
        /// <param name="nd">The operand.</param>
        /// <param name="axis">The reduced axis, already normalized.</param>
        /// <param name="keepdims">Keep the reduced axis with extent 1.</param>
        /// <param name="reduceAll">True for <c>all</c>, false for <c>any</c>.</param>
        /// <returns>The result (owning; a kept axis re-inserted with ExpandDimension), or null when the operand is not bool,
        ///     not C- or F-contiguous, 1-D (its 0-d result stays the engine's), smaller than
        ///     <see cref="LogicalAxisStreamMinElements"/>, reduced along its contiguous axis with rows of at least
        ///     <see cref="LogicalAxisLongRowElements"/>, or dynamic code is unavailable — the engine's kernels then run.</returns>
        /// <exception cref="InvalidOperationException">The stream returned a non-bool array (an internal invariant).</exception>
        private static NDArray<bool> TryStreamLogicalAxis(NDArray nd, int axis, bool keepdims, bool reduceAll)
        {
            var shape = nd.Shape;
            if (nd.typecode != NPTypeCode.Boolean || nd.ndim < 2 || shape.size < LogicalAxisStreamMinElements
                || !(shape.IsContiguous || shape.IsFContiguous) || !DirectILKernelGenerator.Enabled)
                return null;
            // Long contiguous rows: the engine's one-call-per-row SIMD scan is at the bandwidth floor and slightly ahead.
            bool contiguousAxis = shape.IsContiguous ? axis == nd.ndim - 1 : axis == 0;
            if (contiguousAxis && shape.dimensions[axis] >= LogicalAxisLongRowElements)
                return null;
            var source = NDExpr.Arr(nd);
            // Taken WITHOUT keepdims: np.evaluate's keepdims form is a reshaped view, where NumPy's reduction owns its result.
            var result = np.evaluate(reduceAll ? NDExpr.All(source, axis) : NDExpr.Any(source, axis));
            if (keepdims)
                result.Storage.ExpandDimension(axis);
            return result.AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");
        }

        /// <summary>
        ///     The layout of a single-axis logical reduction's result (axis dropped): NumPy's <c>logical_and</c> /
        ///     <c>logical_or.reduce</c> allocates it with <c>npyiter_new_temp_array</c> over the reduce iterator — the input's
        ///     axes in NpyIter's order with the reduced one dropped (<see cref="Shape.ReductionOutputShape"/>), the same rule
        ///     <see cref="AllocateReductionResult"/> applies to the numeric reductions.
        /// </summary>
        /// <param name="reducedDims">The result's dims (the input's without <paramref name="axis"/>); adopted by the
        ///     returned shape, so pass a fresh array.</param>
        /// <param name="input">The reduced array's shape.</param>
        /// <param name="axis">The reduced axis, already normalized.</param>
        /// <returns>A dense, offset-0 shape: the scalar shape for a 1-D input, C for a C-contiguous input or a 1-D result,
        ///     the input's memory order otherwise (F for an F-contiguous input; probed 2.4.2: <c>np.any(f, axis=0)</c> of an
        ///     F <c>(3, 5, 4)</c> array has element strides (1, 5)).</returns>
        /// <remarks>
        ///     A strided input whose axes are in C memory order (a stepped slice) still gets a C result — NpyIter sorts by
        ///     |stride|, and those strides are already descending.
        /// </remarks>
        internal static Shape LogicalReductionShape(long[] reducedDims, Shape input, int axis)
        {
            if (reducedDims.Length == 0)
                return Shape.Scalar;
            // A 1-D result has one layout, and a C-contiguous input's order is C's (NpyIter's sort keeps it).
            if (reducedDims.Length == 1 || input.IsContiguous)
                return Shape.Adopt(reducedDims);
            return Shape.ReductionOutputShape(reducedDims, input, axis);
        }

        // Multi-axis reduction. Matches NumPy: reduces along all listed axes.
        // Fast paths:
        //   1. All axes reduced → 1-D SIMD `all`/`any` over the whole array.
        //   2. Adjacent axes on a C-contig input → reshape to fuse them, then a single
        //      axis reduction. Saves N-1 redundant kernel invocations.
        //   3. Otherwise → chain single-axis reductions with keepdims=true.
        [NDScoped]
        private NDArray<bool> ReduceLogicalMultiAxis(NDArray nd, int[] axis, bool keepdims, bool reduceAll)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));
            // Text: the tuple is checked item by item, then for repeats ("duplicate value in 'axis'"), before NumPy reads
            // the strings' truth values through the bool cast — an empty tuple included, which still casts text to bool:
            // the stand-in is the truth array.
            if (Strings.NDStringUfuncs.ReduceStandIn(reduceAll ? "all" : "any", nd, axis) is { } standIn)
            {
                using (standIn)
                    return ReduceLogicalMultiAxis(standIn, axis, keepdims, reduceAll);
            }

            // Empty tuple: NumPy returns input cast to bool (no reduction).
            if (axis.Length == 0)
                return CastToBoolPreservingShape(nd);

            // 0-d input
            if (nd.ndim == 0)
            {
                // Only valid axis values are 0 / -1 (and only one of them, per NumPy)
                if (axis.Length != 1 || (axis[0] != 0 && axis[0] != -1))
                    throw new AxisError(axis.Length > 0 ? axis[0] : 0, 0);

                return NDArray.Scalar(reduceAll ? All(nd) : Any(nd)).AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");
            }

            int ndim = nd.ndim;
            int[] normalized = NormalizeAndValidateAxes(axis, ndim);
            Array.Sort(normalized);

            // Fast path 1: every axis reduced → 1-D path handles the whole array in one
            // SIMD pass instead of N chained axis reductions through a scalar kernel.
            if (normalized.Length == ndim)
            {
                bool scalar = reduceAll ? All(nd) : Any(nd);
                var result = NDArray.Scalar(scalar).AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");
                if (keepdims)
                {
                    var dims = new long[ndim];
                    for (int i = 0; i < ndim; i++) dims[i] = 1;
                    result.Storage.Reshape(Shape.Adopt(dims));
                }
                return result;
            }

            // Fast path 2: contiguous run of axes on a C-contig input can be fused
            // (the reshape is free, no copy). E.g. axis=(0,1) of (128, 64, 64) C-contig
            // becomes a single axis-0 reduction on (8192, 64).
            if (nd.Shape.IsContiguous && AreContiguousRun(normalized))
            {
                NDArray<bool> result = ReduceContiguousAxisRun(nd, normalized, keepdims, reduceAll);
                return result;
            }

            // Fall back: chain single-axis reductions, highest axis first so lower
            // indices remain valid across passes.
            NDArray<bool> chained = null;
            NDArray current = nd;
            for (int i = normalized.Length - 1; i >= 0; i--)
            {
                chained = ReduceLogicalAxis(current, normalized[i], keepdims: true, reduceAll: reduceAll);
                current = chained;
            }

            if (!keepdims)
            {
                // Drop the kept extent-1 axes WITH their strides: each chained step laid its result out in its input's
                // memory order (LogicalReductionShape), so the surviving axes keep NumPy's K order. A Reshape to fresh C
                // dims would copy a non-C result into C order instead.
                var newDims = new List<long>(ndim - normalized.Length);
                var newStrides = new List<long>(ndim - normalized.Length);
                var axesSet = new HashSet<int>(normalized);
                long[] resultDims = chained.Shape.dimensions;
                long[] resultStrides = chained.Shape.strides;
                for (int d = 0; d < ndim; d++)
                {
                    if (!axesSet.Contains(d))
                    {
                        newDims.Add(resultDims[d]);
                        newStrides.Add(resultStrides[d]);
                    }
                }

                // Removing extent-1 axes never moves an element (each is read at index 0 only), so the buffer is
                // described exactly by the remaining dims and strides; the storage keeps its ownership (SetShapeUnsafe).
                Shape newShape = newDims.Count == 0 ? Shape.Scalar : Shape.Adopt(newDims.ToArray(), newStrides.ToArray());
                chained.Storage.SetShapeUnsafe(newShape);
            }

            return chained;
        }

        // True iff `sorted` (already sorted) is a strict +1 progression: e.g. (1, 2, 3).
        private static bool AreContiguousRun(int[] sorted)
        {
            for (int i = 1; i < sorted.Length; i++)
            {
                if (sorted[i] != sorted[i - 1] + 1)
                    return false;
            }
            return true;
        }

        // Fuse a contiguous run of reduced axes into a single axis via reshape, then
        // run one single-axis reduction. Caller has verified C-contiguous storage.
        [NDScopedCovered] // engine multi-axis All/Any is reached only from the [NDScoped] np.all/np.any entries (the fused reshape view is scope-reclaimed)
        private NDArray<bool> ReduceContiguousAxisRun(NDArray nd, int[] sortedAxes, bool keepdims, bool reduceAll)
        {
            int ndim = nd.ndim;
            long[] origDims = nd.Shape.dimensions;
            int firstAxis = sortedAxes[0];
            int lastAxis = sortedAxes[sortedAxes.Length - 1];

            // Build a reshape that collapses axes [firstAxis..lastAxis] into one axis.
            var newDims = new long[ndim - sortedAxes.Length + 1];
            int w = 0;
            for (int i = 0; i < firstAxis; i++) newDims[w++] = origDims[i];

            long fusedSize = 1;
            for (int i = firstAxis; i <= lastAxis; i++) fusedSize *= origDims[i];
            newDims[w++] = fusedSize;

            for (int i = lastAxis + 1; i < ndim; i++) newDims[w++] = origDims[i];

            // Reshape is a view on C-contig data (no copy).
            NDArray reshaped = nd.reshape(Shape.Adopt(newDims));
            NDArray<bool> reduced = ReduceLogicalAxis(reshaped, firstAxis, keepdims: false, reduceAll: reduceAll);

            if (keepdims)
            {
                var dimsKD = (long[])origDims.Clone();
                foreach (int a in sortedAxes) dimsKD[a] = 1;
                reduced.Storage.Reshape(Shape.Adopt(dimsKD));
            }

            return reduced;
        }

        // Cast every element to bool (truthy = non-zero), preserving the input shape.
        // Used for the np.all(a, axis=()) / np.any(a, axis=()) case where NumPy
        // returns the array reinterpreted as bool without performing any reduction.
        internal static NDArray<bool> CastToBoolPreservingShape(NDArray nd)
        {
            if (nd.GetTypeCode == NPTypeCode.Boolean)
            {
                // Already bool — return a copy so callers can't mutate the input.
                return nd.copy().AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");
            }

            // (nd != 0) returns an NDArray<bool> with element-wise non-zero check,
            // matching NumPy's truthiness semantics (including NaN→True, inf→True).
            return nd != 0;
        }

        /// <summary>
        ///     The tuple axis of np.all / np.any (ufunc reduce): every axis normalized first, then the repeat check.
        /// </summary>
        /// <param name="axes">The axes as given.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns>The normalized axes in input order.</returns>
        /// <exception cref="AxisError">An axis is out of range (the first one, in input order).</exception>
        /// <exception cref="ValueError">An axis appears twice ("duplicate value in 'axis'").</exception>
        /// <remarks>
        ///     NumPy's order: PyUFunc_GenericReduction's tuple branch runs check_and_adjust_axis on every entry, and only
        ///     then PyUFunc_Reduce's flag array rejects a repeat — so <c>(0, 0, 5)</c> reports the axis 5. A per-entry
        ///     check-then-repeat loop (this method's old form) reported the repeat.
        /// </remarks>
        internal static int[] NormalizeAndValidateAxes(int[] axes, int ndim)
        {
            var result = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
                result[i] = NormalizeAxis(axes[i], ndim);

            var seen = new HashSet<int>();
            foreach (int a in result)
                if (!seen.Add(a))
                    throw new ValueError("duplicate value in 'axis'");
            return result;
        }

        private static NDArray<bool> CreateLogicalResult(Shape resultShape, bool fillTrue)
        {
            var result = fillTrue
                ? np.ones(resultShape, NPTypeCode.Boolean)
                : np.zeros(resultShape, NPTypeCode.Boolean);

            return result.AsGeneric<bool>() ?? throw new InvalidOperationException("np.all/np.any expected a Boolean result.");
        }

        private static void ExecuteLogicalAxis<T>(NDArray nd, NDArray<bool> result, int axis, bool reduceAll)
            where T : unmanaged
        {
            if (reduceAll)
                NDAxisIter.ReduceBool<T, NDAllKernel<T>>(nd.Storage, result.Storage, axis);
            else
                NDAxisIter.ReduceBool<T, NDAnyKernel<T>>(nd.Storage, result.Storage, axis);
        }
    }
}
