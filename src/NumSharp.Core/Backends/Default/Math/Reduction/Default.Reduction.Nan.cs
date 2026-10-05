using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.nansum</c>: the sum of <paramref name="a"/>'s elements over <paramref name="axis"/> treating NaN as zero —
        ///     NumPy's <c>np.sum(where(isnan(a), 0, a), axis, keepdims)</c> (<c>_replace_nan</c> then <c>np.sum</c>), so it
        ///     follows np.sum's rules: the result starts from the identity +0.0 (a sum of -0.0 is +0.0), a complex element
        ///     with a NaN in EITHER part counts as zero, and an integer / bool input is plain <c>np.sum</c>.
        /// </summary>
        /// <param name="a">The array to reduce; any dtype, any layout.</param>
        /// <param name="axis">The axis to reduce, or null for every axis (a 0-d input also takes 0 / -1, np.sum's rule).</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <returns>The NaN-skipping sum (a read-only 0-d scalar for a full reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        public override NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;
            var tc = arr.GetTypeCode;

            // Non-float types: fall back to regular sum (no NaN possible)
            if (tc != NPTypeCode.Single && tc != NPTypeCode.Double && tc != NPTypeCode.Half && tc != NPTypeCode.Complex)
                return Sum(arr, axis: axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            // A one-element input is NumPy's identity + replaced element, 0-d (keepdims: (1,)*ndim). The old shortcut
            // returned the input's (1,) clone with its -0.0 sign, and a complex 0-d input with axis=0 looped forever in
            // NanSumComplex's coordinate incrementor. The flat reduction takes the fused NaN-skipping kernels, which start
            // from +0.0 and skip a NaN element — exactly identity + replaced element, with no isnan / where temporaries
            // (0.24 → 2.6 µs through the composition, which NumPy's own 3.2 µs barely beat); an axis keeps the
            // composition, whose np.sum route owns the axis rules (a 0-d input's axis 0 / -1, the shapes).
            if (shape.size == 1)
            {
                if (axis == null)
                    return tc == NPTypeCode.Complex
                        ? NanSumComplex(arr, null, keepdims)
                        : NanReductionElementWise(arr, ReductionOp.NanSum, keepdims);
                return NanReplacedReduction(arr, axis, keepdims, ReductionOp.NanSum);
            }

            // B15: Complex nansum — treat any entry with NaN in real OR imag as zero.
            if (tc == NPTypeCode.Complex)
                return NanSumComplex(arr, axis, keepdims);

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanSum, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanSum);
            }
        }

        /// <summary>
        ///     <c>np.nanprod</c>: the product of <paramref name="a"/>'s elements over <paramref name="axis"/> treating NaN
        ///     as one — NumPy's <c>np.prod(where(isnan(a), 1, a), axis, keepdims)</c>, so it follows np.prod's rules (the
        ///     identity start <c>1</c>, the result shape). An integer / bool input is plain <c>np.prod</c>.
        /// </summary>
        /// <param name="a">The array to reduce; any dtype, any layout.</param>
        /// <param name="axis">The axis to reduce, or null for every axis (a 0-d input also takes 0 / -1, np.prod's rule).</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <returns>The NaN-skipping product (a read-only 0-d scalar for a full reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <remarks>
        ///     A complex input goes to <c>np.prod</c> without the NaN replacement (pre-existing; NumPy replaces a complex
        ///     element with a NaN in either part by <c>1</c>), and a lone complex element skips NumPy's <c>(1+0j) * x</c>
        ///     — pinned by <c>OpenBugsReductionTests.Prod_OneComplexElement_IsTheIdentityTimesTheElement</c>.
        /// </remarks>
        public override NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            // Non-float types: fall back to regular prod (no NaN possible)
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double && arr.GetTypeCode != NPTypeCode.Half)
                return ReduceProduct(arr, axis, keepdims: keepdims);

            if (shape.IsEmpty)
                return arr;

            // A one-element input: NumPy's np.prod(where(isnan(a), 1, a)) — 1 * replaced element, 0-d (keepdims:
            // (1,)*ndim). The old shortcut returned the input's (1,) clone. The flat reduction takes the fused
            // NaN-skipping kernel (it starts from 1 and skips a NaN element: exactly NumPy's value, and a real 1 * x
            // keeps x's sign); an axis keeps the composition, whose np.prod route owns the axis rules.
            if (shape.size == 1)
            {
                if (axis == null)
                    return NanReductionElementWise(arr, ReductionOp.NanProd, keepdims);
                return NanReplacedReduction(arr, axis, keepdims, ReductionOp.NanProd);
            }

            if (axis == null)
            {
                return NanReductionElementWise(arr, ReductionOp.NanProd, keepdims);
            }
            else
            {
                return ExecuteNanAxisReduction(arr, axis.Value, keepdims, ReductionOp.NanProd);
            }
        }

        /// <summary>
        /// Return minimum of an array or minimum along an axis, ignoring NaNs — NumPy's <c>np.nanmin</c>, which on an
        /// ndarray IS <c>np.fmin.reduce</c> (see <see cref="NanMinMax"/>).
        /// </summary>
        /// <param name="a">The array to reduce.</param>
        /// <param name="axis">The axis to reduce (null = all axes).</param>
        /// <param name="keepdims">Keep the reduced axes as size 1.</param>
        /// <returns>The reduced array (a 0-d scalar for a flat reduction without keepdims).</returns>
        /// <exception cref="ArgumentException">The reduction is empty — "zero-size array to reduction operation fmin which
        /// has no identity", NumPy's ValueError.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        public override NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false)
            => NanMinMax(a, axis, keepdims, MinMaxOp.FMin);

        /// <summary>
        /// Return maximum of an array or maximum along an axis, ignoring NaNs — NumPy's <c>np.nanmax</c>, which on an
        /// ndarray IS <c>np.fmax.reduce</c> (see <see cref="NanMinMax"/>).
        /// </summary>
        /// <param name="a">The array to reduce.</param>
        /// <param name="axis">The axis to reduce (null = all axes).</param>
        /// <param name="keepdims">Keep the reduced axes as size 1.</param>
        /// <returns>The reduced array (a 0-d scalar for a flat reduction without keepdims).</returns>
        /// <exception cref="ArgumentException">The reduction is empty — "zero-size array to reduction operation fmax which
        /// has no identity", NumPy's ValueError.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        public override NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false)
            => NanMinMax(a, axis, keepdims, MinMaxOp.FMax);

        /// <summary>
        /// The shared body of <see cref="NanMax"/> / <see cref="NanMin"/>. NumPy's <c>nanmax(a)</c> for a plain ndarray is
        /// <c>np.fmax.reduce(a, axis)</c> plus an "All-NaN slice" RuntimeWarning (NumSharp has no warnings), so float32 /
        /// float64 run NumPy's own fmax / fmin reduction schedules through the exact core
        /// (<see cref="TryExactFlatMinMaxScalar"/>, <see cref="TryExactAxisMinMax"/> with <see cref="MinMaxOp.FMax"/> /
        /// <see cref="MinMaxOp.FMin"/>).
        /// </summary>
        /// <remarks>
        /// <para>
        /// Consequence for floats: the result's BITS are NumPy's — which zero survives a ±0 tie (the CRT scalar op keeps
        /// <c>+0</c> for max / <c>-0</c> for min where the vector op keeps the later operand, so it depends on where
        /// NumPy's calls put each element) and which payload an all-NaN reduction returns (never the canonical NaN: an
        /// all-NaN accumulator hands back its lane 0 as is). The kernels it replaces matched only the value.
        /// </para>
        /// <para>
        /// Edges, probed against NumPy 2.4.2: an empty reduction raises with the <c>fmax</c> / <c>fmin</c> name even for an
        /// integer array (a reduced axis of extent 0, or a flat reduction of an empty array; a non-empty reduced axis of an
        /// empty array yields the empty result); a 0-d or one-element 1-D input returns its element as a 0-d scalar (a
        /// NaN keeps its payload — NumPy copies the first element and never calls the loop). Integer and bool arrays are
        /// the plain <c>np.amax</c> / <c>np.amin</c> (fmax of integers IS max).
        /// </para>
        /// <para>
        /// float16 and complex128 have no SIMD loop in NumPy — <c>HALF_fmax</c> / <c>CDOUBLE_fmax</c> are sequential — and
        /// fold in NumPy's visiting order on every layout: float16 through the bit-level SIMD walks it shares with np.max /
        /// np.min (<see cref="HalfMinMaxReduce"/>), complex128 through <see cref="NanMinMaxSequential"/>. NaN elements
        /// skipped (complex: a NaN in either part), a tie keeping the EARLIER element in NumPy's visiting order, an all-NaN
        /// reduction returning its first NaN verbatim. Complex used
        /// to go to <c>np.amax</c> / <c>np.amin</c>, which PROPAGATE a NaN — a value bug. A broadcast float32 / float64
        /// input — which the exact core declines (a zero stride changes NpyIter's axis sort) — keeps the previous kernels:
        /// the right value, but a ±0 tie's sign and an all-NaN slice's payload are not pinned.
        /// </para>
        /// </remarks>
        /// <param name="a">The array to reduce.</param>
        /// <param name="axis">The axis to reduce (null = all axes).</param>
        /// <param name="keepdims">Keep the reduced axes as size 1.</param>
        /// <param name="op"><see cref="MinMaxOp.FMax"/> or <see cref="MinMaxOp.FMin"/>.</param>
        /// <returns>The reduced array.</returns>
        /// <exception cref="ArgumentException">The reduction is empty (NumPy's ValueError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        private NDArray NanMinMax(NDArray a, int? axis, bool keepdims, MinMaxOp op)
        {
            var arr = a;
            var shape = arr.Shape;
            var tc = arr.GetTypeCode;
            bool isMax = op == MinMaxOp.FMax;
            var nanOp = isMax ? ReductionOp.NanMax : ReductionOp.NanMin;

            if (shape.IsEmpty)
                return arr;

            // fmax / fmin have no identity: an empty reduction raises with THEIR name (not "maximum"), for every dtype —
            // so this precedes the integer delegation below.
            if (shape.size == 0)
                return HandleEmptyArrayMinMaxReduction(arr, axis, keepdims, null, NumPyMinMaxReduce.UfuncName(op));

            // Types that cannot hold a NaN: fmax / fmin of them ARE max / min — the exact amax / amin. Complex is NOT one of
            // them (a NaN in either part makes the element NaN, which fmax SKIPS where amax would propagate it), so it is
            // kept for the sequential fold below.
            if (tc != NPTypeCode.Single && tc != NPTypeCode.Double && tc != NPTypeCode.Half && tc != NPTypeCode.Complex)
                return isMax ? ReduceAMax(arr, axis, keepdims: keepdims) : ReduceAMin(arr, axis, keepdims: keepdims);

            // A single element: NumPy copies it into the result and never calls the loop (bits, NaN payload included, kept).
            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, null, null);

            // float16 / complex128: NumPy's HALF_fmax / CDOUBLE_fmax are plain sequential loops, folded in NumPy's visiting
            // order on every layout (broadcast included). float16 runs the bit-level SIMD walks of
            // Default.Reduction.MinMax.Half.cs (the same walks as np.max / np.min, with the NaN-skipping rule); complex128
            // the scalar fold of Default.Reduction.Nan.Sequential.cs.
            if (tc == NPTypeCode.Half)
                return HalfMinMaxReduce(arr, axis, keepdims, isMax, nanIgnore: true);
            if (tc == NPTypeCode.Complex)
                return NanMinMaxSequential(arr, axis, keepdims, isMax);

            bool exactDtype = tc == NPTypeCode.Single || tc == NPTypeCode.Double;
            if (axis == null)
            {
                if (exactDtype)
                {
                    // NumPy's flat fmax / fmin schedule (null = declined: a broadcast input → the kernels below).
                    var r = TryExactFlatMinMaxScalar(arr, op);
                    if (r is not null)
                    {
                        if (keepdims)
                        {
                            var ks = new long[arr.ndim];
                            for (int i = 0; i < arr.ndim; i++)
                                ks[i] = 1;
                            r.Storage.Reshape(Shape.Adopt(ks));
                        }

                        return r.MarkReductionScalar();
                    }
                }

                return NanReductionElementWise(arr, nanOp, keepdims);
            }

            if (exactDtype)
            {
                int ax = NormalizeAxis(axis.Value, arr.ndim);

                // An extent-1 reduced axis: NumPy copies each lone element (payload kept) — no loop call.
                if (shape[ax] == 1)
                    return HandleTrivialAxisReduction(arr, ax, keepdims, tc, null);

                // NumPy's per-element fmax / fmin schedule (null = declined: a broadcast input → the kernel below).
                var exact = TryExactAxisMinMax(arr, ax, op);
                if (exact is not null)
                {
                    if (keepdims)
                        exact.Storage.ExpandDimension(ax);
                    // Same PyArray_Return rule as ReduceAMax: a fresh 0-d result is a read-only scalar.
                    return exact.MarkReductionScalar();
                }
            }

            return ExecuteNanAxisReduction(arr, axis.Value, keepdims, nanOp);
        }

        /// <summary>
        /// Element-wise NaN reduction (axis=null).
        /// </summary>
        private NDArray NanReductionElementWise(NDArray arr, ReductionOp op, bool keepdims)
        {
            if (DirectILKernelGenerator.Enabled && arr.Shape.IsContiguous)
            {
                object result;
                unsafe
                {
                    switch (arr.GetTypeCode)
                    {
                        case NPTypeCode.Single:
                            result = op switch
                            {
                                ReductionOp.NanSum => DirectILKernelGenerator.NanSumSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanProd => DirectILKernelGenerator.NanProdSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanMin => DirectILKernelGenerator.NanMinSimdHelperFloat((float*)arr.Address, arr.size),
                                ReductionOp.NanMax => DirectILKernelGenerator.NanMaxSimdHelperFloat((float*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        case NPTypeCode.Double:
                            result = op switch
                            {
                                ReductionOp.NanSum => DirectILKernelGenerator.NanSumSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanProd => DirectILKernelGenerator.NanProdSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanMin => DirectILKernelGenerator.NanMinSimdHelperDouble((double*)arr.Address, arr.size),
                                ReductionOp.NanMax => DirectILKernelGenerator.NanMaxSimdHelperDouble((double*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        case NPTypeCode.Half:
                            result = op switch
                            {
                                ReductionOp.NanSum => DirectILKernelGenerator.NanSumHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanProd => DirectILKernelGenerator.NanProdHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanMin => DirectILKernelGenerator.NanMinHalfHelper((Half*)arr.Address, arr.size),
                                ReductionOp.NanMax => DirectILKernelGenerator.NanMaxHalfHelper((Half*)arr.Address, arr.size),
                                _ => throw new NotSupportedException($"Unsupported NaN reduction: {op}")
                            };
                            break;
                        default:
                            throw new NotSupportedException($"NaN reductions only support float/double/half, got {arr.GetTypeCode}");
                    }
                }

                var r = NDArray.Scalar(result);
                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(Shape.Adopt(keepdimsShape));
                }
                return r.MarkReductionScalar();
            }
            else
            {
                return NanReductionScalar(arr, op, keepdims);
            }
        }

        /// <summary>
        /// Scalar fallback for NaN reduction when SIMD is not available.
        /// </summary>
        private NDArray NanReductionScalar(NDArray arr, ReductionOp op, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                    result = NanReduceScalarFloat(arr, op);
                    break;
                case NPTypeCode.Double:
                    result = NanReduceScalarDouble(arr, op);
                    break;
                case NPTypeCode.Half:
                    result = NanReduceScalarHalf(arr, op);
                    break;
                default:
                    throw new NotSupportedException($"NaN reductions only support float/double/half, got {arr.GetTypeCode}");
            }

            var r = NDArray.Scalar(result);
            if (keepdims)
            {
                var keepdimsShape = new long[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    keepdimsShape[i] = 1;
                r.Storage.Reshape(Shape.Adopt(keepdimsShape));
            }
            return r.MarkReductionScalar();
        }

        private static float NanReduceScalarFloat(NDArray arr, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return iter.ExecuteReducing<NanSumFloatKernel, float>(default, 0f);
                }
                case ReductionOp.NanProd:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return iter.ExecuteReducing<NanProdFloatKernel, float>(default, 1f);
                }
                case ReductionOp.NanMin:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMinFloatKernel, NanMinMaxFloatAccumulator>(default, default);
                    return accum.Found ? accum.Value : float.NaN;
                }
                case ReductionOp.NanMax:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMaxFloatKernel, NanMinMaxFloatAccumulator>(default, default);
                    return accum.Found ? accum.Value : float.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        private static double NanReduceScalarDouble(NDArray arr, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return iter.ExecuteReducing<NanSumDoubleKernel, double>(default, 0.0);
                }
                case ReductionOp.NanProd:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return iter.ExecuteReducing<NanProdDoubleKernel, double>(default, 1.0);
                }
                case ReductionOp.NanMin:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMinDoubleKernel, NanMinMaxDoubleAccumulator>(default, default);
                    return accum.Found ? accum.Value : double.NaN;
                }
                case ReductionOp.NanMax:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMaxDoubleKernel, NanMinMaxDoubleAccumulator>(default, default);
                    return accum.Found ? accum.Value : double.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        private static unsafe Half NanReduceScalarHalf(NDArray arr, ReductionOp op)
        {
            // Struct-generic ExecuteReducing (NaN-skipping kernels), accumulating
            // in double for precision then narrowing to Half — same semantics as
            // the Float/Double scalar paths above, ~2.4× the old per-element
            // AsIterator. min/max return Half.NaN when every element was NaN.
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return (Half)iter.ExecuteReducing<NanSumHalfKernel, double>(default, 0.0);
                }
                case ReductionOp.NanProd:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    return (Half)iter.ExecuteReducing<NanProdHalfKernel, double>(default, 1.0);
                }
                case ReductionOp.NanMin:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMinHalfKernel, NanMinMaxDoubleAccumulator>(default, default);
                    return accum.Found ? (Half)accum.Value : Half.NaN;
                }
                case ReductionOp.NanMax:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMaxHalfKernel, NanMinMaxDoubleAccumulator>(default, default);
                    return accum.Found ? (Half)accum.Value : Half.NaN;
                }
                default:
                    throw new NotSupportedException($"Unsupported NaN reduction: {op}");
            }
        }

        /// <summary>
        /// Execute a NaN-aware axis reduction.
        /// </summary>
        private unsafe NDArray ExecuteNanAxisReduction(NDArray arr, int axis, bool keepdims, ReductionOp op)
        {
            var shape = arr.Shape;

            // Normalize axis
            while (axis < 0) axis = arr.ndim + axis;
            if (axis >= arr.ndim) throw new ArgumentOutOfRangeException(nameof(axis));

            // Get kernel
            var inputType = arr.GetTypeCode;
            var key = new AxisReductionKernelKey(inputType, inputType, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetNanAxisReductionKernel(key);

            if (kernel == null)
            {
                return ExecuteNanAxisReductionScalar(arr, axis, keepdims, op);
            }

            // Create output array
            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            // KEEPORDER: allocate the result in the input's memory order (F for an F-contig input),
            // so the strided-output kernel fills it in place with no post-hoc copy (issue #610).
            var result = AllocateReductionResult(inputType, outputDims, shape);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            // ExpandDimension re-inserts the reduced axis preserving the KEEPORDER layout.
            if (keepdims)
                result.Storage.ExpandDimension(axis);
            return result;
        }

        /// <summary>
        /// Scalar fallback for NaN axis reduction.
        /// </summary>
        private NDArray ExecuteNanAxisReductionScalar(NDArray arr, int axis, bool keepdims, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            // KEEPORDER: allocate in the input's memory order (F for an F-contig input); SetAtIndex
            // writes each element through the result's strides, so the F-order buffer fills correctly
            // (issue #610). outputDimStrides below are a C-order decomposition of the LOGICAL output
            // index used only to locate the INPUT slice — independent of the output's physical layout.
            var result = AllocateReductionResult(inputType, outputDims, shape);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;

            long[] outputDimStrides = new long[arr.ndim - 1 > 0 ? arr.ndim - 1 : 1];
            if (arr.ndim > 1)
            {
                outputDimStrides[arr.ndim - 2] = 1;
                for (int d = arr.ndim - 3; d >= 0; d--)
                {
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * shape.dimensions[nextInputDim];
                }
            }

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                long remaining = outIdx;
                long inputBaseOffset = 0;

                for (int d = 0; d < arr.ndim - 1; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    long coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * shape.strides[inputDim];
                }

                object reduced;
                switch (inputType)
                {
                    case NPTypeCode.Single:
                        reduced = ReduceNanAxisScalarFloat(arr, inputBaseOffset, axisSize, shape.strides[axis], op);
                        break;
                    case NPTypeCode.Double:
                        reduced = ReduceNanAxisScalarDouble(arr, inputBaseOffset, axisSize, shape.strides[axis], op);
                        break;
                    case NPTypeCode.Half:
                        reduced = ReduceNanAxisScalarHalf(arr, inputBaseOffset, axisSize, shape.strides[axis], op);
                        break;
                    default:
                        reduced = 0;
                        break;
                }

                result.SetAtIndex(reduced, outIdx);
            }

            // ExpandDimension re-inserts the reduced axis preserving the KEEPORDER layout.
            if (keepdims)
                result.Storage.ExpandDimension(axis);
            return result;
        }

        /// <summary>
        ///     Reads the element at MEMORY offset <paramref name="offset"/> (in elements, relative to the view's own
        ///     <c>Shape.offset</c>) — the coordinate the scalar NaN-axis walk computes from the strides. It must not go
        ///     through <c>GetAtIndex</c>, which takes a LOGICAL C-order index and maps it through the strides a second
        ///     time: on a strided, reversed or transposed view that read other elements (or past the view), so
        ///     <c>np.nansum</c> of a strided float16 slab of negative zeros returned values like 0.0289.
        /// </summary>
        /// <typeparam name="T">The element type (the array's dtype).</typeparam>
        /// <param name="arr">The array whose buffer is read.</param>
        /// <param name="offset">The memory offset in elements, relative to <c>arr.Shape.offset</c>.</param>
        /// <returns>The element stored there.</returns>
        private static unsafe T ReadAtMemoryOffset<T>(NDArray arr, long offset) where T : unmanaged
            => ((T*)arr.Address)[arr.Shape.offset + offset];

        private static float ReduceNanAxisScalarFloat(NDArray arr, long baseOffset, long axisSize, long axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    float sum = 0f;
                    for (long i = 0; i < axisSize; i++)
                    {
                        float val = ReadAtMemoryOffset<float>(arr, baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    float prod = 1f;
                    for (long i = 0; i < axisSize; i++)
                    {
                        float val = ReadAtMemoryOffset<float>(arr, baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    float minVal = float.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        float val = ReadAtMemoryOffset<float>(arr, baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) { if (val < minVal) minVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? minVal : float.NaN;
                }
                case ReductionOp.NanMax:
                {
                    float maxVal = float.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        float val = ReadAtMemoryOffset<float>(arr, baseOffset + i * axisStride);
                        if (!float.IsNaN(val)) { if (val > maxVal) maxVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? maxVal : float.NaN;
                }
                default:
                    return 0f;
            }
        }

        private static double ReduceNanAxisScalarDouble(NDArray arr, long baseOffset, long axisSize, long axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = ReadAtMemoryOffset<double>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) sum += val;
                    }
                    return sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = ReadAtMemoryOffset<double>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) prod *= val;
                    }
                    return prod;
                }
                case ReductionOp.NanMin:
                {
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = ReadAtMemoryOffset<double>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val < minVal) minVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? minVal : double.NaN;
                }
                case ReductionOp.NanMax:
                {
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = ReadAtMemoryOffset<double>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val > maxVal) maxVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? maxVal : double.NaN;
                }
                default:
                    return 0.0;
            }
        }

        /// <summary>
        /// Half-typed scalar NaN axis reduction. Uses double accumulator for precision,
        /// casts final result back to Half to preserve dtype.
        /// </summary>
        private static Half ReduceNanAxisScalarHalf(NDArray arr, long baseOffset, long axisSize, long axisStride, ReductionOp op)
        {
            switch (op)
            {
                case ReductionOp.NanSum:
                {
                    double sum = 0.0;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = (double)ReadAtMemoryOffset<Half>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) sum += val;
                    }
                    return (Half)sum;
                }
                case ReductionOp.NanProd:
                {
                    double prod = 1.0;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = (double)ReadAtMemoryOffset<Half>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) prod *= val;
                    }
                    return (Half)prod;
                }
                case ReductionOp.NanMin:
                {
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = (double)ReadAtMemoryOffset<Half>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val < minVal) minVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? (Half)minVal : Half.NaN;
                }
                case ReductionOp.NanMax:
                {
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    for (long i = 0; i < axisSize; i++)
                    {
                        double val = (double)ReadAtMemoryOffset<Half>(arr, baseOffset + i * axisStride);
                        if (!double.IsNaN(val)) { if (val > maxVal) maxVal = val; foundNonNaN = true; }
                    }
                    return foundNonNaN ? (Half)maxVal : Half.NaN;
                }
                default:
                    return Half.Zero;
            }
        }

        /// <summary>
        /// B15: NumPy-parity Complex nansum. Treats any element with NaN in real OR imag
        /// as zero (skipped). Sum type is Complex.
        /// </summary>
        private unsafe NDArray NanSumComplex(NDArray arr, int? axis, bool keepdims)
        {
            var shape = arr.Shape;
            if (shape.IsEmpty) return arr;

            if (axis == null)
            {
                // Struct-generic NaN-skip Complex sum (~2.6× the old per-element AsIterator).
                System.Numerics.Complex sum;
                using (var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP))
                    sum = iter.ExecuteReducing<NanSumComplexKernel, System.Numerics.Complex>(default, System.Numerics.Complex.Zero);
                var r = NDArray.Scalar(sum);
                if (keepdims)
                {
                    var ks = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++) ks[i] = 1;
                    r.Storage.Reshape(Shape.Adopt(ks));
                }
                return r.MarkReductionScalar();
            }

            // Axis reduction: NumPy's own structure, np.sum(where(isnan(a), 0, a), axis, keepdims). The per-slice
            // NDCoordinatesAxisIncrementor walk it replaces threw on a 1-D input ("vector shape") and could not reduce
            // an empty slab; the composition inherits np.sum's routes (the +0.0 identity, the complex axis kernels,
            // KEEPORDER output, every layout).
            return NanReplacedReduction(arr, axis, keepdims, ReductionOp.NanSum);
        }

        /// <summary>
        ///     NumPy's structure for <c>np.nansum</c> / <c>np.nanprod</c>: <c>_replace_nan(a, identity)</c> — every NaN
        ///     element replaced by the reduction's identity (0 for a sum, 1 for a product; a complex element with a NaN in
        ///     EITHER part, which is what <c>np.isnan</c> tests) — then <c>np.sum</c> / <c>np.prod</c> over the result. The
        ///     reduction therefore follows np.sum's / np.prod's rules exactly: the identity start, the result shape (0-d,
        ///     keepdims, the 0-d input's axis 0 / -1), the axis validation and every layout.
        /// </summary>
        /// <param name="arr">The float16 / float32 / float64 / complex128 input.</param>
        /// <param name="axis">The axis to reduce, or null for every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="op"><see cref="ReductionOp.NanSum"/> or <see cref="ReductionOp.NanProd"/>.</param>
        /// <returns>The reduction of the NaN-replaced array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <remarks>
        ///     Three IL passes (isnan, where, the reduction) where the fused NaN kernels make one; it serves the routes
        ///     those kernels do not cover (a one-element input, the complex axis sum). NumPy itself makes four (a copy,
        ///     isnan, copyto, the reduction). The mask, the fill scalar and the replaced copy are this method's own
        ///     intermediates and are released here; the reduction's result is always a fresh array.
        /// </remarks>
        private NDArray NanReplacedReduction(NDArray arr, int? axis, bool keepdims, ReductionOp op)
        {
            var tc = arr.GetTypeCode;
            using var isNaN = np.isnan(arr);
            using var fill = NDArray.Scalar(op == ReductionOp.NanProd ? tc.GetOneValue() : tc.GetDefaultValue(), tc);
            using var replaced = np.where(isNaN, fill, arr);
            return op == ReductionOp.NanProd
                ? ReduceProduct(replaced, axis, keepdims: keepdims)
                : ReduceAdd(replaced, axis, keepdims);
        }
    }
}
