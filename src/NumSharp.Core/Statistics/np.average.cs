using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the weighted average along the specified axis.
        ///     Equivalent to <c>sum(a * weights) / sum(weights)</c>. When <paramref name="weights"/>
        ///     is null this reduces to <see cref="mean(NDArray)"/> over the same axes.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static NDArray average(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     Compute the weighted average along a tuple of axes. Equivalent to
        ///     <c>np.average(a, axis, weights, keepdims)</c> in NumPy with a tuple axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static NDArray average(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     Compute the weighted average and return a tuple <c>(avg, sum_of_weights)</c>.
        ///     Equivalent to <c>numpy.average(..., returned=True)</c>. When <paramref name="weights"/>
        ///     is null, <c>sum_of_weights</c> is the number of elements per output cell
        ///     (broadcast to the average's shape).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: true);
        }

        /// <summary>
        ///     Tuple-axis overload of <see cref="average_returned(NDArray, int?, NDArray, bool)"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: true);
        }

        private static (NDArray avg, NDArray sumOfWeights) AverageCore(
            NDArray a, int[] axis, NDArray weights, bool keepdims, bool returned)
        {
            int ndim = a.ndim;
            int[] normalizedAxis = NormalizeAxisTuple(axis, ndim);

            if (weights is null)
            {
                NDArray avg = MeanWithAxes(a, normalizedAxis, keepdims);

                // NumPy computes `scl = avg.dtype.type(a.size / avg.size)` unconditionally
                // (before the returned check), so 0/0 raises ZeroDivisionError even when
                // returned=False — see lib/_function_base_impl.average lines 576-578.
                if (avg.size == 0)
                    throw new DivideByZeroException("division by zero");

                if (!returned) return (avg, null);

                double count = (double)a.size / avg.size;
                NDArray scl = NDArray.Scalar(count).astype(avg.typecode);
                if (!scl.shape.SequenceEqual(avg.shape))
                    scl = np.broadcast_to(scl, avg).copy();
                return (avg, scl);
            }

            NDArray wgt = WeightsAreValid(weights, a, normalizedAxis);

            NPTypeCode resultDtype = ComputeResultDtype(a.typecode, wgt.typecode);

            NDArray wgtCast = wgt.typecode == resultDtype ? wgt : wgt.astype(resultDtype);
            NDArray aCast = a.typecode == resultDtype ? a : a.astype(resultDtype);

            // 0-D scalar fast path: reduce-all (axis=None or axes covers all dims)
            // bypasses NDIter + NDArray allocation entirely. Accumulates into
            // two stack-allocated doubles, scalar-divides, wraps once at exit.
            // Cuts ~13 μs of fixed pipeline overhead per call (np.zeros×2 + iter
            // setup + scalar HasZero NDArray equality + scalar NDArray divide).
            bool reduceAll = normalizedAxis is null || normalizedAxis.Length == a.ndim;
            if (reduceAll && aCast.Shape.IsContiguous && wgtCast.Shape.IsContiguous &&
                TryFusedWeightedSumScalar(aCast, wgtCast, resultDtype,
                    out NDArray avgScalar, out NDArray sclScalar))
            {
                if (keepdims)
                {
                    avgScalar = KeepdimsReshape(avgScalar, a.shape, normalizedAxis);
                    sclScalar = KeepdimsReshape(sclScalar, a.shape, normalizedAxis);
                }
                return (avgScalar, returned ? sclScalar : null);
            }

            // Fused fast path via DirectILKernelGenerator: NDIter walks a + w in one
            // pass producing (num, scl) into pre-zeroed output NDArrays. The
            // cached kernel handles per-dtype specialization (SIMD via Vector256<T>
            // for SIMD-capable types, scalar otherwise). When the dtype has no
            // kernel (Bool/Char/Half/Complex/Decimal) we fall back to
            // `aCast * wgtCast → sum` below.
            if (TryFusedWeightedSum(aCast, wgtCast, normalizedAxis, resultDtype, out NDArray numFast, out NDArray sclFast))
            {
                if (HasZero(sclFast))
                    throw new DivideByZeroException("Weights sum to zero, can't be normalized");

                NDArray avgFast = numFast / sclFast;
                if (keepdims)
                {
                    avgFast = KeepdimsReshape(avgFast, a.shape, normalizedAxis);
                    sclFast = KeepdimsReshape(sclFast, a.shape, normalizedAxis);
                }
                if (!returned) return (avgFast, null);
                if (!sclFast.shape.SequenceEqual(avgFast.shape))
                    sclFast = np.broadcast_to(sclFast, avgFast).copy();
                return (avgFast, sclFast);
            }

            // Fallback path for dtypes the IL kernel doesn't cover.
            NDArray scl_ = SumWithAxes(wgtCast, normalizedAxis, resultDtype, keepdims);
            if (HasZero(scl_))
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");
            NDArray prod = aCast * wgtCast;
            NDArray num = SumWithAxes(prod, normalizedAxis, resultDtype, keepdims);
            NDArray avg_ = num / scl_;
            if (!returned) return (avg_, null);
            if (!scl_.shape.SequenceEqual(avg_.shape))
                scl_ = np.broadcast_to(scl_, avg_).copy();
            return (avg_, scl_);
        }

        // Reshape (num, scl) from reduced shape back to keepdims shape.
        // axes==null means "reduce-all" — every dim becomes 1.
        private static NDArray KeepdimsReshape(NDArray reduced, long[] aShape, int[] axes)
        {
            int ndim = aShape.Length;
            long[] kd = new long[ndim];
            int outIdx = 0;
            for (int i = 0; i < ndim; i++)
            {
                bool isReduced = axes is null || Array.IndexOf(axes, i) >= 0;
                if (isReduced) kd[i] = 1L;
                else kd[i] = reduced.shape[outIdx++];
            }
            return reduced.reshape(kd);
        }

        // Scalar (reduce-all, both inputs C-contig) fast path.
        // Stackallocs 2-double output cells, runs the cached IL kernel once with
        // 4 stackalloc'd ptrs/strides (stride=8 for inputs, stride=0 for outputs),
        // scalar-checks for zero, scalar-divides. NO NDArray allocation, NO
        // NDIter setup. Returns the result wrapped as a 0-D NDArray.Scalar.
        //
        // Bails to the NDIter path when the dtype kernel is unavailable
        // (Half/Decimal/Complex/Bool/Char) or when inputs aren't C-contig.
        private static unsafe bool TryFusedWeightedSumScalar(
            NDArray a, NDArray w, NPTypeCode resultDtype,
            out NDArray avg, out NDArray scl)
        {
            NDInnerLoopFunc kernel = DirectILKernelGenerator.GetWeightedSumIterKernel(
                new DirectILKernelGenerator.WeightedSumKernelKey(resultDtype));
            if (kernel is null) { avg = null; scl = null; return false; }

            int elemSize = a.dtypesize;
            byte* ap = (byte*)a.Address + a.Shape.offset * elemSize;
            byte* wp = (byte*)w.Address + w.Shape.offset * elemSize;
            long count = a.size;

            // Stack scratch: 4 ptrs + 4 strides + 2 output cells (max 16 bytes per cell for Complex).
            void** ptrs = stackalloc void*[4];
            long* strs = stackalloc long[4];
            // 2 cells of 16 bytes is enough for everything up to Complex (16B).
            byte* outBuf = stackalloc byte[32];
            byte* numCell = outBuf;
            byte* sclCell = outBuf + 16;
            // Zero them — kernel does `+=` into pre-zeroed cells.
            for (int i = 0; i < 32; i++) outBuf[i] = 0;

            ptrs[0] = ap;       ptrs[1] = wp;
            ptrs[2] = numCell;  ptrs[3] = sclCell;
            strs[0] = elemSize; strs[1] = elemSize;
            strs[2] = 0;        strs[3] = 0;  // pinned outputs → kernel's SIMD fast path
            kernel(ptrs, strs, count, null);

            // Scalar zero check + divide — primitive arithmetic on the stack cells.
            bool isZero = IsScalarZero(numCell: sclCell, resultDtype);
            if (isZero)
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");

            // Build the result NDArrays from the stack cells and return.
            avg = ScalarDivideToNDArray(numCell, sclCell, resultDtype);
            scl = BuildScalarNDArray(sclCell, resultDtype);
            return true;
        }

        // Compute (num / scl) as a primitive scalar and wrap as a 0-D NDArray.
        // The typeof switch is JIT-folded per-call; pointer derefs avoid the
        // NDArray binary-op machinery (~3 μs for a 0-D divide on the heavy path).
        private static unsafe NDArray ScalarDivideToNDArray(byte* numCell, byte* sclCell, NPTypeCode dt)
        {
            switch (dt)
            {
                case NPTypeCode.Double:  return NDArray.Scalar(*(double*)numCell / *(double*)sclCell);
                case NPTypeCode.Single:  return NDArray.Scalar(*(float*) numCell / *(float*) sclCell);
                case NPTypeCode.Int32:   return NDArray.Scalar(*(int*)   numCell / *(int*)   sclCell);
                case NPTypeCode.Int64:   return NDArray.Scalar(*(long*)  numCell / *(long*)  sclCell);
                case NPTypeCode.UInt32:  return NDArray.Scalar(*(uint*)  numCell / *(uint*)  sclCell);
                case NPTypeCode.UInt64:  return NDArray.Scalar(*(ulong*) numCell / *(ulong*) sclCell);
                case NPTypeCode.Int16:   return NDArray.Scalar((short) (*(short*) numCell / *(short*) sclCell));
                case NPTypeCode.UInt16:  return NDArray.Scalar((ushort)(*(ushort*)numCell / *(ushort*)sclCell));
                case NPTypeCode.Byte:    return NDArray.Scalar((byte)  (*(byte*)  numCell / *(byte*)  sclCell));
                case NPTypeCode.SByte:   return NDArray.Scalar((sbyte) (*(sbyte*) numCell / *(sbyte*) sclCell));
                default: throw new NotSupportedException($"Scalar divide for {dt} not in fast path");
            }
        }

        // Wrap a stack cell as a 0-D NDArray. NumSharp's NDArray.Scalar copies the
        // value into freshly-allocated unmanaged storage.
        private static unsafe NDArray BuildScalarNDArray(byte* cell, NPTypeCode dt) => dt switch
        {
            NPTypeCode.Double => NDArray.Scalar(*(double*)cell),
            NPTypeCode.Single => NDArray.Scalar(*(float*)cell),
            NPTypeCode.Int32  => NDArray.Scalar(*(int*)cell),
            NPTypeCode.Int64  => NDArray.Scalar(*(long*)cell),
            NPTypeCode.UInt32 => NDArray.Scalar(*(uint*)cell),
            NPTypeCode.UInt64 => NDArray.Scalar(*(ulong*)cell),
            NPTypeCode.Int16  => NDArray.Scalar(*(short*)cell),
            NPTypeCode.UInt16 => NDArray.Scalar(*(ushort*)cell),
            NPTypeCode.Byte   => NDArray.Scalar(*(byte*)cell),
            NPTypeCode.SByte  => NDArray.Scalar(*(sbyte*)cell),
            _ => throw new NotSupportedException($"BuildScalarNDArray for {dt} not in fast path")
        };

        // Scalar zero check — replaces the full np.any(scl == NDArray.Scalar(0))
        // pipeline (~9 μs at n=100) with a single pointer-deref comparison.
        private static unsafe bool IsScalarZero(byte* numCell, NPTypeCode dt) => dt switch
        {
            NPTypeCode.Double => *(double*)numCell == 0.0,
            NPTypeCode.Single => *(float*)numCell == 0f,
            NPTypeCode.Int32  => *(int*)numCell == 0,
            NPTypeCode.Int64  => *(long*)numCell == 0L,
            NPTypeCode.UInt32 => *(uint*)numCell == 0u,
            NPTypeCode.UInt64 => *(ulong*)numCell == 0UL,
            NPTypeCode.Int16  => *(short*)numCell == 0,
            NPTypeCode.UInt16 => *(ushort*)numCell == 0,
            NPTypeCode.Byte   => *(byte*)numCell == 0,
            NPTypeCode.SByte  => *(sbyte*)numCell == 0,
            _ => throw new NotSupportedException($"IsScalarZero for {dt} not in fast path")
        };

        // Fused weighted sum via DirectILKernelGenerator-cached kernel + NDIter.
        //
        // Setup: 4-operand iter [a, w, num_out, scl_out] with op_axes encoding
        // the reduction axes as -1 for the writable operands. EXTERNAL_LOOP +
        // REDUCE_OK gives the kernel `count == inner-axis size` with output
        // pointers pinned (stride==0) along the reduction axis — the kernel's
        // pinned-output fast path then runs a tight 4×-unrolled SIMD loop.
        // Single source of truth for dtype dispatch lives inside
        // DirectILKernelGenerator.GetWeightedSumIterKernel; this method is dtype-
        // agnostic at the call site.
        private static bool TryFusedWeightedSum(
            NDArray a, NDArray w, int[] axes, NPTypeCode resultDtype,
            out NDArray num, out NDArray scl)
        {
            NDInnerLoopFunc kernel = DirectILKernelGenerator.GetWeightedSumIterKernel(
                new DirectILKernelGenerator.WeightedSumKernelKey(resultDtype));
            if (kernel is null)
            {
                num = null;
                scl = null;
                return false;
            }

            int ndim = a.ndim;
            bool reduceAll = axes is null || axes.Length == ndim;

            // Output shape: a.shape with reduce axes removed. axis=None / reduce-all
            // collapses to a 0-D scalar.
            long[] outShape;
            if (reduceAll)
            {
                outShape = Array.Empty<long>();
            }
            else
            {
                outShape = new long[ndim - axes.Length];
                int oi = 0;
                for (int i = 0; i < ndim; i++)
                    if (Array.IndexOf(axes, i) < 0) outShape[oi++] = a.shape[i];
            }

            num = np.zeros(outShape.Length == 0 ? Shape.Scalar : new Shape(outShape), resultDtype);
            scl = np.zeros(outShape.Length == 0 ? Shape.Scalar : new Shape(outShape), resultDtype);

            // Pre-broadcast w to a's shape so both inputs share the iter shape.
            NDArray wBcast = w.shape.SequenceEqual(a.shape) ? w : np.broadcast_to(w, a.Shape);

            // op_axes: identity for a/w; -1 in reduce axes for num/scl.
            int[] aAxes = new int[ndim];
            int[] wAxes = new int[ndim];
            int[] numAxes = new int[ndim];
            int outAxisCounter = 0;
            for (int i = 0; i < ndim; i++)
            {
                aAxes[i] = i;
                wAxes[i] = i;
                bool isReduced = reduceAll || Array.IndexOf(axes, i) >= 0;
                numAxes[i] = isReduced ? -1 : outAxisCounter++;
            }

            using var iter = NDIterRef.AdvancedNew(
                4,
                new[] { a, wBcast, num, scl },
                NDIterGlobalFlags.REDUCE_OK | NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[]
                {
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READWRITE,
                    NDIterPerOpFlags.READWRITE,
                },
                null,
                ndim,
                new[] { aAxes, wAxes, numAxes, numAxes });

            unsafe { iter.ForEach(kernel); }
            return true;
        }

        private static int[] NormalizeAxisTuple(int[] axis, int ndim)
        {
            if (axis is null) return null;

            int[] normalized = new int[axis.Length];
            for (int i = 0; i < axis.Length; i++)
            {
                int ax = axis[i];
                if (ax < 0) ax += ndim;
                if (ax < 0 || ax >= ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis),
                        $"axis {axis[i]} is out of bounds for array of dimension {ndim}");
                normalized[i] = ax;
            }

            // Duplicate detection
            int[] sorted = (int[])normalized.Clone();
            Array.Sort(sorted);
            for (int i = 1; i < sorted.Length; i++)
                if (sorted[i] == sorted[i - 1])
                    throw new ArgumentException("duplicate value in 'axis'");

            return normalized;
        }

        private static NDArray MeanWithAxes(NDArray a, int[] axes, bool keepdims)
        {
            if (axes is null) return np.mean(a, keepdims);
            if (axes.Length == 1) return np.mean(a, axes[0], keepdims);

            int[] sortedDesc = (int[])axes.Clone();
            Array.Sort(sortedDesc);
            Array.Reverse(sortedDesc);

            NDArray cur = a;
            foreach (int ax in sortedDesc)
                cur = np.mean(cur, ax, keepdims: true);

            if (!keepdims)
            {
                int ndim = a.ndim;
                var kept = new List<long>(ndim - axes.Length);
                for (int i = 0; i < ndim; i++)
                {
                    if (Array.IndexOf(axes, i) < 0)
                        kept.Add(a.shape[i]);
                }
                cur = cur.reshape(kept.ToArray());
            }

            return cur;
        }

        private static NDArray SumWithAxes(NDArray a, int[] axes, NPTypeCode dtype, bool keepdims)
        {
            if (axes is null)
                return np.sum(a, axis: null, keepdims: keepdims, typeCode: dtype);

            if (axes.Length == 0)
            {
                NDArray asd = a.typecode == dtype ? a : a.astype(dtype);
                return asd;
            }

            if (axes.Length == 1)
                return np.sum(a, axis: axes[0], keepdims: keepdims, typeCode: dtype);

            int[] sortedDesc = (int[])axes.Clone();
            Array.Sort(sortedDesc);
            Array.Reverse(sortedDesc);

            NDArray cur = a.typecode == dtype ? a : a.astype(dtype);
            foreach (int ax in sortedDesc)
                cur = np.sum(cur, axis: ax, keepdims: true, typeCode: dtype);

            if (!keepdims)
            {
                int ndim = a.ndim;
                var kept = new List<long>(ndim - axes.Length);
                for (int i = 0; i < ndim; i++)
                {
                    if (Array.IndexOf(axes, i) < 0)
                        kept.Add(a.shape[i]);
                }
                cur = cur.reshape(kept.ToArray());
            }

            return cur;
        }

        // Mirrors numpy's lib/_function_base_impl._weights_are_valid.
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            NDArray wgt = weights;
            if (a.shape.SequenceEqual(wgt.shape))
                return wgt;

            if (axis is null)
                throw new ArgumentException(
                    "Axis must be specified when shapes of a and weights differ.");

            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.shape[axis[i]];

            if (!wgt.shape.SequenceEqual(expected))
                throw new ArgumentException(
                    "Shape of weights must be consistent with shape of a along specified axis.");

            // wgt = wgt.transpose(np.argsort(axis))
            if (axis.Length > 1)
            {
                int[] perm = ArgSort(axis);
                bool identity = true;
                for (int i = 0; i < perm.Length; i++)
                    if (perm[i] != i) { identity = false; break; }
                if (!identity)
                    wgt = wgt.transpose(perm);
            }

            int ndim = a.ndim;
            long[] newShape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                newShape[i] = Array.IndexOf(axis, i) >= 0 ? a.shape[i] : 1L;
            wgt = wgt.reshape(newShape);
            return wgt;
        }

        private static int[] ArgSort(int[] arr)
        {
            int[] idx = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++) idx[i] = i;
            Array.Sort(idx, (x, y) => arr[x].CompareTo(arr[y]));
            return idx;
        }

        private static NPTypeCode ComputeResultDtype(NPTypeCode aType, NPTypeCode wType)
        {
            NPTypeCode common = _FindCommonArrayType(aType, wType);
            if (IsIntegralOrBool(aType))
                return _FindCommonArrayType(common, NPTypeCode.Double);
            return common;
        }

        private static bool IsIntegralOrBool(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        // Dtype-generic zero-detection. Mirrors numpy's `np.any(scl == 0.0)` — uses
        // DirectILKernelGenerator-backed equality + np.any (vacuous-false on empty input).
        // Works for Half/Complex/Decimal where Convert.ToDouble fails (no IConvertible).
        private static bool HasZero(NDArray scl)
        {
            if (scl.size == 0) return false;
            NDArray zero = NDArray.Scalar(0).astype(scl.typecode);
            return np.any(scl == zero);
        }
    }
}
