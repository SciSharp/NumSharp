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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray average(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return AverageOfMatrix(a, axis, weights, keepdims);
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     Compute the weighted average along a tuple of axes. Equivalent to
        ///     <c>np.average(a, axis, weights, keepdims)</c> in NumPy with a tuple axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray average(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            // a numpy.matrix operand: the matrix with every reduced axis kept, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return AverageOfMatrix(a, axis, weights, keepdims);
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     <c>np.average</c> of a <c>numpy.matrix</c> along one axis or none: NumPy calls <c>a.mean(axis)</c>
        ///     (<c>np.multiply(a, w).sum(axis)</c> with weights), the matrix method, so the result is the scalar
        ///     without an axis, a 1 x N row for axis 0 and an N x 1 column for axis 1.
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axis, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is refused as NumPy's matrix method refuses it.</param>
        /// <returns>The average with NumPy's class and shape.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is <c>true</c>: <c>matrix.mean() got an
        ///     unexpected keyword argument 'keepdims'</c> (<c>matrix.sum()</c> with weights).</exception>
        private static NDArray AverageOfMatrix(NDArray a, int? axis, NDArray weights, bool keepdims)
            => NDMatrixBoundary.Reduced((a, axis, weights, keepdims), static s => average(s.a, s.axis, s.weights, s.keepdims),
                axis, keepdims, weights is null ? "mean" : "sum");

        /// <summary>
        ///     <c>np.average</c> of a <c>numpy.matrix</c> over a tuple of axes: a matrix reduction over a TUPLE
        ///     keeps every reduced axis with length 1 (<c>matrix._collapse</c> returns the kept result for any axis
        ///     that is not None; probed 2.4.2: <c>np.average(m, axis=(0, 1))</c> is a 1 x 1 matrix). Without axes
        ///     it is the scalar of the single-axis form.
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axes, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is refused as NumPy's matrix method refuses it.</param>
        /// <returns>The average with NumPy's class and shape.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is <c>true</c>: <c>matrix.mean() got an
        ///     unexpected keyword argument 'keepdims'</c> (<c>matrix.sum()</c> with weights).</exception>
        private static NDArray AverageOfMatrix(NDArray a, int[] axis, NDArray weights, bool keepdims)
        {
            string method = weights is null ? "mean" : "sum";
            if (axis is null)
                return NDMatrixBoundary.Reduced((a, weights, keepdims), static s => average(s.a, (int[])null, s.weights, s.keepdims), null, keepdims, method);
            if (keepdims)
                throw NDMatrixBoundary.KeywordError(method, "keepdims");
            return NDMatrixBoundary.Wrapped((a, axis, weights), static s => average(s.a, s.axis, s.weights, keepdims: true));
        }

        /// <summary>
        ///     The pair <c>np.average(m, ..., returned=True)</c> gives for a <c>numpy.matrix</c> operand: the average
        ///     NumPy's matrix rule returned, and the sum of weights as a PLAIN array of the average's shape
        ///     (NumPy: <c>if scl.shape != avg.shape: scl = np.broadcast_to(scl, avg.shape).copy()</c>; probed
        ///     2.4.2: <c>np.average(m, axis=0, returned=True)</c> is a 1 x N matrix and a 1 x N ndarray).
        /// </summary>
        /// <param name="avg">The average, already computed through <c>np.average</c>'s matrix rule (a matrix, or a
        ///     0-d scalar without an axis). Returned as it is.</param>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axes, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <returns>The average and the sum of weights.</returns>
        /// <remarks>
        ///     The sum of weights comes from a second, plain computation (the matrix path pays one extra pass; the
        ///     plain path is untouched). A matrix <paramref name="weights"/> makes NumPy's sum of weights a matrix
        ///     as well; here it is always a plain array.
        /// </remarks>
        private static (NDArray avg, NDArray sumOfWeights) AverageReturnedOfMatrix(NDArray avg, NDArray a, int[] axis, NDArray weights)
        {
            using var plain = NDMatrixBoundary.Inside();   // the calls below compute plain: this method shapes the pair itself
            var (plainAvg, scl) = AverageCore(a, axis, weights, keepdims: false, returned: true);
            plainAvg.Dispose();   // only the sum of weights is wanted from this pass
            if (scl.Shape.Equals(avg.Shape) || SameDimensions(scl.Shape, avg.Shape))
                return (avg, scl);

            // Same elements in the average's shape: a reshape when the counts agree (a row or a column of sums),
            // else NumPy's broadcast (the element count repeated for every output cell).
            Shape target = Shape.Adopt((long[])avg.Shape.dimensions.Clone());
            NDArray view = scl.size == avg.size ? scl.reshape(target) : np.broadcast_to(scl, target);
            NDArray shaped = view.copy();
            if (!ReferenceEquals(view, scl))
                view.Dispose();
            scl.Dispose();
            return (avg, shaped);
        }

        /// <summary>Whether two shapes have the same dimensions (layout and flags apart).</summary>
        /// <param name="x">The first shape.</param>
        /// <param name="y">The second shape.</param>
        /// <returns>True when rank and every extent agree.</returns>
        private static bool SameDimensions(Shape x, Shape y)
        {
            long[] dx = x.dimensions ?? Array.Empty<long>(), dy = y.dimensions ?? Array.Empty<long>();
            if (dx.Length != dy.Length)
                return false;
            for (int i = 0; i < dx.Length; i++)
                if (dx[i] != dy[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     Compute the weighted average and return a tuple <c>(avg, sum_of_weights)</c>.
        ///     Equivalent to <c>numpy.average(..., returned=True)</c>. When <paramref name="weights"/>
        ///     is null, <c>sum_of_weights</c> is the number of elements per output cell
        ///     (broadcast to the average's shape).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (NDMatrixBoundary.Sees(a))   // a numpy.matrix operand: np.average's matrix rule, then NumPy's sum of weights
                return AverageReturnedOfMatrix(AverageOfMatrix(a, axis, weights, keepdims), a, axis.HasValue ? new[] { axis.Value } : null, weights);
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: true);
        }

        /// <summary>
        ///     Tuple-axis overload of <see cref="average_returned(NDArray, int?, NDArray, bool)"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (NDMatrixBoundary.Sees(a))   // a numpy.matrix operand: np.average's matrix rule, then NumPy's sum of weights
                return AverageReturnedOfMatrix(AverageOfMatrix(a, axis, weights, keepdims), a, axis, weights);
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: true);
        }

        private static (NDArray avg, NDArray sumOfWeights) AverageCore(
            NDArray a, int[] axis, NDArray weights, bool keepdims, bool returned)
        {
            // Boundary scope: reclaims the dtype casts, the fused-kernel numerators, the
            // weighted products and the broadcast/scalar temps; each tuple component is
            // yielded individually (a dropped scl on the returned:false paths is reclaimed).
            using var scope = NDScope.Open();

            // Text: NumPy's average normalizes the axis tuple first (normalize_axis_tuple with argname "axis": "axis: axis 9
            // is out of bounds ...", "repeated axis in `axis` argument"), then a.mean (add.reduce) refuses text — or, with
            // weights, sums them with result_type(a, weights), a SIZED string descriptor ufuncs refuse by class.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.AverageAnswer(a, axis, weights);

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

                if (!returned) return (scope.Returns(avg), null);

                double count = (double)a.size / avg.size;
                NDArray scl = NDArray.Scalar(count).astype(avg.typecode);
                if (!scl.Shape.dimensions.SequenceEqual(avg.Shape.dimensions))
                    scl = np.broadcast_to(scl, avg).copy();
                return (scope.Returns(avg), scope.Returns(scl));
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
                    avgScalar = KeepdimsReshape(avgScalar, a.Shape.dimensions, normalizedAxis);
                    sclScalar = KeepdimsReshape(sclScalar, a.Shape.dimensions, normalizedAxis);
                }
                return (scope.Returns(avgScalar), returned ? scope.Returns(sclScalar) : null);
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
                    avgFast = KeepdimsReshape(avgFast, a.Shape.dimensions, normalizedAxis);
                    sclFast = KeepdimsReshape(sclFast, a.Shape.dimensions, normalizedAxis);
                }
                if (!returned) return (scope.Returns(avgFast), null);
                if (!sclFast.Shape.dimensions.SequenceEqual(avgFast.Shape.dimensions))
                    sclFast = np.broadcast_to(sclFast, avgFast).copy();
                return (scope.Returns(avgFast), scope.Returns(sclFast));
            }

            // Fallback path for dtypes the IL kernel doesn't cover.
            NDArray scl_ = SumWithAxes(wgtCast, normalizedAxis, resultDtype, keepdims);
            if (HasZero(scl_))
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");
            NDArray prod = aCast * wgtCast;
            NDArray num = SumWithAxes(prod, normalizedAxis, resultDtype, keepdims);
            NDArray avg_ = num / scl_;
            if (!returned) return (scope.Returns(avg_), null);
            if (!scl_.Shape.dimensions.SequenceEqual(avg_.Shape.dimensions))
                scl_ = np.broadcast_to(scl_, avg_).copy();
            return (scope.Returns(avg_), scope.Returns(scl_));
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
                else kd[i] = reduced.Shape.dimensions[outIdx++];
            }
            return reduced.reshape(Shape.Adopt(kd));
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
                    if (Array.IndexOf(axes, i) < 0) outShape[oi++] = a.Shape.dimensions[i];
            }

            num = np.zeros(outShape.Length == 0 ? Shape.Scalar : Shape.Adopt(outShape), resultDtype);
            scl = np.zeros(outShape.Length == 0 ? Shape.Scalar : Shape.Adopt(outShape), resultDtype);

            // Pre-broadcast w to a's shape so both inputs share the iter shape.
            NDArray wBcast = w.Shape.dimensions.SequenceEqual(a.Shape.dimensions) ? w : np.broadcast_to(w, a.Shape);

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
                        kept.Add(a.Shape.dimensions[i]);
                }
                cur = cur.reshape(Shape.Adopt(kept.ToArray()));
            }

            return cur;
        }

        private static NDArray SumWithAxes(NDArray a, int[] axes, NPTypeCode dtype, bool keepdims)
        {
            if (axes is null)
                return np.sum(a, axis: null, keepdims: keepdims, dtype: dtype);

            if (axes.Length == 0)
            {
                NDArray asd = a.typecode == dtype ? a : a.astype(dtype);
                return asd;
            }

            if (axes.Length == 1)
                return np.sum(a, axis: axes[0], keepdims: keepdims, dtype: dtype);

            int[] sortedDesc = (int[])axes.Clone();
            Array.Sort(sortedDesc);
            Array.Reverse(sortedDesc);

            NDArray cur = a.typecode == dtype ? a : a.astype(dtype);
            foreach (int ax in sortedDesc)
                cur = np.sum(cur, axis: ax, keepdims: true, dtype: dtype);

            if (!keepdims)
            {
                int ndim = a.ndim;
                var kept = new List<long>(ndim - axes.Length);
                for (int i = 0; i < ndim; i++)
                {
                    if (Array.IndexOf(axes, i) < 0)
                        kept.Add(a.Shape.dimensions[i]);
                }
                cur = cur.reshape(Shape.Adopt(kept.ToArray()));
            }

            return cur;
        }

        // Mirrors numpy's lib/_function_base_impl._weights_are_valid.
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            NDArray wgt = weights;
            if (a.Shape.dimensions.SequenceEqual(wgt.Shape.dimensions))
                return wgt;

            if (axis is null)
                throw new ArgumentException(
                    "Axis must be specified when shapes of a and weights differ.");

            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.Shape.dimensions[axis[i]];

            if (!wgt.Shape.dimensions.SequenceEqual(expected))
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
                newShape[i] = Array.IndexOf(axis, i) >= 0 ? a.Shape.dimensions[i] : 1L;
            wgt = wgt.reshape(Shape.Adopt(newShape));
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
        [NDScoped] // bool boundary: the zero scalar, its astype and the == mask are all reclaimed here
        private static bool HasZero(NDArray scl)
        {
            if (scl.size == 0) return false;
            NDArray zero = NDArray.Scalar(0).astype(scl.typecode);
            return np.any(scl == zero);
        }
    }
}
