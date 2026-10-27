using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
// The moved statements call np's internal promotion table (_FindCommonArrayType) as they did inside np; importing np's
// statics keeps them verbatim.
using static NumSharp.np;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     NumPy's <c>numpy.average</c> (<c>numpy/lib/_function_base_impl.py</c>) after the facades' checks: the axis tuple
    ///     (<c>normalize_axis_tuple</c>), the mean without weights and the element count it stands for, and with weights
    ///     <c>_weights_are_valid</c>'s reshape, the result dtype, the weight sum and its zero check, the weighted sum and the
    ///     division, and <c>returned</c>'s weight sum in the average's shape; over text, NumPy's statements up to the
    ///     reduction that refuses it.
    /// </summary>
    /// <remarks>
    ///     An internal operation class (the np-layer dtype plan's D1: NumPy writes average in Python), moved from the public
    ///     <c>np.average</c> / <c>np.average_returned</c> by P2; the facades keep the numpy.matrix boundary and the null
    ///     check. It sits beside the facades, not under <c>Backends/</c>, because its statements call the public np.*
    ///     functions NumPy's call (np.mean, np.sum, np.broadcast_to, np.zeros, np.any), which ratchet R2 counts under
    ///     <c>Backends/</c>. Where the result dtype has a fused weighted-sum kernel
    ///     (<see cref="DirectILKernelGenerator.GetWeightedSumIterKernel"/>), the weighted sum and the weight sum are computed
    ///     in one pass in the kernel's summation order, not NumPy's pairwise add.reduce.
    /// </remarks>
    internal static class AverageEngine
    {
        /// <summary>
        ///     <c>average(a, axis, weights, returned, keepdims)</c> after the facades' checks: the average and, for
        ///     <paramref name="returned"/>, the sum of weights in the average's shape.
        /// </summary>
        /// <param name="a">The operand (not null): any dtype and layout; text takes NumPy's text statements.</param>
        /// <param name="axis">The axes as given (a negative axis counts from the end), or null for every axis.</param>
        /// <param name="weights">The weights, or null for the plain mean: of <paramref name="a"/>'s shape, or of the
        ///     reduced axes' extents in the order of <paramref name="axis"/>.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="returned">Also compute the sum of weights (NumPy's <c>returned=True</c>); false leaves the pair's
        ///     second element null.</param>
        /// <returns>The average and the sum of weights (null unless <paramref name="returned"/>): arrays the caller owns,
        ///     0-d for a reduction over every axis without keepdims.</returns>
        /// <exception cref="AxisError">An axis out of bounds (<c>axis: axis 9 is out of bounds for array of dimension
        ///     2</c>: NumPy names the argument).</exception>
        /// <exception cref="ValueError">An axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not
        ///     of the reduced axes' shape (<c>Shape of weights must be consistent with shape of a along specified
        ///     axis.</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero (<c>Weights sum to zero, can't be
        ///     normalized</c>), or a mean without weights whose result is empty (<c>division by zero</c>: NumPy's
        ///     <c>a.size / avg.size</c>).</exception>
        /// <exception cref="NotSupportedException">Text weights beside a numeric operand: the numeric promotion table
        ///     (<see cref="np._FindCommonArrayType(NPTypeCode, NPTypeCode)"/>) has no entry for text.</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>); a text operand: the mean's add.reduce refusal, or the weight sum's refusal of a
        ///     sized string or <c>StringDType</c> <c>dtype=</c>.</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> operand with weights it has no common DType
        ///     with.</exception>
        /// <exception cref="InvalidOperationException">Not reached: a <c>StringDType</c> weight sum NumPy refuses that
        ///     computed (a wiring error inside NumSharp).</exception>
        internal static (NDArray avg, NDArray sumOfWeights) Average(
            NDArray a, int[] axis, NDArray weights, bool keepdims, bool returned)
        {
            // Boundary scope: reclaims the dtype casts, the fused-kernel numerators, the
            // weighted products and the broadcast/scalar temps; each tuple component is
            // yielded individually (a dropped scl on the returned:false paths is reclaimed).
            using var scope = NDScope.Open();

            // Text: NumPy's average normalizes the axis tuple first (normalize_axis_tuple with argname "axis": "axis: axis 9
            // is out of bounds ...", "repeated axis in `axis` argument"), then a.mean (add.reduce) refuses text — or, with
            // weights, sums them with result_type(a, weights), a SIZED string descriptor ufuncs refuse by class.
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                // StringDType: the axis tuple normalized (average's argument name), then a.mean(axis) — or, with weights,
                // result_type and the weights' sum — computing until NumPy's statement refuses (NDVStringStatistics.Average).
                Backends.Strings.NDStringUfuncs.ValidateReduceAxes("average", a.ndim, axis);
                var avg = Backends.Strings.NDVStringStatistics.Average(a, Backends.Strings.NDStringUfuncs.ReducedAxes(a, axis),
                    axisGiven: axis is not null, weights, keepdims);
                if (!returned)
                    return (scope.Returns(avg), null);
                // scl = avg_as_array.dtype.type(a.size / avg_as_array.size): only a NaN-like NA mean (np.float64) returns.
                var sclValue = NDArray.Scalar((double)a.size / avg.size);
                return (scope.Returns(avg), scope.Returns(sclValue));
            }
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

        /// <summary>
        ///     Relabels a reduced result (the reduced axes dropped) with keepdims' shape: every reduced axis back with length
        ///     1, the kept axes in order.
        /// </summary>
        /// <param name="reduced">The reduced result: its dims are the kept axes' extents in order.</param>
        /// <param name="aShape">The operand's dims.</param>
        /// <param name="axes">The normalized reduced axes, or null for every axis (each becomes 1).</param>
        /// <returns>A reshape of <paramref name="reduced"/> (a view sharing its buffer: it does not own its data).</returns>
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

        /// <summary>
        ///     The weighted average over every axis of two C-contiguous operands of the result dtype, without an iterator or
        ///     an intermediate array: the cached weighted-sum kernel runs once into two stack cells (the weighted sum and the
        ///     weight sum), the zero check and the division are scalar arithmetic, and each answer is wrapped as a 0-d array.
        /// </summary>
        /// <param name="a">The operand, already of <paramref name="resultDtype"/> and C-contiguous.</param>
        /// <param name="w">The weights, of <paramref name="a"/>'s shape and dtype and C-contiguous.</param>
        /// <param name="resultDtype">The result dtype (the kernel's dtype).</param>
        /// <param name="avg">The average (<c>NDArray.Scalar</c>: a writeable 0-d owner), or null when declined.</param>
        /// <param name="scl">The weight sum (a writeable 0-d owner), or null when declined.</param>
        /// <returns>False when the dtype has no kernel (bool, char, float16, complex, decimal): the caller's iterator
        ///     route runs instead.</returns>
        /// <exception cref="DivideByZeroException">The weights sum to zero (<c>Weights sum to zero, can't be
        ///     normalized</c>).</exception>
        /// <remarks>
        ///     The kernel accumulates with its own SIMD order (several partial sums), not NumPy's pairwise add.reduce. The
        ///     integer cases of the division are never reached: <see cref="ComputeResultDtype"/> lifts an integer or bool
        ///     operand to float64, and a float operand keeps a float dtype.
        /// </remarks>
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

        /// <summary>
        ///     The quotient of two stack cells of one dtype as a 0-d array: C# arithmetic in that dtype (an integer dtype
        ///     divides as integers), without the NDArray binary-operator pipeline.
        /// </summary>
        /// <param name="numCell">The dividend's cell.</param>
        /// <param name="sclCell">The divisor's cell (not zero: the caller checked).</param>
        /// <param name="dt">The cells' dtype: one of the ten dtypes the weighted-sum kernel covers.</param>
        /// <returns>A writeable 0-d owner holding the quotient.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dt"/> is not one of the kernel's dtypes (a wiring
        ///     error: the caller only reaches here with a kernel).</exception>
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

        /// <summary>
        ///     A stack cell as a 0-d array (<c>NDArray.Scalar</c> copies the value into fresh unmanaged storage).
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="dt">The cell's dtype: one of the ten dtypes the weighted-sum kernel covers.</param>
        /// <returns>A writeable 0-d owner holding the cell's value.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dt"/> is not one of the kernel's dtypes (a wiring
        ///     error).</exception>
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

        /// <summary>
        ///     Whether a stack cell holds zero: NumPy's <c>np.any(scl == 0.0)</c> for a 0-d weight sum as one comparison
        ///     (a NaN is not zero; -0.0 is).
        /// </summary>
        /// <param name="numCell">The cell.</param>
        /// <param name="dt">The cell's dtype: one of the ten dtypes the weighted-sum kernel covers.</param>
        /// <returns>True when the value is zero.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dt"/> is not one of the kernel's dtypes (a wiring
        ///     error).</exception>
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

        /// <summary>
        ///     The weighted sum and the weight sum over <paramref name="axes"/> in one iterator pass: a four-operand NDIter
        ///     over <c>[a, w, num, scl]</c> whose outputs have stride 0 along the reduced axes (<c>op_axes</c> -1, REDUCE_OK,
        ///     EXTERNAL_LOOP) drives the cached weighted-sum kernel, which adds into the pre-zeroed outputs.
        /// </summary>
        /// <param name="a">The operand, already of <paramref name="resultDtype"/>.</param>
        /// <param name="w">The weights of <paramref name="resultDtype"/>: <paramref name="a"/>'s shape, or broadcastable to it
        ///     (the reduced axes' extents with 1 elsewhere, from <see cref="WeightsAreValid"/>).</param>
        /// <param name="axes">The normalized reduced axes, or null for every axis.</param>
        /// <param name="resultDtype">The result dtype (the kernel's dtype).</param>
        /// <param name="num">The weighted sum (a fresh array of the kept axes; 0-d for every axis), or null when
        ///     declined.</param>
        /// <param name="scl">The weight sum, of <paramref name="num"/>'s shape, or null when declined.</param>
        /// <returns>False when the dtype has no kernel (bool, char, float16, complex, decimal).</returns>
        /// <remarks>
        ///     The kernel's summation order is its own (SIMD partial sums along a reduced innermost axis, a running total
        ///     otherwise), not NumPy's pairwise add.reduce.
        /// </remarks>
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

        /// <summary>
        ///     np.average's first statement for an axis: <c>normalize_axis_tuple(axis, a.ndim, argname="axis")</c>.
        /// </summary>
        /// <param name="axis">The axes as given, or null for every axis.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns>The normalized axes in input order, or null for null.</returns>
        /// <exception cref="AxisError">An axis is out of range: "axis: axis … is out of bounds for array of dimension …"
        ///     (NumPy names the argument).</exception>
        /// <exception cref="ValueError">An axis appears twice: "repeated axis in `axis` argument".</exception>
        private static int[] NormalizeAxisTuple(int[] axis, int ndim)
        {
            if (axis is null) return null;
            return Backends.DefaultEngine.normalize_axis_tuple(axis, ndim, "axis");
        }

        /// <summary>
        ///     <c>a.mean(axis, keepdims)</c> over normalized axes: np.mean for none or one axis, else one np.mean per axis
        ///     from the last to the first (keepdims between them), reshaped to drop the reduced axes unless
        ///     <paramref name="keepdims"/>.
        /// </summary>
        /// <param name="a">The numeric operand.</param>
        /// <param name="axes">The normalized axes, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The mean. Over two or more axes it is a mean of means (NumPy sums the whole tuple once), and without
        ///     keepdims a reshape view of the last mean; over an empty tuple it is <paramref name="a"/> itself.</returns>
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

        /// <summary>
        ///     <c>a.sum(axis, dtype=dtype, keepdims)</c> over normalized axes: np.sum for none or one axis, the operand cast to
        ///     <paramref name="dtype"/> for an empty tuple, else one np.sum per axis from the last to the first (keepdims
        ///     between them), reshaped to drop the reduced axes unless <paramref name="keepdims"/>.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="axes">The normalized axes, or null for every axis.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The sum (over two or more axes a sum of sums, and without keepdims a reshape view of the last
        ///     one).</returns>
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

        /// <summary>
        ///     NumPy's <c>_weights_are_valid</c>: weights of the operand's shape as they are; otherwise the weights of the
        ///     reduced axes (of shape <c>tuple(a.shape[ax] for ax in axis)</c>), transposed by <c>np.argsort(axis)</c> and
        ///     reshaped so they broadcast against the operand (the reduced axes' extents, 1 elsewhere).
        /// </summary>
        /// <param name="weights">The weights (not null).</param>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The normalized axes, or null for every axis.</param>
        /// <returns><paramref name="weights"/> itself, or a view of it in the operand's rank.</returns>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>; NumPy's TypeError).</exception>
        /// <exception cref="ValueError">Weights not of the reduced axes' shape (<c>Shape of weights must be consistent with
        ///     shape of a along specified axis.</c>). An empty axis tuple is an axis given: weights of another shape then
        ///     meet this check, not the TypeError.</exception>
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            NDArray wgt = weights;
            if (a.Shape.dimensions.SequenceEqual(wgt.Shape.dimensions))
                return wgt;

            // NumPy's _weights_are_valid: `if axis is None: raise TypeError(...)` — the axis tuple as given (an empty
            // tuple is not None), then the shape of the reduced axes.
            if (axis is null)
                throw new TypeError("Axis must be specified when shapes of a and weights differ.");

            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.Shape.dimensions[axis[i]];

            if (!wgt.Shape.dimensions.SequenceEqual(expected))
                throw new ValueError("Shape of weights must be consistent with shape of a along specified axis.");

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

        /// <summary><c>np.argsort</c> of a short int array (the axis tuple): the positions in ascending value order.</summary>
        /// <param name="arr">The values (distinct: a normalized axis tuple).</param>
        /// <returns>The permutation that sorts <paramref name="arr"/>.</returns>
        private static int[] ArgSort(int[] arr)
        {
            int[] idx = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++) idx[i] = i;
            Array.Sort(idx, (x, y) => arr[x].CompareTo(arr[y]));
            return idx;
        }

        /// <summary>
        ///     The weighted average's result dtype: <c>result_type(a.dtype, wgt.dtype, 'f8')</c> for an integer or bool
        ///     operand, else <c>result_type(a.dtype, wgt.dtype)</c>, through the numeric promotion table.
        /// </summary>
        /// <param name="aType">The operand's dtype.</param>
        /// <param name="wType">The weights' dtype.</param>
        /// <returns>The result dtype.</returns>
        /// <exception cref="NotSupportedException">A text dtype: the numeric table has no entry for it.</exception>
        private static NPTypeCode ComputeResultDtype(NPTypeCode aType, NPTypeCode wType)
        {
            NPTypeCode common = _FindCommonArrayType(aType, wType);
            if (IsIntegralOrBool(aType))
                return _FindCommonArrayType(common, NPTypeCode.Double);
            return common;
        }

        /// <summary>NumPy's <c>issubclass(a.dtype.type, (np.integer, np.bool))</c>, with char as NumSharp's integer.</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>True for bool, the integer dtypes and char.</returns>
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

        /// <summary>
        ///     NumPy's <c>np.any(scl == 0.0)</c> for a weight sum of any dtype: the zero of its dtype, the equality and
        ///     np.any (false for an empty sum). Works for float16, complex and decimal, which have no
        ///     <c>Convert.ToDouble</c>.
        /// </summary>
        /// <param name="scl">The weight sum.</param>
        /// <returns>True when some element is zero.</returns>
        [NDScoped] // bool boundary: the zero scalar, its astype and the == mask are all reclaimed here
        private static bool HasZero(NDArray scl)
        {
            if (scl.size == 0) return false;
            NDArray zero = NDArray.Scalar(0).astype(scl.typecode);
            return np.any(scl == zero);
        }
    }
}
