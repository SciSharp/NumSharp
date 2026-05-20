using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Equivalent to <see cref="nonzero(NDArray)"/>: returns the indices where
        ///     <paramref name="condition"/> is non-zero.
        /// </summary>
        /// <param name="condition">Input array. Non-zero entries yield their indices.</param>
        /// <returns>Tuple of arrays with indices where condition is non-zero, one per dimension.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray<long>[] where(NDArray condition)
        {
            return nonzero(condition);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        /// </summary>
        /// <param name="condition">Where True, yield `x`, otherwise yield `y`.</param>
        /// <param name="x">Values from which to choose where condition is True.</param>
        /// <param name="y">Values from which to choose where condition is False.</param>
        /// <returns>An array with elements from `x` where `condition` is True, and elements from `y` elsewhere.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.where.html</remarks>
        public static NDArray where(NDArray condition, NDArray x, NDArray y)
        {
            return where_internal(condition, x, y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for x.
        /// </summary>
        public static NDArray where(NDArray condition, object x, NDArray y)
        {
            return where_internal(condition, asanyarray(x), y);
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for y.
        /// </summary>
        public static NDArray where(NDArray condition, NDArray x, object y)
        {
            return where_internal(condition, x, asanyarray(y));
        }

        /// <summary>
        ///     Return elements chosen from `x` or `y` depending on `condition`.
        ///     Scalar overload for both x and y.
        /// </summary>
        public static NDArray where(NDArray condition, object x, object y)
        {
            return where_internal(condition, asanyarray(x), asanyarray(y));
        }

        /// <summary>
        /// Internal implementation of np.where.
        /// </summary>
        private static NDArray where_internal(NDArray condition, NDArray x, NDArray y)
        {
            // Detect "originally scalar" on the user-supplied operands BEFORE broadcasting
            // expands them into stride-0 views. The scalar fast path below dispatches
            // specialised IL kernels that hoist the scalar into Vector.Create<T>(value) once
            // outside the loop — avoiding the per-element broadcast view dereference that the
            // NpyIter expression kernel would otherwise perform.
            bool xIsScalar = x.size == 1;
            bool yIsScalar = y.size == 1;

            // Skip broadcast_arrays (which allocates 3 NDArrays + helper arrays) when all three
            // already share a shape — the frequent case of np.where(mask, arr, other_arr).
            NDArray cond, xArr, yArr;
            if (condition.Shape == x.Shape && x.Shape == y.Shape)
            {
                cond = condition;
                xArr = x;
                yArr = y;
            }
            else
            {
                var broadcasted = broadcast_arrays(condition, x, y);
                cond = broadcasted[0];
                xArr = broadcasted[1];
                yArr = broadcasted[2];
            }

            // Resolve output layout before dtype casts. Casts of broadcasted scalars can
            // materialize C-contiguous temporaries, but NumPy's iterator ignores those for
            // output-order selection.
            char resultOrder = ResolveWhereOrder(cond, xArr, yArr);

            // Coerce the condition to boolean using NumPy's truthiness rules
            // (0/0.0 → False, everything else including NaN/±Inf → True). The
            // iterator-driven expression kernel requires a bool condition dtype.
            if (cond.GetTypeCode != NPTypeCode.Boolean)
                cond = cond.astype(NPTypeCode.Boolean, copy: false);

            // When x and y already agree, skip the NEP50 promotion lookup. Otherwise defer to
            // _FindCommonType which handles the scalar+array NEP50 rules.
            var outType = x.GetTypeCode == y.GetTypeCode
                ? x.GetTypeCode
                : _FindCommonType(x, y);

            if (xArr.GetTypeCode != outType)
                xArr = xArr.astype(outType, copy: false);
            if (yArr.GetTypeCode != outType)
                yArr = yArr.astype(outType, copy: false);

            // Use cond.shape (dimensions only) not cond.Shape (which may have broadcast strides).
            // NumPy's iterator allocation preserves F-order when all full-size operands agree
            // on F layout; any full C-contiguous or non-contiguous operand falls back to C.
            var result = empty(new Shape((long[])cond.shape.Clone(), resultOrder), outType);

            // Handle empty arrays - nothing to iterate
            if (result.size == 0)
                return result;

            // -----------------------------------------------------------------
            // Scalar-broadcast IL fast path
            // -----------------------------------------------------------------
            // When x or y was a Python literal / 0-d / size-1 array, broadcast_arrays expanded
            // it into a stride-0 view that fails the IsContiguous gate below. Instead of
            // materializing that view into a full contig copy (NpyIter's behaviour) we read
            // the single value, cast it to outType, and dispatch a kernel that broadcasts via
            // V<T>.Create(value) once outside the SIMD loop.
            //
            // The non-scalar operand must be contig (its shape already matches the result
            // because of the broadcast above). Two scalars + contig cond is also covered.
            if (ILKernelGenerator.Enabled &&
                cond.typecode == NPTypeCode.Boolean &&
                cond.Shape.IsContiguous &&
                (xIsScalar || yIsScalar))
            {
                // Promote scalars to outType once (cheap — these are 1-element NDArrays).
                // For each, use the ORIGINAL operand (x/y) so we don't rely on the broadcast
                // view; the cast yields a fresh 1-element NDArray of outType.
                NDArray xScalarSrc = xIsScalar
                    ? (x.GetTypeCode != outType ? x.astype(outType) : x)
                    : null;
                NDArray yScalarSrc = yIsScalar
                    ? (y.GetTypeCode != outType ? y.astype(outType) : y)
                    : null;

                if (xIsScalar && yIsScalar)
                {
                    WhereScalarXYDispatch(cond, xScalarSrc, yScalarSrc, result, outType);
                    return result;
                }
                if (xIsScalar && yArr.Shape.IsContiguous)
                {
                    WhereScalarXDispatch(cond, xScalarSrc, yArr, result, outType);
                    return result;
                }
                if (yIsScalar && xArr.Shape.IsContiguous)
                {
                    WhereScalarYDispatch(cond, xArr, yScalarSrc, result, outType);
                    return result;
                }
            }

            // IL Kernel fast path: all arrays contiguous, bool condition, SIMD enabled
            // Broadcasted arrays (stride=0) are NOT contiguous, so they use iterator path.
            bool canUseKernel = ILKernelGenerator.Enabled &&
                                cond.typecode == NPTypeCode.Boolean &&
                                cond.Shape.IsContiguous &&
                                xArr.Shape.IsContiguous &&
                                yArr.Shape.IsContiguous;

            if (canUseKernel)
            {
                WhereKernelDispatch(cond, xArr, yArr, result, outType);
                return result;
            }

            // Iterator fallback for non-contiguous/broadcasted arrays.
            WhereImpl(cond, xArr, yArr, result);

            return result;
        }

        private static void WhereImpl(NDArray cond, NDArray x, NDArray y, NDArray result)
        {
            // Drive cond + x + y + result in lockstep via a 4-operand NpyIter
            // compiling Where(cond, x, y) → out as a single IL expression kernel.
            // C-order traversal matches NumPy element semantics; WRITEONLY on
            // the output lets the iterator allocate per-inner-loop buffer space
            // when casting is needed.
            var dtype = result.GetTypeCode;
            using var iter = NpyIterRef.MultiNew(
                4, new[] { cond, x, y, result },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2));
            iter.ExecuteExpression(
                expr,
                new[] { NPTypeCode.Boolean, dtype, dtype },
                dtype,
                cacheKey: $"np.where.{dtype}");
        }

        private static char ResolveWhereOrder(params NDArray[] operands)
        {
            bool sawStrictF = false;

            foreach (var operand in operands)
            {
                var shape = operand.Shape;

                // Scalar, 1-D, and broadcasted operands don't force the output layout.
                if (shape.IsScalar || shape.NDim <= 1 || shape.IsBroadcasted)
                    continue;

                bool isC = shape.IsContiguous;
                bool isF = shape.IsFContiguous;

                if (isC && !isF)
                    return 'C';

                if (isF && !isC)
                {
                    sawStrictF = true;
                    continue;
                }

                if (!isC && !isF)
                    return 'C';
            }

            return sawStrictF ? 'F' : 'C';
        }

        /// <summary>
        /// IL Kernel dispatch for contiguous arrays.
        /// Uses IL-generated kernels with SIMD optimization.
        /// </summary>
        private static unsafe void WhereKernelDispatch(NDArray cond, NDArray x, NDArray y, NDArray result, NPTypeCode outType)
        {
            var condPtr = (nint)cond.Address;
            var count = result.size;

            NpFunc.Invoke(outType, WhereKernelExecute<int>, condPtr, (nint)x.Address, (nint)y.Address, (nint)result.Address, count);
        }

        private static unsafe void WhereKernelExecute<T>(nint condPtr, nint xAddr, nint yAddr, nint resultAddr, long count) where T : unmanaged
            => ILKernelGenerator.WhereExecute((bool*)condPtr, (T*)xAddr, (T*)yAddr, (T*)resultAddr, count);

        // -----------------------------------------------------------------
        // Scalar-broadcast dispatch
        // -----------------------------------------------------------------
        // Reads the scalar value from the 1-element NDArray (already promoted to outType)
        // and invokes the appropriate IL kernel. Returns true on success; false if no IL
        // kernel was available (caller falls back to the NpyIter path).

        private static unsafe void WhereScalarXDispatch(NDArray cond, NDArray xScalar, NDArray y, NDArray result, NPTypeCode outType)
        {
            // Attempt the IL kernel; if it returns false (no SIMD / unsupported dtype),
            // materialize the scalar to a broadcast view of cond's shape and fall back
            // to the existing NpyIter expression path.
            bool ok = NpFunc.Invoke(outType, TryWhereScalarXExecute<int>,
                (nint)cond.Address, (nint)xScalar.Address, (nint)y.Address, (nint)result.Address, result.size);
            if (ok) return;

            var xBroadcast = broadcast_to(xScalar, cond.Shape);
            WhereImpl(cond, xBroadcast, y, result);
        }

        private static unsafe void WhereScalarYDispatch(NDArray cond, NDArray x, NDArray yScalar, NDArray result, NPTypeCode outType)
        {
            bool ok = NpFunc.Invoke(outType, TryWhereScalarYExecute<int>,
                (nint)cond.Address, (nint)x.Address, (nint)yScalar.Address, (nint)result.Address, result.size);
            if (ok) return;

            var yBroadcast = broadcast_to(yScalar, cond.Shape);
            WhereImpl(cond, x, yBroadcast, result);
        }

        private static unsafe void WhereScalarXYDispatch(NDArray cond, NDArray xScalar, NDArray yScalar, NDArray result, NPTypeCode outType)
        {
            bool ok = NpFunc.Invoke(outType, TryWhereScalarXYExecute<int>,
                (nint)cond.Address, (nint)xScalar.Address, (nint)yScalar.Address, (nint)result.Address, result.size);
            if (ok) return;

            var xBroadcast = broadcast_to(xScalar, cond.Shape);
            var yBroadcast = broadcast_to(yScalar, cond.Shape);
            WhereImpl(cond, xBroadcast, yBroadcast, result);
        }

        private static unsafe bool TryWhereScalarXExecute<T>(nint condPtr, nint xScalarPtr, nint yPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = ILKernelGenerator.GetWhereScalarXKernel<T>();
            if (kernel == null) return false;

            T scalarX = *(T*)xScalarPtr;
            kernel((bool*)condPtr, scalarX, (T*)yPtr, (T*)resPtr, count);
            return true;
        }

        private static unsafe bool TryWhereScalarYExecute<T>(nint condPtr, nint xPtr, nint yScalarPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = ILKernelGenerator.GetWhereScalarYKernel<T>();
            if (kernel == null) return false;

            T scalarY = *(T*)yScalarPtr;
            kernel((bool*)condPtr, (T*)xPtr, scalarY, (T*)resPtr, count);
            return true;
        }

        private static unsafe bool TryWhereScalarXYExecute<T>(nint condPtr, nint xScalarPtr, nint yScalarPtr, nint resPtr, long count) where T : unmanaged
        {
            var kernel = ILKernelGenerator.GetWhereScalarXYKernel<T>();
            if (kernel == null) return false;

            T scalarX = *(T*)xScalarPtr;
            T scalarY = *(T*)yScalarPtr;
            kernel((bool*)condPtr, scalarX, scalarY, (T*)resPtr, count);
            return true;
        }
    }
}
