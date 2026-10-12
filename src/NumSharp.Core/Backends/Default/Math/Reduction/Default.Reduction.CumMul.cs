using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.cumprod</c> (<c>np.multiply.accumulate</c>): the running products along
        ///     <paramref name="axis_"/>, or along the flattened array when it is null.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative values count from the end), or null for the flattened array. A
        ///     0-d input is treated as 1-D (NumPy's <c>PyArray_CheckAxis</c>), so only 0 and -1 address it.</param>
        /// <param name="dtype">The accumulator dtype (<c>dtype=</c>), or null for NEP 50's accumulator of the input.</param>
        /// <returns>A fresh array of the accumulator dtype: the input's shape with an axis, 1-D for the flattened form (a 0-d
        ///     input gives shape (1,)).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before every shortcut (empty,
        ///     0-d and one-element inputs included).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string input or a string <paramref name="dtype"/> that
        ///     NumPy computes on (the next phase of NumSharp's string support).</exception>
        public override unsafe NDArray ReduceCumMul(NDArray arr, int? axis_, DType dtype = null)
        {
            // Text, or a string dtype= request: NumPy's cumprod checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text or computes on text (the next phase).
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse), then reduces the numbers.
            if (Strings.NDStringUfuncs.ParseForNumericReduce("cumprod", arr, axis_, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => ReduceCumMul(p, axis_, dtype));
            Strings.NDStringUfuncs.ThrowIfReduce("cumprod", arr, axis_, dtype);
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // NumPy: cumprod on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                var int64Arr = arr.astype(NPTypeCode.Int64, copy: true);
                var cum = ReduceCumMul(int64Arr, axis_, (typeCode ?? NPTypeCode.Int64).AsType());
                int64Arr.Dispose();
                return cum;
            }

            var shape = arr.Shape;

            // PyArray_CheckAxis, before any shortcut: the axis addresses rank max(ndim, 1) (a 0-d input ravels to 1-D), so
            // cumprod([5], axis=-3) and cumprod(0-d, axis=1) are NumPy's AxisError, not the value itself. The shortcuts below
            // used to run first, and the later wrap-then-test loop accepted -ndim - 1 (reducing the last axis).
            int axisChecked = axis_ is null ? 0 : check_and_adjust_axis(Math.Max(arr.ndim, 1), axis_.Value);

            // Empty: cumprod returns a FRESH array of the accumulator dtype (NEP50 widening applies
            // even when empty — cumprod(empty int32) is int64, like cumsum). axis=None ravels to a
            // 1-D (0,) result; with an axis the shape is preserved. Returning `arr` unchanged (as
            // before) kept the source dtype AND, for a multi-dim source, the un-raveled shape —
            // e.g. cumprod((0,3)) gave (0,3) where NumPy gives (0,). Mirrors ReduceCumAdd.
            if (shape.IsEmpty || shape.size == 0)
            {
                var emptyRetType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

                return new NDArray(emptyRetType,
                    axis_ == null ? Shape.Vector(shape.size) : Shape.Adopt(shape.dimensions), false);
            }

            // 0-d scalar or single-element 1-D: cumprod is the value itself, promoted to the
            // accumulator dtype and shaped 1-D — cumprod NEVER returns 0-d (NumPy: cumprod(0-d) -> (1,),
            // cumprod([x]) -> [x] int64). Mirrors ReduceCumAdd; previously this returned the input dtype
            // AND a 0-d shape for a 0-d input (NEP50-skip + missing (1,) reshape).
            if (shape.IsScalar || (shape.size == 1 && shape.dimensions.Length == 1))
            {
                var single = Cast(arr, typeCode ?? arr.GetTypeCode.GetAccumulatingType(), copy: true);
                if (single.ndim != 1)
                    single.Storage.Reshape(Shape.Vector(1));   // 0-d -> (1,)
                return single;
            }

            if (axis_ == null)
            {
                var r = cumprod_elementwise(arr, typeCode);
                if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis = axisChecked;

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 - cumprod is just the value itself
                //Return a copy to avoid sharing memory with the original (NumPy behavior)
                return arr.copy();
            }

            // For broadcast arrays, we iterate over the input (which has stride=0 for broadcast dims)
            // but write to a contiguous output array.
            NDArray inputArr = arr;

            // Create output with CONTIGUOUS strides even if input is broadcast.
            var outputShape = Shape.Adopt(shape.dimensions);  // Fresh contiguous shape
            var retTypeCode = typeCode ?? (inputArr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retTypeCode, outputShape, false);

            // Fast path: use IL-generated axis kernel when available
            // Note: We only use the IL kernel for contiguous arrays without offset, as it doesn't
            // handle negative strides or offset-based views correctly.
            if (DirectILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
            {
                bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.Shape.strides[axis] == 1);
                var key = new CumulativeAxisKernelKey(inputArr.GetTypeCode, retTypeCode, ReductionOp.CumProd, innerAxisContiguous);
                var kernel = DirectILKernelGenerator.TryGetCumulativeAxisKernel(key);
                if (kernel != null)
                {
                    fixed (long* inputStrides = arr.Shape.strides)
                    fixed (long* shapePtr = arr.Shape.dimensions)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based axis cumprod (handles broadcast, non-contiguous, edge cases)
            return ExecuteAxisCumProdFallback(inputArr, ret, axis);
        }

        /// <summary>
        /// Fallback axis cumprod on the new axis iterator path.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumProdFallback(NDArray inputArr, NDArray ret, int axis)
        {
            var retType = ret.GetTypeCode;

            if (inputArr.GetTypeCode != retType)
                inputArr = Cast(inputArr, retType, copy: true);

            NpFunc.Invoke(retType, CumProdAxisDispatch<int>, inputArr.Storage, ret.Storage, axis);

            return ret;
        }

        protected unsafe NDArray cumprod_elementwise(NDArray arr, NPTypeCode? typeCode)
            => ScanElementwiseFlat(arr, typeCode, ReductionOp.CumProd);

        /// <summary>
        /// Fallback element-wise cumprod for contiguous input.
        /// </summary>
        private unsafe NDArray cumprod_elementwise_fallback(NDArray arr, NPTypeCode retType)
        {
            if (!arr.Shape.IsContiguous)
                throw new InvalidOperationException("cumprod_elementwise_fallback requires contiguous input.");

            var linearInput = arr.reshape(Shape.Vector(arr.size));
            var converted = linearInput.typecode == retType
                ? linearInput.Clone()
                : Cast(linearInput, retType, copy: true);

            NpFunc.Invoke(retType, CumProdInPlace<int>, (nint)converted.Address, converted.size);

            return converted;
        }

        private static void CumProdAxisDispatch<T>(UnmanagedStorage input, UnmanagedStorage output, int axis) where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
            => NDAxisIter.ExecuteSameType<T, CumProdAxisKernel<T>>(input, output, axis);

        private static unsafe void CumProdInPlace<T>(nint addr, long size) where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
        {
            var p = (T*)addr;
            T product = T.MultiplicativeIdentity;
            for (long i = 0; i < size; i++)
            {
                product *= p[i];
                p[i] = product;
            }
        }
    }
}
