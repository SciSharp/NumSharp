using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;

            if (shape.IsEmpty)
                throw new ArgumentException("attempt to get argmax of an empty sequence");

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.argmax(np.zeros((0,3)), axis=0) raises ValueError (reducing along zero-size axis)
            // NumPy: np.argmax(np.zeros((0,3)), axis=1) returns array([], dtype=int64) with shape (0,)
            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - raise error
                    throw new ArgumentException("attempt to get argmax of an empty sequence");
                }

                // Reducing along non-zero axis - return empty Int64 array with reduced shape
                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var result = np.empty(new Shape(resultShape), NPTypeCode.Int64);

                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++)
                    {
                        if (d == emptyAxis)
                            keepdimsShape[d] = 1;
                        else
                            keepdimsShape[d] = resultShape[sd++];
                    }
                    result.Storage.Reshape(new Shape(keepdimsShape));
                }

                return result;
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                var r = NDArray.Scalar(0L);  // Int64 for NumPy 2.x alignment
                if (keepdims)
                {
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            if (axis_ == null)
            {
                // Use IL-generated kernels for element-wise reduction
                var r = NDArray.Scalar(argmax_elementwise_il(arr));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                return r;
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1, result is all zeros
                if (keepdims)
                {
                    // Keep the axis but reduce to size 1 (it's already 1)
                    return np.zeros(shape.dimensions, NPTypeCode.Int64);
                }
                return np.squeeze_fast(np.zeros(shape.dimensions, NPTypeCode.Int64), axis);
            }

            //handle keepdims - prepare output shape
            Shape axisedShape = Shape.GetAxis(shape, axis);
            Shape outputShape = axisedShape;
            if (keepdims)
            {
                // Insert a 1 at the axis position
                var keepdimsShapeDims = new long[arr.ndim];
                int srcIdx = 0;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (i == axis)
                        keepdimsShapeDims[i] = 1;
                    else
                        keepdimsShapeDims[i] = axisedShape.dimensions[srcIdx++];
                }
                outputShape = new Shape(keepdimsShapeDims);
            }

            // Use IL kernel for axis reduction
            return ExecuteAxisArgReduction(arr, axis, keepdims, outputShape, axisedShape, ReductionOp.ArgMax);
        }

        /// <summary>
        /// Execute axis ArgMax/ArgMin reduction using IL kernels.
        /// </summary>
        private unsafe NDArray ExecuteAxisArgReduction(NDArray arr, int axis, bool keepdims, Shape outputShape, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // Half and Complex have no IL kernel (Bgt/Blt opcodes don't apply to those types
            // and Complex needs lexicographic compare). Route through a stride-aware loop
            // that avoids per-slice NDArray view allocation (R23).
            if (inputType == NPTypeCode.Half)
                return ArgReductionAxisHalf(arr, axis, keepdims, outputShape, axisedShape, op);
            if (inputType == NPTypeCode.Complex)
                return ArgReductionAxisComplex(arr, axis, keepdims, outputShape, axisedShape, op);

            // ArgMax/ArgMin always output Int64
            var key = new AxisReductionKernelKey(inputType, NPTypeCode.Int64, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetAxisReductionKernel(key);

            if (kernel == null)
            {
                return ArgReductionAxisFallback(arr, axis, keepdims, outputShape, axisedShape, op);
            }

            // Use IL kernel path
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);
            long axisSize = shape.dimensions[axis];
            long outputSize = ret.size > 0 ? ret.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = ret.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)ret.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
                ret.Storage.Reshape(outputShape);

            return ret;
        }

        /// <summary>
        /// B7: Fallback argmax/argmin axis reduction when IL kernel not available.
        /// Iterates per slice and calls the scalar argmax_elementwise_il (which has per-dtype
        /// fallbacks for Half, Complex, SByte). Returns an Int64 NDArray with the reduced shape.
        /// </summary>
        private NDArray ArgReductionAxisFallback(NDArray arr, int axis, bool keepdims, Shape outputShape, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new ValueCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
            var slices = iterAxis.Slices;

            do
            {
                // slice is an owning view wrapper into arr's storage — each
                // outer-axis iteration would otherwise queue an NDArray
                // wrapper on the finalizer queue. Storage stays alive via arr.
                using var slice = arr[slices];
                long result = op == ReductionOp.ArgMax
                    ? argmax_elementwise_il(slice)
                    : argmin_elementwise_il(slice);
                ret.SetAtIndex(result, iterIndex[0]);
            } while (iterAxis.Next() != null && iterRet.Next() != null);

            if (keepdims)
                ret.Storage.Reshape(outputShape);

            return ret;
        }

        /// <summary>
        /// Stride-aware axis ArgMax/ArgMin for Half. Avoids per-slice NDArray view allocation:
        /// walks output coordinates and follows the axis stride directly via pointer arithmetic.
        /// NaN-propagating (first NaN on the axis wins) to match NumPy.
        /// </summary>
        private unsafe NDArray ArgReductionAxisHalf(NDArray arr, int axis, bool keepdims, Shape outputShape, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            int ndim = arr.ndim;
            long axisSize = shape.dimensions[axis];
            long axisStride = shape.strides[axis];
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);

            Half* basePtr = (Half*)arr.Address + shape.offset;
            long* retPtr = (long*)ret.Address;

            // Build a stride/dim view excluding the reduction axis so we can iterate
            // outputCount positions via a single coordinate vector.
            int outNdim = ndim - 1;
            long outputCount = ret.size == 0 ? 1 : ret.size;
            long* outDims = stackalloc long[Math.Max(outNdim, 1)];
            long* outStrides = stackalloc long[Math.Max(outNdim, 1)];
            for (int d = 0, k = 0; d < ndim; d++)
            {
                if (d == axis) continue;
                outDims[k] = shape.dimensions[d];
                outStrides[k] = shape.strides[d];
                k++;
            }
            long* coords = stackalloc long[Math.Max(outNdim, 1)];
            for (int d = 0; d < outNdim; d++) coords[d] = 0;

            bool isArgMax = op == ReductionOp.ArgMax;
            for (long outIdx = 0; outIdx < outputCount; outIdx++)
            {
                long baseOffset = 0;
                for (int d = 0; d < outNdim; d++)
                    baseOffset += coords[d] * outStrides[d];

                long bestIdx = 0;
                double best = (double)*(basePtr + baseOffset);
                bool nanSeen = double.IsNaN(best);
                if (!nanSeen)
                {
                    for (long i = 1; i < axisSize; i++)
                    {
                        double v = (double)*(basePtr + baseOffset + i * axisStride);
                        if (double.IsNaN(v)) { bestIdx = i; nanSeen = true; break; }
                        if (isArgMax ? v > best : v < best) { best = v; bestIdx = i; }
                    }
                }
                retPtr[outIdx] = bestIdx;

                // Advance C-order coords
                for (int d = outNdim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < outDims[d]) break;
                    coords[d] = 0;
                }
            }

            if (keepdims)
                ret.Storage.Reshape(outputShape);

            return ret;
        }

        /// <summary>
        /// Stride-aware axis ArgMax/ArgMin for Complex. Uses NumPy's lexicographic compare
        /// (real, then imag). NaN in either component propagates.
        /// </summary>
        private unsafe NDArray ArgReductionAxisComplex(NDArray arr, int axis, bool keepdims, Shape outputShape, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            int ndim = arr.ndim;
            long axisSize = shape.dimensions[axis];
            long axisStride = shape.strides[axis];
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, false);

            System.Numerics.Complex* basePtr = (System.Numerics.Complex*)arr.Address + shape.offset;
            long* retPtr = (long*)ret.Address;

            int outNdim = ndim - 1;
            long outputCount = ret.size == 0 ? 1 : ret.size;
            long* outDims = stackalloc long[Math.Max(outNdim, 1)];
            long* outStrides = stackalloc long[Math.Max(outNdim, 1)];
            for (int d = 0, k = 0; d < ndim; d++)
            {
                if (d == axis) continue;
                outDims[k] = shape.dimensions[d];
                outStrides[k] = shape.strides[d];
                k++;
            }
            long* coords = stackalloc long[Math.Max(outNdim, 1)];
            for (int d = 0; d < outNdim; d++) coords[d] = 0;

            bool isArgMax = op == ReductionOp.ArgMax;
            for (long outIdx = 0; outIdx < outputCount; outIdx++)
            {
                long baseOffset = 0;
                for (int d = 0; d < outNdim; d++)
                    baseOffset += coords[d] * outStrides[d];

                long bestIdx = 0;
                var best = *(basePtr + baseOffset);
                bool nanSeen = double.IsNaN(best.Real) || double.IsNaN(best.Imaginary);
                if (!nanSeen)
                {
                    for (long i = 1; i < axisSize; i++)
                    {
                        var v = *(basePtr + baseOffset + i * axisStride);
                        if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary))
                        {
                            bestIdx = i; nanSeen = true; break;
                        }
                        bool wins = isArgMax
                            ? (v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary))
                            : (v.Real < best.Real || (v.Real == best.Real && v.Imaginary < best.Imaginary));
                        if (wins) { best = v; bestIdx = i; }
                    }
                }
                retPtr[outIdx] = bestIdx;

                for (int d = outNdim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < outDims[d]) break;
                    coords[d] = 0;
                }
            }

            if (keepdims)
                ret.Storage.Reshape(outputShape);

            return ret;
        }

    }
}
