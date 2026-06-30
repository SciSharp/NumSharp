using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null)
        {
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            // Handle empty arrays (size == 0) with axis reduction
            // NumPy: np.var(np.zeros((0,3)), axis=0) returns array([nan, nan, nan]) (reducing along zero-size axis)
            // NumPy: np.var(np.zeros((0,3)), axis=1) returns array([]) with shape (0,) (reducing along non-zero axis)
            if (arr.size == 0)
            {
                if (axis_ == null)
                {
                    // No axis specified - return NaN scalar
                    var r = NDArray.Scalar(double.NaN);
                    if (keepdims)
                    {
                        var keepdimsShape = new long[arr.ndim];
                        for (int i = 0; i < arr.ndim; i++)
                            keepdimsShape[i] = 1;
                        r.Storage.Reshape(new Shape(keepdimsShape));
                    }
                    return r;
                }

                // Axis specified - check if reducing along zero-size axis
                var emptyAxis = axis_.Value;
                while (emptyAxis < 0)
                    emptyAxis = arr.ndim + emptyAxis;
                if (emptyAxis >= arr.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis_));

                var resultShape = Shape.GetAxis(shape, emptyAxis);
                var emptyOutputType = typeCode ?? NPTypeCode.Double;

                NDArray result;
                if (shape[emptyAxis] == 0)
                {
                    // Reducing along a zero-size axis - return NaN filled array
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                    for (long i = 0; i < result.size; i++)
                        result.SetAtIndex(double.NaN, i);
                }
                else
                {
                    // Reducing along non-zero axis - return empty array with reduced shape
                    result = np.empty(new Shape(resultShape), emptyOutputType);
                }

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
                // NumPy: variance of single element with ddof=0 is 0.0
                // With ddof >= size, the divisor is <= 0, which produces NaN
                int _ddof = ddof ?? 0;
                double value = (arr.size - _ddof) <= 0 ? double.NaN : 0.0;
                var r = NDArray.Scalar(value);
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(var_elementwise(arr, typeCode, ddof));
                if (keepdims)
                {
                    // NumPy: keepdims preserves the number of dimensions, all set to 1
                    var keepdimsShape = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShape[i] = 1;
                    r.Storage.Reshape(new Shape(keepdimsShape));
                }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }
            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 - variance of a single element is 0
                //Return zeros with the appropriate shape (NumPy behavior)
                // B23: Complex variance collapses to float64 in NumPy (variance of complex is a
                // real non-negative number). GetComputingType preserves Complex→Complex which
                // would give the wrong dtype here; override to Double for Complex inputs.
                var zerosType = typeCode
                    ?? (arr.GetTypeCode == NPTypeCode.Complex
                        ? NPTypeCode.Double
                        : arr.GetTypeCode.GetComputingType());
                if (keepdims)
                {
                    var keepdimsShapeDims = new long[arr.ndim];
                    for (int i = 0; i < arr.ndim; i++)
                        keepdimsShapeDims[i] = (i == axis) ? 1 : shape[i];
                    return np.zeros(keepdimsShapeDims, zerosType);
                }
                return np.zeros(Shape.GetAxis(shape, axis), zerosType);
            }

            // IL-generated axis reduction fast path - handles all numeric types
            if (DirectILKernelGenerator.Enabled)
            {
                // B16: var axis preserves float input dtype (half → half). Complex → Double (variance
                // is a non-negative real number). Integer → Double.
                var axisOutType = typeCode
                    ?? (arr.GetTypeCode == NPTypeCode.Complex
                        ? NPTypeCode.Double
                        : arr.GetTypeCode.GetComputingType());
                var ilResult = ExecuteAxisVarReductionIL(arr, axis, keepdims, axisOutType, ddof ?? 0);
                if (ilResult is not null)
                    return ilResult;
            }

            // Fallback: iterator-based axis reduction (handles non-contiguous, broadcast, edge cases)
            return ExecuteAxisVarReductionFallback(arr, axis, keepdims, typeCode, ddof);
        }

        /// <summary>
        /// Fallback axis var reduction using iterators. Used when IL kernel not available.
        /// </summary>
        private NDArray ExecuteAxisVarReductionFallback(NDArray arr, int axis, bool keepdims, NPTypeCode? typeCode, int? ddof)
        {
            Shape axisedShape = Shape.GetAxis(arr.Shape, axis);
            var retType = typeCode ?? arr.GetTypeCode.GetComputingType();

            var ret = new NDArray(retType, axisedShape, false);
            int _ddof = ddof ?? 0;
            var input = arr.GetTypeCode == NPTypeCode.Double ? arr : Cast(arr, NPTypeCode.Double, copy: true);
            NDAxisIter.ReduceDouble<VarAxisDoubleKernel>(input.Storage, ret.Storage, axis, _ddof);

            if (keepdims)
                ret.Storage.ExpandDimension(axis);

            return ret;
        }

        public T VarElementwise<T>(NDArray arr, NPTypeCode? typeCode, int? ddof) where T : unmanaged
        {
            return (T)Converts.ChangeType(var_elementwise(arr, typeCode, ddof), InfoOf<T>.NPTypeCode);
        }

        protected object var_elementwise(NDArray arr, NPTypeCode? typeCode, int? ddof)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
            {
                // With ddof >= size, divisor is <= 0, which produces NaN
                int _ddof = ddof ?? 0;
                return (arr.size - _ddof) <= 0 ? double.NaN : 0.0;
            }

            var retType = typeCode ?? (arr.GetTypeCode).GetComputingType();

            // SIMD fast-path for contiguous arrays
            if (DirectILKernelGenerator.Enabled && arr.Shape.IsContiguous)
            {
                int _ddof = ddof ?? 0;
                double variance;

                unsafe
                {
                    switch (arr.GetTypeCode)
                    {
                        case NPTypeCode.Single:
                            variance = DirectILKernelGenerator.VarSimdHelper((float*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Double:
                            variance = DirectILKernelGenerator.VarSimdHelper((double*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Byte:
                            variance = DirectILKernelGenerator.VarSimdHelper((byte*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.SByte:
                            variance = DirectILKernelGenerator.VarSimdHelper((sbyte*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int16:
                            variance = DirectILKernelGenerator.VarSimdHelper((short*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt16:
                            variance = DirectILKernelGenerator.VarSimdHelper((ushort*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int32:
                            variance = DirectILKernelGenerator.VarSimdHelper((int*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt32:
                            variance = DirectILKernelGenerator.VarSimdHelper((uint*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.Int64:
                            variance = DirectILKernelGenerator.VarSimdHelper((long*)arr.Address, arr.size, _ddof);
                            break;
                        case NPTypeCode.UInt64:
                            variance = DirectILKernelGenerator.VarSimdHelper((ulong*)arr.Address, arr.size, _ddof);
                            break;
                        default:
                            goto fallback;
                    }
                }

                // Convert to requested return type
                return Converts.ChangeType(variance, retType);

                fallback:;
            }

            // Fallback: iterator-based (handles non-contiguous, decimal, char, bool)
            return var_elementwise_fallback(arr, retType, ddof);
        }

        /// <summary>
        /// Fallback element-wise var. The input is iterated in place through its strides
        /// (<see cref="FlatStrideOffset"/>) — strided / transposed / sliced / reversed views are
        /// visited via coordinate decode rather than materialized to a contiguous copy. var is an
        /// order-independent two-pass reduction, so any visiting order is valid; this mirrors
        /// NumPy, which reduces strided arrays in place instead of copying.
        /// </summary>
        private unsafe object var_elementwise_fallback(NDArray arr, NPTypeCode retType, int? ddof)
        {
            int _ddof = ddof ?? 0;
            var tc = arr.GetTypeCode;
            byte* basePtr = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
            bool contig = arr.Shape.IsContiguous;
            var dims = arr.shape;
            var strides = arr.strides;
            int ndim = arr.ndim;
            long n = arr.size;

            if (tc == NPTypeCode.Decimal)
                return Converts.ChangeType(VarMomentsDecimal((decimal*)basePtr, dims, strides, ndim, contig, n, _ddof), retType);

            // Complex var uses |x - mean|^2 and returns float64.
            if (tc == NPTypeCode.Complex)
                return VarMomentsComplex((Complex*)basePtr, dims, strides, ndim, contig, n, _ddof);

            double variance = VarMomentsRealDispatch(tc, basePtr, dims, strides, ndim, contig, n, _ddof);
            return Converts.ChangeType(variance, retType);
        }

        /// <summary>
        /// C-order visitation offset (in elements) for the <paramref name="linear"/>-th element of
        /// a strided view. Decodes coordinates last-axis-fastest and folds them through the strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long FlatStrideOffset(long linear, long[] dims, long[] strides, int ndim)
        {
            long off = 0;
            for (int d = ndim - 1; d >= 0; d--)
            {
                long dim = dims[d];
                off += (linear % dim) * strides[d];
                linear /= dim;
            }
            return off;
        }

        /// <summary>Two-pass variance over a strided real-typed buffer, accumulating in double.</summary>
        private static unsafe double VarMomentsReal<TIn>(TIn* p, long[] dims, long[] strides, int ndim, bool contig, long n, int ddof)
            where TIn : unmanaged, INumberBase<TIn>
        {
            double sum = 0;
            for (long i = 0; i < n; i++)
                sum += double.CreateChecked(p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]);
            double mean = sum / n;
            double sq = 0;
            for (long i = 0; i < n; i++)
            {
                double v = double.CreateChecked(p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)]) - mean;
                sq += v * v;
            }
            return sq / (n - ddof);
        }

        /// <summary>
        /// Two-pass variance over a strided bool buffer. A boolean's numeric value is 0 or 1, never
        /// its raw storage byte, so each element is normalized via '!= 0' before accumulation —
        /// matching NumPy, where var/std of a bool array operate on the {0,1} values even when the
        /// underlying buffer (np.frombuffer view / interop) holds non-0/1 bytes.
        /// </summary>
        private static unsafe double VarMomentsBool(byte* p, long[] dims, long[] strides, int ndim, bool contig, long n, int ddof)
        {
            double sum = 0;
            for (long i = 0; i < n; i++)
                sum += p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)] != 0 ? 1.0 : 0.0;
            double mean = sum / n;
            double sq = 0;
            for (long i = 0; i < n; i++)
            {
                double v = (p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)] != 0 ? 1.0 : 0.0) - mean;
                sq += v * v;
            }
            return sq / (n - ddof);
        }

        /// <summary>Dispatch the real-typed strided two-pass on the input dtype (bool→byte, char→ushort).</summary>
        private static unsafe double VarMomentsRealDispatch(NPTypeCode tc, byte* basePtr, long[] dims, long[] strides, int ndim, bool contig, long n, int ddof)
            => tc switch
            {
                // bool is NOT byte here: its numeric value is 0/1, not the raw storage byte. A bool
                // buffer may hold non-0/1 bytes (np.frombuffer view / interop); normalize via '!= 0'
                // so var/std count True as 1 (NumPy parity). See VarMomentsBool.
                NPTypeCode.Boolean => VarMomentsBool((byte*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Byte    => VarMomentsReal((byte*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.SByte   => VarMomentsReal((sbyte*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Int16   => VarMomentsReal((short*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.UInt16  => VarMomentsReal((ushort*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Char    => VarMomentsReal((ushort*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Int32   => VarMomentsReal((int*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.UInt32  => VarMomentsReal((uint*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Int64   => VarMomentsReal((long*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.UInt64  => VarMomentsReal((ulong*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Half    => VarMomentsReal((Half*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Single  => VarMomentsReal((float*)basePtr, dims, strides, ndim, contig, n, ddof),
                NPTypeCode.Double  => VarMomentsReal((double*)basePtr, dims, strides, ndim, contig, n, ddof),
                _ => throw new NotSupportedException($"var/std not supported for {tc}")
            };

        /// <summary>Two-pass variance over a strided decimal buffer, accumulating in decimal.</summary>
        private static unsafe decimal VarMomentsDecimal(decimal* p, long[] dims, long[] strides, int ndim, bool contig, long n, int ddof)
        {
            decimal mean = 0;
            for (long i = 0; i < n; i++)
                mean += p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)];
            mean /= n;
            decimal sum = 0;
            for (long i = 0; i < n; i++)
            {
                decimal a = p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)] - mean;
                sum += a * a;
            }
            return sum / ((decimal)n - ddof);
        }

        /// <summary>Two-pass variance over a strided complex buffer; returns float64 of |x-mean|^2.</summary>
        private static unsafe double VarMomentsComplex(Complex* p, long[] dims, long[] strides, int ndim, bool contig, long n, int ddof)
        {
            var xmean = Complex.Zero;
            for (long i = 0; i < n; i++)
                xmean += p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)];
            xmean /= n;
            double sum = 0;
            for (long i = 0; i < n; i++)
            {
                var diff = p[contig ? i : FlatStrideOffset(i, dims, strides, ndim)] - xmean;
                sum += diff.Real * diff.Real + diff.Imaginary * diff.Imaginary;
            }
            return sum / (n - ddof);
        }

        /// <summary>
        /// IL-generated axis variance reduction. Returns null if kernel not available.
        /// </summary>
        private unsafe NDArray ExecuteAxisVarReductionIL(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, int ddof)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // Var axis reduction always outputs double for accuracy
            var key = new AxisReductionKernelKey(inputType, NPTypeCode.Double, ReductionOp.Var, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetAxisReductionKernel(key);

            if (kernel == null)
                return null;

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? new Shape(outputDims) : Shape.Scalar;
            var result = new NDArray(NPTypeCode.Double, outputShape, false);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                // The kernel computes variance with ddof=0 by default
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);

                // For ddof != 0, adjust: var_ddof = var_0 * n / max(n - ddof, 0)
                // B24: clamp (n - ddof) to 0 to match NumPy, which uses max(n-ddof, 0) as the
                // divisor. For ddof >= n the divisor is 0 → IEEE yields +inf (var is unbounded
                // when degrees of freedom are exhausted). Without the clamp, ddof > n gives a
                // negative adjustment and therefore negative variance — wrong sign AND wrong value.
                if (ddof != 0)
                {
                    double* resultPtr = (double*)result.Address;
                    double divisor = Math.Max(axisSize - ddof, 0);
                    double adjustment = (double)axisSize / divisor;
                    for (long i = 0; i < outputSize; i++)
                        resultPtr[i] *= adjustment;
                }
            }

            // Convert to requested output type if different from double
            if (outputType != NPTypeCode.Double)
            {
                result = Cast(result, outputType, copy: true);
            }

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : (sd < outputDims.Length ? outputDims[sd++] : 1);
                result.Storage.Reshape(new Shape(ks));
            }

            return result;
        }
    }
}
