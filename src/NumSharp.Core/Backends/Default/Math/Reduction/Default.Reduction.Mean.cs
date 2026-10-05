using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.mean</c> / <c>ndarray.mean</c>: NumPy's <c>_methods._mean</c> — the add reduction (identity-seeded, see
        ///     <see cref="ReduceAdd"/>) divided by the count of the reduced elements. A mean of only negative zeros is
        ///     therefore +0.0, and a complex mean divides with NumPy's complex loop (Smith's algorithm with the divisor
        ///     <c>n+0j</c>), so an infinite or NaN part poisons the other one even for a single element.
        /// </summary>
        /// <param name="arr">The array to average; any dtype, any layout.</param>
        /// <param name="axis_">The axis to reduce, or null for every axis. Unlike np.sum, a 0-d array takes no axis
        /// (np.mean normalizes it against rank 0 and raises).</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="dtype">The dtype the sum is computed in and the mean returned in; null selects float64 for bool /
        /// integers / char and the input's own dtype for float16 / float32 / float64 / complex128 / decimal.</param>
        /// <returns>The mean: a read-only 0-d scalar for a full reduction without keepdims, otherwise a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range (every axis is, for a 0-d input).</exception>
        public override NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            var shape = arr.Shape;
            // np.mean's result dtype: dtype=, else float64 for bool / integers / char, the input's own otherwise (NumPy
            // computes a float16 mean through a float32 sum and returns float16).
            var meanType = typeCode ?? arr.GetTypeCode.GetComputingType();

            if (shape.IsEmpty)
                return NDArray.Scalar(double.NaN);

            if (shape.size == 0)
            {
                // NumPy: the identity-seeded sum (0) divided by a count of 0 — NaN in the mean's dtype ((nan+nanj) for
                // complex, through CDOUBLE_divide's zero-divisor branch). A dtype that cannot hold NaN (decimal, an
                // explicit integer dtype) keeps the float64 NaN this path always returned.
                var nan = NaNOfMeanType(meanType, out var nanType);
                if (axis_ == null)
                {
                    var r = NDArray.Scalar(nan, nanType);
                    if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                    return r.MarkReductionScalar();
                }
                var axis = NormalizeAxis(axis_.Value, arr.ndim);
                var resultShape = Shape.GetAxis(shape, axis);
                var outputType = typeCode.HasValue ? typeCode.Value : nanType;
                NDArray result;
                if (shape[axis] == 0)
                {
                    result = np.empty(Shape.Adopt(resultShape), outputType);
                    for (long i = 0; i < result.size; i++) result.SetAtIndex(nan, i);
                }
                else result = np.empty(Shape.Adopt(resultShape), outputType);
                if (keepdims)
                {
                    var ks = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(Shape.Adopt(ks));
                }
                return result;
            }

            // ONE element per output — a one-element input reduced flat or over its only axis: NumPy's mean is
            // (identity + x) / 1, the identity add of np.sum and then np.mean's own division (a no-op for every real
            // dtype; NumPy's complex division by 1+0j for complex). Never a copy of x: -0.0 becomes +0.0.
            if (shape.size == 1 && (axis_ == null || shape.NDim <= 1))
            {
                if (axis_ != null)
                {
                    // np.mean normalizes the axis itself (_count_reduce_items), so — unlike np.sum — a 0-d input
                    // refuses even axis 0 / -1.
                    if (shape.NDim == 0)
                        throw new AxisError(axis_.Value, 0);
                    NormalizeAxis(axis_.Value, shape.NDim);
                }

                return SumOfOneElement(arr, keepdims, meanType, null, divideByCount: true);
            }

            if (axis_ == null)
            {
                // The flat mean of a float loop is NumPy's _mean statement for statement: its exact add.reduce in the loop
                // _mean picks (float64 for bool / integers, float32 for float16), then the float64 division by the count.
                var exact = TryExactFlatMean(arr, keepdims, typeCode);
                if (exact is not null)
                    return exact;

                var result = mean_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r.MarkReductionScalar();
            }

            var axis2 = NormalizeAxis(axis_.Value, arr.ndim);
            var inputTc = arr.GetTypeCode;

            // B2: Complex mean axis is handled by the NDIter REDUCE path (ExecuteAxisReduction
            // → ExecuteAxisReductionNDIter), which runs a one-pass complex Sum kernel then
            // divides by the axis length — both components preserved (NumPy parity). This
            // replaced the per-output-row-allocating MeanAxisComplex (15–45× slower).

            // B16: Half mean axis computes in Double then casts back to preserve Half dtype.
            bool needsCast = !typeCode.HasValue && inputTc == NPTypeCode.Half;
            // NumPy parity: mean preserves float input dtype (float32→float32, float64→float64);
            // integer inputs promote to float64. GetComputingType() encodes exactly this rule and
            // also keeps InputType == AccumulatorType for floats, which is what unlocks the
            // SIMD same-type axis-reduction kernel. Forcing Double here was a 2576× regression
            // on mean(float32, axis=0) because it dropped into the scalar promoted helper.
            var outputType2 = needsCast ? NPTypeCode.Double : (typeCode ?? inputTc.GetComputingType());

            NDArray result2;
            if (shape[axis2] == 1)
                // Every output owns one element: (identity + x) / 1, never a copy of x (a -0.0 slab is +0.0); the
                // count division only changes a complex loop (NumPy's complex division by 1+0j).
                result2 = SumOverLengthOneAxis(arr, axis2, keepdims, outputType2, null, divideByCount: true);
            else
                result2 = ExecuteAxisReduction(arr, axis2, keepdims, outputType2, null, ReductionOp.Mean);

            if (needsCast)
            {
                // The double-precision intermediate is consumed by the narrowing COPY cast — leaving
                // it undisposed stranded one pooled buffer per f16 axis mean (the sum path's twin
                // leak, caught by UndisposedIntermediateTests via the instance oracle tier,
                // 2026-09-18).
                var wide2 = result2;
                try
                {
                    result2 = Cast(result2, inputTc, copy: true);
                }
                finally
                {
                    wide2.Dispose();
                }
            }
            return result2;
        }

        /// <summary>
        ///     The value an EMPTY mean takes: NumPy divides the identity-seeded sum (0) by a count of 0, which is NaN in
        ///     the mean's own dtype — <c>(nan+nanj)</c> for complex128 (CDOUBLE_divide's zero-divisor branch divides each
        ///     part by 0). A dtype without a NaN (decimal, or an explicit integer dtype) falls back to the float64 NaN this
        ///     route always returned.
        /// </summary>
        /// <param name="meanType">The mean's result dtype.</param>
        /// <param name="nanType">Receives the dtype the returned value is boxed as.</param>
        /// <returns>The boxed NaN of <paramref name="nanType"/>.</returns>
        private static object NaNOfMeanType(NPTypeCode meanType, out NPTypeCode nanType)
        {
            nanType = meanType;
            switch (meanType)
            {
                case NPTypeCode.Half: return Half.NaN;
                case NPTypeCode.Single: return float.NaN;
                case NPTypeCode.Double: return double.NaN;
                case NPTypeCode.Complex: return new System.Numerics.Complex(double.NaN, double.NaN);
                default:
                    nanType = NPTypeCode.Double;
                    return double.NaN;
            }
        }

        /// <summary>
        /// Element-wise mean for typed result. Compatibility method for Std/Var.
        /// </summary>
        public T MeanElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var result = mean_elementwise_il(arr, typeCode);
            return (T)Converts.ChangeType(result, typeof(T).GetTypeCode());
        }
    }
}
