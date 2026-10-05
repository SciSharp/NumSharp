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

            var inT = arr.GetTypeCode;
            // Decimal (no NumPy dtype) and a char loop (NumSharp's, reachable only through dtype=) keep the routes below.
            bool portable = inT != NPTypeCode.Decimal && meanType != NPTypeCode.Decimal && meanType != NPTypeCode.Char;

            // An integer or bool loop (only through dtype=) is _mean statement for statement on every shape: the sum in the
            // loop (wrapping, or logical OR for bool), the float64 quotient, NumPy's unsafe cast back (an empty slice's
            // 0 / 0 = NaN becomes INT_MIN, or True). The routes below assumed a float result: an empty input returned a
            // float64 NaN or threw InvalidCastException, a flat sum did not wrap, an axis sum read every element as a
            // double.
            if (portable && (meanType.IsInteger() || meanType == NPTypeCode.Boolean))
                return MeanLikeNumPy(arr, MeanAxis(axis_, shape.NDim), keepdims, meanType, float16Result: false);

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

                // What the exact engine declines (a broadcast operand, a decimal one) keeps the elementwise fold below
                // where that fold is NumPy's value: it sums an integer / bool / char input in its native 64-bit
                // accumulator and divides in float64, which equals NumPy's float64 sum only in the float64 loop and only
                // while the sum is exact in float64 (size × max |x| <= 2^53). Anything else takes _mean's statements
                // through np.sum's own route for the operand: an int64 / uint64 input, whose native sum wraps (np.mean
                // of a broadcast [0, 2**64-1, 127, 128, 255] gave 101.8, NumPy 3.69e18), another dtype= loop (the fold
                // divided before converting), and decimal, whose fold returned float64 where the axis route (and the
                // decimal oracle) keep decimal.
                if (inT == NPTypeCode.Decimal ? meanType == NPTypeCode.Decimal
                    : portable && IntegerMaxMagnitude(inT) >= 0
                      && !(meanType == NPTypeCode.Double && IntegerMaxMagnitude(inT) > 0
                           && arr.size * IntegerMaxMagnitude(inT) <= 9007199254740992.0))
                    return MeanLikeNumPy(arr, null, keepdims, meanType, float16Result: false);

                var result = mean_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r.MarkReductionScalar();
            }

            var axis2 = NormalizeAxis(axis_.Value, arr.ndim);
            var inputTc = arr.GetTypeCode;

            // Every float or complex loop along an axis (but a length-1 axis, whose identity add above is exact) is _mean
            // statement for statement: np.sum's exact add reduction in the loop (a cast loop casts each element first;
            // float16 sums in float32 by default), then the division. The kernel route below summed a cast loop's
            // elements as doubles in one sequential pass and divided a complex sum part by part; and it walked a reversed
            // axis in memory order, which ReduceAdd's flat engine walks in NumPy's logical order.
            if (portable && shape[axis2] != 1)
                return MeanLikeNumPy(arr, axis2, keepdims, typeCode ?? (inputTc == NPTypeCode.Half ? NPTypeCode.Single : meanType),
                                     float16Result: !typeCode.HasValue && inputTc == NPTypeCode.Half);

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
        ///     np.mean's axis check: <c>_count_reduce_items</c> normalizes the axis against the operand's rank before
        ///     anything is summed, so a 0-d operand refuses every axis (np.sum would take 0 and -1 there).
        /// </summary>
        /// <param name="axis">The requested axis, negative counting from the end, or null for a flat mean.</param>
        /// <param name="ndim">The operand's rank.</param>
        /// <returns>The non-negative axis, or null.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand (every axis is, for a 0-d one).</exception>
        private static int? MeanAxis(int? axis, int ndim) => axis is null ? null : NormalizeAxis(axis.Value, ndim);

        /// <summary>
        ///     NumPy's <c>_methods._mean</c>, statement for statement:
        ///     <code>
        ///     rcount = _count_reduce_items(arr, axis, keepdims=keepdims)     # np.intp: a STRONG int64
        ///     ret = umr_sum(arr, axis, dtype, out, keepdims)                 # np.sum in the loop
        ///     if isinstance(ret, ndarray):
        ///         ret = um.true_divide(ret, rcount, out=ret, casting='unsafe')
        ///         if is_float16_result: ret = arr.dtype.type(ret)            # float32 -> float16
        ///     else:                                                          # a scalar
        ///         ret = (arr.dtype if is_float16_result else ret.dtype).type(ret / rcount)
        ///     </code>
        /// </summary>
        /// <param name="arr">The operand; any dtype but decimal, any layout, not the uninitialized-shape sentinel.</param>
        /// <param name="axis">The normalized axis, or null for a flat mean.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="loop">The sum's loop dtype (the <c>dtype=</c> request, or <c>_mean</c>'s default: float64 for
        /// bool / integers / char, float32 for float16, the input's own otherwise).</param>
        /// <param name="float16Result">NumPy's <c>is_float16_result</c>: a float16 input without <c>dtype=</c>, summed in
        /// float32 and returned as float16.</param>
        /// <returns>The mean in <paramref name="loop"/> (float16 for <paramref name="float16Result"/>), laid out as
        /// np.sum lays out its result; 0-d and read-only (NumPy's scalar) when every axis is reduced without
        /// keepdims.</returns>
        /// <remarks>
        ///     The two branches differ only for <paramref name="float16Result"/>: the array branch rounds the float64
        ///     quotient into the float32 sum and then into float16 (twice), the scalar branch rounds the float64 quotient
        ///     into float16 once. For every other loop both are the float64 quotient rounded once into the loop
        ///     (<see cref="DivideByCountLikeNumPy"/>: complex divides by <c>rcount + 0j</c>, an integer or bool loop casts
        ///     unsafely, so 0 / 0 is INT_MIN or True).
        /// </remarks>
        private NDArray MeanLikeNumPy(NDArray arr, int? axis, bool keepdims, NPTypeCode loop, bool float16Result)
        {
            // rcount: the reduced extent. An empty slice divides by 0 (NumPy's NaN, then the loop's cast of it).
            long count = axis is null ? arr.size : arr.Shape.dimensions[axis.Value];
            var ret = ReduceAdd(arr, axis, keepdims, loop, @out: null);
            bool scalar = ret.ndim == 0;

            if (float16Result && scalar)
            {
                // arr.dtype.type(ret / rcount): np.float32 / np.intp promotes to float64, rounded once into float16.
                double q = ret.GetSingle() / (double)count;
                ret.Dispose();
                var h = new NDArray(NPTypeCode.Half, Shape.Scalar, false);
                unsafe { *(ushort*)h.Address = DirectILKernelGenerator.DoubleToHalfBits(q); }
                return h.MarkReductionScalar();
            }

            // A 0-d sum is the read-only reduction scalar; 0-d has no layout to keep, so dividing a copy is exact.
            if (!ret.Shape.IsWriteable)
            {
                var owned = ret.copy();
                ret.Dispose();
                ret = owned;
            }

            // The engine's dense-block division is NumPy's quotient where it is exact: x / (double)n for the float64 loop,
            // the complex loop's division by n + 0j, and x / (float)n for float32 while n <= 2^24 (n is then exact in
            // float32, and float64 division rounded once to float32 equals float32 division: 53 >= 2*24 + 2 bits). Every
            // other loop (and a larger float32 count) takes the float64 loop with NumPy's unsafe cast back.
            if (count > 0 && (loop == NPTypeCode.Double || loop == NPTypeCode.Complex
                              || (loop == NPTypeCode.Single && count <= 1L << 24)))
                ILKernelGenerator.MeanDivideByCount(ret, count);
            else
                DivideByCountLikeNumPy(ret, count);

            if (float16Result)
            {
                // arr.dtype.type(ret) on the float32 array: a K-order cast (the layout np.sum gave the sum stays).
                var narrowed = ret.astype(np.float16, copy: true);
                ret.Dispose();
                ret = narrowed;
            }

            return scalar ? ret.MarkReductionScalar() : ret;
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
