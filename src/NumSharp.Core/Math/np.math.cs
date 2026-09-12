using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.add.html</remarks>
        /// <param name="@out">A location into which the result is stored (must broadcast with the inputs without being stretched; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Add(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.divide.html</remarks>
        public static NDArray divide(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Divide(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html</remarks>
        public static NDArray true_divide(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Divide(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.multiply.html</remarks>
        public static NDArray multiply(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Multiply(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.subtract.html</remarks>
        public static NDArray subtract(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Subtract(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.mod.html</remarks>
        public static NDArray mod(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Mod(x1, x2, dtype, @out, where);

        public static NDArray mod(NDArray x1, float x2)
            => x1.TensorEngine.Mod(x1, x2);

        /// <summary>
        /// Return element-wise remainder of division (the floored remainder — result has the same
        /// sign as the divisor, Python's <c>%</c>). In NumPy <c>remainder</c> IS <c>mod</c> (the same
        /// ufunc), so this is a straight alias of <see cref="mod(NDArray, NDArray, NDArray, NDArray, DType)"/>.
        /// The complement <see cref="fmod(NDArray, NDArray, NDArray, NDArray, DType)"/> uses the
        /// dividend's sign instead.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (NumPy ufunc out=; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.remainder.html</remarks>
        public static NDArray remainder(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Mod(x1, x2, dtype, @out, where);

        public static NDArray remainder(NDArray x1, float x2)
            => x1.TensorEngine.Mod(x1, x2);

        /// <summary>
        /// Return the element-wise C-library remainder of division. Unlike
        /// <see cref="mod(NDArray, NDArray, NDArray, NDArray, DType)"/> /
        /// <see cref="remainder(NDArray, NDArray, NDArray, NDArray, DType)"/> (floored — result has the
        /// DIVISOR's sign), <c>fmod</c> uses truncated division so the result has the DIVIDEND's sign:
        /// <c>fmod(-7, 3) == -1</c> where <c>mod(-7, 3) == 2</c>. Integer inputs stay integer (NEP50;
        /// bool -> int8); floats follow C <c>fmod</c> (<c>fmod(x, 0) == NaN</c>, etc.). Complex is not
        /// supported (NumPy has no complex fmod loop).
        /// Mirrors NumPy's ufunc signature: <c>fmod(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (NumPy ufunc out=; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): computation runs in this dtype.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fmod.html</remarks>
        public static NDArray fmod(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Fmod(x1, x2, dtype, @out, where);

        // Scalar / array-like divisor convenience (the a % obj analog; asanyarray mints the operand).
        [NDScoped]
        public static NDArray fmod(NDArray x1, object x2)
            => x1.TensorEngine.Fmod(x1, np.asanyarray(x2));

        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        /// 
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static NDArray convolve(NDArray a, NDArray v, string mode = "full")
            => a.convolve(v, mode);

        /// <summary>
        ///     Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis or axes along which a product is performed. The default, axis=None, will calculate the product of all the elements in the input array. If axis is negative it counts from the last to the first axis.</param>
        /// <param name="dtype">The type of the returned array, as well as of the accumulator in which the elements are multiplied. The dtype of a is used by default unless a has an integer dtype of less precision than the default platform integer. In that case, if a is signed then the platform integer is used while if a is unsigned then an unsigned integer of the same precision as the platform integer is used.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.</param>
        /// <returns>An array shaped as a but with the specified axis removed.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.prod.html</remarks>
        public static NDArray prod(NDArray a, int? axis = null, DType dtype = null, bool keepdims = false) //todo impl a version with keepdims
            => a.TensorEngine.ReduceProduct(a, axis, keepdims, dtype);

        /// <summary>
        ///     Numerical positive, element-wise (identity: returns +x, a copy).
        ///     Mirrors NumPy's ufunc signature: <c>positive(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): positive(i32, dtype: float64) widens; bool loop requests raise NumPy's did-not-contain-a-loop TypeError (positive has no bool loop, but positive(bool, dtype: float64) is legal).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.positive.html</remarks>
        public static NDArray positive(NDArray nd, NDArray @out = null, NDArray where = null, DType dtype = null)
            => nd.TensorEngine.Positive(nd, dtype, @out, where);

        /// <summary>
        ///     Numerical negative, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>negative(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop, so negative(bool, dtype: float64) is legal while plain negative(bool) raises (NumPy parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.negative.html</remarks>
        public static NDArray negative(NDArray nd, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // ufunc out=/where=: the provided out is returned as-is (no
            // layout post-processing — NumPy returns out untouched).
            if (@out is not null || where is not null)
                return nd.TensorEngine.Negate(nd, dtype, @out, where);

            // Route through the engine (same path as the unary `-` operator and nd.negate()):
            // the IL kernel negates unsigned integers by two's-complement wrap (NumPy: -1u -> 255)
            // and handles non-contiguous operands via NDIter. The legacy hand-written nd.negative()
            // threw NotSupportedException for unsigned dtypes and required a flat Address.
            var result = nd.TensorEngine.Negate(nd, dtype);
            // NumPy-aligned layout preservation: negative preserves F-contig input.
            if (nd.Shape.NDim > 1 && nd.size > 1
                && nd.Shape.IsFContiguous && !nd.Shape.IsContiguous
                && result.Shape.NDim > 1 && !result.Shape.IsFContiguous)
            {
                return result.copy('F');
            }
            return result;
        }
    }
}
