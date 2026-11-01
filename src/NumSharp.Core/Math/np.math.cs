using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.add.html</remarks>
        /// <param name="@out">A location into which the result is stored (must broadcast with the inputs without being stretched; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> input gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>), a plain array of a result of another dtype; a chararray
        ///     <paramref name="out"/> comes back as a new chararray view of it.
        /// </remarks>
        /// <exception cref="ValueError">A short <c>numpy.char.chararray</c> input (0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements) that NumPy's trivial loop copies into a loop dtype
        ///     that is not string data — bool for the logical ufuncs, StringDType beside a
        ///     StringDType operand — without a <paramref name="where"/> mask: the finalize error
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => add(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : NDCharArrayBoundary.Sees(x1, x2, @out)   // a numpy.char.chararray operand or out=: chararray.__array_wrap__'s view of a text result, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.WrappedUfunc((x1, x2, @out, where, dtype), static s => add(s.x1, s.x2, s.@out, s.where, s.dtype), x1, x2, @out)
            : x1.TensorEngine.Add(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.divide.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray divide(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => divide(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Divide(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray true_divide(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => true_divide(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Divide(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.multiply.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> input gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>), a plain array of a result of another dtype; a chararray
        ///     <paramref name="out"/> comes back as a new chararray view of it.
        /// </remarks>
        /// <exception cref="ValueError">A short <c>numpy.char.chararray</c> input (0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements) that NumPy's trivial loop copies into a loop dtype
        ///     that is not string data — bool for the logical ufuncs, StringDType beside a
        ///     StringDType operand — without a <paramref name="where"/> mask: the finalize error
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        public static NDArray multiply(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => multiply(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : NDCharArrayBoundary.Sees(x1, x2, @out)   // a numpy.char.chararray operand or out=: chararray.__array_wrap__'s view of a text result, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.WrappedUfunc((x1, x2, @out, where, dtype), static s => multiply(s.x1, s.x2, s.@out, s.where, s.dtype), x1, x2, @out)
            : x1.TensorEngine.Multiply(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.subtract.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray subtract(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => subtract(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Subtract(x1, x2, dtype, @out, where);

        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.mod.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray mod(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => mod(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Mod(x1, x2, dtype, @out, where);

        public static NDArray mod(NDArray x1, float x2)
            => NDMatrixBoundary.Sees(x1)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2), static s => mod(s.x1, s.x2))
            : x1.TensorEngine.Mod(x1, x2);

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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray remainder(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => remainder(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Mod(x1, x2, dtype, @out, where);

        public static NDArray remainder(NDArray x1, float x2)
            => NDMatrixBoundary.Sees(x1)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2), static s => remainder(s.x1, s.x2))
            : x1.TensorEngine.Mod(x1, x2);

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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray fmod(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(x1, x2)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => fmod(s.x1, s.x2, s.@out, s.where, s.dtype), @out)
            : x1.TensorEngine.Fmod(x1, x2, dtype, @out, where);

        // Scalar / array-like divisor convenience (the a % obj analog; asanyarray mints the operand).
        [NDScoped]
        public static NDArray fmod(NDArray x1, object x2)
            => NDMatrixBoundary.Sees(x1)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((x1, x2), static s => fmod(s.x1, s.x2))
            : x1.TensorEngine.Fmod(x1, np.asanyarray(x2));

        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        ///
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="a">The first one-dimensional input sequence (a 0-d array is one element, NumPy's <c>ndmin=1</c>).</param>
        /// <param name="v">The second one-dimensional input sequence (likewise); it is the one reversed, after the swap that
        ///     makes the longer sequence the data.</param>
        /// <param name="mode"><c>'full'</c> (the default: every overlap, <c>len(a) + len(v) - 1</c> points), <c>'same'</c>
        ///     (<c>max(len(a), len(v))</c> points, centred) or <c>'valid'</c> (only the full overlaps). The name is
        ///     case-sensitive, as in NumPy.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype (<c>result_type(a, v)</c>): the discrete, linear
        ///     convolution of <paramref name="a"/> and <paramref name="v"/>.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        ///
        /// NumPy's Python body (<c>numpy/_core/numeric.py</c>) runs in the engine member
        /// <see cref="TensorEngine.Convolve"/> (the engine of <paramref name="a"/>); its last
        /// statement, <c>multiarray.correlate(a, v[::-1], mode)</c>, is the engine member
        /// <see cref="TensorEngine.Correlate"/>. Unlike <see cref="correlate"/>, a complex kernel is NOT
        /// conjugated, and the swap needs no reversal of the result (convolution is commutative). The order of the errors
        /// is NumPy's: the two emptiness texts (on the operands as given) come before the mode, and the mode before the
        /// depth check. Both functions share one kernel; see <see cref="correlate"/> for its bit parity.
        /// </remarks>
        /// <exception cref="ValueError"><c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands as given, before
        ///     the length swap), then an unknown <paramref name="mode"/> (<c>Use one of 'valid', 'same', or 'full' for
        ///     convolve/correlate mode</c> when its first letter names a mode, else <c>mode must be one of 'valid', 'same', or
        ///     'full' (got 'x')</c> with the argument in Python's <c>repr</c>), <c>object too deep for desired array</c> for
        ///     an operand of two or more dimensions (a <c>numpy.matrix</c> operand included, as in NumPy), or — for a
        ///     fixed-width string operand, after every check above — <c>data type must provide an itemsize</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate (CPython's UTF-8 encoding
        ///     error, raised before the mode is read).</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>): <c>convolve/correlate mode not
        ///     understood</c>. A <c>StringDType</c> operand (after the emptiness checks and the swap): NumPy's legacy-only
        ///     descriptor refusal, or Python's error from the first product of the OBJECT fallback.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes (an
        ///     object array, which NumSharp does not have).</exception>
        public static NDArray convolve(NDArray a, NDArray v, string mode = "full")
            => a.TensorEngine.Convolve(a, v, mode);

        /// <summary>
        ///     Return the product of array elements over a given axis — NumPy's <c>np.prod(a, axis=None, dtype=None,
        ///     out=None, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e.
        ///     <c>np.multiply.reduce(a, axis, dtype, out, keepdims=keepdims, initial=initial, where=where)</c>. This is the only
        ///     overload: NumPy's whole signature, with NumPy's names, order and defaults (np-layer dtype plan P10, D10).
        /// </summary>
        /// <param name="a">Input data (any numeric dtype; text has no multiply loop and is refused).</param>
        /// <param name="axis">The axis to multiply over (negative counts from the end); <c>null</c> multiplies every element.
        ///     A 0-d array accepts 0 and -1 as every axis (ufunc.reduce's rule).</param>
        /// <param name="dtype">The accumulator and result dtype: every element is cast to it before it is multiplied.
        ///     <c>null</c> is NumPy's default: int64 for bool and the signed integers narrower than 64 bits, uint64 for the
        ///     unsigned ones, the input's own dtype otherwise — or, with <paramref name="out"/>, <c>promote_types(out, a)</c>
        ///     (no small-integer widening).</param>
        /// <param name="out">The array the product is written into and returned (the result's shape, any dtype: the values
        ///     are cast in unsafely). Footgun: the FOURTH positional parameter, as in NumPy, so write
        ///     <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every product starts from instead of 1, converted into the loop dtype as NumPy's
        ///     element setter converts a Python value (a C# integer range-checked, a <c>double</c> truncated into an integer
        ///     loop, a 0-d array cast). An empty product is the initial value. <c>null</c> is the identity 1.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: only the selected elements are multiplied
        ///     in; an output with none selected is the start value. The C# <c>true</c> literal is NumPy's default.
        ///     <c>null</c> is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a
        ///     product over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; a complex <paramref name="initial"/> for
        ///     a real loop; text (NumPy's refusal); a <c>numpy.matrix</c> operand with <paramref name="keepdims"/>,
        ///     <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.prod() got an unexpected keyword argument
        ///     'keepdims'</c>).</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes than
        ///     <paramref name="a"/>; an unconvertible <paramref name="initial"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.prod.html. With <paramref name="initial"/> or
        ///     <paramref name="where"/> the product is NumPy's sequential multiply loop through its reduction iterator (a
        ///     NaN keeps the payload of the first NaN met); without them a flat float product multiplies in SIMD lane order,
        ///     which can differ from NumPy's in the last bit (the open NL3).
        /// </remarks>
        public static NDArray prod(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (NDMatrixBoundary.Sees(a))
            {
                // NumPy hands a numpy.matrix to matrix.prod(axis, dtype, out) with every keyword the caller passed, in its
                // signature's order; the method takes none of them, so the first one passed is its TypeError.
                if (keepdims)
                    throw NDMatrixBoundary.KeywordError("prod", "keepdims");
                if (initial is not null)
                    throw NDMatrixBoundary.KeywordError("prod", "initial");
                if (@where is not null)
                    throw NDMatrixBoundary.KeywordError("prod", "where");
                if (@out is not null)
                    return NDMatrixBoundary.ReducedInto(a, @out, (axis, dtype),
                        static (m, o, s) => m.prod(s.axis, s.dtype, o),
                        static (m, o, s) => m.TensorEngine.ReduceProduct(m, s.axis, keepdims: true, s.dtype, o));
                // a row, a column or the scalar, as NumPy's matrix method returns
                return NDMatrixBoundary.Reduced((a, axis, dtype), static s => prod(s.a, s.axis, s.dtype), axis, false, "prod");
            }

            // The whole call is the engine's: TensorEngine.ReduceProduct of a's engine takes every keyword.
            return a.TensorEngine.ReduceProduct(a, axis, keepdims, dtype, @out, initial, @where);
        }

        /// <summary>
        ///     Numerical positive, element-wise (identity: returns +x, a copy).
        ///     Mirrors NumPy's ufunc signature: <c>positive(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): positive(i32, dtype: float64) widens; bool loop requests raise NumPy's did-not-contain-a-loop TypeError (positive has no bool loop, but positive(bool, dtype: float64) is legal).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.positive.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray positive(NDArray nd, NDArray @out = null, NDArray where = null, DType dtype = null)
            => NDMatrixBoundary.Sees(nd)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((nd, @out, where, dtype), static s => positive(s.nd, s.@out, s.where, s.dtype), @out)
            : nd.TensorEngine.Positive(nd, dtype, @out, where);

        /// <summary>
        ///     Numerical negative, element-wise.
        ///     Mirrors NumPy's ufunc signature: <c>negative(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): selects the loop, so negative(bool, dtype: float64) is legal while plain negative(bool) raises (NumPy parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.negative.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray negative(NDArray nd, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(nd))
                return NDMatrixBoundary.Wrapped((nd, @out, where, dtype), static s => negative(s.nd, s.@out, s.where, s.dtype), @out);
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
