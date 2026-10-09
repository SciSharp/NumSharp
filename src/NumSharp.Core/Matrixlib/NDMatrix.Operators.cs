using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

// =============================================================================
// NDMatrix.Operators.cs — matrix.__mul__ / __rmul__ / __imul__ / __pow__ / __ipow__ / __rpow__ and the operators a
// matrix inherits from ndarray
// =============================================================================
//
// `*` is the one operator numpy.matrix redefines (numpy/matrixlib/defmatrix.py):
//
//     def __mul__(self, other):
//         if isinstance(other, (N.ndarray, list, tuple)):
//             return N.dot(self, asmatrix(other))      # a 1-d operand becomes a ROW vector
//         if isscalar(other) or not hasattr(other, '__rmul__'):
//             return N.dot(self, other)                # scaling, in np.dot's dtype
//         return NotImplemented
//     def __rmul__(self, other):
//         return N.dot(other, self)                    # the left operand is NOT made a matrix
//
// Two consequences worth knowing:
//   * m * v with a 1-d v of length N is dot((M, N), (1, N)), which raises unless N == 1; write m * v.T-shaped
//     data as a column matrix. v * m is dot((N,), (N, K)) and works.
//   * scaling goes through np.dot, whose result dtype is np.promote_types of the two operand dtypes. It is NOT
//     the weak-scalar rule of the `*` ufunc: an int8 matrix times a 0-d int64 is int64, where an int8 array times
//     the same scalar stays int8. A C# literal carries its C# type (2 is int32, 2.5 is float64), the library-wide
//     convention; NumPy's Python int is int64.
//
// Every other operator is ndarray's ufunc, and its result is WRAPPED back into a matrix (__array_wrap__), so
// m + x, m / x, m > x ... are elementwise with broadcasting and return a matrix. A result that needs more than
// two axes raises "shape too large to be a matrix.".
//
// OVERLOADS. Each binary operator has five forms — (matrix, matrix), (matrix, array), (array, matrix),
// (matrix, object) and (object, matrix) — so an expression with a matrix on EITHER side is a matrix expression,
// whatever the other operand is. A C# number binds the (matrix, matrix) form through the implicit conversion to
// a 0-d matrix. C# has no `**`: use __pow__.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDMatrix
    {
        #region Helpers

        /// <summary>
        ///     The matrix for an elementwise (ufunc) result: a view of the hidden result, folded to two axes. A 0-d
        ///     result of two scalars stays a scalar.
        /// </summary>
        /// <param name="result">The plain result.</param>
        /// <param name="left">The left operand, or the only one.</param>
        /// <param name="right">The right operand, or <c>null</c> for a unary operator.</param>
        /// <returns>The matrix.</returns>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        private static NDMatrix Elementwise(NDArray result, NDArray left, NDArray right = null)
            => FromWrapped(result, left, right, scalarStaysScalar: true);

        /// <summary>
        ///     NumPy's <c>matrix.__mul__</c> and <c>__rmul__</c> in one place: the matrix product, or scaling when
        ///     either operand is 0-d.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand. At least one of the two is a matrix.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ (NumPy's
        ///     <c>shapes (2,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)</c>).</exception>
        /// <exception cref="ValueError">The right operand of a matrix has more than two axes longer than 1
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        [NDScopedCovered]
        private static NDMatrix Product(NDArray left, NDArray right)
        {
            if (left is null)
                throw new ArgumentNullException(nameof(left));
            if (right is null)
                throw new ArgumentNullException(nameof(right));

            if (left.ndim == 0 || right.ndim == 0)
            {
                // A plain 0-d array on the LEFT stands for a NumPy scalar (m.sum() * m, np.float64(2) * m). Python
                // asks the scalar first, and a NumPy scalar does not defer to matrix.__rmul__: it runs the multiply
                // ufunc and the matrix wraps the result. So this is the elementwise product (no np.dot zero-sign
                // rule, the operand's memory order, a view of the hidden result), in the dtype two NumPy scalars
                // promote to.
                if (left.ndim == 0 && left is not NDMatrix && right.ndim != 0)
                    return FromWrapped(np.multiply(left, right, dtype: np.promote_types(left.dtype, right.dtype)), left, right);

                // Every other 0-d operand is N.dot(self, scalar) / N.dot(scalar, self): a C# number on either side
                // (the 0-d matrix stand-in), or a 0-d array on the right. NumPy raises for an explicit 0-d ndarray
                // on the right (asmatrix makes it (1, 1)); NumSharp cannot tell it from a NumPy scalar and scales.
                return Scale(left, right);
            }

            if (left is NDMatrix)
            {
                // __mul__: the other operand is seen as a matrix first, which turns a 1-d array into a row.
                NDArray other = right is NDMatrix ? right : ViewOf(right);
                return From(np.dot(left, other), left, right);
            }

            // __rmul__: N.dot(other, self) with the left operand as it is; a 1-d result is folded to a row.
            return From(np.dot(left, right), left, right);
        }

        /// <summary>
        ///     <c>N.dot(a, scalar)</c>: the scaled array in the dtype <c>np.dot</c> gives two arrays,
        ///     <c>np.promote_types(a.dtype, scalar.dtype)</c>, computed the way NumPy's two routes compute it.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand; one of the two is 0-d.</param>
        /// <returns>The scaled matrix; 0-d when both operands are.</returns>
        /// <remarks>
        ///     <para>
        ///     <b>float32, float64, complex128</b> (<c>cblas_matrixproduct</c>'s scalar branch). The result is a new
        ///     C-contiguous array the matrix OWNS, whatever the operand's layout. NumPy fills it with zeros and
        ///     accumulates <c>?axpy</c> into it, so each element is <c>0 + scalar * x</c>: a product of
        ///     <c>-0.0</c> comes out as <c>+0.0</c>. A result of ONE element is the exception: it is the direct
        ///     product, which keeps <c>-0.0</c>.
        ///     </para>
        ///     <para>
        ///     <b>Every other dtype</b> (<c>PyArray_MatrixProduct2</c> without a BLAS type): the multiply ufunc,
        ///     whose result the matrix wraps as a view, in the operand's memory order.
        ///     </para>
        ///     <para>
        ///     <b>Three float routes, one result.</b>
        ///     (1) A C-contiguous float32 / float64 operand that already has the loop dtype (a float matrix times a
        ///     number, the common case) runs ONE direct pass over raw memory (<see cref="ScaleContiguous"/>).
        ///     (2) Any other operand of the loop dtype (<c>m.T</c>, a stepped view, every complex128 matrix) runs
        ///     one fused <c>np.evaluate</c> pass from <see cref="FusedScaleMinElements"/> elements on
        ///     (<see cref="FusedScaleMinElementsComplex"/> for complex128).
        ///     (3) The rest (an operand of another dtype, a smaller view or complex matrix, a host without
        ///     dynamic code) runs the multiply ufunc and then adds the zero in place.
        ///     All three give the bytes of <c>np.add(np.multiply(x, s, dtype: loop), 0)</c> (28,548 comparisons
        ///     over float32 / float64 / complex128, nine layouts, both operand orders, scalars of every dtype
        ///     that keeps the loop, special values, 2 to 300,300 elements), with one exception: when the scalar
        ///     AND an element are NaN, the payload that survives can differ (12 of the 28,548). NumPy's own bits
        ///     there come from the BLAS <c>?axpy</c> kernel, which no route models, and the oracle compares a NaN
        ///     as a NaN.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        private static NDMatrix Scale(NDArray left, NDArray right)
        {
            DType loop = np.promote_types(left.dtype, right.dtype);
            NPTypeCode code = loop.GetTypeCode();
            if (code != NPTypeCode.Single && code != NPTypeCode.Double && code != NPTypeCode.Complex)
                return FromWrapped(np.multiply(left, right, dtype: loop), left, right, scalarStaysScalar: true);

            // NumPy makes the scalar its second operand whichever side it stood on, so both orders give the
            // same bytes. When both operands are 0-d either choice is the scalar.
            NDArray array = left.ndim == 0 ? right : left;
            NDArray scalar = left.ndim == 0 ? left : right;

            if (array.size == 1)
            {
                // `if (l == 1)`: one product written straight into the result, the scalar first. For complex128
                // NumPy spells the textbook formula out; the ufunc's fused multiply could round differently.
                NDArray single = code == NPTypeCode.Complex
                    ? ComplexScalarProduct(scalar, array)
                    : np.multiply(scalar, array, dtype: loop);
                return From(single, left, right, scalarStaysScalar: true);
            }

            // AsClean: the operand's dimensions in C order, never its view layout.
            var result = new NDArray(loop, array.Shape.AsClean(), false);
            bool real = code != NPTypeCode.Complex;

            if (real && array.typecode == code && array.Shape.IsContiguous && !DisableDirectScale)
            {
                // The common case, a C-contiguous float matrix times a number of the same loop: ?axpy into zeros
                // as ONE pass over raw memory, with no ufunc dispatch and no 0-d operand to allocate.
                DirectScaleRuns++;
                ScaleContiguous(array, scalar, result, code);
                return From(result, left, right, scalarStaysScalar: true);
            }

            // The fused pass pays from the measured crossover on: earlier for a complex operand, whose second
            // pass costs more against its multiply than a float one's.
            long fusedFrom = FusedScaleMinElementsTestOverride > 0
                ? FusedScaleMinElementsTestOverride
                : real ? FusedScaleMinElements : FusedScaleMinElementsComplex;
            if (array.size >= fusedFrom && array.typecode == code && Backends.Kernels.DirectILKernelGenerator.Enabled)
            {
                FusedScaleRuns++;
                // One fused pass for an operand of the loop dtype that the direct pass cannot take (m.T, a
                // stepped view, a complex matrix): the scalar and the zero are hoisted 0-d parameters, the
                // operand is read once through its strides and the result written once.
                NDArray zero = code switch
                {
                    NPTypeCode.Single => NDArray.Scalar(0f),
                    NPTypeCode.Double => NDArray.Scalar(0d),
                    _ => NDArray.Scalar(Complex.Zero),
                };
                np.evaluate((NDExpr)array * (NDExpr)scalar + (NDExpr)zero, result);
            }
            else
            {
                // An operand of another dtype (an integer matrix times 2.5), a small float one in another layout,
                // or a host without dynamic code: the ufunc's loop, then the accumulator's addition in place.
                np.multiply(array, scalar, @out: result, dtype: loop);
                AddZeroInPlace(result, code);
            }

            return From(result, left, right, scalarStaysScalar: true);
        }

        /// <summary>
        ///     <c>result[i] = 0 + scalar * array[i]</c> for a C-contiguous float32 / float64 operand that already
        ///     has the loop dtype, as one pass over raw memory. The multiply and the add are rounded separately
        ///     (never a fused multiply-add), which is what NumPy's <c>?axpy</c> into zeros and the two ufunc passes
        ///     of <see cref="Scale"/> compute, so the bytes are the same; only the payload of a NaN that meets a
        ///     NaN scalar can differ.
        /// </summary>
        /// <param name="array">The operand: C-contiguous, of dtype <paramref name="code"/>, any rank, at any
        ///     offset into its buffer.</param>
        /// <param name="scalar">The 0-d operand, of any dtype that promotes into the loop; it is converted to the
        ///     loop dtype the way NumPy casts <c>np.dot</c>'s operands (a C cast).</param>
        /// <param name="result">A fresh C-contiguous array of the operand's size and dtype, at offset 0.</param>
        /// <param name="code"><see cref="NPTypeCode.Double"/> or <see cref="NPTypeCode.Single"/>.</param>
        private static unsafe void ScaleContiguous(NDArray array, NDArray scalar, NDArray result, NPTypeCode code)
        {
            long n = array.size;
            // A view at an offset keeps the buffer's base address and a non-zero Shape.offset, or re-seats the
            // address and keeps offset 0: base + offset * itemsize is the first element in both cases (the
            // operand and the 0-d scalar alike).
            if (code == NPTypeCode.Double)
            {
                // A scalar of the loop dtype (the C# double of `m * 2.0`) is read in place; any other dtype goes
                // through the house converter, which is the C cast NumPy applies to np.dot's operands.
                double factor = scalar.typecode == NPTypeCode.Double
                    ? ((double*)scalar.Storage.Address)[scalar.Shape.offset]
                    : Utilities.Converts.ChangeType<double>(scalar.Storage.GetAtIndex(0));
                double* source = (double*)array.Storage.Address + array.Shape.offset;
                Backends.Kernels.DirectILKernelGenerator.PolyAffineDouble(source, n, factor, 0.0, (double*)result.Storage.Address);
            }
            else
            {
                float factor = scalar.typecode == NPTypeCode.Single
                    ? ((float*)scalar.Storage.Address)[scalar.Shape.offset]
                    : Utilities.Converts.ChangeType<float>(scalar.Storage.GetAtIndex(0));
                float* source = (float*)array.Storage.Address + array.Shape.offset;
                Backends.Kernels.DirectILKernelGenerator.PolyAffineSingle(source, n, factor, 0f, (float*)result.Storage.Address);
            }

            // The kernel read raw pointers. Nothing above uses the operands after their addresses were taken, so
            // without this the GC may finalize a temporary operand (and pool its buffer) while the kernel runs.
            GC.KeepAlive(array);
            GC.KeepAlive(scalar);
            GC.KeepAlive(result);
        }

        /// <summary>
        ///     The second pass of the two-pass route: <c>r[i] = 0 + r[i]</c> in place, which turns a <c>-0.0</c>
        ///     product into NumPy's <c>+0.0</c> and changes nothing else. It runs the affine kernel with scale 1
        ///     (<c>x * 1</c> is exact for every value and keeps a NaN's payload), so no 0-d zero is allocated and no
        ///     ufunc is dispatched.
        /// </summary>
        /// <param name="result">A C-contiguous float32 / float64 / complex128 array at offset 0 that the caller
        ///     just filled.</param>
        /// <param name="code">The array's dtype: <see cref="NPTypeCode.Double"/>, <see cref="NPTypeCode.Single"/> or
        ///     <see cref="NPTypeCode.Complex"/>.</param>
        /// <remarks>A complex128 array is passed as twice as many float64 values: adding <c>0+0j</c> is adding
        ///     <c>0.0</c> to each part, so the float64 kernel computes exactly NumPy's complex addition.</remarks>
        private static unsafe void AddZeroInPlace(NDArray result, NPTypeCode code)
        {
            if (code == NPTypeCode.Single)
            {
                float* r = (float*)result.Storage.Address;
                Backends.Kernels.DirectILKernelGenerator.PolyAffineSingle(r, result.size, 1f, 0f, r);
            }
            else
            {
                double* r = (double*)result.Storage.Address;
                long parts = code == NPTypeCode.Complex ? result.size * 2 : result.size;
                Backends.Kernels.DirectILKernelGenerator.PolyAffineDouble(r, parts, 1.0, 0.0, r);
            }

            GC.KeepAlive(result);
        }

        /// <summary>
        ///     From this many elements on, a FLOAT operand that the direct pass cannot take (not C-contiguous)
        ///     is scaled in one fused <c>np.evaluate</c> pass instead of a multiply pass and an add pass. Raise it
        ///     and mid-size views pay a second pass over the result; lower it and small ones pay the fused call's
        ///     fixed cost. A complex128 operand uses <see cref="FusedScaleMinElementsComplex"/>.
        /// </summary>
        /// <remarks>
        ///     Measured on 2026-10-09 (<c>m * 2.0</c> with the literal's temporary included, two pinned P-cores,
        ///     the routes in one process, best of 21 interleaved rounds after each was compiled at tier 1;
        ///     microseconds, fused / two passes). A transposed float64 matrix: 16 elements 2.7 / 1.6; 256
        ///     elements 2.2 / 2.0; 1,024 elements 3.1 / 3.1; 2,025 elements 4.0 / 4.1; 4,096 elements 5.4 / 6.0;
        ///     8,100 elements 6.8 / 7.6; 65,536 elements 60 / 72; 262,144 elements 417 / 461.
        ///     For reference, the direct pass over a C-contiguous float64 matrix: 16 elements 0.74; 4,096
        ///     elements 2.8; 65,536 elements 16; 262,144 elements 117 (it beats both other routes at every size).
        ///     The fused call's cost at 16 elements moved between 1.4 and 2.8 microseconds from run to run; a
        ///     cell timed before tier 1 reads up to three times too slow, so warm every route twice, with a
        ///     pause, before timing.
        /// </remarks>
        private const long FusedScaleMinElements = 4096;

        /// <summary>
        ///     <see cref="FusedScaleMinElements"/> for a complex128 operand (any layout: the direct pass does not
        ///     take complex operands, because its kernel multiplies in the other operand order, which can round
        ///     the last bit differently).
        /// </summary>
        /// <remarks>
        ///     Measured with the float cells (microseconds, fused / two passes). A C-contiguous complex128
        ///     matrix: 16 elements 2.4 / 1.6; 256 elements 2.3 / 2.5; 1,024 elements 3.7 / 4.3; 4,096 elements
        ///     7.6 / 10.1; 65,536 elements 87 / 121; 262,144 elements 352 / 534. A transposed one: 16 elements
        ///     3.0 / 1.7; 256 elements 2.9 / 2.7; 1,024 elements 4.4 / 4.7; 4,096 elements 9.9 / 11.6; 262,144
        ///     elements 1,109 / 1,270. The routes cross between 256 and 1,024 elements.
        /// </remarks>
        private const long FusedScaleMinElementsComplex = 512;

        /// <summary>
        ///     Test hook: when positive it replaces <see cref="FusedScaleMinElements"/> for float AND complex
        ///     operands, so the oracle can replay its small matrices through the fused pass (2, in
        ///     <c>FuzzCorpusTests.Matrix_FusedScaling</c>) or keep every operand off it
        ///     (<see cref="long.MaxValue"/>, in <c>Matrix_TwoPassScaling</c>). Zero in production. Not thread-safe
        ///     by design: set it only around a serial replay.
        /// </summary>
        internal static long FusedScaleMinElementsTestOverride;

        /// <summary>How many times <see cref="Scale"/> took the fused pass (the proof for the test hook's replay).</summary>
        internal static int FusedScaleRuns;

        /// <summary>
        ///     Test hook: when set, <see cref="Scale"/> skips the direct pass over a C-contiguous float operand, so
        ///     the oracle can replay those operands through the fused pass and through the two ufunc passes
        ///     (<c>FuzzCorpusTests.Matrix_FusedScaling</c>, <c>Matrix_TwoPassScaling</c>). False in production. Not
        ///     thread-safe by design: set it only around a serial replay.
        /// </summary>
        internal static bool DisableDirectScale;

        /// <summary>How many times <see cref="Scale"/> took the direct pass (the proof that the default replay
        ///     of the oracle reaches it).</summary>
        internal static int DirectScaleRuns;

        /// <summary>
        ///     The one-element complex product of <c>cblas_matrixproduct</c>: <c>re = s.re*x.re - s.im*x.im</c>,
        ///     <c>im = s.re*x.im + s.im*x.re</c>, each multiplication and the sum rounded on its own.
        /// </summary>
        /// <param name="scalar">The 0-d operand, of any dtype (converted to complex128 first).</param>
        /// <param name="element">The one-element operand, of any dtype and any rank.</param>
        /// <returns>A new complex128 array of <paramref name="element"/>'s shape holding the product.</returns>
        private static NDArray ComplexScalarProduct(NDArray scalar, NDArray element)
        {
            Complex s = Utilities.Converts.ChangeType<Complex>(scalar.Storage.GetAtIndex(0));
            Complex x = Utilities.Converts.ChangeType<Complex>(element.Storage.GetAtIndex(0));
            var product = new Complex(s.Real * x.Real - s.Imaginary * x.Imaginary,
                                      s.Real * x.Imaginary + s.Imaginary * x.Real);

            var result = new NDArray(NPTypeCode.Complex, element.Shape.AsClean(), false);
            result.Storage.SetAtIndex(product, 0);
            return result;
        }

        /// <summary>
        ///     Python's <c>operator.index(n)</c> for a matrix power: an integer, a bool, or a 0-d integer array.
        /// </summary>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="TypeError">The exponent is anything else (a float, a list, a matrix): NumPy's
        ///     <c>exponent must be an integer</c>.</exception>
        /// <exception cref="OverflowException">The integer does not fit <see cref="int"/>.</exception>
        private static int PowerIndex(object exponent)
        {
            switch (exponent)
            {
                case int i: return i;
                case bool b: return b ? 1 : 0;
                case sbyte or byte or short or ushort or uint or long or ulong:
                    return checked((int)System.Convert.ToInt64(exponent));
                case BigInteger big:
                    return checked((int)big);
                // A 0-d integer array is an index; a matrix never is (it is two-dimensional).
                case NDArray { ndim: 0 } scalar when scalar is not NDMatrix && IsIntegerLike(scalar.typecode):
                    return checked((int)System.Convert.ToInt64(scalar.GetAtIndex(0)));
                default:
                    throw new TypeError("exponent must be an integer");
            }
        }

        #endregion

        #region Matrix product and power

        /// <summary>
        ///     NumPy's <c>matrix * matrix</c>: the MATRIX product (<c>np.dot</c>), not the elementwise one. A 0-d
        ///     operand (a C# number) scales. For the elementwise product use <c>np.multiply(a, b)</c>.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A new matrix of shape <c>(left rows, right columns)</c> in the promoted dtype.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">An array operand cannot be seen as a matrix.</exception>
        [NDScoped]
        public static NDMatrix operator *(NDMatrix left, NDMatrix right) => Product(left, right);

        /// <summary>
        ///     NumPy's <c>matrix * ndarray</c>: <c>np.dot(left, asmatrix(right))</c>. A 1-d <paramref name="right"/>
        ///     becomes a ROW <c>(1, N)</c> first, so the product only exists when the matrix has one column.
        /// </summary>
        /// <param name="left">The matrix.</param>
        /// <param name="right">The array; 0-d scales.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError"><paramref name="right"/> has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator *(NDMatrix left, NDArray right) => Product(left, right);

        /// <summary>
        ///     NumPy's <c>ndarray * matrix</c> (<c>matrix.__rmul__</c>): <c>np.dot(left, right)</c> with the array as
        ///     it is, so a 1-d <paramref name="left"/> of length M times an <c>(M, K)</c> matrix is a <c>(1, K)</c>
        ///     row.
        /// </summary>
        /// <param name="left">The array; 0-d scales.</param>
        /// <param name="right">The matrix.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">The product has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator *(NDArray left, NDMatrix right) => Product(left, right);

        /// <summary>NumPy's <c>matrix * other</c> for a list, a tuple, a C# array or a boxed number; the operand is
        ///     converted as <see cref="np.matrix(object, DType, bool)"/> converts its data.</summary>
        /// <param name="left">The matrix.</param>
        /// <param name="right">The other operand.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">The operand is ragged or cannot be seen as a matrix.</exception>
        /// <exception cref="NotSupportedException">The operand holds strings or other objects.</exception>
        [NDScoped]
        public static NDMatrix operator *(NDMatrix left, object right) => Product(left, AsOperand(right));

        /// <summary>NumPy's <c>other * matrix</c> for a list, a tuple, a C# array or a boxed number.</summary>
        /// <param name="left">The other operand.</param>
        /// <param name="right">The matrix.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException">An operand is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">The operand is ragged, or the product cannot be a matrix.</exception>
        /// <exception cref="NotSupportedException">The operand holds strings or other objects.</exception>
        [NDScoped]
        public static NDMatrix operator *(object left, NDMatrix right) => Product(AsOperand(left), right);

        /// <summary>Python's <c>m.__mul__(other)</c>: the same as <c>m * other</c>.</summary>
        /// <param name="other">A matrix, an array, a sequence or a number.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">The operand is ragged or cannot be seen as a matrix.</exception>
        /// <exception cref="NotSupportedException">The operand holds strings or other objects.</exception>
        [NDScoped]
        public NDMatrix __mul__(object other) => Product(this, AsOperand(other));

        /// <summary>Python's <c>m.__rmul__(other)</c>: <c>other * m</c>, with <paramref name="other"/> on the LEFT.</summary>
        /// <param name="other">An array, a sequence or a number.</param>
        /// <returns>The product as a new matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">The operand is ragged, or the product cannot be a matrix.</exception>
        /// <exception cref="NotSupportedException">The operand holds strings or other objects.</exception>
        [NDScoped]
        public NDMatrix __rmul__(object other) => Product(AsOperand(other), this);

        /// <summary>
        ///     Python's <c>m *= other</c>: computes <c>m * other</c> and stores it IN this matrix
        ///     (<c>self[:] = self * other</c>). C#'s own <c>m *= x</c> rebinds the variable to a new matrix instead;
        ///     call this method when other references to the matrix must see the product.
        /// </summary>
        /// <param name="other">A matrix, an array, a sequence or a number.</param>
        /// <returns>This matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The inner dimensions differ.</exception>
        /// <exception cref="ValueError">This matrix is read-only, or the product's shape is not this matrix's shape
        ///     (<c>could not broadcast input array from shape (2,3) into shape (2,2)</c>).</exception>
        /// <exception cref="NotSupportedException">The operand holds strings or other objects.</exception>
        /// <remarks>The product is cast to this matrix's dtype with an unsafe cast, as NumPy's assignment does:
        ///     an int matrix times 2.5 keeps truncated integers.</remarks>
        [NDScoped]
        public NDMatrix __imul__(object other)
        {
            NDMatrix product = Product(this, AsOperand(other));
            np.copyto(this, product, casting: "unsafe");
            return this;
        }

        /// <summary>
        ///     NumPy's <c>m ** n</c>: the matrix power (<c>np.linalg.matrix_power</c>), <c>m</c> multiplied by itself
        ///     <paramref name="n"/> times. Zero gives the identity, a negative power inverts first.
        /// </summary>
        /// <param name="n">The integer power.</param>
        /// <returns>This matrix ITSELF for <c>n == 1</c> (as NumPy); a new matrix otherwise. A negative power of
        ///     an integer matrix is float64.</returns>
        /// <exception cref="LinAlgError">The matrix is not square (<c>Last 2 dimensions of the array must be
        ///     square</c>), or a negative power of a singular matrix (<c>Singular matrix</c>).</exception>
        /// <exception cref="TypeError">A negative power of a dtype <c>np.linalg</c> does not support.</exception>
        [NDScoped]
        public NDMatrix __pow__(int n)
        {
            np.linalg.AssertStackedSquare(this);
            if (n == 1)
                return this;
            // a ** -1 is inv(a) itself in NumPy (the n == 1 shortcut after inverting), not a copy of it.
            if (n == -1)
                return FromWrapped(np.linalg.inv(this), this);
            NDArray power = np.linalg.matrix_power(this, n);
            // n == 0 builds the identity with empty_like (the matrix owns it); every other power is a matmul
            // ufunc result, which NumPy wraps.
            return n == 0 ? From(power, this) : FromWrapped(power, this);
        }

        /// <summary>NumPy's <c>m ** other</c> for an exponent of any kind: an integer, a bool or a 0-d integer
        ///     array is the power; anything else is refused.</summary>
        /// <param name="other">The exponent.</param>
        /// <returns>The power; see <see cref="__pow__(int)"/>.</returns>
        /// <exception cref="LinAlgError">The matrix is not square (tested BEFORE the exponent, as NumPy does), or
        ///     a negative power of a singular matrix.</exception>
        /// <exception cref="TypeError">The exponent is not an integer (<c>exponent must be an integer</c>).</exception>
        /// <exception cref="OverflowException">The exponent does not fit <see cref="int"/>.</exception>
        [NDScoped]
        public NDMatrix __pow__(object other)
        {
            np.linalg.AssertStackedSquare(this);
            return __pow__(PowerIndex(other));
        }

        /// <summary>
        ///     Python's <c>m **= other</c>: computes the power and stores it IN this matrix
        ///     (<c>self[:] = self ** other</c>).
        /// </summary>
        /// <param name="other">The exponent (see <see cref="__pow__(object)"/>).</param>
        /// <returns>This matrix.</returns>
        /// <exception cref="LinAlgError">The matrix is not square, or a negative power of a singular matrix.</exception>
        /// <exception cref="TypeError">The exponent is not an integer.</exception>
        /// <exception cref="ValueError">This matrix is read-only.</exception>
        /// <remarks>The result is cast to this matrix's dtype with an unsafe cast: the inverse of an integer
        ///     matrix is truncated.</remarks>
        [NDScoped]
        public NDMatrix __ipow__(object other)
        {
            NDMatrix power = __pow__(other);
            // n == 1 returns this matrix itself: nothing to store.
            if (!ReferenceEquals(power, this))
                np.copyto(this, power, casting: "unsafe");
            return this;
        }

        /// <summary>
        ///     Python's <c>other ** m</c>. NumPy's matrix does not define it (<c>__rpow__</c> returns
        ///     <c>NotImplemented</c>), so a number base is an error; an ARRAY base is ndarray's own elementwise power,
        ///     wrapped as a matrix.
        /// </summary>
        /// <param name="other">The base.</param>
        /// <returns><c>np.power(other, m)</c> as a matrix when <paramref name="other"/> is an array.</returns>
        /// <exception cref="TypeError"><paramref name="other"/> is not an array: Python's
        ///     <c>unsupported operand type(s) for ** or pow(): 'int' and 'matrix'</c>.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public NDMatrix __rpow__(object other)
        {
            if (other is NDArray array)
                return Elementwise(np.power(array, (NDArray)this), array, this);

            string name = other switch
            {
                null => "NoneType",
                bool => "bool",
                sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => "int",
                float or double => "float",
                Complex => "complex",
                string => "str",
                _ => other.GetType().Name,
            };
            throw new TypeError($"unsupported operand type(s) for ** or pow(): '{name}' and 'matrix'");
        }

        #endregion

        #region Arithmetic (elementwise)

        /// <summary>
        ///     Elementwise addition with broadcasting, returned as a matrix (NumPy's <c>matrix + x</c>). The dtype
        ///     follows the array operator of the same name.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator +(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left + (NDArray)right, left, right);

        /// <inheritdoc cref="operator +(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator +(NDMatrix left, NDArray right) => Elementwise((NDArray)left + right, left, right);

        /// <inheritdoc cref="operator +(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator +(NDArray left, NDMatrix right) => Elementwise(left + (NDArray)right, left, right);

        /// <inheritdoc cref="operator +(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator +(NDMatrix left, object right) => Elementwise((NDArray)left + right, left);

        /// <inheritdoc cref="operator +(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator +(object left, NDMatrix right) => Elementwise(left + (NDArray)right, right);

        /// <summary>Elementwise subtraction with broadcasting, returned as a matrix (NumPy's <c>matrix - x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator -(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left - (NDArray)right, left, right);

        /// <inheritdoc cref="operator -(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator -(NDMatrix left, NDArray right) => Elementwise((NDArray)left - right, left, right);

        /// <inheritdoc cref="operator -(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator -(NDArray left, NDMatrix right) => Elementwise(left - (NDArray)right, left, right);

        /// <inheritdoc cref="operator -(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator -(NDMatrix left, object right) => Elementwise((NDArray)left - right, left);

        /// <inheritdoc cref="operator -(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator -(object left, NDMatrix right) => Elementwise(left - (NDArray)right, right);

        /// <summary>
        ///     Elementwise TRUE division with broadcasting, returned as a matrix (NumPy's <c>matrix / x</c>).
        ///     Integer operands give float64. It is not a multiplication by an inverse: use <c>m * x.I</c> for that.
        /// </summary>
        /// <param name="left">The dividend.</param>
        /// <param name="right">The divisor.</param>
        /// <returns>The quotient as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator /(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left / (NDArray)right, left, right);

        /// <inheritdoc cref="operator /(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator /(NDMatrix left, NDArray right) => Elementwise((NDArray)left / right, left, right);

        /// <inheritdoc cref="operator /(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator /(NDArray left, NDMatrix right) => Elementwise(left / (NDArray)right, left, right);

        /// <inheritdoc cref="operator /(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator /(NDMatrix left, object right) => Elementwise((NDArray)left / right, left);

        /// <inheritdoc cref="operator /(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator /(object left, NDMatrix right) => Elementwise(left / (NDArray)right, right);

        /// <summary>Elementwise remainder with broadcasting, returned as a matrix (NumPy's <c>matrix % x</c>; the
        ///     result takes the divisor's sign).</summary>
        /// <param name="left">The dividend.</param>
        /// <param name="right">The divisor.</param>
        /// <returns>The remainder as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator %(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left % (NDArray)right, left, right);

        /// <inheritdoc cref="operator %(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator %(NDMatrix left, NDArray right) => Elementwise((NDArray)left % right, left, right);

        /// <inheritdoc cref="operator %(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator %(NDArray left, NDMatrix right) => Elementwise(left % (NDArray)right, left, right);

        /// <inheritdoc cref="operator %(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator %(NDMatrix left, object right) => Elementwise((NDArray)left % right, left);

        /// <inheritdoc cref="operator %(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator %(object left, NDMatrix right) => Elementwise(left % (NDArray)right, right);

        /// <summary>Elementwise negation, returned as a matrix (NumPy's <c>-matrix</c>).</summary>
        /// <param name="operand">The matrix.</param>
        /// <returns>The negated matrix (a view of the computed array).</returns>
        [NDScoped]
        public static NDMatrix operator -(NDMatrix operand) => Elementwise(-(NDArray)operand, operand);

        /// <summary>NumPy's <c>+matrix</c>: the <c>np.positive</c> ufunc, a copy returned as a matrix.</summary>
        /// <param name="operand">The matrix.</param>
        /// <returns>A new matrix with the same elements (a view of the computed array).</returns>
        /// <exception cref="TypeError">The matrix is bool: <c>np.positive</c> has no bool loop, as in NumPy
        ///     (<c>ufunc 'positive' did not contain a loop with signature matching types ...</c>).</exception>
        [NDScoped]
        public static NDMatrix operator +(NDMatrix operand) => Elementwise(np.positive(operand), operand);

        #endregion

        #region Bitwise and shifts (elementwise)

        /// <summary>Elementwise AND with broadcasting, returned as a matrix (NumPy's <c>matrix &amp; x</c>;
        ///     logical for bool, bitwise for integers).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator &(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left & (NDArray)right, left, right);

        /// <inheritdoc cref="operator &amp;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator &(NDMatrix left, NDArray right) => Elementwise((NDArray)left & right, left, right);

        /// <inheritdoc cref="operator &amp;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator &(NDArray left, NDMatrix right) => Elementwise(left & (NDArray)right, left, right);

        /// <inheritdoc cref="operator &amp;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator &(NDMatrix left, object right) => Elementwise((NDArray)left & right, left);

        /// <inheritdoc cref="operator &amp;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator &(object left, NDMatrix right) => Elementwise(left & (NDArray)right, right);

        /// <summary>Elementwise OR with broadcasting, returned as a matrix (NumPy's <c>matrix | x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator |(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left | (NDArray)right, left, right);

        /// <inheritdoc cref="operator |(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator |(NDMatrix left, NDArray right) => Elementwise((NDArray)left | right, left, right);

        /// <inheritdoc cref="operator |(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator |(NDArray left, NDMatrix right) => Elementwise(left | (NDArray)right, left, right);

        /// <inheritdoc cref="operator |(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator |(NDMatrix left, object right) => Elementwise((NDArray)left | right, left);

        /// <inheritdoc cref="operator |(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator |(object left, NDMatrix right) => Elementwise(left | (NDArray)right, right);

        /// <summary>Elementwise XOR with broadcasting, returned as a matrix (NumPy's <c>matrix ^ x</c>). It is NOT
        ///     a power: Python's <c>**</c> is <see cref="__pow__(int)"/>.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator ^(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left ^ (NDArray)right, left, right);

        /// <inheritdoc cref="operator ^(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ^(NDMatrix left, NDArray right) => Elementwise((NDArray)left ^ right, left, right);

        /// <inheritdoc cref="operator ^(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ^(NDArray left, NDMatrix right) => Elementwise(left ^ (NDArray)right, left, right);

        /// <inheritdoc cref="operator ^(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ^(NDMatrix left, object right) => Elementwise((NDArray)left ^ right, left);

        /// <inheritdoc cref="operator ^(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ^(object left, NDMatrix right) => Elementwise(left ^ (NDArray)right, right);

        /// <summary>Elementwise bitwise NOT, returned as a matrix (NumPy's <c>~matrix</c>; logical NOT for bool).</summary>
        /// <param name="operand">The matrix.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        [NDScoped]
        public static NDMatrix operator ~(NDMatrix operand) => Elementwise(~(NDArray)operand, operand);

        /// <summary>Elementwise logical NOT as a bool matrix (NumSharp's <c>!array</c>, NumPy's
        ///     <c>np.logical_not</c>).</summary>
        /// <param name="operand">The matrix.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        [NDScoped]
        public static NDMatrix operator !(NDMatrix operand) => Elementwise(!(NDArray)operand, operand);

        /// <summary>Elementwise left shift with broadcasting, returned as a matrix (NumPy's <c>matrix &lt;&lt; x</c>).
        ///     C# requires the left operand of a shift operator to be the declaring type, so there is no
        ///     (array, matrix) form: write <c>np.asmatrix(np.left_shift(a, m))</c>.</summary>
        /// <param name="left">The values.</param>
        /// <param name="right">The shift counts.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator <<(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left << (NDArray)right, left, right);

        /// <inheritdoc cref="operator &lt;&lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <<(NDMatrix left, NDArray right) => Elementwise((NDArray)left << right, left, right);

        /// <inheritdoc cref="operator &lt;&lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <<(NDMatrix left, object right) => Elementwise((NDArray)left << right, left);

        /// <summary>Elementwise right shift with broadcasting, returned as a matrix (NumPy's <c>matrix &gt;&gt; x</c>;
        ///     arithmetic for signed integers).</summary>
        /// <param name="left">The values.</param>
        /// <param name="right">The shift counts.</param>
        /// <returns>The result as a matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator >>(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left >> (NDArray)right, left, right);

        /// <inheritdoc cref="operator &gt;&gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >>(NDMatrix left, NDArray right) => Elementwise((NDArray)left >> right, left, right);

        /// <inheritdoc cref="operator &gt;&gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >>(NDMatrix left, object right) => Elementwise((NDArray)left >> right, left);

        #endregion

        #region Comparisons (elementwise, bool matrix)

        // REFLECTED FORMS. With the matrix on the RIGHT (array < matrix, 2 < matrix, [1, 2] == matrix) Python
        // asks the matrix first, because its type is a subclass of the left operand's or the left operand cannot
        // answer: the call is matrix.__gt__(left), np.greater(matrix, left). The values are those of the
        // written comparison, but the matrix is the FIRST ufunc operand, which is the order NumPy's broadcast
        // error lists the shapes in. The (array, matrix) and (object, matrix) operators below therefore
        // evaluate the mirrored comparison with the matrix on the left.

        /// <summary>
        ///     Elementwise equality with broadcasting as a BOOL MATRIX (NumPy's <c>matrix == x</c>). It is not a
        ///     reference or a whole-matrix test: use <c>is null</c> for a null check, <see cref="Equals(object)"/>
        ///     or <c>np.array_equal</c> for "all elements equal".
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array). A comparison with <c>null</c> is a 0-d bool.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator ==(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left == (NDArray)right, left, right);

        /// <inheritdoc cref="operator ==(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ==(NDMatrix left, NDArray right) => Elementwise((NDArray)left == right, left, right);

        /// <inheritdoc cref="operator ==(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ==(NDArray left, NDMatrix right) => Elementwise((NDArray)right == left, left, right);

        /// <inheritdoc cref="operator ==(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ==(NDMatrix left, object right) => Elementwise((NDArray)left == right, left);

        /// <inheritdoc cref="operator ==(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator ==(object left, NDMatrix right) => Elementwise((NDArray)right == left, right);

        /// <summary>Elementwise inequality with broadcasting as a bool matrix (NumPy's <c>matrix != x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator !=(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left != (NDArray)right, left, right);

        /// <inheritdoc cref="operator !=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator !=(NDMatrix left, NDArray right) => Elementwise((NDArray)left != right, left, right);

        /// <inheritdoc cref="operator !=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator !=(NDArray left, NDMatrix right) => Elementwise((NDArray)right != left, left, right);

        /// <inheritdoc cref="operator !=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator !=(NDMatrix left, object right) => Elementwise((NDArray)left != right, left);

        /// <inheritdoc cref="operator !=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator !=(object left, NDMatrix right) => Elementwise((NDArray)right != left, right);

        /// <summary>Elementwise <c>&lt;</c> with broadcasting as a bool matrix (NumPy's <c>matrix &lt; x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator <(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left < (NDArray)right, left, right);

        /// <inheritdoc cref="operator &lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <(NDMatrix left, NDArray right) => Elementwise((NDArray)left < right, left, right);

        /// <inheritdoc cref="operator &lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <(NDArray left, NDMatrix right) => Elementwise((NDArray)right > left, left, right);

        /// <inheritdoc cref="operator &lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <(NDMatrix left, object right) => Elementwise((NDArray)left < right, left);

        /// <inheritdoc cref="operator &lt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <(object left, NDMatrix right) => Elementwise((NDArray)right > left, right);

        /// <summary>Elementwise <c>&gt;</c> with broadcasting as a bool matrix (NumPy's <c>matrix &gt; x</c>). The
        ///     result indexes a matrix directly: <c>m[m &gt; 5]</c>.</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator >(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left > (NDArray)right, left, right);

        /// <inheritdoc cref="operator &gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >(NDMatrix left, NDArray right) => Elementwise((NDArray)left > right, left, right);

        /// <inheritdoc cref="operator &gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >(NDArray left, NDMatrix right) => Elementwise((NDArray)right < left, left, right);

        /// <inheritdoc cref="operator &gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >(NDMatrix left, object right) => Elementwise((NDArray)left > right, left);

        /// <inheritdoc cref="operator &gt;(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >(object left, NDMatrix right) => Elementwise((NDArray)right < left, right);

        /// <summary>Elementwise <c>&lt;=</c> with broadcasting as a bool matrix (NumPy's <c>matrix &lt;= x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator <=(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left <= (NDArray)right, left, right);

        /// <inheritdoc cref="operator &lt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <=(NDMatrix left, NDArray right) => Elementwise((NDArray)left <= right, left, right);

        /// <inheritdoc cref="operator &lt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <=(NDArray left, NDMatrix right) => Elementwise((NDArray)right >= left, left, right);

        /// <inheritdoc cref="operator &lt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <=(NDMatrix left, object right) => Elementwise((NDArray)left <= right, left);

        /// <inheritdoc cref="operator &lt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator <=(object left, NDMatrix right) => Elementwise((NDArray)right >= left, right);

        /// <summary>Elementwise <c>&gt;=</c> with broadcasting as a bool matrix (NumPy's <c>matrix &gt;= x</c>).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A bool matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">The broadcast result has more than two axes longer than 1.</exception>
        [NDScoped]
        public static NDMatrix operator >=(NDMatrix left, NDMatrix right) => Elementwise((NDArray)left >= (NDArray)right, left, right);

        /// <inheritdoc cref="operator &gt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >=(NDMatrix left, NDArray right) => Elementwise((NDArray)left >= right, left, right);

        /// <inheritdoc cref="operator &gt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >=(NDArray left, NDMatrix right) => Elementwise((NDArray)right <= left, left, right);

        /// <inheritdoc cref="operator &gt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >=(NDMatrix left, object right) => Elementwise((NDArray)left >= right, left);

        /// <inheritdoc cref="operator &gt;=(NDMatrix, NDMatrix)"/>
        [NDScoped]
        public static NDMatrix operator >=(object left, NDMatrix right) => Elementwise((NDArray)right <= left, right);

        #endregion
    }
}
