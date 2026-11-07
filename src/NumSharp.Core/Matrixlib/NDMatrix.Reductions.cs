using System;
using System.Globalization;
using NumSharp.Backends;

// =============================================================================
// NDMatrix.Reductions.cs — matrix.sum / mean / prod / std / var / any / all / max / min / argmax / argmin / ptp
// =============================================================================
//
// NumPy's matrix reductions keep the result two-dimensional (numpy/matrixlib/defmatrix.py):
//
//     def sum(self, axis=None, dtype=None, out=None):
//         return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)
//     def _collapse(self, axis):  return self[0, 0] if axis is None else self
//
//     def argmax(self, axis=None, out=None):
//         return N.ndarray.argmax(self, axis, out)._align(axis)
//     def _align(self, axis):
//         if axis is None: return self[0, 0]
//         elif axis == 0:  return self
//         elif axis == 1:  return self.transpose()
//         else:            raise ValueError("unsupported axis")
//
// So axis=None gives a SCALAR, axis 0 a row (1, N) and axis 1 a column (M, 1). The two families differ for a
// negative axis: sum(-1) is the column, while argmax(-1) and ptp(-1) raise "unsupported axis" AFTER the reduction
// ran (so an axis that is out of bounds is still NumPy's AxisError).
//
// C# TYPING. Each reduction has two overloads: (int axis, ...) returns the matrix, and (int? axis = null, ...)
// returns an NDArray, which is the 0-d scalar for axis null and the matrix (as an NDArray reference) otherwise.
// The base class's overloads with keepdims / where / initial are NOT hidden: a call that names one of those
// keywords binds the NDArray member and reduces the matrix as a plain array, where NumPy's matrix method raises
// TypeError for the unknown keyword.
//
// out=. NumPy passes out through with keepdims=True, so out must be a MATRIX of the kept shape ((1, N), (M, 1),
// or (1, 1) for axis=None, where the call still returns the scalar). The parameter is typed NDMatrix for the
// reductions where that works; a plain array there makes NumPy write the result and then fail on out._collapse.
// For any / all / argmax / argmin / ptp NumPy's own code cannot succeed with an out at all; the parameter exists
// for signature parity and reproduces NumPy's failure.
//
// THE LOOP WITH out=. A reduction without dtype= resolves its loop from out AND the operand
// (ufunc_object.c reducelike_promote_and_resolve: the operand dtypes it promotes are (out, arr, out)), and it does
// not widen small integers when out is given. So the loop is promote_types(out.dtype, arr.dtype): an int8 matrix
// summed into an int8 out wraps in int8, into a bool out it sums in int8 and stores the truth value, into a
// float64 out a float32 matrix is summed in float64. The result reaches out with an UNSAFE cast. That rule, the text
// routes into out and _mean / _var / _std's statements around it are the engine's out= forms (TensorEngine.ReduceAdd /
// ReduceProduct / ReduceMean / ReduceVar / ReduceStd, DefaultEngine's Default.Reduction.Out.cs since the np-layer
// dtype plan's P1d): the members below call them with keepdims=True and read the scalar of axis=None from out.
//
// SCALARS. axis=None is `...(keepdims=True)[0, 0]`: the reduction always runs in its ARRAY form and the scalar is
// read from the (1, 1) result. That matters where NumPy's array and scalar branches differ: a float16 mean rounds
// twice (float32 sum to float16), and std with an integer dtype= raises the sqrt cast error where np.std of the same
// data as a plain array returns a truncated scalar.
//
// THE PLAIN COMPUTATION. NumPy's method body is `N.ndarray.sum(self, ...)`: the ndarray computation runs, and the matrix
// method shapes its result. Each member below computes on `self = A` (the matrix's plain base view) and a plain view of
// a matrix out (NDSubclass.PlainView): a public function handed the matrix itself would take its own subclass path,
// whose generated method dispatch calls this very member again (np.max(m) -> m.max(...) -> np.max(m) -> ...). The
// views are made inside the member's NDScope, which disposes them on return; a result never views them (a reduction
// allocates), and the member returns the caller's out, never its view.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDMatrix
    {
        #region Helpers

        /// <summary>Whether <paramref name="code"/> is an integer dtype (signed, unsigned or NumSharp's char).</summary>
        /// <param name="code">The type code.</param>
        /// <returns><c>true</c> for the eight integer widths and <see cref="NPTypeCode.Char"/>.</returns>
        private static bool IsIntegerLike(NPTypeCode code) => code switch
        {
            NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char => true,
            _ => false,
        };

        /// <summary>
        ///     NumPy's <c>self[0, 0]</c> of a <c>(1, 1)</c> reduction result that was written into
        ///     <paramref name="out"/>: the scalar a matrix reduction with <c>axis=None</c> returns.
        /// </summary>
        /// <param name="out">The <c>(1, 1)</c> matrix holding the result.</param>
        /// <returns>A new 0-d array holding a COPY of the element, as a NumPy scalar is independent of the
        ///     array it was read from.</returns>
        [NDScopedCovered]
        private static NDArray ScalarFrom(NDMatrix @out)
        {
            var element = new NDArray(@out.Storage.GetView(Slice.Index(0), Slice.Index(0)));
            return element.copy();
        }

        /// <summary>
        ///     NumPy's <c>_collapse(None)</c>, <c>self[0, 0]</c>, of a <c>(1, 1)</c> reduction result this class
        ///     computed itself: the scalar a matrix reduction with <c>axis=None</c> returns.
        /// </summary>
        /// <param name="kept">The fresh <c>(1, 1)</c> result of a reduction run with <c>keepdims: true</c>.</param>
        /// <returns>A new read-only 0-d array holding the element (NumSharp's form of a NumPy scalar).</returns>
        [NDScopedCovered]
        private static NDArray Collapse(NDArray kept)
        {
            NDArray element = kept.reshape(Shape.NewScalar());
            return element.copy().MarkReductionScalar();
        }

        /// <summary>
        ///     NumPy's <c>_align(axis)</c> for a reduction that drops the axis (<c>argmax</c>, <c>argmin</c>,
        ///     <c>ptp</c>): the 1-d result as a row for axis 0, as a column for axis 1.
        /// </summary>
        /// <param name="result">The 1-d result of the plain reduction.</param>
        /// <param name="axis">The axis exactly as the caller passed it.</param>
        /// <param name="wrapped">The reduction is a ufunc expression whose result the matrix wraps as a view
        ///     (<c>ptp</c>: <c>subtract(maximum.reduce, minimum.reduce)</c>). Otherwise NumPy allocates the result
        ///     as a matrix that owns its data (<c>argmax</c>, <c>argmin</c>).</param>
        /// <returns>The matrix: <c>(1, N)</c> for axis 0, <c>(M, 1)</c> (the transposed row, a view) for axis 1.</returns>
        /// <exception cref="ValueError">Any other axis value, also a valid negative one (<c>unsupported axis</c>).</exception>
        [NDScopedCovered]
        private NDMatrix Align(NDArray result, int axis, bool wrapped)
        {
            if (axis != 0 && axis != 1)
                throw new ValueError("unsupported axis");

            NDMatrix row;
            if (wrapped)
                row = FromWrapped(result, this);
            else
            {
                // The index reduction of a length-1 axis may hand back a view of a scratch buffer; NumPy's result
                // is always an owner.
                row = From(result.Storage.OwnsData ? result : result.copy(), this);
            }

            return axis == 0 ? row : row.T;
        }

        #endregion

        #region sum / prod / mean

        /// <summary>
        ///     NumPy's <c>matrix.sum(axis, dtype, out)</c> for a given axis: the sums as a matrix that keeps the
        ///     reduced axis with length 1 (<c>(1, N)</c> for axis 0, <c>(M, 1)</c> for axis 1 or -1).
        /// </summary>
        /// <param name="axis">The axis to sum along (0, 1, -1 or -2).</param>
        /// <param name="dtype">The dtype the sum is computed in, or <c>null</c> for NumPy's default (integers
        ///     narrower than 64 bits and bool accumulate in int64 / uint64).</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>. Without
        ///     <paramref name="dtype"/> the sum is computed in <c>promote_types(out.dtype, this.dtype)</c> and
        ///     stored with an unsafe cast: an int8 matrix into an int8 out wraps, into a bool out it stores
        ///     whether the int8 sum is non-zero.</param>
        /// <returns>The sums; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape
        ///     (NumPy's reduction texts).</exception>
        public NDMatrix sum(int axis, DType dtype = null, NDMatrix @out = null) => (NDMatrix)sum((int?)axis, dtype, @out);

        /// <summary>
        ///     NumPy's <c>matrix.sum(axis=None, dtype=None, out=None)</c>. Without an axis the result is the sum of
        ///     all elements as a scalar.
        /// </summary>
        /// <param name="axis">The axis, or <c>null</c> to sum every element.</param>
        /// <param name="dtype">The dtype the sum is computed in, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A matrix of the kept shape (<c>(1, 1)</c> without an axis), or <c>null</c>.</param>
        /// <returns>A 0-d array (the scalar) when <paramref name="axis"/> is <c>null</c>; otherwise the matrix of
        ///     <see cref="sum(int, DType, NDMatrix)"/> as an <see cref="NDArray"/> reference.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray sum(int? axis = null, DType dtype = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                if (axis is null)
                    return TensorEngine.ReduceAdd(self, null, keepdims: false, dtype);
                return FromWrapped(TensorEngine.ReduceAdd(self, axis, keepdims: true, dtype), this);
            }

            // add.reduce(self, axis, dtype, out, keepdims=True) is the engine's out= route: the text loop resolution (the
            // concatenation computes into out, everything else is NumPy's refusal), the reduce prologue, the loop
            // promote_types(out, self) when dtype is null (the file header's out= rule), the unsafe store.
            TensorEngine.ReduceAdd(self, axis, keepdims: true, dtype, outView);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        /// <summary>NumPy's <c>matrix.prod(axis, dtype, out)</c> for a given axis: the products as a matrix that
        ///     keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to multiply along.</param>
        /// <param name="dtype">The dtype the product is computed in, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <returns>The products; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        public NDMatrix prod(int axis, DType dtype = null, NDMatrix @out = null) => (NDMatrix)prod((int?)axis, dtype, @out);

        /// <summary>NumPy's <c>matrix.prod(axis=None, dtype=None, out=None)</c>; a scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> to multiply every element.</param>
        /// <param name="dtype">The dtype the product is computed in, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A matrix of the kept shape, or <c>null</c>.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray prod(int? axis = null, DType dtype = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                if (axis is null)
                    return TensorEngine.ReduceProduct(self, null, keepdims: false, dtype);
                return FromWrapped(TensorEngine.ReduceProduct(self, axis, keepdims: true, dtype), this);
            }

            // multiply.reduce(self, axis, dtype, out, keepdims=True) is the engine's out= route: the text loop resolution
            // (integer counts into a text out repeat, everything else is NumPy's refusal), the reduce prologue, the loop
            // promote_types(out, self) when dtype is null (the file header's out= rule), the unsafe store.
            TensorEngine.ReduceProduct(self, axis, keepdims: true, dtype, outView);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        /// <summary>NumPy's <c>matrix.mean(axis, dtype, out)</c> for a given axis: the means as a matrix that
        ///     keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to average along.</param>
        /// <param name="dtype">The dtype the mean is computed in, or <c>null</c>: float64 for an integer or bool
        ///     matrix, the matrix's own dtype for a float one.</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <returns>The means; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        public NDMatrix mean(int axis, DType dtype = null, NDMatrix @out = null) => (NDMatrix)mean((int?)axis, dtype, @out);

        /// <summary>NumPy's <c>matrix.mean(axis=None, dtype=None, out=None)</c>; a scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> for the mean of every element.</param>
        /// <param name="dtype">The dtype the mean is computed in, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A matrix of the kept shape, or <c>null</c>. NumPy stores the SUM into it and then
        ///     divides it in place, both with an unsafe cast, so an integer <paramref name="out"/> holds
        ///     truncated values.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray mean(int? axis = null, DType dtype = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                // Always the array form (keepdims), see the file header: a float16 mean rounds twice there.
                NDArray kept = TensorEngine.ReduceMean(self, axis, keepdims: true, dtype);
                if (axis is null)
                    return Collapse(kept);
                // _mean ends a float16 mean with `arr.dtype.type(ret)`, a conversion that allocates the matrix;
                // every other mean is the true_divide ufunc's result, which the matrix wraps. Which one ran is _mean's
                // own rule (is_float16_result), the engine's.
                return DefaultEngine.MeanIsFloat16Result(this, dtype) ? From(kept, this) : FromWrapped(kept, this);
            }

            // _mean(self, axis, dtype, out, keepdims=True) is the engine's out= route: umr_sum(arr, axis, dtype, out,
            // keepdims) stores the sum into out (dtype=, else float64 for integers and bool, float32 for float16, else the
            // reduce rule for a missing dtype with an out), then true_divide(out, rcount, out=out, casting='unsafe').
            TensorEngine.ReduceMean(self, axis, keepdims: true, dtype, outView);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        #endregion

        #region var / std

        /// <summary>NumPy's <c>matrix.var(axis, dtype, out, ddof)</c> for a given axis: the variances as a matrix
        ///     that keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to compute along.</param>
        /// <param name="dtype">The dtype of the computation, or <c>null</c>: float64 for an integer or bool
        ///     matrix, the matrix's dtype for a float one.</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>N - ddof</c>.</param>
        /// <returns>The variances; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        public NDMatrix var(int axis, DType dtype = null, NDMatrix @out = null, int ddof = 0) => (NDMatrix)var((int?)axis, dtype, @out, ddof);

        /// <summary>NumPy's <c>matrix.var(axis=None, dtype=None, out=None, ddof=0)</c>; a scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> for the variance of every element.</param>
        /// <param name="dtype">The dtype of the computation, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A matrix of the kept shape, or <c>null</c>.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <remarks>
        ///     With <paramref name="out"/> NumPy hands it to the LAST sum only (the sum of the squared deviations):
        ///     the mean and the deviations are computed as without it, that sum runs in
        ///     <c>promote_types(out.dtype, deviation dtype)</c> unless <paramref name="dtype"/> is given, it is
        ///     stored into <paramref name="out"/> with an unsafe cast, and <paramref name="out"/> is then divided
        ///     in place by <c>max(N - ddof, 0)</c>.
        /// </remarks>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray var(int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                NDArray kept = TensorEngine.ReduceVar(self, axis, keepdims: true, ddof, dtype);
                return axis is null ? Collapse(kept) : FromWrapped(kept, this);
            }

            // _var(self, axis, dtype, out, ddof, keepdims=True) is the engine's out= route: out joins the last sum only,
            // then is divided in place.
            TensorEngine.ReduceVar(self, axis, keepdims: true, ddof, dtype, outView);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        /// <summary>NumPy's <c>matrix.std(axis, dtype, out, ddof)</c> for a given axis: the standard deviations
        ///     as a matrix that keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to compute along.</param>
        /// <param name="dtype">The dtype of the computation, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A FLOAT or complex matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <returns>The standard deviations; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> or <paramref name="dtype"/> is an integer or
        ///     bool dtype: NumPy's <c>Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64') with
        ///     casting rule 'same_kind'</c>.</exception>
        public NDMatrix std(int axis, DType dtype = null, NDMatrix @out = null, int ddof = 0) => (NDMatrix)std((int?)axis, dtype, @out, ddof);

        /// <summary>NumPy's <c>matrix.std(axis=None, dtype=None, out=None, ddof=0)</c>; a scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> for the standard deviation of every element.</param>
        /// <param name="dtype">The dtype of the computation, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">A float or complex matrix of the kept shape, or <c>null</c>.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="TypeError">
        ///     An integer or bool <paramref name="out"/> or <paramref name="dtype"/>: NumPy takes the square root
        ///     in place under the <c>same_kind</c> rule, which refuses the float root for an integer array
        ///     (<c>Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>). With <paramref name="out"/> this is raised AFTER the variance was stored, so
        ///     <paramref name="out"/> holds the variance then. It is raised for <c>axis: null</c> too: a matrix
        ///     reduction is always the array form (file header), where <c>np.std</c> of a plain array returns a
        ///     truncated scalar.
        /// </exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray std(int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                NDArray kept = TensorEngine.ReduceStd(self, axis, keepdims: true, ddof, dtype);
                return axis is null ? Collapse(kept) : FromWrapped(kept, this);
            }

            // _std(self, axis, dtype, out, ddof, keepdims=True) is the engine's out= route: _var into out, then
            // sqrt(out, out=out), whose ufunc refuses an integer or bool out after the variance is in it.
            TensorEngine.ReduceStd(self, axis, keepdims: true, ddof, dtype, outView);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        #endregion

        #region any / all

        /// <summary>NumPy's <c>matrix.any(axis)</c> for a given axis: whether any element along the axis is true,
        ///     as a bool matrix that keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to test along.</param>
        /// <param name="out">Must be <c>null</c>. NumPy's <c>matrix.any</c> hands an out to the argument slot of
        ///     <c>dtype</c>, so every call with one fails.</param>
        /// <returns>The bool matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is not <c>null</c> (NumPy's
        ///     <c>Cannot construct a dtype from an array</c>).</exception>
        public NDMatrix any(int axis, NDArray @out = null) => (NDMatrix)any((int?)axis, @out);

        /// <summary>NumPy's <c>matrix.any(axis=None, out=None)</c>; a bool scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> to test every element.</param>
        /// <param name="out">Must be <c>null</c> (see <see cref="any(int, NDArray)"/>).</param>
        /// <returns>A 0-d bool array when <paramref name="axis"/> is <c>null</c>; otherwise the bool matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is not <c>null</c>.</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray any(int? axis = null, NDArray @out = null)
        {
            NDArray self = A;   // the plain computation (file header)
            if (@out is not null)
                throw new TypeError("Cannot construct a dtype from an array");
            if (axis is null)
                return np.any(self, axis: null, @out: null, keepdims: false);
            return FromWrapped(np.any(self, axis: axis, @out: null, keepdims: true), this);
        }

        /// <summary>NumPy's <c>matrix.all(axis)</c> for a given axis: whether every element along the axis is
        ///     true, as a bool matrix that keeps the reduced axis with length 1.</summary>
        /// <param name="axis">The axis to test along.</param>
        /// <param name="out">Must be <c>null</c> (NumPy's <c>matrix.all</c> cannot take one; see
        ///     <see cref="any(int, NDArray)"/>).</param>
        /// <returns>The bool matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is not <c>null</c>.</exception>
        public NDMatrix all(int axis, NDArray @out = null) => (NDMatrix)all((int?)axis, @out);

        /// <summary>NumPy's <c>matrix.all(axis=None, out=None)</c>; a bool scalar without an axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> to test every element.</param>
        /// <param name="out">Must be <c>null</c>.</param>
        /// <returns>A 0-d bool array when <paramref name="axis"/> is <c>null</c>; otherwise the bool matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is not <c>null</c>.</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray all(int? axis = null, NDArray @out = null)
        {
            NDArray self = A;   // the plain computation (file header)
            if (@out is not null)
                throw new TypeError("Cannot construct a dtype from an array");
            if (axis is null)
                return np.all(self, axis: null, @out: null, keepdims: false);
            return FromWrapped(np.all(self, axis: axis, @out: null, keepdims: true), this);
        }

        #endregion

        #region max / min

        /// <summary>NumPy's <c>matrix.max(axis, out)</c> for a given axis: the maxima as a matrix that keeps the
        ///     reduced axis with length 1.</summary>
        /// <param name="axis">The axis to reduce along.</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <returns>The maxima; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">The reduced axis is empty (<c>zero-size array to reduction operation
        ///     maximum which has no identity</c>), or <paramref name="out"/> is read-only or misshapen.</exception>
        public NDMatrix max(int axis, NDMatrix @out = null) => (NDMatrix)max((int?)axis, @out);

        /// <summary>NumPy's <c>matrix.max(axis=None, out=None)</c>; the largest element as a scalar without an
        ///     axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> for the maximum of every element.</param>
        /// <param name="out">A matrix of the kept shape, or <c>null</c>.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An empty reduction, or <paramref name="out"/> is read-only or misshapen.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray max(int? axis = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                if (axis is null)
                    return np.max(self, axis: null, @out: null, keepdims: false);
                return FromWrapped(np.max(self, axis: axis, @out: null, keepdims: true), this);
            }

            np.max(self, axis: axis, @out: outView, keepdims: true);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        /// <summary>NumPy's <c>matrix.min(axis, out)</c> for a given axis: the minima as a matrix that keeps the
        ///     reduced axis with length 1.</summary>
        /// <param name="axis">The axis to reduce along.</param>
        /// <param name="out">A matrix of the result's shape to store into, or <c>null</c>.</param>
        /// <returns>The minima; <paramref name="out"/> itself when it is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An empty reduction, or <paramref name="out"/> is read-only or misshapen.</exception>
        public NDMatrix min(int axis, NDMatrix @out = null) => (NDMatrix)min((int?)axis, @out);

        /// <summary>NumPy's <c>matrix.min(axis=None, out=None)</c>; the smallest element as a scalar without an
        ///     axis.</summary>
        /// <param name="axis">The axis, or <c>null</c> for the minimum of every element.</param>
        /// <param name="out">A matrix of the kept shape, or <c>null</c>.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An empty reduction, or <paramref name="out"/> is read-only or misshapen.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain (non-matrix) array: NumPy's reduction stores
        ///     into it, then <c>_collapse</c> does not exist on it (<c>'numpy.ndarray' object has no attribute
        ///     '_collapse'</c>).</exception>
        [NDScoped]
        [NumPyMethod]
        public NDArray min(int? axis = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (@out is null)
            {
                if (axis is null)
                    return np.min(self, axis: null, @out: null, keepdims: false);
                return FromWrapped(np.min(self, axis: axis, @out: null, keepdims: true), this);
            }

            np.min(self, axis: axis, @out: outView, keepdims: true);
            // matrix._collapse(axis) on the out NumPy's reduction returned: a plain out has no _collapse (after the store)
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_collapse'");
            return axis is null ? ScalarFrom(matrixOut) : matrixOut;
        }

        #endregion

        #region nanmax / nanmin (numpy/lib/_nanfunctions_impl.py's branch for subclasses)

        /// <summary>
        ///     np.nanmax of a matrix: NumPy's nanmax sends any operand that is not a base ndarray down its branch "slow, but
        ///     safe for subclasses of ndarray" (<see cref="NanExtreme"/>), which ends in this class's own <c>max</c>. A 0-d
        ///     matrix is a NumPy scalar (a C# number converted to the class), computed as a base ndarray.
        /// </summary>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">NumPy's <c>out=</c>, or <c>null</c>: a matrix gets the method's answer, a plain array the
        ///     values and then the method's AttributeError.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is matrix.max's TypeError.</param>
        /// <param name="initial">The caller's initial value; any value is matrix.max's TypeError.</param>
        /// <param name="where">The caller's mask; any mask is matrix.max's TypeError.</param>
        /// <returns>See <see cref="NanExtreme"/>.</returns>
        /// <exception cref="TypeError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="AttributeError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="ValueError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        internal override NDArray NanMax(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where)
            => Storage.Shape.NDim == 0
                ? base.NanMax(axis, @out, keepdims, initial, @where)
                : NanExtreme(axis, @out, keepdims, initial, @where, isMax: true);

        /// <summary>
        ///     np.nanmin of a matrix: NumPy's branch for subclasses (<see cref="NanExtreme"/>), ending in this class's own
        ///     <c>min</c>; a 0-d matrix is a NumPy scalar, computed as a base ndarray.
        /// </summary>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">NumPy's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is matrix.min's TypeError.</param>
        /// <param name="initial">The caller's initial value; any value is matrix.min's TypeError.</param>
        /// <param name="where">The caller's mask; any mask is matrix.min's TypeError.</param>
        /// <returns>See <see cref="NanExtreme"/>.</returns>
        /// <exception cref="TypeError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="AttributeError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="ValueError">See <see cref="NanExtreme"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        internal override NDArray NanMin(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where)
            => Storage.Shape.NDim == 0
                ? base.NanMin(axis, @out, keepdims, initial, @where)
                : NanExtreme(axis, @out, keepdims, initial, @where, isMax: false);

        /// <summary>
        ///     NumPy's np.nanmax / np.nanmin for a matrix, its branch for anything that is not a base ndarray
        ///     (numpy/lib/_nanfunctions_impl.py), statement for statement:
        ///     <code>
        ///     a, mask = _replace_nan(a, -np.inf)          # +np.inf for nanmin
        ///     res = np.amax(a, axis=axis, out=out, **kwargs)
        ///     if mask is None:
        ///         return res
        ///     kwargs.pop("initial", None)
        ///     mask = np.all(mask, axis=axis, **kwargs)
        ///     if np.any(mask):
        ///         res = _copyto(res, np.nan, mask)
        ///     return res
        ///     </code>
        ///     np.amax of the matrix copy IS this class's <see cref="max(int?, NDArray)"/> (the method NumPy's
        ///     <c>_wrapreduction</c> finds), so its keyword TypeErrors, its out= rules and its 'maximum' texts are the
        ///     answer's. Footgun: those bits are not fmax's - a <c>±0</c> tie follows maximum, an all-NaN slice is NumPy's
        ///     positive quiet NaN whatever NaN it held, and the copy and two mask reductions are paid on every call (NumPy's
        ///     slow route, which a plain array never takes).
        /// </summary>
        /// <param name="axis">The axis, or <c>null</c> for the scalar of every element.</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is the method's TypeError.</param>
        /// <param name="initial">The caller's initial value; any value is the method's TypeError.</param>
        /// <param name="where">The caller's mask; any mask is the method's TypeError.</param>
        /// <param name="isMax"><c>true</c> for np.nanmax (<c>-inf</c>, <c>max</c>), <c>false</c> for np.nanmin (<c>+inf</c>,
        ///     <c>min</c>).</param>
        /// <returns>Without <paramref name="out"/>: the scalar for a <c>null</c> axis, else a row <c>(1, N)</c> or a column
        ///     <c>(N, 1)</c> matrix. With a matrix <paramref name="out"/>: that out along an axis, a 0-d copy of its element
        ///     without one. An all-NaN slice is <c>np.nan</c> in the result's dtype: a NEW NaN scalar for a <c>null</c> axis
        ///     (a matrix out keeps the <c>-inf</c> / <c>+inf</c> the method stored, as NumPy's does), NaN written into the
        ///     row, the column or the matrix out along one.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is true, or <paramref name="initial"/> or
        ///     <paramref name="where"/> is given (<c>matrix.max() got an unexpected keyword argument 'initial'</c>), the first
        ///     in that order, before anything is computed.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array: the method stores the replaced copy's
        ///     extremes into it (an all-NaN slice as <c>-inf</c> / <c>+inf</c>), then fails on <c>out._collapse</c>.</exception>
        /// <exception cref="ValueError">An empty reduction, named after <c>maximum</c> / <c>minimum</c> (the method's
        ///     reduction), or a read-only or misshapen out.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        private NDArray NanExtreme(int? axis, NDArray @out, bool keepdims, object initial, NDArray @where, bool isMax)
        {
            // res = np.amax(a, ..., **kwargs) hands the caller's keywords to matrix.max, which takes none of them: the first
            // one passed, in amax's order (keepdims, initial, where), is its TypeError. Raising it before the copy is NumPy's
            // answer too - the copy _replace_nan makes first is not observable.
            string method = isMax ? "max" : "min";
            if (keepdims)
                throw NDSubclass.KeywordError(__name__, method, "keepdims");
            if (initial is not null)
                throw NDSubclass.KeywordError(__name__, method, "initial");
            if (@where is not null)
                throw NDSubclass.KeywordError(__name__, method, "where");

            // a, mask = _replace_nan(a, -inf / +inf): the array object's statement, on this matrix's plain view (the dtype
            // decision lives there, below the facades - ratchet R1). An inexact matrix gives the NaN mask and the replaced
            // copy; any other dtype no mask and the view itself.
            NDArray replaced = NDArray.ReplaceNaN(A, isMax ? double.NegativeInfinity : double.PositiveInfinity, out NDArray mask);

            // res = np.amax(a, axis=axis, out=out): the operand itself without a mask, else the copy as a matrix (NumPy's
            // np.array(a, subok=True, copy=True) is one), reduced by this class's own method - its out rules, its
            // _collapse, its texts.
            NDMatrix reduced = mask is null ? this : From(replaced, this);
            NDArray res = isMax ? reduced.max(axis, @out) : reduced.min(axis, @out);
            if (mask is null)
                return res;

            // mask = np.all(mask, axis=axis): matrix.all is ndarray.all(keepdims=True)._collapse(axis), the scalar without an
            // axis and the kept-axis array along one, which lines up with the row / column result or the matrix out (no
            // keyword is left to pass: any of them raised above). np.any(mask) asks whether any slice was all NaN.
            NDArray allNaN = axis is null ? np.all(mask) : np.all(mask, axis, keepdims: true);
            if (!(bool)np.any(allNaN))
                return res;

            // res = _copyto(res, np.nan, mask). np.nan is Python's positive quiet NaN (C#'s double.NaN carries the sign bit),
            // cast to the result's dtype: 0x7FF8000000000000, 0x7FC00000 for float32, 0x7E00 for float16, nan+0j for complex.
            double positiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            if (axis is null)
                // matrix._collapse(None) made res a NumPy scalar, not an ndarray: `a = a.dtype.type(val)` is a NEW scalar, and
                // a matrix out keeps the -inf / +inf the method stored.
                return NDArray.Scalar(positiveNaN, res.dtype).MarkReductionScalar();

            // An ndarray (the row / column, or the matrix out itself): np.copyto(res, np.nan, where=mask, casting='unsafe')
            // sets the all-NaN slices in place - through the plain view, as this class's own code never hands a matrix on.
            np.copyto(NDSubclass.PlainView(res), positiveNaN, "unsafe", allNaN);
            return res;
        }

        #endregion

        #region argmax / argmin / ptp

        /// <summary>
        ///     NumPy's <c>matrix.argmax(axis, out)</c> for a given axis: the index of the first maximum along the
        ///     axis, as a row <c>(1, N)</c> for axis 0 and a column <c>(M, 1)</c> for axis 1. The flat form
        ///     <c>m.argmax()</c> is the inherited member and returns a <see cref="long"/>.
        /// </summary>
        /// <param name="axis">0 or 1. NumPy's matrix rejects every other value, also -1 and -2.</param>
        /// <param name="out">Must be <c>null</c> in practice: NumPy's method fails for every out (see the
        ///     exceptions).</param>
        /// <returns>The int64 indices as a matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> is a valid negative axis (<c>unsupported
        ///     axis</c>), the reduced axis is empty, or <paramref name="out"/> does not have the 1-d result's shape
        ///     (<c>output array does not match result of np.argmax.</c>; a matrix never has it).</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array of the right shape: NumPy
        ///     writes the indices into it and then fails on <c>out._align</c>.</exception>
        [NDScoped]
        public NDMatrix argmax(int axis, NDArray @out = null) => ArgReduce(axis, @out, max: true);

        /// <summary>NumPy's <c>matrix.argmin(axis, out)</c> for a given axis; see
        ///     <see cref="argmax(int, NDArray)"/>.</summary>
        /// <param name="axis">0 or 1.</param>
        /// <param name="out">Must be <c>null</c> in practice.</param>
        /// <returns>The int64 indices as a matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An unsupported axis, an empty reduction, or a misshapen
        ///     <paramref name="out"/>.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array of the right shape.</exception>
        [NDScoped]
        public NDMatrix argmin(int axis, NDArray @out = null) => ArgReduce(axis, @out, max: false);

        /// <summary>
        ///     NumPy's <c>matrix.argmax(axis=None, out=None)</c> as the generated dispatch of <c>np.argmax</c> calls it
        ///     (<c>_wrapfunc</c>): along an axis the aligned index matrix of <see cref="argmax(int, NDArray)"/>; without one
        ///     <c>_align(None)</c>, the flat index as a 0-d array.
        /// </summary>
        /// <param name="axis">The axis (0 or 1), or <c>null</c> for the flat index. Deliberately without a default: an
        ///     <c>m.argmax()</c> call keeps binding the inherited flat member.</param>
        /// <param name="out">Must be <c>null</c> in practice: NumPy's method fails on every out (see the exceptions).</param>
        /// <returns>The 0-d flat index without an axis; otherwise the int64 index matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An unsupported (negative) axis, an empty reduction, or a misshapen out.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array NumPy's reduction accepted: <c>_align</c>
        ///     does not exist on it.</exception>
        [NumPyMethod]
        [NDScoped]
        public NDArray argmax(int? axis, NDArray @out = null) => axis is int along ? ArgReduce(along, @out, max: true) : FlatArgReduce(@out, max: true);

        /// <summary>NumPy's <c>matrix.argmin(axis=None, out=None)</c> as <c>np.argmin</c>'s generated dispatch calls it; see
        ///     <see cref="argmax(int?, NDArray)"/>.</summary>
        /// <param name="axis">The axis (0 or 1), or <c>null</c> for the flat index (no default, see argmax).</param>
        /// <param name="out">Must be <c>null</c> in practice.</param>
        /// <returns>The 0-d flat index without an axis; otherwise the int64 index matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An unsupported axis, an empty reduction, or a misshapen out.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array NumPy's reduction accepted.</exception>
        [NumPyMethod]
        [NDScoped]
        public NDArray argmin(int? axis, NDArray @out = null) => axis is int along ? ArgReduce(along, @out, max: false) : FlatArgReduce(@out, max: false);

        /// <summary>
        ///     <c>ndarray.argmax(self, None, out)._align(None)</c>: the flat index of the first extreme element as a 0-d array;
        ///     with an <paramref name="out"/> the index is stored into it and <c>_align</c> runs on it.
        /// </summary>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="max">Maximum instead of minimum.</param>
        /// <returns>The 0-d index, or the matrix out's scalar.</returns>
        /// <exception cref="ValueError">An empty matrix (NumPy's empty-sequence text), or a misshapen out.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array (it holds the index afterwards, as in
        ///     NumPy).</exception>
        [NDScopedCovered]
        private NDArray FlatArgReduce(NDArray @out, bool max)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            NDArray index = max ? np.argmax(self, (int?)null, outView) : np.argmin(self, (int?)null, outView);
            if (@out is null)
                return index;
            // _align(None) is self[0, 0]: a plain out has no _align.
            if (@out is not NDMatrix matrixOut)
                throw new AttributeError("'numpy.ndarray' object has no attribute '_align'");
            return ScalarFrom(matrixOut);
        }

        /// <summary>The shared body of <see cref="argmax(int, NDArray)"/> and <see cref="argmin(int, NDArray)"/>.</summary>
        /// <param name="axis">The axis as passed.</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="max">Maximum instead of minimum.</param>
        /// <returns>The aligned index matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An unsupported axis, an empty reduction, or a misshapen out.</exception>
        /// <exception cref="AttributeError">A plain out of the right shape.</exception>
        [NDScopedCovered]
        private NDMatrix ArgReduce(int axis, NDArray @out, bool max)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            // The plain reduction validates the axis first; only then does _align reject -1 / -2.
            DefaultEngine.NormalizeAxis(axis, ndim);
            NDArray indices = max ? np.argmax(self, axis) : np.argmin(self, axis);
            if (@out is not null)
            {
                if (!indices.Shape.Equals(@out.Shape))
                    throw new ValueError($"output array does not match result of np.{(max ? "argmax" : "argmin")}.");
                np.copyto(outView, indices, casting: "unsafe");
                throw new AttributeError("'numpy.ndarray' object has no attribute '_align'");
            }

            return Align(indices, axis, wrapped: false);
        }

        /// <summary>
        ///     NumPy's <c>matrix.ptp(axis, out)</c> for a given axis: maximum minus minimum along the axis, as a
        ///     row for axis 0 and a column for axis 1.
        /// </summary>
        /// <param name="axis">0 or 1. NumPy's matrix rejects every other value, also -1 and -2.</param>
        /// <param name="out">Must be <c>null</c> in practice: NumPy's method fails for every out.</param>
        /// <returns>The ranges as a matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> is a valid negative axis (<c>unsupported
        ///     axis</c>), the reduced axis is empty, or <paramref name="out"/> is given (NumPy's reduction or
        ///     broadcast text; a plain out of the right shape holds the MAXIMA afterwards, as in NumPy).</exception>
        /// <exception cref="TypeError">The matrix is bool (NumPy's boolean subtract is not defined).</exception>
        public NDMatrix ptp(int axis, NDArray @out = null) => (NDMatrix)ptp((int?)axis, @out);

        /// <summary>NumPy's <c>matrix.ptp(axis=None, out=None)</c>; the range of every element as a scalar without
        ///     an axis.</summary>
        /// <param name="axis">The axis (0 or 1), or <c>null</c>.</param>
        /// <param name="out">Must be <c>null</c> in practice.</param>
        /// <returns>A 0-d array when <paramref name="axis"/> is <c>null</c>; otherwise the matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">An unsupported axis, an empty reduction, or an <paramref name="out"/>.</exception>
        /// <exception cref="TypeError">The matrix is bool.</exception>
        [NDScoped]
        public NDArray ptp(int? axis = null, NDArray @out = null)
        {
            NDArray self = A, outView = NDSubclass.PlainView(@out);   // the plain computation (file header)
            if (axis is not null)
                DefaultEngine.NormalizeAxis(axis.Value, ndim);

            if (@out is not null)
            {
                // _ptp: subtract(maximum.reduce(a, axis, None, out), minimum.reduce(a, axis), out). The first
                // reduction has no keepdims, so it rejects an out of another rank; when it accepts the out it
                // returns that plain array, and subtracting the (wrapped, two-dimensional) minima from it cannot
                // be stored back into it.
                np.max(self, axis: axis, @out: outView, keepdims: false);
                string wide = axis is null ? "(1,1)" : string.Create(CultureInfo.InvariantCulture, $"(1,{@out.size})");
                throw new ValueError(
                    $"non-broadcastable output operand with shape {@out.Shape.ToPythonTuple()} doesn't match the broadcast shape {wide}");
            }

            NDArray range = np.ptp(self, axis);
            return axis is null ? range : Align(range, axis.Value, wrapped: true);
        }

        #endregion
    }
}
