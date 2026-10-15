using System;
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
// float64 out a float32 matrix is summed in float64. The result reaches out with an UNSAFE cast.
//
// SCALARS. axis=None is `...(keepdims=True)[0, 0]`: the reduction always runs in its ARRAY form and the scalar is
// read from the (1, 1) result. That matters where NumPy's array and scalar branches differ: a float16 mean rounds
// twice (float32 sum to float16), and std with an integer dtype= raises the sqrt cast error where np.std of the same
// data as a plain array returns a truncated scalar.
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
        ///     The loop of a reduction that was given <c>out=</c> and no <c>dtype=</c>: NumPy promotes the dtype
        ///     of <paramref name="out"/> with the operand's (file header), without the small-integer widening.
        /// </summary>
        /// <param name="out">The destination.</param>
        /// <returns><c>np.promote_types(out.dtype, this.dtype)</c>.</returns>
        private DType LoopWith(NDArray @out) => np.promote_types(@out.dtype, dtype);

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

        /// <summary>The number of elements a reduction over <paramref name="axis"/> combines into one result
        ///     element (NumPy's <c>_count_reduce_items</c>).</summary>
        /// <param name="axis">The axis, or <c>null</c> for all elements.</param>
        /// <returns>The count.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        private long ReduceCount(int? axis)
            => axis is null ? size : Shape.dimensions[DefaultEngine.NormalizeAxis(axis.Value, ndim)];

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
        [NDScoped]
        public NDArray sum(int? axis = null, DType dtype = null, NDMatrix @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                if (axis is null)
                    return TensorEngine.ReduceAdd(this, null, keepdims: false, dtype);
                return FromWrapped(TensorEngine.ReduceAdd(this, axis, keepdims: true, dtype), this);
            }

            // NumPy's reduce prologue runs before an element is read; the loop and the unsafe store are the
            // file header's out= rule.
            ReductionArguments.Validate(this, axis, keepdims: true, @out, null, "add");
            NDArray total = TensorEngine.ReduceAdd(this, axis, keepdims: true, dtype ?? LoopWith(@out));
            np.copyto(@out, total, casting: "unsafe");
            return axis is null ? ScalarFrom(@out) : @out;
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
        [NDScoped]
        public NDArray prod(int? axis = null, DType dtype = null, NDMatrix @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                if (axis is null)
                    return TensorEngine.ReduceProduct(this, null, keepdims: false, dtype);
                return FromWrapped(TensorEngine.ReduceProduct(this, axis, keepdims: true, dtype), this);
            }

            // NumPy's reduce prologue runs before an element is read; the loop is the file header's out= rule.
            ReductionArguments.Validate(this, axis, keepdims: true, @out, null, "multiply");
            NDArray product = TensorEngine.ReduceProduct(this, axis, keepdims: true, dtype ?? LoopWith(@out));
            np.copyto(@out, product, casting: "unsafe");
            return axis is null ? ScalarFrom(@out) : @out;
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
        [NDScoped]
        public NDArray mean(int? axis = null, DType dtype = null, NDMatrix @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                // Always the array form (keepdims), see the file header: a float16 mean rounds twice there.
                NDArray kept = TensorEngine.ReduceMean(this, axis, keepdims: true, dtype);
                if (axis is null)
                    return Collapse(kept);
                // _mean ends a float16 mean with `arr.dtype.type(ret)`, a conversion that allocates the matrix;
                // every other mean is the true_divide ufunc's result, which the matrix wraps.
                return dtype is null && typecode == NPTypeCode.Half ? From(kept, this) : FromWrapped(kept, this);
            }

            // _mean: ret = umr_sum(arr, axis, dtype, out, keepdims); ret = true_divide(ret, rcount, out=ret,
            // casting='unsafe'). The sum's loop is dtype=, else float64 for integers and bool, float32 for
            // float16, else the reduce rule for a missing dtype with an out (file header).
            DType loop = dtype;
            if (loop is null)
            {
                if (typecode == NPTypeCode.Boolean || IsIntegerLike(typecode))
                    loop = np.float64;
                else if (typecode == NPTypeCode.Half)
                    loop = np.float32;
            }

            ReductionArguments.Validate(this, axis, keepdims: true, @out, null, "add");
            long count = ReduceCount(axis);
            NDArray total = TensorEngine.ReduceAdd(this, axis, keepdims: true, loop ?? LoopWith(@out));
            np.copyto(@out, total, casting: "unsafe");
            // The division runs ON out: the sum as out's dtype holds it, divided by the strong int64 count.
            DefaultEngine.DivideByCountLikeNumPy(@out, count);
            return axis is null ? ScalarFrom(@out) : @out;
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
        [NDScoped]
        public NDArray var(int? axis = null, DType dtype = null, NDMatrix @out = null, int ddof = 0)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                NDArray kept = TensorEngine.ReduceVar(this, axis, keepdims: true, ddof, dtype);
                return axis is null ? Collapse(kept) : FromWrapped(kept, this);
            }

            StoreVariance(axis, dtype, @out, ddof);
            return axis is null ? ScalarFrom(@out) : @out;
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
        /// <exception cref="ArgumentException"><paramref name="out"/> or <paramref name="dtype"/> is an integer or
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
        /// <exception cref="ArgumentException">
        ///     An integer or bool <paramref name="out"/> or <paramref name="dtype"/>: NumPy takes the square root
        ///     in place under the <c>same_kind</c> rule, which refuses the float root for an integer array
        ///     (<c>Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>). With <paramref name="out"/> this is raised AFTER the variance was stored, so
        ///     <paramref name="out"/> holds the variance then. It is raised for <c>axis: null</c> too: a matrix
        ///     reduction is always the array form (file header), where <c>np.std</c> of a plain array returns a
        ///     truncated scalar.
        /// </exception>
        [NDScoped]
        public NDArray std(int? axis = null, DType dtype = null, NDMatrix @out = null, int ddof = 0)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                NDArray kept = TensorEngine.ReduceStd(this, axis, keepdims: true, ddof, dtype);
                return axis is null ? Collapse(kept) : FromWrapped(kept, this);
            }

            // _std: ret = _var(..., out=out); ret = um.sqrt(ret, out=ret). The ufunc itself refuses an integer or
            // bool out (its loop is a float one that same_kind cannot cast back), after the variance is in out.
            StoreVariance(axis, dtype, @out, ddof);
            np.sqrt(@out, @out: @out);
            return axis is null ? ScalarFrom(@out) : @out;
        }

        /// <summary>The variance of this matrix stored into <paramref name="out"/> with NumPy's reduce checks
        ///     and an unsafe cast (the shared body of <see cref="var(int?, DType, NDMatrix, int)"/> and
        ///     <see cref="std(int?, DType, NDMatrix, int)"/> with an out).</summary>
        /// <param name="axis">The axis, or <c>null</c>.</param>
        /// <param name="dtype">The requested dtype, or <c>null</c>.</param>
        /// <param name="out">The destination matrix.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or misshapen.</exception>
        /// <exception cref="NotSupportedException">The matrix is bound to an engine that is not NumSharp's
        ///     <see cref="DefaultEngine"/> (the out= form needs its sum-only variance).</exception>
        [NDScopedCovered]
        private void StoreVariance(int? axis, DType dtype, NDMatrix @out, int ddof)
        {
            ReductionArguments.Validate(this, axis, keepdims: true, @out, null, "add");
            if (TensorEngine is not DefaultEngine engine)
                throw new NotSupportedException("matrix.var / matrix.std with out= need NumSharp's DefaultEngine.");

            // _var: ret = umr_sum(x, axis, dtype, out, keepdims) — the sum of the squared deviations, whose loop
            // out joins — then ret = true_divide(ret, max(rcount - ddof, 0), out=ret, casting='unsafe').
            NDArray squares = engine.VarStdNumPy(this, axis, keepdims: true, ddof, dtype, takeSqrt: false,
                outDtype: @out.dtype, sumOnly: true);
            np.copyto(@out, squares, casting: "unsafe");
            DefaultEngine.DivideByCountLikeNumPy(@out, Math.Max(ReduceCount(axis) - ddof, 0));
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
        public NDArray any(int? axis = null, NDArray @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is not null)
                throw new TypeError("Cannot construct a dtype from an array");
            if (axis is null)
                return np.any((NDArray)this, axis: null, @out: null, keepdims: false);
            return FromWrapped(np.any((NDArray)this, axis: axis, @out: null, keepdims: true), this);
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
        public NDArray all(int? axis = null, NDArray @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is not null)
                throw new TypeError("Cannot construct a dtype from an array");
            if (axis is null)
                return np.all((NDArray)this, axis: null, @out: null, keepdims: false);
            return FromWrapped(np.all((NDArray)this, axis: axis, @out: null, keepdims: true), this);
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
        [NDScoped]
        public NDArray max(int? axis = null, NDMatrix @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                if (axis is null)
                    return np.max((NDArray)this, axis: null, @out: null, keepdims: false);
                return FromWrapped(np.max((NDArray)this, axis: axis, @out: null, keepdims: true), this);
            }

            np.max((NDArray)this, axis: axis, @out: @out, keepdims: true);
            return axis is null ? ScalarFrom(@out) : @out;
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
        [NDScoped]
        public NDArray min(int? axis = null, NDMatrix @out = null)
        {
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (@out is null)
            {
                if (axis is null)
                    return np.min((NDArray)this, axis: null, @out: null, keepdims: false);
                return FromWrapped(np.min((NDArray)this, axis: axis, @out: null, keepdims: true), this);
            }

            np.min((NDArray)this, axis: axis, @out: @out, keepdims: true);
            return axis is null ? ScalarFrom(@out) : @out;
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
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            // The plain reduction validates the axis first; only then does _align reject -1 / -2.
            DefaultEngine.NormalizeAxis(axis, ndim);
            NDArray indices = max ? np.argmax((NDArray)this, axis) : np.argmin((NDArray)this, axis);
            if (@out is not null)
            {
                if (!indices.Shape.Equals(@out.Shape))
                    throw new ValueError($"output array does not match result of np.{(max ? "argmax" : "argmin")}.");
                np.copyto(@out, indices, casting: "unsafe");
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
            using var plain = NDMatrixBoundary.Inside();   // the np.* calls below compute plain; this member builds the matrix
            if (axis is not null)
                DefaultEngine.NormalizeAxis(axis.Value, ndim);

            if (@out is not null)
            {
                // _ptp: subtract(maximum.reduce(a, axis, None, out), minimum.reduce(a, axis), out). The first
                // reduction has no keepdims, so it rejects an out of another rank; when it accepts the out it
                // returns that plain array, and subtracting the (wrapped, two-dimensional) minima from it cannot
                // be stored back into it.
                np.max((NDArray)this, axis: axis, @out: @out, keepdims: false);
                string wide = axis is null ? "(1,1)" : $"(1,{@out.size})";
                throw new ValueError(
                    $"non-broadcastable output operand with shape {@out.Shape.ToPythonTuple()} doesn't match the broadcast shape {wide}");
            }

            NDArray range = np.ptp((NDArray)this, axis);
            return axis is null ? range : Align(range, axis.Value, wrapped: true);
        }

        #endregion
    }
}
