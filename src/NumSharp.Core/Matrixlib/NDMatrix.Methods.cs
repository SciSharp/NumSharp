using System;

// =============================================================================
// NDMatrix.Methods.cs — the ndarray methods a matrix inherits, returned as matrices
// =============================================================================
//
// numpy.matrix overrides only squeeze / flatten / ravel (and the reductions); every other method is ndarray's.
// The result is still a matrix, because ndarray creates results with the SUBCLASS and calls __array_finalize__,
// which folds them onto two axes. Three results that look odd follow from that rule and are NumPy's:
//
//     m.reshape(6)        (1, 6)      a 1-d result is a row
//     m.diagonal()        (1, K)      a read-only view
//     m.take(1)           (1, 1)      a 0-d result is a 1x1 matrix; m.trace() too
//     m.dot(v)            (1, M)      for a 1-d v: the 1-d product is folded to a ROW
//
// C# binds these members by static type, so each one is re-declared here with `new` and returns NDMatrix. They
// take the same arguments as the NDArray members, minus out=: a call that passes an out binds the NDArray
// member and returns what that returns.
//
// OWNERSHIP follows NumPy: a method that allocates its result (copy, astype, take, repeat, flatten, argsort ...)
// returns a matrix that owns its data (From); a method implemented by a ufunc or a reduction (cumsum, clip,
// conj, round, trace) returns a view of the hidden result (FromWrapped).
//
// =============================================================================

namespace NumSharp
{
    public partial class NDMatrix
    {
        #region Shape (in place)

        /// <summary>
        ///     The dimensions. Assigning reshapes the matrix IN PLACE and folds the new shape back onto two axes,
        ///     as NumPy does: <c>m.shape = (6,)</c> gives <c>(1, 6)</c>.
        /// </summary>
        /// <value>A copy of the dimensions (<c>(rows, columns)</c>).</value>
        /// <exception cref="AttributeError">(set) The new dimensions need a copy (a non-contiguous matrix).</exception>
        /// <exception cref="IncorrectShapeException">(set) The new size differs from the matrix's size.</exception>
        /// <exception cref="ValueError">(set) More than two of the new dimensions are longer than 1
        ///     (<c>shape too large to be a matrix.</c>); the matrix keeps its old shape.</exception>
        public new long[] shape
        {
            get => base.shape;
            set
            {
                Shape before = Storage.Shape;
                base.shape = value;
                FoldInPlace(before);
            }
        }

        /// <summary>
        ///     The shape. Assigning reshapes IN PLACE to the given dimensions and folds them onto two axes; see
        ///     <see cref="shape"/>.
        /// </summary>
        /// <value>The shape descriptor (dimensions, strides, flags).</value>
        /// <exception cref="AttributeError">(set) The new dimensions need a copy.</exception>
        /// <exception cref="IncorrectShapeException">(set) The new size differs from the matrix's size.</exception>
        /// <exception cref="ValueError">(set) More than two of the new dimensions are longer than 1; the matrix
        ///     keeps its old shape.</exception>
        public new Shape Shape
        {
            get => base.Shape;
            set
            {
                Shape before = Storage.Shape;
                base.Shape = value;
                FoldInPlace(before);
            }
        }

        /// <summary>
        ///     Folds this matrix's shape onto two axes after an in-place shape assignment (NumPy assigns through a
        ///     temporary view of the subclass, whose finalize does the folding).
        /// </summary>
        /// <param name="before">The shape to restore when the new one cannot be a matrix.</param>
        /// <exception cref="ValueError">The assigned shape has more than two axes longer than 1.</exception>
        /// <exception cref="IncorrectShapeException">More than two axes and one of extent 0.</exception>
        private void FoldInPlace(Shape before)
        {
            Shape current = Storage.Shape;
            if (current.NDim == 2)
                return;
            try
            {
                Shape folded = current.AsMatrix();
                Storage.SetShapeUnsafe(ref folded);
            }
            catch
            {
                // NumPy raises before self changes: put the old shape back.
                Storage.SetShapeUnsafe(ref before);
                throw;
            }
        }

        #endregion

        #region matrix's own overrides: squeeze / flatten / ravel

        /// <summary>
        ///     NumPy's <c>matrix.squeeze(axis=None)</c>. A matrix cannot lose an axis, so a squeezed row stays a row
        ///     and a squeezed COLUMN becomes a row <c>(1, N)</c>.
        /// </summary>
        /// <param name="axis">The axis to remove, or <c>null</c> for every axis of length 1.</param>
        /// <returns>This matrix ITSELF when it has no axis of length 1 and <paramref name="axis"/> is
        ///     <c>null</c> (as NumPy); otherwise a view.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> names an axis longer than 1 (<c>cannot select an
        ///     axis to squeeze out which has size not equal to one</c>).</exception>
        [NDScoped]
        public new NDMatrix squeeze(int? axis = null)
        {
            long[] dims = Storage.Shape.dimensions;
            if (axis is null)
            {
                // ndarray.squeeze returns self when no axis has length 1.
                bool unit = false;
                foreach (long extent in dims)
                    unit |= extent == 1;
                if (!unit)
                    return this;
            }
            else
            {
                // PyArray_SqueezeSelected: the axis is normalized first (AxisError), then it must have length 1.
                int selected = Backends.DefaultEngine.NormalizeAxis(axis.Value, dims.Length);
                if (dims[selected] != 1)
                    throw new ValueError("cannot select an axis to squeeze out which has size not equal to one");
            }

            return From(base.squeeze(axis), this);
        }

        /// <summary>NumPy's <c>matrix.flatten(order='C')</c>: a COPY of the elements as one row
        ///     <c>(1, size)</c>.</summary>
        /// <param name="order"><c>'C'</c> row-major, <c>'F'</c> column-major, <c>'A'</c> column-major when the
        ///     matrix is F-contiguous, <c>'K'</c> memory order.</param>
        /// <returns>A new <c>(1, size)</c> matrix that owns its data.</returns>
        /// <exception cref="ArgumentException"><paramref name="order"/> is not one of the four codes.</exception>
        [NDScoped]
        public new NDMatrix flatten(char order = 'C') => From(base.flatten(order), this);

        /// <summary>NumPy's <c>matrix.ravel(order='C')</c>: the elements as one row <c>(1, size)</c>, a VIEW
        ///     when the memory layout allows it and a copy otherwise.</summary>
        /// <returns>A <c>(1, size)</c> matrix; writes reach this matrix when it is a view.</returns>
        [NDScoped]
        public new NDMatrix ravel() => From(base.ravel(), this);

        /// <inheritdoc cref="ravel()"/>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <exception cref="ArgumentException"><paramref name="order"/> is not one of the four codes.</exception>
        [NDScoped]
        public new NDMatrix ravel(char order) => From(base.ravel(order), this);

        #endregion

        #region Conversion and copies

        /// <summary>Copy of the matrix cast to <paramref name="dtype"/>, as a matrix.</summary>
        /// <param name="dtype">The target dtype.</param>
        /// <param name="copy"><c>false</c> returns this matrix ITSELF when it already has the dtype.</param>
        /// <returns>A new matrix that owns its data, or this matrix.</returns>
        [NDScoped]
        public new NDMatrix astype(DType dtype, bool copy = true) => From(base.astype(dtype, copy), this);

        /// <summary>Copy of the matrix cast to <paramref name="dtype"/> with a memory order and a casting
        ///     rule, as a matrix.</summary>
        /// <param name="dtype">The target dtype.</param>
        /// <param name="copy"><c>false</c> returns this matrix itself when nothing has to change.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="casting">NumPy's casting rule name (<c>"unsafe"</c>, <c>"same_kind"</c>, <c>"safe"</c>,
        ///     <c>"equiv"</c>, <c>"no"</c>).</param>
        /// <returns>A new matrix that owns its data, or this matrix.</returns>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/>.</exception>
        /// <exception cref="ArgumentException">An unknown order or casting name.</exception>
        [NDScoped]
        public new NDMatrix astype(DType dtype, bool copy, char order, string casting = "unsafe")
            => From(base.astype(dtype, copy, order, casting), this);

        /// <summary>A copy of the matrix, as a matrix (NumPy's <c>m.copy(order)</c>).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>A new matrix that owns its data.</returns>
        /// <exception cref="ArgumentException">An unknown order.</exception>
        [NDScoped]
        public new NDMatrix copy(char order = 'C') => From(base.copy(order), this);

        /// <summary>A new matrix object over the same memory (NumPy's <c>m.view()</c>), optionally reading the
        ///     bytes as another dtype.</summary>
        /// <param name="dtype">The dtype to read the bytes as, or <c>null</c> to keep it. A dtype of another
        ///     size changes the number of columns.</param>
        /// <returns>A view; writes reach this matrix.</returns>
        /// <exception cref="ArgumentException">The new item size does not divide the last axis.</exception>
        /// <exception cref="InvalidOperationException">A dtype of another size over a non-contiguous last axis.</exception>
        [NDScoped]
        public new NDMatrix view(DType dtype = null) => From(base.view(dtype), this);

        /// <summary>The matrix with its bytes swapped, as a matrix.</summary>
        /// <param name="inplace"><c>true</c> swaps this matrix's own bytes and returns this matrix.</param>
        /// <returns>A new matrix, or this matrix when <paramref name="inplace"/> is <c>true</c>.</returns>
        /// <exception cref="ValueError"><paramref name="inplace"/> on a read-only matrix.</exception>
        [NDScoped]
        public new NDMatrix byteswap(bool inplace = false) => From(base.byteswap(inplace), this);

        /// <summary>A field of the matrix's elements read as <paramref name="dtype"/> at a byte offset, as a
        ///     matrix (a view).</summary>
        /// <param name="dtype">The dtype of the field.</param>
        /// <param name="offset">The byte offset inside each element.</param>
        /// <returns>A view of the field.</returns>
        /// <exception cref="ValueError">The field does not fit inside an element.</exception>
        [NDScoped]
        public new NDMatrix getfield(DType dtype, int offset = 0) => From(base.getfield(dtype, offset), this);

        /// <summary>Array-API device transfer: only <c>"cpu"</c> exists, and the matrix itself is returned.</summary>
        /// <param name="device">Must be <c>"cpu"</c>.</param>
        /// <param name="stream">Must be <c>null</c>.</param>
        /// <returns>This matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="device"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Another device, or a stream.</exception>
        public new NDMatrix to_device(string device, object stream = null) => (NDMatrix)base.to_device(device, stream);

        #endregion

        #region Axes

        /// <summary>The matrix with its axes permuted (NumPy's <c>m.transpose(axes)</c>); without an argument the
        ///     transpose.</summary>
        /// <param name="premute">The permutation (<c>{1, 0}</c> or <c>{0, 1}</c>), or <c>null</c> to reverse the
        ///     axes.</param>
        /// <returns>A view.</returns>
        /// <exception cref="ArgumentException">The permutation does not match the matrix's axes.</exception>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        [NDScoped]
        public new NDMatrix transpose(int[] premute = null) => From(base.transpose(premute), this);

        /// <summary>The matrix with two axes exchanged (NumPy's <c>m.swapaxes(a, b)</c>).</summary>
        /// <param name="axis1">The first axis.</param>
        /// <param name="axis2">The second axis.</param>
        /// <returns>A view.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        [NDScoped]
        public new NDMatrix swapaxes(int axis1, int axis2) => From(base.swapaxes(axis1, axis2), this);

        /// <summary>NumPy's <c>m.mT</c>: the matrix transpose; for a matrix the same as <see cref="T"/>.</summary>
        /// <value>A view.</value>
        /// <exception cref="ArgumentException">The matrix object has fewer than two axes (the 0-d scalar
        ///     stand-in).</exception>
        public new NDMatrix mT
        {
            [NDScoped] get => From(base.mT, this);
        }

        /// <summary>
        ///     The elements under another shape, folded back onto two axes (NumPy's <c>m.reshape(shape)</c>):
        ///     <c>m.reshape(6)</c> is the row <c>(1, 6)</c>, <c>m.reshape(1, 2, 3)</c> is <c>(2, 3)</c>.
        /// </summary>
        /// <param name="newShape">The requested dimensions; one may be -1 (inferred).</param>
        /// <returns>A view when the layout allows it, otherwise a copy.</returns>
        /// <exception cref="IncorrectShapeException">The requested size differs from the matrix's size.</exception>
        /// <exception cref="ValueError">More than two requested dimensions are longer than 1.</exception>
        [NDScoped]
        public new NDMatrix reshape(Shape newShape) => From(base.reshape(newShape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="newShape">The requested dimensions.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c> or <c>'A'</c>: the index order the elements are read and
        ///     placed in.</param>
        [NDScoped]
        public new NDMatrix reshape(Shape newShape, char order) => From(base.reshape(newShape, order), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="shape">The requested dimensions; one may be -1.</param>
        [NDScoped]
        public new NDMatrix reshape(int[] shape) => From(base.reshape(shape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="shape">The requested dimensions; one may be -1.</param>
        [NDScoped]
        public new NDMatrix reshape(params long[] shape) => From(base.reshape(shape), this);

        #endregion

        #region Selection

        /// <summary>The diagonal as a ROW <c>(1, K)</c> (NumPy's <c>m.diagonal()</c>).</summary>
        /// <param name="offset">0 for the main diagonal, positive above it, negative below.</param>
        /// <param name="axis1">The first axis of the plane.</param>
        /// <param name="axis2">The second axis of the plane.</param>
        /// <returns>A READ-ONLY view of the diagonal.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ArgumentException">The two axes are the same.</exception>
        [NDScoped]
        public new NDMatrix diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) => From(base.diagonal(offset, axis1, axis2), this);

        /// <summary>The sum of the diagonal as a <c>(1, 1)</c> matrix (NumPy's <c>m.trace()</c>; <c>np.trace(m)</c>
        ///     is a scalar).</summary>
        /// <param name="offset">The diagonal's offset.</param>
        /// <param name="axis1">The first axis of the plane.</param>
        /// <param name="axis2">The second axis of the plane.</param>
        /// <param name="dtype">The dtype of the sum, or <c>null</c> for NumPy's default.</param>
        /// <returns>A new <c>(1, 1)</c> matrix.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ArgumentException">The two axes are the same.</exception>
        [NDScoped]
        public NDMatrix trace(int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null)
            => FromWrapped(base.trace(offset, axis1, axis2, dtype), this);

        /// <summary>
        ///     Elements selected by index along an axis, as a matrix (NumPy's <c>m.take(indices, axis, mode)</c>).
        ///     Without an axis the matrix is read flat and the result is a row.
        /// </summary>
        /// <param name="indices">The indices; a C# integer selects one element (a <c>(1, 1)</c> result).</param>
        /// <param name="axis">The axis, or <c>null</c> to index the flattened matrix.</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> for an index out of bounds.</param>
        /// <returns>A new matrix that owns its data.</returns>
        /// <exception cref="IndexError">An index is out of bounds under <c>"raise"</c>. Without an axis the
        ///     text names axis 1 (<c>index 12 is out of bounds for axis 1 with size 12</c>): NumPy ravels the
        ///     matrix to its <c>(1, size)</c> row first and takes along that row.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">An index array of a non-integer dtype, or an unknown mode.</exception>
        [NDScoped]
        public NDMatrix take(NDArray indices, int? axis = null, string mode = "raise")
        {
            if (axis is not null || ndim != 2)
                return From(base.take(indices, axis, null, mode), this);

            // PyArray_CheckAxis for axis=None: temp = ravel(self), which is the (1, size) ROW for a matrix, and
            // the axis becomes temp.ndim - 1 = 1. The values are those of a flat take; the axis shows in the
            // bounds error.
            NDArray row = base.ravel().reshape(1, size);
            return From(row.take(indices, 1, null, mode), this);
        }

        /// <summary>The rows or columns where <paramref name="condition"/> is true, as a matrix (NumPy's
        ///     <c>m.compress(condition, axis)</c>); without an axis the flattened elements as a row.</summary>
        /// <param name="condition">A 1-d bool array.</param>
        /// <param name="axis">The axis to select along, or <c>null</c>.</param>
        /// <returns>A new matrix that owns its data.</returns>
        /// <exception cref="ValueError"><paramref name="condition"/> is not 1-d.</exception>
        /// <exception cref="IndexError"><paramref name="condition"/> is longer than the axis.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public NDMatrix compress(NDArray condition, int? axis = null) => From(base.compress(condition, axis, null), this);

        /// <summary>Builds a matrix by picking, per element, from <paramref name="choices"/> the array this
        ///     matrix's integer value names (NumPy's <c>m.choose(choices, mode)</c>).</summary>
        /// <param name="choices">The arrays to choose from; they broadcast against this matrix.</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> for a value out of range.</param>
        /// <returns>A new matrix.</returns>
        /// <exception cref="ValueError">A value is out of range under <c>"raise"</c>, or the shapes do not
        ///     broadcast.</exception>
        /// <exception cref="ArgumentException">An unknown mode.</exception>
        [NDScoped]
        public NDMatrix choose(NDArray[] choices, string mode = "raise") => From(base.choose(choices, null, mode), this);

        /// <summary>Each element repeated, as a matrix (NumPy's <c>m.repeat(repeats, axis)</c>); without an axis
        ///     the flattened elements as a row.</summary>
        /// <param name="repeats">The repeat count per element (one value, or one per element of the axis).</param>
        /// <param name="axis">The axis to repeat along, or <c>null</c>.</param>
        /// <returns>A new matrix that owns its data.</returns>
        /// <exception cref="ValueError">A negative count, or counts that do not match the axis.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public new NDMatrix repeat(NDArray repeats, int? axis = null) => From(base.repeat(repeats, axis), this);

        /// <inheritdoc cref="repeat(NDArray, int?)"/>
        [NDScoped]
        public new NDMatrix repeat(int repeats, int? axis = null) => From(base.repeat(repeats, axis), this);

        /// <inheritdoc cref="repeat(NDArray, int?)"/>
        [NDScoped]
        public new NDMatrix repeat(long repeats, int? axis = null) => From(base.repeat(repeats, axis), this);

        #endregion

        #region Sorting

        /// <summary>The indices that sort the matrix along an axis, as a matrix (NumPy's <c>m.argsort(axis)</c>);
        ///     with <c>null</c> the indices of the flattened matrix as a row.</summary>
        /// <param name="axis">The axis (default: the last), or <c>null</c>.</param>
        /// <returns>A new int64 matrix.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public new NDMatrix argsort(int? axis = -1) => From(base.argsort(axis), this);

        /// <summary>The indices that partition the matrix around the <paramref name="kth"/> element along an
        ///     axis, as a matrix (NumPy's <c>m.argpartition(kth, axis)</c>).</summary>
        /// <param name="kth">The element position that ends up in its sorted place.</param>
        /// <param name="axis">The axis (default: the last), or <c>null</c> for the flattened matrix.</param>
        /// <param name="kind">Only <c>"introselect"</c>.</param>
        /// <param name="order">Must be <c>null</c> (no structured dtypes).</param>
        /// <returns>A new int64 matrix.</returns>
        /// <exception cref="ValueError"><paramref name="kth"/> is out of bounds, an unknown
        ///     <paramref name="kind"/>, or an <paramref name="order"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public new NDMatrix argpartition(int kth, int? axis = -1, string kind = "introselect", string order = null)
            => From(base.argpartition(kth, axis, kind, order), this);

        #endregion

        #region Elementwise methods and scans

        /// <summary>The running sum along an axis, as a matrix (NumPy's <c>m.cumsum(axis, dtype)</c>); without an
        ///     axis the running sum of the flattened matrix as a row.</summary>
        /// <param name="axis">The axis, or <c>null</c>.</param>
        /// <param name="dtype">The dtype of the sums, or <c>null</c> for NumPy's default.</param>
        /// <returns>A matrix (a view of the computed array).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public NDMatrix cumsum(int? axis = null, DType dtype = null) => FromWrapped(base.cumsum(axis, dtype), this);

        /// <summary>The running product along an axis, as a matrix (NumPy's <c>m.cumprod(axis, dtype)</c>).</summary>
        /// <param name="axis">The axis, or <c>null</c> for the flattened matrix (a row).</param>
        /// <param name="dtype">The dtype of the products, or <c>null</c> for NumPy's default.</param>
        /// <returns>A matrix (a view of the computed array).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        public NDMatrix cumprod(int? axis = null, DType dtype = null) => FromWrapped(base.cumprod(axis, dtype), this);

        /// <summary>The complex conjugate, as a matrix. A matrix that is not complex is returned ITSELF (as
        ///     NumPy does), not a copy.</summary>
        /// <returns>The conjugated matrix, or this matrix.</returns>
        [NDScoped]
        public NDMatrix conj() => FromWrapped(base.conj(), this);

        /// <inheritdoc cref="conj()"/>
        [NDScoped]
        public NDMatrix conjugate() => FromWrapped(base.conjugate(), this);

        /// <summary>The elements limited to an interval, as a matrix (NumPy's <c>m.clip(min, max)</c>).</summary>
        /// <param name="min">The lower bound, or <c>null</c> for none. It broadcasts against the matrix.</param>
        /// <param name="max">The upper bound, or <c>null</c> for none. With no bound at all the result is a copy.</param>
        /// <returns>A matrix (a view of the computed array).</returns>
        /// <exception cref="IncorrectShapeException">A bound does not broadcast against the matrix.</exception>
        [NDScoped]
        public NDMatrix clip(NDArray min = null, NDArray max = null) => FromWrapped(base.clip(min, max), this);

        /// <summary>The elements rounded to the nearest integer (half to even), as a matrix (NumPy's
        ///     <c>m.round()</c>).</summary>
        /// <returns>A matrix: a copy that owns its data for an integer or complex matrix, a view of the computed
        ///     array for a float one. A bool matrix gives float16, as NumPy's <c>rint</c> does.</returns>
        [NDScoped]
        public NDMatrix round() => (NDMatrix)RoundLikeNumPy(0);

        /// <summary>
        ///     The elements rounded to <paramref name="decimals"/> places (NumPy's <c>m.round(decimals, out)</c>).
        ///     The CLASS of the result depends on the route NumPy's <c>PyArray_Round</c> takes, so the static type
        ///     is <see cref="NDArray"/>.
        /// </summary>
        /// <param name="decimals">The number of decimal places; negative rounds left of the point.</param>
        /// <param name="out">An array to store into, or <c>null</c>.</param>
        /// <returns>
        ///     <paramref name="out"/> when given. Otherwise:
        ///     a MATRIX that owns its data for a complex matrix (NumPy copies the matrix and rounds the two parts
        ///     in place) and for an integer matrix with <paramref name="decimals"/> &gt;= 0 (a copy);
        ///     a MATRIX viewing the computed array for <paramref name="decimals"/> 0 of a float or bool matrix
        ///     (the <c>rint</c> ufunc);
        ///     a plain ARRAY for every other case (NumPy computes those into an array it allocates itself).
        /// </returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or misshapen.</exception>
        /// <exception cref="InvalidCastException">The result cannot be cast to <paramref name="out"/>'s dtype.</exception>
        [NDScoped]
        public new NDArray round(int decimals, NDArray @out = null)
            => @out is not null ? base.round(decimals, @out) : RoundLikeNumPy(decimals);

        /// <summary>The class and ownership rule of <see cref="round(int, NDArray)"/> without an out.</summary>
        /// <param name="decimals">The number of decimal places.</param>
        /// <returns>A matrix or a plain array, as documented on <see cref="round(int, NDArray)"/>.</returns>
        [NDScopedCovered]
        private NDArray RoundLikeNumPy(int decimals)
        {
            NDArray rounded = base.round(decimals, null);
            NPTypeCode code = typecode;
            if (code == NPTypeCode.Complex || (decimals >= 0 && IsIntegerLike(code)))
            {
                // PyArray_NewCopy(a, NPY_KEEPORDER) as the subclass: always a new array that owns its data.
                bool fresh = !ReferenceEquals(rounded, this) && rounded.Storage.OwnsData;
                return From(fresh ? rounded : rounded.copy('K'), this);
            }

            return decimals == 0 ? FromWrapped(rounded, this) : rounded;
        }

        /// <summary>
        ///     NumPy's <c>m.dot(b)</c>: the dot product through the ARRAY method, folded onto two axes. With a 1-d
        ///     <paramref name="b"/> of length N the product of an <c>(M, N)</c> matrix is the 1-d vector of length M,
        ///     which comes back as a ROW <c>(1, M)</c> — not the column a matrix product would give.
        /// </summary>
        /// <param name="b">The right operand, as it is (it is NOT seen as a matrix first, unlike <c>m * b</c>).</param>
        /// <returns>A new matrix.</returns>
        /// <exception cref="ArgumentException">The shapes are not aligned.</exception>
        /// <exception cref="ValueError">The product has more than two axes longer than 1.</exception>
        [NDScoped]
        public new NDMatrix dot(NDArray b) => From(base.dot(b), this, b);

        /// <summary>
        ///     The real part, as a matrix. For a complex matrix a VIEW of the real components (writes reach the
        ///     matrix); for every other dtype the matrix ITSELF. Assigning stores into the real parts.
        /// </summary>
        /// <value>The real part.</value>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast.</exception>
        public new NDMatrix real
        {
            [NDScoped] get => From(base.real, this);
            set => base.real = value;
        }

        /// <summary>
        ///     The imaginary part, as a matrix. For a complex matrix a VIEW of the imaginary components; for every
        ///     other dtype a READ-ONLY matrix of zeros. Assigning stores into the imaginary parts of a complex
        ///     matrix.
        /// </summary>
        /// <value>The imaginary part.</value>
        /// <exception cref="TypeError">(set) The matrix is not complex (<c>array does not have imaginary part to
        ///     set</c>).</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast.</exception>
        public new NDMatrix imag
        {
            [NDScoped] get => From(base.imag, this);
            set => base.imag = value;
        }

        #endregion
    }
}
