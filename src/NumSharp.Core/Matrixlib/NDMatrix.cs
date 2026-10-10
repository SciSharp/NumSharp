using System;
using System.Collections;
using System.Numerics;
using NumSharp.Backends;

// =============================================================================
// NDMatrix.cs — numpy.matrix (numpy/matrixlib/defmatrix.py), the class and its matrix-only members
// =============================================================================
//
// NumPy's matrix is an ndarray SUBCLASS that changes three things: it is always two-dimensional
// (__array_finalize__ folds every result back onto two axes), `*` is the matrix product and `**` the matrix
// power, and it adds .T / .H / .I / .A / .A1. Python finds the subclass at RUN time. NumSharp finds it at COMPILE
// time: the members and operators declared on NDMatrix give the matrix semantics and return NDMatrix, and they are
// bound by the STATIC type of the expression.
//
//     NDMatrix m = np.matrix("1 2; 3 4");
//     var p = m * m;          // matrix product: NDMatrix.operator *
//     NDArray a = m;          // the same object, seen as an ndarray
//     var q = a * a;          // elementwise: NDArray.operator *, exactly as for any other NDArray
//
// The reason is safety. Every kernel and helper of this library is written against NDArray, where `x * y` is the
// elementwise product and `x[i]` drops an axis. A run-time dispatch (virtual members, a dynamic `*`) would change
// the meaning of that code for a caller who passed a matrix in. With static binding a matrix passed to an np.*
// function is an ordinary 2-D array there, which is also what NumPy's own C code sees.
//
// The consequence: np.* functions return NDArray, where NumPy's would hand a matrix back (np.sum(m, 0),
// np.transpose(m), np.linalg.inv(m)). Use the members (m.sum(0), m.T, m.I) or np.asmatrix(result), which is an
// O(1) view.
//
// THE THREE RESULT FORMS. Every member that returns a matrix builds it through From / FromWrapped / ViewOf:
//   ViewOf(a)        NumPy's a.view(matrix): a new array object over a's memory, never an owner.
//   From(r)          NumPy allocated the result as a matrix (copy, astype, dot, take ...): the matrix takes over
//                    the result's storage object, so it owns its data when the result did.
//   FromWrapped(r)   NumPy computed a plain ndarray and its __array_wrap__ returned a VIEW of it (every ufunc:
//                    the arithmetic operators, the reductions, cumsum, clip ...): the matrix never owns its data
//                    and its base is the hidden result.
// All three apply the finalize rule (Shape.AsMatrix) and none disposes the result it is given: their callers are
// [NDScoped], so the temporary NDArray wrapper is reclaimed by the scope while the matrix keeps its own counted
// reference on the buffer.
//
// THE 0-d MATRIX. NumPy returns a SCALAR where a matrix expression has no axes left (m[0, 0], m.sum()). C# needs
// one static return type, so:
//   * members whose NumPy result is always a scalar return a 0-d NDArray (m[0, 0], m.sum(), m.max());
//   * members that return a matrix or a scalar depending on their arguments (the general indexers) return an
//     NDMatrix that is 0-d in the scalar case;
//   * an implicitly converted C# number (m[mask] = 0, m + 2) is a 0-d NDMatrix too.
// A 0-d NDMatrix is a scalar, not a matrix: `*` scales by it, and an operation whose result is 0-d stays 0-d.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.matrix</c>: a two-dimensional array whose <c>*</c> is the matrix product, whose rows and
    ///     columns stay two-dimensional when indexed, and which adds <see cref="T"/>, <see cref="H"/>,
    ///     <see cref="I"/>, <see cref="A"/> and <see cref="A1"/>. Create one with <see cref="np.matrix(NDArray, DType, bool)"/>,
    ///     <see cref="np.asmatrix(NDArray, DType)"/> or <see cref="np.bmat(NDArray[][])"/>. Use it to port NumPy code
    ///     written against <c>np.matrix</c>; for new code NumPy recommends plain arrays and <c>np.matmul</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The matrix semantics follow the static type.</b> The members and operators of this class are chosen by
    ///     the compiler when the expression is typed <see cref="NDMatrix"/>. The same object seen through an
    ///     <see cref="NDArray"/> reference (a parameter of an <c>np.*</c> function, a cast, <see cref="A"/>) is a
    ///     plain 2-D array: <c>*</c> is elementwise there and an integer index drops an axis. Therefore the
    ///     <c>np.*</c> functions return <see cref="NDArray"/>; wrap a result with <see cref="np.asmatrix(NDArray, DType)"/>
    ///     (a view, no copy) or use the member (<c>m.sum(0)</c>, <c>m.T</c>, <c>m.I</c>).
    ///     </para>
    ///     <para>
    ///     <b>Always two-dimensional.</b> Every matrix result goes through NumPy's finalize rule
    ///     (<see cref="Shape.AsMatrix"/>): a 1-d result becomes a row <c>(1, N)</c>, a 0-d result <c>(1, 1)</c>, and a
    ///     result with more than two axes longer than 1 raises <c>shape too large to be a matrix.</c>. Two NumPy
    ///     quirks are kept: indexing with a new axis (<c>m[Slice.NewAxis]</c>) and an in-place
    ///     <c>resize</c> can leave a matrix object with another rank.
    ///     </para>
    ///     <para>
    ///     <b>Scalars.</b> Where NumPy returns a scalar, a member returns a 0-d <see cref="NDArray"/> when the result
    ///     is always a scalar (<c>m[0, 0]</c>, <c>m.sum()</c>) and a 0-d <see cref="NDMatrix"/> when it depends on the
    ///     arguments (<c>m["0, 0"]</c>). A C# number converts implicitly to a 0-d <see cref="NDMatrix"/>, so
    ///     <c>m * 2</c>, <c>m + 1.5</c> and <c>m[mask] = 0</c> compile. A 0-d matrix is a scalar: <c>*</c> scales by
    ///     it, and so does every 0-d <see cref="NDArray"/> that a computation returned (<c>m * m.sum()</c>,
    ///     <c>m * np.linalg.norm(a)</c>, <c>m * a[0, 0]</c>). The exception is NumPy's: an explicit 0-d ARRAY
    ///     (<c>np.array(2)</c>, <c>np.asarray(2)</c>; <see cref="NDArray.IsExplicitZeroDimArray"/>) is an ndarray to
    ///     <c>matrix.__mul__</c>, so <c>m * np.array(2)</c> is the product with a 1 x 1 matrix and raises
    ///     <c>shapes (2,3) and (1,1) not aligned</c> unless <c>m</c> has one column. A 0-d array built another way
    ///     (<c>np.zeros(())</c>, <c>x.reshape(())</c>) still scales, where NumPy raises.
    ///     </para>
    ///     <para>
    ///     <b>Lifetime.</b> A matrix is an <see cref="NDArray"/>: it holds one counted reference on its buffer and
    ///     is released with <see cref="NDArray.Dispose"/> or an <see cref="NDScope"/>. A matrix that views another
    ///     array (<see cref="np.asmatrix(NDArray, DType)"/>, <see cref="T"/>, a row) keeps that array's buffer alive.
    ///     </para>
    /// </remarks>
    /// <seealso href="https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"/>
    public partial class NDMatrix : NDArray, IEnumerable
    {
        /// <summary>
        ///     NumPy's <c>matrix.__array_priority__</c>: the value that makes a mixed <c>ndarray</c> / <c>matrix</c>
        ///     operation return a matrix. NumSharp resolves that at compile time through the operators of this
        ///     class, so the constant is informational; it exists so ported code that reads it compiles.
        /// </summary>
        public const double __array_priority__ = 10.0;

        /// <summary>
        ///     Wraps <paramref name="storage"/> as a matrix WITHOUT applying the finalize rule and without copying.
        ///     Callers must pass a storage object that no other array wraps: an in-place shape change of either
        ///     array would otherwise show through the other.
        /// </summary>
        /// <param name="storage">The storage to wrap. Its shape becomes the matrix's shape as it is (two axes,
        ///     or zero axes for the scalar stand-in; see the file header).</param>
        /// <exception cref="NullReferenceException"><paramref name="storage"/> is <c>null</c>.</exception>
        internal NDMatrix(UnmanagedStorage storage) : base(storage)
        {
        }

        /// <inheritdoc/>
        /// <remarks>NumPy prints a matrix as <c>matrix([[1, 2], [3, 4]])</c>: the subclass name, with the
        ///     continuation lines indented one column further than an array's.</remarks>
        internal override string ReprClassName => "matrix";

        #region Result construction (view / allocated / wrapped)

        /// <summary>
        ///     NumPy's <c>a.view(matrix)</c>: a NEW matrix object over the memory of <paramref name="a"/>, with the
        ///     finalize rule applied. Use it to see an array as a matrix without copying; the result never owns its
        ///     data and keeps the writeability of <paramref name="a"/>.
        /// </summary>
        /// <param name="a">The array to view. Not disposed, not modified.</param>
        /// <returns>A matrix sharing memory with <paramref name="a"/>. Never <paramref name="a"/> itself, also when
        ///     it is a matrix already (NumPy's <c>view</c> always makes a new object).</returns>
        /// <exception cref="ValueError">More than two axes of <paramref name="a"/> are longer than 1.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="a"/> has more than two axes and one has
        ///     extent 0.</exception>
        internal static NDMatrix ViewOf(NDArray a)
        {
            Shape shape = a.Storage.Shape;
            return new NDMatrix(shape.NDim == 2 ? a.Storage.Alias() : a.Storage.Alias(shape.AsMatrix()));
        }

        /// <summary>
        ///     The matrix for a result NumPy ALLOCATES as a matrix (<c>copy</c>, <c>astype</c>, <c>dot</c>,
        ///     <c>take</c>, a view-returning method such as <c>transpose</c>): the matrix takes over the storage
        ///     object of <paramref name="result"/>, so it owns its data exactly when the result did.
        /// </summary>
        /// <param name="result">What the underlying array operation returned. The caller's scope reclaims its
        ///     wrapper; this method does not dispose it.</param>
        /// <param name="operand0">An operand of that operation, or <c>null</c>. When the operation returned this
        ///     very object, NumPy's identity is kept (see the returns).</param>
        /// <param name="operand1">A second operand, or <c>null</c>.</param>
        /// <param name="scalarStaysScalar">Keep a 0-d result 0-d (the scalar stand-in) where the finalize rule
        ///     would make it <c>(1, 1)</c>.</param>
        /// <returns>
        ///     <paramref name="result"/> itself when it is a matrix already or one of the operands that is a matrix
        ///     (NumPy returns <c>self</c> from <c>astype(copy=False)</c>, <c>conj()</c> of a real matrix,
        ///     <c>to_device</c>); a view of it when it is an operand that is not a matrix; otherwise a new matrix
        ///     over the result's storage.
        /// </returns>
        /// <exception cref="ValueError">The result cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">The result has more than two axes and one has extent 0.</exception>
        internal static NDMatrix From(NDArray result, NDArray operand0 = null, NDArray operand1 = null, bool scalarStaysScalar = false)
            => Build(result, operand0, operand1, wrapped: false, scalarStaysScalar);

        /// <summary>
        ///     The matrix for a result NumPy computes as a plain array and then WRAPS (<c>__array_wrap__</c>, every
        ///     ufunc: the arithmetic and comparison operators, the reductions, <c>cumsum</c>, <c>clip</c>): a view of
        ///     the hidden result. Unlike <see cref="From"/> the matrix never owns its data, which is what
        ///     <c>flags.owndata</c> and <c>base</c> report in NumPy for these results.
        /// </summary>
        /// <param name="result">What the underlying array operation returned (not disposed here).</param>
        /// <param name="operand0">An operand of that operation, or <c>null</c> (identity rule of <see cref="From"/>).</param>
        /// <param name="operand1">A second operand, or <c>null</c>.</param>
        /// <param name="scalarStaysScalar">Keep a 0-d result 0-d (the scalar stand-in).</param>
        /// <returns>The matrix; see <see cref="From"/> for the identity cases.</returns>
        /// <exception cref="ValueError">The result cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">The result has more than two axes and one has extent 0.</exception>
        internal static NDMatrix FromWrapped(NDArray result, NDArray operand0 = null, NDArray operand1 = null, bool scalarStaysScalar = false)
            => Build(result, operand0, operand1, wrapped: true, scalarStaysScalar);

        /// <summary>
        ///     The matrix for an indexing result, where NumPy's <c>matrix.__getitem__</c> switches the finalize rule
        ///     OFF (<c>self._getitem = True</c>) and applies its own: a 0-d result is a scalar, a 1-d result becomes a
        ///     row or a column, and every other rank is returned as it is (so a new axis gives a 3-d matrix object).
        /// </summary>
        /// <param name="result">What the plain array indexing returned (not disposed here).</param>
        /// <param name="self">The indexed matrix (identity rule of <see cref="From"/>).</param>
        /// <param name="column">Whether a 1-d result is a column <c>(N, 1)</c>: NumPy's
        ///     <c>len(index) &gt; 1 and isscalar(index[1])</c>. Otherwise it is a row <c>(1, N)</c>.</param>
        /// <param name="scalarArrayCopy">The index holds a 0-d integer ARRAY next to integers and slices only
        ///     (NumPy's <c>HAS_SCALAR_ARRAY</c>): NumPy selects the view and returns a COPY of it, so the matrix
        ///     owns its data. Every other advanced index gives a matrix that views the gathered array.</param>
        /// <returns>The matrix; 0-d for a scalar result.</returns>
        [NDScopedCovered]
        internal static NDMatrix FromIndexing(NDArray result, NDMatrix self, bool column, bool scalarArrayCopy = false)
        {
            if (ReferenceEquals(result, self))
                return self;

            UnmanagedStorage storage = result.Storage;
            if (scalarArrayCopy)
            {
                // mapping.c: `if (index_type & HAS_SCALAR_ARRAY) result = PyArray_NewCopy(view, NPY_KEEPORDER)`.
                // The copy is made as the subclass, so the matrix is the owner. A result that already is a fresh
                // owner is that copy.
                if (!storage.OwnsData)
                    storage = result.copy('K').Storage;
            }
            else if (ReferenceEquals(storage, self.Storage) || storage.OwnsData)
            {
                // Two reasons for a new storage object over the same memory.
                //  * Never share a storage OBJECT with another live array: a later in-place shape change would
                //    show through both.
                //  * A result that owns its data is what an index array or a mask gathered. NumPy builds the
                //    subclass result of such an index as a second array object over the gathered one (mapping.c
                //    wrap_out_array: PyArray_NewFromDescr(Py_TYPE(self), ..., base = result)), so the matrix
                //    never owns its data and its base is the hidden array.
                storage = storage.Alias();
            }

            ApplyIndexingRule(storage, column);
            return new NDMatrix(storage);
        }

        /// <summary>
        ///     Applies <c>matrix.__getitem__</c>'s shape rule to a storage object in place: a 1-d shape becomes a
        ///     row or a column, every other rank is kept.
        /// </summary>
        /// <param name="storage">A storage object that only the result being built wraps.</param>
        /// <param name="column">Column <c>(N, 1)</c> instead of row <c>(1, N)</c> for a 1-d shape.</param>
        internal static void ApplyIndexingRule(UnmanagedStorage storage, bool column)
        {
            Shape shape = storage.Shape;
            if (shape.NDim != 1)
                return;
            Shape folded = column ? shape.AsMatrixColumn() : shape.AsMatrix();
            storage.SetShapeUnsafe(ref folded);
        }

        /// <summary>The shared body of <see cref="From"/> and <see cref="FromWrapped"/>.</summary>
        /// <param name="result">The plain result.</param>
        /// <param name="operand0">First operand or <c>null</c>.</param>
        /// <param name="operand1">Second operand or <c>null</c>.</param>
        /// <param name="wrapped">Build a view of the result (never an owner) instead of taking its storage over.</param>
        /// <param name="scalarStaysScalar">Keep a 0-d result 0-d.</param>
        /// <returns>The matrix.</returns>
        /// <exception cref="ValueError">The result cannot be a matrix.</exception>
        /// <exception cref="IncorrectShapeException">The result has more than two axes and one has extent 0.</exception>
        private static NDMatrix Build(NDArray result, NDArray operand0, NDArray operand1, bool wrapped, bool scalarStaysScalar)
        {
            // The operation handed an operand back (astype(copy: false), conj() of a real array, to_device):
            // NumPy returns that very object, so a matrix operand IS the result.
            if (ReferenceEquals(result, operand0) || ReferenceEquals(result, operand1))
                return result as NDMatrix ?? ViewOf(result);
            if (result is NDMatrix already)
            {
                // A matrix member returned a matrix (m.T inside a composition): an allocated result is that
                // matrix; a wrapped one must not own, so it is a view of it.
                return wrapped && already.Storage.OwnsData ? new NDMatrix(already.Storage.Alias()) : already;
            }

            UnmanagedStorage storage = result.Storage;
            Shape shape = storage.Shape;
            bool fold = shape.NDim != 2 && !(scalarStaysScalar && shape.NDim == 0);

            // A storage object another live array wraps is never taken over or reshaped in place: NDArray.flat of
            // a vector and a few engine shortcuts return a second wrapper over the operand's own storage.
            bool shared = (operand0 is not null && ReferenceEquals(storage, operand0.Storage))
                          || (operand1 is not null && ReferenceEquals(storage, operand1.Storage));

            // NumSharp hands a computed scalar back as a READ-ONLY 0-d array (a NumPy scalar is immutable). Folded
            // to (1, 1) it is an ordinary matrix element, which NumPy lets the caller write (m.trace()[0, 0] = 1).
            // Only a 0-d result that owns its element is such a scalar; a 0-d VIEW keeps what it views.
            if (fold && shape.NDim == 0 && !shape.IsWriteable && !shared && storage.OwnsData)
            {
                shape = shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
                storage.SetShapeUnsafe(ref shape);
            }
            if (wrapped || shared)
                return new NDMatrix(fold ? storage.Alias(shape.AsMatrix()) : storage.Alias());

            if (fold)
            {
                // NumPy assigns self.shape in place. The result's wrapper is about to be reclaimed by the caller's
                // scope, so changing its storage is not observable through it.
                Shape folded = shape.AsMatrix();
                storage.SetShapeUnsafe(ref folded);
            }

            return new NDMatrix(storage);
        }

        #endregion

        #region Scalars (the 0-d stand-in)

        /// <summary>A 0-d matrix holding <paramref name="value"/>: the scalar stand-in a C# number becomes.</summary>
        /// <typeparam name="T">The element type, one of the 15 NumSharp dtypes.</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>A new 0-d matrix that owns its one element.</returns>
        private static NDMatrix ScalarOf<T>(T value) where T : unmanaged => new NDMatrix(UnmanagedStorage.Scalar(value));

        /// <summary>A C# <see cref="bool"/> as a 0-d matrix scalar (NumPy's Python <c>bool</c> operand of a matrix
        ///     expression). It is NOT <c>np.matrix(value)</c>, which is <c>(1, 1)</c>.</summary>
        /// <param name="value">The value.</param>
        public static implicit operator NDMatrix(bool value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(sbyte value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(byte value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(short value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(ushort value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(int value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(uint value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(long value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(ulong value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(char value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(Half value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(float value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(double value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(decimal value) => ScalarOf(value);

        /// <inheritdoc cref="implicit operator NDMatrix(bool)"/>
        public static implicit operator NDMatrix(Complex value) => ScalarOf(value);

        #endregion

        #region A / A1 / T / H / I

        /// <summary>
        ///     NumPy's <c>matrix.A</c>: this matrix as a plain <see cref="NDArray"/> of the same shape. Use it to
        ///     leave the matrix semantics: <c>m.A * m.A</c> is elementwise and <c>m.A[0]</c> is 1-d.
        /// </summary>
        /// <value>A new array object over the same memory (a view; writes reach the matrix). It keeps the
        ///     matrix's buffer alive while it lives.</value>
        /// <remarks>A cast (<c>(NDArray)m</c>) gives the plain semantics too, without a new object.</remarks>
        public NDArray A => base.view();

        /// <summary>
        ///     NumPy's <c>matrix.A1</c>: the elements as a 1-d <see cref="NDArray"/> in C order (<c>m.A.ravel()</c>).
        /// </summary>
        /// <value>A view when the matrix is C-contiguous (writes reach the matrix), otherwise a copy, exactly as
        ///     <c>ravel</c> decides.</value>
        public NDArray A1 => base.ravel();

        /// <summary>NumPy's <c>matrix.T</c>: the transpose, as a matrix.</summary>
        /// <value>A view with the two axes exchanged (writes reach this matrix).</value>
        public new NDMatrix T
        {
            [NDScoped] get => From(base.transpose(), this);
        }

        /// <summary>
        ///     NumPy's <c>matrix.H</c>: the conjugate transpose. For a matrix that is not complex it is the plain
        ///     transpose.
        /// </summary>
        /// <value>A new matrix holding the conjugated elements for a complex matrix; for every other dtype the
        ///     same VIEW as <see cref="T"/>, so writes reach this matrix there and not for a complex one.</value>
        public NDMatrix H
        {
            [NDScoped]
            get => typecode == NPTypeCode.Complex
                ? FromWrapped(base.transpose().conj(), this)
                : From(base.transpose(), this);
        }

        /// <summary>
        ///     NumPy's <c>matrix.I</c>: the multiplicative inverse of a square matrix (<c>np.linalg.inv</c>), the
        ///     Moore-Penrose pseudo-inverse of any other (<c>np.linalg.pinv</c>).
        /// </summary>
        /// <value>A new matrix. An integer or bool matrix gives float64; float32 stays float32.</value>
        /// <exception cref="LinAlgError">A square matrix is singular.</exception>
        /// <exception cref="TypeError">The dtype is not supported by <c>np.linalg</c> (float16, decimal, char).</exception>
        /// <exception cref="NotSupportedException">The matrix is not square and no LAPACK backend is installed:
        ///     the pseudo-inverse needs a singular value decomposition, which NumSharp.Core does not carry
        ///     (reference <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        /// <exception cref="ValueError">The matrix object is not two-dimensional (possible only after the rank
        ///     quirks named on the class): NumPy's <c>M, N = self.shape</c> fails to unpack.</exception>
        public NDMatrix I
        {
            [NDScoped]
            get
            {
                long[] dims = Shape.dimensions;
                // NumPy: M, N = self.shape. A matrix of another rank fails that unpacking before any algebra.
                if (dims.Length < 2)
                    throw new ValueError($"not enough values to unpack (expected 2, got {dims.Length})");
                if (dims.Length > 2)
                    throw new ValueError("too many values to unpack (expected 2)");

                NDArray inverse = dims[0] == dims[1] ? np.linalg.inv(this) : np.linalg.pinv(this);
                return FromWrapped(inverse, this);
            }
        }

        /// <summary>NumPy's <c>matrix.getA()</c>: the same as <see cref="A"/>.</summary>
        /// <returns>A plain array view of this matrix.</returns>
        public NDArray getA() => A;

        /// <summary>NumPy's <c>matrix.getA1()</c>: the same as <see cref="A1"/>.</summary>
        /// <returns>The elements as a 1-d array.</returns>
        public NDArray getA1() => A1;

        /// <summary>NumPy's <c>matrix.getT()</c>: the same as <see cref="T"/>.</summary>
        /// <returns>The transposed matrix (a view).</returns>
        public NDMatrix getT() => T;

        /// <summary>NumPy's <c>matrix.getH()</c>: the same as <see cref="H"/>.</summary>
        /// <returns>The conjugate transpose.</returns>
        public NDMatrix getH() => H;

        /// <summary>NumPy's <c>matrix.getI()</c>: the same as <see cref="I"/>.</summary>
        /// <returns>The inverse or pseudo-inverse.</returns>
        /// <exception cref="LinAlgError">A square matrix is singular.</exception>
        /// <exception cref="TypeError">The dtype is not supported by <c>np.linalg</c>.</exception>
        /// <exception cref="NotSupportedException">A non-square matrix without a LAPACK backend.</exception>
        /// <exception cref="ValueError">The matrix object is not two-dimensional.</exception>
        public NDMatrix getI() => I;

        #endregion

        #region Object protocol (clone, iteration, equality)

        /// <summary>
        ///     A deep copy that is a matrix again. It HIDES <see cref="NDArray.Clone"/> and does not override it,
        ///     on purpose: library code that clones an array it holds as <see cref="NDArray"/> must get a plain
        ///     array back, or the <c>np.*</c> functions that happen to clone their operand would return matrices
        ///     and the others would not.
        /// </summary>
        /// <returns>A new matrix owning a copy of the data.</returns>
        public new NDMatrix Clone() => new NDMatrix(Storage.Clone()) { TensorEngine = TensorEngine };

        /// <summary>
        ///     Iterates the rows. Each item is a <c>(1, N)</c> <see cref="NDMatrix"/> VIEW, as in NumPy, where
        ///     iterating an <see cref="NDArray"/> of the same shape yields 1-d rows.
        /// </summary>
        /// <returns>An enumerator over the rows; nothing for a matrix with no rows.</returns>
        /// <exception cref="TypeError">The matrix is the 0-d scalar stand-in (<c>iteration over a 0-d array</c>).</exception>
        /// <remarks>Each row holds a counted reference on the buffer: dispose the rows in a hot loop, or iterate
        ///     inside an <see cref="NDScope"/>.</remarks>
        public new IEnumerator GetEnumerator()
        {
            // The rank quirks (a 3-d matrix after m[newaxis], a 1-d one after resize) iterate like NumPy's base
            // class does; only a real matrix has rows to keep two-dimensional.
            if (ndim != 2)
                return base.GetEnumerator();
            return Rows().GetEnumerator();

            IEnumerable Rows()
            {
                long count = Shape.dimensions[0];
                for (long i = 0; i < count; i++)
                    yield return this[i];
            }
        }

        /// <summary>The non-generic enumeration of the rows; makes <c>foreach</c> over an
        ///     <see cref="IEnumerable"/> reference use the matrix rows too.</summary>
        /// <returns>The enumerator of <see cref="GetEnumerator"/>.</returns>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Python's <c>iter(m)</c>: the same as <see cref="GetEnumerator"/>.</summary>
        /// <returns>An enumerator over the rows as <c>(1, N)</c> matrices.</returns>
        /// <exception cref="TypeError">The matrix is the 0-d scalar stand-in.</exception>
        public new IEnumerator __iter__() => GetEnumerator();

        /// <summary>Value equality, as for <see cref="NDArray"/>: true when every element compares equal after
        ///     broadcasting. Declared because this class declares <c>==</c>.</summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><c>true</c> when all elements are equal; <c>false</c> for <c>null</c>.</returns>
        public override bool Equals(object obj) => base.Equals(obj);

        /// <summary>Always throws: an array is mutable and therefore unhashable, as in NumPy.</summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override int GetHashCode() => base.GetHashCode();

        #endregion
    }
}
