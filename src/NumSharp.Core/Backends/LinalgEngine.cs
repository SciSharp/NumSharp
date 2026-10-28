// numpy.linalg's functions as a sub-engine, reached through TensorEngine.Linalg (docs/plans/np-layer-dtype-logic.md,
// section 1 "Which O" and phase PE, E2): NumPy writes them in Python (numpy/linalg/_linalg.py) around the
// _umath_linalg gufuncs, and DefaultLinalgEngine runs those statements over protected virtual kernels that ask
// TensorEngine.Blas first. The views numpy.linalg.diagonal and numpy.linalg.matrix_transpose are not members: they are
// functions of the array object, facades over np.diagonal and np.matrix_transpose.

using System;

namespace NumSharp
{
    /// <summary>
    ///     The contract of <c>numpy.linalg</c>: one member per function that computes (29 of NumPy's 31; the views
    ///     <c>diagonal</c> and <c>matrix_transpose</c> are the array object's), plus a second C# form where NumPy's
    ///     argument is an int or a tuple (<see cref="Tensordot(NDArray, NDArray, int[], int[])"/>,
    ///     <see cref="VectorNorm(NDArray, int, bool, object)"/>). An engine reaches it through
    ///     <see cref="TensorEngine.Linalg"/>, and every public <c>np.linalg</c> function applies NumPy's class boundary
    ///     (<c>numpy.matrix</c>, <c>numpy.char.chararray</c>) and then calls one member.
    /// </summary>
    /// <remarks>
    ///     <para><b>The default</b> is <see cref="Backends.DefaultLinalgEngine"/>: NumPy's statements in NumPy's order (the
    ///     argument checks, <c>_commonType</c>'s dtype rule, the gufuncs' text refusals and each function's tail) over ten
    ///     kernels that stand for the <c>_umath_linalg</c> gufuncs and ask <see cref="TensorEngine.Blas"/> first, so the
    ///     bundled OpenBLAS of <c>NumSharp.Interop.OpenBLAS</c> needs no subclass.</para>
    ///     <para><b>Providing one.</b> Derive from <see cref="Backends.DefaultLinalgEngine"/> and override its kernels to
    ///     compute the factorisations another way while NumPy's statements stay; derive from this class to replace whole
    ///     functions. Assign the instance to <see cref="TensorEngine.Linalg"/> of the cached engine: it reaches every
    ///     array bound to that engine, existing ones included.</para>
    ///     <para><b>Operands.</b> A member receives plain arrays: the <c>np.linalg</c> functions decide the
    ///     <c>numpy.matrix</c> and <c>numpy.char.chararray</c> classes before they call it (wrapping its result where NumPy
    ///     returns the class), so an override never sees those classes from them. A member called directly computes
    ///     with whatever it is given.</para>
    ///     <para><b>Results</b> are NumPy's objects: dtype (float64 for bool / integer operands, float32 kept, complex
    ///     stays complex128; NumSharp has no complex64), shape, memory order and ownership, and NumPy's 0-d scalar
    ///     stand-ins (read-only 0-d arrays) where NumPy returns a scalar.</para>
    /// </remarks>
    public abstract class LinalgEngine
    {
        /// <summary>
        ///     NumPy's <c>np.linalg.cholesky(a, upper=False)</c>: the Cholesky factor of a Hermitian positive-definite
        ///     matrix, or of each matrix in a stack.
        /// </summary>
        /// <param name="a">The matrix or stack (the last two axes, square). Only the lower triangle is read, so a
        ///     non-Hermitian operand is factorised as its own reflection rather than rejected.</param>
        /// <param name="upper">False (NumPy's default) for the lower factor <c>L</c> with <c>a = L L*</c>; true for the
        ///     upper <c>U</c> with <c>a = U* U</c>.</param>
        /// <returns>The triangular factor, a fresh array of the loop dtype (float64 for bool / integer operands).</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a matrix that is not positive
        ///     definite (<c>Matrix is not positive definite</c>).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>array type float16 is unsupported in
        ///     linalg</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal, a UFuncTypeError).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty
        ///     operand or a zero-size batch needs none).</exception>
        public abstract NDArray Cholesky(NDArray a, bool upper = false);

        /// <summary>
        ///     NumPy's <c>np.linalg.cond(x, p=None)</c>: the condition number of a matrix in the given norm, or of each
        ///     matrix in a stack.
        /// </summary>
        /// <param name="x">The matrix or stack.</param>
        /// <param name="p">The order: null (NumPy's default) and ±2 from the singular values; ±1, ±inf and <c>"fro"</c> /
        ///     <c>"nuc"</c> as <c>norm(x, p) * norm(inv(x), p)</c>.</param>
        /// <returns>The condition number, real: a 0-d array for one matrix (NumPy's scalar), one per matrix of a stack; inf
        ///     for a singular matrix, NaN only where the matrix itself holds a NaN.</returns>
        /// <exception cref="LinAlgError">An empty operand (<c>cond is not defined on empty arrays</c>), a non-square one
        ///     for an inverse-based order, or the nuclear norm of a singular matrix (the SVD of its NaN inverse: <c>SVD did
        ///     not converge</c>).</exception>
        /// <exception cref="TypeError">An array or list order (NumPy's <c>p in {2, -2}</c> hashes it: <c>unhashable type:
        ///     'numpy.ndarray'</c> / <c>'list'</c>); a float16, decimal or char operand.</exception>
        /// <exception cref="ValueError">An order the norm refuses (<c>Invalid norm order for matrices.</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">An SVD-based order (null, ±2, <c>"nuc"</c>) over a non-empty
        ///     operand that no matrix backend serves. The inverse-based orders need none: the managed LU inverts.</exception>
        /// <remarks>The inverse runs in float64 (complex128 for a complex operand) whatever the result dtype, and a
        ///     singular matrix's inverse is NaN rather than an error, as NumPy computes it under
        ///     <c>errstate(all='ignore')</c>.</remarks>
        public abstract NDArray Cond(NDArray x, object p = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.cross(x1, x2, axis=-1)</c>, the Array-API cross product: stricter than
        ///     <see cref="np.cross"/>, it takes 3-vectors only, over one axis shared by both operands and the result.
        /// </summary>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="axis">The axis holding the three components, in both operands and in the result (negative counts
        ///     from the end).</param>
        /// <returns>np.cross's result: <c>promote_types(x1.dtype, x2.dtype)</c> (a strong promotion), a view of a fresh
        ///     C-ordered array whose component axis is moved to <paramref name="axis"/>.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range for either operand (<c>tuple index out of
        ///     range</c>: NumPy indexes <c>x1.shape[axis]</c> first).</exception>
        /// <exception cref="ValueError">The component axis of either operand does not have length 3, or the other axes do
        ///     not broadcast.</exception>
        /// <exception cref="TypeError">Either operand holds text (the multiply has no string loop).</exception>
        public abstract NDArray Cross(NDArray x1, NDArray x2, int axis = -1);

        /// <summary>
        ///     NumPy's <c>np.linalg.det(a)</c>: the determinant of a matrix, or of each matrix in a stack, from an LU
        ///     factorisation (<c>getrf</c>), so a large or ill-conditioned matrix can overflow where <see cref="Slogdet"/>
        ///     stays finite.
        /// </summary>
        /// <param name="a">The square matrix or stack.</param>
        /// <returns>The determinant in the loop dtype: NumPy's scalar (a read-only 0-d array) for one matrix, one value per
        ///     matrix of a stack.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <remarks>Needs no backend: without one the managed LU computes it, allclose to NumPy's blocked LAPACK (a
        ///     matrix whose zero pivot appears only after several eliminations can give a tiny value where LAPACK gives
        ///     0).</remarks>
        public abstract NDArray Det(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.linalg.eig(a)</c>: the eigenvalues and right eigenvectors of a general (not necessarily
        ///     symmetric) matrix, or of each matrix in a stack.
        /// </summary>
        /// <param name="a">The square matrix or stack; it must hold no inf or NaN.</param>
        /// <returns><c>(eigenvalues, eigenvectors)</c>, the columns of the second being the vectors. The dtype is
        ///     DATA-dependent: real (float64, float32 kept) for a real operand whose eigenvalues are all real, else
        ///     complex128; a float32 operand's complex result carries NumPy's complex64 values.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, an operand holding inf / NaN
        ///     (<c>Array must not contain infs or NaNs</c>), or no convergence.</exception>
        /// <exception cref="TypeError">A text operand (<c>isfinite</c> has no string loop), or a float16, decimal or char
        ///     operand.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract (NDArray eigenvalues, NDArray eigenvectors) Eig(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.linalg.eigh(a, UPLO='L')</c>: the eigenvalues and eigenvectors of a real symmetric or complex
        ///     Hermitian matrix, or of each matrix in a stack. Faster than <see cref="Eig"/> and always real.
        /// </summary>
        /// <param name="a">The symmetric / Hermitian matrix or stack.</param>
        /// <param name="UPLO">The triangle that holds the data, <c>'L'</c> (NumPy's default) or <c>'U'</c>, either case.
        ///     The other triangle is NOT read, so a non-symmetric operand is interpreted as its own reflection.</param>
        /// <returns><c>(eigenvalues, eigenvectors)</c>: the eigenvalues ascending and real (the real counterpart dtype
        ///     for a complex operand).</returns>
        /// <exception cref="ValueError"><paramref name="UPLO"/> is neither 'L' nor 'U' (<c>UPLO argument must be 'L' or
        ///     'U'</c>, checked before the operand).</exception>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or no convergence.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract (NDArray eigenvalues, NDArray eigenvectors) Eigh(NDArray a, char UPLO = 'L');

        /// <summary>
        ///     NumPy's <c>np.linalg.eigvals(a)</c>: <see cref="Eig"/>'s eigenvalues without the eigenvectors.
        /// </summary>
        /// <param name="a">The square matrix or stack; it must hold no inf or NaN.</param>
        /// <returns>The eigenvalues: real for a real operand whose eigenvalues are all real, else complex128.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, an operand holding inf / NaN, or no
        ///     convergence.</exception>
        /// <exception cref="TypeError">A text operand, or a float16, decimal or char operand.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract NDArray Eigvals(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.linalg.eigvalsh(a, UPLO='L')</c>: <see cref="Eigh"/>'s eigenvalues without the eigenvectors.
        ///     LAPACK computes them another way when it computes no vectors, so they can differ from Eigh's in the last
        ///     bits, in NumPy as here.
        /// </summary>
        /// <param name="a">The symmetric / Hermitian matrix or stack.</param>
        /// <param name="UPLO">The triangle that holds the data, <c>'L'</c> or <c>'U'</c>, either case.</param>
        /// <returns>The eigenvalues, ascending and real.</returns>
        /// <exception cref="ValueError"><paramref name="UPLO"/> is neither 'L' nor 'U'.</exception>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or no convergence.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract NDArray Eigvalsh(NDArray a, char UPLO = 'L');

        /// <summary>
        ///     NumPy's <c>np.linalg.inv(a)</c>: the multiplicative inverse of a matrix, or of each matrix in a stack, by
        ///     solving <c>a x = I</c> (<c>gesv</c>), so a singular operand raises from the factorisation, not from a
        ///     determinant test.
        /// </summary>
        /// <param name="a">The square matrix or stack.</param>
        /// <returns>The inverse, a fresh array of the loop dtype (float64 for bool / integer operands).</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a singular matrix (<c>Singular
        ///     matrix</c>).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <remarks>Needs no backend: without one the managed LU computes it.</remarks>
        public abstract NDArray Inv(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.linalg.lstsq(a, b, rcond=None)</c>: the least-squares solution to <c>a x = b</c>. It takes a
        ///     single matrix, not a stack.
        /// </summary>
        /// <param name="a">The coefficient matrix <c>(M, N)</c>, exactly two-dimensional.</param>
        /// <param name="b">The ordinate values, <c>(M,)</c> or <c>(M, K)</c>.</param>
        /// <param name="rcond">Singular values below <c>rcond</c> times the largest are treated as zero; null for NumPy's
        ///     <c>eps * max(M, N)</c> (float64's eps, the compute type's).</param>
        /// <returns>NumPy's four-tuple <c>(solution, residuals, rank, singularValues)</c>: the residuals EMPTY unless the
        ///     system is overdetermined and of full rank; the rank NumPy's scalar.</returns>
        /// <exception cref="LinAlgError">An operand that is not two-dimensional (after a 1-D <paramref name="b"/> became a
        ///     column), <paramref name="b"/>'s rows differ from <paramref name="a"/>'s (<c>Incompatible dimensions</c>), or
        ///     no convergence.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(
            NDArray a, NDArray b, double? rcond = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.matmul(x1, x2)</c>, the Array-API spelling of <see cref="np.matmul"/>: the same
        ///     operation with every keyword at its default.
        /// </summary>
        /// <param name="x1">The left operand (at least one-dimensional).</param>
        /// <param name="x2">The right operand (at least one-dimensional).</param>
        /// <returns>np.matmul's product: a fresh array that owns its data, NumPy's 0-d scalar stand-in for two
        ///     vectors.</returns>
        /// <exception cref="ValueError">A 0-d operand, or core dimensions that do not match, or loop axes that do not
        ///     broadcast (NumPy's gufunc texts).</exception>
        /// <exception cref="TypeError">Text operands (matmul has no string loop).</exception>
        /// <exception cref="NullReferenceException">An operand is null.</exception>
        public abstract NDArray Matmul(NDArray x1, NDArray x2);

        /// <summary>
        ///     NumPy's <c>np.linalg.matrix_norm(x, keepdims=False, ord='fro')</c>, the Array-API matrix norm: always over
        ///     the LAST TWO axes.
        /// </summary>
        /// <param name="x">The matrix or stack.</param>
        /// <param name="keepdims">Leave the two reduced axes in the result with length 1.</param>
        /// <param name="ord">The matrix order; null stands for NumPy's default <c>"fro"</c> (C# cannot spell a non-null
        ///     default for an <c>object</c> parameter).</param>
        /// <returns>The matrix norm over the last two axes.</returns>
        /// <exception cref="ValueError">An invalid matrix order, or a min order of an empty matrix.</exception>
        /// <exception cref="AxisError">An operand of fewer than two axes.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="OpenBlasMissingBackendException">An order 2, -2 or <c>"nuc"</c> over a non-empty matrix with no
        ///     matrix backend installed.</exception>
        public abstract NDArray MatrixNorm(NDArray x, bool keepdims = false, object ord = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.matrix_power(a, n)</c>: a square matrix (or each of a stack) raised to an integer power
        ///     by binary exponentiation; a negative power inverts first.
        /// </summary>
        /// <param name="a">The square matrix or stack (the last two axes).</param>
        /// <param name="n">Any integer: 0 gives the identity in <paramref name="a"/>'s dtype and memory order.</param>
        /// <returns>The power. <b>Footgun (NumPy's own):</b> <c>n == 1</c> returns <paramref name="a"/> ITSELF, so writing
        ///     to the result writes to <paramref name="a"/>; <c>n == -1</c> returns the fresh inverse.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a negative power of a singular
        ///     matrix.</exception>
        /// <exception cref="TypeError">A text matrix with <c>n</c> of 2 or more (matmul has no string loop), or with a
        ///     negative <c>n</c> (inv's input cast).</exception>
        public abstract NDArray MatrixPower(NDArray a, int n);

        /// <summary>
        ///     NumPy's <c>np.linalg.matrix_rank(A, tol=None, hermitian=False, rtol=None)</c>: the number of singular values
        ///     above the tolerance, per matrix.
        /// </summary>
        /// <param name="A">The matrix or stack. Below two axes NumPy's answer is a PREDICATE, <c>int(not all(A ==
        ///     0))</c>: <c>matrix_rank([1, 2, 3])</c> is 1, an empty operand 0.</param>
        /// <param name="tol">The absolute threshold; null for <c>S.max() * max(M, N) * eps</c> (eps of the singular values'
        ///     dtype).</param>
        /// <param name="hermitian">Take the singular values from the Hermitian (real symmetric) matrix whose lower
        ///     triangle <paramref name="A"/> holds (eigvalsh's square check applies).</param>
        /// <param name="rtol">The relative threshold (the Array-API spelling).</param>
        /// <returns>The rank: an int64 0-d array for one matrix (or below two axes), one per matrix of a stack.</returns>
        /// <exception cref="ValueError">Both <paramref name="tol"/> and <paramref name="rtol"/> are set.</exception>
        /// <exception cref="LinAlgError">A non-square operand with <paramref name="hermitian"/>.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand of two or more axes (NumPy's gufunc input-cast
        ///     refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand of two or more axes that no matrix backend
        ///     serves.</exception>
        public abstract NDArray MatrixRank(NDArray A, double? tol = null, bool hermitian = false, double? rtol = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.multi_dot(arrays, out=None)</c>: a chained matrix product in the cheapest association
        ///     order (the matrix-chain dynamic program; a direct comparison for three).
        /// </summary>
        /// <param name="arrays">Two or more arrays. With three or more, every one but the ends must be 2-D; a 1-D first
        ///     operand is a row and a 1-D last one a column, the added axis removed again afterwards. Two arrays are a
        ///     plain <c>np.dot</c>.</param>
        /// <param name="out">The final product's out array (np.dot's strict out: the product's exact dtype and rank,
        ///     C-contiguous, the TWO-dimensional product's shape), or null.</param>
        /// <returns>The product: <paramref name="out"/> itself when given (a two-array chain, or no vector end); a view of it
        ///     raveled when one end was a vector; NumPy's scalar stand-in when both were; otherwise a fresh array.</returns>
        /// <exception cref="ValueError">Fewer than two arrays (<c>Expecting at least two arrays.</c>), or an out array
        ///     np.dot refuses.</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands (np.dot's <c>shapes (…) and (…) not
        ///     aligned</c>).</exception>
        /// <exception cref="LinAlgError">With three or more arrays, an operand other than a vector end that is not
        ///     two-dimensional.</exception>
        public abstract NDArray MultiDot(NDArray[] arrays, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.norm(x, ord=None, axis=None, keepdims=False)</c>: a matrix or vector norm. The number of
        ///     reduced axes decides the vocabulary of orders: one axis is a vector norm, two a matrix norm.
        /// </summary>
        /// <param name="x">The operand: integer, bool and text (parsed by <c>astype(float)</c>) operands compute in
        ///     float64.</param>
        /// <param name="ord">The order: null (Frobenius for a matrix, 2 for a vector), a number (an infinity included), or
        ///     <c>"fro"</c> / <c>"f"</c> / <c>"nuc"</c>.</param>
        /// <param name="axis">The reduced axes (one or two), or null for every axis (NumPy's int axis is a one-element
        ///     array here).</param>
        /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
        /// <returns>The norm: a 0-d array for a whole-array norm, else the reduced array. An empty operand's inf-, 1- and
        ///     2-norms are 0.0 (NumPy 2.x reduces those with <c>initial=0</c>).</returns>
        /// <exception cref="ValueError">An order that is invalid for the number of reduced axes, duplicate axes, more than
        ///     two reduced axes (<c>Improper number of dimensions to norm.</c>), a min order of an empty operand, or a text
        ///     element that does not parse as a float.</exception>
        /// <exception cref="AxisError">An axis out of range.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A matrix order 2, -2 or <c>"nuc"</c> over a non-empty matrix
        ///     with no matrix backend installed; every other order is a reduction and needs none.</exception>
        public abstract NDArray Norm(NDArray x, object ord = null, int[] axis = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.linalg.outer(x1, x2)</c>: the outer product of two VECTORS. Not <see cref="np.outer"/>, which
        ///     flattens whatever it is given; this one refuses anything but 1-D.
        /// </summary>
        /// <param name="x1">The first vector.</param>
        /// <param name="x2">The second vector.</param>
        /// <returns>The <c>(len(x1), len(x2))</c> outer product, a fresh array.</returns>
        /// <exception cref="ValueError">Either operand is not one-dimensional (<c>Input arrays must be one-dimensional,
        ///     but they are x1.ndim={n} and x2.ndim={m}.</c>).</exception>
        public abstract NDArray Outer(NDArray x1, NDArray x2);

        /// <summary>
        ///     NumPy's <c>np.linalg.pinv(a, rcond=None, hermitian=False, rtol=None)</c>: the Moore-Penrose pseudo-inverse,
        ///     from the singular value decomposition.
        /// </summary>
        /// <param name="a">The matrix or stack.</param>
        /// <param name="rcond">Singular values at or below <c>rcond</c> times the largest are treated as zero; with
        ///     <paramref name="rtol"/> null too, NumPy's <c>1e-15</c>.</param>
        /// <param name="hermitian">Decompose the Hermitian (real symmetric) matrix whose lower triangle the conjugated
        ///     operand holds, through the symmetric eigensolver.</param>
        /// <param name="rtol">The Array-API spelling of <paramref name="rcond"/>; setting both is an error.</param>
        /// <returns>The pseudo-inverse, shape <c>(..., N, M)</c>: float64 / float32 / complex128 by the loop dtype, the
        ///     operand's own dtype for an empty operand.</returns>
        /// <exception cref="ValueError">Both <paramref name="rcond"/> and <paramref name="rtol"/> are set, or an empty
        ///     operand of fewer than two axes.</exception>
        /// <exception cref="TypeError">A text operand (<c>cannot conjugate non-numeric dtype</c>), or a float16, decimal or
        ///     char operand.</exception>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand with
        ///     <paramref name="hermitian"/>.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        /// <remarks>C# cannot tell NumPy's <c>rtol=_NoValue</c> (1e-15) from <c>rtol=None</c> (the Array-API default
        ///     <c>max(M, N) * eps</c>): with neither tolerance set, NumPy's 1e-15 applies.</remarks>
        public abstract NDArray Pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.qr(a, mode='reduced')</c>: <c>a = Q R</c> with Q orthonormal and R upper triangular,
        ///     per matrix of a stack.
        /// </summary>
        /// <param name="a">The matrix or stack.</param>
        /// <param name="mode"><c>"reduced"</c> (Q <c>(..., M, K)</c>, R <c>(..., K, N)</c>, <c>K = min(M, N)</c>),
        ///     <c>"complete"</c> (a square Q), <c>"r"</c> (R alone), <c>"raw"</c> (LAPACK's packed <c>(h, tau)</c>), or
        ///     the legacy aliases NumPy still honours (<c>"f"</c> / <c>"full"</c> for reduced, <c>"e"</c> /
        ///     <c>"economic"</c>).</param>
        /// <returns><c>(Q, R)</c>: for <c>"r"</c> the Q slot is null, for <c>"raw"</c> the pair is <c>(h, tau)</c>, for
        ///     <c>"economic"</c> the packed matrix and null.</returns>
        /// <exception cref="ValueError">An unknown or null <paramref name="mode"/> (<c>Unrecognized mode '...'</c>,
        ///     checked before the operand).</exception>
        /// <exception cref="LinAlgError">Fewer than two axes.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract (NDArray Q, NDArray R) Qr(NDArray a, string mode = "reduced");

        /// <summary>
        ///     NumPy's <c>np.linalg.slogdet(a)</c>: the sign and the natural log of the absolute determinant, the
        ///     overflow-safe <see cref="Det"/>.
        /// </summary>
        /// <param name="a">The square matrix or stack.</param>
        /// <returns><c>(sign, logabsdet)</c>, such that <c>sign * exp(logabsdet)</c> is the determinant: NumPy's scalars
        ///     for one matrix, arrays for a stack. A singular matrix gives <c>(0, -inf)</c>; a complex operand a
        ///     unit-modulus complex sign.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <remarks>Needs no backend: without one the managed LU computes it.</remarks>
        public abstract (NDArray sign, NDArray logabsdet) Slogdet(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.linalg.solve(a, b)</c>: the solution of <c>a x = b</c>. <paramref name="b"/> is a stack of
        ///     vectors only when it is exactly 1-D (NumPy 2.0's rule); any other rank is a stack of matrices, so the two
        ///     gufuncs are chosen by rank alone and report their own core-dimension errors.
        /// </summary>
        /// <param name="a">The coefficient matrix or stack <c>(..., M, M)</c>.</param>
        /// <param name="b">The ordinate values, <c>(..., M)</c> or <c>(..., M, K)</c>; the stacked axes broadcast.</param>
        /// <returns>The solution, a fresh array of the loop dtype (float64 for bool / integer operands).</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square <paramref name="a"/>, or a singular matrix.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="ValueError">A core dimension of <paramref name="b"/> that does not match, a
        ///     <paramref name="b"/> of too few axes for the matrix gufunc, or stacked axes that do not broadcast (NumPy's
        ///     gufunc texts).</exception>
        /// <remarks>Needs no backend: without one the managed LU computes it.</remarks>
        public abstract NDArray Solve(NDArray a, NDArray b);

        /// <summary>
        ///     NumPy's <c>np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)</c>: the singular value
        ///     decomposition <c>a = U diag(S) Vh</c>, per matrix of a stack; the routine behind pinv, matrix_rank, cond and
        ///     the spectral / nuclear norms.
        /// </summary>
        /// <param name="a">The matrix or stack.</param>
        /// <param name="full_matrices">True (NumPy's default) for square U and Vh, false for <c>K = min(M, N)</c> columns /
        ///     rows; ignored with <paramref name="hermitian"/>.</param>
        /// <param name="compute_uv">False for the singular values alone (U and Vh null).</param>
        /// <param name="hermitian">Decompose the Hermitian (real symmetric) matrix whose lower triangle
        ///     <paramref name="a"/> holds, through eigh / eigvalsh as NumPy does: the upper triangle is not read, so a
        ///     non-Hermitian operand gives another matrix's factors without an error.</param>
        /// <returns><c>(U, S, Vh)</c>, the singular values descending and real.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand with <paramref name="hermitian"/>, or
        ///     no convergence.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool full_matrices = true, bool compute_uv = true,
            bool hermitian = false);

        /// <summary>
        ///     NumPy's <c>np.linalg.svdvals(x)</c>, the Array-API spelling of <c>svd(x, compute_uv=False)</c>: the singular
        ///     values alone.
        /// </summary>
        /// <param name="x">The matrix or stack.</param>
        /// <returns>The singular values, descending and real.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or no convergence.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        public abstract NDArray Svdvals(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.linalg.tensordot(x1, x2, axes=2)</c> with an integer <paramref name="axes"/>, the Array-API
        ///     spelling of <see cref="np.tensordot(NDArray, NDArray, int)"/>: a sum product over the last
        ///     <paramref name="axes"/> axes of <paramref name="x1"/> and the first ones of <paramref name="x2"/>.
        /// </summary>
        /// <param name="x1">The first tensor.</param>
        /// <param name="x2">The second tensor.</param>
        /// <param name="axes">How many axes to contract: 0 is the outer product, 1 <c>np.dot</c>; a negative count
        ///     contracts nothing.</param>
        /// <returns>The contraction, shape <c>x1.shape[:-axes] + x2.shape[axes:]</c>.</returns>
        /// <exception cref="ValueError">The contracted extents differ (<c>shape-mismatch for sum</c>).</exception>
        /// <exception cref="IndexError">The count exceeds an operand's rank (<c>tuple index out of range</c>).</exception>
        public abstract NDArray Tensordot(NDArray x1, NDArray x2, int axes = 2);

        /// <summary>
        ///     NumPy's <c>np.linalg.tensordot(x1, x2, axes=(axesA, axesB))</c>, the Array-API spelling of
        ///     <see cref="np.tensordot(NDArray, NDArray, int[], int[])"/>: <paramref name="axesA"/> of
        ///     <paramref name="x1"/> contracted element by element with <paramref name="axesB"/> of <paramref name="x2"/>.
        /// </summary>
        /// <param name="x1">The first tensor.</param>
        /// <param name="x2">The second tensor.</param>
        /// <param name="axesA">The contracted axes of <paramref name="x1"/>.</param>
        /// <param name="axesB">The contracted axes of <paramref name="x2"/>, paired in order.</param>
        /// <returns>The contraction: <paramref name="x1"/>'s free axes, then <paramref name="x2"/>'s.</returns>
        /// <exception cref="ValueError">A repeated axis (<c>duplicate axes are not allowed in tensordot</c>, checked first),
        ///     or lists of unequal length or mismatched extents (<c>shape-mismatch for sum</c>).</exception>
        /// <exception cref="IndexError">A contraction axis out of range (<c>tuple index out of range</c>).</exception>
        public abstract NDArray Tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB);

        /// <summary>
        ///     NumPy's <c>np.linalg.tensorinv(a, ind=2)</c>: the inverse of an N-dimensional array with respect to a
        ///     tensordot contraction over its last <c>a.ndim - ind</c> axes.
        /// </summary>
        /// <param name="a">The tensor to invert.</param>
        /// <param name="ind">How many LEADING axes form the "row" side; must be positive.</param>
        /// <returns>The inverse, shape <c>a.shape[ind:] + a.shape[:ind]</c>: a view of the inverse of the square matrix
        ///     (NumPy's <c>ia.reshape(*invshape)</c>).</returns>
        /// <exception cref="ValueError"><paramref name="ind"/> is not positive (<c>Invalid ind argument.</c>), or the
        ///     reshape to a square matrix is impossible.</exception>
        /// <exception cref="LinAlgError">The reshaped matrix is not square, or is singular.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand, or a 0-d operand whose 1x1 inverse succeeded
        ///     (NumPy's <c>reshape() takes exactly 1 argument (0 given)</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (inv's gufunc input-cast refusal).</exception>
        public abstract NDArray Tensorinv(NDArray a, int ind = 2);

        /// <summary>
        ///     NumPy's <c>np.linalg.tensorsolve(a, b, axes=None)</c>: the solution x of the tensor equation
        ///     <c>tensordot(a, x, x.ndim) = b</c>, a reshape of <see cref="Solve"/>.
        /// </summary>
        /// <param name="a">The coefficient tensor; its leading <c>b.ndim</c> axes (after <paramref name="axes"/> moved some
        ///     to the end) index the equations.</param>
        /// <param name="b">The right-hand side.</param>
        /// <param name="axes">Axes of <paramref name="a"/> to move to the right first, as LIST VALUES (NumPy never
        ///     normalizes them: only <c>0 .. a.ndim - 1</c> are found), or null.</param>
        /// <returns>The solution, of NumPy's <c>oldshape = a.shape[-(a.ndim - b.ndim):]</c> (after the axis move); it owns
        ///     its data (NumPy's <c>res.shape = oldshape</c>).</returns>
        /// <exception cref="ValueError">An axis that is not one of <c>0 .. a.ndim - 1</c> (<c>list.remove(x): x not in
        ///     list</c>), or a right-hand side that does not match the square system (solve's gufunc text).</exception>
        /// <exception cref="LinAlgError"><c>a.size</c> is not <c>prod(oldshape) ** 2</c>, or the system is
        ///     singular.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
        /// <exception cref="ArgumentException">A text operand (solve's gufunc input-cast refusal).</exception>
        public abstract NDArray Tensorsolve(NDArray a, NDArray b, int[] axes = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.trace(x, offset=0, dtype=None)</c>, the Array-API trace: the sum along the diagonals of
        ///     the LAST two axes, where <see cref="np.trace"/> defaults to the first two (a <c>(2, 3, 3)</c> stack gives
        ///     <c>(2,)</c> here and <c>(3,)</c> there).
        /// </summary>
        /// <param name="x">The operand, at least two-dimensional.</param>
        /// <param name="offset">The diagonal's offset from the main one (positive above).</param>
        /// <param name="dtype">The accumulator and result dtype, or null for np.trace's rule (NumPy's sum rule).</param>
        /// <returns>The traces, one per matrix (a 0-d array for one matrix).</returns>
        /// <exception cref="AxisError">An operand of fewer than two axes.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="TypeError">A text operand with a non-text <paramref name="dtype"/>, or a string
        ///     <paramref name="dtype"/>.</exception>
        public abstract NDArray Trace(NDArray x, int offset = 0, DType dtype = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.vecdot(x1, x2, axis=-1)</c>, the Array-API spelling of <see cref="np.vecdot"/>: the
        ///     dot product of vectors along <paramref name="axis"/>, conjugating <paramref name="x1"/>.
        /// </summary>
        /// <param name="x1">The first operand (conjugated when complex).</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="axis">The axis of the vectors in both operands (negative counts from the end).</param>
        /// <returns>The dot products: the operands' other axes broadcast, in the loop dtype (an int32 pair stays
        ///     int32).</returns>
        /// <exception cref="ValueError">A 0-d operand, vectors of different lengths, or other axes that do not broadcast
        ///     (NumPy's gufunc texts).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">Text operands (no vecdot loop).</exception>
        /// <exception cref="NullReferenceException">An operand is null.</exception>
        public abstract NDArray Vecdot(NDArray x1, NDArray x2, int axis = -1);

        /// <summary>
        ///     NumPy's <c>np.linalg.vector_norm(x, axis=None, keepdims=False, ord=2)</c> with a tuple of axes (or none): the
        ///     reduced axes moved to the front and merged into one, then a VECTOR norm over it, whatever their number.
        /// </summary>
        /// <param name="x">The operand: integer and bool operands compute in float64.</param>
        /// <param name="axis">The reduced axes, any number of them (negative counts from the end), or null for the
        ///     flattened array. An empty array reduces nothing (the elementwise magnitude).</param>
        /// <param name="keepdims">Leave the reduced axes in the result with length 1.</param>
        /// <param name="ord">The vector order; null stands for NumPy's default of 2.</param>
        /// <returns>The vector norm over the axes.</returns>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">An axis appears twice (<c>repeated axis</c>), a text order, or -inf of an empty
        ///     operand.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <remarks>NumPy's int axis and its one-element tuple take different statements (the tuple is moved and merged
        ///     first), which reduce in different layouts: the int is <see cref="VectorNorm(NDArray, int, bool, object)"/>.</remarks>
        public abstract NDArray VectorNorm(NDArray x, int[] axis = null, bool keepdims = false, object ord = null);

        /// <summary>
        ///     NumPy's <c>np.linalg.vector_norm(x, axis=axis, keepdims=False, ord=2)</c> with an integer axis: NumPy's
        ///     <c>norm(x, axis=axis, ord=ord)</c>, then for keepdims a reshape that puts the axis back with length 1.
        /// </summary>
        /// <param name="x">The operand: integer and bool operands compute in float64.</param>
        /// <param name="axis">The reduced axis (negative counts from the end).</param>
        /// <param name="keepdims">Leave the reduced axis in the result with length 1.</param>
        /// <param name="ord">The vector order; null stands for NumPy's default of 2.</param>
        /// <returns>The norm along <paramref name="axis"/>; with <paramref name="keepdims"/> a view of it (NumPy's
        ///     reshape, which does not own its data).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range (a 0-d operand has no axis).</exception>
        /// <exception cref="ValueError">A text order, or -inf of an empty axis.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        public abstract NDArray VectorNorm(NDArray x, int axis, bool keepdims = false, object ord = null);
    }
}
