using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: with <c>hermitian</c> the eigh-based branch (<see cref="SvdHermitian"/>,
        ///     <see cref="SvdvalsHermitian"/>, through this engine's <see cref="Eigh"/> / <see cref="Eigvalsh"/>);
        ///     otherwise the rank check, <c>_commonType</c>, the text refusal of the <c>svd_f</c> / <c>svd_s</c> / <c>svd</c>
        ///     gufunc's input cast, then <see cref="SvdKernel"/> at the loop dtype.
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp; the (U, S, Vh) tuple is yielded component-wise
        public override (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool full_matrices = true, bool compute_uv = true,
            bool hermitian = false)
        {
            // NumPy's hermitian=True branch runs before any check of svd's own: eigh / eigvalsh make theirs (the square
            // check, _commonType, their gufunc's input cast).
            if (hermitian)
                return compute_uv ? SvdHermitian(a) : (null, SvdvalsHermitian(a), null);

            AssertStacked2d(a);
            var common = CommonType(a);
            ThrowIfTextGufunc(compute_uv ? (full_matrices ? "svd_f" : "svd_s") : "svd", a);
            return SvdKernel(ToCommon(a, common), full_matrices, compute_uv);
        }

        /// <summary>
        ///     NumPy's <c>svd(a, hermitian=True)</c> with <c>compute_uv=True</c>, statement for statement:
        ///     <code>
        ///     s, u = eigh(a)
        ///     sgn = np.copysign(1.0, s)   # NumPy 2.5; 2.4: sign(s)
        ///     s = abs(s)
        ///     sidx = argsort(s)[..., ::-1]
        ///     sgn = np.take_along_axis(sgn, sidx, axis=-1)
        ///     s = np.take_along_axis(s, sidx, axis=-1)
        ///     u = np.take_along_axis(u, sidx[..., None, :], axis=-1)
        ///     vt = transpose(u * sgn[..., None, :]).conjugate()
        ///     </code>
        /// </summary>
        /// <param name="a">The Hermitian (real symmetric) matrix, or a stack of them; only its lower triangle is read
        ///     (eigh's <c>UPLO='L'</c>).</param>
        /// <returns><c>(U, S, Vh)</c>: S the absolute eigenvalues descending, U the eigenvectors in that order, Vh the
        ///     conjugate transpose of U with each row multiplied by its eigenvalue's sign. For a real operand Vh is a
        ///     transposed VIEW of that product (<c>ndarray.conjugate</c> returns a real array itself), for a complex one a
        ///     new array in the view's order (the conjugate ufunc keeps it).</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand (eigh's checks).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="TypeError">A text operand (eigh's gufunc input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        /// <remarks>
        ///     The sign is <c>np.copysign(1.0, s)</c> since NumPy 2.5 ("avoid zero sign"): a zero eigenvalue signs its row
        ///     of Vh by the zero's own sign (+1 for 0.0, -1 for -0.0), so that row is its eigenvector and Vh stays unitary.
        ///     NumPy 2.4's <c>sign(s)</c> was 0 there and zeroed the row. The <c>1.0</c> is a Python float, weak, so the
        ///     sign keeps s's dtype (float32 for a float32 operand). NumPy's
        ///     <c>argsort</c> is its default kind, x86-simd-sort's bitonic argsort on an AVX2 host, which keeps equal values
        ///     in order up to three elements and reorders them from four; NumSharp's argsort is stable. Where two absolute
        ///     eigenvalues of a matrix of four or more rows are equal, S is the same and the tied columns of U, with their
        ///     rows of Vh, can come in another order.
        ///     <para>Each statement calls the member its main-namespace function dispatches to, on eigh's plain factors:
        ///     the operand engine's copysign, abs, argsort (ndarray.argsort), multiply, swapaxes and conjugate, the array
        ///     object's take_along_axis, expand_dims and empty_like allocation.</para>
        /// </remarks>
        [NDScoped] // the eigh factors, the sign / abs / argsort temps and the product behind Vh; U, S and Vh are yielded
        private (NDArray U, NDArray S, NDArray Vh) SvdHermitian(NDArray a)
        {
            var (w, v) = Eigh(a, 'L');
            // np.copysign(1.0, s): the weak Python float adopts s's dtype; a ±0 eigenvalue gives ±1, never 0.
            var sgn = w.TensorEngine.CopySign(NDArray.PythonLiteral(1.0), w);
            var s = w.TensorEngine.Abs(w);
            var sidx = ReverseLastAxis(s.TensorEngine.ArgSort(s, -1, null, null, null));
            sgn = NDArray.take_along_axis(sgn, sidx, -1);
            s = NDArray.take_along_axis(s, sidx, -1);
            // sidx[..., None, :] and sgn[..., None, :]: one index / sign row broadcast over the rows of each matrix.
            var u = NDArray.take_along_axis(v, NDArray.expand_dims(sidx, -2), -1);
            // transpose(u * sgn[..., None, :]): the product, then the swap of its last two axes.
            var scaled = u.TensorEngine.Multiply(u, NDArray.expand_dims(sgn, -2));
            var vt = scaled.TensorEngine.SwapAxes(scaled, -1, -2);
            // transpose(...).conjugate(): the conjugate ufunc for a complex product; a real one is returned itself, so Vh
            // stays the transposed view (the scope's release of the product leaves the view's own reference).
            if (vt.typecode == NPTypeCode.Complex)
            {
                // NumPy's ufunc allocates its result in the view's K order (F for one matrix, the swapped permutation for
                // a stack). NumSharp's eager unary ufunc writes C order for a permuted operand (the library's layout gap
                // that the matrix tier's MX05 records for np.conjugate), so the result is allocated in that order (np.empty_like's
                // own allocation: vt's dtype over its K layout, uninitialised) and filled through out=.
                var conj = vt.TensorEngine.NewArray(vt.dtype, Shape.NewLikeLayout(vt.Shape, 'K', (long[])vt.Shape.dimensions.Clone()), false);
                vt.TensorEngine.Conjugate(vt, null, conj, null);
                vt = conj;
            }
            return (u, s, vt);
        }

        /// <summary>
        ///     NumPy's <c>svd(a, compute_uv=False, hermitian=True)</c>: <c>s = eigvalsh(a); s = abs(s); return
        ///     sort(s)[..., ::-1]</c>.
        /// </summary>
        /// <param name="a">The Hermitian (real symmetric) matrix, or a stack of them; only its lower triangle is read.</param>
        /// <returns>The absolute eigenvalues descending: a view of the sorted copy with its last axis reversed (a negative
        ///     stride, not an owner), as NumPy returns them.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand (eigvalsh's checks).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="TypeError">A text operand (eigvalsh's gufunc input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        /// <remarks>eigvalsh is not eigh without the vectors: LAPACK computes the values in another way when it computes no
        ///     vectors, so these can differ from <see cref="SvdHermitian"/>'s S in the last bits, in NumPy as here. The
        ///     absolute value and the sort are the members np.abs and np.sort dispatch to, on the plain eigenvalues.</remarks>
        [NDScoped] // the eigenvalues and their absolute values; the reversed view of the sorted copy is yielded
        private NDArray SvdvalsHermitian(NDArray a)
        {
            var w = Eigvalsh(a, 'L');
            var s = w.TensorEngine.Abs(w);
            return ReverseLastAxis(s.TensorEngine.Sort(s, -1, null, null, null));
        }

        /// <summary>NumPy's <c>x[..., ::-1]</c>: a view of <paramref name="x"/> with its last axis reversed.</summary>
        /// <param name="x">An array of at least one axis.</param>
        /// <returns>The view (shares <paramref name="x"/>'s memory; the last axis's stride negated).</returns>
        private static NDArray ReverseLastAxis(NDArray x) => x[new[] {Slice.Ellipsis, new Slice(null, null, -1)}];

        /// <inheritdoc/>
        /// <remarks>
        ///     <c>svd(x, compute_uv=False)</c>'s statements: the rank check, <c>_commonType</c>, the text refusal of the
        ///     <c>svd</c> gufunc's input cast, then <see cref="SvdKernel"/> without U and Vh.
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp (U/Vh are null under compute_uv: false)
        public override NDArray Svdvals(NDArray x)
        {
            AssertStacked2d(x);
            var common = CommonType(x);
            // NumPy's svd(x, compute_uv=False) calls _umath_linalg.svd: its input cast refuses text.
            ThrowIfTextGufunc("svd", x);
            return SvdKernel(ToCommon(x, common), fullMatrices: false, computeUv: false).S;
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the tolerance arguments, NumPy's predicate below two axes (<c>int(not all(A == 0))</c>),
        ///     and otherwise the count of singular values (this engine's <see cref="Svd(NDArray, bool, bool, bool)"/>)
        ///     above the tolerance, through the members np.amax, np.multiply, np.greater and np.count_nonzero dispatch to.
        ///     An empty matrix (and a stack of them) is rank 0: the largest singular value starts from NumPy 2.5's
        ///     <c>initial=0</c>, where NumPy 2.4 raised the no-identity <c>ValueError</c>.
        /// </remarks>
        [NDScoped]
        public override NDArray MatrixRank(NDArray A, double? tol = null, bool hermitian = false, double? rtol = null)
        {
            if (tol is not null && rtol is not null)
                throw new ValueError("`tol` and `rtol` can't be both set.");

            // NumPy short-circuits below rank 2 without any decomposition: `int(not all(A == 0))`.
            // That is a PREDICATE, not a count — a 0-d operand and a vector alike are rank 1
            // unless every element is zero, so matrix_rank([1,2,3]) is 1 and not 3. An empty
            // operand is rank 0, because all([]) is true.
            if (A.ndim < 2)
            {
                // Text: NumPy's `A == 0` has no string loop, so ndarray.__eq__ answers all-False (a 0-d operand:
                // Python's False) without reading the text — rank 1, or 0 for an empty operand (all([]) is True).
                if (Backends.Strings.NDStringUfuncs.IsText(A))
                    return NDArray.Scalar(A.size == 0 ? 0L : 1L);
                return NDArray.Scalar(A.TensorEngine.CountNonZero(A) == 0 ? 0L : 1L);
            }

            // NumPy's `S = svd(A, compute_uv=False, hermitian=hermitian)`: with hermitian, the absolute eigenvalues of the
            // matrix A's lower triangle holds (eigvalsh's checks first), else the svd gufunc's values.
            var s = Svd(A, full_matrices: true, compute_uv: false, hermitian).S;

            int nd = A.ndim;
            long m = A.Shape.dimensions[nd - 2];
            long n = A.Shape.dimensions[nd - 1];

            NDArray tolArr;
            if (tol is null)
            {
                // Default: max(M,N) * eps(S.dtype) * largest singular value (MATLAB's rule). The eps is
                // the singular values' own — float32's for a float32 operand, float64's otherwise.
                double eps = s.typecode == NPTypeCode.Single ? 1.1920928955078125e-07 : 2.220446049250313e-16;
                // NumPy's default `max(A.shape[-2:]) * finfo(S.dtype).eps` is a Python int times a NumPy scalar of S's
                // dtype: a STRONG scalar of that dtype (the int scales eps by a power of two's multiple, exactly, in either
                // width), so the product below stays in S's dtype (float32 for a float32 operand). A given rtol is
                // `asarray(rtol)[..., newaxis]`, a strong float64 array, which promotes a float32 S to float64.
                NDArray rtolArr = rtol.HasValue
                    ? NDArray.Scalar(rtol.Value)
                    : NDArray.Scalar(Math.Max(m, n) * eps).astype(s.typecode);
                // S.max(axis=-1, keepdims=True, initial=0) * rtol: ndarray.max's member, then np.multiply's. NumPy 2.5's
                // `initial=0` (2.4.5, numpy#30422) gives an EMPTY matrix's empty singular-value axis a largest value of 0
                // instead of the no-identity ValueError, so its rank counts nothing above 0 and is 0 (a stack of empty
                // matrices: one 0 per matrix). A non-empty axis is unchanged: every singular value is >= 0.
                var largest = s.TensorEngine.ReduceAMax(s, -1, true, null, null, 0, null);
                tolArr = largest.TensorEngine.Multiply(largest, rtolArr);
            }
            else
            {
                // asarray(tol)[..., newaxis] — a scalar threshold broadcasts against the last axis.
                tolArr = NDArray.Scalar(tol.Value);
            }

            // count_nonzero(S > tol, axis=-1): np.greater's member, then np.count_nonzero's.
            var above = s.TensorEngine.Greater(s, tolArr);
            return above.TensorEngine.CountNonZero(above, -1, false);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy 2.5's statements: the empty-operand refusal, the hashability of the order, the singular-value orders
        ///     (<c>_assert_stacked_2d</c>, then <see cref="CondSvdvals"/>, NumPy's <c>_cond_svdvals(x)</c>: a non-finite or
        ///     non-converging matrix reads NaN where <c>svd</c> raises), <c>sum(_cond_svdvals(x)) * sum(_cond_svdvals(invx))</c>
        ///     for <c>'nuc'</c>, or <c>norm(x, p) * norm(inv(x), p)</c> for the others (the inverse by <see cref="InvKernel"/>
        ///     in float64 / complex128, NaN for a singular matrix; the norms by this engine's <see cref="Norm"/>), and NumPy's
        ///     NaN-to-inf tail. The divisions, the sums and the product are np.divide's, np.sum's and np.multiply's members.
        ///     NumPy 2.4 took the singular values from <c>svd</c> and the nuclear order from <c>norm</c>, so a NaN matrix, and
        ///     a singular one under <c>'nuc'</c>, raised <c>SVD did not converge</c> (gh-32591).
        /// </remarks>
        [NDScoped]
        public override NDArray Cond(NDArray x, object p = null)
        {
            if (IsEmpty2d(x))
                throw new LinAlgError("cond is not defined on empty arrays");
            // `p is None or p in {2, -2}`: the set membership hashes p, which an ndarray (a typed C# array too, the house
            // map) and a Python list cannot be — whatever they hold, text included.
            if (p is NDArray || (p is Array && p is not string) || PolySequenceLike(p))
                throw new TypeError($"unhashable type: '{(p is NDArray || p.GetType().IsArray && p.GetType().GetElementType() != typeof(object) ? "numpy.ndarray" : "list")}'");

            NDArray r;
            if (p is null || IsOrder(p, 2) || IsOrder(p, -2))
            {
                // The default and ±2 orders come straight from the singular values: the ratio of the
                // largest to the smallest (2/None), or its reciprocal (-2). Division by a zero
                // singular value gives inf/nan here rather than raising, matching NumPy's errstate.
                // NumPy 2.5: `_assert_stacked_2d(x); s = _cond_svdvals(x)` — a failed matrix's NaN values reach the tail.
                AssertStacked2d(x);
                var s = CondSvdvals(x);
                // NumPy's `s[..., 0]` / `s[..., -1]`. Built as Slice[] rather than the "..., 0"
                // STRING form, whose per-call Regex.Split+Regex.Match parse (Slice.ParseSlices)
                // costs ~1.7µs EACH — the dominant managed overhead of this tiny wrapper, on top
                // of an already parity-or-faster gesdd. The Slice[] path resolves identically
                // (`Slice.Index` is index-reduction on the last axis after Ellipsis fills the
                // leading ones), so the extracted first/last singular values are byte-identical.
                var sMax = s[new[] {Slice.Ellipsis, Slice.Index(0)}];
                var sMin = s[new[] {Slice.Ellipsis, Slice.Index(-1)}];
                r = IsOrder(p, -2) ? sMin.TensorEngine.Divide(sMin, sMax) : sMax.TensorEngine.Divide(sMax, sMin);
            }
            else
            {
                AssertStackedSquare(x);
                var common = CommonType(x);
                // NumPy calls _umath_linalg.inv(x, signature=...) next: its input cast refuses text.
                ThrowIfTextGufunc("inv", x);
                // result_t = _realType(result_t): the condition number is real.
                var realT = common == NPTypeCode.Complex ? NPTypeCode.Double : common;

                // NumPy 2.5's statements, in order:
                //     signature = 'D->D' if isComplexType(t) else 'd->d'
                //     with errstate(all='ignore'):
                //         invx = _umath_linalg.inv(x, signature=signature)
                //         if p == 'nuc':
                //             r = (sum(_cond_svdvals(x), axis=-1, initial=0)
                //                  * sum(_cond_svdvals(invx), axis=-1, initial=0))
                //         else:
                //             r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
                //     r = r.astype(result_t, copy=False)
                // The inverse runs in float64 / complex128 whatever result_t is (a float32 matrix's inverse and its norm
                // are float64), and a singular matrix's inverse is NaN instead of an error. Both factors are evaluated,
                // x's first: a bad order is refused by norm ("Invalid norm order for matrices.") even for a singular
                // matrix. The nuclear order sums the singular values itself, because norm's svd would raise for the NaNs
                // of a failed inversion (NumPy 2.4 did: "SVD did not converge"); a NaN sum is then the tail's inf.
                var invx = InvIgnoringErrors(ToCommon(x, common == NPTypeCode.Complex ? NPTypeCode.Complex : NPTypeCode.Double));
                NDArray factorX, factorInv;
                if (p is string nuc && nuc == "nuc")
                {
                    // sum(..., axis=-1, initial=0): np.sum's member over the last axis, the Python int 0 its start.
                    var sX = CondSvdvals(x);
                    var sInv = CondSvdvals(invx);
                    factorX = sX.TensorEngine.ReduceAdd(sX, -1, false, null, null, 0, null);
                    factorInv = sInv.TensorEngine.ReduceAdd(sInv, -1, false, null, null, 0, null);
                }
                else
                {
                    factorX = Norm(x, p, new[] {-2, -1}, false);
                    factorInv = Norm(invx, p, new[] {-2, -1}, false);
                }
                // One matrix's two factors are NumPy scalars, so `*` is scalarmath (the float64 product: a float32 factor
                // widens exactly) and astype keeps a scalar; a stack's are arrays, multiplied by the ufunc (np.multiply's
                // member).
                r = factorX.ndim == 0
                    ? NDArray.Scalar(factorX.astype(np.float64).GetDouble() * factorInv.GetDouble()).astype(realT).MarkReductionScalar()
                    : factorX.TensorEngine.Multiply(factorX, factorInv).astype(realT, false);
            }

            return CondNanToInf(r, x);
        }

        /// <summary>
        ///     NumPy 2.5's <c>_cond_svdvals(x)</c> (gh-32591): <c>svd(x, compute_uv=False)</c>'s singular values under
        ///     <c>errstate(all='ignore')</c>, so a matrix the svd gufunc cannot factorise — a non-finite one (umath_linalg's
        ///     <c>all_finite</c> check), or one <c>gesdd</c> does not converge on — reads NaN where np.linalg.svd raises.
        /// </summary>
        /// <param name="x">The matrix or stack (two axes or more: the caller ran <c>_assert_stacked_2d</c>, or
        ///     <c>_assert_stacked_square</c> for the inverse).</param>
        /// <returns>The singular values, descending, of shape <c>x.shape[:-2] + (min(M, N),)</c> in
        ///     <c>_realType(result_t)</c> (float32 for a float32 operand, float64 otherwise); NumPy's positive quiet NaN
        ///     throughout each matrix that failed (the gufunc's <c>nan_matrix</c> fill).</returns>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>), or a text operand (the svd
        ///     gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A finite matrix that no backend serves (Core has no managed
        ///     SVD).</exception>
        /// <remarks>
        ///     The common case is one <see cref="SvdKernel"/> call over the whole stack: every matrix finite and convergent.
        ///     Otherwise the stack is factorised matrix by matrix, as the gufunc's loop does, and each failed matrix's slot
        ///     is filled with NaN; every other matrix's values are the same bits either way (each is factorised on its own).
        ///     <para>The reshapes, the allocation and the per-matrix stores are the array object's members (np.reshape's
        ///     <c>reshape</c>, np.empty's allocation, np.copyto's <c>UnmanagedStorage.Copyto</c> with its default
        ///     <c>same_kind</c> rule, under which the singular values or the NaN into their own dtype are always
        ///     safe).</para>
        /// </remarks>
        [NDScopedCovered /* only caller: [NDScoped] cond; the cast, the per-matrix views, values and the NaN scalar ride its scope */]
        private NDArray CondSvdvals(NDArray x)
        {
            var common = CommonType(x);
            // NumPy's _umath_linalg.svd(x, signature=...): its input cast refuses text.
            ThrowIfTextGufunc("svd", x);
            var xc = ToCommon(x, common);
            if (Backends.Kernels.FiniteScan.IsAllFinite(xc))
            {
                try
                {
                    return SvdKernel(xc, fullMatrices: false, computeUv: false).S;
                }
                catch (LinAlgError e) when (e.Message == "SVD did not converge")
                {
                    // errstate(all='ignore') swallows only the gufunc's invalid flag, which a failed gesdd sets; the
                    // matrix-by-matrix pass below finds which matrices failed.
                }
            }

            // At least one matrix failed, so there is at least one element: M, N > 0 (an empty stack is all-finite, and
            // SvdKernel answers it through LinAlgEmpty without raising).
            int nd = xc.ndim;
            long m = xc.Shape.dimensions[nd - 2];
            long n = xc.Shape.dimensions[nd - 1];
            long k = Math.Min(m, n);
            long count = xc.size / (m * n);
            var matrices = xc.reshape(new Shape(new[] {count, m, n}, copy: false));
            // _realType(result_t): the singular values' dtype, float32 only for a float32 operand.
            var realT = common == NPTypeCode.Single ? NPTypeCode.Single : NPTypeCode.Double;
            // np.empty's own allocation: uninitialised, every row is written by the loop below.
            var values = xc.TensorEngine.NewArray(realT, new Shape(new[] {count, k}, copy: false), false);
            // NPY_NAN in the values' dtype (.NET's double.NaN / float.NaN are negative quiet NaNs).
            var nanValue = realT == NPTypeCode.Single
                ? NDArray.Scalar(BitConverter.Int32BitsToSingle(0x7FC00000))
                : NDArray.Scalar(BitConverter.Int64BitsToDouble(0x7FF8000000000000L));
            for (long i = 0; i < count; i++)
            {
                var slot = values[new[] {Slice.Index(i)}];
                var matrix = matrices[new[] {Slice.Index(i)}];
                NDArray s = null;
                if (Backends.Kernels.FiniteScan.IsAllFinite(matrix))
                {
                    try
                    {
                        s = SvdKernel(matrix, fullMatrices: false, computeUv: false).S;
                    }
                    catch (LinAlgError e) when (e.Message == "SVD did not converge")
                    {
                        // gesdd did not converge on this finite matrix: NaN, as the gufunc fills it.
                    }
                }
                UnmanagedStorage.Copyto(slot, s ?? nanValue, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
            }

            // Back to x.shape[:-2] + (min(M, N),).
            var dims = new long[nd - 1];
            for (int d = 0; d < nd - 2; d++)
                dims[d] = xc.Shape.dimensions[d];
            dims[nd - 2] = k;
            return values.reshape(new Shape(dims, copy: false));
        }

        /// <summary>
        ///     NumPy's <c>_umath_linalg.inv(x, signature=...)</c> under <c>errstate(all='ignore')</c>, as np.linalg.cond calls
        ///     it: every matrix is inverted on its own (the gufunc's loop), and one that the factorisation finds singular is
        ///     filled with NaN (umath_linalg's <c>nan_matrix</c>) where np.linalg.inv raises.
        /// </summary>
        /// <param name="x">The square matrices, float64 or complex128 (the <c>'d->d'</c> / <c>'D->D'</c> loop).</param>
        /// <returns>The inverses, of <paramref name="x"/>'s shape and dtype; NaN (NumPy's positive quiet NaN, both parts of a
        ///     complex element) throughout each singular matrix.</returns>
        /// <remarks>
        ///     Nothing is raised for a singular matrix, and no backend is needed: without one the managed LU inverts
        ///     float64 and complex128 matrices. The engine's inverse raises <c>LinAlgError("Singular matrix")</c> at the
        ///     first singular matrix, so a stack is inverted whole first (one call, the common case) and only after that
        ///     error matrix by matrix. Each matrix is factorised independently on both routes (the OpenBLAS <c>gesv</c> loop
        ///     and the managed LU), so every regular matrix's inverse is the same bits either way.
        ///     <para>The reshapes, the allocation and the per-matrix stores are the array object's members (np.reshape's
        ///     <c>reshape</c>, np.empty's allocation, np.copyto's <c>UnmanagedStorage.Copyto</c> with its default
        ///     <c>same_kind</c> rule, under which a float64 / complex128 inverse or NaN into its own dtype is always
        ///     safe).</para>
        /// </remarks>
        /// <exception cref="LinAlgError">Any factorisation error but <c>Singular matrix</c>, which this swallows.</exception>
        [NDScopedCovered /* only caller: [NDScoped] cond; the per-matrix views, inverses and the NaN scalar ride its scope */]
        private NDArray InvIgnoringErrors(NDArray x)
        {
            try
            {
                return InvKernel(x);
            }
            catch (LinAlgError e) when (e.Message == "Singular matrix")
            {
                // errstate(all='ignore') swallows only the gufunc's invalid flag, which a singular factorisation sets;
                // any other error still propagates.
            }

            int nd = x.ndim;
            long m = x.Shape.dimensions[nd - 1];
            // a singular matrix was found, so there is at least one element: m > 0 and the matrix count is size / m^2
            long count = x.size / (m * m);
            var matrices = x.reshape(new Shape(new[] {count, m, m}, copy: false));
            // np.empty(shape, x.dtype)'s own allocation: uninitialised, every matrix is written by the loop below.
            var inverses = x.TensorEngine.NewArray(x.dtype, new Shape(new[] {count, m, m}, copy: false), false);
            double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000L); // NPY_NAN; .NET's double.NaN is negative
            var nanValue = x.typecode == NPTypeCode.Complex
                ? NDArray.Scalar(new System.Numerics.Complex(nan, nan))
                : NDArray.Scalar(nan);
            for (long k = 0; k < count; k++)
            {
                var slot = inverses[new[] {Slice.Index(k)}];
                try
                {
                    UnmanagedStorage.Copyto(slot, InvKernel(matrices[new[] {Slice.Index(k)}]), NPY_CASTING.NPY_SAME_KIND_CASTING, null);
                }
                catch (LinAlgError e) when (e.Message == "Singular matrix")
                {
                    UnmanagedStorage.Copyto(slot, nanValue, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
                }
            }

            return inverses.reshape(new Shape(x.Shape.dimensions, copy: false));
        }

        /// <summary>Whether an order is the number <paramref name="value"/> (NumPy's <c>p in {2, -2}</c> and <c>p == -2</c>).</summary>
        /// <param name="p">The order argument.</param>
        /// <param name="value">The number to compare with.</param>
        /// <returns>True for a non-null, non-string order equal to <paramref name="value"/>.</returns>
        /// <exception cref="InvalidCastException">An order that does not convert to a double.</exception>
        /// <exception cref="FormatException">An order whose conversion to a double parses text and fails.</exception>
        private static bool IsOrder(object p, double value)
            => p is not null && p is not string && Convert.ToDouble(p) == value;

        /// <summary>Whether <paramref name="p"/> is a Python list for NumPy's purposes (an <c>object[]</c> or another
        ///     non-array <see cref="System.Collections.IList"/>): unhashable, unlike a tuple.</summary>
        /// <param name="p">The order argument.</param>
        /// <returns>True for a list.</returns>
        private static bool PolySequenceLike(object p)
            => p is System.Collections.IList && p is not System.Runtime.CompilerServices.ITuple;

        /// <summary>
        ///     NumPy's <c>_is_empty_2d</c> — <c>size == 0</c> AND the product of the last two
        ///     dimensions is 0 (so a zero-length BATCH like <c>(0,3,3)</c> is NOT empty here).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>True for an empty operand whose last two dimensions hold no element.</returns>
        private static bool IsEmpty2d(NDArray x)
        {
            if (x.size != 0)
                return false;

            int nd = x.ndim;
            long prod = 1;
            for (int i = Math.Max(0, nd - 2); i < nd; i++)
                prod *= x.Shape.dimensions[i];
            return prod == 0;
        }

        /// <summary>
        ///     NumPy's tail of <c>cond</c>: turn a NaN condition number into <c>inf</c> — UNLESS the
        ///     matrix itself held a NaN (over its last two axes), in which case the NaN is kept. A
        ///     NaN arises from a <c>0/0</c> (an all-zero matrix) or from a singular matrix's NaN
        ///     inverse, both infinitely ill-conditioned.
        /// </summary>
        /// <param name="r">The condition numbers.</param>
        /// <param name="x">The operand they came from.</param>
        /// <returns><paramref name="r"/> itself when it holds no NaN to replace; else, for one matrix, NumPy's scalar
        ///     <c>r.dtype.type(inf)</c> (a read-only 0-d array), and for a stack a fresh array with the NaNs replaced.</returns>
        /// <remarks>The predicates, the reductions and the selection are the members np.isnan, np.any, np.logical_not,
        ///     np.logical_and and np.where dispatch to, over the plain condition numbers and the cond facade's plain
        ///     operand.</remarks>
        [NDScopedCovered /* only caller: [NDScoped] cond (the isnan/logical/inf-scalar temps are scope-reclaimed) */]
        private static NDArray CondNanToInf(NDArray r, NDArray x)
        {
            var nanMask = r.TensorEngine.IsNan(r);
            if (!nanMask.TensorEngine.Any(nanMask))
                return r;

            NDArray apply = nanMask;
            if (x.typecode is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half or NPTypeCode.Complex)
            {
                // `x_nan = isnan(x).any(axis=(-2, -1))`, then `nan_mask & ~x_nan`.
                var xIsNan = x.TensorEngine.IsNan(x);
                var xNan = xIsNan.TensorEngine.Any(xIsNan, new[] {-2, -1}, null, false, null);
                var notXNan = xNan.TensorEngine.LogicalNot(xNan);
                apply = nanMask.TensorEngine.LogicalAnd(nanMask, notXNan);
            }

            // NumPy's `elif nan_mask: r = r.dtype.type(inf)` ("Convention is to return scalars instead of 0d arrays"): a
            // new scalar of r's dtype, or r itself when the matrix held the NaN.
            if (r.ndim == 0)
                return apply.GetBoolean()
                    ? NDArray.Scalar(double.PositiveInfinity).astype(r.typecode).MarkReductionScalar()
                    : r;

            return NDArray.where(apply, NDArray.Scalar(double.PositiveInfinity).astype(r.typecode), r);
        }
    }
}
