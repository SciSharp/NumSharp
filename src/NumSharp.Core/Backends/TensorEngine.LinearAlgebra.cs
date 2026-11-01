using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     The matrix products beyond <c>dot</c> / <c>matmul</c> - <c>inner</c>, <c>vdot</c>, <c>vecdot</c>, <c>matvec</c>,
    ///     <c>vecmat</c> - NumPy's C core: each asks <see cref="Blas"/> first and falls back to NumSharp's managed kernels, so
    ///     a backend changes which implementation runs and never whether an answer exists.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every member reads <see cref="Blas"/> into a LOCAL before using it. That is not style -
    ///     the property is settable and a concurrent <c>OpenBlasEngine.Disable()</c> turns a test-then-call
    ///     into a null dereference (measured at ~2 % of calls; see <c>docs/stale-docs/GEMM_PARITY.md</c> section 9).
    ///     </para>
    ///     <para>
    ///     numpy.linalg's factorisations are not here: they are the <see cref="Linalg"/> sub-engine's
    ///     (<see cref="LinalgEngine"/>), whose default, <see cref="DefaultLinalgEngine"/>, runs NumPy's
    ///     <c>_linalg.py</c> statements over ten kernels that ask the same <see cref="Blas"/> property. The ten
    ///     <c>_umath_linalg</c> members this class shipped in 0.70.0 (Cholesky, Det, Slogdet, Eig, Eigh, Inv, Solve,
    ///     Lstsq, Qr, Svd) moved there in 0.71.0 (the np-layer dtype plan's phase PE, E2) as
    ///     <c>DefaultLinalgEngine.CholeskyKernel</c> ... <c>SvdKernel</c>.
    ///     </para>
    /// </remarks>
    public abstract partial class TensorEngine
    {
        #region Products — managed fallback, backend optional

        /// <summary>
        ///     <c>np.inner</c> — a sum product over the LAST axis of both operands.
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>PyArray_InnerProduct</c> swaps the last two axes of <paramref name="b"/>
        ///     (when <c>a.ndim >= 1</c> and <c>b.ndim >= 2</c>) and hands the pair to the very same
        ///     <c>PyArray_MatrixProduct2</c> that backs <c>np.dot</c> — which is why an unaligned
        ///     <c>inner</c> reports <paramref name="b"/>'s shape ALREADY TRANSPOSED.
        ///     <para><see cref="NDAnchoredAttribute"/>: the managed product reads both operands' raw addresses in
        ///     <c>Dot</c>, an engine method that opens no scope, so the weaver roots <paramref name="a"/> and
        ///     <paramref name="b"/> here until the product returns — a caller's temporary operand
        ///     (<c>np.inner(a, np.arange(n))</c>) cannot be collected under the kernel. An override does not inherit
        ///     the mark.</para>
        /// </remarks>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand; its last two axes are swapped (a view) before the product.</param>
        /// <returns>The inner product, shape <c>a.shape[:-1] + b.shape[:-1]</c>.</returns>
        /// <exception cref="IncorrectShapeException">The last axes do not align (NumPy's "shapes ... not aligned" text,
        ///     with <paramref name="b"/>'s shape transposed).</exception>
        /// <exception cref="ArgumentException">The same misalignment on the vector-times-matrix path of <c>Dot</c>, which
        ///     raises the house type of that path with the same NumPy text.</exception>
        /// <exception cref="TypeError">A fixed-width string operand beside a 0-d one: NumPy's no-loop refusal of the
        ///     <c>multiply</c> it computes.</exception>
        /// <exception cref="ValueError">A fixed-width string operand otherwise: "data type must provide an itemsize" (the
        ///     string product has no item size to allocate).</exception>
        [NDAnchored]
        public virtual NDArray Inner(NDArray a, NDArray b)
        {
            // inner of a 0-d operand is a multiply (its string loops answer); otherwise NumPy's dot checks run on the
            // TRANSPOSED b it builds first (so an alignment error names b's shape transposed), and the result allocation
            // needs a fixed item size. Before the BLAS hook, which never takes text.
            var right = (a.ndim >= 1 && b.ndim >= 2) ? SwapAxes(b, -1, -2) : b;
            Backends.Strings.NDStringUfuncs.ThrowIfProduct(a, right);
            var blas = Blas;
            if (blas != null && blas.TryInner(a, b, out var accelerated))
                return accelerated;

            return Dot(a, right);
        }

        /// <summary>
        ///     <c>np.vdot</c> — both operands flattened to 1-D, <paramref name="a"/> conjugated when
        ///     complex, then a vector dot product. Always 0-d.
        /// </summary>
        [NDScoped] // engine boundary (also called directly, not only via np.vdot): the ravel/reshape/conjugate temps are reclaimed
        public virtual NDArray Vdot(NDArray a, NDArray b)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b);   // an engine method computes plain: np.ravel / np.reshape below must not apply its matrix rule
            // A StringDType operand: NumPy's legacy descriptor lookup refuses it, or falls back to an OBJECT product that
            // NumSharp cannot hold; VStringVdotAnswer gives NumPy's answer in NumPy's order.
            if (Backends.Strings.NDStringUfuncs.IsVString(a) || Backends.Strings.NDStringUfuncs.IsVString(b))
                throw Backends.Strings.NDStringUfuncs.VStringVdotAnswer(a, b);
            // vdot flattens both operands first (one reused dims buffer: a length mismatch is b's reshape error), so even
            // a 0-d string then fails NumPy's result allocation for want of an item size.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(b))
            {
                long n = a.size;
                if (b.size != n)
                    b.reshape(new long[] {n});
                throw Backends.Strings.NDStringUfuncs.NoItemsize();
            }
            var blas = Blas;
            if (blas != null && blas.TryVdot(a, b, out var accelerated))
                return accelerated;

            // NumPy's array_vdot reshapes BOTH operands with one reused PyArray_Dims buffer whose
            // single entry starts at -1. _fix_unknown_dimension writes the resolved length back into
            // that buffer, so the second reshape no longer asks for (-1,) but for (a.size,) — which
            // is why a length mismatch surfaces as a reshape error and never as the
            // "vectors have different lengths" the C code below it still carries.
            var left = NDArray.ravel(a, 'C');
            var right = b.reshape(new long[] {left.size});
            return Dot(LinAlgHelper.Conjugate(left), right);
        }

        /// <summary>
        ///     The <c>np.vecdot</c> gufunc <c>(n),(n)->()</c> — <paramref name="x1"/> conjugated when
        ///     complex. Operands arrive validated; leading axes broadcast.
        /// </summary>
        [NDScoped] // engine boundary: the conjugate + elementwise product feeding the reduction are reclaimed
        public virtual NDArray Vecdot(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryVecdot(x1, x2, out var accelerated))
                return accelerated;

            // NumPy's float16 loop is HALF_dot (arraytypes.c.src, no BLAS): each pair widened to float32 and multiplied
            // there, the products summed in float32 in order, the sum narrowed to float16 once. The elementwise product
            // below would round every product to float16 first (probed 2.4.2: np.vecdot of random float16 rows differs
            // from it for 3 terms already). The matmul kernel's float16 path is that same sum (HALF matmul_inner_noblas
            // narrows once too), so each output is a (…,1,n) row times an (…,n,1) column.
            if (x1.typecode == NPTypeCode.Half && x2.typecode == NPTypeCode.Half)
            {
                var halfProduct = Matmul(LinAlgHelper.UnitAxisView(x1, x1.ndim - 1), LinAlgHelper.UnitAxisView(x2, x2.ndim));
                halfProduct = LinAlgHelper.DropAxis(halfProduct, halfProduct.ndim - 1);
                return LinAlgHelper.DropAxis(halfProduct, halfProduct.ndim - 1);
            }

            // The reduction dtype is the LOOP dtype, not NEP50's accumulator: np.vecdot's registered
            // loops are 'ii->i', so an int32 pair reduces to int32 where np.sum would give int64.
            var product = Multiply(LinAlgHelper.Conjugate(x1), x2);
            return ReduceAdd(product, -1, false, product.typecode.AsType());
        }

        /// <summary>
        ///     The <c>np.matvec</c> gufunc <c>(m,n),(n)->(m)</c> — NumPy's <c>gemv</c> route.
        /// </summary>
        [NDScoped] // engine boundary: the column-promoted operand and the pre-DropAxis product are reclaimed
        public virtual NDArray Matvec(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryMatvec(x1, x2, out var accelerated))
                return accelerated;

            // (…,m,n) @ (…,n,1) -> (…,m,1), then drop the trailing length-1 axis. Promoting x2 to a
            // column first is what keeps matmul out of its own 1-D special case.
            var product = Matmul(x1, NDArray.expand_dims(x2, -1));
            return LinAlgHelper.DropAxis(product, product.ndim - 1);
        }

        /// <summary>
        ///     The <c>np.vecmat</c> gufunc <c>(n),(n,m)->(m)</c> — <paramref name="x1"/> conjugated
        ///     when complex.
        /// </summary>
        [NDScoped] // engine boundary: the conjugate/expand_dims operand and the pre-DropAxis product are reclaimed
        public virtual NDArray Vecmat(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryVecmat(x1, x2, out var accelerated))
                return accelerated;

            // (…,1,n) @ (…,n,m) -> (…,1,m), then drop the length-1 axis at -2.
            var product = Matmul(NDArray.expand_dims(LinAlgHelper.Conjugate(x1), -2), x2);
            return LinAlgHelper.DropAxis(product, product.ndim - 2);
        }

        #endregion

        // np.einsum's contraction does NOT live here — it is a pure composition over the matrix
        // products (Matmul/Multiply) with no engine state and no seam, so it sits in the np layer next
        // to np.tensordot and np.linalg.multi_dot (LinearAlgebra/np.einsum.Contract.cs). It inherits
        // Blas routing the same indirect way they do, through np.matmul, exactly as NumPy's
        // einsumfunc.py composes over c_einsum + matmul rather than adding a low-level kernel.
    }
}

namespace NumSharp.Backends
{
    /// <summary>
    ///     Shape and dtype plumbing shared by the linear-algebra entry points.
    /// </summary>
    internal static class LinAlgHelper
    {
        /// <summary>
        ///     The complex conjugate, and the identity at every other dtype — which is exactly
        ///     NumPy's rule for <c>vdot</c>/<c>vecdot</c>/<c>vecmat</c>, whose "conjugate the first
        ///     operand" step is a no-op for real loops.
        /// </summary>
        /// <remarks>
        ///     <para>Kept internal: <c>np.conj</c>/<c>np.conjugate</c> is a NumPy function in its own right
        ///     and adding it belongs to its own pass, not to this one.</para>
        ///     <para>The conjugates go into a fresh C-ordered array: the products this feeds see the layout the former
        ///     <c>a.copy()</c> gave them (a product's reduction order follows its operands' layout, so the layout is part of
        ///     the answer). One pass, through the engine's own conjugate kernel: a C-contiguous operand is read linearly by
        ///     the unary kernel itself (no ufunc argument handling); any other layout goes through the engine member, which
        ///     routes it through the iterator (the unary kernel's own strided walk is the engine's slow fallback, about four
        ///     times slower on an F-ordered operand).</para>
        /// </remarks>
        /// <param name="a">The operand.</param>
        /// <returns><paramref name="a"/> itself for a dtype other than complex128; otherwise a fresh C-ordered array of
        ///     the conjugates.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled (the engine's conjugate kernel, as for
        ///     np.conjugate).</exception>
        internal static NDArray Conjugate(NDArray a)
        {
            if (a.typecode != NPTypeCode.Complex)
                return a;

            var result = new NDArray(NPTypeCode.Complex, a.Shape.Clean(), false);
            try
            {
                if (!a.Shape.IsContiguous)
                    return a.TensorEngine.Conjugate(a, null, result);
                var kernel = DirectILKernelGenerator.GetUnaryKernel(
                    new UnaryKernelKey(NPTypeCode.Complex, NPTypeCode.Complex, UnaryOp.Conjugate, IsContiguous: true));
                DefaultEngine.ExecuteUnaryKernel(kernel, a, result);
                return result;
            }
            catch
            {
                // The result never reaches a caller: its buffer goes back now, not at finalization.
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Drops a length-1 axis from a freshly computed product — the step that
        ///     turns matmul's <c>(…,m,1)</c> back into <c>matvec</c>'s <c>(…,m)</c> and its
        ///     <c>(…,1,m)</c> back into <c>vecmat</c>'s <c>(…,m)</c>.
        /// </summary>
        /// <param name="a">The product: consumed (it is returned relabelled, or disposed after a copy).</param>
        /// <param name="axis">The length-1 axis to drop.</param>
        /// <returns>The product without the axis, owning its buffer and keeping its other strides: relabelled in place
        ///     (<see cref="NDArray.AdoptDroppedAxis"/>), as NumPy's gufunc output owns its data (probed 2.4.2:
        ///     <c>np.matvec(a, v).flags.owndata</c> is True); a product that is not a fresh owner of its elements is
        ///     copied in K order first.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not a length-1 axis of
        ///     <paramref name="a"/>.</exception>
        internal static NDArray DropAxis(NDArray a, int axis) => NDArray.AdoptDroppedAxis(a, axis);

        /// <summary>
        ///     A view of <paramref name="x"/> with a length-1 axis inserted at <paramref name="axis"/> — what
        ///     <c>np.expand_dims</c> returns, built here without the public function (an engine kernel calls no np.*
        ///     layer). The view reads the same memory; the new axis's stride is 0, which an extent-1 axis never uses.
        /// </summary>
        /// <param name="x">The array.</param>
        /// <param name="axis">The position of the new axis: 0 to <c>x.ndim</c>.</param>
        /// <returns>A view of <c>x.ndim + 1</c> dimensions (the caller's scope reclaims it).</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is outside 0 to <c>x.ndim</c> (a
        ///     caller's bug).</exception>
        internal static NDArray UnitAxisView(NDArray x, int axis)
        {
            if ((uint)axis > (uint)x.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "the new axis must lie between 0 and ndim");

            var src = x.Shape;
            var dims = new long[src.NDim + 1];
            var strides = new long[src.NDim + 1];
            for (int i = 0, j = 0; i < dims.Length; i++)
            {
                if (i == axis)
                {
                    dims[i] = 1;
                    continue;   // stride 0
                }

                dims[i] = src.dimensions[j];
                strides[i] = src.strides[j++];
            }

            return new NDArray(x.Storage.Alias(new Shape(dims, strides, src.offset, src.bufferSize)))
            {
                TensorEngine = x.TensorEngine
            };
        }
    }
}
