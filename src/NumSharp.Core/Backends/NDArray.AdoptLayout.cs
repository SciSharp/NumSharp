using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Turns a kernel's C-ordered result, computed over operands transposed into <paramref name="layout"/>'s
        ///     memory order, into an array laid out as <paramref name="layout"/>: relabels its storage in place
        ///     (<see cref="UnmanagedStorage.SetShapeUnsafe(Shape)"/>, the ndarray.shape setter's mechanism). No
        ///     element moves, and the array keeps owning its buffer, as a NumPy ufunc result does. Use it wherever
        ///     NumSharp reproduces NumPy's ufunc OUTPUT layout (<see cref="Shape.NpyIterOutputShape"/>) with a kernel
        ///     that can only write C order: run the kernel over <c>np.transpose(operand, layout.DenseAxisOrder())</c>,
        ///     then adopt the layout here.
        /// </summary>
        /// <param name="r">
        ///     The kernel's result over the transposed operands, so its dims are <paramref name="layout"/>'s in
        ///     <see cref="Shape.DenseAxisOrder"/> order. Consumed: it is returned relabelled, or disposed after its
        ///     values are copied out.
        /// </param>
        /// <param name="layout">
        ///     The target layout: dense (no stride-0 axis of extent &gt; 1) and non-empty, as
        ///     <see cref="Shape.NpyIterOutputShape"/> builds it.
        /// </param>
        /// <returns>
        ///     The values of <paramref name="r"/> in <paramref name="layout"/>: <paramref name="r"/> itself (relabelled)
        ///     on the normal path, a fresh array otherwise.
        /// </returns>
        /// <remarks>
        ///     <para><b>Why the relabel is correct.</b> A C-contiguous buffer of the transposed dims stores element
        ///     <c>a[j0, …, jn]</c> at <c>Σ j_k·layout.strides[k]</c>: the transposed dims put the axes in the layout's
        ///     memory order, so each axis's C stride over them is the product of the extents inside it, which is the
        ///     layout's stride. An extent-1 axis is only ever read at index 0, so its stride (which the two shapes may
        ///     disagree on) never addresses anything.</para>
        ///     <para><b>Why not write the layout directly.</b> NDIter walks a pair of operands that share a permuted
        ///     layout in LOGICAL order and gathers every element from both, which measured 2.7× slower than the
        ///     transposed C pass (32³ float64 transposed (2,0,1): 40.6 µs vs 14.8 µs). The transposed views of dense
        ///     permuted operands are C-ordered, so the kernel runs at memory speed.</para>
        ///     <para><b>The contract is checked, not trusted.</b> The kernels this serves allocate a fresh
        ///     C-contiguous, offset-0 result. Should one ever return anything else (an operator fallback that keeps
        ///     an input's F order, a view), the values are copied through the C-contiguous transposed view of a fresh
        ///     <paramref name="layout"/> array instead: a copy between two same-ordered views, never NDIter's
        ///     permuted walk.</para>
        /// </remarks>
        internal static NDArray AdoptLayout(NDArray r, Shape layout)
        {
            if (r.Shape.IsContiguous && r.Shape.offset == 0 && r.Storage.Count == layout.size && r.size == layout.size)
            {
                r.Storage.SetShapeUnsafe(layout);
                return r;
            }

            NDArray o = null, ov = null;
            try
            {
                o = new NDArray(r.typecode, layout, false);
                ov = np.transpose(o, layout.DenseAxisOrder());   // C-contiguous over r's (transposed) dims
                np.copyto(ov, r);
                var done = o;
                o = null;
                return done;
            }
            finally
            {
                ov?.Dispose();
                o?.Dispose();   // non-null only when the copy threw
                r.Dispose();
            }
        }

        /// <summary>
        ///     Inserts a length-1 axis into a freshly computed result by relabelling its storage in place: the other axes
        ///     keep their strides, so a K-ordered result stays K-ordered, and the result keeps owning its buffer, as
        ///     NumPy's allocated gufunc output does (probed 2.4.2: <c>np.vecdot(a, b, keepdims=True).flags.owndata</c> is
        ///     True). The new axis takes the stride of one element, because NumPy's iterator allocates a kept core axis
        ///     innermost: <c>np.vecdot(m, m, axis=0, keepdims=True)</c> on a (3,4) <c>m</c> has strides (8,8), where C
        ///     order would give (32,8). An extent-1 axis never addresses an element, so only <c>.strides</c> shows the
        ///     choice.
        /// </summary>
        /// <param name="r">
        ///     The result, normally a fresh owner (a kernel's output, in any dense layout). Consumed: it is returned
        ///     relabelled; when it is not such an owner, it is first copied in K order (see <see cref="FreshOwner"/>) and
        ///     disposed.
        /// </param>
        /// <param name="axis">The position of the new axis in the result: 0 to <c>r.ndim</c>.</param>
        /// <returns>The result with one more axis, owning its buffer.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is outside 0 to <c>r.ndim</c> (a caller's
        ///     bug: the gufunc checks validated the axis before the product ran).</exception>
        internal static NDArray AdoptInsertedAxis(NDArray r, int axis)
        {
            if ((uint)axis > (uint)r.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "the new axis must lie between 0 and ndim");

            r = FreshOwner(r);
            long[] dims = r.Shape.dimensions, strides = r.Shape.strides;
            var newDims = new long[dims.Length + 1];
            var newStrides = new long[dims.Length + 1];
            for (int i = 0, j = 0; i < newDims.Length; i++)
            {
                if (i == axis)
                {
                    newDims[i] = 1;
                    newStrides[i] = 1;   // one element: NumPy's innermost core axis
                }
                else
                {
                    newDims[i] = dims[j];
                    newStrides[i] = strides[j++];
                }
            }

            r.Storage.SetShapeUnsafe(Shape.Adopt(newDims, newStrides));
            return r;
        }

        /// <summary>
        ///     Drops a length-1 axis from a freshly computed result by relabelling its storage in place: the other axes
        ///     keep their strides, and the result keeps owning its buffer, as NumPy's gufunc output does (probed 2.4.2:
        ///     <c>np.matvec(a, v).flags.owndata</c> is True, where a reshape of the <c>(…,m,1)</c> product would be a
        ///     view). Dropping the last axis of a <c>(1,)</c> result gives the 0-d scalar shape.
        /// </summary>
        /// <param name="r">
        ///     The result, normally a fresh owner (a kernel's output, in any dense layout). Consumed: it is returned
        ///     relabelled; when it is not such an owner, it is first copied in K order (see <see cref="FreshOwner"/>) and
        ///     disposed.
        /// </param>
        /// <param name="axis">The length-1 axis to drop: 0 to <c>r.ndim - 1</c>.</param>
        /// <returns>The result with one axis fewer, owning its buffer.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not a length-1 axis of
        ///     <paramref name="r"/> (a caller's bug: dropping any other axis would lose elements).</exception>
        internal static NDArray AdoptDroppedAxis(NDArray r, int axis)
        {
            if ((uint)axis >= (uint)r.ndim || r.Shape.dimensions[axis] != 1)
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "the dropped axis must be a length-1 axis");

            r = FreshOwner(r);
            long[] dims = r.Shape.dimensions, strides = r.Shape.strides;
            var newDims = new long[dims.Length - 1];
            var newStrides = new long[dims.Length - 1];
            for (int i = 0, j = 0; i < dims.Length; i++)
            {
                if (i == axis)
                    continue;
                newDims[j] = dims[i];
                newStrides[j++] = strides[i];
            }

            r.Storage.SetShapeUnsafe(Shape.Adopt(newDims, newStrides));
            return r;
        }

        /// <summary>
        ///     Permutes a freshly computed result's axes by relabelling its storage in place — the dims and strides of
        ///     <c>np.transpose(r, permutation)</c> become <paramref name="r"/>'s own — so the result keeps owning its
        ///     buffer, as NumPy's gufunc output does when <c>axes=</c> remaps its core axes (probed 2.4.2:
        ///     <c>np.matmul(ones((2,3,4)), ones((2,4,5)), axes=[(1,2),(1,2),(0,1)])</c> has shape (3,5,2), strides
        ///     (40,8,120), OWNDATA True). Unlike <see cref="AdoptLayout"/> it accepts any dense layout of
        ///     <paramref name="r"/>, so it can follow a relabel that already gave the result NumPy's K order.
        /// </summary>
        /// <param name="r">
        ///     The result, normally a fresh owner (a kernel's output, in any dense layout). Consumed: it is returned
        ///     relabelled; when it is not such an owner, it is first copied in K order (see <see cref="FreshOwner"/>) and
        ///     disposed, and the permutation applies to the copy.
        /// </param>
        /// <param name="permutation">
        ///     NumPy's <c>np.transpose</c> axes: axis <c>i</c> of the answer is axis <c>permutation[i]</c> of
        ///     <paramref name="r"/>. Each axis once.
        /// </param>
        /// <returns>The result with its axes permuted, owning its buffer.</returns>
        /// <exception cref="ArgumentException"><paramref name="permutation"/> is not a permutation of <paramref name="r"/>'s
        ///     axes (a caller's bug: the gufunc checks validated the axes).</exception>
        internal static NDArray AdoptAxisPermutation(NDArray r, int[] permutation)
        {
            int nd = r.ndim;
            if (permutation.Length != nd)
                throw new ArgumentException($"a permutation of {nd} axes was expected, got {permutation.Length}",
                    nameof(permutation));
            var seen = new bool[nd];
            foreach (int ax in permutation)
            {
                if ((uint)ax >= (uint)nd || seen[ax])
                    throw new ArgumentException("each axis must appear exactly once", nameof(permutation));
                seen[ax] = true;
            }

            r = FreshOwner(r);
            r.Storage.SetShapeUnsafe(DefaultEngine.TransposedShape(r.Shape, permutation));
            return r;
        }

        /// <summary>
        ///     The precondition of the unit-axis and permutation relabels: an array that owns a buffer of exactly its
        ///     elements, at offset 0, not broadcast. Every kernel's fresh output is one, and then its strides describe a
        ///     dense layout, so a relabel that only inserts or drops a length-1 axis, or permutes the axes, is a valid
        ///     shape over the same bytes.
        /// </summary>
        /// <param name="r">The result to relabel. Disposed when it has to be copied.</param>
        /// <returns><paramref name="r"/> itself when it is such an owner; otherwise its copy in K order (no current
        ///     caller passes one), which keeps the dense layout NumPy would allocate and owns its buffer, so the relabel
        ///     never changes the shape of memory that another array reads.</returns>
        private static NDArray FreshOwner(NDArray r)
        {
            if (r.Storage.OwnsData && r.Shape.offset == 0 && r.Storage.Count == r.size && !r.Shape.IsBroadcasted)
                return r;

            var fresh = r.copy('K');
            r.Dispose();
            return fresh;
        }
    }
}
