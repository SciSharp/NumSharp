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
    }
}
