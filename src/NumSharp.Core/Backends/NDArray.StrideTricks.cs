using NumSharp.Backends;

namespace NumSharp
{
    // ============================== lib/_stride_tricks_impl.py ==============================
    // NumPy's broadcast_to and broadcast_arrays as members of the array object (P6's structural part): broadcast_to,
    // the read-only view np.broadcast_to returns, over Shape.BroadcastTo's check and layout
    // (View/Shape.Broadcasting.cs), and broadcast_arrays, the writeable views np.broadcast_arrays returns for a list of
    // arrays, over Shape.Broadcast. The facades keep their overloads and their numpy.matrix regions and call them; the
    // engine and the array object's storage partials call them where they broadcast through np.broadcast_to /
    // np.broadcast_arrays. np.broadcast_shapes' engine callers call Shape.BroadcastShape, its body.
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>numpy.broadcast_to(array, shape)</c> after the facade: a READ-ONLY view of
        ///     <paramref name="from"/>'s buffer at <paramref name="against"/>'s dimensions, the stretched axes and the
        ///     prepended ones read with stride 0.
        /// </summary>
        /// <param name="from">The source storage (not null); the view shares its buffer, and its counted reference keeps the
        ///     buffer alive after the source array is disposed or collected.</param>
        /// <param name="against">The target; only its dimensions are read.</param>
        /// <returns>A new array over <paramref name="from"/>'s buffer, never writeable: several of its elements may be one
        ///     memory location, so a caller that writes must copy first (<c>.copy()</c>). It is a new object even when the
        ///     dimensions already agree, read-only all the same.</returns>
        /// <exception cref="ValueError">A dimension of <paramref name="against"/> is negative (NumPy's <c>all elements of
        ///     broadcast shape must be non-negative</c>), checked first.</exception>
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the target one way
        ///     (<see cref="Shape.BroadcastTo"/>), checked before the view is built.</exception>
        /// <remarks>No numpy.matrix rule runs here: the result is a plain array whatever <paramref name="from"/>'s class,
        ///     which is NumPy's <c>subok=False</c> answer. The facades open their matrix region around this call, an engine
        ///     caller needs none (no public function runs inside).</remarks>
        internal static NDArray broadcast_to(UnmanagedStorage from, Shape against) =>
            new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from, Shape.BroadcastTo(from.Shape, against).WithFlags(flagsToClear: ArrayFlags.WRITEABLE)));

        /// <summary>
        ///     <see cref="broadcast_to(UnmanagedStorage, Shape)"/> over an array: <paramref name="from"/>'s storage viewed
        ///     read-only at <paramref name="against"/>'s dimensions.
        /// </summary>
        /// <param name="from">The source array (not null). Disposing it later leaves the view valid: the view holds its own
        ///     reference on the buffer.</param>
        /// <param name="against">The target; only its dimensions are read.</param>
        /// <returns>A new read-only view; see <see cref="broadcast_to(UnmanagedStorage, Shape)"/>.</returns>
        /// <exception cref="ValueError">A dimension of <paramref name="against"/> is negative.</exception>
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the target one way.</exception>
        internal static NDArray broadcast_to(NDArray from, Shape against) => broadcast_to(from.Storage, against);

        /// <summary>
        ///     NumPy 2.4.2's <c>numpy.broadcast_arrays(*args)</c> for a list of arrays, after the facade: every array at the
        ///     shape they broadcast to together — an array already of that shape is returned AS ITSELF, every other one is
        ///     a view whose stretched and prepended axes read with stride 0, writeable exactly when its array is.
        /// </summary>
        /// <param name="ndArrays">The arrays (not null, no element null).</param>
        /// <returns>One array per operand, in order: the operand itself when its shape is the broadcast shape (NumPy's
        ///     <c>array if array.shape == shape</c>), else a new view over its buffer (keeping it alive). Footgun: several
        ///     elements of a stretched view are one memory location, so a write through one element writes every element
        ///     that aliases it (NumPy warns on such a write); copy first to write independently. Do not dispose a result
        ///     without checking it is not the caller's own operand.</returns>
        /// <exception cref="IncorrectShapeException">The arrays do not broadcast together, checked before any view is built,
        ///     with NumPy's text naming the first pair that disagrees (<c>shape mismatch: objects cannot be broadcast to a
        ///     single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).</c>,
        ///     <see cref="Shape.BroadcastShape"/>; parity QA-30).</exception>
        /// <exception cref="ValueError">The broadcast shape's element count does not fit a 64-bit integer
        ///     (<c>broadcast dimensions too large.</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy builds each stretched view with <c>_broadcast_to(array, shape, readonly=False)</c>, which sets
        ///     <c>writeable = True</c> only <c>if array.flags._writeable_no_warn</c>: the view of a read-only operand (one
        ///     whose WRITEABLE flag was cleared, a <c>np.broadcast_to</c> view) stays read-only, so a write through it raises
        ///     NumPy's <c>ValueError: assignment destination is read-only</c>. Every view used to be forced writeable, and
        ///     such a write landed in the memory its owner had made read-only (parity QA-34).
        ///     </para>
        ///     <para>
        ///     No numpy.matrix rule runs here: a matrix or chararray operand is first viewed as a plain array, as NumPy's
        ///     <c>np.array(_m, copy=None, subok=False)</c> does, so the results are plain arrays whatever the operands'
        ///     classes. The facade opens its matrix region around this call; an engine or storage caller needs none (no
        ///     public function runs inside).
        ///     </para>
        /// </remarks>
        internal static NDArray[] broadcast_arrays(NDArray[] ndArrays)
        {
            int len = ndArrays.Length;
            int i;
            var inputShapes = new Shape[len];
            for (i = 0; i < len; i++)
                inputShapes[i] = ndArrays[i].Shape;
            // `shape = _broadcast_shape(*args)` is NumPy's first statement, and its refusal is the text a mismatch reports
            // (np.broadcast's PyArray_Broadcast, naming the pair that disagrees; parity QA-30). The per-operand layouts still
            // come from Shape.Broadcast, which cannot refuse after this check and keeps every view's strides as they were.
            Shape.BroadcastShape(inputShapes);
            var outputShapes = Shape.Broadcast(inputShapes);

            var list = new NDArray[len];
            for (i = 0; i < len; i++)
                list[i] = BroadcastArraysMember(ndArrays[i], outputShapes[i]);

            return list;
        }

        /// <summary>
        ///     One result of NumPy 2.4.2's <c>broadcast_arrays</c>: <c>array if array.shape == shape else
        ///     _broadcast_to(array, shape, subok=False, readonly=False)</c>.
        /// </summary>
        /// <param name="operand">The operand as the caller passed it (not null).</param>
        /// <param name="broadcast">The operand's layout at the broadcast shape (<see cref="Shape.Broadcast(Shape[])"/>'s entry
        ///     for it: the operand's strides with 0 on every stretched or prepended axis).</param>
        /// <returns>The operand itself (or its plain view, for a subclass operand) when its dimensions already are the
        ///     broadcast shape's; else a new view over its buffer, WRITEABLE exactly when the operand is.</returns>
        /// <remarks>
        ///     <c>_writeable_no_warn</c> is the operand's own WRITEABLE flag (<c>flags.writeable</c>, the shape's flag here):
        ///     a stretched view of a writeable array stays writeable, as NumPy 2.4.2 returns it (with a FutureWarning on the
        ///     first write that NumSharp does not model), while <see cref="Shape.Broadcast(Shape[])"/> marks every stretched
        ///     layout read-only by itself, so the flag is set or cleared explicitly either way.
        /// </remarks>
        internal static NDArray BroadcastArraysMember(NDArray operand, Shape broadcast)
        {
            // `args = [np.array(_m, copy=None, subok=False) for _m in args]`: a plain array is itself, a numpy.matrix or a
            // numpy.char.chararray becomes a new plain view of its memory (owned by the caller like any view).
            var plain = NDCharArrayBoundary.PlainViewOf(NDMatrixBoundary.PlainOf(operand));
            // `array if array.shape == shape`: the operand itself, identity included (`r[0] is a` in NumPy).
            if (SameDims(plain.Shape.dimensions, broadcast.dimensions))
                return plain;
            // `_broadcast_to(..., readonly=False)`: writeable only `if array.flags._writeable_no_warn`.
            var layout = plain.Shape.IsWriteable
                ? broadcast.WithFlags(flagsToSet: ArrayFlags.WRITEABLE)
                : broadcast.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(plain.Storage, layout));
        }
    }
}
