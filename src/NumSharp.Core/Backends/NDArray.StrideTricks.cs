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
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the target one way.</exception>
        internal static NDArray broadcast_to(NDArray from, Shape against) => broadcast_to(from.Storage, against);

        /// <summary>
        ///     NumPy's <c>numpy.broadcast_arrays(*args)</c> for a list of arrays, after the facade: every array viewed at
        ///     the shape they broadcast to together, the stretched and the prepended axes read with stride 0. The views are
        ///     WRITEABLE, as NumPy 2.4.2 still returns them (with a FutureWarning that a later version will not), unlike
        ///     <see cref="broadcast_to(NDArray, Shape)"/>'s.
        /// </summary>
        /// <param name="ndArrays">The arrays (not null, no element null).</param>
        /// <returns>One new view per array, in order, each over its array's buffer (and keeping it alive). Footgun:
        ///     several elements of a stretched view are one memory location, so a write through one element writes every
        ///     element that aliases it; copy first to write independently.</returns>
        /// <exception cref="IncorrectShapeException">The arrays do not broadcast together
        ///     (<see cref="Shape.Broadcast(Shape[])"/>), checked before any view is built.</exception>
        /// <remarks>No numpy.matrix rule runs here: the result is a list of plain arrays whatever the operands' classes.
        ///     The facade opens its matrix region around this call; an engine or storage caller needs none (no public
        ///     function runs inside).</remarks>
        internal static NDArray[] broadcast_arrays(NDArray[] ndArrays)
        {
            int len = ndArrays.Length;
            int i;
            var inputShapes = new Shape[len];
            for (i = 0; i < len; i++)
                inputShapes[i] = ndArrays[i].Shape;
            var outputShapes = Shape.Broadcast(inputShapes);

            var list = new NDArray[len];
            for (i = 0; i < len; i++)
                // NumPy 2.4.2 parity: broadcast_arrays results are WRITEABLE (unlike broadcast_to),
                // even for stretched (stride-0) dimensions. NumPy emits a FutureWarning that a future
                // version will change this, but 2.4.2 still returns writeable=True. Force the flag on so
                // even a broadcasted subshape (which Shape.Broadcast marks read-only) stays writeable.
                list[i] = new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(ndArrays[i].Storage, outputShapes[i].WithFlags(flagsToSet: ArrayFlags.WRITEABLE)));

            return list;
        }
    }
}
