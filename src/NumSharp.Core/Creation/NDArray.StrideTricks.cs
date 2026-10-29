using NumSharp.Backends;

namespace NumSharp
{
    // ============================== lib/_stride_tricks_impl.py ==============================
    // NumPy's broadcast_to as a member of the array object (P6's structural part): BroadcastTo, the read-only view
    // np.broadcast_to returns, over Shape.BroadcastTo's check and layout (View/Shape.Broadcasting.cs). The np.broadcast_to
    // facades keep their overloads and their numpy.matrix regions and call it; the engine calls it where it broadcast an
    // operand through np.broadcast_to. np.broadcast_shapes' engine callers call Shape.BroadcastShape, its body.
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
        internal static NDArray BroadcastTo(UnmanagedStorage from, Shape against) =>
            new NDArray(UnmanagedStorage.CreateBroadcastedUnsafe(from, Shape.BroadcastTo(from.Shape, against).WithFlags(flagsToClear: ArrayFlags.WRITEABLE)));

        /// <summary>
        ///     <see cref="BroadcastTo(UnmanagedStorage, Shape)"/> over an array: <paramref name="from"/>'s storage viewed
        ///     read-only at <paramref name="against"/>'s dimensions.
        /// </summary>
        /// <param name="from">The source array (not null). Disposing it later leaves the view valid: the view holds its own
        ///     reference on the buffer.</param>
        /// <param name="against">The target; only its dimensions are read.</param>
        /// <returns>A new read-only view; see <see cref="BroadcastTo(UnmanagedStorage, Shape)"/>.</returns>
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the target one way.</exception>
        internal static NDArray BroadcastTo(NDArray from, Shape against) => BroadcastTo(from.Storage, against);
    }
}
