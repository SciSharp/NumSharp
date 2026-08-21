namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>PyArray_Return</c> analog for the reduction family: whenever a reduction
        ///     hands back a FRESH 0-d result (axis=None; axis reducing the last dimension away;
        ///     keepdims over a 0-d input — all of which NumPy converts to a numpy SCALAR at the API
        ///     boundary), the result reports the scalar's flags — <c>num=263</c>, WRITEABLE off
        ///     (probed 2.4.2: <c>np.sum(a).flags.writeable is False</c>, and even
        ///     <c>np.sum(np.array(5), keepdims=True)</c> comes back a read-only scalar). NumSharp
        ///     has no scalar type, so its 0-d NDArray plays that role and carries the flags.
        /// </summary>
        /// <remarks>
        ///     No-op for non-0-d results, so every exit of a reduction can route through here —
        ///     the keepdims path reshapes to <c>(1,)*ndim</c> first and passes untouched. Must NOT
        ///     be applied to an <c>out=</c> operand: NumPy returns the out array itself, writeable
        ///     (probed: <c>np.sum(a, out=np.array(0)).flags.writeable is True</c>).
        /// </remarks>
        internal NDArray MarkReductionScalar()
        {
            if (Shape.IsScalar && Shape.IsWriteable)
                Storage.SetShapeUnsafe(Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            return this;
        }
    }
}
