namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     Return an array copy of the given object.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="order">
        ///     Controls the memory layout of the copy.
        ///     'C' - row-major, 'F' - column-major, 'A' - 'F' if source is F-contiguous else 'C',
        ///     'K' - match source layout as closely as possible.
        /// </param>
        /// <param name="subok">
        ///     True to keep the class of a subclass object (NumPy's <c>array(a, order=order, subok=True, copy=True)</c>):
        ///     the copy of a <c>numpy.char.chararray</c> is a chararray that owns its data, the copy of a
        ///     <c>numpy.matrix</c> a matrix. False (the default, as in NumPy) returns a plain array whatever
        ///     <paramref name="a"/> is — use it when the copy is passed to code written for plain arrays.
        /// </param>
        /// <returns>Array interpretation of a: a new array holding a copy of its data.</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of NumPy's order codes
        ///     (<c>order must be one of 'C', 'F', 'A', or 'K' (got 'Q')</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.copy.html</remarks>
        public static NDArray copy(NDArray a, char order = 'K', bool subok = false)
            // The copy is the array object's; subok decides the class of a subclass operand's copy (the computation's
            // [SubclassRule]: NumPy's array(..., subok=subok)).
            => NDArray.copy_(a, order);
    }

    public partial class NDArray
    {
        /// <summary>
        ///     <c>np.copy</c>'s computation: NumPy's <c>array(a, order=order, copy=True)</c> on the array, a new array holding a
        ///     copy of its data in the requested layout.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>The copy, a new owning array.</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of NumPy's order codes.</exception>
        /// <remarks>An NDArray reference binds the instance <see cref="NDArray.copy(char)"/> (the subclasses' own copy
        ///     members hide it, they do not override it), which is the plain copy. The class of a subclass operand's copy is
        ///     <c>np.copy</c>'s <c>subok</c> (<see cref="NDSubclassRules.Copy"/>).</remarks>
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.Copy))]
        internal static NDArray copy_(NDArray a, char order) => a.copy(order);
    }
}
