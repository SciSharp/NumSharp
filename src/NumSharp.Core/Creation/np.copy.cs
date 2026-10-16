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
        {
            if (subok)
            {
                // subok=True creates the copy as a's class. Each rule computes the plain copy inside its region of plain
                // semantics, then hands it out as the subclass (NDCharArrayBoundary / NDMatrixBoundary).
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Kept((a, order), static s => copy(s.a, s.order));
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, order), static s => copy(s.a, s.order));
            }
            // An NDArray reference binds NDArray.copy (the subclasses' own copy members hide it, they do not override
            // it), which is the plain copy NumPy's subok=False makes.
            return a.copy(order);
        }
    }
}
