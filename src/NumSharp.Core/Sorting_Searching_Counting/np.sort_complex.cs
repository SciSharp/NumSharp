namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Sort a complex array using the real part first, then the imaginary part.
        ///     Port of NumPy's <c>numpy/lib/_function_base_impl.py</c>: <c>b = array(a, copy=True);
        ///     b.sort()</c> — i.e. sorted along the LAST axis in the input's own dtype — then up-cast
        ///     to complex. Always returns a fresh complex array; the input is never mutated.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Sorted complex array (sorted along the last axis).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.sort_complex.html<br></br>
        ///     NumPy widths: int8/uint8/int16/uint16 up-cast to complex64, longdouble to clongdouble,
        ///     everything else to complex128; NumSharp has the single <see cref="System.Numerics.Complex"/>
        ///     (complex128) width, so every non-complex dtype lands there — the VALUES are identical
        ///     (all four narrow-int ranges are exact in float32 and float64 alike), only the width differs.
        ///     Char/Decimal (no NumPy analog) take the same route: sort in their own dtype, then cast.
        ///     NaN floats sort to the end (→ <c>nan+0j</c>); complex sorts lexicographically with NumPy's
        ///     any-NaN-part-last ordering. A 0-d input leaks <c>ndarray.sort()</c>'s own
        ///     "axis -1 is out of bounds for array of dimension 0" — exactly as NumPy leaks it.
        /// </remarks>
        /// <exception cref="AxisError">A 0-d string array (<c>axis -1 is out of bounds for array of dimension 0</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty string array: NumPy sorts the text and parses it
        ///     into complex numbers (the next phase of NumSharp's string support). An empty one gives an empty complex128
        ///     array.</exception>
        [NDScoped]
        public static NDArray sort_complex(NDArray a)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Plain(a, static s => sort_complex(s));
            // Text: NumPy's b.sort() checks its axis (a 0-d input refuses -1) and sorts the text (the next phase of the
            // string support); lines of at most one element are left as they are, and astype('D') then parses the text,
            // which the cast below defers for a non-empty array (text conversion) and does for an empty one.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                Backends.Strings.NDStringUfuncs.ValidateReduceAxis("sort", a.ndim, -1);
                if (!Backends.Strings.NDStringUfuncs.ShapeDecides("sort", a, -1))
                    throw Backends.Strings.NDStringUfuncs.NextPhase("sort_complex", a);
            }
            // np.sort == array(a, copy=True) + in-place sort along the default last axis; the sort
            // runs in the ORIGINAL dtype (an int64 past 2^53 must order by its exact value, not by
            // its double image), and only the sorted result is widened.
            var b = np.sort(a, -1);
            return b.typecode == NPTypeCode.Complex ? b : b.astype(NPTypeCode.Complex);
        }
    }
}
