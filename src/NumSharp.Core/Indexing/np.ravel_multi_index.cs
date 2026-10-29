using System;
using NumSharp.Backends.Iteration;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Converts a tuple of coordinate arrays into an array of flat indices, applying boundary modes per axis —
        ///     NumPy's <c>arr_ravel_multi_index</c> (<c>numpy/_core/src/multiarray/compiled_base.c</c>), statement for
        ///     statement. Inverse of <see cref="unravel_index(NDArray, int[], char)"/>.
        /// </summary>
        /// <param name="multi_index">
        ///     The coordinate arrays, one per dimension of <paramref name="dims"/>: integer or bool, any layout; they
        ///     broadcast against each other, and the broadcast shape is the result's shape. A null entry is NumPy's
        ///     <c>None</c>, which is refused (<see cref="TypeError"/>).
        /// </param>
        /// <param name="dims">The shape of the array the coordinates index. An empty shape (NumPy's <c>()</c>) takes no
        ///     coordinate array and gives the flat index 0.</param>
        /// <param name="mode">
        ///     Boundary mode for every axis: <c>"raise"</c> (default — an out-of-range coordinate raises),
        ///     <c>"wrap"</c> (modulo with sign correction) or <c>"clip"</c> (saturate).
        /// </param>
        /// <param name="order">
        ///     <c>'C'</c> (row-major, default) or <c>'F'</c> (column-major), in either case: the stride ordering used to
        ///     fold coordinates into a flat index.
        /// </param>
        /// <returns>
        ///     The int64 flat indices, in the coordinates' broadcast shape, laid out as NumPy's iterator allocates them
        ///     (KEEPORDER over the coordinate arrays: F-ordered coordinates give an F-ordered result, a transposed set the
        ///     same permutation, mixed or broadcast ones C order). Coordinates that are all 0-d give NumPy's scalar: a
        ///     fresh read-only 0-d array.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     In NumPy's order: an order letter that is not C / F / A / K (<c>order must be one of 'C', 'F', 'A', or 'K'
        ///     (got 'X')</c>); a bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipmodeConverter"/>); an <c>'A'</c> or <c>'K'</c> order (<c>only 'C' or 'F' order is
        ///     permitted</c>); a dims product that overflows int64 (<c>invalid dims: array size defined by dims is larger
        ///     than the maximum possible size.</c>); a coordinate count that differs from <paramref name="dims"/>' length
        ///     (<c>parameter multi_index must be a sequence of length N</c>); a zero dimension while there is a coordinate
        ///     to fold (<c>cannot unravel if shape has zero entries (is empty).</c>, whatever the mode); and (mode
        ///     <c>raise</c>) a coordinate out of range (<c>invalid entry in coordinates array</c>).
        /// </exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool, or a null entry (NumPy's
        ///     <c>None</c>): <c>only int indices permitted</c>, checked in order after the count.</exception>
        /// <exception cref="IncorrectShapeException">The coordinate arrays do not broadcast together (NumPy's
        ///     <c>ValueError</c> text from its iterator, every coordinate shape listed: <c>operands could not be broadcast
        ///     together with shapes (2,) (3,) </c>).</exception>
        /// <exception cref="NotSupportedException">Runtime code generation is unavailable (the IL kernel cannot be
        ///     emitted).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
        ///     <para><b>BREAKING (NumPy parity):</b> the result used to be C-ordered for any coordinate layout, and an
        ///     <c>'A'</c> / <c>'K'</c> order and a dims product overflow were refused only after the coordinates were
        ///     read. A lowercase order letter is accepted, an empty <paramref name="dims"/> is legal, a zero dimension in
        ///     wrap or clip mode raises NumPy's ValueError (it used to divide by zero), and the errors are
        ///     <see cref="ValueError"/> (an <see cref="ArgumentException"/>).</para>
        ///     <para><b>Deliberate divergence:</b> NumPy refuses more than 63 dimensions (<c>too many dimensions passed to
        ///     ravel_multi_index</c>, its fixed iterator operand limit); NumSharp has no dimension cap.</para>
        ///     <para>The body is <see cref="TensorEngine.RavelMultiIndex"/> (the np-layer dtype plan): this facade computes a
        ///     <c>numpy.matrix</c> operand as a plain array and calls the first coordinate array's engine.</para>
        /// </remarks>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string mode = "raise", char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // The engine of the first coordinate array (the default one for an empty, null or null-led list) computes the
            // call, its argument checks included; one mode string serves every dimension.
            return EngineOf(multi_index is { Length: > 0 } ? multi_index[0] : null).RavelMultiIndex(multi_index, dims, mode, order);
        }

        /// <summary>
        ///     Per-axis mode overload. <paramref name="modes"/> length must match
        ///     <paramref name="dims"/> length.
        /// </summary>
        /// <param name="multi_index">The coordinate arrays, one per dimension (integer or bool; a null entry is NumPy's
        ///     <c>None</c>, refused).</param>
        /// <param name="dims">Shape of the array the indices are unravelling into.</param>
        /// <param name="modes">One boundary mode per dimension (NumPy's <c>mode</c> tuple); <c>null</c> is NumPy's
        ///     <c>None</c>, which raises on every dimension.</param>
        /// <param name="order"><c>'C'</c> (default) or <c>'F'</c>, in either case.</param>
        /// <returns>The flat indices, laid out as <see cref="ravel_multi_index(NDArray[], int[], string, char)"/> lays them
        ///     out.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="multi_index"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: a bad order letter; a mode count that differs from the dimensions
        ///     (<c>list of clipmodes has wrong length (1 instead of 2)</c>) or a bad mode (<see cref="ClipmodeConverter"/>);
        ///     an <c>'A'</c> / <c>'K'</c> order; a dims product overflow; a coordinate count that differs from
        ///     <paramref name="dims"/>' length; a zero dimension with a coordinate to fold; (mode <c>raise</c>) a coordinate
        ///     out of range.</exception>
        /// <exception cref="TypeError">A coordinate array that is neither integer nor bool, or a null entry (NumPy's
        ///     <c>None</c>): <c>only int indices permitted</c>.</exception>
        /// <exception cref="IncorrectShapeException">The coordinate arrays do not broadcast together (NumPy's iterator
        ///     text).</exception>
        /// <exception cref="NotSupportedException">The IL kernel cannot be emitted.</exception>
        /// <remarks>The body is <see cref="TensorEngine.RavelMultiIndex"/>, as for the single-mode overload.</remarks>
        public static NDArray<long> ravel_multi_index(NDArray[] multi_index, int[] dims, string[] modes, char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(multi_index);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // The engine of the first coordinate array computes the call; the modes array is NumPy's mode tuple (null is
            // NumPy's None, which raises on every dimension).
            return EngineOf(multi_index is { Length: > 0 } ? multi_index[0] : null).RavelMultiIndex(multi_index, dims, modes, order);
        }

        /// <summary>
        ///     Scalar convenience overload — folds a single coordinate tuple into a
        ///     flat index. Equivalent to wrapping each coord in a 0-d NDArray but
        ///     returns a <see cref="long"/> directly.
        /// </summary>
        /// <param name="coords">One coordinate per dimension.</param>
        /// <param name="dims">The shape the coordinates index; an empty shape takes no coordinate and gives 0.</param>
        /// <param name="mode"><c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>, for every axis.</param>
        /// <param name="order"><c>'C'</c> (default) or <c>'F'</c>, in either case.</param>
        /// <returns>The flat index.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="coords"/> or <paramref name="dims"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: a bad order letter; a bad mode; an <c>'A'</c> / <c>'K'</c> order;
        ///     a dims product overflow; a coordinate count that differs from <paramref name="dims"/>' length
        ///     (<c>parameter multi_index must be a sequence of length N</c>); a zero dimension (<c>cannot unravel if shape
        ///     has zero entries (is empty).</c>); (mode <c>raise</c>) a coordinate out of range (<c>invalid entry in
        ///     coordinates array</c>).</exception>
        public static long ravel_multi_index(long[] coords, int[] dims, string mode = "raise", char order = 'C')
        {
            if (coords == null) throw new ArgumentNullException(nameof(coords));
            if (dims == null) throw new ArgumentNullException(nameof(dims));
            // NumPy's order, as in the array form: the order letter, the mode, the ravel strides, then the coordinates.
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            int modeInt = ClipmodeConverter(mode);
            var ravelStrides = RavelStrides(dims, npyOrder);
            if (coords.Length != dims.Length)
                throw new ValueError($"parameter multi_index must be a sequence of length {dims.Length}");

            int ndim = dims.Length;
            // A scalar tuple is one coordinate to fold, so NumPy's zero-dimension check applies (in every mode).
            ThrowIfZeroDimension(dims);

            long raveled = 0;
            for (int d = 0; d < ndim; d++)
            {
                long j = coords[d];
                long m = dims[d];
                j = ApplyMode(j, m, modeInt, d);
                raveled += j * ravelStrides[d];
            }
            return raveled;
        }

        /// <summary>
        ///     The ravel strides <c>arr_ravel_multi_index</c> computes before it reads the coordinates, with NumPy's two
        ///     refusals: the order, and the dims product (checked on every multiplication, the last one included).
        /// </summary>
        /// <param name="dims">The shape being indexed.</param>
        /// <param name="order">The converted order.</param>
        /// <returns>One stride per dimension (C: the last dimension's is 1; F: the first one's).</returns>
        /// <exception cref="ValueError"><c>'A'</c> / <c>'K'</c> (<c>only 'C' or 'F' order is permitted</c>), or a product
        ///     that overflows int64 (<c>invalid dims: array size defined by dims is larger than the maximum possible
        ///     size.</c>).</exception>
        internal static long[] RavelStrides(int[] dims, NPY_ORDER order)
        {
            bool cOrder = RequireCOrFOrder(order);
            int ndim = dims.Length;
            var ravelStrides = new long[ndim];
            long s = 1;
            for (int i = 0; i < ndim; i++)
            {
                int d = cOrder ? ndim - 1 - i : i;
                ravelStrides[d] = s;
                long next = unchecked(s * dims[d]);
                if (dims[d] != 0 && next / dims[d] != s)
                    throw new ValueError("invalid dims: array size defined by dims is larger than the maximum possible size.");
                s = next;
            }
            return ravelStrides;
        }

        /// <summary>
        ///     <c>ravel_multi_index_loop</c>'s first check: a shape with a zero dimension has no element to index.
        /// </summary>
        /// <param name="dims">The shape being indexed.</param>
        /// <exception cref="ValueError">A zero dimension (<c>cannot unravel if shape has zero entries (is empty).</c>).</exception>
        /// <remarks>NumPy makes it only when there is a coordinate to fold (an empty broadcast skips the loop), in every
        ///     mode: wrap would otherwise divide by the zero dimension.</remarks>
        internal static void ThrowIfZeroDimension(int[] dims)
        {
            for (int d = 0; d < dims.Length; d++)
                if (dims[d] == 0)
                    throw new ValueError("cannot unravel if shape has zero entries (is empty).");
        }

        /// <summary>
        ///     One coordinate under its boundary mode, as <c>ravel_multi_index_loop</c> treats it.
        /// </summary>
        /// <param name="j">The coordinate.</param>
        /// <param name="m">The dimension (not zero: <see cref="ThrowIfZeroDimension"/> ran first).</param>
        /// <param name="mode">0 raise, 1 wrap, 2 clip.</param>
        /// <param name="dim">The axis (unused in the text: NumPy names neither the axis nor the value).</param>
        /// <returns>The coordinate brought into range (wrap / clip) or unchanged (raise, in range).</returns>
        /// <exception cref="ValueError">(raise) the coordinate is outside <c>[0, m)</c>: <c>invalid entry in coordinates
        ///     array</c>.</exception>
        private static long ApplyMode(long j, long m, int mode, int dim)
        {
            switch (mode)
            {
                case 0: // raise
                    if (j < 0 || j >= m)
                        throw new ValueError("invalid entry in coordinates array");
                    return j;
                case 1: // wrap (NumPy staged path)
                    if (j < 0)
                    {
                        j += m;
                        if (j < 0)
                        {
                            j %= m;
                            if (j != 0) j += m;
                        }
                    }
                    else if (j >= m)
                    {
                        j -= m;
                        if (j >= m) j %= m;
                    }
                    return j;
                case 2: // clip
                    if (j < 0) return 0;
                    if (j >= m) return m - 1;
                    return j;
                default:
                    return j;
            }
        }
    }
}
