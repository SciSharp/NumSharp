using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Repeat each element of an array after themselves.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">Axis along which to repeat values. <c>null</c> (NumPy <c>None</c>) flattens the input and returns a flat array.</param>
        /// <returns>Output array which has the same shape as <paramref name="a"/>, except along <paramref name="axis"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray repeat(NDArray a, int repeats, int? axis = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis))
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis))
            : repeat(a, (long)repeats, axis);

        /// <summary>
        ///     Repeat each element of an array after themselves.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">Axis along which to repeat values. <c>null</c> (NumPy <c>None</c>) flattens the input.</param>
        /// <returns>Output array.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // calculating overload: reclaims the ravel/ascontiguousarray/reshape temps
        public static NDArray repeat(NDArray a, long repeats, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis));
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return UnmanagedStorage.Repeat(a, repeats, axis);
        }

        /// <summary>
        ///     Repeat elements of an array with per-element repeat counts. Mirrors NumPy
        ///     <c>np.repeat(a, repeats, axis)</c>: scalar / size-1 <paramref name="repeats"/> broadcasts to
        ///     every element along the (flattened or selected) axis; otherwise the length must match.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">Repeat counts. Either a 0-d/size-1 array (broadcast) or a 1-D array of length equal to <c>a.size</c> (axis=None) or <c>a.shape[axis]</c>.</param>
        /// <param name="axis">Axis along which to repeat. <c>null</c> flattens the input.</param>
        /// <returns>A new array with elements repeated according to <paramref name="repeats"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="repeats"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="repeats"/> has more than one dimension (<c>object too deep for desired
        ///     array</c>, checked first).</exception>
        /// <exception cref="TypeError"><paramref name="repeats"/> does not cast 'safely' to int64 — uint64, a float, complex,
        ///     decimal or text (<c>Cannot cast array data from dtype('float64') to dtype('int64') according to the rule
        ///     'safe'</c>; <c>Cannot cast scalar …</c> for a 0-d repeats).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">The repeats' length does not match, or a count is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        ///     <para>
        ///     NumPy converts the repeats with <c>PyArray_ContiguousFromAny(repeats, NPY_INTP, 0, 1)</c> before it reads the axis:
        ///     the rank (at most 1) first, then the 'safe' cast — so a 2-D repeats is the rank's ValueError whatever its dtype,
        ///     and both precede an out-of-range axis. (A 2-D repeats used to be flattened.)
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // calculating overload: reclaims the ravel temps
        public static NDArray repeat(NDArray a, NDArray repeats, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, repeats, axis), static s => repeat(s.a, s.repeats, s.axis));
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (repeats is null)
                throw new ArgumentNullException(nameof(repeats));
            return UnmanagedStorage.Repeat(a, repeats, axis);
        }

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static unsafe NDArray repeat<T>(T a, int repeats) where T : unmanaged
            => repeat(a, (long)repeats);

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static unsafe NDArray repeat<T>(T a, long repeats) where T : unmanaged
        {
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");

            if (repeats == 0)
                return new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(0));

            var ret = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(repeats));
            var dst = (T*)ret.Address;
            for (long j = 0; j < repeats; j++)
                dst[j] = a;
            return ret;
        }
    }
}
