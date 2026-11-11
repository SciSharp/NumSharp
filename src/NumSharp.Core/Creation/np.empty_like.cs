using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array with the same shape and type as a given array, of uninitialized (arbitrary) data: NumPy's
        ///     <c>empty_like(prototype, /, dtype=None, order='K', subok=True, shape=None, *, device=None)</c>.
        /// </summary>
        /// <param name="prototype">The shape and data-type of <paramref name="prototype"/> define these same attributes of the
        ///     returned array.</param>
        /// <param name="dtype">
        ///     Overrides the dtype of the result — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>,
        ///     an <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        /// </param>
        /// <param name="order">Memory layout: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> (F when <paramref name="prototype"/> is F-contiguous and not
        ///     C-contiguous, else C) or <c>'K'</c> (the default: <paramref name="prototype"/>'s axis order, kept also for a
        ///     <paramref name="shape"/> of the same rank), in either case.</param>
        /// <param name="shape">Overrides the shape of the result (an int, an int[] or a tuple converts to
        ///     <see cref="Shape"/>), or null for <paramref name="prototype"/>'s. With <c>'K'</c> and a different rank the result is C-ordered,
        ///     as NumPy's.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity;
        ///     NumPy's is keyword-only).</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype. A
        ///     <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the subclass),
        ///     owning its data exactly when the plain result would.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order; <paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>); a negative dimension in <paramref name="shape"/>
        ///     (<c>negative dimensions are not allowed</c>); a <c>numpy.matrix</c> operand whose result cannot be a matrix
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces the former <c>(prototype, dtype, Shape shape = default, device)</c> / <c>(prototype, dtype, shape, order, device)</c>
        ///     pair,
        ///     so <c>order</c> no longer needs a <c>dtype</c> before it and <c>shape</c> takes None; a positional
        ///     <c>device</c> is written <c>device: "cpu"</c>. <c>subok</c> stays out (the parity plan's
        ///     exclusion).</para>
        /// </remarks>
        public static NDArray empty_like(NDArray prototype, DType dtype = null, char order = 'K', Shape? shape = null, string device = null)
            // NumPy's statements (the argument checks in their order, the layout, the allocation) are the array object's;
            // `default` is its "no shape override".
            => NDArray.empty_like(prototype, dtype, shape ?? default, order, device);
    }
}
