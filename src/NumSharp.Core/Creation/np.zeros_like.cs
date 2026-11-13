using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array of zeros with the same shape and type as a given array: NumPy's
        ///     <c>zeros_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)</c>.
        /// </summary>
        /// <param name="a">The shape and data-type of <paramref name="a"/> define these same attributes of the returned array.</param>
        /// <param name="dtype">Overrides the data type of the result, or null for <paramref name="a"/>'s.</param>
        /// <param name="order">Memory layout: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> (F when <paramref name="a"/> is F-contiguous and not
        ///     C-contiguous, else C) or <c>'K'</c> (the default: <paramref name="a"/>'s axis order, kept also for a
        ///     <paramref name="shape"/> of the same rank), in either case.</param>
        /// <param name="shape">Overrides the shape of the result (an int, an int[] or a tuple converts to
        ///     <see cref="Shape"/>), or null for <paramref name="a"/>'s. With <c>'K'</c> and a different rank the result is C-ordered,
        ///     as NumPy's.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity;
        ///     NumPy's is keyword-only).</param>
        /// <returns>A new, owned, writeable array of zeros (a text dtype's empty strings) in the requested layout. A
        ///     <c>numpy.char.chararray</c> operand gives a chararray (NumPy allocates the result as the subclass).</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order; <paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>); a negative dimension in <paramref name="shape"/>
        ///     (<c>negative dimensions are not allowed</c>); a <c>numpy.matrix</c> operand whose result cannot be a matrix
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand with a dtype that is not string data: NumPy's
        ///     finalize error (<c>Can only create a chararray from string data.</c>), before anything is filled.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null (its layout is read after the order and device checks).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces the former <c>(a, dtype, device)</c> / <c>(a, dtype, order, device)</c> pair,
        ///     so <c>order</c> no longer needs a <c>dtype</c> before it and <c>shape</c> takes None; a positional
        ///     <c>device</c> is written <c>device: "cpu"</c>. <c>subok</c> stays out (the parity plan's
        ///     exclusion).</para>
        /// </remarks>
        public static NDArray zeros_like(NDArray a, DType dtype = null, char order = 'K', Shape? shape = null, string device = null)
        {
            // NumPy's statements (empty_like's argument checks and layout, then the fill with 0) are the array object's;
            // `default` is its "no shape override" (a 0-d shape is a real override: its hash is not 0).
            return NDArray.zeros_like(a, dtype, order, shape ?? default, device);
        }
    }
}
