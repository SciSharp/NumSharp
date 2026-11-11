using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a full array with the same shape and type as a given array: NumPy's
        ///     <c>full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None, *, device=None)</c>.
        /// </summary>
        /// <param name="a">The shape and data-type of <paramref name="a"/> define these same attributes of the returned array.</param>
        /// <param name="fill_value">Fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array) that
        ///     broadcasts into the result.</param>
        /// <param name="dtype">Overrides the data type of the result, or null for <paramref name="a"/>'s.</param>
        /// <param name="order">Memory layout: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> (F when <paramref name="a"/> is F-contiguous and not
        ///     C-contiguous, else C) or <c>'K'</c> (the default: <paramref name="a"/>'s axis order, kept also for a
        ///     <paramref name="shape"/> of the same rank), in either case.</param>
        /// <param name="shape">Overrides the shape of the result (an int, an int[] or a tuple converts to
        ///     <see cref="Shape"/>), or null for <paramref name="a"/>'s. With <c>'K'</c> and a different rank the result is C-ordered,
        ///     as NumPy's.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity;
        ///     NumPy's is keyword-only).</param>
        /// <returns>Array of <paramref name="fill_value"/> with the requested shape and type. A <c>numpy.char.chararray</c>
        ///     operand gives a chararray (NumPy allocates the result as the subclass).</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K in either case (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'Z')</c>), reported before the device, as NumPy's empty_like converts its
        ///     arguments in order; <paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's <c>Device not
        ///     understood. Only "cpu" is allowed, but received: ...</c>); a negative dimension in <paramref name="shape"/>
        ///     (<c>negative dimensions are not allowed</c>); a <c>numpy.matrix</c> operand whose result cannot be a matrix
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A <c>numpy.char.chararray</c> operand with a dtype that is not string data: NumPy's
        ///     finalize error (<c>Can only create a chararray from string data.</c>), before anything is filled; or an array
        ///     fill value that does not broadcast into the result, or whose text does not parse into a numeric result.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) result.</exception>
        /// <exception cref="OverflowError">A Python int outside the result's integer range
        ///     (<c>np.full_like(int8_array, 1000)</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full_like.html
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces the former <c>(a, fill_value, dtype, device)</c> / <c>(a, fill_value, dtype, order, device)</c>
        ///     pair,
        ///     so <c>order</c> no longer needs a <c>dtype</c> before it and <c>shape</c> takes None; a positional
        ///     <c>device</c> is written <c>device: "cpu"</c>. <c>subok</c> stays out (the parity plan's
        ///     exclusion).</para>
        /// </remarks>
        public static NDArray full_like(NDArray a, object fill_value, DType dtype = null, char order = 'K', Shape? shape = null,
            string device = null)
        {
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            // `default` is the array object's "no shape override" (a 0-d shape is a real one: its hash is not 0).
            return NDArray.full_like(a, fill_value, dtype, order, shape ?? default);
        }
    }
}
