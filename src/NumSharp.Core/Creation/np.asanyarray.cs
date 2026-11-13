using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through — NumPy's one signature,
        ///     <c>np.asanyarray(a, dtype=None, order=None, *, device=None, copy=None)</c>.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array: scalars, C# arrays and lists (Python
        ///     lists), tuples, nested object arrays, text, and arrays — an array subclass (<see cref="NDMatrix"/>,
        ///     <see cref="NDCharArray"/>) passes through as its class. Null is Python's <c>None</c>.</param>
        /// <param name="dtype">The requested dtype; null infers it from <paramref name="a"/>.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (either case); null is NumPy's None,
        ///     <c>'K'</c>: the operand's own layout is kept.</param>
        /// <param name="device">Null or <c>"cpu"</c> (the Array-API device; NumPy has no other).</param>
        /// <param name="copy">NumPy's copy mode: null (None, the default) copies only when the dtype or layout needs it;
        ///     true always returns a new array (of the operand's class); false never copies and raises when a copy is
        ///     needed — always for a scalar, a list or any other non-array input, which must be converted into a new array.</param>
        /// <returns>The array interpretation of <paramref name="a"/>: <paramref name="a"/> itself when it is an array whose
        ///     dtype and layout satisfy the request (and <paramref name="copy"/> is not true), else a new array — of
        ///     <paramref name="a"/>'s class for a subclass operand.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="order"/> is not one of C, F, A, K (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'X')</c>); <paramref name="device"/> is not <c>"cpu"</c>; <paramref name="copy"/>
        ///     is false and a copy is needed (<c>Unable to avoid copy while creating an array as requested.</c> and NumPy's two
        ///     migration lines); a ragged list; a <c>numpy.matrix</c> operand whose result cannot be a matrix (<c>shape too large
        ///     to be a matrix.</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null and <paramref name="dtype"/> is null (NumPy's
        ///     object array).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype: Python's <c>int(None)</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html — NumPy's <c>array_asanyarray</c>, over
        ///     <c>_array_fromobject_generic</c> with <c>subok</c>.
        ///     <para>A null <paramref name="a"/> is Python's <c>None</c>: with a dtype it is the 0-d array NumPy's
        ///     <c>np.asanyarray(None, dtype)</c> gives (NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a
        ///     <c>StringDType</c> NA object); without one NumPy builds an object array, which NumSharp refuses.</para>
        ///     <para>One signature since API-parity batch 8 (2026-11-12): the <c>(a, dtype, device)</c> form without an order
        ///     and the <c>(a, dtype, order, device)</c> form whose dtype was required (so <c>np.asanyarray(a, order: 'F')</c>
        ///     did not compile) went, and <paramref name="copy"/> arrived. The device moved from the third position to the
        ///     fourth.</para>
        /// </remarks>
        public static NDArray asanyarray(in object a, DType dtype = null, char? order = null, string device = null, bool? copy = null)
        {
            // NumPy's argument parser converts the order letter (PyArray_OrderConverter) before the device, and both before
            // the operand is converted, so an unknown letter is reported first. None is NPY_KEEPORDER.
            char requested = order ?? 'K';
            OrderResolver.Convert(requested);
            DeviceArguments.ValidateDevice(device);
            // A numpy.matrix passes through as a matrix and a numpy.char.chararray as a chararray: the computation is the
            // subclass-aware conversion ([SubclassAware]), which keeps the class np.asarray would strip.
            return NDArray.asanyarray(a, dtype, requested, copy);
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) to an ndarray, passing ndarray
        ///     subclasses through — the consumer round-trip of <c>ndarray.data</c>, NumPy's <c>np.asanyarray(a.data)</c> (a
        ///     zero-copy VIEW sharing memory, copying only if a dtype / layout / copy request forces it). Equivalent to
        ///     <c>np.asanyarray(a.obj, …)</c>.
        /// </summary>
        /// <param name="a">A <see cref="MemoryView"/> over an array. Null is Python's <c>None</c>, as for the <c>object</c>
        ///     overload: C# binds a bare <c>np.asanyarray(null, dtype)</c> HERE (a <see cref="MemoryView"/> is the more specific
        ///     parameter), so the port of NumPy's <c>np.asanyarray(None, dtype)</c> must mean the same.</param>
        /// <param name="dtype">The requested dtype; null keeps the source's.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>; null is <c>'K'</c>.</param>
        /// <param name="device">Null or <c>"cpu"</c>.</param>
        /// <param name="copy">NumPy's copy mode (see <see cref="asanyarray(in object, DType, char?, string, bool?)"/>).</param>
        /// <returns>The array the memory view exposes (a view sharing its memory), or a copy when the request forces one.</returns>
        /// <exception cref="ValueError">An unknown order letter, a device other than <c>"cpu"</c>, or <paramref name="copy"/>
        ///     false when a copy is needed.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null and <paramref name="dtype"/> is null (NumPy's
        ///     object array).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype.</exception>
        /// <remarks>A dedicated overload is required because a <see cref="MemoryView"/> is not otherwise resolvable by the
        ///     <c>object</c> converter. Its parameter took NumPy's name <c>a</c> (it was <c>buffer</c>) and its
        ///     <paramref name="copy"/> arrived in API-parity batch 8.</remarks>
        public static NDArray asanyarray(MemoryView a, DType dtype = null, char? order = null, string device = null, bool? copy = null)
            // None: the object overload's rule (its 0-d setitem value in dtype, or the object-dtype refusal).
            => asanyarray(a is null ? null : a.obj, dtype, order, device, copy);
    }
}
