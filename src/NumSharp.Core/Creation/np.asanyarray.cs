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
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array. This includes scalars, lists, lists of tuples, tuples, tuples of tuples, tuples of lists, and ndarrays.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <returns>Array interpretation of a. If a is an ndarray or a subclass of ndarray, it is returned as-is and no copy is performed.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html</remarks>
        public static NDArray asanyarray(in object a, DType dtype = null, string device = null)
            => asanyarray(in a, dtype, 'K', device);

        /// <summary>
        ///     Convert the input to an ndarray with a specified memory layout.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <param name="order">'C', 'F', 'A' or 'K' (default — resolved against a).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array interpretation of a in the requested layout.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html
        ///     <para>A null <paramref name="a"/> is Python's <c>None</c>: with a dtype it is the 0-d array NumPy's
        ///     <c>np.asanyarray(None, dtype)</c> gives (NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a
        ///     <c>StringDType</c> NA object); without one NumPy builds an object array, which NumSharp refuses.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null and <paramref name="dtype"/> is null (NumPy's
        ///     object array).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype: Python's <c>int(None)</c>.</exception>
        public static NDArray asanyarray(in object a, DType dtype, char order, string device = null)
        {
            DeviceArguments.ValidateDevice(device);
            // a numpy.matrix passes through as a matrix and a numpy.char.chararray as a chararray (np.asarray, which the
            // body calls for an array, strips the class); None is no array, so the boundaries run first as before
            if (a is NDArray nd)
            {
                if (NDMatrixBoundary.Sees(nd))
                    return NDMatrixBoundary.Wrapped((nd, dtype, order), static s => asarray(s.nd, s.dtype, s.order));
                if (NDCharArrayBoundary.Sees(nd))
                    return NDCharArrayBoundary.Kept((nd, dtype, order), static s => asarray(s.nd, s.dtype, s.order));
            }
            return NDArray.AnyArrayFromObject(a, dtype, order);
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) to an ndarray,
        ///     passing ndarray subclasses through — the consumer round-trip of <c>ndarray.data</c>. Matches
        ///     NumPy's <c>np.asanyarray(a.data)</c> (a zero-copy VIEW sharing memory, copying only if a
        ///     dtype/layout change forces it). A dedicated overload is required because a <see cref="MemoryView"/>
        ///     is not otherwise resolvable by the <c>object</c> converter above. Equivalent to
        ///     <c>np.asanyarray(buffer.obj, …)</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array. <c>null</c> is Python's <c>None</c>, as for the
        ///     <c>object</c> overload: C# binds a bare <c>np.asanyarray(null, dtype)</c> HERE (a <see cref="MemoryView"/> is
        ///     the more specific parameter), so the port of NumPy's <c>np.asanyarray(None, dtype)</c> must mean the same.</param>
        /// <param name="dtype">By default, the data-type is inferred from the source.</param>
        /// <param name="order">'C', 'F', 'A' or 'K' (default).</param>
        /// <param name="device">Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html</remarks>
        /// <exception cref="NotSupportedException"><paramref name="buffer"/> is null and <paramref name="dtype"/> is null
        ///     (NumPy's object array).</exception>
        /// <exception cref="TypeError"><paramref name="buffer"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype.</exception>
        public static NDArray asanyarray(MemoryView buffer, DType dtype = null, char order = 'K', string device = null)
        {
            // None: the object overload's rule (its 0-d setitem value in dtype, or the object-dtype refusal).
            if (buffer is null)
                return asanyarray((object)null, dtype, order, device);
            return asanyarray(buffer.obj, dtype, order, device);
        }
    }
}
