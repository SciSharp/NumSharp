using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Replace elements of <paramref name="a"/> with given values, at the
        ///     specified flat indices. In-place — modifies <paramref name="a"/>.
        ///     Equivalent to <c>a.flat[ind] = v</c> with cyclic
        ///     broadcasting of <paramref name="v"/>.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="ind">
        ///     NumPy's <c>ind</c>: the flat indices, an integer array or a C# integer (a 0-d index), cast to int64
        ///     internally, in C order whatever its layout. Indexing is into the C-order flattening of
        ///     <paramref name="a"/>.
        /// </param>
        /// <param name="v">
        ///     NumPy's <c>v</c>: the values to write, read in C order. Cast to <paramref name="a"/>'s dtype (NumPy's
        ///     FORCECAST); a C# literal is packed as the Python scalar it stands for (range-checked, never wrapped). Cycles
        ///     modulo its size — shorter than <paramref name="ind"/> is fine; an empty array writes nothing (NumPy's no-op). <c>null</c> is
        ///     Python's <c>None</c>, converted into <paramref name="a"/>'s dtype (<see cref="NDArray.NoneForStore"/>: NaN,
        ///     <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a <c>StringDType</c> NA object; int()'s TypeError for an integer
        ///     array — raised before the indices' bounds and even for an empty index array, as NumPy converts first).
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>; <c>null</c> is NumPy's <c>None</c>
        ///     (raise).
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="ind"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="v"/> is null (<c>None</c>) and <paramref name="a"/> is an integer
        ///     (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipModes.ClipmodeConverter"/>) — reported first, as NumPy parses it with the arguments —, then a read-only
        ///     <paramref name="a"/> (<c>put: output array is read-only</c>).</exception>
        /// <exception cref="TypeError">The index dtype cannot be cast to int64 under the <c>safe</c> rule (float, complex,
        ///     uint64), an empty index array included.</exception>
        /// <exception cref="IndexError"><paramref name="a"/> is empty while <paramref name="ind"/> is not
        ///     (<c>cannot replace elements of an empty array</c>), or an index is out of bounds under <c>mode="raise"</c>
        ///     (<c>index 9 is out of bounds for axis 0 with size 4</c>).</exception>
        /// <exception cref="OverflowError">A C# literal value out of <paramref name="a"/>'s integer range (NumPy's
        ///     OverflowError: "Python integer 1000 out of bounds for int8"), checked after the indices.</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer array ("cannot convert float NaN to
        ///     integer").</exception>
        /// <exception cref="OverflowError">A float infinity literal into an integer array ("cannot convert float infinity
        ///     to integer").</exception>
        /// <exception cref="TypeError">A complex literal into a real array (Python's <c>float()</c> / <c>int()</c>
        ///     text).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.put.html
        ///     <para>The work is <see cref="UnmanagedStorage.PutTo"/>, NumPy's <c>PyArray_PutTo</c>: see there for the check order, the copy
        ///     NumPy makes when the target overlaps the values or the indices, and which writes survive an out-of-bounds
        ///     index.</para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the contiguous scratch of a non-contiguous or overlapping target and
        ///     the int64 and dtype-converted copies are released on return, exceptions included. The caller's arrays were
        ///     built before the scope opened, so the scope never disposes them.</para>
        ///     <para>One overload, NumPy's signature <c>put(a, ind, v, mode='raise')</c> keyword for keyword (2026-11-11).
        ///     A scalar index and value need no overload of their own: a C# literal converts to a 0-d array marked as the
        ///     Python scalar it stands for, so <c>np.put(a, 1, 1000)</c> raises NumPy's OverflowError for an int8 target
        ///     exactly as NumPy's Python int does. The former <c>(long index, object value)</c> overload made that very
        ///     call ambiguous (CS0121: the literal index preferred <c>long</c>, the literal value <c>NDArray</c>), so only
        ///     cast arguments ever reached it; a value held in an <c>object</c> variable goes through
        ///     <see cref="np.asanyarray(in object, DType, char?, string, bool?)"/>.</para>
        /// </remarks>
        [NDScoped]
        public static void put(NDArray a, NDArray ind, NDArray v, string mode = "raise")
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (ind is null) throw new ArgumentNullException(nameof(ind));

            // A null `v` is Python's None: PutTo converts it at NumPy's conversion point (PutValuesLikeNumPy), after the
            // mode, the writeability and the empty-target checks, as PyArray_PutTo does.
            UnmanagedStorage.PutTo(a, ind, v, mode);
        }
    }
}
