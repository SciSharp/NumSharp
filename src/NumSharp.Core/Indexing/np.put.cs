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
        ///     Equivalent to <c>a.flat[indices] = values</c> with cyclic
        ///     broadcasting of <paramref name="values"/>.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="indices">
        ///     Integer array of flat indices (cast to int64 internally, in C order whatever its layout). Indexing is into
        ///     the C-order flattening of <paramref name="a"/>.
        /// </param>
        /// <param name="values">
        ///     Values to write, read in C order. Cast to <paramref name="a"/>'s dtype (NumPy's FORCECAST); a C# literal is
        ///     packed as the Python scalar it stands for (range-checked, never wrapped). Cycles modulo its size — shorter
        ///     than <paramref name="indices"/> is fine; an empty array writes nothing (NumPy's no-op). <c>null</c> is
        ///     Python's <c>None</c>, converted into <paramref name="a"/>'s dtype (<see cref="NDArray.NoneForStore"/>: NaN,
        ///     <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a <c>StringDType</c> NA object; int()'s TypeError for an integer
        ///     array — raised before the indices' bounds and even for an empty index array, as NumPy converts first).
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>; <c>null</c> is NumPy's <c>None</c>
        ///     (raise).
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="values"/> is null (<c>None</c>) and <paramref name="a"/> is an integer
        ///     (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipModes.ClipmodeConverter"/>) — reported first, as NumPy parses it with the arguments —, then a read-only
        ///     <paramref name="a"/> (<c>put: output array is read-only</c>).</exception>
        /// <exception cref="TypeError">The index dtype cannot be cast to int64 under the <c>safe</c> rule (float, complex,
        ///     uint64), an empty index array included.</exception>
        /// <exception cref="IndexError"><paramref name="a"/> is empty while <paramref name="indices"/> is not
        ///     (<c>cannot replace elements of an empty array</c>), or an index is out of bounds under <c>mode="raise"</c>
        ///     (<c>index 9 is out of bounds for axis 0 with size 4</c>).</exception>
        /// <exception cref="OverflowException">A C# literal value out of <paramref name="a"/>'s integer range (NumPy's
        ///     OverflowError: "Python integer 1000 out of bounds for int8"), checked after the indices.</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer array ("cannot convert float NaN to
        ///     integer").</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.put.html
        ///     <para>The work is <see cref="UnmanagedStorage.PutTo"/>, NumPy's <c>PyArray_PutTo</c>: see there for the check order, the copy
        ///     NumPy makes when the target overlaps the values or the indices, and which writes survive an out-of-bounds
        ///     index.</para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the contiguous scratch of a non-contiguous or overlapping target and
        ///     the int64 and dtype-converted copies are released on return, exceptions included. The caller's arrays were
        ///     built before the scope opened, so the scope never disposes them.</para>
        /// </remarks>
        [NDScoped]
        public static void put(NDArray a, NDArray indices, NDArray values, string mode = "raise")
        {
            using var plain = NDMatrixBoundary.PlainFor(a, indices, values);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            // A null `values` is Python's None: PutTo converts it at NumPy's conversion point (PutValuesLikeNumPy).
            UnmanagedStorage.PutTo(a, indices, values, mode);
        }

        /// <summary>
        ///     Scalar-index, scalar-value convenience overload.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="index">The flat index into the C-order flattening of <paramref name="a"/>.</param>
        /// <param name="value">The value to write. A C# number is packed into <paramref name="a"/>'s dtype by NumPy's
        ///     <c>PyArray_Pack</c> rules for its kind (<see cref="NDArray.AsStoreValue"/>); <c>null</c> is Python's
        ///     <c>None</c> (<see cref="NDArray.NoneForStore"/>); anything else is converted with
        ///     <see cref="np.asanyarray(in object, DType, string)"/> and cast.</param>
        /// <param name="mode">Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="value"/> is null (<c>None</c>) and <paramref name="a"/> is an integer
        ///     (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/>, then a read-only <paramref name="a"/>.</exception>
        /// <exception cref="IndexError"><paramref name="a"/> is empty, or <paramref name="index"/> is out of bounds under
        ///     <c>mode="raise"</c>.</exception>
        /// <exception cref="OverflowException">A C# integer out of <paramref name="a"/>'s range, or a float infinity into an
        ///     integer array (NumPy's OverflowError texts).</exception>
        /// <exception cref="ValueError">A float NaN into an integer array.</exception>
        /// <exception cref="TypeError">A complex into a real array.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the 0-d index array and the conversion of <paramref name="value"/> are built
        ///     in THIS frame, so this method needs a scope of its own to release them. An <see cref="NDArray"/> passed as
        ///     <paramref name="value"/> comes back from <see cref="NDArray.AsStoreValue"/> unchanged and is never disposed.
        /// </remarks>
        [NDScoped]
        public static void put(NDArray a, long index, object value, string mode = "raise")
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy converts the index (a Python int) and the value inside PyArray_PutTo, after the mode, the
            // writeability and the empty-target checks: PutTo runs them in that order (a null value is None, converted there).
            UnmanagedStorage.PutTo(a, NDArray.Scalar(index), value, mode);
        }
    }
}
