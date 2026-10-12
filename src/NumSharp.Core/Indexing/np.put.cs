using System;
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
        ///     than <paramref name="indices"/> is fine; an empty array writes nothing (NumPy's no-op).
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>; <c>null</c> is NumPy's <c>None</c>
        ///     (raise).
        /// </param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (NumPy's <c>clipmode_parser</c> texts, see
        ///     <see cref="ClipModeConverter"/>) — reported first, as NumPy parses it with the arguments —, then a read-only
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
        ///     <para>The work is <see cref="PutTo"/>, NumPy's <c>PyArray_PutTo</c>: see there for the check order, the copy
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
            if (values is null) throw new ArgumentNullException(nameof(values));

            PutTo(a, indices, values, mode);
        }

        /// <summary>
        ///     Scalar-index, scalar-value convenience overload.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="index">The flat index into the C-order flattening of <paramref name="a"/>.</param>
        /// <param name="value">The value to write. A C# number is packed into <paramref name="a"/>'s dtype by NumPy's
        ///     <c>PyArray_Pack</c> rules for its kind (<see cref="NDArray.AsStoreValue"/>); anything else is converted with
        ///     <see cref="np.asanyarray(in object, DType, string)"/> and cast.</param>
        /// <param name="mode">Boundary mode: <c>"raise"</c> (default), <c>"wrap"</c>, or <c>"clip"</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="value"/> is null.</exception>
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
            if (value is null) throw new ArgumentNullException(nameof(value));

            // NumPy converts the index (a Python int) and the value inside PyArray_PutTo, after the mode, the
            // writeability and the empty-target checks: PutTo runs them in that order.
            PutTo(a, NDArray.Scalar(index), value, mode);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_PutTo</c> (<c>item_selection.c</c>) behind <c>ndarray.put</c> and <c>np.put</c>, with
        ///     <c>array_put</c>'s clip-mode conversion, statement for statement. numpy.ma's <c>put</c> calls it for the data
        ///     and for the mask (<c>self._data.put</c> / <c>m.put</c>).
        /// </summary>
        /// <param name="a">The target, written in place.</param>
        /// <param name="indices">The flat indices: any integer or bool array that casts to int64 under <c>safe</c>, any
        ///     layout (read in C order).</param>
        /// <param name="values">The values: an <see cref="NDArray"/> (a C# literal's marked 0-d array is packed as the
        ///     Python scalar it stands for), a boxed C# number (packed by its kind), or anything
        ///     <see cref="NDArray.AsStoreValue"/> converts. Read in C order and cycled.</param>
        /// <param name="mode">The clip mode string (<see cref="ClipModeConverter"/>).</param>
        /// <exception cref="ValueError">A bad <paramref name="mode"/>; then a read-only <paramref name="a"/>.</exception>
        /// <exception cref="TypeError">An index dtype that does not cast to int64 under <c>safe</c>.</exception>
        /// <exception cref="IndexError">A non-empty <paramref name="indices"/> into an empty <paramref name="a"/>, or an
        ///     out-of-bounds index under raise.</exception>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="a"/>'s range.</exception>
        /// <exception cref="NotSupportedException">No IL put kernel exists for the element width.</exception>
        /// <remarks>
        ///     <para>
        ///     The order is NumPy's: the mode (parsed with <c>array_put</c>'s arguments), the writeability, the index cast
        ///     (<c>PyArray_ContiguousFromAny(indices, NPY_INTP)</c>, so an empty float index array is the TypeError too), the
        ///     empty-target IndexError, the value conversion (<c>PyArray_FromAny(values, a.dtype, DEFAULT | FORCECAST)</c>: a
        ///     literal is range-checked even when nothing is written), and only then the no-op of an empty value array.
        ///     </para>
        ///     <para>
        ///     NumPy writes into a copy of the target (<c>NPY_ARRAY_ENSURECOPY | NPY_ARRAY_WRITEBACKIFCOPY</c>) when the
        ///     target is not C-contiguous or when it may share memory with the converted values or indices
        ///     (<c>arrays_overlap</c>, a bounds check: <see cref="NDMemOverlap.SolveMayShareMemory"/> with
        ///     <c>maxWork: 0</c>), so a cyclic read of values that alias the target sees the ORIGINAL elements
        ///     (<c>np.put(a, [1, 2, 3], a[0:3])</c> shifts, <c>np.put(a, a, v)</c> reads the original indices). The copy is
        ///     written back only on success, as NumPy discards it on an error; a contiguous target without overlap is
        ///     written in place, so the indices before an out-of-bounds one are written (NumPy's raise loop writes as it
        ///     goes). An int32 index array NumSharp reads in place (NumPy copies it to intp) counts as overlap when it may
        ///     share the target's memory, which gives NumPy's answer: its values are read before any write.
        ///     </para>
        ///     <para>
        ///     Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>np.put</c> overloads and
        ///     numpy.ma's <c>put</c>, which opens a scope of its own; every converted copy and the scratch target are
        ///     released by that scope, which also keeps them reachable while the kernel reads them through raw pointers.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        internal static void PutTo(NDArray a, NDArray indices, object values, string mode)
        {
            // array_put converts `mode` with PyArray_ClipmodeConverter while it parses its arguments: a bad mode is
            // reported before PyArray_PutTo looks at the target.
            int modeInt = ClipModeConverter(mode);

            // PyArray_FailUnlessWriteable(self, "put: output array") is PyArray_PutTo's first check (probed 2.4.2:
            // put(ro, [], []) and put(ro, [oob], v) report read-only).
            NumSharpException.ThrowIfNotWriteable(a.Shape, "put: output array");

            // PyArray_ContiguousFromAny(indices0, NPY_INTP): the 'safe' cast is checked for every index array, an empty
            // one too, before the empty target and the values (probed: an empty float index array raises TypeError).
            var idx64 = CastIndicesToInt64(indices, "safe", out _, out bool idx32);
            long ni = indices.size;
            if (ni > 0 && a.size == 0)
                throw new IndexError("cannot replace elements of an empty array");

            // PyArray_FromAny(values0, a.dtype, DEFAULT | FORCECAST), then `nv <= 0` returns without writing: a literal is
            // packed (range-checked) even when nothing is written (np.put(int8, [], 1000) raises).
            var vals = PutValuesLikeNumPy(values, a);
            if (vals.size == 0 || ni == 0)
                return;

            // overlap = arrays_overlap(self, values) || arrays_overlap(self, indices); with a non-contiguous target NumPy
            // also writes into a C-contiguous copy and writes it back.
            bool overlaps =
                NDMemOverlap.SolveMayShareMemory(a, vals, maxWork: 0) != MemOverlap.No ||
                NDMemOverlap.SolveMayShareMemory(a, idx64, maxWork: 0) != MemOverlap.No;
            if (a.Storage.IsVString)
            {
                // A StringDType target: the same put runs on marks (the values' element numbers, cycled by the numeric put
                // with its mode), and the marked elements receive the strings. NumPy writes a contiguous target that
                // overlaps nothing as it goes, so the elements before an out-of-bounds index stay written; a copy (a
                // non-contiguous target, or an overlap) is discarded on the error.
                Backends.Strings.NDVStringMove.ScatterVia(a, vals, (marks, index) => PutTo(marks, indices, index, mode),
                    keepPartialOnError: !overlaps && a.Shape.IsContiguous);
                return;
            }
            if (overlaps || !a.Shape.IsContiguous)
            {
                // A contiguous target is copied with copy() (ascontiguousarray would hand back the target itself); the
                // scratch is released by the caller's scope. A throw from the kernel skips the write-back: NumPy's
                // PyArray_DiscardWritebackIfCopy, so nothing reaches `a`.
                var scratch = a.Shape.IsContiguous ? a.copy() : np.ascontiguousarray(a);
                PutKernel(scratch, idx64, idx32, ni, vals, modeInt);
                np.copyto(a, scratch, casting: "unsafe");
                return;
            }

            PutKernel(a, idx64, idx32, ni, vals, modeInt);
        }

        /// <summary>
        ///     <c>PyArray_PutTo</c>'s value conversion, <c>PyArray_FromAny(values0, a.dtype, NPY_ARRAY_DEFAULT |
        ///     NPY_ARRAY_FORCECAST)</c>: the values as a C-contiguous array of <paramref name="a"/>'s element type.
        /// </summary>
        /// <param name="values">An <see cref="NDArray"/> or any value <see cref="NDArray.AsStoreValue"/> accepts.</param>
        /// <param name="a">The target, whose dtype the values take.</param>
        /// <returns>The caller's array when it already is C-contiguous of <paramref name="a"/>'s element type, else a new
        ///     array (released by the caller's scope).</returns>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="a"/>'s range.</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer array.</exception>
        /// <exception cref="TypeError">A complex literal into a real array.</exception>
        /// <remarks>
        ///     The cast is made in C ORDER: the put kernels read the converted values linearly, and <c>astype</c>'s default
        ///     <c>'K'</c> keeps an F-order input F-ordered, which put an F-order int32 array's values into an int64 target
        ///     in memory order (<c>[10, 12, 11, 13]</c> where NumPy writes <c>[10, 11, 12, 13]</c>). The same-element-type
        ///     route takes <c>ascontiguousarray</c>, already C order.
        /// </remarks>
        [NDScopedCovered]
        private static NDArray PutValuesLikeNumPy(object values, NDArray a)
        {
            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar and is packed with PyArray_Pack's range
            // checks; a boxed C# number is packed by its kind; any other value is converted, for the cast below.
            var v = values is NDArray nd
                ? NDArray.PackForStore(nd, a.typecode, pyArrayPackRoute: false)
                : NDArray.AsStoreValue(values, a.typecode);

            // Same element type (for strings: same class AND width) reads the values as they are; anything else is
            // converted to a's dtype first (NumPy's FORCECAST: a string is truncated / NUL-padded, S <-> U checked).
            if (v.HasSameElementType(a))
                return v.Shape.IsContiguous ? v : np.ascontiguousarray(v);
            return v.astype(a.dtype, copy: true, order: 'C');
        }

        /// <summary>
        ///     The contiguous core of <see cref="PutTo"/>: runs the IL scatter kernel straight into <paramref name="a"/>'s
        ///     buffer, cycling <paramref name="vals"/>.
        /// </summary>
        /// <param name="a">The C-contiguous target (the caller's array, or the scratch copy).</param>
        /// <param name="idx64">The C-contiguous indices: int64, or int32 read in place when <paramref name="idx32"/>.</param>
        /// <param name="idx32">True when <paramref name="idx64"/> holds int32 values.</param>
        /// <param name="ni">The number of indices.</param>
        /// <param name="vals">The C-contiguous values of <paramref name="a"/>'s element type, at least one.</param>
        /// <param name="mode">The parsed boundary mode (<see cref="ClipModeRaise"/>, <see cref="ClipModeWrap"/>,
        ///     <see cref="ClipModeClip"/>).</param>
        /// <exception cref="NotSupportedException">No IL put kernel exists for the element width.</exception>
        /// <exception cref="IndexError">An index is out of bounds under mode raise (NumPy's text; the indices before it
        ///     have been written).</exception>
        private static unsafe void PutKernel(NDArray a, NDArray idx64, bool idx32, long ni, NDArray vals, int mode)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            // idx32: the index buffer holds int32 values; the kernels' `long*` parameter is a reinterpret.
            long* idxPtr = idx32
                ? (long*)((int*)idx64.Storage.Address + idx64.Shape.offset)
                : (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* valsPtr = (byte*)vals.Storage.Address + vals.Shape.offset * vals.dtypesize;

            long status;
            // The lean flat scatter (DirectILKernelGenerator.GatherFlat.cs) serves RAISE mode with a
            // compile-time element width; wrap/clip keep the general kernel.
            var flat = mode == ClipModeRaise ? DirectILKernelGenerator.GetPutFlatKernel(copyKind, idx32) : null;
            if (flat != null)
            {
                status = flat(dstPtr, idxPtr, ni, valsPtr, vals.size, a.size);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetPutKernel(copyKind, idx32);
                if (kernel == null)
                    throw new NotSupportedException("np.put: IL kernel unavailable");
                status = kernel(dstPtr, idxPtr, ni,
                                valsPtr, vals.size,
                                a.size, a.dtypesize, mode);
            }
            if (status < ni)
            {
                long badI = status;
                long badVal = idx32 ? ((int*)idxPtr)[badI] : idxPtr[badI];
                // NumPy's check_and_adjust_index text (an IndexError), the ORIGINAL value of the bad index.
                throw new IndexError($"index {badVal} is out of bounds for axis 0 with size {a.size}");
            }
        }
    }
}
