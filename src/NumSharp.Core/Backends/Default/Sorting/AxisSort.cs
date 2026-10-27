using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    /// Along-axis sort / argsort, structured exactly like NumPy's <c>_new_sortlike</c> /
    /// <c>_new_argsortlike</c> (item_selection.c): an NDIter <b>IterAllButAxis</b> drive (the
    /// sort axis is dropped from the iterator via <c>op_axes</c> so it can't coalesce) hands one
    /// 1-D line per call to a high-performance inner sort kernel.
    ///
    /// Inner kernel = LSD <see cref="RadixSort"/> for every fixed-width numeric dtype (POC: int
    /// sort 0.7-0.9x NumPy, argsort 2-4x FASTER than NumPy); scalar BCL introsort with the exact
    /// NumPy comparators for the Half/Complex/Decimal value sorts and the Decimal argsort (the Half
    /// and Complex argsorts are radix too: their keys fold the values NumPy's comparators call
    /// equal). Floats partition NaN to the end (NumPy sorts NaN last). Radix is stable, so argsort
    /// ties resolve in ascending index order (NumPy 'stable').
    /// </summary>
    internal static unsafe class AxisSort
    {
        // ----- key adapters: monotonic unsigned transform so ascending key order == NumPy value order -----
        private interface IKey32<T> where T : unmanaged { uint To(T v); T From(uint k); int Bytes { get; } }
        private interface IKey64<T> where T : unmanaged { ulong To(T v); T From(ulong k); }

        private readonly struct KBool : IKey32<byte> { public uint To(byte v) => v; public byte From(uint k) => (byte)k; public int Bytes => 1; }
        private readonly struct KU8 : IKey32<byte> { public uint To(byte v) => v; public byte From(uint k) => (byte)k; public int Bytes => 1; }
        private readonly struct KI8 : IKey32<sbyte> { public uint To(sbyte v) => (byte)(v ^ unchecked((sbyte)0x80)); public sbyte From(uint k) => (sbyte)((byte)k ^ 0x80); public int Bytes => 1; }
        private readonly struct KI16 : IKey32<short> { public uint To(short v) => (ushort)(v ^ unchecked((short)0x8000)); public short From(uint k) => (short)((ushort)k ^ 0x8000); public int Bytes => 2; }
        private readonly struct KU16 : IKey32<ushort> { public uint To(ushort v) => v; public ushort From(uint k) => (ushort)k; public int Bytes => 2; }
        private readonly struct KChar : IKey32<char> { public uint To(char v) => v; public char From(uint k) => (char)k; public int Bytes => 2; }
        private readonly struct KI32 : IKey32<int> { public uint To(int v) => (uint)v ^ 0x80000000u; public int From(uint k) => (int)(k ^ 0x80000000u); public int Bytes => 4; }
        private readonly struct KU32 : IKey32<uint> { public uint To(uint v) => v; public uint From(uint k) => k; public int Bytes => 4; }
        private readonly struct KI64 : IKey64<long> { public ulong To(long v) => (ulong)v ^ 0x8000000000000000UL; public long From(ulong k) => (long)(k ^ 0x8000000000000000UL); }
        private readonly struct KU64 : IKey64<ulong> { public ulong To(ulong v) => v; public ulong From(ulong k) => k; }

        // ============================ public entry points ============================

        /// <summary>np.sort: copy(order='K') then sort in place (axis=null flattens) — NumPy's
        /// fromnumeric.sort verbatim, so an F-contiguous input keeps its F layout, a 3-D transpose
        /// keeps its exact stride order, and strided/negative-stride inputs come back C (probed
        /// 2.4.2). The line driver walks the copy's own strides, so the in-place pass is
        /// layout-agnostic.</summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>The sorted copy.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d input refuses every axis: NumPy's sort does not ravel
        ///     it first).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements holding an NA whose NA object is
        ///     neither NaN-like nor a string.</exception>
        public static NDArray Sort(NDArray a, int? axis) => Sort(a, axis, NPY_SORTKIND.NPY_QUICKSORT);

        /// <summary>
        ///     np.sort with a resolved kind (<see cref="SortKindArguments.Resolve"/>): copy(order='K') then sort in place
        ///     (axis=null flattens), as NumPy's <c>fromnumeric.sort</c> does. Over <c>S</c> / <c>U</c> text the result's
        ///     lines are gathered sorted straight from <paramref name="a"/> into a fresh array of the copy's layout
        ///     (<see cref="TextSort.SortLinesInto"/>), which skips the copy NumPy sorts.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The kind. It changes the answer only where NumPy's does: the stable kind's float value sort keeps
        ///     every NaN's bits (the default kind writes NumPy's canonical NaN), and <c>StringDType</c> refuses the heapsort
        ///     kind.</param>
        /// <returns>The sorted copy.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (a 0-d input refuses every axis: NumPy's sort does not ravel
        ///     it first).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>),
        ///     after the axis check.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements holding an NA whose NA object is
        ///     neither NaN-like nor a string (<c>Cannot compare null that is not a nan-like value</c>).</exception>
        [NDScoped]
        public static NDArray Sort(NDArray a, int? axis, NPY_SORTKIND kind)
        {
            if (Strings.NDStringUfuncs.IsVString(a))
            {
                // StringDType: the sort loop's refusal of the heapsort kind follows the axis check (PyArray_Sort resolves
                // the loop after check_and_adjust_axis); then the copy is sorted with NumPy's NA ordering (VStringSort).
                Strings.NDStringUfuncs.ValidateReduceAxis("sort", a.ndim, axis);
                if (kind == NPY_SORTKIND.NPY_HEAPSORT)
                    throw UnsupportedSortKind();
                return SortVString(a, axis, kind == NPY_SORTKIND.NPY_STABLESORT);
            }
            if (Strings.NDStringUfuncs.IsString(a))
                return SortText(a, axis);

            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                SortInPlace(flat, 0, kind);
                return flat;
            }
            var res = a.copy('K');
            SortInPlace(res, NormalizeAxis(axis.Value, a.ndim), kind);
            return res;
        }

        /// <summary>ndarray.sort: sorts <paramref name="a"/> in place along the axis.</summary>
        /// <param name="a">The array, sorted in place.</param>
        /// <param name="axis">The axis, or null (NumSharp's flatten-in-place extension).</param>
        /// <exception cref="ValueError">The array is read-only (<c>sort array is read-only</c>), or a <c>StringDType</c> line
        ///     of two or more elements holds an NA whose NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static void SortInPlace(NDArray a, int? axis) => SortInPlace(a, axis, NPY_SORTKIND.NPY_QUICKSORT);

        /// <summary>
        ///     ndarray.sort with a resolved kind: sorts <paramref name="a"/> in place along the axis, in NumPy's
        ///     <c>PyArray_Sort</c> order — the axis, then writeability, then (for <c>StringDType</c>) the kind, then the sort,
        ///     which leaves lines of at most one element as they are.
        /// </summary>
        /// <param name="a">The array, sorted in place.</param>
        /// <param name="axis">The axis, or null (NumSharp's flatten-in-place extension).</param>
        /// <param name="kind">The kind (see <see cref="Sort(NDArray, int?, NPY_SORTKIND)"/>).</param>
        /// <exception cref="ValueError">The array is read-only (<c>sort array is read-only</c>), or a <c>StringDType</c> line
        ///     of two or more elements holds an NA whose NA object is neither NaN-like nor a string (<c>Cannot compare null
        ///     that is not a nan-like value</c>: the lines before it are sorted, that line is left as it was where NumPy
        ///     leaves it in the order its failing comparisons produced).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>).</exception>
        [NDScoped]
        public static void SortInPlace(NDArray a, int? axis, NPY_SORTKIND kind)
        {
            if (Strings.NDStringUfuncs.IsVString(a))
            {
                // StringDType: NumPy's checks in PyArray_Sort's order (axis, writeability, the loop's kind), then every
                // line's cells permuted in place (VStringSort; a line of at most one element is left as it is).
                Strings.NDStringUfuncs.ValidateReduceAxis("sort", a.ndim, axis);
                if (!a.Shape.IsWriteable)
                    throw new ValueError("sort array is read-only");
                if (kind == NPY_SORTKIND.NPY_HEAPSORT)
                    throw UnsupportedSortKind();
                bool stable = kind == NPY_SORTKIND.NPY_STABLESORT;
                if (axis == null)
                {
                    // NumSharp's flatten-in-place extension: one line over the whole (contiguous) buffer.
                    VStringSort.SortLinesInPlace(a.reshape(a.size), 0, stable);
                    return;
                }
                VStringSort.SortLinesInPlace(a, NormalizeAxis(axis.Value, a.ndim), stable);
                return;
            }
            if (Strings.NDStringUfuncs.IsString(a))
            {
                // S / U: NumPy's PyArray_Sort checks the axis, then writeability, then sorts every line by code points
                // (_new_sortlike leaves a line of at most one element, or an empty array, as it is).
                Strings.NDStringUfuncs.ValidateReduceAxis("sort", a.ndim, axis);
                if (!a.Shape.IsWriteable)
                    throw new ValueError("sort array is read-only");
                if (Strings.NDStringUfuncs.ShapeDecides("sort", a, axis))
                    return;
                if (axis == null)
                {
                    // NumSharp's flatten-in-place extension: one line over the whole (contiguous) buffer.
                    TextSort.SortLines(a.reshape(a.size), 0);
                    return;
                }
                TextSort.SortLines(a, NormalizeAxis(axis.Value, a.ndim));
                return;
            }
            if (!a.Shape.IsWriteable)
                throw new ValueError("sort array is read-only"); // NumPy 2.4.2 verbatim (probed)
            if (axis == null)
            {
                // In-place flatten-sort only well-defined for contiguous; NumPy raises otherwise.
                // The flatten view is a transient wrapper — scope-reclaimed once the sort lands.
                SortInPlace(a.reshape(a.size), 0, kind);
                return;
            }
            SortInPlace(a, NormalizeAxis(axis.Value, a.ndim), kind);
        }

        /// <summary>
        ///     np.argsort: int64 indices that sort <paramref name="a"/> along <paramref name="axis"/> (same shape;
        ///     <c>null</c> flattens). A 0-d input ravels to one dimension first, as NumPy's <c>PyArray_ArgSort</c> does through
        ///     <c>PyArray_CheckAxis</c>: <c>np.argsort(np.array(5))</c> is <c>[0]</c>, and an axis other than 0 / -1 reports
        ///     dimension 1.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>A fresh int64 array of indices.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (for a 0-d input, outside rank 1).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements holding an NA whose NA object is
        ///     neither NaN-like nor a string.</exception>
        public static NDArray ArgSort(NDArray a, int? axis) => ArgSort(a, axis, NPY_SORTKIND.NPY_QUICKSORT);

        /// <summary>
        ///     np.argsort with a resolved kind. Over <c>S</c> / <c>U</c> text the default and heapsort kinds give NumPy's
        ///     introsort permutation (ties in its order) and the stable kind the stable one (<see cref="TextSort.ArgSortLines"/>);
        ///     over <c>StringDType</c> text the default kind gives NumPy's introsort permutation and the stable kind the stable
        ///     one, the NaN-like NAs last (<see cref="VStringSort.ArgSortLines"/>); NumSharp's numeric argsort is one stable
        ///     order for every kind.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The kind.</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (for a 0-d input, outside rank 1).</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort kind</c>):
        ///     NumPy's argsort resolves its loop BEFORE it checks the axis.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> line of two or more elements holding an NA whose NA object is
        ///     neither NaN-like nor a string (<c>Cannot compare null that is not a nan-like value</c>).</exception>
        [NDScoped]
        public static NDArray ArgSort(NDArray a, int? axis, NPY_SORTKIND kind)
        {
            // PyArray_ArgSort resolves the dtype's argsort loop (StringDType refuses the heapsort kind) before
            // PyArray_CheckAxis.
            if (kind == NPY_SORTKIND.NPY_HEAPSORT && Strings.NDStringUfuncs.IsVString(a))
                throw UnsupportedSortKind();
            // PyArray_CheckAxis ravels a 0-d input before the axis is checked (the reshape is a view; the scope reclaims it).
            if (a.ndim == 0)
                a = a.reshape(1);
            if (Strings.NDStringUfuncs.IsVString(a))
                return ArgSortVString(a, axis, kind == NPY_SORTKIND.NPY_STABLESORT);
            if (Strings.NDStringUfuncs.IsString(a))
                return ArgSortText(a, axis, kind == NPY_SORTKIND.NPY_STABLESORT);
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, Shape.Vector(a.size), false);
                ArgSortInto(flat, outFlat, 0);
                return outFlat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])a.Shape.dimensions.Clone()), false);
            ArgSortInto(src, ret, ax);
            return ret;
        }

        /// <summary>
        ///     np.sort over <c>S</c> / <c>U</c> text: NumPy's axis check (a 0-d input refuses every axis; <c>axis=None</c>
        ///     sorts the C-order flattening), then every line gathered sorted into a fresh array laid out as
        ///     <c>copy(order='K')</c> lays out the input — the layout NumPy's sorted copy has.
        /// </summary>
        /// <param name="a">The text array (only read).</param>
        /// <param name="axis">The axis, or null.</param>
        /// <returns>The sorted copy.</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <remarks>Internal: <see cref="AxisPartition"/>'s text route is this sort (a string has no partition function, so
        ///     NumPy sorts each line).</remarks>
        internal static NDArray SortText(NDArray a, int? axis)
        {
            Strings.NDStringUfuncs.ValidateReduceAxis("sort", a.ndim, axis);
            if (axis == null)
            {
                // ravel is a view of a C-contiguous input and a C-order copy of any other; the sort only reads it.
                var line = a.ravel();
                var flat = new NDArray(a.dtype, Shape.Vector(a.size), false);
                TextSort.SortLinesInto(line, flat, 0);
                return flat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var res = new NDArray(a.dtype, a.Shape.KeepOrder(), false);
            TextSort.SortLinesInto(a, res, ax);
            return res;
        }

        /// <summary>
        ///     np.argsort over <c>S</c> / <c>U</c> text (a 0-d input already ravelled): NumPy's axis check, then every line's
        ///     indices — a line of at most one element gets zeros, as <c>_new_argsortlike</c> writes them.
        /// </summary>
        /// <param name="a">The text array (only read; any layout).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <remarks>Internal: <see cref="AxisPartition"/>'s text route is this argsort with the default kind (NumPy's generic
        ///     introsort fallback compares as the string introsort does).</remarks>
        internal static NDArray ArgSortText(NDArray a, int? axis, bool stable)
        {
            Strings.NDStringUfuncs.ValidateReduceAxis("argsort", a.ndim, axis);
            if (axis == null)
            {
                var line = a.ravel();
                var outFlat = new NDArray(NPTypeCode.Int64, Shape.Vector(a.size), false);
                TextSort.ArgSortLines(line, outFlat, 0, stable);
                return outFlat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var ret = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])a.Shape.dimensions.Clone()), false);
            TextSort.ArgSortLines(a, ret, ax, stable);
            return ret;
        }

        /// <summary>
        ///     np.sort over <c>StringDType</c> text once the axis and the kind were checked: NumPy's
        ///     <c>a.flatten()</c> (<c>axis=None</c>) or <c>a.copy(order='K')</c>, then every line of the copy sorted in place
        ///     (<see cref="VStringSort.SortLinesInPlace"/>: the copy's cells permuted, NumPy's NA order).
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array (only read).</param>
        /// <param name="axis">The axis (already validated), or null for the flattened array.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>The sorted copy (its own allocator).</returns>
        /// <exception cref="ValueError">A line of two or more elements holding a None-like NA (NumPy's comparison refusal).</exception>
        /// <remarks>Internal: <see cref="AxisPartition"/>'s <c>StringDType</c> route is this sort with the default kind
        ///     (NumPy's partition falls back to its quicksort).</remarks>
        internal static NDArray SortVString(NDArray a, int? axis, bool stable)
        {
            if (axis == null)
            {
                // flatten is a C-order copy with its own allocator, sorted in place below.
                var flat = a.flatten();
                VStringSort.SortLinesInPlace(flat, 0, stable);
                return flat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var res = a.copy('K');
            VStringSort.SortLinesInPlace(res, ax, stable);
            return res;
        }

        /// <summary>
        ///     np.argsort over <c>StringDType</c> text (a 0-d input already ravelled): NumPy's axis check, then every line's
        ///     indices (<see cref="VStringSort.ArgSortLines"/>) — a line of one element gets 0, as <c>_new_argsortlike</c>
        ///     writes it.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array (only read; any layout).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>A fresh, C-contiguous int64 array of indices.</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">A line of two or more elements holding a None-like NA.</exception>
        /// <remarks>Internal: <see cref="AxisPartition"/>'s <c>StringDType</c> route is this argsort with the default kind,
        ///     and <c>np.unique</c>'s sort path uses it.</remarks>
        internal static NDArray ArgSortVString(NDArray a, int? axis, bool stable)
        {
            Strings.NDStringUfuncs.ValidateReduceAxis("argsort", a.ndim, axis);
            if (axis == null)
            {
                // ravel is a view of a C-contiguous input and a C-order copy of any other; the sort only reads it.
                var line = a.ravel();
                var outFlat = new NDArray(NPTypeCode.Int64, Shape.Vector(a.size), false);
                VStringSort.ArgSortLines(line, outFlat, 0, stable);
                return outFlat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var ret = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])a.Shape.dimensions.Clone()), false);
            VStringSort.ArgSortLines(a, ret, ax, stable);
            return ret;
        }

        /// <summary>
        ///     NumPy's refusal of the heapsort kind by a <c>StringDType</c> sort or argsort (<c>stringdtype_get_sort_loop</c>
        ///     accepts only the default and the stable kinds).
        /// </summary>
        /// <returns>The exception (not thrown).</returns>
        private static RuntimeError UnsupportedSortKind() => new RuntimeError("unsupported sort kind");

        /// <summary>
        ///     The sort family's axis check — NumPy's <c>check_and_adjust_axis</c> in <c>PyArray_Sort</c> /
        ///     <c>PyArray_ArgSort</c> / <c>PyArray_Partition</c> / <c>PyArray_LexSort</c>.
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="ndim">The rank it addresses (the caller passes 1 for a 0-d input NumPy ravels first).</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndim, ndim)</c>.</exception>
        internal static int NormalizeAxis(int axis, int ndim) => DefaultEngine.check_and_adjust_axis(ndim, axis);

        // ============================ in-place line sort ============================

        /// <summary>
        /// Radix key width (bytes) the line kernel needs for <paramref name="tc"/>, or 0 for the
        /// scalar BCL path (Half/Complex/Decimal sort their own buffer). 8 for the 8-byte numeric
        /// dtypes (Int64/UInt64/Double), else 4 (1/2/4-byte ints, Single via its float32 key).
        /// Drives which scratch buffers <see cref="SortInPlace"/>/<see cref="ArgSortInto"/> allocate:
        /// the old code allocated BOTH the u32 and u64 double-buffers (plus the histogram) for every
        /// dtype, so a 10M-element int32 sort allocated 160 MB of u64 buffers it never touched — and
        /// the CLR zero-fills new arrays, ≈16 ms of pure memset wasted per large sort.
        /// </summary>
        private static int KeyWidth(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Double => 8,
            NPTypeCode.Half or NPTypeCode.Complex or NPTypeCode.Decimal => 0,
            _ => 4,
        };

        /// <summary>
        ///     The radix key width (bytes) the ARGSORT line kernel of <paramref name="tc"/> needs, or 0 for the scalar
        ///     comparison path (decimal only): <see cref="KeyWidth"/>, except float16 (a 16-bit key in the 4-byte core,
        ///     <see cref="ArgLineF16"/>) and complex128 (8-byte keys, <see cref="ArgLineC128"/>). Their VALUE sorts stay on
        ///     the comparison sort: a value sort rebuilds the values from their keys, and these keys fold <c>-0.0</c> onto
        ///     <c>+0.0</c> and every NaN onto its class.
        /// </summary>
        /// <param name="tc">The element type.</param>
        /// <returns>4 or 8 for a radix argsort, 0 for the comparison argsort.</returns>
        private static int ArgKeyWidth(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Half => 4,
            NPTypeCode.Complex => 8,
            _ => KeyWidth(tc),
        };

        /// <summary>
        ///     Sorts every line of the numeric array <paramref name="target"/> along <paramref name="axis"/> in place.
        /// </summary>
        /// <param name="target">The array (writeable, already checked).</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="kind">The kind: only the stable kind changes the answer, keeping every float NaN's bits in the
        ///     sorted line's NaN tail (NumPy's stable kinds do not canonicalize; its default kind does).</param>
        /// <exception cref="NotSupportedException">A Half / Complex / Decimal line longer than <see cref="int.MaxValue"/>.</exception>
        private static void SortInPlace(NDArray target, int axis, NPY_SORTKIND kind)
        {
            long N = target.Shape.dimensions[axis];
            if (N <= 1 || target.size == 0) return;

            var tc = target.GetTypeCode;
            int elsize = tc.SizeOf();
            long axisStride = (long)target.Shape.strides[axis] * elsize; // byte stride along the sort axis

            // A line length is a `long`, so an axis may exceed int.MaxValue. The radix scratch is
            // therefore UNMANAGED (managed uint[]/ulong[] element counts cap at ~2^31); the histogram
            // is long* because one byte-bucket can hold up to N counts. The scalar BCL path
            // (Half/Complex/Decimal) sorts through managed Span/Array and so cannot exceed int range.
            int w = KeyWidth(tc);
            var ctx = new LineCtx { n = N, inStride = axisStride, outStride = axisStride, keepNaNBits = kind == NPY_SORTKIND.NPY_STABLESORT };

            if (w == 0)
            {
                if (N > int.MaxValue)
                    throw new NotSupportedException(
                        $"sort of a {tc} axis longer than int.MaxValue ({N}) is not supported: the scalar "
                        + "comparison sort uses managed buffers, which cap at int.MaxValue elements.");
                NDInnerLoopFunc scalarKern = GetSortKernel(tc);
                DriveAllButAxis(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, scalarKern, &ctx);
                return;
            }

            // single-pass radix builds ALL byte-histograms at once: nbytes·256 longs (8·256 for the
            // 8-byte path, 4·256 covers the ≤4-byte paths). BuildHist zeroes it, so a dirty buffer serves.
            int histLen = (w == 8 ? 8 : 4) * 256;
            // Pooled scratch (see ArgSortInto): fresh C-allocator columns are first-touch page faults on every call.
            long keyBytes = N * w, histBytes = (long)histLen * sizeof(long);
            void* pk = null, pt = null; long* pc = null;
            try
            {
                pk = (void*)SizeBucketedBufferPool.Take(keyBytes);
                pt = (void*)SizeBucketedBufferPool.Take(keyBytes);
                pc = (long*)SizeBucketedBufferPool.Take(histBytes);
                if (w == 4) { ctx.k32 = (uint*)pk; ctx.t32 = (uint*)pt; }
                else { ctx.k64 = (ulong*)pk; ctx.t64 = (ulong*)pt; }
                ctx.count = pc;
                NDInnerLoopFunc kern = GetSortKernel(tc);
                DriveAllButAxis(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, kern, &ctx);
            }
            finally
            {
                if (pk != null) SizeBucketedBufferPool.Return((IntPtr)pk, keyBytes);
                if (pt != null) SizeBucketedBufferPool.Return((IntPtr)pt, keyBytes);
                if (pc != null) SizeBucketedBufferPool.Return((IntPtr)pc, histBytes);
            }
        }

        /// <summary>
        /// Write the indices that sort every line of <paramref name="src"/> along
        /// <paramref name="axis"/> into <paramref name="dst"/> (stable radix for the integer, float, float16 and complex
        /// dtypes with NaN last; a managed comparison sort for Decimal).
        /// </summary>
        /// <param name="src">The C-contiguous array to argsort (read only).</param>
        /// <param name="dst">An int64 array with <paramref name="src"/>'s dims; every line along <paramref name="axis"/> is overwritten.</param>
        /// <param name="axis">The already-normalised sort axis.</param>
        /// <exception cref="NotSupportedException">A Decimal line is longer than <see cref="int.MaxValue"/> (the comparison sort uses managed buffers).</exception>
        /// <remarks>
        /// Radix scratch is unmanaged and sized for ONE line; the line kernels reuse it for every line,
        /// sorting <c>LineCtx.n</c> elements regardless of the iterator's per-call count. That is why
        /// an input with no lines (<c>size == 0</c>) must return before the drive: see the guard below.
        /// </remarks>
        private static void ArgSortInto(NDArray src, NDArray dst, int axis)
        {
            long N = src.Shape.dimensions[axis];
            var tc = src.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new LineCtx
            {
                n = N,
                inStride = (long)src.Shape.strides[axis] * elsize,
                outStride = (long)dst.Shape.strides[axis] * sizeof(long),
            };
            // An empty axis OR zero lines (another dimension is 0, e.g. (0,3) along the last axis):
            // dst has src's dims, so it is empty too and there is nothing to write. The size guard is
            // load-bearing, not an optimisation: the line kernels ignore the iterator's per-call count
            // and always write a full line of N indices, so letting a zero-line drive through wrote N
            // int64s into dst's ZERO-byte buffer — native heap corruption (Windows killed the process
            // with 0xC0000374 STATUS_HEAP_CORRUPTION; glibc stayed silent). SortInPlace and
            // AxisPartition already carried this guard; argsort was the one sibling without it.
            if (N == 0 || src.size == 0) return;

            // Radix scratch is UNMANAGED so a line may exceed int.MaxValue (the index column idx/it
            // and the key column are both element-sized and would otherwise cap at ~2^31). The scalar
            // BCL argsort path (Decimal, w==0) allocates its own managed buffers and so cannot exceed
            // int range. float16 and complex128 argsort by radix too (ArgKeyWidth), while their value
            // sorts keep the comparison sort.
            int w = ArgKeyWidth(tc);

            if (w == 0)
            {
                if (N > int.MaxValue)
                    throw new NotSupportedException(
                        $"argsort of a {tc} axis longer than int.MaxValue ({N}) is not supported: the scalar "
                        + "comparison sort uses managed buffers, which cap at int.MaxValue elements.");
                NDInnerLoopFunc scalarKern = GetArgSortKernel(tc);
                DriveAllButAxis(new[] { src, dst }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, scalarKern, &ctx);
                return;
            }

            // single-pass radix builds ALL byte-histograms at once: nbytes·256 longs. BuildHist zeroes it.
            int histLen = (w == 8 ? 8 : 4) * 256;
            // The scratch comes from the buffer pool, not straight from the C allocator: a line of 100K float64 needs four
            // 800 KB columns, which the allocator serves as fresh pages on every call, and touching them was a large part of
            // the argsort (measured A/B, one process each, best of two runs: 100K float64 3.9 -> 2.8 ms, 1M 45 -> 39 ms;
            // 100K int32 1.8 -> 1.3 ms). The pool hands the same warm buffers back to the next sort. Every Return passes the
            // size its Take got (the pool's contract).
            long keyBytes = N * w, idxBytes = N * sizeof(long), histBytes = (long)histLen * sizeof(long);
            void* pk = null, pt = null; long* pi = null, pit = null, pc = null;
            try
            {
                pk = (void*)SizeBucketedBufferPool.Take(keyBytes);
                pt = (void*)SizeBucketedBufferPool.Take(keyBytes);
                pi = (long*)SizeBucketedBufferPool.Take(idxBytes);
                pit = (long*)SizeBucketedBufferPool.Take(idxBytes);
                pc = (long*)SizeBucketedBufferPool.Take(histBytes);
                if (w == 4) { ctx.k32 = (uint*)pk; ctx.t32 = (uint*)pt; }
                else { ctx.k64 = (ulong*)pk; ctx.t64 = (ulong*)pt; }
                ctx.idx = pi; ctx.it = pit; ctx.count = pc;
                NDInnerLoopFunc kern = GetArgSortKernel(tc);
                DriveAllButAxis(new[] { src, dst }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, kern, &ctx);
            }
            finally
            {
                if (pk != null) SizeBucketedBufferPool.Return((IntPtr)pk, keyBytes);
                if (pt != null) SizeBucketedBufferPool.Return((IntPtr)pt, keyBytes);
                if (pi != null) SizeBucketedBufferPool.Return((IntPtr)pi, idxBytes);
                if (pit != null) SizeBucketedBufferPool.Return((IntPtr)pit, idxBytes);
                if (pc != null) SizeBucketedBufferPool.Return((IntPtr)pc, histBytes);
            }
        }

        /// <summary>NumPy IterAllButAxis: iterate every axis EXCEPT <paramref name="axis"/> (dropped via
        /// op_axes so it can't coalesce); the kernel receives each operand's line start per call.
        /// Internal: <see cref="AxisPartition"/> drives its partition/argpartition line kernels through
        /// the same loop (NumPy routes _new_sortlike AND _new_partitionlike through one drive too).</summary>
        /// <param name="ops">The operands whose lines are visited together (all share <c>ops[0]</c>'s dims).</param>
        /// <param name="flags">Per-operand NDIter access flags (READONLY / WRITEONLY / READWRITE).</param>
        /// <param name="axis">The normalised line axis — the one axis the iterator does NOT walk.</param>
        /// <param name="kern">The line kernel, called once per line with each operand's line start.</param>
        /// <param name="aux">The kernel's context (a <c>LineCtx*</c> or the partition equivalent).</param>
        /// <remarks>
        /// A zero-size <c>ops[0]</c> returns without calling <paramref name="kern"/>: there are no lines,
        /// and the line kernels would otherwise sort/write a full line into an empty buffer.
        /// </remarks>
        [NDScoped]
        internal static void DriveAllButAxis(NDArray[] ops, NDIterPerOpFlags[] flags, int axis, NDInnerLoopFunc kern, void* aux)
        {
            // Scope: the 1-D (1, N) promotion views below are pure iteration aids over the caller's
            // operands — without eager reclamation a dropped view held the result buffer's last
            // extra ref until the finalizer, pinning every 1-D sort/argsort/partition result.

            // Defence in depth for every caller: a zero-size operand means there are no lines to
            // visit, but the iterator still makes a kernel call for it, and the line kernels ignore
            // that call's count and sort/write a WHOLE line (LineCtx.n elements). Returning here keeps
            // an unguarded caller from overrunning a zero-byte buffer. The callers also check size
            // themselves; this only catches one that forgets (argsort did — see ArgSortInto).
            if (ops[0].size == 0)
                return;

            int ndim = ops[0].ndim;

            // 1-D input drops its only axis -> a 0-dimensional all-but-axis iterator. Our NDIter
            // mis-drives that degenerate shape as `size` single-element iterations, and because each
            // line kernel re-sorts the whole line (it walks by LineCtx.n, ignoring the per-call count)
            // the total work is N x O(N) = O(N^2). Promote every operand to a (1, N) memory-sharing
            // view (expand_dims aliases storage, so an in-place sort still mutates the original, and a
            // leading size-1 axis leaves the data-axis stride -> LineCtx in/out strides unchanged): now
            // exactly one axis (size 1) is kept, so the iterator makes exactly one line kernel call.
            if (ndim == 1)
            {
                var promoted = new NDArray[ops.Length];
                for (int i = 0; i < ops.Length; i++) promoted[i] = np.expand_dims(ops[i], 0);
                ops = promoted;
                axis = 1;
                ndim = 2;
            }

            var kept = new int[ndim - 1];
            for (int d = 0, w = 0; d < ndim; d++) if (d != axis) kept[w++] = d;
            var opAxes = new int[ops.Length][];
            for (int i = 0; i < ops.Length; i++) opAxes[i] = kept;

            // A fixed-width string operand has no type-code element size (its width is its descriptor's): the descriptor
            // construction sizes it, a null entry keeping each operand's own descriptor (no cast).
            DType[] descrs = null;
            var globalFlags = NDIterGlobalFlags.None;
            foreach (var op in ops)
            {
                if (op.typecode == NPTypeCode.String)
                {
                    descrs ??= new DType[ops.Length];
                    // A NEP 55 StringDType operand "holds references" (its cells point into an arena or the heap): NDIter
                    // refuses it without REFS_OK. The line kernels read its cells in place (no cast, no buffer).
                    if (op.Storage.IsVString)
                        globalFlags |= NDIterGlobalFlags.REFS_OK;
                }
            }
            var iter = NDIterRef.AdvancedNew(ops.Length, ops, globalFlags,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, flags, null, ndim - 1, opAxes, opDescrs: descrs);
            try { iter.ForEach(kern, aux); }
            finally { iter.Dispose(); }
        }

        // line-sort context carried via NDIter auxdata (no per-call captures/allocations)
        private struct LineCtx
        {
            public uint* k32; public uint* t32; public ulong* k64; public ulong* t64;
            public long* idx; public long* it; public long* count;
            public long inStride; public long outStride; public long n;

            /// <summary>The float value sorts' NaN tail: true keeps every NaN's bits in input order (NumPy's stable kinds),
            ///     false writes NumPy's canonical quiet NaN (its default kind).</summary>
            public bool keepNaNBits;
        }

        // ============================ generic line kernels ============================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortLine32<T, K>(byte* line, LineCtx* c) where T : unmanaged where K : struct, IKey32<T>
        {
            K k = default; long n = c->n; long s = c->inStride;
            for (long i = 0; i < n; i++) c->k32[i] = k.To(*(T*)(line + i * s));
            uint* r = RadixSort.SortU32(c->k32, c->t32, n, k.Bytes, c->count);
            for (long i = 0; i < n; i++) *(T*)(line + i * s) = k.From(r[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortLine64<T, K>(byte* line, LineCtx* c) where T : unmanaged where K : struct, IKey64<T>
        {
            K k = default; long n = c->n; long s = c->inStride;
            for (long i = 0; i < n; i++) c->k64[i] = k.To(*(T*)(line + i * s));
            ulong* r = RadixSort.SortU64(c->k64, c->t64, n, c->count);
            for (long i = 0; i < n; i++) *(T*)(line + i * s) = k.From(r[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgLine32<T, K>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where K : struct, IKey32<T>
        {
            K k = default; long n = c->n; long si = c->inStride, so = c->outStride;
            for (long i = 0; i < n; i++) { c->k32[i] = k.To(*(T*)(inLine + i * si)); c->idx[i] = i; }
            long* r = RadixSort.ArgSortU32(c->k32, c->t32, c->idx, c->it, n, k.Bytes, c->count);
            for (long i = 0; i < n; i++) *(long*)(outLine + i * so) = r[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgLine64<T, K>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where K : struct, IKey64<T>
        {
            K k = default; long n = c->n; long si = c->inStride, so = c->outStride;
            for (long i = 0; i < n; i++) { c->k64[i] = k.To(*(T*)(inLine + i * si)); c->idx[i] = i; }
            long* r = RadixSort.ArgSortU64(c->k64, c->t64, c->idx, c->it, n, c->count);
            for (long i = 0; i < n; i++) *(long*)(outLine + i * so) = r[i];
        }

        // ----- float radix with NaN-last partition -----

        /// <summary>
        ///     The NaN NumPy's float32 sort writes: the positive quiet NaN <c>0x7FC00000</c>. NumPy's default sort
        ///     (x86-simd-sort for 'quicksort' / 'heapsort') replaces every NaN of a line longer than one element —
        ///     negative, signalling or payload-bearing — with <c>std::numeric_limits&lt;float&gt;::quiet_NaN()</c> after
        ///     sorting. .NET's <see cref="float.NaN"/> is the NEGATIVE <c>0xFFC00000</c>, so writing it gave every sorted
        ///     NaN the opposite sign of NumPy's (np.sort, ndarray.sort, and everything that sorts first — polyfromroots'
        ///     <c>roots.sort()</c> then carried the wrong sign through <c>-r</c> into every coefficient).
        /// </summary>
        private static readonly float NumPySortNaN32 = BitConverter.UInt32BitsToSingle(0x7FC00000U);

        /// <summary>
        ///     The NaN NumPy's float64 sort writes: the positive quiet NaN <c>0x7FF8000000000000</c> (see
        ///     <see cref="NumPySortNaN32"/>; .NET's <see cref="double.NaN"/> is the negative <c>0xFFF8…</c>).
        /// </summary>
        private static readonly double NumPySortNaN64 = BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL);

        /// <summary>
        ///     Sorts one float32 line in place: the non-NaN values by radix, then the NaN tail.
        /// </summary>
        /// <param name="line">The line's first element (byte address).</param>
        /// <param name="c">The line context: length, stride and the radix scratch buffers.</param>
        /// <remarks>
        ///     The default kind's tail is NumPy's canonical NaN (<see cref="NumPySortNaN32"/>), not the input NaNs: NumPy's
        ///     default kind rewrites them all. The stable kind (<see cref="LineCtx.keepNaNBits"/>) keeps each NaN's bits
        ///     in input order, as NumPy's stable sort (timsort with <c>npy::float_tag::less</c>, where two NaNs compare
        ///     equal) does. The NaN bits wait in the key buffer's unused top slots while the radix sorts the bottom ones:
        ///     the m keys fill slots [0, m) upward and the n - m NaNs fill [m, n) downward, so the two never meet. A
        ///     one-element line never reaches here (the driver leaves it untouched, as NumPy does).
        /// </remarks>
        private static void SortLineF32(byte* line, LineCtx* c)
        {
            long n = c->n; long s = c->inStride; long m = 0, q = 0;
            bool keep = c->keepNaNBits;
            for (long i = 0; i < n; i++)
            {
                float v = *(float*)(line + i * s);
                if (!float.IsNaN(v)) c->k32[m++] = FKey32(v);
                else if (keep) c->k32[n - 1 - q++] = BitConverter.SingleToUInt32Bits(v);
            }
            // The radix reads and writes slots [0, m) of k32 and t32 only, so the NaN bits above stay put.
            uint* r = RadixSort.SortU32(c->k32, c->t32, m, 4, c->count);
            for (long i = 0; i < m; i++) *(float*)(line + i * s) = FVal32(r[i]);
            if (keep)
            {
                for (long i = 0; i < n - m; i++) *(uint*)(line + (m + i) * s) = c->k32[n - 1 - i];
                return;
            }
            float nan = NumPySortNaN32;
            for (long i = m; i < n; i++) *(float*)(line + i * s) = nan;
        }

        /// <summary>
        ///     Sorts one float64 line in place: the non-NaN values by radix, then the NaN tail.
        /// </summary>
        /// <param name="line">The line's first element (byte address).</param>
        /// <param name="c">The line context: length, stride and the radix scratch buffers.</param>
        /// <remarks>The default kind's tail is NumPy's canonical NaN (<see cref="NumPySortNaN64"/>), the stable kind's the
        ///     input NaNs in input order; see <see cref="SortLineF32"/>.</remarks>
        private static void SortLineF64(byte* line, LineCtx* c)
        {
            long n = c->n; long s = c->inStride; long m = 0, q = 0;
            bool keep = c->keepNaNBits;
            for (long i = 0; i < n; i++)
            {
                double v = *(double*)(line + i * s);
                if (!double.IsNaN(v)) c->k64[m++] = FKey64(v);
                else if (keep) c->k64[n - 1 - q++] = BitConverter.DoubleToUInt64Bits(v);
            }
            ulong* r = RadixSort.SortU64(c->k64, c->t64, m, c->count);
            for (long i = 0; i < m; i++) *(double*)(line + i * s) = FVal64(r[i]);
            if (keep)
            {
                for (long i = 0; i < n - m; i++) *(ulong*)(line + (m + i) * s) = c->k64[n - 1 - i];
                return;
            }
            double nan = NumPySortNaN64;
            for (long i = m; i < n; i++) *(double*)(line + i * s) = nan;
        }
        /// <summary>
        ///     Argsorts one float32 line: the non-NaN indices by a stable radix over <see cref="FArgKey32"/>, then the NaN
        ///     indices in input order.
        /// </summary>
        /// <param name="inLine">The input line's first element (byte address).</param>
        /// <param name="outLine">The int64 index line's first element (byte address).</param>
        /// <param name="c">The line context: length, strides and the radix scratch buffers.</param>
        /// <remarks>
        ///     The order is NumPy's stable argsort (<c>kind='stable'</c>, timsort with <c>npy::float_tag::less</c>):
        ///     equal values, <c>-0.0</c> and <c>+0.0</c> included, keep their input order, and every NaN comes last in
        ///     input order. The radix key therefore folds <c>-0.0</c> onto <c>+0.0</c>; the raw bit key put every
        ///     <c>-0.0</c> first (<c>np.argsort([0.0, -0.0])</c> was <c>[1, 0]</c>, NumPy <c>[0, 1]</c>), which
        ///     <c>np.lexsort</c>'s tie order and the numpy.ma set operations inherited.
        /// </remarks>
        private static void ArgLineF32(byte* inLine, byte* outLine, LineCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride; long m = 0;
            for (long i = 0; i < n; i++) { float v = *(float*)(inLine + i * si); if (!float.IsNaN(v)) { c->k32[m] = FArgKey32(v); c->idx[m] = i; m++; } }
            long* r = RadixSort.ArgSortU32(c->k32, c->t32, c->idx, c->it, m, 4, c->count);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = r[i];
            long q = m; for (long i = 0; i < n; i++) if (float.IsNaN(*(float*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        /// <summary>
        ///     Argsorts one float64 line: the non-NaN indices by a stable radix over <see cref="FArgKey64"/>, then the NaN
        ///     indices in input order (see <see cref="ArgLineF32"/> for the order this reproduces).
        /// </summary>
        /// <param name="inLine">The input line's first element (byte address).</param>
        /// <param name="outLine">The int64 index line's first element (byte address).</param>
        /// <param name="c">The line context: length, strides and the radix scratch buffers.</param>
        private static void ArgLineF64(byte* inLine, byte* outLine, LineCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride; long m = 0;
            for (long i = 0; i < n; i++) { double v = *(double*)(inLine + i * si); if (!double.IsNaN(v)) { c->k64[m] = FArgKey64(v); c->idx[m] = i; m++; } }
            long* r = RadixSort.ArgSortU64(c->k64, c->t64, c->idx, c->it, m, c->count);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = r[i];
            long q = m; for (long i = 0; i < n; i++) if (double.IsNaN(*(double*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        /// <summary>
        ///     Argsorts one float16 line: the non-NaN indices by a stable two-byte radix over <see cref="HArgKey16"/>, then
        ///     the NaN indices in input order. That is NumPy's stable argsort order (timsort with <c>npy::half_tag::less</c>:
        ///     NaNs last and equal to each other, <c>-0.0</c> equal to <c>+0.0</c>, ties in input order), the order the
        ///     comparison argsort this replaces gave as well (<see cref="HalfCmp"/>, ties by index).
        /// </summary>
        /// <param name="inLine">The input line's first element (byte address).</param>
        /// <param name="outLine">The int64 index line's first element (byte address).</param>
        /// <param name="c">The line context: length, strides and the 4-byte radix scratch buffers.</param>
        /// <remarks>The comparison argsort called an interface comparer per comparison through a managed index array: 23 ms
        ///     for 100K elements, about twice NumPy's scalar argsort.</remarks>
        private static void ArgLineF16(byte* inLine, byte* outLine, LineCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride; long m = 0;
            for (long i = 0; i < n; i++)
            {
                ushort b = *(ushort*)(inLine + i * si);
                // A NaN has every exponent bit and a mantissa bit set: the magnitude bits exceed +inf's 0x7C00.
                if ((b & 0x7FFF) <= 0x7C00) { c->k32[m] = HArgKey16(b); c->idx[m] = i; m++; }
            }
            long* r = RadixSort.ArgSortU32(c->k32, c->t32, c->idx, c->it, m, 2, c->count);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = r[i];
            long q = m;
            for (long i = 0; i < n; i++)
                if ((*(ushort*)(inLine + i * si) & 0x7FFF) > 0x7C00) *(long*)(outLine + (q++) * so) = i;
        }

        /// <summary>
        ///     The argsort radix key of a non-NaN float16's bits: monotonic in the value (the sign-magnitude bits mapped to
        ///     an unsigned order, as <see cref="FKey32"/> maps a float32's), with <c>-0.0</c> folded onto <c>+0.0</c> so the
        ///     two zeros compare equal and the stable radix keeps their input order.
        /// </summary>
        /// <param name="b">The bits of a non-NaN value (<c>(b &amp; 0x7FFF) &lt;= 0x7C00</c>).</param>
        /// <returns>The key in the low 16 bits: <c>-inf</c> is <c>0x03FF</c>, both zeros <c>0x8000</c>, <c>+inf</c>
        ///     <c>0xFC00</c>.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static uint HArgKey16(ushort b)
        {
            if ((b & 0x7FFF) == 0) return 0x8000u;                            // -0.0 and +0.0: one key
            return (b & 0x8000) != 0 ? (uint)(ushort)~b : (uint)(b | 0x8000);  // negative: flip every bit; positive: set the sign bit
        }

        /// <summary>
        ///     Argsorts one complex128 line in NumPy's stable argsort order (timsort with <c>CDOUBLE_LT</c>, the order of
        ///     <see cref="ComplexCmp"/>; equal values in input order) by radix. <c>CDOUBLE_LT</c> is lexicographic on
        ///     (class, first key, second key): the values without a NaN part first, by (real, imaginary); then the
        ///     <c>R+nanj</c> values by R; then the <c>nan+Rj</c> values by R; then the <c>nan+nanj</c> values, all equal. So
        ///     the line is partitioned by class (stably), the first class is radix-sorted by its imaginary key and then,
        ///     stably, by its real key (LSD), the second and third by their one key, and the fourth keeps its input order.
        ///     The keys are <see cref="FArgKey64"/>: both zeros compare equal in either part, as in <c>CDOUBLE_LT</c>.
        /// </summary>
        /// <param name="inLine">The input line's first element (byte address).</param>
        /// <param name="outLine">The int64 index line's first element (byte address).</param>
        /// <param name="c">The line context: length, strides and the 8-byte radix scratch buffers.</param>
        /// <remarks>
        ///     The comparison argsort this replaces (<see cref="ComplexCmp"/> through an interface comparer and a managed index
        ///     array, ties by index) gave the same indices at about half NumPy's speed. The classes occupy consecutive
        ///     regions of the key and index buffers, and each radix touches only its own region, so the four never overlap.
        ///     The first class's second radix reads each element's real part again through the first radix's permutation.
        /// </remarks>
        private static void ArgLineC128(byte* inLine, byte* outLine, LineCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride;

            // Class sizes: A (no NaN part), B (R+nanj), C (nan+Rj); D (nan+nanj) is the rest.
            long nA = 0, nB = 0, nC = 0;
            for (long i = 0; i < n; i++)
            {
                double* z = (double*)(inLine + i * si);
                bool reNaN = double.IsNaN(z[0]), imNaN = double.IsNaN(z[1]);
                if (!reNaN) { if (!imNaN) nA++; else nB++; }
                else if (!imNaN) nC++;
            }
            long offB = nA, offC = nA + nB, offD = nA + nB + nC;

            // Stable partition: each element's key for its class's first radix, its index as the payload.
            long a = 0, b = offB, cc = offC, d = offD;
            for (long i = 0; i < n; i++)
            {
                double* z = (double*)(inLine + i * si);
                double re = z[0], im = z[1];
                if (!double.IsNaN(re))
                {
                    if (!double.IsNaN(im)) { c->k64[a] = FArgKey64(im); c->idx[a++] = i; }     // A: the imaginary key first (LSD)
                    else { c->k64[b] = FArgKey64(re); c->idx[b++] = i; }                       // B: the real part
                }
                else if (!double.IsNaN(im)) { c->k64[cc] = FArgKey64(im); c->idx[cc++] = i; }  // C: the imaginary part
                else c->idx[d++] = i;                                                         // D: input order
            }

            if (nA > 0)
            {
                long* r1 = RadixSort.ArgSortU64(c->k64, c->t64, c->idx, c->it, nA, c->count);
                // The second radix carries r1 as its payload and uses the other index buffer as its scratch; the key
                // buffers' first nA slots are free again (only r1's order matters now).
                long* other = r1 == c->idx ? c->it : c->idx;
                for (long j = 0; j < nA; j++) c->k64[j] = FArgKey64(*(double*)(inLine + r1[j] * si));
                long* r2 = RadixSort.ArgSortU64(c->k64, c->t64, r1, other, nA, c->count);
                for (long j = 0; j < nA; j++) *(long*)(outLine + j * so) = r2[j];
            }
            if (nB > 0)
            {
                long* rB = RadixSort.ArgSortU64(c->k64 + offB, c->t64 + offB, c->idx + offB, c->it + offB, nB, c->count);
                for (long j = 0; j < nB; j++) *(long*)(outLine + (offB + j) * so) = rB[j];
            }
            if (nC > 0)
            {
                long* rC = RadixSort.ArgSortU64(c->k64 + offC, c->t64 + offC, c->idx + offC, c->it + offC, nC, c->count);
                for (long j = 0; j < nC; j++) *(long*)(outLine + (offC + j) * so) = rC[j];
            }
            for (long j = offD; j < n; j++) *(long*)(outLine + j * so) = c->idx[j];
        }

        /// <summary>
        ///     The argsort radix key of a non-NaN float32: <see cref="FKey32"/> of the value with <c>-0.0</c> folded onto
        ///     <c>+0.0</c>, so the two zeros compare EQUAL (NumPy's <c>less</c>) and the stable radix keeps their input order.
        /// </summary>
        /// <param name="v">A non-NaN value.</param>
        /// <returns>The monotonic unsigned key.</returns>
        /// <remarks>Only for argsort: a value sort rebuilds the values from their keys (<see cref="FVal32"/>), so folding
        /// there would turn every <c>-0.0</c> into <c>+0.0</c>.</remarks>
        [MethodImpl(OptimizeAndInline)] private static uint FArgKey32(float v) => FKey32(v == 0f ? 0f : v);

        /// <summary>
        ///     The argsort radix key of a non-NaN float64, <c>-0.0</c> folded onto <c>+0.0</c> (see <see cref="FArgKey32"/>).
        /// </summary>
        /// <param name="v">A non-NaN value.</param>
        /// <returns>The monotonic unsigned key.</returns>
        [MethodImpl(OptimizeAndInline)] private static ulong FArgKey64(double v) => FKey64(v == 0d ? 0d : v);

        [MethodImpl(OptimizeAndInline)] private static uint FKey32(float v) { uint b = BitConverter.SingleToUInt32Bits(v); return b ^ ((uint)((int)b >> 31) | 0x80000000u); }
        [MethodImpl(OptimizeAndInline)] private static float FVal32(uint k) { uint b = k ^ (((k >> 31) - 1) | 0x80000000u); return BitConverter.UInt32BitsToSingle(b); }
        [MethodImpl(OptimizeAndInline)] private static ulong FKey64(double v) { ulong b = BitConverter.DoubleToUInt64Bits(v); return b ^ ((ulong)((long)b >> 63) | 0x8000000000000000UL); }
        [MethodImpl(OptimizeAndInline)] private static double FVal64(ulong k) { ulong b = k ^ (((k >> 63) - 1) | 0x8000000000000000UL); return BitConverter.UInt64BitsToDouble(b); }

        // ----- scalar BCL introsort + exact NumPy comparators (Half / Complex / Decimal) -----
        private static void SortLineScalar<T, TCmp>(byte* line, LineCtx* c) where T : unmanaged where TCmp : struct, IComparer<T>
        {
            // c->n is a long, but this scalar (Half/Complex/Decimal) path is only reached after
            // SortInPlace has guarded N <= int.MaxValue (managed Span/Array are int-capped).
            int n = (int)c->n; long s = c->inStride;
            Span<T> buf = n <= 1024 ? stackalloc T[n] : new T[n];
            for (int i = 0; i < n; i++) buf[i] = *(T*)(line + i * s);
            buf.Sort(default(TCmp));
            for (int i = 0; i < n; i++) *(T*)(line + i * s) = buf[i];
        }
        private static void ArgLineScalar<T, TCmp>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where TCmp : struct, IComparer<T>
        {
            // Guarded to N <= int.MaxValue upstream (ArgSortInto), as SortLineScalar above.
            int n = (int)c->n; long si = c->inStride, so = c->outStride;
            var buf = new T[n]; var ix = new long[n];
            for (int i = 0; i < n; i++) { buf[i] = *(T*)(inLine + i * si); ix[i] = i; }
            Array.Sort(ix, new IndexedCmp<T, TCmp>(buf));
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = ix[i];
        }

        // stable indirect comparer for scalar argsort (ties -> ascending index)
        private sealed class IndexedCmp<T, TCmp> : IComparer<long> where TCmp : struct, IComparer<T>
        {
            private readonly T[] _v; public IndexedCmp(T[] v) { _v = v; }
            public int Compare(long i, long j) { int r = default(TCmp).Compare(_v[i], _v[j]); return r != 0 ? r : i.CompareTo(j); }
        }

        private readonly struct HalfCmp : IComparer<Half>
        {
            public int Compare(Half a, Half b)
            {
                float x = (float)a, y = (float)b;
                if (x < y) return -1; if (x > y) return 1;
                bool xn = float.IsNaN(x), yn = float.IsNaN(y);
                if (xn && yn) return 0; if (xn) return 1; if (yn) return -1; return 0;
            }
        }
        private readonly struct DecimalCmp : IComparer<decimal> { public int Compare(decimal a, decimal b) => a.CompareTo(b); }
        internal readonly struct ComplexCmp : IComparer<Complex>
        {
            // NumPy CDOUBLE_LT (npysort_common.h): lexicographic real-then-imag, any-NaN-part sorts last.
            // Internal: AxisPartition's Complex introselect comparator is this same ordering.
            public int Compare(Complex a, Complex b) { if (Lt(a, b)) return -1; if (Lt(b, a)) return 1; return 0; }
            private static bool Lt(Complex a, Complex b)
            {
                double ar = a.Real, ai = a.Imaginary, br = b.Real, bi = b.Imaginary;
                if (ar < br) return ai == ai || bi != bi;
                if (ar > br) return bi != bi && ai == ai;
                if (ar == br || (ar != ar && br != br)) return ai < bi || (bi != bi && ai == ai);
                return br != br;
            }
        }

        // ============================ dtype dispatch (one type-switch each) ============================

        private static NDInnerLoopFunc GetSortKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => SortLine32<byte, KBool>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => SortLine32<byte, KU8>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => SortLine32<sbyte, KI8>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => SortLine32<short, KI16>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => SortLine32<ushort, KU16>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => SortLine32<char, KChar>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => SortLine32<int, KI32>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => SortLine32<uint, KU32>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => SortLine64<long, KI64>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => SortLine64<ulong, KU64>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => SortLineF32((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => SortLineF64((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => SortLineScalar<Half, HalfCmp>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => SortLineScalar<Complex, ComplexCmp>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => SortLineScalar<decimal, DecimalCmp>((byte*)p[0], (LineCtx*)a),
            _ => throw new NotSupportedException($"sort not supported for dtype {tc}"),
        };

        private static NDInnerLoopFunc GetArgSortKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => ArgLine32<byte, KBool>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => ArgLine32<byte, KU8>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => ArgLine32<sbyte, KI8>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => ArgLine32<short, KI16>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => ArgLine32<ushort, KU16>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => ArgLine32<char, KChar>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => ArgLine32<int, KI32>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => ArgLine32<uint, KU32>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => ArgLine64<long, KI64>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => ArgLine64<ulong, KU64>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => ArgLineF32((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => ArgLineF64((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            // float16 and complex128 argsort by radix (ArgKeyWidth gives them radix scratch); their value sorts above stay
            // on the comparison sort.
            NPTypeCode.Half => static (p, s, c, a) => ArgLineF16((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => ArgLineC128((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => ArgLineScalar<decimal, DecimalCmp>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            _ => throw new NotSupportedException($"argsort not supported for dtype {tc}"),
        };
    }
}
