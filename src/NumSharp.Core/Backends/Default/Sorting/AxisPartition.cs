using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    /// Along-axis partition / argpartition, structured exactly like NumPy's
    /// <c>PyArray_Partition</c> / <c>PyArray_ArgPartition</c> (item_selection.c): the SAME
    /// <see cref="AxisSort.DriveAllButAxis"/> NDIter drive hands one 1-D line per call to an
    /// introselect kernel (<see cref="QuickSelect"/> — the primitive already backing
    /// median/percentile/quantile), around the caller's pre-validated, pre-sorted kth list.
    ///
    /// Validation is the port of <c>partition_prep_kth_array</c> + <c>PyArray_Partition</c>'s own
    /// checks, in NumPy's probed order: kind → order → axis → writeable → kth (kth bounds are
    /// SKIPPED for a size-0 array, negative kth wraps once and the error reports the post-wrap
    /// value, and the kth list is sorted so successive partitions cannot trample each other).
    ///
    /// Float lines NaN-compact first (NumPy's LT sorts any NaN last; original NaN bit patterns are
    /// preserved in the tail, in encounter order — the same policy as <see cref="AxisSort"/>'s
    /// argsort NaN tail); Complex lines use the exact NumPy CDOUBLE_LT comparator
    /// (<see cref="AxisSort.ComplexCmp"/>), whose NaN ordering is NOT a compactable
    /// "any-NaN-part last" — (1, NaN) orders before (NaN, 1).
    /// </summary>
    internal static unsafe class AxisPartition
    {
        // ============================ public entry points ============================

        /// <summary>np.partition: fresh C-contiguous partitioned copy (axis=null flattens first).</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <returns>The partitioned copy.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static NDArray Partition(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            // Text: NumPy checks the axis and the kth bounds, then (a string having no partition function) sorts every
            // line with npy_quicksort, so the partition IS the sorted copy (StringDType first: IsString refuses it).
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextChecks("partition", a, axis, () => WidenKth(kth));
                return Strings.NDStringUfuncs.IsVString(a) ? AxisSort.SortVString(a, axis, stable: false) : AxisSort.SortText(a, axis);
            }
            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                PartitionInPlaceCore(flat, 0, WidenKth(kth));
                return flat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            // NumPy partition = copy(order='K') + in-place (fromnumeric.partition verbatim):
            // an F input keeps F layout, strided/negative come back C (probed 2.4.2).
            var res = a.copy('K');
            PartitionInPlaceCore(res, ax, WidenKth(kth));
            return res;
        }

        /// <summary>np.partition with an NDArray kth — NumPy's array form, which is what makes its
        /// kth-dtype rejections REACHABLE (a typed C# int[] cannot be bool/float/2-D). Validation is
        /// NumPy's probed STAGING: kind → order → too-deep (before axis!) → axis → bool/integer dtype
        /// → wrap/bounds — a bool kth with a bad axis reports the AXIS, a 2-D kth reports too-deep.</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <returns>The partitioned copy.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static NDArray Partition(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            // Text: the axis, then the kth dtype and bounds, then the sort NumPy partitions a string with.
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextChecks("partition", a, axis, () => ExtractKthValues(kth));
                return Strings.NDStringUfuncs.IsVString(a) ? AxisSort.SortVString(a, axis, stable: false) : AxisSort.SortText(a, axis);
            }
            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                PartitionInPlaceCore(flat, 0, ExtractKthValues(kth));
                return flat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            // NumPy partition = copy(order='K') + in-place — same rule as the scalar-kth overload.
            var res = a.copy('K');
            PartitionInPlaceCore(res, ax, ExtractKthValues(kth));
            return res;
        }

        /// <summary>ndarray.partition: partitions <paramref name="a"/> in place along the axis
        /// (null flattens in place — the NumSharp extension ndarray.sort already carries).</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static void PartitionInPlace(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            // Text: the axis, writeability and the kth bounds, then NumPy sorts every line in place.
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextPartitionInPlace(a, axis, () => WidenKth(kth));
                return;
            }
            if (axis == null)
            {
                // In-place flatten-partition only well-defined for contiguous (ndarray.sort house rule).
                if (!a.Shape.IsWriteable)
                    throw new ValueError("partition array is read-only");
                PartitionInPlaceCore(a.reshape(a.size), 0, WidenKth(kth));
                return;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            if (!a.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            PartitionInPlaceCore(a, ax, WidenKth(kth));
        }

        /// <summary>ndarray.partition with an NDArray kth (staged validation — see the np.partition form).</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static void PartitionInPlace(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            // Text: the axis, writeability, then the kth dtype and bounds, then the in-place sort of every line.
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextPartitionInPlace(a, axis, () => ExtractKthValues(kth));
                return;
            }
            if (axis == null)
            {
                if (!a.Shape.IsWriteable)
                    throw new ValueError("partition array is read-only");
                PartitionInPlaceCore(a.reshape(a.size), 0, ExtractKthValues(kth));
                return;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            if (!a.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            PartitionInPlaceCore(a, ax, ExtractKthValues(kth));
        }

        /// <summary>np.argpartition: int64 indices that would partition <paramref name="a"/>
        /// (axis=null flattens; the input is only read). A 0-d input ravels to (1,) first —
        /// NumPy's arg-side PyArray_CheckAxis quirk, so kth/axis errors report dimension 1.</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static NDArray ArgPartition(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            if (a.ndim == 0)
                a = a.reshape(1);
            // Text: the axis (a 0-d input ravelled above), then the kth bounds, then NumPy's generic introsort fallback
            // (npy_aquicksort), which compares as the string introsort does: argpartition is the default argsort. Over
            // StringDType that fallback receives the real array, so where the argsort loop crashes NumPy (its heapsort
            // fallback) argpartition still answers — with the port's heapsort.
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextChecks("argpartition", a, axis, () => WidenKth(kth));
                return Strings.NDStringUfuncs.IsVString(a)
                    ? AxisSort.ArgSortVString(a, axis, stable: false)
                    : AxisSort.ArgSortText(a, axis, stable: false);
            }
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, Shape.Vector(a.size), false);
                ArgPartitionInto(flat, outFlat, 0, WidenKth(kth));
                return outFlat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])a.Shape.dimensions.Clone()), false);
            ArgPartitionInto(src, ret, ax, WidenKth(kth));
            return ret;
        }

        /// <summary>np.argpartition with an NDArray kth (staged validation — see the np.partition form).</summary>
        /// <param name="a">The array.</param>
        /// <param name="kth">The element index (or indices) to put in sorted position.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>introselect</c>).</param>
        /// <param name="order">Field names (no NumSharp dtype has fields: any value refuses).</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/>, an <paramref name="order"/>, or a kth out of
        ///     bounds (and NumPy's kth dtype errors for an array kth).</exception>
        /// <exception cref="AxisError">A string array with an out-of-bounds axis.</exception>
        public static NDArray ArgPartition(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            if (a.ndim == 0)
                a = a.reshape(1);
            // Text: the axis, then the kth dtype and bounds, then the default argsort (NumPy's generic introsort fallback).
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsString(a))
            {
                TextChecks("argpartition", a, axis, () => ExtractKthValues(kth));
                return Strings.NDStringUfuncs.IsVString(a)
                    ? AxisSort.ArgSortVString(a, axis, stable: false)
                    : AxisSort.ArgSortText(a, axis, stable: false);
            }
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, Shape.Vector(a.size), false);
                ArgPartitionInto(flat, outFlat, 0, ExtractKthValues(kth));
                return outFlat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])a.Shape.dimensions.Clone()), false);
            ArgPartitionInto(src, ret, ax, ExtractKthValues(kth));
            return ret;
        }

        // ============================ validation (NumPy's order) ============================

        private static void ValidateKindOrder(string kind, string order)
        {
            // NumPy converts `kind` while parsing arguments, so it fires before order/axis/kth.
            if (kind is not null && kind != "introselect")
                throw new ValueError($"select kind must be 'introselect' (got '{kind}')");
            // `order` exists for structured dtypes only, which NumSharp does not have.
            if (order is not null)
                throw new ValueError("Cannot specify order when the array has no fields.");
        }

        /// <summary>
        ///     The checks NumPy makes before <c>np.partition</c> / <c>np.argpartition</c> touch text: the axis (<c>axis=None</c>
        ///     flattens), the kth values (an array kth's dtype checks run here, after the axis) and their bounds against the
        ///     partitioned length (<c>partition_prep_kth_array</c> runs "even if using sorting"). The caller then sorts: a
        ///     string has no partition function, so NumPy sorts every line (<c>npy_quicksort</c> / <c>npy_aquicksort</c>).
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The string array (a 0-d argpartition input already ravelled).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kth">Produces the kth values (raising NumPy's kth dtype errors).</param>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">A kth is out of bounds, or a bool kth array.</exception>
        /// <exception cref="TypeError">A non-integer kth array.</exception>
        private static void TextChecks(string op, NDArray a, int? axis, Func<long[]> kth)
        {
            // NumPy's AxisError (the numeric path's NormalizeAxis keeps its house ArgumentException).
            Strings.NDStringUfuncs.ValidateReduceAxis(op, a.ndim, axis);
            long n = axis is not { } ax ? a.size : a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
            PrepKth(kth(), n, a.size);
        }

        /// <summary>
        ///     The in-place <c>ndarray.partition</c> over text, in NumPy's <c>PyArray_Partition</c> order: the axis
        ///     (<c>check_and_adjust_axis</c>), the array's writeability (<c>partition array is read-only</c>), the kth values
        ///     and their bounds; then (a string having no partition function) the in-place sort of every line, which leaves
        ///     a line of at most one element, or an empty array, as it is.
        /// </summary>
        /// <param name="a">The string array partitioned in place.</param>
        /// <param name="axis">The axis, or null (NumSharp's flatten-in-place extension: no axis check).</param>
        /// <param name="kth">Produces the kth values (raising NumPy's kth dtype errors).</param>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">The array is read-only, a kth is out of bounds, or a bool kth array.</exception>
        /// <exception cref="TypeError">A non-integer kth array.</exception>
        private static void TextPartitionInPlace(NDArray a, int? axis, Func<long[]> kth)
        {
            Strings.NDStringUfuncs.ValidateReduceAxis("partition", a.ndim, axis);
            if (!a.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            long n = axis is not { } ax ? a.size : a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
            PrepKth(kth(), n, a.size);
            if (a.size == 0 || n <= 1)
                return;
            bool vstring = Strings.NDStringUfuncs.IsVString(a);
            if (axis == null)
            {
                // NumSharp's flatten-in-place extension: one line over the whole (contiguous) buffer.
                if (vstring)
                    VStringSort.SortLinesInPlace(a.reshape(a.size), 0, stable: false);
                else
                    TextSort.SortLines(a.reshape(a.size), 0);
                return;
            }
            int line = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            if (vstring)
                VStringSort.SortLinesInPlace(a, line, stable: false);
            else
                TextSort.SortLines(a, line);
        }

        /// <summary>Widen an int[] kth to the long[] the cores take (NumPy's kth is intp).
        /// Null rides through so the ArgumentNullException still fires at the kth-validation stage.</summary>
        private static long[] WidenKth(int[] kth)
        {
            if (kth is null)
                return null;
            var k64 = new long[kth.Length];
            for (int i = 0; i < kth.Length; i++)
                k64[i] = kth[i];
            return k64;
        }

        /// <summary>The pre-axis stage of NumPy's array-kth conversion: a &gt;1-D kth is rejected
        /// with the verbatim conversion error BEFORE the axis is even looked at (probed:
        /// <c>np.partition(a, [[1]], axis=9)</c> reports too-deep, not the axis). A null kth
        /// passes: it rides through to the kth stage (<see cref="ExtractKthValues"/> hands it on,
        /// <see cref="PrepKth"/> refuses it), where NumPy refuses kth=None ("Partition index must be
        /// integer", after the axis), as the int[] form's null already did.</summary>
        /// <param name="kth">The kth index array, or null.</param>
        /// <exception cref="ValueError"><paramref name="kth"/> has more than one dimension.</exception>
        private static void KthTooDeepCheck(NDArray kth)
        {
            // A null is not refused here: NumPy converts None to a 0-d object array, which passes this stage, and
            // checks the axis before it refuses the dtype.
            if (kth is null)
                return;
            if (kth.ndim > 1)
                throw new ValueError("object too deep for desired array");
        }

        /// <summary>The kth-stage of NumPy's <c>partition_prep_kth_array</c> for an array kth:
        /// bool rejects first ("Booleans unacceptable as partition index"), any non-integer dtype
        /// next ("Partition index must be integer" — TypeError), then the values cast to intp with
        /// NumPy's MODULAR wrap (a uint64 2^63 becomes a negative kth and its bounds error reports
        /// the wrapped value — probed: <c>kth(=-9223372036854775804) out of bounds (4)</c>, while
        /// uint64 2^64-1 wraps to -1 and is a LEGAL last-element kth). Char (no NumPy dtype) is
        /// integer-family, so it rides the same route (house call). Any layout/0-d is normalized
        /// by the astype+ravel.</summary>
        /// <param name="kth">The kth index array, or null.</param>
        /// <returns>The positions as int64 values in C order (wrapped modularly), or null for a null
        ///     <paramref name="kth"/>, which <see cref="PrepKth"/> then refuses.</returns>
        /// <exception cref="ValueError">A bool index array.</exception>
        /// <exception cref="TypeError">An index array of a non-integer dtype.</exception>
        private static long[] ExtractKthValues(NDArray kth)
        {
            // Null rides through, as WidenKth's does for the int[] form, so PrepKth's ArgumentNullException fires at the
            // kth stage, after the axis.
            if (kth is null)
                return null;
            if (kth.typecode == NPTypeCode.Boolean)
                throw new ValueError("Booleans unacceptable as partition index");
            switch (kth.typecode)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    break;
                default:
                    throw new TypeError("Partition index must be integer");
            }

            // The int64 conversion (a copy unless kth already is int64) and its flat view are this method's own
            // intermediates, both disposed here — never the caller's kth (ravel always returns a new array object, and a
            // view's Dispose releases only its own reference). The conversion used to be left to the finalizer: one pooled
            // buffer per call whose kth has another integer dtype (a C# int literal converts to an int32 kth).
            var wide = kth.typecode == NPTypeCode.Int64 ? kth : kth.astype(NPTypeCode.Int64);
            try
            {
                using var k64 = wide.ravel();
                var result = new long[k64.size];
                for (int i = 0; i < result.Length; i++)
                    result[i] = k64.GetValue<long>(i);
                return result;
            }
            finally
            {
                if (!ReferenceEquals(wide, kth))
                    wide.Dispose();
            }
        }

        /// <summary>
        ///     Port of <c>partition_prep_kth_array</c>: wrap negatives once, bounds-check against
        ///     the axis length UNLESS the array is empty (NumPy's <c>PyArray_SIZE(op) != 0</c> guard),
        ///     report the post-wrap value verbatim, and sort ascending so the partitions compose.
        ///     An empty kth is NumPy's <c>np.array([], dtype=intp)</c> — a valid no-op (Python's
        ///     bare <c>[]</c> is float64 and raises; a typed C# int[] carries no such ambiguity).
        /// </summary>
        private static long[] PrepKth(long[] kth, long axisLength, long size)
        {
            // NumPy's np.partition(a, None) is a TypeError; a null C# array is the same caller bug
            // (house ArgumentNullException precedent: fill_diagonal/lexsort). Guarded HERE so it
            // fires at NumPy's kth-validation stage — after kind/order/axis, like every kth error.
            if (kth is null)
                throw new ArgumentNullException(nameof(kth));
            var ks = new long[kth.Length];
            for (int i = 0; i < kth.Length; i++)
            {
                long k = kth[i];
                if (k < 0)
                    k += axisLength;
                if (size != 0 && (k < 0 || k >= axisLength))
                    throw new ValueError($"kth(={k}) out of bounds ({axisLength})");
                ks[i] = k; // a kth position can exceed int.MaxValue for an axis longer than 2^31
            }
            if (ks.Length > 1)
                Array.Sort(ks);
            return ks;
        }

        // ============================ drive ============================

        private static void PartitionInPlaceCore(NDArray target, int axis, long[] kth)
        {
            long N = target.Shape.dimensions[axis];
            var ks = PrepKth(kth, N, target.size);
            if (target.size == 0 || N <= 1 || ks.Length == 0)
                return;

            var tc = target.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new PartCtx
            {
                n = N,
                inStride = (long)target.Shape.strides[axis] * elsize,
                outStride = 0,
                nKth = ks.Length,
            };

            // per-call scratch, reused across every line (the AxisSort buffer policy): the value
            // gather buffer always, the NaN stash only for the float dtypes that compact. UNMANAGED
            // so a line may exceed int.MaxValue (a managed byte[] element count caps at ~2^31), and
            // POOLED: a fresh C-allocator buffer is first-touch page faults on every call (see
            // AxisSort.ArgSortInto). Every Return passes the size its Take got.
            long lineBytes = N * elsize;
            void* ps = null, pn = null;
            try
            {
                ps = (void*)SizeBucketedBufferPool.Take(lineBytes);
                if (NeedsNanTail(tc)) pn = (void*)SizeBucketedBufferPool.Take(lineBytes);
                fixed (long* pk = ks)
                {
                    ctx.scratch = (byte*)ps;
                    ctx.nanTail = (byte*)pn;
                    ctx.kth = pk;
                    NDInnerLoopFunc kern = GetPartitionKernel(tc);
                    AxisSort.DriveAllButAxis(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, kern, &ctx);
                }
            }
            finally
            {
                if (ps != null) SizeBucketedBufferPool.Return((IntPtr)ps, lineBytes);
                if (pn != null) SizeBucketedBufferPool.Return((IntPtr)pn, lineBytes);
            }
        }

        private static void ArgPartitionInto(NDArray src, NDArray dst, int axis, long[] kth)
        {
            long N = src.Shape.dimensions[axis];
            var ks = PrepKth(kth, N, src.size);
            if (src.size == 0)
                return;

            var tc = src.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new PartCtx
            {
                n = N,
                inStride = (long)src.Shape.strides[axis] * elsize,
                outStride = (long)dst.Shape.strides[axis] * sizeof(long),
                nKth = ks.Length,
            };

            // argpartition always gathers (the source is read-only); idx is the answer column. Both
            // are UNMANAGED so a line may exceed int.MaxValue (the int64 idx column would otherwise
            // cap at ~2^31 elements), and POOLED (see PartitionInPlaceCore).
            long lineBytes = N * elsize, idxBytes = N * sizeof(long);
            void* ps = null; long* pi = null;
            try
            {
                ps = (void*)SizeBucketedBufferPool.Take(lineBytes);
                pi = (long*)SizeBucketedBufferPool.Take(idxBytes);
                fixed (long* pk = ks)
                {
                    ctx.scratch = (byte*)ps;
                    ctx.idx = pi;
                    ctx.kth = pk;
                    NDInnerLoopFunc kern = GetArgPartitionKernel(tc);
                    AxisSort.DriveAllButAxis(new[] { src, dst },
                        new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, kern, &ctx);
                }
            }
            finally
            {
                if (ps != null) SizeBucketedBufferPool.Return((IntPtr)ps, lineBytes);
                if (pi != null) SizeBucketedBufferPool.Return((IntPtr)pi, idxBytes);
            }
        }

        private static bool NeedsNanTail(NPTypeCode tc)
            => tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;

        // line-partition context carried via NDIter auxdata (no per-call captures/allocations)
        private struct PartCtx
        {
            public byte* scratch;   // N-element value gather buffer
            public byte* nanTail;   // N-element NaN stash (float dtypes only)
            public long* idx;       // N-element original-index column (argpartition only)
            public long* kth;       // wrapped, validated, sorted ascending (a position can exceed int.MaxValue)
            public int nKth;
            public long inStride;   // byte stride along the partition axis
            public long outStride;  // byte stride of the int64 output line (argpartition only)
            public long n;
        }

        // NumPy CDOUBLE_LT as a Comparison<Complex> for the QuickSelect comparator path.
        private static readonly Comparison<Complex> ComplexLess =
            static (a, b) => default(AxisSort.ComplexCmp).Compare(a, b);

        // ============================ partition line kernels ============================

        /// <summary>Raw-compare dtypes (ints/bool/char/decimal — no NaN representation): partition the
        /// line in place when it is contiguous, else gather → introselect → scatter.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PartLine<T>(byte* line, PartCtx* c) where T : unmanaged, IComparable<T>
        {
            long n = c->n;
            long s = c->inStride;
            if (s == sizeof(T))
            {
                QuickSelect.PartitionAtMany((T*)line, n, c->kth, c->nKth);
                return;
            }
            T* buf = (T*)c->scratch;
            for (long i = 0; i < n; i++) buf[i] = *(T*)(line + i * s);
            QuickSelect.PartitionAtMany(buf, n, c->kth, c->nKth);
            for (long i = 0; i < n; i++) *(T*)(line + i * s) = buf[i];
        }

        /// <summary>Float dtypes: stable NaN-compact (non-NaN prefix keeps encounter order, NaN bit
        /// patterns stashed verbatim), introselect the prefix with raw compares, NaN tail written back.
        /// kth entries pointing into the NaN tail are correct by construction (every slot there is NaN,
        /// which is where a full NaN-last sort would put them).</summary>
        private static void PartLineF32(byte* line, PartCtx* c)
        {
            long n = c->n; long s = c->inStride;
            float* buf = (float*)c->scratch; float* nans = (float*)c->nanTail;
            long m = 0, q = 0;
            for (long i = 0; i < n; i++) { float v = *(float*)(line + i * s); if (float.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(float*)(line + i * s) = buf[i];
            for (long i = 0; i < q; i++) *(float*)(line + (m + i) * s) = nans[i];
        }

        private static void PartLineF64(byte* line, PartCtx* c)
        {
            long n = c->n; long s = c->inStride;
            double* buf = (double*)c->scratch; double* nans = (double*)c->nanTail;
            long m = 0, q = 0;
            for (long i = 0; i < n; i++) { double v = *(double*)(line + i * s); if (double.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(double*)(line + i * s) = buf[i];
            for (long i = 0; i < q; i++) *(double*)(line + (m + i) * s) = nans[i];
        }

        private static void PartLineF16(byte* line, PartCtx* c)
        {
            long n = c->n; long s = c->inStride;
            Half* buf = (Half*)c->scratch; Half* nans = (Half*)c->nanTail;
            long m = 0, q = 0;
            for (long i = 0; i < n; i++) { Half v = *(Half*)(line + i * s); if (Half.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(Half*)(line + i * s) = buf[i];
            for (long i = 0; i < q; i++) *(Half*)(line + (m + i) * s) = nans[i];
        }

        /// <summary>Complex: NumPy's CDOUBLE_LT comparator drives the introselect directly — its NaN
        /// ordering ((1,NaN) before (NaN,1)) is positional, not a compactable any-NaN-last.</summary>
        private static void PartLineComplex(byte* line, PartCtx* c)
        {
            long n = c->n; long s = c->inStride;
            if (s == sizeof(Complex))
            {
                QuickSelect.PartitionAtMany((Complex*)line, n, c->kth, c->nKth, ComplexLess);
                return;
            }
            Complex* buf = (Complex*)c->scratch;
            for (long i = 0; i < n; i++) buf[i] = *(Complex*)(line + i * s);
            QuickSelect.PartitionAtMany(buf, n, c->kth, c->nKth, ComplexLess);
            for (long i = 0; i < n; i++) *(Complex*)(line + i * s) = buf[i];
        }

        // ============================ argpartition line kernels ============================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgPartLine<T>(byte* inLine, byte* outLine, PartCtx* c) where T : unmanaged, IComparable<T>
        {
            long n = c->n; long si = c->inStride, so = c->outStride;
            T* buf = (T*)c->scratch; long* idx = c->idx;
            for (long i = 0; i < n; i++) { buf[i] = *(T*)(inLine + i * si); idx[i] = i; }
            QuickSelect.PartitionAtMany(buf, idx, n, c->kth, c->nKth);
            for (long i = 0; i < n; i++) *(long*)(outLine + i * so) = idx[i];
        }

        /// <summary>Float argpartition: NaN indices go to the tail in encounter order — the exact
        /// policy of <see cref="AxisSort"/>'s ArgLineF32/F64 argsort kernels.</summary>
        private static void ArgPartLineF32(byte* inLine, byte* outLine, PartCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride;
            float* buf = (float*)c->scratch; long* idx = c->idx;
            long m = 0;
            for (long i = 0; i < n; i++) { float v = *(float*)(inLine + i * si); if (!float.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            long q = m;
            for (long i = 0; i < n; i++) if (float.IsNaN(*(float*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineF64(byte* inLine, byte* outLine, PartCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride;
            double* buf = (double*)c->scratch; long* idx = c->idx;
            long m = 0;
            for (long i = 0; i < n; i++) { double v = *(double*)(inLine + i * si); if (!double.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            long q = m;
            for (long i = 0; i < n; i++) if (double.IsNaN(*(double*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineF16(byte* inLine, byte* outLine, PartCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride;
            Half* buf = (Half*)c->scratch; long* idx = c->idx;
            long m = 0;
            for (long i = 0; i < n; i++) { Half v = *(Half*)(inLine + i * si); if (!Half.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (long i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            long q = m;
            for (long i = 0; i < n; i++) if (Half.IsNaN(*(Half*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineComplex(byte* inLine, byte* outLine, PartCtx* c)
        {
            long n = c->n; long si = c->inStride, so = c->outStride;
            Complex* buf = (Complex*)c->scratch; long* idx = c->idx;
            for (long i = 0; i < n; i++) { buf[i] = *(Complex*)(inLine + i * si); idx[i] = i; }
            QuickSelect.PartitionAtMany(buf, idx, n, c->kth, c->nKth, ComplexLess);
            for (long i = 0; i < n; i++) *(long*)(outLine + i * so) = idx[i];
        }

        // ============================ dtype dispatch (one type-switch each) ============================

        private static NDInnerLoopFunc GetPartitionKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => PartLine<bool>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => PartLine<byte>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => PartLine<sbyte>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => PartLine<short>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => PartLine<ushort>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => PartLine<char>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => PartLine<int>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => PartLine<uint>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => PartLine<long>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => PartLine<ulong>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => PartLineF32((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => PartLineF64((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => PartLineF16((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => PartLineComplex((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => PartLine<decimal>((byte*)p[0], (PartCtx*)a),
            _ => throw new NotSupportedException($"partition not supported for dtype {tc}"),
        };

        private static NDInnerLoopFunc GetArgPartitionKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => ArgPartLine<bool>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => ArgPartLine<byte>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => ArgPartLine<sbyte>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => ArgPartLine<short>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => ArgPartLine<ushort>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => ArgPartLine<char>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => ArgPartLine<int>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => ArgPartLine<uint>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => ArgPartLine<long>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => ArgPartLine<ulong>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => ArgPartLineF32((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => ArgPartLineF64((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => ArgPartLineF16((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => ArgPartLineComplex((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => ArgPartLine<decimal>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            _ => throw new NotSupportedException($"argpartition not supported for dtype {tc}"),
        };
    }
}
