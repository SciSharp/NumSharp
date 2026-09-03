using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     The kernel-backed route for a single integer index array over a C-contiguous source —
    ///     the shape behind <c>a[idx]</c>, <c>a[idx] = v</c>, <c>np.take(a, idx[, axis=0])</c> and
    ///     <c>np.put(a, idx, v)</c>. It hands the gather / scatter to the IL kernels
    ///     (<see cref="DirectILKernelGenerator.GetTakeFlatKernel"/> and friends) and reproduces the
    ///     two NumPy behaviours the fancy indexer must keep on top of them:
    ///     <list type="bullet">
    ///       <item>the index array is read <b>in place</b> at its own width (int32 or int64), with
    ///       its view offset honoured — no widening copy;</item>
    ///       <item>a fancy <em>assignment</em> never writes partially: NumPy's
    ///       <c>mapiter_trivial_set</c> validates every index BEFORE the first store
    ///       ("Check the indices beforehand"), so <see cref="FirstOutOfRange(long*, long, long)"/>
    ///       runs a SIMD bounds scan first and the scatter kernel then cannot fail.</item>
    ///     </list>
    ///     Everything here is layout-agnostic in dtype: a gather moves <c>slabBytes</c> per index
    ///     (one element, or one whole trailing sub-array when the index selects rows).
    /// </summary>
    internal static unsafe class FancyIndexKernels
    {
        /// <summary>
        ///     Gathered-region footprint above which the take kernels emit software prefetch (the same
        ///     2 MiB knee <c>np.take</c> uses: below it the source is cache-resident and the prefetch
        ///     is pure per-element overhead, above it the random gather stalls on memory latency).
        /// </summary>
        internal const long PrefetchThresholdBytes = 2L * 1024 * 1024;

        /// <summary>
        ///     Resolves a 1-D index array to an in-place pointer when the kernels can read it directly:
        ///     int32 or int64 dtype, C-contiguous (a view offset is folded into the pointer). Any
        ///     other layout (a strided / reversed index view, a narrower integer dtype) returns false
        ///     and the caller keeps its general route.
        /// </summary>
        internal static bool TryGetIndexPointer(NDArray indices, out void* ptr, out bool idx32)
        {
            ptr = null;
            idx32 = false;
            var tc = indices.typecode;
            if (tc != NPTypeCode.Int32 && tc != NPTypeCode.Int64)
                return false;
            if (!indices.Shape.IsContiguous)
                return false;
            idx32 = tc == NPTypeCode.Int32;
            long itemBytes = idx32 ? 4 : 8;
            ptr = indices.Storage.Address + indices.Shape.offset * itemBytes;
            return true;
        }

        /// <summary>The index value at position <paramref name="j"/> of an int32/int64 index buffer.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ReadIndex(void* indices, bool idx32, long j)
            => idx32 ? ((int*)indices)[j] : ((long*)indices)[j];

        /// <summary>
        ///     Gathers <paramref name="count"/> slabs of <paramref name="slabBytes"/> bytes, addressed by
        ///     the index array, from the C-contiguous <paramref name="src"/> (whose indexed axis has
        ///     <paramref name="maxItem"/> slabs) into the contiguous <paramref name="dst"/>, under RAISE
        ///     semantics (a negative index is normalised once, then bounds-checked). Picks the lean flat
        ///     kernel for a primitive-width slab and the general take kernel (runtime-sized <c>cpblk</c>)
        ///     otherwise. Returns -1 on success, else the position of the first out-of-range index (the
        ///     destination is then partially written — a fresh gather result is simply discarded).
        ///     Returns -2 when runtime IL generation is unavailable (nothing touched).
        /// </summary>
        internal static long Gather(byte* src, void* indices, bool idx32, long count, long maxItem, long slabBytes, byte* dst)
        {
            bool prefetch = maxItem * slabBytes > PrefetchThresholdBytes;
            int copyKind = DirectILKernelGenerator.CopyKindFor(slabBytes);
            if (copyKind != 0)
            {
                var flat = DirectILKernelGenerator.GetTakeFlatKernel(copyKind, idx32, prefetch);
                if (flat == null)
                    return -2;
                long done = flat(src, indices, count, maxItem, dst);
                return done == count ? -1 : done;
            }

            var general = DirectILKernelGenerator.GetTakeKernel(copyKind, prefetch, idx32);
            if (general == null)
                return -2;
            long status = general(src, (long*)indices, count, 1, maxItem, slabBytes, 0, dst);
            return status == count ? -1 : status;
        }

        /// <summary>
        ///     Scatters <paramref name="count"/> slabs of <paramref name="slabBytes"/> bytes from
        ///     <paramref name="values"/> (a wrapping cursor over <paramref name="valuesCount"/> slabs:
        ///     1 broadcasts one slab to every index, <c>count</c> is a straight copy) into the
        ///     C-contiguous <paramref name="dst"/> at the index array's positions (RAISE semantics).
        ///     Returns -1 on success, the first failing position (earlier slabs ARE written — callers
        ///     that must not write partially validate first with <see cref="FirstOutOfRange(long*, long, long)"/>),
        ///     or -2 when runtime IL generation is unavailable.
        /// </summary>
        internal static long Scatter(byte* dst, void* indices, bool idx32, long count, byte* values, long valuesCount, long maxItem, long slabBytes)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(slabBytes);
            if (copyKind != 0)
            {
                var flat = DirectILKernelGenerator.GetPutFlatKernel(copyKind, idx32);
                if (flat == null)
                    return -2;
                long done = flat(dst, indices, count, values, valuesCount, maxItem);
                return done == count ? -1 : done;
            }

            var general = DirectILKernelGenerator.GetPutKernel(copyKind, idx32);
            if (general == null)
                return -2;
            long status = general(dst, (long*)indices, count, values, valuesCount, maxItem, slabBytes, 0);
            return status == count ? -1 : status;
        }

        /// <summary>
        ///     Position of the first index outside <c>[-n, n)</c>, or -1 when every index is in range —
        ///     NumPy's "check the indices beforehand" pass, vectorised: <see cref="Vector{T}"/> lanes
        ///     are tested against both bounds and only a failing block is re-walked one element at a
        ///     time to name the exact position.
        /// </summary>
        internal static long FirstOutOfRange(long* indices, long count, long n)
        {
            long i = 0;
            if (Vector.IsHardwareAccelerated && count >= Vector<long>.Count * 2)
            {
                int lanes = Vector<long>.Count;
                var lo = new Vector<long>(-n);
                var hi = new Vector<long>(n);
                long blockEnd = count - lanes;
                for (; i <= blockEnd; i += lanes)
                {
                    var v = Unsafe.ReadUnaligned<Vector<long>>(indices + i);
                    var bad = Vector.BitwiseOr(Vector.LessThan(v, lo), Vector.GreaterThanOrEqual(v, hi));
                    if (!Vector.EqualsAll(bad, Vector<long>.Zero))
                        break;
                }
            }
            for (; i < count; i++)
            {
                long v = indices[i];
                if (v < -n || v >= n)
                    return i;
            }
            return -1;
        }

        /// <summary>
        ///     <see cref="FirstOutOfRange(long*, long, long)"/> for an int32 index array. An axis longer
        ///     than <see cref="int.MaxValue"/> cannot be exceeded by any int32 index, so the scan is
        ///     skipped outright there.
        /// </summary>
        internal static long FirstOutOfRange(int* indices, long count, long n)
        {
            if (n > int.MaxValue)
                return -1;
            int n32 = (int)n;
            long i = 0;
            if (Vector.IsHardwareAccelerated && count >= Vector<int>.Count * 2)
            {
                int lanes = Vector<int>.Count;
                var lo = new Vector<int>(-n32);
                var hi = new Vector<int>(n32);
                long blockEnd = count - lanes;
                for (; i <= blockEnd; i += lanes)
                {
                    var v = Unsafe.ReadUnaligned<Vector<int>>(indices + i);
                    var bad = Vector.BitwiseOr(Vector.LessThan(v, lo), Vector.GreaterThanOrEqual(v, hi));
                    if (!Vector.EqualsAll(bad, Vector<int>.Zero))
                        break;
                }
            }
            for (; i < count; i++)
            {
                int v = indices[i];
                if (v < -n32 || v >= n32)
                    return i;
            }
            return -1;
        }

        /// <summary>Dispatches on the index width.</summary>
        internal static long FirstOutOfRange(void* indices, bool idx32, long count, long n)
            => idx32 ? FirstOutOfRange((int*)indices, count, n) : FirstOutOfRange((long*)indices, count, n);
    }
}
