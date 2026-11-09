using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// AxisSort.Descending.cs — NumPy 2.5's descending sorts (np.sort / np.argsort / ndarray.sort / ndarray.argsort with
// descending=True, gh-31345) of the numeric dtypes and of fixed-width text (S / U)
// =============================================================================
//
// WHAT NUMPY DOES (2.5.4, npysort/npysort_methods.cpp, quicksort.hpp, timsort.hpp, numpy_tag.h; probed)
//   * array_sort / array_argsort fold descending=True into NPY_SORT_DESCENDING on top of the default or the stable kind
//     (a kind= string together with either keyword is refused before the array is read: SortKindArguments.Resolve).
//   * The comparator is Tag::greater, which keeps NaN LAST: a is "greater" than b when it is non-NaN and either b < a or b
//     is NaN; -0.0 and +0.0 are equal; complex keeps its NaN classes in the ascending order (no NaN part, R+nanj,
//     nan+Rj, nan+nanj) and orders each class descending (lexicographic for the first, by its one non-NaN part for the
//     next two, the last all equal); text compares its code units unsigned, reversed.
//   * The default kind runs the scalar introsort (quicksort_ / aquicksort_ / string_quicksort_ / string_aquicksort_
//     with reverse = true) on every x86 host: x86-simd-sort is ascending-only (quicksort_dispatch skips it when reverse),
//     so ties — equal values, -0.0 against +0.0, NaN payloads — land where that introsort puts them. The stable kind runs
//     timsort (radix sort for the 1- and 2-byte integers): equal values keep their input order.
//   * StringDType loops refuse the descending flag (RuntimeError "unsupported sort kind"), object arrays find no sort
//     function (TypeError); numpy.ma refuses descending=True with a ValueError. Those refusals live with their dtypes.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   1. The STABLE ASCENDING permutation of each line comes from the existing argsort (AxisSort.ArgSort for the numeric
//      dtypes — one stable order for every kind, NaN classes last in input order — and TextSort for S / U).
//   2. Inside each NaN class the ascending permutation is a sequence of RUNS of equal values; emitting the runs of every
//      class in reverse order, each run's own order kept, is the stable descending permutation — NumPy's stable kind.
//   3. The default kind needs NumPy's introsort tie order once a line has more than SMALL_QUICKSORT + 1 = 16 elements and
//      a run of two or more (a shorter line is one insertion sort, which is stable; a line without ties has one sorted
//      order). Each element then gets its dense RANK in the descending order, and NumPy's introsort, ported statement for
//      statement over (rank, index) records (TextSort.IntroArgSortRecords, the port the text and StringDType sorts use),
//      leaves NumPy's permutation: Tag::greater(a, b) is rank(a) < rank(b), so every comparison it makes has NumPy's
//      answer. The value sort gathers the line through that permutation, so -0.0 / +0.0 and the NaN payloads land where
//      NumPy's value introsort puts them (its value and index introsorts make the same comparisons and the same moves).
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    internal static unsafe partial class AxisSort
    {
        /// <summary>
        ///     The descending line kernel's context, carried through NDIter's auxdata (no per-line allocation): the line
        ///     geometry, the kind, and one line's scratch.
        /// </summary>
        private struct DescCtx
        {
            /// <summary>The line length.</summary>
            public long n;

            /// <summary>The byte distance between consecutive elements of a source line.</summary>
            public long srcStride;

            /// <summary>The byte distance between consecutive int64 slots of an ascending-permutation line.</summary>
            public long ascStride;

            /// <summary>The byte distance between consecutive slots of an output line (int64 indices, or elements for the
            ///     in-place value sort, where the output IS the source).</summary>
            public long dstStride;

            /// <summary>The element size in bytes (the text width for <c>S</c> / <c>U</c>).</summary>
            public int elsize;

            /// <summary>True for the stable kind (timsort's ties in input order); false for the default kind (NumPy's
            ///     introsort tie order).</summary>
            public bool stable;

            /// <summary>True for the in-place value sort (gather the elements); false for the argsort (write indices).</summary>
            public bool values;

            /// <summary>One line's stable descending permutation (n slots).</summary>
            public long* order;

            /// <summary>One line's (rank, index) records for the introsort port (2 · n slots).</summary>
            public ulong* rec;

            /// <summary>The value sort's gathered line (n · elsize bytes; null for an argsort).</summary>
            public byte* gather;
        }

        /// <summary>
        ///     How the descending kernel tells NumPy's ties and NaN classes apart for one element type: the class of an
        ///     element (the NaN classes after the class of values without a NaN part) and the equality of two elements of
        ///     one class under NumPy's comparator (<c>!greater(a, b) &amp;&amp; !greater(b, a)</c>).
        /// </summary>
        private interface IDescendingOrder
        {
            /// <summary>The NaN class of an element: 0 for a value without a NaN part, higher for the NaN classes in their
            ///     sort order.</summary>
            /// <param name="p">The element.</param>
            /// <returns>The class.</returns>
            int Class(byte* p);

            /// <summary>Whether two elements of one class are equal under NumPy's comparator.</summary>
            /// <param name="p">The first element.</param>
            /// <param name="q">The second element (same class).</param>
            /// <returns>True for a tie.</returns>
            bool Equal(byte* p, byte* q);
        }

        /// <summary>Elements equal exactly when their bytes are (the integers, bool and char): one class.</summary>
        private readonly struct DescBits8 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => *p == *q;
        }

        /// <summary>Two-byte elements equal exactly when their bits are (int16, uint16, char): one class.</summary>
        private readonly struct DescBits16 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => *(ushort*)p == *(ushort*)q;
        }

        /// <summary>Four-byte integers: one class, bit equality.</summary>
        private readonly struct DescBits32 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => *(uint*)p == *(uint*)q;
        }

        /// <summary>Eight-byte integers: one class, bit equality.</summary>
        private readonly struct DescBits64 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => *(ulong*)p == *(ulong*)q;
        }

        /// <summary>float32: the NaNs are one class after the values, all equal; <c>-0.0 == +0.0</c>.</summary>
        private readonly struct DescF32 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => float.IsNaN(*(float*)p) ? 1 : 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q)
            {
                float x = *(float*)p, y = *(float*)q;
                // Same class: two values compare IEEE-equal (the zeros alike), two NaNs are equal.
                return x == y || (x != x && y != y);
            }
        }

        /// <summary>float64: the NaNs are one class after the values, all equal; <c>-0.0 == +0.0</c>.</summary>
        private readonly struct DescF64 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => double.IsNaN(*(double*)p) ? 1 : 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q)
            {
                double x = *(double*)p, y = *(double*)q;
                return x == y || (x != x && y != y);
            }
        }

        /// <summary>float16 (<c>npy::half_tag</c>): the NaNs are one class after the values, all equal; the zeros equal.</summary>
        private readonly struct DescF16 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => (*(ushort*)p & 0x7FFF) > 0x7C00 ? 1 : 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q)
            {
                ushort a = *(ushort*)p, b = *(ushort*)q;
                // Two NaNs (same class) are equal; two values are equal when their bits are, or when both are zeros.
                if ((a & 0x7FFF) > 0x7C00)
                    return true;
                return a == b || ((a & 0x7FFF) == 0 && (b & 0x7FFF) == 0);
            }
        }

        /// <summary>
        ///     complex128 (<c>npy::cdouble_tag</c>): class 0 without a NaN part, 1 <c>R+nanj</c>, 2 <c>nan+Rj</c>, 3
        ///     <c>nan+nanj</c>; two elements of one class tie when their non-NaN parts compare equal.
        /// </summary>
        private readonly struct DescC128 : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p)
            {
                double re = ((double*)p)[0], im = ((double*)p)[1];
                return (double.IsNaN(re) ? 2 : 0) + (double.IsNaN(im) ? 1 : 0);
            }

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q)
            {
                double ar = ((double*)p)[0], ai = ((double*)p)[1], br = ((double*)q)[0], bi = ((double*)q)[1];
                // Within one class a NaN part faces a NaN part, which the comparator treats as equal.
                return (ar == br || (ar != ar && br != br)) && (ai == bi || (ai != ai && bi != bi));
            }
        }

        /// <summary>decimal: one class, value equality (<c>1.0m</c> equals <c>1.00m</c>).</summary>
        private readonly struct DescDecimal : IDescendingOrder
        {
            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => *(decimal*)p == *(decimal*)q;
        }

        /// <summary><c>S</c> / <c>U</c> text of one width: one class, equal when every byte is (equal code units).</summary>
        private readonly struct DescText : IDescendingOrder
        {
            /// <summary>The element width in bytes.</summary>
            private readonly int _width;

            /// <summary>Creates the order for one width.</summary>
            /// <param name="width">The element width in bytes.</param>
            public DescText(int width) => _width = width;

            /// <inheritdoc/>
            public int Class(byte* p) => 0;

            /// <inheritdoc/>
            [MethodImpl(OptimizeAndInline)]
            public bool Equal(byte* p, byte* q) => new ReadOnlySpan<byte>(p, _width).SequenceEqual(new ReadOnlySpan<byte>(q, _width));
        }

        /// <summary>
        ///     np.argsort with <c>descending=True</c> over a numeric or <c>S</c> / <c>U</c> array (a 0-d input already
        ///     ravelled): the axis check, then every line's descending permutation — the stable one for the stable kind,
        ///     NumPy's introsort order for the default kind.
        /// </summary>
        /// <param name="a">The array (only read; any layout).</param>
        /// <param name="axis">The axis, or null for the C-order flattening.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <returns>A fresh, C-contiguous int64 array of indices (one-dimensional for a null axis).</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="NotSupportedException">A Decimal line longer than <see cref="int.MaxValue"/> (the ascending
        ///     permutation's comparison sort uses managed buffers).</exception>
        [NDScoped]
        internal static NDArray ArgSortDescending(NDArray a, int? axis, bool stable)
        {
            bool text = Strings.NDStringUfuncs.IsString(a);
            if (text)
                Strings.NDStringUfuncs.ValidateReduceAxis("argsort", a.ndim, axis);
            // The ascending stable permutation (its own axis check for the numeric dtypes); the flattening for a null axis
            // is the same C-order line the ascending route sorts.
            var src = axis == null ? a.ravel() : a;
            int ax = axis == null ? 0 : NormalizeAxis(axis.Value, a.ndim);
            var ret = a.TensorEngine.NewArray(NPTypeCode.Int64, new Shape(src.Shape.dimensions, copy: false), false);
            if (src.size == 0)
                return ret;
            using var asc = text ? ArgSortText(src, ax, stable: true) : ArgSort(src, ax, NPY_SORTKIND.NPY_SORT_STABLE);
            DriveDescending(src, asc, ret, ax, stable, values: false);
            return ret;
        }

        /// <summary>
        ///     ndarray.sort with <c>descending=True</c> over a numeric or <c>S</c> / <c>U</c> array: every line along
        ///     <paramref name="axis"/> rewritten in descending order, in place — the stable order for the stable kind,
        ///     NumPy's introsort arrangement (ties included: <c>-0.0</c> / <c>+0.0</c> and the NaN payloads) for the default
        ///     kind.
        /// </summary>
        /// <param name="target">The writeable array (already checked).</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <exception cref="NotSupportedException">A Decimal line longer than <see cref="int.MaxValue"/>.</exception>
        [NDScoped]
        internal static void SortInPlaceDescending(NDArray target, int axis, bool stable)
        {
            if (target.Shape.dimensions[axis] <= 1 || target.size == 0)
                return;
            bool text = Strings.NDStringUfuncs.IsString(target);
            using var asc = text ? ArgSortText(target, axis, stable: true) : ArgSort(target, axis, NPY_SORTKIND.NPY_SORT_STABLE);
            DriveDescending(target, asc, target, axis, stable, values: true);
        }

        /// <summary>
        ///     Drives the descending line kernel over every line of <paramref name="src"/> along <paramref name="axis"/>,
        ///     with one line's scratch from the buffer pool.
        /// </summary>
        /// <param name="src">The sorted array (read; for the value sort also written).</param>
        /// <param name="asc">The stable ascending permutation of <paramref name="src"/>'s lines (int64, its dimensions).</param>
        /// <param name="dst">The int64 output for an argsort, or <paramref name="src"/> itself for the value sort.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="stable">True for the stable kind.</param>
        /// <param name="values">True for the in-place value sort, false for the argsort.</param>
        /// <remarks>The scratch is returned to the pool whatever the kernel throws (it throws nothing for valid input).</remarks>
        private static void DriveDescending(NDArray src, NDArray asc, NDArray dst, int axis, bool stable, bool values)
        {
            long n = src.Shape.dimensions[axis];
            int elsize = Strings.NDStringUfuncs.IsString(src) ? src.dtype.itemsize : src.GetTypeCode.SizeOf();
            var ctx = new DescCtx
            {
                n = n,
                srcStride = (long)src.Shape.strides[axis] * elsize,
                ascStride = (long)asc.Shape.strides[axis] * sizeof(long),
                dstStride = values ? (long)src.Shape.strides[axis] * elsize : (long)dst.Shape.strides[axis] * sizeof(long),
                elsize = elsize,
                stable = stable,
                values = values,
            };
            long orderBytes = n * sizeof(long), recBytes = 2 * n * sizeof(ulong), gatherBytes = values ? n * elsize : 0;
            try
            {
                ctx.order = (long*)SizeBucketedBufferPool.Take(orderBytes);
                ctx.rec = (ulong*)SizeBucketedBufferPool.Take(recBytes);
                if (values)
                    ctx.gather = (byte*)SizeBucketedBufferPool.Take(gatherBytes);
                NDInnerLoopFunc kern = GetDescendingKernel(src);
                if (values)
                    DriveAllButAxis(new[] { src, asc }, new[] { NDIterPerOpFlags.READWRITE, NDIterPerOpFlags.READONLY }, axis, kern, &ctx);
                else
                    DriveAllButAxis(new[] { src, asc, dst },
                        new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, kern, &ctx);
            }
            finally
            {
                if (ctx.order != null) SizeBucketedBufferPool.Return((IntPtr)ctx.order, orderBytes);
                if (ctx.rec != null) SizeBucketedBufferPool.Return((IntPtr)ctx.rec, recBytes);
                if (ctx.gather != null) SizeBucketedBufferPool.Return((IntPtr)ctx.gather, gatherBytes);
            }
        }

        /// <summary>The descending line kernel for <paramref name="a"/>'s element type (one type switch).</summary>
        /// <param name="a">The sorted array.</param>
        /// <returns>The kernel: operand 0 the source line, operand 1 the ascending permutation line, operand 2 (argsort) the
        ///     index line.</returns>
        /// <exception cref="NotSupportedException">An element type without a sort.</exception>
        private static NDInnerLoopFunc GetDescendingKernel(NDArray a)
        {
            if (Strings.NDStringUfuncs.IsString(a))
                return static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], ((DescCtx*)x)->values ? (byte*)p[0] : (byte*)p[2],
                    (DescCtx*)x, new DescText(((DescCtx*)x)->elsize));
            return a.GetTypeCode switch
            {
                NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte
                    => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescBits8)),
                NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                    => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescBits16)),
                NPTypeCode.Int32 or NPTypeCode.UInt32
                    => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescBits32)),
                NPTypeCode.Int64 or NPTypeCode.UInt64
                    => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescBits64)),
                NPTypeCode.Single => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescF32)),
                NPTypeCode.Double => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescF64)),
                NPTypeCode.Half => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescF16)),
                NPTypeCode.Complex => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescC128)),
                NPTypeCode.Decimal => static (p, s, c, x) => DescendingLine((byte*)p[0], (byte*)p[1], Out(p, x), (DescCtx*)x, default(DescDecimal)),
                var tc => throw new NotSupportedException($"sort not supported for dtype {tc}"),
            };
        }

        /// <summary>The output line of a descending kernel call: the source line for the value sort, operand 2 for the argsort.</summary>
        /// <param name="p">The iterator's data pointers.</param>
        /// <param name="x">The <see cref="DescCtx"/>.</param>
        /// <returns>The output line's first slot.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static byte* Out(void** p, void* x) => ((DescCtx*)x)->values ? (byte*)p[0] : (byte*)p[2];

        /// <summary>
        ///     One line of a descending sort: the stable descending permutation from the ascending one (each NaN class's runs
        ///     of equal values emitted in reverse order, each run's own order kept), NumPy's introsort permutation over the
        ///     elements' ranks for the default kind when it can differ, then the indices or the gathered elements written.
        /// </summary>
        /// <typeparam name="TOrder">The element type's classes and ties.</typeparam>
        /// <param name="src">The source line's first element.</param>
        /// <param name="asc">The ascending permutation line's first slot.</param>
        /// <param name="dst">The output line's first slot (the source line itself for the value sort).</param>
        /// <param name="c">The context.</param>
        /// <param name="order">The element type's order.</param>
        private static void DescendingLine<TOrder>(byte* src, byte* asc, byte* dst, DescCtx* c, TOrder order)
            where TOrder : struct, IDescendingOrder
        {
            long n = c->n, ss = c->srcStride, sa = c->ascStride;
            if (n == 0)
                return;
            long* perm = c->order;
            ulong* rec = c->rec;
            long o = 0, rank = 0;
            bool ties = false;
            long i = 0;
            while (i < n)
            {
                // One NaN class: the slots [i, j) of the ascending order.
                int cls = order.Class(src + *(long*)(asc + i * sa) * ss);
                long j = i + 1;
                while (j < n && order.Class(src + *(long*)(asc + j * sa) * ss) == cls)
                    j++;
                // Its runs of equal values, from the last (the greatest) to the first, each in its ascending (input) order.
                long e = j;
                while (e > i)
                {
                    byte* last = src + *(long*)(asc + (e - 1) * sa) * ss;
                    long s = e - 1;
                    while (s > i && order.Equal(src + *(long*)(asc + (s - 1) * sa) * ss, last))
                        s--;
                    ties |= e - s > 1;
                    for (long k = s; k < e; k++)
                    {
                        long index = *(long*)(asc + k * sa);
                        perm[o++] = index;
                        // NumPy's initial index order (index i at record i), the dense descending rank as its key.
                        rec[2 * index] = (ulong)rank;
                        rec[2 * index + 1] = (ulong)index;
                    }
                    rank++;
                    e = s;
                }
                i = j;
            }

            // The default kind partitions a line of more than SMALL_QUICKSORT + 1 elements, so only there can a tie leave the
            // stable order; Tag::greater(a, b) is rank(a) < rank(b), so the port's comparisons are NumPy's.
            if (!c->stable && ties && n > TextSort.SmallQuicksort + 1)
            {
                TextSort.IntroArgSortRecords(rec, n);
                for (long k = 0; k < n; k++)
                    perm[k] = (long)rec[2 * k + 1];
            }

            if (!c->values)
            {
                long sd = c->dstStride;
                for (long k = 0; k < n; k++)
                    *(long*)(dst + k * sd) = perm[k];
                return;
            }
            // The value sort: gather the elements (their bits, so equal values keep their own) and write them back.
            int w = c->elsize;
            byte* g = c->gather;
            for (long k = 0; k < n; k++)
                Buffer.MemoryCopy(src + perm[k] * ss, g + k * w, w, w);
            long sv = c->dstStride;
            for (long k = 0; k < n; k++)
                Buffer.MemoryCopy(g + k * w, dst + k * sv, w, w);
        }
    }
}
