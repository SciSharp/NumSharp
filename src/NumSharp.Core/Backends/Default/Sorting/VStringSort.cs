using System;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// VStringSort.cs — sort / argsort / partition / lexsort of NEP 55 StringDType text (T), along an axis
// =============================================================================
//
// WHAT NUMPY DOES (2.4.2, numpy/_core/src/multiarray/stringdtype/dtype.c + npysort; probed)
//   * StringDType registers its own sort and argsort ArrayMethods: the default kind runs the GENERIC introsort
//     (npy_quicksort_impl / npy_aquicksort_impl: median of three, insertion sort below SMALL_QUICKSORT + 1 = 16 elements,
//     the heapsort fallback once the depth budget 2 * msb(n) is spent), the stable kind the generic mergesort
//     (npy_mergesort_impl / npy_amergesort_impl); the heapsort kind raises RuntimeError("unsupported sort kind").
//   * Every comparison is `_sort_compare(a, b) < 0`, which loads both cells (NpyString_load) and compares their UTF-8
//     bytes (NpyString_cmp: memcmp of the common prefix, then the shorter first). A missing (NA) cell follows the
//     descriptor's rule: a NaN-like NA object makes `_compare` return 1 when the FIRST operand is NA (even against
//     another NA) and -1 when only the second is — so under `< 0` tests every NA is "not less" than everything and
//     every non-NA is less than an NA: the NAs behave as ONE greatest value; any other non-string NA object (None, a
//     custom object) raises ValueError("Cannot compare null that is not a nan-like value") at the first comparison that
//     touches it; a string NA object (and a descriptor without one) reads the NA as its default string.
//   * np.partition / np.argpartition have no selection loop for StringDType: PyArray_(Arg)Partition falls back to the
//     legacy npy_quicksort / npy_aquicksort with the dtype's compare function — the same introsort, so argpartition is
//     the default argsort and partition the sorted values (probed).
//   * np.lexsort runs every key's NPY_STABLESORT argsort, and StringDType has no legacy one: npy_atimsort with the
//     compare function, re-sorting the running index array. Its run detection tests `cmp(a, b) <= 0` (ascending) and
//     `cmp(b, a) < 0` (strictly descending), so two adjacent NaN-like NAs (cmp(NA, NA) = 1 both ways) open a descending
//     run and are REVERSED: np.lexsort((np.array([nan, nan, 'a'], dtype=nan),)) is [2, 1, 0] where argsort(kind='stable')
//     gives [2, 0, 1].
//   * NumPy 2.4.2 bug: the StringDType sort and argsort loops hand npy_(a)quicksort_impl the METHOD CONTEXT as its array
//     argument, and the heapsort fallback (npy_heapsort / npy_aheapsort) reads that pointer as a PyArrayObject: an input
//     that exhausts the depth budget crashes the interpreter (0xC0000005, probed with McIlroy's adversary). The
//     partition fallback passes the real array and does not crash.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   1. LOAD. Each element of a line is loaded once (its text pointer and size, NDVStringCells.Load with the arena base read
//      under the allocator lock); an NA cell follows the descriptor's rule: a NaN-like NA is set aside (in input order),
//      a None-like NA in a line of two or more elements raises NumPy's ValueError (every one of NumPy's sorts compares
//      every element of such a line), any other NA reads as the default string. Each text gets an order-preserving key:
//      its first 8 bytes read big-endian, zero-padded.
//   2. STABLE ORDER. A stable radix sort of the keys (RadixSort, only the bytes in which the keys differ), then a stable
//      TIE PASS over every run of equal keys: the elements that END within the bytes the keys covered come first, by
//      size (equal keys make a shorter text a prefix of a longer one, and NpyString_cmp puts a prefix first); the others
//      are keyed on their next 8 bytes and radix-sorted inside the run, recursively; runs of at most 16 elements are
//      insertion-sorted by comparing their remaining bytes. The NaN-like NAs follow in input order. This is the order of
//      NumPy's stable kind (its mergesort only tests `< 0`, so the NAs keep their order), and of every VALUE sort.
//   3. NUMPY'S TIE ORDER for the default kind: a line of at most 16 elements is one insertion sort in NumPy (stable). A
//      longer line whose elements all differ (one NA at most) has one sorted order. Otherwise each element gets its dense
//      rank in the stable order (the NAs one rank above every text), and NumPy's introsort, ported statement for statement
//      over (rank, index) records (TextSort.IntroArgSortRecords), leaves NumPy's permutation: its every comparison is a
//      `< 0` test, which on ranks has the answer the string compare has. Where NumPy crashes (the heapsort fallback),
//      the port's heapsort answers — the order np.argpartition gives.
//   4. LEXSORT. The key's ranks, the running permutation's ranks, then npy_atimsort ported statement for statement over
//      them with NumPy's asymmetric NA compare when the line holds two or more NaN-like NAs (anything else makes the
//      compare a total order, under which timsort is the stable sort: the stable radix of the ranks answers).
//
// LINES
//   DriveLines walks every line along the axis: operands of at most two dimensions directly, others through NDIter with
//   the axis dropped (op_axes) and EXTERNAL_LOOP, so one call of a Job's block kernel receives a run of lines. StringDType
//   operands hold references (their cells point into an arena or the heap): the iterator needs REFS_OK, and the kernels
//   read the cells in place (no cast, no buffer). The Job owns one line's scratch and reuses it for every line.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     Sorting of NEP 55 <c>StringDType</c> text along an axis with NumPy 2.4.2's answers: the value sort, the stable and
    ///     the default argsort (NumPy's introsort permutation, ties included), and the <c>np.lexsort</c> pass of a text key
    ///     (NumPy's timsort with its asymmetric NA compare). See the file header for the algorithm.
    /// </summary>
    internal static unsafe class VStringSort
    {
        /// <summary>NumPy's <c>SMALL_QUICKSORT</c>: a line of at most <c>SmallQuicksort + 1</c> elements is one insertion
        ///     sort in NumPy's introsort, which is stable.</summary>
        private const int SmallQuicksort = TextSort.SmallQuicksort;

        /// <summary>The longest run of equal keys the tie pass insertion-sorts by comparing the remaining bytes; longer runs
        ///     key their next 8 bytes and radix-sort them.</summary>
        private const int TieInsertionMax = 16;

        /// <summary>NumPy's <c>RUN_STACK_SIZE</c> (timsort.cpp): the timsort's pending-run stack.</summary>
        private const int TimRunStack = 128;

        /// <summary>The rank of a NaN-like NA in a line's rank table (above every text's dense rank).</summary>
        private const long NARank = long.MaxValue;

        /// <summary>
        ///     Test hook: lines whose default argsort ran NumPy's introsort port (ties present). A plain counter, not
        ///     synchronized: tests read the difference around calls they make on one thread.
        /// </summary>
        internal static long PortRuns;

        /// <summary>Test hook: lines whose default argsort was answered by the stable order (no tie, or at most 16
        ///     elements). See <see cref="PortRuns"/>.</summary>
        internal static long ShortcutRuns;

        /// <summary>Test hook: lexsort lines that ran NumPy's timsort port (two or more NaN-like NAs). See
        ///     <see cref="PortRuns"/>.</summary>
        internal static long TimPortRuns;

        /// <summary>Test hook: block-kernel calls the line driver made (one per run of lines). See <see cref="PortRuns"/>.</summary>
        internal static long BlockCalls;

        /// <summary>
        ///     Test hook: when true, the default argsort runs NumPy's introsort port even when the stable order is the answer
        ///     (no tie, or a short line); the permutation is the same. Process-wide; set it only from single-threaded test
        ///     code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool ShortcutDisabled;

        /// <summary>
        ///     Test hook: when true, a lexsort line runs NumPy's timsort port whatever its NAs (the stable radix gives the
        ///     same permutation without two NaN-like NAs). Process-wide; set it only from single-threaded test code and
        ///     restore it in a <c>finally</c>.
        /// </summary>
        internal static bool TimShortcutDisabled;

        /// <summary>A static empty text (a non-null pointer to nothing), for the NA of a one-element line, never compared.</summary>
        /// <remarks>The allocator's one shared constant (<see cref="VStringAllocator.EmptyString"/>, NumPy's
        ///     <c>EMPTY_STRING</c>) rather than a block of its own, so the sort adds no raw native allocation. Never written
        ///     through, never freed.</remarks>
        private static readonly byte* s_empty = VStringAllocator.EmptyString;

        // ============================== entry points ==============================

        /// <summary>
        ///     Writes the indices that sort every line of <paramref name="src"/> along <paramref name="axis"/> into
        ///     <paramref name="dst"/>: NumPy's stable order for the stable kind, NumPy's introsort permutation otherwise.
        /// </summary>
        /// <param name="src">The <c>StringDType</c> array (only read; any layout, broadcast included).</param>
        /// <param name="dst">An int64 array of <paramref name="src"/>'s dimensions; every line is overwritten.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="stable">True for the stable kind (<c>kind='stable'</c> / <c>'mergesort'</c>, <c>stable=True</c>);
        ///     false for the default kind and for <c>np.argpartition</c>.</param>
        /// <exception cref="ValueError">A line of two or more elements holding an NA of a descriptor whose NA object is
        ///     neither NaN-like nor a string (<c>Cannot compare null that is not a nan-like value</c>).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell (an arena cell without an arena: NumPy's <c>Failed to load
        ///     string in string comparison</c>), or the scratch cannot be allocated.</exception>
        internal static void ArgSortLines(NDArray src, NDArray dst, int axis, bool stable)
        {
            long n = src.Shape.dimensions[axis];
            if (n == 0 || src.size == 0)
                return;
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var allocator = src.Storage.VStringSlice.Allocator;
            // The arena moves only when its allocator allocates; nothing here allocates in it, so one base read under the
            // lock serves the whole call (the lock also keeps other threads' writers out while the cells are read).
            allocator.acquire_allocator();
            try
            {
                using var job = new Job(src.dtype, n, src.Shape.strides[axis] * (long)cell, dst.Shape.strides[axis] * (long)sizeof(long),
                    allocator.ArenaBase, stable ? JobMode.ArgStable : JobMode.ArgDefault);
                DriveLines(new[] { src, dst }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, job.ArgSortBlock);
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>
        ///     Sorts every line of <paramref name="target"/> along <paramref name="axis"/> in place by permuting its cells
        ///     (<c>ndarray.sort</c>, and <c>np.sort</c> / <c>np.partition</c> over their copy): the cells keep pointing into
        ///     the array's own allocator, so nothing is allocated.
        /// </summary>
        /// <param name="target">A writeable <c>StringDType</c> array.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <param name="stable">True for the stable kind; false for the default kind and for <c>np.partition</c>.</param>
        /// <exception cref="ValueError">See <see cref="ArgSortLines"/>. NumPy raises after sorting the line with the failing
        ///     comparisons answered "equal"; NumSharp leaves the failing line as it was (the lines before it are sorted in
        ///     both).</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="ArgSortLines"/>.</exception>
        /// <remarks>
        ///     Equal texts are equal values, and every NaN-like NA is the same value, so the stable order writes NumPy's
        ///     values for either kind; only a line whose NA cells read as a default string (a string NA object, or a
        ///     descriptor without one) can hold an NA cell equal to a text cell, which <c>repr</c> cannot tell apart but
        ///     identity can: the default kind then places the cells as NumPy's introsort does.
        /// </remarks>
        internal static void SortLinesInPlace(NDArray target, int axis, bool stable)
        {
            long n = target.Shape.dimensions[axis];
            if (n <= 1 || target.size == 0)
                return;
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var allocator = target.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                using var job = new Job(target.dtype, n, target.Shape.strides[axis] * (long)cell, 0, allocator.ArenaBase,
                    stable ? JobMode.ValueStable : JobMode.ValueDefault);
                DriveLines(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, job.SortBlockInPlace);
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>
        ///     One <c>np.lexsort</c> pass of a <c>StringDType</c> key: every line of <paramref name="perm"/> (the running
        ///     permutation) is re-sorted by the key's text as NumPy's <c>npy_atimsort</c> re-sorts its index array.
        /// </summary>
        /// <param name="key">The <c>StringDType</c> key (only read; any layout).</param>
        /// <param name="perm">The running permutation (int64, the key's dimensions), rewritten in place; each line holds a
        ///     permutation of its line's indices.</param>
        /// <param name="axis">The normalised axis.</param>
        /// <exception cref="ValueError">See <see cref="ArgSortLines"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="ArgSortLines"/>.</exception>
        internal static void LexPass(NDArray key, NDArray perm, int axis)
        {
            long n = key.Shape.dimensions[axis];
            if (n <= 1 || key.size == 0)
                return;
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            var allocator = key.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                using var job = new Job(key.dtype, n, key.Shape.strides[axis] * (long)cell, perm.Shape.strides[axis] * (long)sizeof(long),
                    allocator.ArenaBase, JobMode.Lex);
                DriveLines(new[] { key, perm }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE }, axis, job.LexBlock);
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>
        ///     Walks every line of <paramref name="ops"/> along <paramref name="axis"/>, handing <paramref name="kernel"/> a
        ///     run of lines per call: the first line's start in each operand, the byte distance between consecutive lines,
        ///     and their count (operands of at most two dimensions directly, others through NDIter with the axis dropped and
        ///     <c>EXTERNAL_LOOP</c>).
        /// </summary>
        /// <param name="ops">The operands, all of one shape (a <c>StringDType</c> array first).</param>
        /// <param name="flags">Each operand's NDIter flags.</param>
        /// <param name="axis">The normalised axis the lines run along.</param>
        /// <param name="kernel">The block kernel.</param>
        /// <remarks>The iterator gets <c>REFS_OK</c> for a <c>StringDType</c> operand (it holds references) and NO_CASTING
        ///     with each operand's own descriptor: the kernels read and write the cells in place.</remarks>
        private static void DriveLines(NDArray[] ops, NDIterPerOpFlags[] flags, int axis, NDInnerLoopFunc kernel)
        {
            if (ops[0].size == 0)
                return;
            int ndim = ops[0].ndim;
            if (ndim <= 2)
            {
                // One line (1-D) or the lines of one kept axis (2-D): the pointers and the step between lines are the
                // operands' own; no iterator is needed.
                int keptAxis = ndim == 2 ? 1 - axis : -1;
                long count = keptAxis >= 0 ? ops[0].Shape.dimensions[keptAxis] : 1;
                void** pointers = stackalloc void*[ops.Length];
                long* steps = stackalloc long[ops.Length];
                for (int k = 0; k < ops.Length; k++)
                {
                    long elementBytes = NDIterRef.ElementBytes(ops[k]);
                    pointers[k] = (byte*)ops[k].Address + ops[k].Shape.offset * elementBytes;
                    steps[k] = keptAxis >= 0 ? ops[k].Shape.strides[keptAxis] * elementBytes : 0;
                }
                kernel(pointers, steps, count, null);
                return;
            }
            var kept = new int[ndim - 1];
            for (int d = 0, w = 0; d < ndim; d++)
                if (d != axis)
                    kept[w++] = d;
            var opAxes = new int[ops.Length][];
            for (int i = 0; i < ops.Length; i++)
                opAxes[i] = kept;
            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP;
            foreach (var op in ops)
                if (op.typecode == NPTypeCode.String && op.Storage.IsVString)
                    globalFlags |= NDIterGlobalFlags.REFS_OK;
            // Null descriptors keep each operand's own (no cast).
            var descrs = new DType[ops.Length];
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, globalFlags, NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING, flags, null, ndim - 1, opAxes, opDescrs: descrs);
            iter.ForEach(kernel);
        }

        /// <summary>NumPy's comparison refusal of a None-like NA.</summary>
        /// <returns>The exception (not thrown).</returns>
        internal static ValueError NullCompareError() => new ValueError("Cannot compare null that is not a nan-like value");

        /// <summary>NumPy's <c>_compare</c> refusal of a corrupt cell (its <c>MemoryError</c>).</summary>
        /// <returns>The exception (not thrown).</returns>
        internal static OutOfMemoryException CorruptCellError() => new OutOfMemoryException("Failed to load string in string comparison");

        /// <summary>
        ///     The 8 bytes of a text from <paramref name="offset"/> on, read big-endian and zero-padded past the text's end:
        ///     an order-preserving key of that window (unsigned key order is byte order).
        /// </summary>
        /// <param name="buf">The text's first byte.</param>
        /// <param name="size">The text's size.</param>
        /// <param name="offset">The window's first byte.</param>
        /// <returns>The key (0 when the text ends before the window).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static ulong KeyAt(byte* buf, long size, long offset)
        {
            long left = size - offset;
            if (left >= 8)
                return BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(buf + offset));
            if (left <= 0)
                return 0;
            // A partial window: byte by byte (reading 8 bytes could pass the end of an arena or heap allocation).
            ulong k = 0;
            byte* p = buf + offset;
            for (int b = 0; b < left; b++)
                k |= (ulong)p[b] << (56 - 8 * b);
            return k;
        }

        /// <summary>
        ///     The first-window key of a cell's text (<see cref="KeyAt"/> at offset 0), with the short-cell fast path: a text
        ///     stored IN its 16-byte cell can be read as one word and masked to its size.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="buf">The text's first byte (the cell itself for a short string).</param>
        /// <param name="size">The text's size.</param>
        /// <returns>The key.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static ulong FirstKey(byte* cell, byte* buf, long size)
        {
            if (buf != cell)
                return KeyAt(buf, size, 0);
            // A short cell: 16 readable bytes, the text in bytes 0..size-1 and stale bytes after it — masked away.
            ulong raw = BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(cell));
            if (size >= 8)
                return raw;
            return size == 0 ? 0 : raw & (ulong.MaxValue << (int)(64 - 8 * size));
        }

        /// <summary>
        ///     NumPy's <c>NpyString_cmp</c> of two texts whose bytes before <paramref name="from"/> are known equal (as far as
        ///     both reach): the bytes from <paramref name="from"/> on, then the shorter text first.
        /// </summary>
        /// <param name="a">The first text.</param>
        /// <param name="sa">Its size.</param>
        /// <param name="b">The second text.</param>
        /// <param name="sb">Its size.</param>
        /// <param name="from">The first byte that may differ.</param>
        /// <returns>Negative, zero or positive.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int CompareFrom(byte* a, long sa, byte* b, long sb, long from)
        {
            long ra = sa - from, rb = sb - from;
            long common = Math.Min(ra, rb);
            if (common > 0)
            {
                int c = NDVStringCells.Order(a + from, common, b + from, common);
                if (c != 0)
                    return c;
            }
            return sa.CompareTo(sb);
        }

        // ============================== the line job ==============================

        /// <summary>What a <see cref="Job"/> writes for each line.</summary>
        private enum JobMode : byte
        {
            /// <summary>The stable argsort (operand 1: int64 indices).</summary>
            ArgStable,

            /// <summary>The default argsort: NumPy's introsort permutation (operand 1: int64 indices).</summary>
            ArgDefault,

            /// <summary>The in-place value sort, stable kind (operand 0 rewritten).</summary>
            ValueStable,

            /// <summary>The in-place value sort, default kind (operand 0 rewritten).</summary>
            ValueDefault,

            /// <summary>A lexsort pass (operand 1: the running permutation, rewritten).</summary>
            Lex,
        }

        /// <summary>
        ///     One sort call's state: its constants, its NA rule and ONE line's scratch (pooled buffers), reused for every line
        ///     the driver hands it. Disposing returns the scratch.
        /// </summary>
        private sealed class Job : IDisposable
        {
            /// <summary>The line length (at least 1).</summary>
            private readonly long _n;

            /// <summary>The byte distance between consecutive cells of a line.</summary>
            private readonly long _cellStride;

            /// <summary>The byte distance between consecutive int64 slots of an output line (argsort / lexsort).</summary>
            private readonly long _outStride;

            /// <summary>The arena base of the cells' allocator (read under its lock).</summary>
            private readonly byte* _arena;

            /// <summary>The NA rule of the descriptor.</summary>
            private readonly VStringNARule _rule;

            /// <summary>The UTF-8 default string an NA reads as under the <c>None</c> / <c>String</c> rules (pooled; null when
            ///     empty).</summary>
            private byte* _defaultText;

            /// <summary>The byte length of <see cref="_defaultText"/>.</summary>
            private readonly long _defaultSize;

            /// <summary>What each line produces.</summary>
            private readonly JobMode _mode;

            /// <summary>Per element: the text's first byte (the default string for an NA read as text).</summary>
            private byte** _bufs;

            /// <summary>Per element: the text's size, or -1 for a NaN-like NA.</summary>
            private long* _sizes;

            /// <summary>Per element: the first-window key.</summary>
            private ulong* _key0;

            /// <summary>The text elements' indices: the stable order once <see cref="StableOrder"/> ran (the first
            ///     <see cref="_textCount"/> slots), followed by the NaN-like NAs in input order.</summary>
            private long* _order;

            /// <summary>Radix keys (one per slot of a run being sorted).</summary>
            private ulong* _keys;

            /// <summary>The radix's second key buffer.</summary>
            private ulong* _keyTmp;

            /// <summary>The radix's second index buffer (also a partition scratch).</summary>
            private long* _idxTmp;

            /// <summary>The radix histograms (8 · 256 counts).</summary>
            private long* _hist;

            /// <summary>Per element: its dense rank in the stable order (<see cref="NARank"/> for a NaN-like NA); allocated by the
            ///     modes that need ranks.</summary>
            private long* _ranks;

            /// <summary>(rank, index) records for the introsort port (2 · n), allocated on first use.</summary>
            private ulong* _rec;

            /// <summary>The value sort's cell scratch (16 · n bytes), allocated on first use.</summary>
            private byte* _cellTmp;

            /// <summary>The lexsort pass's working permutation and its timsort buffer (2 · n), allocated on first use.</summary>
            private long* _perm;

            /// <summary>The lexsort pass's ranks in the running permutation's order (n), allocated on first use.</summary>
            private ulong* _seq;

            /// <summary>The number of text elements of the current line (NAs read as text included).</summary>
            private long _textCount;

            /// <summary>The number of NaN-like NAs of the current line.</summary>
            private long _naCount;

            /// <summary>Whether the current line holds an NA cell read as a default string (the <c>None</c> / <c>String</c>
            ///     rules): such a cell can equal a text cell in value while being a different cell.</summary>
            private bool _defaultReadNA;

            /// <summary>Byte sizes of the pooled buffers, kept for <see cref="Dispose"/>.</summary>
            private readonly long _ptrBytes, _wordBytes;

            /// <summary>The histogram scratch's byte size.</summary>
            private const long HistBytes = 8 * 256 * sizeof(long);

            /// <summary>
            ///     Allocates the scratch of one line of <paramref name="n"/> elements and reads the descriptor's NA rule.
            /// </summary>
            /// <param name="descr">The <c>StringDType</c> descriptor of the lines.</param>
            /// <param name="n">The line length (at least 1).</param>
            /// <param name="cellStride">The byte distance between consecutive cells of a line.</param>
            /// <param name="outStride">The byte distance between consecutive int64 output slots (0 for a value sort).</param>
            /// <param name="arena">The cells' arena base (read under the allocator lock).</param>
            /// <param name="mode">What each line produces.</param>
            /// <exception cref="OutOfMemoryException">The scratch cannot be allocated (whatever was taken is returned).</exception>
            internal Job(DType descr, long n, long cellStride, long outStride, byte* arena, JobMode mode)
            {
                _n = n;
                _cellStride = cellStride;
                _outStride = outStride;
                _arena = arena;
                _mode = mode;
                _rule = NDVStringCells.RuleOf(descr);
                _ptrBytes = n * sizeof(long);
                _wordBytes = n * sizeof(long);
                try
                {
                    byte[] text = NDUtf8.Encode(descr.VString.default_string);
                    _defaultSize = text.Length;
                    if (text.Length > 0)
                    {
                        // A pooled copy (no pin for the call's lifetime): the cells' reads hand out this pointer.
                        _defaultText = (byte*)SizeBucketedBufferPool.Take(text.Length);
                        fixed (byte* t = text)
                            Buffer.MemoryCopy(t, _defaultText, text.Length, text.Length);
                    }
                    _bufs = (byte**)SizeBucketedBufferPool.Take(_ptrBytes);
                    _sizes = (long*)SizeBucketedBufferPool.Take(_wordBytes);
                    _key0 = (ulong*)SizeBucketedBufferPool.Take(_wordBytes);
                    _order = (long*)SizeBucketedBufferPool.Take(_wordBytes);
                    _keys = (ulong*)SizeBucketedBufferPool.Take(_wordBytes);
                    _keyTmp = (ulong*)SizeBucketedBufferPool.Take(_wordBytes);
                    _idxTmp = (long*)SizeBucketedBufferPool.Take(_wordBytes);
                    _hist = (long*)SizeBucketedBufferPool.Take(HistBytes);
                    if (mode is JobMode.ArgDefault or JobMode.ValueDefault or JobMode.Lex)
                        _ranks = (long*)SizeBucketedBufferPool.Take(_wordBytes);
                }
                catch
                {
                    // A failed allocation leaves the job half-built: return what was taken before rethrowing.
                    Dispose();
                    throw;
                }
            }

            /// <summary>Returns every pooled buffer (idempotent).</summary>
            public void Dispose()
            {
                if (_defaultText != null) { SizeBucketedBufferPool.Return((IntPtr)_defaultText, _defaultSize); _defaultText = null; }
                if (_bufs != null) { SizeBucketedBufferPool.Return((IntPtr)_bufs, _ptrBytes); _bufs = null; }
                if (_sizes != null) { SizeBucketedBufferPool.Return((IntPtr)_sizes, _wordBytes); _sizes = null; }
                if (_key0 != null) { SizeBucketedBufferPool.Return((IntPtr)_key0, _wordBytes); _key0 = null; }
                if (_order != null) { SizeBucketedBufferPool.Return((IntPtr)_order, _wordBytes); _order = null; }
                if (_keys != null) { SizeBucketedBufferPool.Return((IntPtr)_keys, _wordBytes); _keys = null; }
                if (_keyTmp != null) { SizeBucketedBufferPool.Return((IntPtr)_keyTmp, _wordBytes); _keyTmp = null; }
                if (_idxTmp != null) { SizeBucketedBufferPool.Return((IntPtr)_idxTmp, _wordBytes); _idxTmp = null; }
                if (_hist != null) { SizeBucketedBufferPool.Return((IntPtr)_hist, HistBytes); _hist = null; }
                if (_ranks != null) { SizeBucketedBufferPool.Return((IntPtr)_ranks, _wordBytes); _ranks = null; }
                if (_rec != null) { SizeBucketedBufferPool.Return((IntPtr)_rec, 2 * _wordBytes); _rec = null; }
                if (_cellTmp != null) { SizeBucketedBufferPool.Return((IntPtr)_cellTmp, 2 * _wordBytes); _cellTmp = null; }
                if (_perm != null) { SizeBucketedBufferPool.Return((IntPtr)_perm, 2 * _wordBytes); _perm = null; }
                if (_seq != null) { SizeBucketedBufferPool.Return((IntPtr)_seq, _wordBytes); _seq = null; }
            }

            // ---------------------------------------------------------------- block kernels

            /// <summary>
            ///     The block kernel of an argsort (an <see cref="NDInnerLoopFunc"/> run by <see cref="DriveLines"/>): operand 0
            ///     holds the cell lines, operand 1 the int64 index lines.
            /// </summary>
            /// <param name="dataptrs">The first line's start in each operand.</param>
            /// <param name="strides">The byte distance between consecutive lines in each operand.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void ArgSortBlock(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* cells = (byte*)dataptrs[0], output = (byte*)dataptrs[1];
                long cellStep = strides[0], outStep = strides[1];
                for (long l = 0; l < count; l++)
                    ArgSortLine(cells + l * cellStep, output + l * outStep);
            }

            /// <summary>The block kernel of the in-place value sort: operand 0 holds the cell lines.</summary>
            /// <param name="dataptrs">The first line's start.</param>
            /// <param name="strides">The byte distance between consecutive lines.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void SortBlockInPlace(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* cells = (byte*)dataptrs[0];
                long step = strides[0];
                for (long l = 0; l < count; l++)
                    SortLineInPlace(cells + l * step);
            }

            /// <summary>The block kernel of a lexsort pass: operand 0 holds the key's cell lines, operand 1 the running
            ///     permutation's int64 lines (rewritten).</summary>
            /// <param name="dataptrs">The first line's start in each operand.</param>
            /// <param name="strides">The byte distance between consecutive lines in each operand.</param>
            /// <param name="count">The number of lines in the block.</param>
            /// <param name="aux">Unused.</param>
            internal void LexBlock(void** dataptrs, long* strides, long count, void* aux)
            {
                BlockCalls++;
                byte* cells = (byte*)dataptrs[0], perm = (byte*)dataptrs[1];
                long cellStep = strides[0], permStep = strides[1];
                for (long l = 0; l < count; l++)
                    LexLine(cells + l * cellStep, perm + l * permStep);
            }

            // ---------------------------------------------------------------- per line

            /// <summary>Argsorts one line: the stable order, then (default kind) the shortcut or NumPy's introsort port.</summary>
            /// <param name="cells">The line's cell 0.</param>
            /// <param name="output">The index line's slot 0.</param>
            private void ArgSortLine(byte* cells, byte* output)
            {
                long n = _n;
                if (n == 1)
                {
                    // NumPy's sorts make no comparison over one element: index 0, whatever the cell holds.
                    *(long*)output = 0;
                    return;
                }
                Load(cells);
                StableOrder();
                if (_mode == JobMode.ArgDefault && NeedsIntrosort())
                {
                    long* perm = IntrosortPermutation();
                    for (long k = 0; k < n; k++)
                        *(long*)(output + k * _outStride) = perm[2 * k + 1];
                    return;
                }
                for (long k = 0; k < n; k++)
                    *(long*)(output + k * _outStride) = _order[k];
            }

            /// <summary>Sorts one line's cells in place: the order of the kind (see <see cref="SortLinesInPlace"/>), applied by
            ///     gathering the cells into scratch and writing them back.</summary>
            /// <param name="cells">The line's cell 0.</param>
            private void SortLineInPlace(byte* cells)
            {
                long n = _n;
                Load(cells);
                StableOrder();
                const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                if (_cellTmp == null)
                    _cellTmp = (byte*)SizeBucketedBufferPool.Take(2 * _wordBytes);
                // The default kind's own permutation only where it can differ in what the cells hold (see the remarks of
                // SortLinesInPlace): an NA cell read as a default string in a line with ties.
                if (_mode == JobMode.ValueDefault && _defaultReadNA && NeedsIntrosort())
                {
                    long* rec = IntrosortPermutation();
                    for (long k = 0; k < n; k++)
                        Buffer.MemoryCopy(cells + rec[2 * k + 1] * _cellStride, _cellTmp + k * cell, cell, cell);
                }
                else
                {
                    for (long k = 0; k < n; k++)
                        Buffer.MemoryCopy(cells + _order[k] * _cellStride, _cellTmp + k * cell, cell, cell);
                }
                for (long k = 0; k < n; k++)
                    Buffer.MemoryCopy(_cellTmp + k * cell, cells + k * _cellStride, cell, cell);
            }

            /// <summary>
            ///     One lexsort pass over one line: the key's ranks, the running permutation re-sorted by them — the stable radix
            ///     of its ranks, or NumPy's timsort port when two or more NaN-like NAs make its compare asymmetric.
            /// </summary>
            /// <param name="cells">The key line's cell 0.</param>
            /// <param name="permLine">The permutation line's slot 0 (rewritten).</param>
            private void LexLine(byte* cells, byte* permLine)
            {
                long n = _n;
                Load(cells);
                StableOrder();
                ComputeRanks();
                if (_perm == null)
                    _perm = (long*)SizeBucketedBufferPool.Take(2 * _wordBytes);
                if (_seq == null)
                    _seq = (ulong*)SizeBucketedBufferPool.Take(_wordBytes);
                for (long k = 0; k < n; k++)
                {
                    long index = *(long*)(permLine + k * _outStride);
                    _perm[k] = index;
                    _seq[k] = (ulong)_ranks[index];
                }
                long* result;
                if (_naCount >= 2 || TimShortcutDisabled)
                {
                    TimPortRuns++;
                    TimSort(_perm, n, _perm + n);
                    result = _perm;
                }
                else
                {
                    // At most one NaN-like NA: the compare is a total order (an NA greater than every text), under which
                    // timsort is stable — the stable radix of the ranks (in the permutation's order) gives its answer.
                    ulong varying = 0, all = ulong.MaxValue;
                    for (long k = 0; k < n; k++)
                    {
                        varying |= _seq[k];
                        all &= _seq[k];
                    }
                    result = RadixSort.ArgSortU64(_seq, _keyTmp, _perm, _idxTmp, n, _hist, varying ^ all, out _);
                }
                for (long k = 0; k < n; k++)
                    *(long*)(permLine + k * _outStride) = result[k];
            }

            // ---------------------------------------------------------------- load

            /// <summary>
            ///     Loads one line's cells: each text's pointer, size and first-window key; a NaN-like NA is set aside in input
            ///     order (size -1); a None-like NA raises NumPy's ValueError (the line has two or more elements, and every one
            ///     of NumPy's sorts compares every element then); any other NA reads as the default string.
            /// </summary>
            /// <param name="cells">The line's cell 0.</param>
            /// <exception cref="ValueError">A None-like NA (<see cref="NullCompareError"/>).</exception>
            /// <exception cref="OutOfMemoryException">A corrupt cell (<see cref="CorruptCellError"/>).</exception>
            private void Load(byte* cells)
            {
                long n = _n, t = 0, na = 0;
                bool defaultRead = false;
                // The NaN-like NAs fill the order buffer from its end backwards (reversed afterwards into input order):
                // the texts fill it from the front, so one buffer holds both without a second pass.
                for (long i = 0; i < n; i++)
                {
                    byte* cell = cells + i * _cellStride;
                    int r = NDVStringCells.Load(cell, _arena, out byte* buf, out long size);
                    if (r != 0)
                    {
                        if (r < 0)
                            throw CorruptCellError();
                        switch (_rule)
                        {
                            case VStringNARule.NaN:
                                _sizes[i] = -1;
                                _bufs[i] = null;
                                _order[n - 1 - na] = i;
                                na++;
                                continue;
                            case VStringNARule.Null:
                                throw NullCompareError();
                            default:
                                // No NA object, or a string one: the NA reads as the descriptor's default string.
                                buf = _defaultSize == 0 ? s_empty : _defaultText;
                                size = _defaultSize;
                                defaultRead = true;
                                _bufs[i] = buf;
                                _sizes[i] = size;
                                _key0[i] = KeyAt(buf, size, 0);
                                _order[t++] = i;
                                continue;
                        }
                    }
                    _bufs[i] = buf;
                    _sizes[i] = size;
                    _key0[i] = FirstKey(cell, buf, size);
                    _order[t++] = i;
                }
                // The NAs were stored from the end backwards: reverse them into input order behind the texts.
                for (long lo = t, hi = n - 1; lo < hi; lo++, hi--)
                    (_order[lo], _order[hi]) = (_order[hi], _order[lo]);
                _textCount = t;
                _naCount = na;
                _defaultReadNA = defaultRead;
            }

            // ---------------------------------------------------------------- stable order

            /// <summary>
            ///     Orders the line's texts stably (the first <see cref="_textCount"/> slots of <see cref="_order"/>): a radix sort
            ///     of their first-window keys, then the tie pass over each run of equal keys. The NaN-like NAs stay behind them
            ///     in input order.
            /// </summary>
            private void StableOrder()
            {
                long t = _textCount;
                if (t <= 1)
                    return;
                ulong varying = 0, all = ulong.MaxValue;
                for (long j = 0; j < t; j++)
                {
                    ulong k = _key0[_order[j]];
                    _keys[j] = k;
                    varying |= k;
                    all &= k;
                }
                long* sorted = RadixSort.ArgSortU64(_keys, _keyTmp, _order, _idxTmp, t, _hist, varying ^ all, out ulong* sortedKeys);
                if (sorted != _order)
                    Buffer.MemoryCopy(sorted, _order, t * sizeof(long), t * sizeof(long));
                if (sortedKeys != _keys)
                    Buffer.MemoryCopy(sortedKeys, _keys, t * sizeof(ulong), t * sizeof(ulong));
                TieRuns(0, t, 8);
            }

            /// <summary>
            ///     Runs the tie pass over every run of equal keys in <c>_keys[s, s + len)</c> (the keys of <c>_order</c>'s slots
            ///     there, sorted), whose texts are equal before byte <paramref name="covered"/>.
            /// </summary>
            /// <param name="s">The first slot.</param>
            /// <param name="len">The number of slots.</param>
            /// <param name="covered">The bytes the keys covered (the next window starts there).</param>
            /// <remarks>A run's own pass rewrites the key buffers only inside the run, and the scan below has already moved
            ///     past it, so one set of buffers serves every level.</remarks>
            private void TieRuns(long s, long len, long covered)
            {
                long end = s + len;
                long start = s;
                while (start < end)
                {
                    ulong k = _keys[start];
                    long stop = start + 1;
                    while (stop < end && _keys[stop] == k)
                        stop++;
                    if (stop - start >= 2)
                        TieSort(start, stop - start, covered);
                    start = stop;
                }
            }

            /// <summary>
            ///     Orders a run of texts whose keys were equal over the first <paramref name="covered"/> bytes, stably: insertion
            ///     by the remaining bytes for a short run; otherwise the texts that END within the covered bytes first, by size
            ///     (equal keys make each a prefix of the longer texts), then the others keyed on their next window, radix-sorted
            ///     and tie-passed again.
            /// </summary>
            /// <param name="s">The run's first slot of <see cref="_order"/>.</param>
            /// <param name="len">The run's length (at least 2).</param>
            /// <param name="covered">The bytes the run's keys covered.</param>
            private void TieSort(long s, long len, long covered)
            {
                long* run = _order + s;
                if (len <= TieInsertionMax)
                {
                    InsertionFrom(run, len, covered);
                    return;
                }
                long enders = 0;
                for (long k = 0; k < len; k++)
                    if (_sizes[run[k]] <= covered)
                        enders++;
                if (enders == len)
                {
                    SortBySize(s, len);
                    return;
                }
                if (enders > 0)
                {
                    // Stable partition: the enders to the front (through the index scratch), then sorted by size.
                    long* tmp = _idxTmp + s;
                    long e = 0, c = enders;
                    for (long k = 0; k < len; k++)
                    {
                        long index = run[k];
                        if (_sizes[index] <= covered)
                            tmp[e++] = index;
                        else
                            tmp[c++] = index;
                    }
                    Buffer.MemoryCopy(tmp, run, len * sizeof(long), len * sizeof(long));
                    if (enders >= 2)
                        SortBySize(s, enders);
                }
                // The continuers: keyed on the next window, radix-sorted inside their slots.
                long cs = s + enders, cl = len - enders;
                ulong varying = 0, all = ulong.MaxValue;
                for (long k = 0; k < cl; k++)
                {
                    long index = _order[cs + k];
                    ulong key = KeyAt(_bufs[index], _sizes[index], covered);
                    _keys[cs + k] = key;
                    varying |= key;
                    all &= key;
                }
                if (cl >= 2)
                {
                    long* sorted = RadixSort.ArgSortU64(_keys + cs, _keyTmp + cs, _order + cs, _idxTmp + cs, cl, _hist,
                        varying ^ all, out ulong* sortedKeys);
                    if (sorted != _order + cs)
                        Buffer.MemoryCopy(sorted, _order + cs, cl * sizeof(long), cl * sizeof(long));
                    if (sortedKeys != _keys + cs)
                        Buffer.MemoryCopy(sortedKeys, _keys + cs, cl * sizeof(ulong), cl * sizeof(ulong));
                    TieRuns(cs, cl, covered + 8);
                }
            }

            /// <summary>Stable insertion sort of a short run by <see cref="CompareFrom"/> from <paramref name="covered"/>.</summary>
            /// <param name="run">The run's first slot.</param>
            /// <param name="len">The run's length.</param>
            /// <param name="covered">The bytes known equal.</param>
            private void InsertionFrom(long* run, long len, long covered)
            {
                for (long i = 1; i < len; i++)
                {
                    long v = run[i];
                    byte* vb = _bufs[v];
                    long vs = _sizes[v];
                    long j = i - 1;
                    // Strict less: equal texts keep their order (stable).
                    while (j >= 0 && CompareFrom(vb, vs, _bufs[run[j]], _sizes[run[j]], covered) < 0)
                    {
                        run[j + 1] = run[j];
                        j--;
                    }
                    run[j + 1] = v;
                }
            }

            /// <summary>Stable sort of a run by text size (texts that are prefixes of each other: the shorter first).</summary>
            /// <param name="s">The run's first slot of <see cref="_order"/>.</param>
            /// <param name="len">The run's length.</param>
            private void SortBySize(long s, long len)
            {
                long* run = _order + s;
                if (len <= 64)
                {
                    for (long i = 1; i < len; i++)
                    {
                        long v = run[i];
                        long vs = _sizes[v];
                        long j = i - 1;
                        while (j >= 0 && _sizes[run[j]] > vs)
                        {
                            run[j + 1] = run[j];
                            j--;
                        }
                        run[j + 1] = v;
                    }
                    return;
                }
                ulong varying = 0, all = ulong.MaxValue;
                for (long k = 0; k < len; k++)
                {
                    ulong size = (ulong)_sizes[run[k]];
                    _keys[s + k] = size;
                    varying |= size;
                    all &= size;
                }
                long* sorted = RadixSort.ArgSortU64(_keys + s, _keyTmp + s, run, _idxTmp + s, len, _hist, varying ^ all, out _);
                if (sorted != run)
                    Buffer.MemoryCopy(sorted, run, len * sizeof(long), len * sizeof(long));
            }

            // ---------------------------------------------------------------- ranks and NumPy's tie order

            /// <summary>
            ///     The dense ranks of the line's elements in the stable order (by element index): equal texts share a rank, the
            ///     NaN-like NAs take <see cref="NARank"/>.
            /// </summary>
            /// <returns>The number of distinct values (the NAs counting as one).</returns>
            private long ComputeRanks()
            {
                long t = _textCount, distinct = 0;
                long prev = -1;
                for (long p = 0; p < t; p++)
                {
                    long index = _order[p];
                    if (prev < 0 || !SameText(prev, index))
                        distinct++;
                    _ranks[index] = distinct - 1;
                    prev = index;
                }
                for (long p = t; p < _n; p++)
                    _ranks[_order[p]] = NARank;
                return distinct + (_naCount > 0 ? 1 : 0);
            }

            /// <summary>Whether two text elements hold the same text (equal first keys, sizes and remaining bytes).</summary>
            /// <param name="a">An element index.</param>
            /// <param name="b">Another element index.</param>
            /// <returns>True for equal texts.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private bool SameText(long a, long b)
            {
                long size = _sizes[a];
                if (size != _sizes[b] || _key0[a] != _key0[b])
                    return false;
                return size <= 8 || NDVStringCells.Order(_bufs[a] + 8, size - 8, _bufs[b] + 8, size - 8) == 0;
            }

            /// <summary>
            ///     Whether the default kind's permutation can differ from the stable order: a line of more than
            ///     <c>SmallQuicksort + 1</c> elements (NumPy partitions it) with two equal values. Computes the ranks.
            /// </summary>
            /// <returns>True when NumPy's introsort port must run.</returns>
            private bool NeedsIntrosort()
            {
                long n = _n;
                if (ShortcutDisabled)
                {
                    ComputeRanks();
                    return true;
                }
                if (n <= SmallQuicksort + 1)
                {
                    ShortcutRuns++;
                    return false;
                }
                long distinct = ComputeRanks();
                // Every NaN-like NA shares one rank: two of them are a tie.
                if (distinct == n && _naCount <= 1)
                {
                    ShortcutRuns++;
                    return false;
                }
                return true;
            }

            /// <summary>
            ///     NumPy's introsort permutation of the line (the ranks computed): (rank, index) records in input order run
            ///     through <see cref="TextSort.IntroArgSortRecords"/>, NumPy's <c>npy_aquicksort_impl</c> statement for
            ///     statement (its extra <c>pi &lt; pj</c> scan guards stop where the unguarded scans stop under a strict weak
            ///     order, which ranks are).
            /// </summary>
            /// <returns>The records (indices in the odd slots, in NumPy's sorted order).</returns>
            private long* IntrosortPermutation()
            {
                long n = _n;
                if (_rec == null)
                    _rec = (ulong*)SizeBucketedBufferPool.Take(2 * _wordBytes);
                for (long i = 0; i < n; i++)
                {
                    _rec[2 * i] = (ulong)_ranks[i];
                    _rec[2 * i + 1] = (ulong)i;
                }
                PortRuns++;
                TextSort.IntroArgSortRecords(_rec, n);
                return (long*)_rec;
            }

            // ---------------------------------------------------------------- NumPy's timsort (npy_atimsort)

            /// <summary>
            ///     NumPy's <c>_compare</c> of two elements through their ranks: a NaN-like NA as the FIRST operand is greater (1)
            ///     — even against another NA —, as the second operand smaller (-1); texts by rank.
            /// </summary>
            /// <param name="a">An element index.</param>
            /// <param name="b">Another element index.</param>
            /// <returns>-1, 0 or 1.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int Cmp(long a, long b)
            {
                long ra = _ranks[a], rb = _ranks[b];
                if (ra == NARank)
                    return 1;
                if (rb == NARank)
                    return -1;
                return ra < rb ? -1 : ra > rb ? 1 : 0;
            }

            /// <summary>NumPy's <c>compute_min_run_short</c>.</summary>
            /// <param name="num">The number of elements.</param>
            /// <returns>The minimum run length.</returns>
            private static long ComputeMinRunShort(long num)
            {
                long r = 0;
                while (16 < num)
                {
                    r |= num & 1;
                    num >>= 1;
                }
                return num + r;
            }

            /// <summary>
            ///     NumPy's <c>npy_atimsort</c> (timsort.cpp) over <paramref name="tosort"/>, statement for statement, with
            ///     <see cref="Cmp"/> as its compare function: it re-sorts the index array it receives (the running permutation).
            /// </summary>
            /// <param name="tosort">The index array (rewritten).</param>
            /// <param name="num">Its length.</param>
            /// <param name="buffer">Scratch of at least <paramref name="num"/> slots (NumPy's growing <c>buffer_intp</c>; every
            ///     merge needs at most the smaller run).</param>
            private void TimSort(long* tosort, long num, long* buffer)
            {
                long* runS = stackalloc long[TimRunStack];
                long* runL = stackalloc long[TimRunStack];
                long stackPtr = 0;
                long minrun = ComputeMinRunShort(num);
                for (long l = 0; l < num;)
                {
                    long n = CountRun(tosort, l, num, minrun);
                    runS[stackPtr] = l;
                    runL[stackPtr] = n;
                    ++stackPtr;
                    TryCollapse(tosort, runS, runL, ref stackPtr, buffer);
                    l += n;
                }
                ForceCollapse(tosort, runS, runL, ref stackPtr, buffer);
            }

            /// <summary>NumPy's <c>npy_acount_run</c>: the natural run starting at <paramref name="l"/> (a strictly descending
            ///     one reversed), extended to <paramref name="minrun"/> by insertion.</summary>
            /// <param name="tosort">The index array.</param>
            /// <param name="l">The run's first slot.</param>
            /// <param name="num">The array's length.</param>
            /// <param name="minrun">The minimum run length.</param>
            /// <returns>The run's length.</returns>
            private long CountRun(long* tosort, long l, long num, long minrun)
            {
                if (num - l == 1)
                    return 1;
                long* pl = tosort + l;
                long* pi;
                if (Cmp(pl[0], pl[1]) <= 0)
                {
                    // (not strictly) ascending
                    for (pi = pl + 1; pi < tosort + num - 1 && Cmp(pi[0], pi[1]) <= 0; ++pi)
                    {
                    }
                }
                else
                {
                    // (strictly) descending
                    for (pi = pl + 1; pi < tosort + num - 1 && Cmp(pi[1], pi[0]) < 0; ++pi)
                    {
                    }
                    for (long* pj = pl, pr = pi; pj < pr; ++pj, --pr)
                        (*pj, *pr) = (*pr, *pj);
                }
                ++pi;
                long sz = pi - pl;
                if (sz < minrun)
                {
                    sz = l + minrun < num ? minrun : num - l;
                    long* pr = pl + sz;
                    // insertion sort
                    for (; pi < pr; ++pi)
                    {
                        long vi = *pi;
                        long* pj = pi;
                        while (pl < pj && Cmp(vi, pj[-1]) < 0)
                        {
                            *pj = pj[-1];
                            --pj;
                        }
                        *pj = vi;
                    }
                }
                return sz;
            }

            /// <summary>NumPy's <c>npy_agallop_left</c>: where <paramref name="key"/> belongs in a sorted run, before its
            ///     equals.</summary>
            /// <param name="tosort">The run's first slot.</param>
            /// <param name="size">The run's length.</param>
            /// <param name="key">The element searched for.</param>
            /// <returns>The insertion slot.</returns>
            private long GallopLeft(long* tosort, long size, long key)
            {
                if (Cmp(tosort[size - 1], key) < 0)
                    return size;
                long lastOfs = 0, ofs = 1;
                for (;;)
                {
                    if (size <= ofs || ofs < 0)
                    {
                        ofs = size;
                        break;
                    }
                    if (Cmp(tosort[size - ofs - 1], key) < 0)
                        break;
                    lastOfs = ofs;
                    ofs = (ofs << 1) + 1;
                }
                long l = size - ofs - 1, r = size - lastOfs - 1;
                while (l + 1 < r)
                {
                    long m = l + ((r - l) >> 1);
                    if (Cmp(tosort[m], key) < 0)
                        l = m;
                    else
                        r = m;
                }
                return r;
            }

            /// <summary>NumPy's <c>npy_agallop_right</c>: where <paramref name="key"/> belongs in a sorted run, after its
            ///     equals.</summary>
            /// <param name="tosort">The run's first slot.</param>
            /// <param name="size">The run's length.</param>
            /// <param name="key">The element searched for.</param>
            /// <returns>The insertion slot.</returns>
            private long GallopRight(long* tosort, long size, long key)
            {
                if (Cmp(key, tosort[0]) < 0)
                    return 0;
                long lastOfs = 0, ofs = 1;
                for (;;)
                {
                    if (size <= ofs || ofs < 0)
                    {
                        ofs = size;
                        break;
                    }
                    if (Cmp(key, tosort[ofs]) < 0)
                        break;
                    lastOfs = ofs;
                    ofs = (ofs << 1) + 1;
                }
                while (lastOfs + 1 < ofs)
                {
                    long m = lastOfs + ((ofs - lastOfs) >> 1);
                    if (Cmp(key, tosort[m]) < 0)
                        ofs = m;
                    else
                        lastOfs = m;
                }
                return ofs;
            }

            /// <summary>NumPy's <c>npy_amerge_left</c>: merges the run at <paramref name="p1"/> (copied to
            ///     <paramref name="p3"/>) with the following run, from the left.</summary>
            /// <param name="p1">The first run (its first element belongs after <paramref name="p2"/>'s first).</param>
            /// <param name="l1">Its length.</param>
            /// <param name="p2">The second run.</param>
            /// <param name="l2">Its length.</param>
            /// <param name="p3">Scratch of <paramref name="l1"/> slots.</param>
            private void MergeLeft(long* p1, long l1, long* p2, long l2, long* p3)
            {
                long* end = p2 + l2;
                Buffer.MemoryCopy(p1, p3, l1 * sizeof(long), l1 * sizeof(long));
                // first element must be in p2 otherwise skipped in the caller
                *p1++ = *p2++;
                while (p1 < p2 && p2 < end)
                {
                    if (Cmp(*p2, *p3) < 0)
                        *p1++ = *p2++;
                    else
                        *p1++ = *p3++;
                }
                if (p1 != p2)
                    Buffer.MemoryCopy(p3, p1, (p2 - p1) * sizeof(long), (p2 - p1) * sizeof(long));
            }

            /// <summary>NumPy's <c>npy_amerge_right</c>: merges the run at <paramref name="p1"/> with the following run (copied
            ///     to <paramref name="p3"/>), from the right.</summary>
            /// <param name="p1">The first run.</param>
            /// <param name="l1">Its length.</param>
            /// <param name="p2">The second run (its last element belongs before <paramref name="p1"/>'s last).</param>
            /// <param name="l2">Its length.</param>
            /// <param name="p3">Scratch of <paramref name="l2"/> slots.</param>
            private void MergeRight(long* p1, long l1, long* p2, long l2, long* p3)
            {
                long* start = p1 - 1;
                Buffer.MemoryCopy(p2, p3, l2 * sizeof(long), l2 * sizeof(long));
                p1 += l1 - 1;
                p2 += l2 - 1;
                p3 += l2 - 1;
                // first element must be in p1 otherwise skipped in the caller
                *p2-- = *p1--;
                while (p1 < p2 && start < p1)
                {
                    if (Cmp(*p3, *p1) < 0)
                        *p2-- = *p1--;
                    else
                        *p2-- = *p3--;
                }
                if (p1 != p2)
                {
                    long ofs = p2 - start;
                    Buffer.MemoryCopy(p3 - ofs + 1, start + 1, ofs * sizeof(long), ofs * sizeof(long));
                }
            }

            /// <summary>NumPy's <c>npy_amerge_at</c>: merges the runs at stack entries <paramref name="at"/> and
            ///     <paramref name="at"/> + 1, galloping past the elements already in place.</summary>
            /// <param name="tosort">The index array.</param>
            /// <param name="runS">The runs' first slots.</param>
            /// <param name="runL">The runs' lengths.</param>
            /// <param name="at">The first run's stack entry.</param>
            /// <param name="buffer">The merge scratch.</param>
            private void MergeAt(long* tosort, long* runS, long* runL, long at, long* buffer)
            {
                long s1 = runS[at], l1 = runL[at], s2 = runS[at + 1], l2 = runL[at + 1];
                // tosort[s2] belongs to tosort[s1+k]
                long k = GallopRight(tosort + s1, l1, tosort[s2]);
                if (l1 == k)
                    return; // already sorted
                long* p1 = tosort + s1 + k;
                l1 -= k;
                long* p2 = tosort + s2;
                // tosort[s2-1] belongs to tosort[s2+l2]
                l2 = GallopLeft(tosort + s2, l2, tosort[s2 - 1]);
                if (l2 < l1)
                    MergeRight(p1, l1, p2, l2, buffer);
                else
                    MergeLeft(p1, l1, p2, l2, buffer);
            }

            /// <summary>NumPy's <c>npy_atry_collapse</c>: merges runs while the stack's invariants are broken.</summary>
            /// <param name="tosort">The index array.</param>
            /// <param name="runS">The runs' first slots.</param>
            /// <param name="runL">The runs' lengths.</param>
            /// <param name="stackPtr">The stack's height (updated).</param>
            /// <param name="buffer">The merge scratch.</param>
            private void TryCollapse(long* tosort, long* runS, long* runL, ref long stackPtr, long* buffer)
            {
                long top = stackPtr;
                while (1 < top)
                {
                    long b = runL[top - 2];
                    long c = runL[top - 1];
                    if ((2 < top && runL[top - 3] <= b + c) || (3 < top && runL[top - 4] <= runL[top - 3] + b))
                    {
                        long a = runL[top - 3];
                        if (a <= c)
                        {
                            MergeAt(tosort, runS, runL, top - 3, buffer);
                            runL[top - 3] += b;
                            runS[top - 2] = runS[top - 1];
                            runL[top - 2] = runL[top - 1];
                            --top;
                        }
                        else
                        {
                            MergeAt(tosort, runS, runL, top - 2, buffer);
                            runL[top - 2] += c;
                            --top;
                        }
                    }
                    else if (1 < top && b <= c)
                    {
                        MergeAt(tosort, runS, runL, top - 2, buffer);
                        runL[top - 2] += c;
                        --top;
                    }
                    else
                        break;
                }
                stackPtr = top;
            }

            /// <summary>NumPy's <c>npy_aforce_collapse</c>: merges every pending run.</summary>
            /// <param name="tosort">The index array.</param>
            /// <param name="runS">The runs' first slots.</param>
            /// <param name="runL">The runs' lengths.</param>
            /// <param name="stackPtr">The stack's height.</param>
            /// <param name="buffer">The merge scratch.</param>
            private void ForceCollapse(long* tosort, long* runS, long* runL, ref long stackPtr, long* buffer)
            {
                long top = stackPtr;
                while (2 < top)
                {
                    if (runL[top - 3] <= runL[top - 1])
                    {
                        MergeAt(tosort, runS, runL, top - 3, buffer);
                        runL[top - 3] += runL[top - 2];
                        runS[top - 2] = runS[top - 1];
                        runL[top - 2] = runL[top - 1];
                        --top;
                    }
                    else
                    {
                        MergeAt(tosort, runS, runL, top - 2, buffer);
                        runL[top - 2] += runL[top - 1];
                        --top;
                    }
                }
                if (1 < top)
                    MergeAt(tosort, runS, runL, top - 2, buffer);
                stackPtr = top;
            }
        }
    }
}
