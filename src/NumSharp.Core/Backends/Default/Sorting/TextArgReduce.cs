using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// TextArgReduce.cs — np.argmax / np.argmin of fixed-width text (S / U): NumPy's STRING_argmax / UNICODE_argmax answers
// =============================================================================
//
// WHAT NUMPY DOES (2.4.2, arraytypes.c.src; probed)
//   PyArray_ArgMax / PyArray_ArgMin move the reduced axis last, make the array C-contiguous (a copy unless it already is)
//   and call the dtype's argmax / argmin once per line: the running best is copied aside and every later element is
//   compared with it by STRING_compare / UNICODE_compare (unsigned code units over the whole item width, the zero padding
//   read like any unit); a STRICTLY greater element (argmax) or strictly smaller one (argmin) replaces it. So the answer is
//   the index of the FIRST extreme element of each line. axis=None ravels in C order first. np.nanargmax / np.nanargmin of
//   text are argmax / argmin (_replace_nan leaves an array that is not inexact alone).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The first extreme element of a line is the same whatever order the comparisons are made in, so a line is split into
//   BLOCKS of up to 1,024 elements, the first extreme element of each block is found, and the block winners are merged in
//   line order with NumPy's strict comparison (a later winner replaces the running best only when it is strictly more
//   extreme, so ties keep the earlier element: the first occurrence).
//   Inside a block, the common prefix of its strings is measured first (whole-vector compares against the block's first
//   element, inline; the exact length only on a mismatch; stopping as soon as it is empty — for most text after the first
//   two elements): every element agrees on those units, so the first unit that can differ is the one after them (a block
//   of URLs sharing "https://site.example/" is decided by the units after it). Then one of two filters finds the block's
//   winner:
//     * GATHER (U, with AVX2): the unit after the prefix of eight elements at a time is gathered into one vector and
//       compared with the running best's; only elements whose unit equals or beats the best's are looked at, one by one —
//       a larger unit replaces the best, an equal one is decided by the units after it (an element of at most eight units
//       in ONE masked 256-bit compare against the best, held in a register). One gather a group of eight, with no key
//       written: about a fifth of what keying a U element costs (it narrows 32 bytes to a 64-bit key). Text whose best
//       string recurs (categorical data) pays a full comparison per copy here, so once a block finds one element in eight
//       equal to its best, the call's later blocks take the KEYS filter.
//     * KEYS (S, U without AVX2, tie-heavy U): each element gets an order-preserving 64-bit key of the units after the
//       prefix (TextKeyer: S 8 bytes; U 8 units of 8 bits, else 16 / 21 / 32 bits) and the first index of the extreme key
//       is found with vector loads. When the keys cover the strings to their ends they ARE the strings' order; otherwise
//       the elements tied on the extreme key are keyed again from the units after it, until one candidate is left or the
//       keys reach the end of the strings (the remaining candidates are then equal, and the first of them wins).
//   Short lines (at most 16 elements: rows of a 2-D array, an argmax along a short axis) are keyed many rows per kernel
//   call when they lie end to end (from the chunk's common prefix on); each row's extreme key is found by a branch-free
//   scan, ties resolved by comparing the units after the keys.
//   A line of fresh text at 100K elements reads every cache line of it (each element's first differing unit, or all of
//   it, must be compared): those cells run at the memory's bandwidth, as NumPy's loop does.
//
// LAYOUTS
//   Lines along an axis are visited by TextSort's line driver (NDIter over every axis but the line's, EXTERNAL_LOOP; the
//   trivial-loop bypass up to two dimensions), reading each line through its own stride: no contiguous copy, whatever the
//   layout (stepped, reversed, transposed, F-ordered). axis=None over a C-contiguous array, or any 1-D view, is one line.
//   Over any other N-D layout NumPy's index is the C-order index, but the lines are walked along the axis whose elements are
//   closest in MEMORY (an F-ordered matrix by its columns): the first extreme element of a line along any axis is its first
//   in C order too (the C index grows with every coordinate), so the line winners are merged by value first and by C index
//   on a tie.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     <c>np.argmax</c> / <c>np.argmin</c> of fixed-width text with NumPy 2.4.2's answers: the index of the first extreme
    ///     element of each line (see the file header for the algorithm).
    /// </summary>
    internal static unsafe class TextArgReduce
    {
        /// <summary>
        ///     The elements a long line is handled in at a time: the common prefix is measured per block (a block of 1,024
        ///     elements stays in L1/L2 while it is read twice), and 1,024 keys (8 KB) stay in L1 while their extreme and its
        ///     candidates are found. The per-block work (a prefix check, a merge comparison) is a few nanoseconds.
        /// </summary>
        internal const int BlockSize = 1024;

        /// <summary>
        ///     The longest line the short-row route handles: rows of at most 16 elements are keyed many rows per kernel call
        ///     and scanned with scalar code, where the block route's per-call work would dominate.
        /// </summary>
        internal const int ShortLineMax = 16;

        /// <summary>Test hook: the number of blocks the KEYS filter handled.</summary>
        internal static long BlockRuns;

        /// <summary>Test hook: the number of blocks the GATHER filter handled.</summary>
        internal static long GatherRuns;

        /// <summary>Test hook: the number of rows the short-row route handled.</summary>
        internal static long ShortRowRuns;

        /// <summary>Test hook: the number of refinement passes (candidates keyed again from the units after a tied key).</summary>
        internal static long RefineRuns;

        /// <summary>
        ///     Test hook: when true, blocks are filtered from unit 0 instead of from the end of their common prefix (the
        ///     refinement passes and tie comparisons then do the prefix's work), so tests can hold both paths to one answer.
        /// </summary>
        internal static bool PrefixSkipDisabled;

        /// <summary>Test hook: when true, <c>U</c> blocks take the KEYS filter even where AVX2 gathers are available.</summary>
        internal static bool GatherDisabled;


        /// <summary>
        ///     <c>np.argmax</c> (or <c>np.argmin</c>) of an <c>S</c> / <c>U</c> array whose lines hold two or more elements
        ///     (the caller has run NumPy's checks — the axis, the empty-line refusal — and answered the shapes that compare
        ///     nothing).
        /// </summary>
        /// <param name="a">The text array (any layout).</param>
        /// <param name="axis">The reduced axis (validated, possibly negative), or null for the C-order flattened array.</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one (every axis for the flattened form).</param>
        /// <param name="max">True for argmax, false for argmin.</param>
        /// <returns>A fresh C-contiguous int64 array: 0-d for the flattened form without <paramref name="keepdims"/> (marked
        ///     as a reduction scalar, as the numeric argmax marks it), else the input's dimensions with the axis removed (or
        ///     kept with length one).</returns>
        /// <exception cref="OutOfMemoryException">The scratch cannot be allocated.</exception>
        [NDScoped]
        internal static NDArray ArgReduce(NDArray a, int? axis, bool keepdims, bool max)
        {
            int nd = a.ndim;
            int width = a.dtype.itemsize;
            int unit = a.dtype.kind == 'U' ? 4 : 1;
            if (axis is null || nd == 0)
            {
                long index = FlatIndex(a, unit, width, max);
                var flat = NDArray.Scalar(index);
                if (keepdims && nd > 0)
                {
                    var ones = new long[nd];
                    Array.Fill(ones, 1L);
                    flat.Storage.Reshape(Shape.Adopt(ones));
                }
                return flat.MarkReductionScalar();
            }

            int ax = axis.Value < 0 ? axis.Value + nd : axis.Value;
            long n = a.Shape.dimensions[ax];
            // The result laid out with the axis kept (length one), so that the line driver pairs every line of the input
            // with its slot; the axis is dropped afterwards by relabelling the same C-contiguous buffer.
            var keepDims = (long[])a.Shape.dimensions.Clone();
            keepDims[ax] = 1;
            var result = new NDArray(NPTypeCode.Int64, Shape.Adopt(keepDims), false);
            if (result.size > 0)
            {
                long stride = a.Shape.strides[ax] * (long)width;
                using var job = new Job(unit, width, n, stride, max);
                TextSort.DriveLines(new[] { a, result }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, ax,
                    job.ArgLines);
            }
            if (!keepdims)
            {
                var outDims = new long[nd - 1];
                for (int d = 0, w = 0; d < nd; d++)
                    if (d != ax)
                        outDims[w++] = a.Shape.dimensions[d];
                result.Storage.Reshape(outDims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(outDims));
            }
            return result.MarkReductionScalar();
        }

        /// <summary>
        ///     The C-order flat index of the first extreme element (<c>axis=None</c>; a 0-d input is its one element).
        /// </summary>
        /// <param name="a">The text array (at least one element).</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="max">True for argmax.</param>
        /// <returns>The index into the C-order ravel of <paramref name="a"/>.</returns>
        private static long FlatIndex(NDArray a, int unit, int width, bool max)
        {
            long size = a.size;
            if (size <= 1)
                return 0;
            int nd = a.ndim;
            byte* first = (byte*)a.Address + a.Shape.offset * (long)width;
            if (nd == 1 || a.Shape.IsContiguous)
            {
                // One line: a C-contiguous array in its memory order (C order IS its ravel), a 1-D view through its stride.
                long stride = nd == 1 ? a.Shape.strides[0] * (long)width : width;
                using var job = new Job(unit, width, size, stride, max);
                return job.LineWinner(first, stride, size);
            }

            // Any other N-D layout: lines along the axis whose elements are closest in memory (the smallest stride among
            // the axes longer than one; the later axis on a tie, so a C-ordered stack of views keeps its rows). Their
            // winners are merged by value and, on a tie, by C index (see the file header).
            int lineAxis = -1;
            long bestStride = long.MaxValue;
            for (int d = 0; d < nd; d++)
            {
                if (a.Shape.dimensions[d] <= 1)
                    continue;
                long s = Math.Abs(a.Shape.strides[d]);
                if (s <= bestStride)
                {
                    bestStride = s;
                    lineAxis = d;
                }
            }
            long lineLength = a.Shape.dimensions[lineAxis];
            long lineStride = a.Shape.strides[lineAxis] * (long)width;
            using var lines = new Job(unit, width, lineLength, lineStride, max);
            lines.StartFlat(a.Shape.dimensions, lineAxis);
            TextSort.DriveLines(new[] { a }, new[] { NDIterPerOpFlags.READONLY }, lineAxis, lines.FlatLines);
            return lines.FlatBestIndex;
        }

        /// <summary>
        ///     One arg-reduction call's state: its constants, the key scratch reused for every line, and — for the flattened
        ///     form over lines — the running best and the C-order odometer of the lines. Disposing returns the scratch.
        /// </summary>
        private sealed class Job : IDisposable
        {
            /// <summary>The code-unit size (1 or 4).</summary>
            private readonly int _unit;

            /// <summary>The element width in bytes.</summary>
            private readonly int _width;

            /// <summary>The element width in code units.</summary>
            private readonly int _units;

            /// <summary>The line length (every line of one call has it).</summary>
            private readonly long _n;

            /// <summary>The byte distance between consecutive elements of a line.</summary>
            private readonly long _stride;

            /// <summary>True for argmax, false for argmin.</summary>
            private readonly bool _max;

            /// <summary>
            ///     Whether <c>U</c> blocks take the GATHER filter: AVX2 is supported, the test hook allows it, and eight
            ///     elements' offsets from the first fit a 32-bit gather index.
            /// </summary>
            private readonly bool _gather;

            /// <summary>The prefix keys of the dtype.</summary>
            private readonly TextKeyer _keyer;

            /// <summary>The elements of the current GATHER block found equal to the running best (see <see cref="BlockWinner"/>).</summary>
            private long _ties;

            /// <summary>Set once a GATHER block found one element in eight equal to its best: the call's later blocks take the
            ///     KEYS filter.</summary>
            private bool _tieHeavy;

            /// <summary><see cref="BlockSize"/> keys plus vector padding: a block's (or a chunk of short rows') keys.</summary>
            private ulong* _keys;

            /// <summary><see cref="BlockSize"/> keys: the refinement keys of a block's candidates.</summary>
            private ulong* _keys2;

            /// <summary><see cref="BlockSize"/> indices: a block's candidates (elements whose key equals the extreme key).</summary>
            private long* _cand;

            /// <summary>The flattened form's running best element (null before the first line).</summary>
            private byte* _flatBest;

            /// <summary>The C-order index of <see cref="_flatBest"/>.</summary>
            internal long FlatBestIndex;

            /// <summary>The flattened form's kept axes, in C order (every axis but the line axis).</summary>
            private long[] _keptDims;

            /// <summary>The C-order element stride of each kept axis.</summary>
            private long[] _keptCStrides;

            /// <summary>The kept coordinates of the next line (the odometer the driver's C order advances).</summary>
            private long[] _coords;

            /// <summary>The C-order element stride of the line axis.</summary>
            private long _lineCStride;

            /// <summary>The bytes of each key buffer: <see cref="BlockSize"/> slots plus four of vector padding.</summary>
            private const long KeyBytes = (BlockSize + 4) * sizeof(ulong);

            /// <summary>The bytes of the candidate buffer.</summary>
            private const long CandBytes = BlockSize * sizeof(long);

            /// <summary>Allocates the scratch.</summary>
            /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
            /// <param name="width">The element width in bytes.</param>
            /// <param name="n">The line length.</param>
            /// <param name="stride">The byte distance between a line's elements.</param>
            /// <param name="max">True for argmax.</param>
            /// <exception cref="OutOfMemoryException">The scratch cannot be allocated (the partial allocation is returned).</exception>
            internal Job(int unit, int width, long n, long stride, bool max)
            {
                _unit = unit;
                _width = width;
                _units = width / unit;
                _n = n;
                _stride = stride;
                _max = max;
                _gather = unit == 4 && Avx2.IsSupported && !GatherDisabled && stride != long.MinValue
                          && Math.Abs(stride) <= int.MaxValue / 8;
                _keyer = new TextKeyer(unit, _units);
                try
                {
                    _keys = (ulong*)SizeBucketedBufferPool.Take(KeyBytes);
                    _keys2 = (ulong*)SizeBucketedBufferPool.Take(KeyBytes);
                    _cand = (long*)SizeBucketedBufferPool.Take(CandBytes);
                }
                catch
                {
                    Dispose();
                    throw;
                }
            }

            /// <summary>Returns the scratch to the buffer pool (idempotent).</summary>
            public void Dispose()
            {
                if (_keys != null) { SizeBucketedBufferPool.Return((IntPtr)_keys, KeyBytes); _keys = null; }
                if (_keys2 != null) { SizeBucketedBufferPool.Return((IntPtr)_keys2, KeyBytes); _keys2 = null; }
                if (_cand != null) { SizeBucketedBufferPool.Return((IntPtr)_cand, CandBytes); _cand = null; }
            }

            /// <summary>
            ///     Prepares the flattened form over lines along <paramref name="lineAxis"/>: the kept axes in C order, their C
            ///     strides, and the odometer at the first line.
            /// </summary>
            /// <param name="dims">The input's dimensions.</param>
            /// <param name="lineAxis">The axis the lines run along.</param>
            internal void StartFlat(long[] dims, int lineAxis)
            {
                int nd = dims.Length;
                var cStrides = new long[nd];
                long s = 1;
                for (int d = nd - 1; d >= 0; d--)
                {
                    cStrides[d] = s;
                    s *= dims[d];
                }
                _keptDims = new long[nd - 1];
                _keptCStrides = new long[nd - 1];
                for (int d = 0, w = 0; d < nd; d++)
                {
                    if (d == lineAxis)
                        continue;
                    _keptDims[w] = dims[d];
                    _keptCStrides[w++] = cStrides[d];
                }
                _coords = new long[nd - 1];
                _lineCStride = cStrides[lineAxis];
            }

            /// <summary>
            ///     The block kernel of an axis reduction (run by the line driver): operand 0 holds the text lines, operand 1 the
            ///     int64 result, one slot per line.
            /// </summary>
            /// <param name="dataptrs">The first line's element 0 in the text, and its result slot.</param>
            /// <param name="strides">The byte distance between consecutive lines in each operand.</param>
            /// <param name="count">The number of lines.</param>
            /// <param name="aux">Unused.</param>
            internal void ArgLines(void** dataptrs, long* strides, long count, void* aux)
            {
                byte* text = (byte*)dataptrs[0];
                byte* output = (byte*)dataptrs[1];
                long textStep = strides[0], outputStep = strides[1];
                if (_n <= ShortLineMax)
                {
                    ShortRows(text, textStep, count, output, outputStep);
                    return;
                }
                for (long r = 0; r < count; r++)
                    *(long*)(output + r * outputStep) = LineWinner(text + r * textStep, _stride, _n);
            }

            /// <summary>
            ///     The block kernel of the flattened form over lines (run by the line driver, which visits the kept axes in C
            ///     order): each line's winner is merged into the running best — a strictly more extreme element replaces it,
            ///     an equal one replaces it only with a smaller C index.
            /// </summary>
            /// <param name="dataptrs">The first line's element 0.</param>
            /// <param name="strides">The byte distance between consecutive lines.</param>
            /// <param name="count">The number of lines.</param>
            /// <param name="aux">Unused.</param>
            internal void FlatLines(void** dataptrs, long* strides, long count, void* aux)
            {
                byte* text = (byte*)dataptrs[0];
                long textStep = strides[0];
                if (_n <= ShortLineMax)
                {
                    // The short lines' winners land in the candidate buffer (as indices within their lines), a chunk at a time.
                    long chunk = Math.Max(1, BlockSize / _n);
                    for (long r0 = 0; r0 < count; r0 += chunk)
                    {
                        long rows = Math.Min(chunk, count - r0);
                        ShortRows(text + r0 * textStep, textStep, rows, (byte*)_cand, sizeof(long));
                        for (long r = 0; r < rows; r++)
                            MergeFlat(text + (r0 + r) * textStep, _cand[r]);
                    }
                    return;
                }
                for (long r = 0; r < count; r++)
                {
                    byte* line = text + r * textStep;
                    MergeFlat(line, LineWinner(line, _stride, _n));
                }
            }

            /// <summary>
            ///     Merges one line's winner into the flattened form's running best and advances the odometer to the next line.
            /// </summary>
            /// <param name="line">The line's element 0.</param>
            /// <param name="j">The winner's index within the line.</param>
            private void MergeFlat(byte* line, long j)
            {
                long cIndex = j * _lineCStride;
                for (int d = 0; d < _coords.Length; d++)
                    cIndex += _coords[d] * _keptCStrides[d];
                byte* candidate = line + j * _stride;
                if (_flatBest == null)
                {
                    _flatBest = candidate;
                    FlatBestIndex = cIndex;
                }
                else
                {
                    int c = CompareFrom(candidate, _flatBest, 0, _width, _unit);
                    if ((_max ? c > 0 : c < 0) || (c == 0 && cIndex < FlatBestIndex))
                    {
                        _flatBest = candidate;
                        FlatBestIndex = cIndex;
                    }
                }
                // The driver hands the lines over in C order of the kept axes: advance the odometer.
                for (int d = _coords.Length - 1; d >= 0; d--)
                {
                    if (++_coords[d] < _keptDims[d])
                        break;
                    _coords[d] = 0;
                }
            }

            /// <summary>
            ///     The index of the first extreme element of one line: the block winners merged in line order.
            /// </summary>
            /// <param name="text">The line's element 0.</param>
            /// <param name="stride">The byte distance between its elements (any sign).</param>
            /// <param name="n">Its length (at least 1).</param>
            /// <returns>The index of the first maximum (argmax) or minimum (argmin).</returns>
            internal long LineWinner(byte* text, long stride, long n)
            {
                if (n <= ShortLineMax)
                    return ShortRowWinner(text, stride, n);
                byte* best = null;
                long bestIndex = 0;
                for (long s = 0; s < n; s += BlockSize)
                {
                    long count = Math.Min(BlockSize, n - s);
                    byte* block = text + s * stride;
                    long j = BlockWinner(block, stride, count);
                    byte* candidate = block + j * stride;
                    // A later block's winner replaces the running best only when strictly more extreme: NumPy's first
                    // occurrence across blocks.
                    if (best == null || Beats(candidate, best, 0))
                    {
                        best = candidate;
                        bestIndex = s + j;
                    }
                }
                return bestIndex;
            }

            /// <summary>
            ///     The index of the first extreme element of one block: its common prefix skipped, then the GATHER filter
            ///     (<c>U</c> with AVX2) or the KEYS filter.
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="count">Its length (1 to <see cref="BlockSize"/>).</param>
            /// <returns>The index within the block.</returns>
            private long BlockWinner(byte* block, long stride, long count)
            {
                if (count == 1)
                    return 0;
                int prefix = PrefixSkipDisabled ? 0 : CommonPrefixUnits(block, stride, count);
                // Every element of the block is the same string: the first one wins.
                if (prefix >= _units)
                    return 0;
                if (!_gather || _tieHeavy || count <= 8)
                    return KeyedWinner(block, stride, count, prefix);
                _ties = 0;
                long winner = GatherWinner(block, stride, count, prefix);
                // Text whose best string recurs (categorical data: every copy of the best word ties with it on every unit)
                // spends the GATHER filter on a full comparison per copy and on the branches they mispredict, where the KEYS
                // filter finds the first of the equal keys with vector compares. Past one copy in eight elements, the call's
                // remaining blocks are keyed. (Ties that a later unit decides — a small alphabet — stay with the GATHER
                // filter, which decides them in one masked compare.)
                if (_ties > count / 8)
                    _tieHeavy = true;
                return winner;
            }

            /// <summary>
            ///     The GATHER filter (<c>U</c>, AVX2): unit <paramref name="prefix"/> of eight elements at a time is gathered
            ///     and compared with the running best's (unsigned, by flipping the sign bit); only elements whose unit equals
            ///     or beats it are looked at, in index order — a better unit replaces the best, an equal one is decided by the
            ///     units after it (NumPy's strict replacement, so the first extreme element stays).
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements (fits a 32-bit gather index eight times over).</param>
            /// <param name="count">Its length (more than 8).</param>
            /// <param name="prefix">The units every element of the block shares (below the width).</param>
            /// <returns>The index within the block.</returns>
            private long GatherWinner(byte* block, long stride, long count, int prefix)
            {
                GatherRuns++;
                // An element of at most eight units is compared on a tie as ONE masked 256-bit load against the running
                // best's (held in a register), inline; a longer one through CompareFrom. Two methods, so that neither tie
                // path's registers crowd the other's gather loop (one method holding both spilled the loop's state).
                return _width <= 32 ? GatherWinnerShort(block, stride, count, prefix) : GatherWinnerLong(block, stride, count, prefix);
            }

            /// <summary>
            ///     The first extreme element among <paramref name="m"/> candidates that agree on every unit before
            ///     <paramref name="from"/>: they are keyed again from that unit, and the ones tying on the extreme key are kept
            ///     in index order, until one is left or the keys reach the end of the strings (the remaining candidates are then
            ///     equal, and the first of them wins).
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="m">The number of candidates in <c>_cand</c> (at least 1), in index order.</param>
            /// <param name="from">The first unit not known equal.</param>
            /// <returns>The index within the block.</returns>
            private long RefineCandidates(byte* block, long stride, int m, int from)
            {
                while (m > 1 && from < _units)
                {
                    RefineRuns++;
                    int covered = _keyer.ComputeKeys(block, stride, _cand, m, from, _keys2);
                    from += covered;
                    ulong extreme = _max ? MaxKey(_keys2, m) : MinKey(_keys2, m);
                    // Keep the candidates that tie on the new keys, in index order (the first of them is the earliest).
                    int kept = 0;
                    for (int c = 0; c < m; c++)
                        if (_keys2[c] == extreme)
                            _cand[kept++] = _cand[c];
                    m = kept;
                }
                return _cand[0];
            }

            /// <summary>
            ///     The GATHER filter for elements of at most eight units (32 bytes): a tie on the gathered unit is decided by
            ///     comparing the whole candidate with the running best, both as masked 256-bit loads (<c>vpmaskmovd</c>: the
            ///     lanes past the width read as zero and never fault, so an element at the end of mapped memory is safe).
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements (fits a 32-bit gather index eight times over).</param>
            /// <param name="count">Its length (more than 8).</param>
            /// <param name="prefix">The units every element of the block shares (below the width).</param>
            /// <returns>The index within the block.</returns>
            private long GatherWinnerShort(byte* block, long stride, long count, int prefix)
            {
                bool lastUnit = prefix + 1 >= _units;
                byte* units = block + prefix * 4;
                uint bestUnit = Unsafe.ReadUnaligned<uint>(units);
                long bi = 0;
                var offsets = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7) * (int)stride;
                var bias = Vector256.Create(int.MinValue);
                // Lanes below the width load the element's units; the rest load zero in both operands and compare equal.
                var widthMask = Vector256.LessThan(Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7), Vector256.Create(_units));
                var bestVec = Avx2.MaskLoad((int*)block, widthMask);
                long i = 1;
                for (; i + 8 <= count; i += 8)
                {
                    var v = Avx2.GatherVector256((int*)(units + i * stride), offsets, 1) ^ bias;
                    var b = Vector256.Create((int)(bestUnit ^ 0x80000000u));
                    // Lanes whose unit can replace the best: strictly better when the gathered unit is the element's last
                    // (an equal one is an equal string, which never replaces), else equal or better (an equal unit is decided
                    // by the units after it).
                    uint mask = (lastUnit
                        ? (_max ? Vector256.GreaterThan(v, b) : Vector256.LessThan(v, b))
                        : (_max ? Vector256.GreaterThanOrEqual(v, b) : Vector256.LessThanOrEqual(v, b))).ExtractMostSignificantBits();
                    while (mask != 0)
                    {
                        int lane = BitOperations.TrailingZeroCount(mask);
                        mask &= mask - 1;
                        long idx = i + lane;
                        byte* e = block + idx * stride;
                        uint u = Unsafe.ReadUnaligned<uint>(units + idx * stride);
                        // The mask was taken against the group's starting best: an earlier lane of the group may have
                        // raised it since, so each lane is tested again against the current best.
                        if ((_max ? u > bestUnit : u < bestUnit)
                            || (u == bestUnit && !lastUnit && TieBeatsShort(e, block + bi * stride, bestVec, widthMask)))
                        {
                            bestUnit = u;
                            bi = idx;
                            bestVec = Avx2.MaskLoad((int*)e, widthMask);
                        }
                    }
                }
                for (; i < count; i++)
                {
                    byte* e = block + i * stride;
                    uint u = Unsafe.ReadUnaligned<uint>(units + i * stride);
                    if (_max ? u > bestUnit : u < bestUnit)
                    {
                        bestUnit = u;
                        bi = i;
                        bestVec = Avx2.MaskLoad((int*)e, widthMask);
                    }
                    else if (u == bestUnit && !lastUnit && TieBeatsShort(e, block + bi * stride, bestVec, widthMask))
                    {
                        bi = i;
                        bestVec = Avx2.MaskLoad((int*)e, widthMask);
                    }
                }
                return bi;
            }

            /// <summary>
            ///     Whether a candidate that ties with the running best on the gathered unit is strictly more extreme: the whole
            ///     element is compared in one masked load; the first differing unit decides, compared unsigned.
            /// </summary>
            /// <param name="candidate">The later element.</param>
            /// <param name="best">The running best.</param>
            /// <param name="bestVec">The running best's units, loaded with <paramref name="widthMask"/>.</param>
            /// <param name="widthMask">The lanes below the element width.</param>
            /// <returns>True when the candidate replaces the best (an equal element never does).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private bool TieBeatsShort(byte* candidate, byte* best, Vector256<int> bestVec, Vector256<int> widthMask)
            {
                var c = Avx2.MaskLoad((int*)candidate, widthMask);
                uint differ = ~Vector256.Equals(c, bestVec).ExtractMostSignificantBits() & 0xFFu;
                if (differ == 0)
                {
                    _ties++;
                    return false;
                }
                int lane = BitOperations.TrailingZeroCount(differ);
                uint cu = ((uint*)candidate)[lane], bu = ((uint*)best)[lane];
                return _max ? cu > bu : cu < bu;
            }

            /// <summary>A tie of the long GATHER filter, decided like <see cref="Beats"/>; an element equal to the best is
            ///     counted (see <see cref="BlockWinner"/>).</summary>
            /// <param name="candidate">The later element.</param>
            /// <param name="best">The running best.</param>
            /// <param name="from">The first byte compared (after the tied unit).</param>
            /// <returns>True when the candidate replaces the best.</returns>
            private bool TieBeatsLong(byte* candidate, byte* best, int from)
            {
                int c = CompareFrom(candidate, best, from, _width, _unit);
                if (c == 0)
                    _ties++;
                return _max ? c > 0 : c < 0;
            }

            /// <summary>
            ///     The GATHER filter for elements longer than eight units: a tie on the gathered unit is decided by the units
            ///     after it (<see cref="Beats"/>).
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements (fits a 32-bit gather index eight times over).</param>
            /// <param name="count">Its length (more than 8).</param>
            /// <param name="prefix">The units every element of the block shares (below the width).</param>
            /// <returns>The index within the block.</returns>
            private long GatherWinnerLong(byte* block, long stride, long count, int prefix)
            {
                int unitBytes = prefix * 4;
                bool lastUnit = prefix + 1 >= _units;
                byte* units = block + unitBytes;
                uint bestUnit = Unsafe.ReadUnaligned<uint>(units);
                long bi = 0;
                var offsets = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7) * (int)stride;
                var bias = Vector256.Create(int.MinValue);
                long i = 1;
                for (; i + 8 <= count; i += 8)
                {
                    var v = Avx2.GatherVector256((int*)(units + i * stride), offsets, 1) ^ bias;
                    var b = Vector256.Create((int)(bestUnit ^ 0x80000000u));
                    uint mask = (lastUnit
                        ? (_max ? Vector256.GreaterThan(v, b) : Vector256.LessThan(v, b))
                        : (_max ? Vector256.GreaterThanOrEqual(v, b) : Vector256.LessThanOrEqual(v, b))).ExtractMostSignificantBits();
                    while (mask != 0)
                    {
                        int lane = BitOperations.TrailingZeroCount(mask);
                        mask &= mask - 1;
                        long idx = i + lane;
                        uint u = Unsafe.ReadUnaligned<uint>(units + idx * stride);
                        // Tested again against the current best: an earlier lane of the group may have raised it.
                        if ((_max ? u > bestUnit : u < bestUnit)
                            || (u == bestUnit && !lastUnit && TieBeatsLong(block + idx * stride, block + bi * stride, unitBytes + 4)))
                        {
                            bestUnit = u;
                            bi = idx;
                        }
                    }
                }
                for (; i < count; i++)
                {
                    uint u = Unsafe.ReadUnaligned<uint>(units + i * stride);
                    if (_max ? u > bestUnit : u < bestUnit)
                    {
                        bestUnit = u;
                        bi = i;
                    }
                    else if (u == bestUnit && !lastUnit && TieBeatsLong(block + i * stride, block + bi * stride, unitBytes + 4))
                    {
                        bi = i;
                    }
                }
                return bi;
            }

            /// <summary>
            ///     The KEYS filter: keys from unit <paramref name="prefix"/> on, the first index of the extreme key, and the
            ///     candidates that tie on it refined by the next units until one is left or the strings end.
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="count">Its length.</param>
            /// <param name="prefix">The units every element of the block shares (below the width).</param>
            /// <returns>The index within the block.</returns>
            private long KeyedWinner(byte* block, long stride, long count, int prefix)
            {
                BlockRuns++;
                int covered = _keyer.ComputeKeys(block, stride, null, count, prefix, _keys);
                int from = prefix + covered;
                ulong extreme = _max ? MaxKey(_keys, count) : MinKey(_keys, count);
                long first = new ReadOnlySpan<ulong>(_keys, checked((int)count)).IndexOf(extreme);
                // Keys that reach the end of the strings are the strings' order: the first extreme key is the answer.
                if (from >= _units)
                    return first;
                int m = GatherEqual(_keys, count, extreme, first, _cand);
                return RefineCandidates(block, stride, m, from);
            }

            /// <summary>
            ///     Short rows: when the rows lie end to end they are keyed many rows per kernel call (a chunk of up to
            ///     <see cref="BlockSize"/> elements), else one row per call; each row's first extreme element is found by
            ///     <see cref="RowFromKeys"/> and written to its slot.
            /// </summary>
            /// <param name="text">The first row's element 0.</param>
            /// <param name="textStep">The byte distance between consecutive rows.</param>
            /// <param name="count">The number of rows.</param>
            /// <param name="output">The first row's slot (an int64 index within the row is written there).</param>
            /// <param name="outputStep">The byte distance between consecutive slots.</param>
            private void ShortRows(byte* text, long textStep, long count, byte* output, long outputStep)
            {
                long n = _n;
                if (n == 1 || _units == 0)
                {
                    // One element a row, or zero-width strings that are all equal: every row's winner is its element 0.
                    for (long r = 0; r < count; r++)
                        *(long*)(output + r * outputStep) = 0;
                    return;
                }
                if (count > 1 && textStep == n * _stride)
                {
                    long rowsPerChunk = Math.Max(1, BlockSize / n);
                    for (long r0 = 0; r0 < count; r0 += rowsPerChunk)
                    {
                        long rows = Math.Min(rowsPerChunk, count - r0);
                        byte* chunk = text + r0 * textStep;
                        // The chunk's strings share their first `prefix` units (most text: none): key from the unit after them.
                        int prefix = PrefixSkipDisabled ? 0 : CommonPrefixUnits(chunk, _stride, rows * n);
                        if (prefix >= _units)
                        {
                            // Every string of the chunk is the same: each row's winner is its element 0.
                            for (long r = 0; r < rows; r++)
                                *(long*)(output + (r0 + r) * outputStep) = 0;
                            continue;
                        }
                        int covered = prefix + _keyer.ComputeKeys(chunk, _stride, null, rows * n, prefix, _keys);
                        for (long r = 0; r < rows; r++)
                            *(long*)(output + (r0 + r) * outputStep) = RowFromKeys(_keys + r * n, chunk + r * textStep, _stride, n, covered);
                    }
                    ShortRowRuns += count;
                    return;
                }
                for (long r = 0; r < count; r++)
                    *(long*)(output + r * outputStep) = ShortRowWinner(text + r * textStep, _stride, n);
            }

            /// <summary>The first extreme element of one short row, keyed on its own.</summary>
            /// <param name="row">The row's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="n">Its length (1 to <see cref="ShortLineMax"/>).</param>
            /// <returns>The index within the row.</returns>
            private long ShortRowWinner(byte* row, long stride, long n)
            {
                if (n == 1 || _units == 0)
                    return 0;
                ShortRowRuns++;
                int covered = _keyer.ComputeKeys(row, stride, null, n, 0, _keys);
                return RowFromKeys(_keys, row, stride, n, covered);
            }

            /// <summary>
            ///     The first extreme element of a short row from its keys: a branch-free scan for the first extreme key (a
            ///     strictly better key replaces, an equal one only marks a tie); when the keys do not reach the end of the
            ///     strings, the elements tied on the extreme key are compared from the units after the keys.
            /// </summary>
            /// <param name="keys">The row's keys, one per element.</param>
            /// <param name="row">The row's element 0.</param>
            /// <param name="stride">The byte distance between the row's elements.</param>
            /// <param name="n">The row length.</param>
            /// <param name="covered">The units known equal or decided by the keys: every string agrees before the keys' first
            ///     unit, so ties on the keys are compared from this unit on.</param>
            /// <returns>The index within the row.</returns>
            private long RowFromKeys(ulong* keys, byte* row, long stride, long n, int covered)
            {
                // One branch-free pass of fixed length: a better key selects itself, its index and a cleared tie flag through
                // masks made from the comparisons (the JIT does not turn selects inside a loop into conditional moves, and the
                // branches they replace mispredict about once a row on random or categorical rows; a second pass over the
                // keys after the winner ends at a different index each row, and that exit mispredicted as often).
                // `tie` ends set when a key after the first extreme key equals it.
                ulong best = keys[0];
                long bi = 0;
                ulong tie = 0;
                for (long j = 1; j < n; j++)
                {
                    ulong k = keys[j];
                    bool better = _max ? k > best : k < best;
                    bool equal = k == best;
                    ulong take = 0UL - Unsafe.As<bool, byte>(ref better);
                    best = (k & take) | (best & ~take);
                    bi = (j & (long)take) | (bi & ~(long)take);
                    tie = (tie & ~take) | (0UL - Unsafe.As<bool, byte>(ref equal));
                }
                if (tie == 0 || covered >= _units)
                    return bi;
                // Keys that end before the strings do: the elements after the first extreme key that tie with it are decided
                // by the units after the keys, the earlier element winning equality.
                int from = covered * _unit;
                for (long j = bi + 1; j < n; j++)
                    if (keys[j] == best && Beats(row + j * stride, row + bi * stride, from))
                        bi = j;
                return bi;
            }

            /// <summary>
            ///     NumPy's strict replacement test: whether <paramref name="candidate"/> is strictly greater (argmax) or
            ///     strictly smaller (argmin) than <paramref name="best"/>, compared from byte <paramref name="from"/> on (the
            ///     bytes before it are known equal).
            /// </summary>
            /// <param name="candidate">The later element.</param>
            /// <param name="best">The running best.</param>
            /// <param name="from">The first byte compared (a unit boundary).</param>
            /// <returns>True when the candidate replaces the best.</returns>
            private bool Beats(byte* candidate, byte* best, int from)
            {
                int c = CompareFrom(candidate, best, from, _width, _unit);
                return _max ? c > 0 : c < 0;
            }

            /// <summary>
            ///     The common prefix of a block's strings, in code units: each element is compared with the block's first by
            ///     32-byte vector loads (an overlapping last load covers the rest), narrowing the prefix, until it is empty
            ///     (for most text after the first two elements) or the block ends.
            /// </summary>
            /// <param name="block">The block's element 0.</param>
            /// <param name="stride">The byte distance between its elements.</param>
            /// <param name="count">Its length.</param>
            /// <returns>The units every element of the block shares with the first (the width when all are equal).</returns>
            private int CommonPrefixUnits(byte* block, long stride, long count)
            {
                int bytes = _units * _unit;
                long i = 1;
                if (Vector256.IsHardwareAccelerated)
                {
                    // While the prefix spans a vector or more, an element that agrees with element 0 is confirmed by whole
                    // 256-bit compares inline (the last one overlapping); only a mismatch measures the new prefix exactly.
                    // A block of URLs sharing 84 bytes runs three compares an element here, where a call per element cost more
                    // than reading the element.
                    for (; i < count && bytes >= 32; i++)
                    {
                        byte* e = block + i * stride;
                        bool same = true;
                        int k = 0;
                        for (; k + 32 <= bytes; k += 32)
                        {
                            if (Vector256.Load(block + k) != Vector256.Load(e + k))
                            {
                                same = false;
                                break;
                            }
                        }
                        if (same && k < bytes && Vector256.Load(block + bytes - 32) != Vector256.Load(e + bytes - 32))
                            same = false;
                        if (!same)
                            bytes = CommonPrefixBytes(block, e, bytes);
                    }
                }
                for (; i < count && bytes >= _unit; i++)
                    bytes = CommonPrefixBytes(block, block + i * stride, bytes);
                return bytes / _unit;
            }
        }

        /// <summary>
        ///     The length of the common prefix of two byte ranges of <paramref name="length"/> bytes, with overlapping loads
        ///     so that no length takes a byte loop: 32-byte vectors (the last one overlapping the previous), else 16-byte
        ///     vectors, 8-byte words or 4-byte words the same way; single bytes only below 4.
        /// </summary>
        /// <param name="a">The first range.</param>
        /// <param name="b">The second range.</param>
        /// <param name="length">The bytes to compare.</param>
        /// <returns>The index of the first differing byte, or <paramref name="length"/>.</returns>
        /// <remarks>
        ///     An overlapping last load re-reads bytes already found equal, so a difference it reports is the first one at or
        ///     after its start — the first one overall, since every byte before its start was equal. A tie of two <c>U</c>
        ///     elements compares 28 bytes after the first unit: the byte loop it replaces cost three words and four bytes.
        /// </remarks>
        internal static int CommonPrefixBytes(byte* a, byte* b, int length)
        {
            if (Vector256.IsHardwareAccelerated && length >= 32)
            {
                int k = 0;
                for (; k + 32 <= length; k += 32)
                {
                    uint diff = ~Vector256.Equals(Vector256.Load(a + k), Vector256.Load(b + k)).ExtractMostSignificantBits();
                    if (diff != 0)
                        return k + BitOperations.TrailingZeroCount(diff);
                }
                if (k < length)
                {
                    // The last 32 bytes again, overlapping what was checked: only a difference at or past k is new.
                    int at = length - 32;
                    uint diff = ~Vector256.Equals(Vector256.Load(a + at), Vector256.Load(b + at)).ExtractMostSignificantBits();
                    if (diff != 0)
                        return at + BitOperations.TrailingZeroCount(diff);
                }
                return length;
            }
            if (Vector128.IsHardwareAccelerated && length >= 16)
            {
                int k = 0;
                for (; k + 16 <= length; k += 16)
                {
                    uint diff = ~Vector128.Equals(Vector128.Load(a + k), Vector128.Load(b + k)).ExtractMostSignificantBits() & 0xFFFFu;
                    if (diff != 0)
                        return k + BitOperations.TrailingZeroCount(diff);
                }
                if (k < length)
                {
                    int at = length - 16;
                    uint diff = ~Vector128.Equals(Vector128.Load(a + at), Vector128.Load(b + at)).ExtractMostSignificantBits() & 0xFFFFu;
                    if (diff != 0)
                        return at + BitOperations.TrailingZeroCount(diff);
                }
                return length;
            }
            if (length >= 8)
            {
                int k = 0;
                for (; k + 8 <= length; k += 8)
                {
                    ulong x = Unsafe.ReadUnaligned<ulong>(a + k) ^ Unsafe.ReadUnaligned<ulong>(b + k);
                    if (x != 0)
                        return k + BitOperations.TrailingZeroCount(x) / 8;
                }
                if (k < length)
                {
                    int at = length - 8;
                    ulong x = Unsafe.ReadUnaligned<ulong>(a + at) ^ Unsafe.ReadUnaligned<ulong>(b + at);
                    if (x != 0)
                        return at + BitOperations.TrailingZeroCount(x) / 8;
                }
                return length;
            }
            if (length >= 4)
            {
                uint x = Unsafe.ReadUnaligned<uint>(a) ^ Unsafe.ReadUnaligned<uint>(b);
                if (x != 0)
                    return BitOperations.TrailingZeroCount(x) / 8;
                int at = length - 4;
                x = Unsafe.ReadUnaligned<uint>(a + at) ^ Unsafe.ReadUnaligned<uint>(b + at);
                return x != 0 ? at + BitOperations.TrailingZeroCount(x) / 8 : length;
            }
            for (int i = 0; i < length; i++)
                if (a[i] != b[i])
                    return i;
            return length;
        }

        /// <summary>
        ///     NumPy's <c>STRING_compare</c> / <c>UNICODE_compare</c> of two elements of one width from byte
        ///     <paramref name="from"/> on: the first differing byte is found with vector compares
        ///     (<see cref="CommonPrefixBytes"/>, inlined — a span-based compare cost a call and two span constructions per
        ///     tie), and its code unit decides, compared unsigned.
        /// </summary>
        /// <param name="a">The first element.</param>
        /// <param name="b">The second element.</param>
        /// <param name="from">The first byte compared (a unit boundary; the bytes before it are known equal).</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int CompareFrom(byte* a, byte* b, int from, int width, int unit)
        {
            if (from >= width)
                return 0;
            // An element of up to 64 bytes is compared from its first byte: the bytes before `from` are equal, so the first
            // difference is the same, and whole-width loads (one 32-byte vector for a U8 element) beat a ragged remainder.
            int start = width <= 64 ? 0 : from;
            int d = start + CommonPrefixBytes(a + start, b + start, width - start);
            if (d >= width)
                return 0;
            if (unit == 1)
                return a[d] < b[d] ? -1 : 1;
            int u = d & ~3;
            return Unsafe.ReadUnaligned<uint>(a + u) < Unsafe.ReadUnaligned<uint>(b + u) ? -1 : 1;
        }

        /// <summary>The largest of <paramref name="count"/> keys (unsigned), with 256-bit vectors where they are accelerated.</summary>
        /// <param name="keys">The keys.</param>
        /// <param name="count">The number of keys (at least 1).</param>
        /// <returns>The maximum key.</returns>
        /// <remarks>
        ///     AVX2 has no unsigned 64-bit compare, and <c>Vector256.Max&lt;ulong&gt;</c> ran ten times slower than a signed
        ///     compare (0.58 ns a key): the keys are biased by flipping their top bit, which maps unsigned order onto signed
        ///     order, compared signed (one <c>vpcmpgtq</c> and one blend a vector), and flipped back.
        /// </remarks>
        internal static ulong MaxKey(ulong* keys, long count) => ExtremeKey(keys, count, max: true);

        /// <summary>The smallest of <paramref name="count"/> keys (unsigned), with 256-bit vectors where they are accelerated.</summary>
        /// <param name="keys">The keys.</param>
        /// <param name="count">The number of keys (at least 1).</param>
        /// <returns>The minimum key.</returns>
        internal static ulong MinKey(ulong* keys, long count) => ExtremeKey(keys, count, max: false);

        /// <summary>
        ///     The extreme (largest or smallest, unsigned) of <paramref name="count"/> keys: two accumulators of biased signed
        ///     lanes over 256-bit loads (see <see cref="MaxKey"/>), folded at the end, then a scalar tail.
        /// </summary>
        /// <param name="keys">The keys.</param>
        /// <param name="count">The number of keys (at least 1).</param>
        /// <param name="max">True for the largest key, false for the smallest.</param>
        /// <returns>The extreme key.</returns>
        private static ulong ExtremeKey(ulong* keys, long count, bool max)
        {
            long i = 0;
            ulong best = keys[0];
            if (Vector256.IsHardwareAccelerated && count >= 8)
            {
                var bias = Vector256.Create(long.MinValue);
                var acc0 = Vector256.Load((long*)keys) ^ bias;
                var acc1 = Vector256.Load((long*)keys + 4) ^ bias;
                for (i = 8; i + 8 <= count; i += 8)
                {
                    var v0 = Vector256.Load((long*)keys + i) ^ bias;
                    var v1 = Vector256.Load((long*)keys + i + 4) ^ bias;
                    acc0 = Vector256.ConditionalSelect(max ? Vector256.GreaterThan(v0, acc0) : Vector256.LessThan(v0, acc0), v0, acc0);
                    acc1 = Vector256.ConditionalSelect(max ? Vector256.GreaterThan(v1, acc1) : Vector256.LessThan(v1, acc1), v1, acc1);
                }
                acc0 = Vector256.ConditionalSelect(max ? Vector256.GreaterThan(acc1, acc0) : Vector256.LessThan(acc1, acc0), acc1, acc0);
                long b = acc0.GetElement(0);
                for (int lane = 1; lane < 4; lane++)
                {
                    long e = acc0.GetElement(lane);
                    if (max ? e > b : e < b)
                        b = e;
                }
                best = (ulong)(b ^ long.MinValue);
            }
            for (; i < count; i++)
            {
                ulong k = keys[i];
                if (max ? k > best : k < best)
                    best = k;
            }
            return best;
        }

        /// <summary>
        ///     Collects, in index order, the indices from <paramref name="first"/> on whose key equals
        ///     <paramref name="value"/> (vector compares; a vector without a match costs one compare and one mask test).
        /// </summary>
        /// <param name="keys">The keys.</param>
        /// <param name="count">The number of keys.</param>
        /// <param name="value">The key to match.</param>
        /// <param name="first">The first matching index (known).</param>
        /// <param name="outIdx">Receives the matching indices.</param>
        /// <returns>The number of matches (at least 1).</returns>
        internal static int GatherEqual(ulong* keys, long count, ulong value, long first, long* outIdx)
        {
            int m = 0;
            long i = first;
            if (Vector256.IsHardwareAccelerated)
            {
                var target = Vector256.Create(value);
                for (; i + 4 <= count; i += 4)
                {
                    uint mask = Vector256.Equals(Vector256.Load(keys + i), target).ExtractMostSignificantBits();
                    while (mask != 0)
                    {
                        int lane = BitOperations.TrailingZeroCount(mask);
                        outIdx[m++] = i + lane;
                        mask &= mask - 1;
                    }
                }
            }
            for (; i < count; i++)
                if (keys[i] == value)
                    outIdx[m++] = i;
            return m;
        }
    }
}
