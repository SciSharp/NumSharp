using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    // ── NumPy's introselect_, statement for statement (numpy/_core/src/npysort/selection.cpp, 2.4.2) ──────────────────
    //
    // np.partition / np.argpartition with SEVERAL kth, and every median / percentile / quantile (which partition at
    // `kth + [-1]` or `unique([0, -1, prev, next])`), run NumPy's scalar introselect_ once per kth on EVERY host: only a
    // single kth goes to x86-simd-sort (`if (nkth == 1 && quickselect_dispatch(...))`), and on a host without AVX-512 the
    // 8- and 16-bit types (float16 included) take introselect_ even then. Its swaps decide where equal values land: the
    // sign of the zero a ±0 tie leaves at a kth (the bits np.partition returns, and the zero np.quantile / np.percentile
    // interpolate from), the index np.argpartition reports for a tie, and which NaN ends a line. So the select below runs
    // NumPy's statements in NumPy's order, for the value path and the index path alike:
    //   * the pivot stack (store_pivot, NPY_MAX_PIVOT_STACK = 50) narrows [low, high] before each kth;
    //   * kth - low < 3 → dumb_select_: pass i swaps the window's FIRST minimum to low + i;
    //   * an inexact type (float32 / float64 / complex128; NumPy's npy_half is a uint16_t, so float16 is not) with
    //     kth == num - 1 → the LAST maximum swaps to kth, and nothing is stored;
    //   * otherwise the loop `for (; low + 1 < high;)`: median3_swap_, or median_of_median5_ once 2 * msb(num) rounds are
    //     spent and the window still has 7 elements; unguarded_partition_ (a Hoare scan stopping on equal values); the
    //     pivot swapped to hh and stored; high = hh - 1 and / or low = ll (the LEFT scanner's stop, which can be hh + 2);
    //   * the two-element compare-swap, then store_pivot(kth).
    // There is no insertion sort and no heap sort anywhere: those arrange ties their own way.
    //
    // Speed is kept where it does not move a tie. A long window's Hoare scan runs as the branchless block partition
    // (BlockQuicksort): a block scan records the same stops Hoare's scanners stop at (left: not less than the pivot,
    // right: not greater), pairs them in the same order, and hands the crossing to a scalar Hoare loop started from the
    // scanners' exact state, so the swaps are Hoare's (see BlockScan). The dumb_select and last-maximum passes keep their
    // vector minimum / maximum searches over NaN-free lines (the first / last position of the extreme is the same).
    //
    // The comparison is a policy struct (INpyLess): the raw `<` of the element type (NaN-free lines, integers, bool),
    // NumPy's NaN-aware DOUBLE_LT / FLOAT_LT / HALF_LT (a float line holding NaN, partitioned in place as NumPy does), or
    // CDOUBLE_LT (complex128). The column is a second policy (ILineColumn): the values alone (np.partition, the quantile
    // kernels) or the values and their original indices moved in lockstep (np.argpartition: NumPy's arg form moves only
    // `tosort` and reads `v[tosort[i]]`, the same arrangement as moving both). Each instantiation is specialized by the JIT.
    internal static partial class QuickSelect
    {
        /// <summary>
        ///     NumPy's <c>Tag::less</c> for one element type (<c>npysort_common.h</c> / <c>numpy_tag.h</c>): the only
        ///     comparison introselect_ makes. A struct type argument, so every call inlines to the type's compare.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        private unsafe interface INpyLess<T> where T : unmanaged
        {
            /// <summary>Whether <c>*a</c> orders strictly before <c>*b</c>.</summary>
            /// <param name="a">The first element.</param>
            /// <param name="b">The second element.</param>
            /// <returns>NumPy's <c>Tag::less(*a, *b)</c>.</returns>
            bool Lt(T* a, T* b);

            /// <summary>
            ///     True when this comparison is the type's raw <c>&lt;</c> over values that hold no NaN, so equality is the
            ///     type's <c>==</c> and the extreme-value passes may run as vector minimum / maximum searches. Constant per
            ///     struct, so the JIT folds every test of it.
            /// </summary>
            bool IsRawOrder { get; }
        }

        /// <summary>
        ///     The element type's own <c>&lt;</c> (<see cref="LtPtr{T}"/>): NumPy's <c>Tag::less</c> for the integers, bool
        ///     and char, and for a float line known to hold no NaN (where <c>DOUBLE_LT</c> reduces to <c>&lt;</c>).
        ///     FOOTGUN: a NaN in a line handed this comparison breaks the unguarded scans' sentinels (a NaN is neither
        ///     below nor above anything); a float line that may hold NaN takes <see cref="NaNLess{T}"/>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        private readonly unsafe struct RawLess<T> : INpyLess<T> where T : unmanaged
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public bool Lt(T* a, T* b) => LtPtr(a, b);

            /// <inheritdoc/>
            public bool IsRawOrder => true;
        }

        /// <summary>
        ///     NumPy's NaN-aware float comparison: <c>DOUBLE_LT</c> / <c>FLOAT_LT</c> (<c>a &lt; b || (b != b &amp;&amp; a ==
        ///     a)</c>) and <c>HALF_LT</c>, under which every NaN orders after every number and NaNs are equal to each other.
        ///     np.partition / np.argpartition of a float line holding NaN partition it in place under this order, so the
        ///     NaNs take part in the swaps (they are not compacted away first) and the numbers land where NumPy puts them.
        /// </summary>
        /// <typeparam name="T"><see cref="double"/>, <see cref="float"/> or <see cref="Half"/>.</typeparam>
        private readonly unsafe struct NaNLess<T> : INpyLess<T> where T : unmanaged
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public bool Lt(T* a, T* b)
            {
                if (typeof(T) == typeof(double))
                {
                    double x = *(double*)a, y = *(double*)b;
                    return x < y || (y != y && x == x);
                }
                if (typeof(T) == typeof(float))
                {
                    float x = *(float*)a, y = *(float*)b;
                    return x < y || (y != y && x == x);
                }
                if (typeof(T) == typeof(Half))
                {
                    // HALF_LT: a NaN b orders every number before it; otherwise a NaN a is never below, and two numbers
                    // compare by HalfOrderKey (IEEE order with the zeros equal, _npy_half_lt_nonan).
                    ushort x = *(ushort*)a, y = *(ushort*)b;
                    bool xNaN = (x & 0x7FFF) > 0x7C00, yNaN = (y & 0x7FFF) > 0x7C00;
                    return yNaN ? !xNaN : !xNaN && HalfOrderKey(x) < HalfOrderKey(y);
                }
                // Every other type has no NaN: NumPy's less is its plain one.
                return LtPtr(a, b);
            }

            /// <inheritdoc/>
            public bool IsRawOrder => false;
        }

        /// <summary>
        ///     NumPy's <c>CDOUBLE_LT</c>: complex128 ordered by the real part, then the imaginary part, with a NaN in either
        ///     part ordering after every number (<c>(1, NaN)</c> orders before <c>(NaN, 1)</c>: a NaN real part sorts last,
        ///     a NaN imaginary part sorts last among its real part). The comparison np.partition / np.argpartition run on a
        ///     complex line, in place.
        /// </summary>
        private readonly unsafe struct CDoubleLess : INpyLess<Complex>
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public bool Lt(Complex* a, Complex* b)
            {
                // System.Numerics.Complex is (real, imaginary), two doubles in that order.
                double ar = ((double*)a)[0], ai = ((double*)a)[1], br = ((double*)b)[0], bi = ((double*)b)[1];
                if (ar < br)
                    return ai == ai || bi != bi;
                if (ar > br)
                    return bi != bi && ai == ai;
                if (ar == br || (ar != ar && br != br))
                    return ai < bi || (bi != bi && ai == ai);
                return br != br;
            }

            /// <inheritdoc/>
            public bool IsRawOrder => false;
        }

        /// <summary>
        ///     What a swap of the select moves: the values alone, or the values and the original-index column beside them.
        ///     A struct type argument, so the value-only swap carries no index work at all.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TSelf">The implementing struct (its <see cref="Offset"/> returns its own type).</typeparam>
        private unsafe interface ILineColumn<T, TSelf> where T : unmanaged where TSelf : struct, ILineColumn<T, TSelf>
        {
            /// <summary>Exchanges positions <paramref name="i"/> and <paramref name="j"/> (NumPy's <c>std::swap(sortee(i),
            ///     sortee(j))</c>).</summary>
            /// <param name="v">The values, based at the same position as the column.</param>
            /// <param name="i">The first position.</param>
            /// <param name="j">The second position.</param>
            void Swap(T* v, long i, long j);

            /// <summary>The column re-based <paramref name="by"/> positions on, for a select over <c>v + by</c> (NumPy's
            ///     <c>tosort + (arg ? by : 0)</c> in median_of_median5_ and its recursion).</summary>
            /// <param name="by">The number of positions to move the base.</param>
            /// <returns>The re-based column.</returns>
            TSelf Offset(long by);
        }

        /// <summary>The value-only column (np.partition, the quantile kernels): a swap exchanges two values.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        private readonly unsafe struct ValueColumn<T> : ILineColumn<T, ValueColumn<T>> where T : unmanaged
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public void Swap(T* v, long i, long j)
            {
                T t = v[i];
                v[i] = v[j];
                v[j] = t;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public ValueColumn<T> Offset(long by) => default;
        }

        /// <summary>
        ///     The argpartition column: every swap exchanges two values and the same two entries of the original-index
        ///     column, so the index column records the permutation the value select performs.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        private readonly unsafe struct IndexColumn<T> : ILineColumn<T, IndexColumn<T>> where T : unmanaged
        {
            /// <summary>The original-index column, based at the same position as the values.</summary>
            private readonly long* _idx;

            /// <summary>Wraps an index column.</summary>
            /// <param name="idx">The column's first entry (the entry of the values' first element).</param>
            public IndexColumn(long* idx) => _idx = idx;

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public void Swap(T* v, long i, long j)
            {
                T t = v[i];
                v[i] = v[j];
                v[j] = t;
                long u = _idx[i];
                _idx[i] = _idx[j];
                _idx[j] = u;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public IndexColumn<T> Offset(long by) => new IndexColumn<T>(_idx + by);
        }

        /// <summary>
        ///     NumPy's <c>npy_get_msb</c>: the index of the highest set bit (<c>floor(log2(n))</c>), whose double is
        ///     introselect_'s median-of-3 budget before median_of_median5_ takes over. NOT <see cref="Log2"/>, which
        ///     counts bits (one more).
        /// </summary>
        /// <param name="n">A positive length.</param>
        /// <returns>The highest set bit's index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Msb(long n)
        {
            int depth = 0;
            while ((n >>= 1) != 0)
                depth++;
            return depth;
        }

        /// <summary>
        ///     NumPy's <c>store_pivot</c>: records a placed position for the later (larger) kth of the same line. A full
        ///     stack (<see cref="PivotStackMax"/>, NumPy's <c>NPY_MAX_PIVOT_STACK</c>) only lets the kth itself overwrite
        ///     its top, and positions below kth are never stored, so the stack stays descending with the tightest upper
        ///     bound on top.
        /// </summary>
        /// <param name="pivot">The placed position.</param>
        /// <param name="kth">The kth being selected.</param>
        /// <param name="pivots">The stack, or null for a select without one (median_of_median5_'s recursion).</param>
        /// <param name="npiv">The stack's depth (null with <paramref name="pivots"/>).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void StorePivot(long pivot, long kth, long* pivots, int* npiv)
        {
            if (pivots == null)
                return;
            if (pivot == kth && *npiv == PivotStackMax)
                pivots[*npiv - 1] = pivot;
            else if (pivot >= kth && *npiv < PivotStackMax)
            {
                pivots[*npiv] = pivot;
                *npiv += 1;
            }
        }

        /// <summary>
        ///     NumPy's <c>introselect_</c> for one kth: places the kth smallest element of <c>v[0, num)</c> at
        ///     <paramref name="kth"/>, everything before it not above and everything after it not below, moving equal values
        ///     exactly as NumPy does (see the block above the class).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line (<see cref="RawLess{T}"/>,
        ///     <see cref="NaNLess{T}"/>, <see cref="CDoubleLess"/>).</typeparam>
        /// <typeparam name="TCol">What a swap moves (<see cref="ValueColumn{T}"/> or <see cref="IndexColumn{T}"/>).</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move, based at <paramref name="v"/>.</param>
        /// <param name="num">The line's length (at least 1).</param>
        /// <param name="kth">The position to place, in <c>[0, num)</c>.</param>
        /// <param name="pivots">The line's pivot stack (shared by its kths, ascending), or null for a one-off select.</param>
        /// <param name="npiv">The stack's depth, or null with <paramref name="pivots"/>.</param>
        /// <remarks>
        ///     Successive calls for one line must come in ascending kth order with the same stack, as NumPy's
        ///     <c>_new_sortlike</c> / <c>_new_argsortlike</c> make them: a stored pivot is only correct for the
        ///     arrangement the previous calls left. A repeated kth returns at once (its own stored pivot), except after the
        ///     inexact last-maximum pass, which stores nothing and, run again over the whole line, finds the maximum
        ///     already at the end: the same arrangement either way.
        ///     <para>
        ///     Split for speed, not structure: this head (the pivot pops and NumPy's two linear shortcuts) inlines into
        ///     the callers' kth loops, and the quickselect loop (<see cref="IntroselectLoop{T, TLess, TCol}"/>) stays a
        ///     call. The quantile family's many short rows (np.quantile over axis 1 of a (100000, 5) array partitions each
        ///     row at NumPy's [0, 2, 3, 4]) are almost all dumb_select passes, which then cost no call at all.
        ///     </para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Introselect<T, TLess, TCol>(T* v, TCol col, long num, long kth, long* pivots, int* npiv)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            long low = 0;
            long high = num - 1;
            if (npiv == null)
                pivots = null;

            // Pop the pivots at or below kth (each raises low past itself); the first one above kth bounds high. An exact
            // hit means a previous kth already placed this one.
            while (pivots != null && *npiv > 0)
            {
                long top = pivots[*npiv - 1];
                if (top > kth)
                {
                    high = top - 1;
                    break;
                }
                if (top == kth)
                    return;
                low = top + 1;
                *npiv -= 1;
            }

            // dumb_select_ for a kth within 3 of the window's start (e.g. the interpolating quantiles' neighbour pairs).
            if (kth - low < 3)
            {
                DumbSelect<T, TLess, TCol>(v, col, low, high, kth);
                StorePivot(kth, kth, pivots, npiv);
                return;
            }
            // The inexact types' NaN check `partition(d, (x, -1))`: the window's last maximum to the end, nothing stored.
            // No pivot above kth = num - 1 can exist, so the window ends at the line's end, which NumPy scans to.
            if (IsInexact<T>() && kth == num - 1)
            {
                col.Swap(v, kth, LastMaxIndex<T, TLess>(v, low, num - 1));
                return;
            }

            IntroselectLoop<T, TLess, TCol>(v, col, num, kth, pivots, npiv, low, high);
        }

        /// <summary>
        ///     The quickselect half of <see cref="Introselect{T, TLess, TCol}"/>, run once neither linear shortcut applies:
        ///     median-of-3 (median-of-medians-5 once the budget is spent), the Hoare scan, the narrowing, the two-element
        ///     finish and the kth's own <c>store_pivot</c>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move.</param>
        /// <param name="num">The line's length (its median-of-3 budget is <c>2 * msb(num)</c>).</param>
        /// <param name="kth">The position to place.</param>
        /// <param name="pivots">The line's pivot stack, or null.</param>
        /// <param name="npiv">The stack's depth, or null.</param>
        /// <param name="low">The window's first position after the pivot pops.</param>
        /// <param name="high">The window's last position after the pivot pops.</param>
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void IntroselectLoop<T, TLess, TCol>(T* v, TCol col, long num, long kth, long* pivots,
            int* npiv, long low, long high)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            int depthLimit = Msb(num) * 2;
            // Three elements or more: median-of-3 (or median-of-medians-5) pivot, Hoare partition, narrow.
            while (low + 1 < high)
            {
                long ll = low + 1;
                long hh = high;
                if (depthLimit > 0 || hh - ll < 5)
                {
                    long mid = low + (high - low) / 2;
                    // median3_swap_ leaves v[low + 1] <= pivot (at low) <= v[high]: the unguarded scans' sentinels.
                    Median3Swap<T, TLess, TCol>(v, col, low, mid, high);
                }
                else
                {
                    // median_of_median5_ over [ll, hh), then its median to low; no sentinels, so the scans start one
                    // further out on both sides and stop on the medians' guaranteed spread.
                    long mid = ll + MedianOfMedians5<T, TLess, TCol>(v + ll, col.Offset(ll), hh - ll);
                    col.Swap(v, mid, low);
                    ll--;
                    hh++;
                }
                depthLimit--;

                // The pivot stays at v[low] for the whole scan (the scans start above low and the right one stops on it),
                // so comparing against its address is NumPy's by-value pivot.
                UnguardedPartition<T, TLess, TCol>(v, col, v + low, ref ll, ref hh);
                col.Swap(v, low, hh);

                // The kth's own position is stored after the loop.
                if (hh != kth)
                    StorePivot(hh, kth, pivots, npiv);
                if (hh >= kth)
                    high = hh - 1;
                // ll, not hh + 1: when the scanners met on an element equal to the pivot ll is hh + 2, and NumPy narrows
                // past that element (it is in place); a kth equal to hh + 1 then leaves a window that no longer holds it,
                // which NumPy partitions anyway until it is two elements or fewer: those swaps are part of its arrangement.
                if (hh <= kth)
                    low = ll;
            }

            // Two elements left.
            if (high == low + 1 && default(TLess).Lt(v + high, v + low))
                col.Swap(v, high, low);
            StorePivot(kth, kth, pivots, npiv);
        }

        /// <summary>
        ///     NumPy's <c>dumb_select_</c> over the window <c>[low, high]</c>: for every position from <paramref name="low"/>
        ///     to <paramref name="kth"/>, the FIRST minimum of the rest of the window is swapped into it (a strict less never
        ///     replaces an equal candidate). O(n·(kth − low + 1)), which is why NumPy takes it only for <c>kth - low &lt; 3</c>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="high">The window's last position (at least <paramref name="kth"/>).</param>
        /// <param name="kth">The last position to fill, within 3 of <paramref name="low"/>.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DumbSelect<T, TLess, TCol>(T* v, TCol col, long low, long high, long kth)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            for (long i = low; i <= kth; i++)
                col.Swap(v, i, FirstMinIndex<T, TLess>(v, i, high));
        }

        /// <summary>
        ///     The index of the first minimum of <c>v[from..to]</c> under <typeparamref name="TLess"/>, the element NumPy's
        ///     <c>dumb_select_</c> scan settles on (it replaces its candidate only on a strict less, so among equal values,
        ///     every zero included, the first wins).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="from">The first position scanned.</param>
        /// <param name="to">The last position scanned (at least <paramref name="from"/>).</param>
        /// <returns>The position of the first element no other element is below.</returns>
        /// <remarks>A raw order over a type with vector lanes searches with vectors (<see cref="FirstMinIndexVector{T}"/>):
        ///     the same position, since equality there is the type's <c>==</c>. A NaN-aware or complex order runs NumPy's
        ///     loop.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe long FirstMinIndex<T, TLess>(T* v, long from, long to)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
        {
            long count = to - from + 1;
            if (default(TLess).IsRawOrder && HasVectorLanes<T>() && Vector.IsHardwareAccelerated && count >= 2L * Vector<T>.Count)
                return from + FirstMinIndexVector(v + from, count);
            // NumPy's loop (comparing against v[minIdx] equals comparing against its copy: nothing moves during the scan).
            TLess less = default;
            long minIdx = from;
            for (long k = from + 1; k <= to; k++)
                if (less.Lt(v + k, v + minIdx))
                    minIdx = k;
            return minIdx;
        }

        /// <summary>
        ///     The index of the last maximum of <c>v[from..to]</c> under <typeparamref name="TLess"/>, the element NumPy's
        ///     inexact <c>kth == num - 1</c> pass settles on (it replaces its candidate whenever the element is not less, so
        ///     among equal values the last wins; under the NaN-aware order the last NaN).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="from">The first position scanned.</param>
        /// <param name="to">The last position scanned (at least <paramref name="from"/>).</param>
        /// <returns>The position of the last element no other element is above.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe long LastMaxIndex<T, TLess>(T* v, long from, long to)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
        {
            long count = to - from + 1;
            if (default(TLess).IsRawOrder && HasVectorLanes<T>() && Vector.IsHardwareAccelerated && count >= 2L * Vector<T>.Count)
                return from + LastMaxIndexVector(v + from, count);
            TLess less = default;
            long maxIdx = from;
            for (long k = from + 1; k <= to; k++)
                if (!less.Lt(v + k, v + maxIdx))
                    maxIdx = k;
            return maxIdx;
        }

        /// <summary>
        ///     NumPy's <c>median3_swap_</c>: orders <c>{low, mid, high}</c> enough to move their median to
        ///     <paramref name="low"/> (the pivot) and their smallest to <c>low + 1</c>, leaving <c>v[low + 1] ≤ pivot ≤
        ///     v[high]</c> as the unguarded scans' sentinels.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move.</param>
        /// <param name="low">The window's first position.</param>
        /// <param name="mid">The window's middle (<c>low + (high - low) / 2</c>).</param>
        /// <param name="high">The window's last position.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Median3Swap<T, TLess, TCol>(T* v, TCol col, long low, long mid, long high)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            TLess less = default;
            if (less.Lt(v + high, v + mid))
                col.Swap(v, high, mid);
            if (less.Lt(v + high, v + low))
                col.Swap(v, high, low);
            // The pivot to low.
            if (less.Lt(v + low, v + mid))
                col.Swap(v, low, mid);
            // The smallest of the three to low + 1.
            col.Swap(v, mid, low + 1);
        }

        /// <summary>
        ///     NumPy's <c>median5_</c>: partially sorts the five elements at <paramref name="v"/> with its fixed exchange
        ///     network and returns the position (1, 2 or 3) of their median.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The group's first element.</param>
        /// <param name="col">The column, based at <paramref name="v"/>.</param>
        /// <returns>The median's offset within the group.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long Median5<T, TLess, TCol>(T* v, TCol col)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            TLess less = default;
            if (less.Lt(v + 1, v + 0)) col.Swap(v, 1, 0);
            if (less.Lt(v + 4, v + 3)) col.Swap(v, 4, 3);
            if (less.Lt(v + 3, v + 0)) col.Swap(v, 3, 0);
            if (less.Lt(v + 4, v + 1)) col.Swap(v, 4, 1);
            if (less.Lt(v + 2, v + 1)) col.Swap(v, 2, 1);
            if (less.Lt(v + 3, v + 2))
                return less.Lt(v + 3, v + 1) ? 1 : 3;
            // v[1] and v[2] were swapped into order above.
            return 2;
        }

        /// <summary>
        ///     NumPy's <c>median_of_median5_</c>: the median of each whole group of five moves to the front (group i's to
        ///     position i), and those medians are selected around their own middle by a stack-less <see
        ///     cref="Introselect{T, TLess, TCol}"/>. Its result as a pivot splits a window at least 30 / 70, which bounds
        ///     introselect_ to linear time once its median-of-3 budget is spent (adversarial or heavily repeated data).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The span's first element.</param>
        /// <param name="col">The column, based at <paramref name="v"/>.</param>
        /// <param name="num">The span's length (a partial last group is ignored, as NumPy ignores it).</param>
        /// <returns>The offset of the median of medians from <paramref name="v"/> (<c>nmed / 2</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe long MedianOfMedians5<T, TLess, TCol>(T* v, TCol col, long num)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            long right = num - 1;
            long nmed = (right + 1) / 5;
            for (long i = 0, subleft = 0; i < nmed; i++, subleft += 5)
            {
                long m = Median5<T, TLess, TCol>(v + subleft, col.Offset(subleft));
                col.Swap(v, subleft + m, i);
            }
            // NumPy recurses without its caller's pivot stack (pivots and npiv are NULL here).
            if (nmed > 2)
                Introselect<T, TLess, TCol>(v, col, nmed, nmed / 2, null, null);
            return nmed / 2;
        }

        /// <summary>
        ///     NumPy's <c>unguarded_partition_</c>: the Hoare scan around <paramref name="pivot"/>, the left scanner stopping
        ///     on every element not less than the pivot and the right one on every element not greater, each stop pair
        ///     swapped until the scanners cross. No bounds tests: the caller's sentinels (median3_swap_) or the medians'
        ///     spread (median_of_median5_) stop both scans.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move.</param>
        /// <param name="pivot">The pivot's address (it stays in place during the scan).</param>
        /// <param name="ll">In: the position before the left scanner's first look. Out: its last stop (where the scans
        ///     crossed, hh + 1 or hh + 2).</param>
        /// <param name="hh">In: the position after the right scanner's first look. Out: its last stop, the pivot's final
        ///     position.</param>
        /// <remarks>
        ///     A window with room for two disjoint blocks first runs <see cref="BlockScan{T, TLess, TCol}"/>, which performs
        ///     the scan's swaps without its branch mispredictions and hands this loop the scanners' state; the swaps and
        ///     the final <paramref name="ll"/> / <paramref name="hh"/> are the ones the plain loop would produce.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void UnguardedPartition<T, TLess, TCol>(T* v, TCol col, T* pivot, ref long ll, ref long hh)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            TLess less = default;
            long l = ll, h = hh;
            // The unscanned span is [l + 1, h - 1].
            if (h - l - 1 > 2 * BlockSize)
                BlockScan<T, TLess, TCol>(v, col, pivot, ref l, ref h);
            for (;;)
            {
                do { l++; } while (less.Lt(v + l, pivot));
                do { h--; } while (less.Lt(pivot, v + h));
                if (h < l)
                    break;
                col.Swap(v, l, h);
            }
            ll = l;
            hh = h;
        }

        /// <summary>
        ///     The bulk of a long Hoare scan as a branchless block partition (BlockQuicksort), with exactly Hoare's swaps:
        ///     each side's next <see cref="BlockSize"/> unscanned elements are scanned at once into the offsets of its
        ///     stops (left: not less than the pivot; right: not greater), with no data-dependent branch, and the i-th
        ///     remaining left stop is swapped with the i-th remaining right stop, which is the pair Hoare's two scanners
        ///     would swap next. On return <paramref name="ll"/> / <paramref name="hh"/> are the scanners' positions in
        ///     Hoare's own run at this point, so the caller's scalar loop finishes the crossing exactly.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLess">NumPy's <c>Tag::less</c> for the line.</typeparam>
        /// <typeparam name="TCol">What a swap moves.</typeparam>
        /// <param name="v">The line.</param>
        /// <param name="col">The column the swaps move.</param>
        /// <param name="pivot">The pivot's address.</param>
        /// <param name="ll">In: the position before the left scanner's first look; out: the position before its next
        ///     look.</param>
        /// <param name="hh">In: the position after the right scanner's first look; out: the position after its next
        ///     look.</param>
        /// <remarks>
        ///     <para>
        ///     Why the swaps are Hoare's: a scanner stops on the stops of the ORIGINAL values in order (a swap only writes
        ///     positions both scanners have passed), and Hoare swaps its i-th left stop with its i-th right stop until they
        ///     cross. The blocks are disjoint while the unscanned span exceeds two blocks, so every pair they swap lies
        ///     left-before-right, before the crossing; the stops come out of the scans in Hoare's order (left ascending,
        ///     right descending) and are consumed in that order. The crossing itself is left to the scalar loop.
        ///     </para>
        ///     <para>
        ///     The hand-over: a partly consumed left block's next stop P is where Hoare's left scanner stands next (the
        ///     positions between the last consumed stop and P are not stops), so the scalar loop restarts it at P - 1; the
        ///     right scanner restarts after the first unscanned position. A partly consumed right block mirrors that. (The
        ///     earlier pdqsort-style drain paired a block's LAST remaining offset first, which is not Hoare's pairing and
        ///     moved ties elsewhere on lines longer than ~260 elements.)
        ///     </para>
        ///     <para>[SkipLocalsInit]: both offset blocks are written before they are read.</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void BlockScan<T, TLess, TCol>(T* v, TCol col, T* pivot, ref long ll, ref long hh)
            where T : unmanaged
            where TLess : struct, INpyLess<T>
            where TCol : struct, ILineColumn<T, TCol>
        {
            TLess less = default;
            byte* offL = stackalloc byte[BlockSize];
            byte* offR = stackalloc byte[BlockSize];
            // The first unscanned positions on each side.
            long lo = ll + 1, hi = hh - 1;
            int numL = 0, numR = 0, startL = 0, startR = 0;
            long baseL = lo, baseR = hi;

            while (hi - lo + 1 > 2 * BlockSize)
            {
                if (numL == 0)
                {
                    startL = 0;
                    baseL = lo;
                    // Data-independent write: every offset is stored, the count advances only for a stop.
                    for (int i = 0; i < BlockSize; i++)
                    {
                        offL[numL] = (byte)i;
                        numL += less.Lt(v + lo + i, pivot) ? 0 : 1;
                    }
                }
                if (numR == 0)
                {
                    startR = 0;
                    baseR = hi;
                    for (int i = 0; i < BlockSize; i++)
                    {
                        offR[numR] = (byte)i;
                        numR += less.Lt(pivot, v + hi - i) ? 0 : 1;
                    }
                }
                int num = Math.Min(numL, numR);
                for (int i = 0; i < num; i++)
                    col.Swap(v, baseL + offL[startL + i], baseR - offR[startR + i]);
                numL -= num;
                numR -= num;
                startL += num;
                startR += num;
                // A consumed block's elements are all on their side now; the other side's block (if any stops remain)
                // stays where it is.
                if (numL == 0)
                    lo += BlockSize;
                if (numR == 0)
                    hi -= BlockSize;
            }

            // Hoare's state: at most one side has stops left in its block, and that side's scanner stands right before
            // its next stop; the other scanner stands before its first unscanned position.
            ll = numL > 0 ? baseL + offL[startL] - 1 : lo - 1;
            hh = numR > 0 ? baseR - offR[startR] + 1 : hi + 1;
        }
    }
}
