using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;
using NumSharp.Backends.Unmanaged;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // ============================================================================
        //  HASH fast path for unique_values / unique_counts on INTEGER-FAMILY dtypes.
        //
        //  NumPy's _unique_hash (numpy/_core/src/multiarray/unique.cpp) dedups with a
        //  std::unordered_set for exactly the integer + complex dtypes — plain floats are
        //  NOT in its function map, so they fall through to the sort path. We mirror that
        //  structure: integers dedup through a purpose-built open-addressing table (then a
        //  sort of the typically-small unique set), which is far cheaper than the O(n log n)
        //  sort of the whole array and beats NumPy's own hash on integers. Floats/half/
        //  complex/decimal keep the existing sort path (their equal_nan=False NaN semantics,
        //  -0.0 collapsing and NumPy's own float→sort fallback make bitwise-hash dedup wrong).
        //
        //  Semantics are IDENTICAL to the sort path for integers: the output is the SORTED
        //  set of distinct values (counts aligned to it). Integer bit patterns are a bijection
        //  with their values, so bitwise dedup + numeric sort == the sort-path result.
        // ============================================================================

        /// <summary>
        ///     Sorted unique values for <c>np.unique_values</c>. Integer-family dtypes take the
        ///     hash path; everything else routes through the existing sort path (equal_nan=False).
        /// </summary>
        /// <param name="a">The numeric operand (any layout; only read).</param>
        /// <returns>The sorted distinct values, a fresh 1-D array of <paramref name="a"/>'s dtype (a float's NaNs last,
        ///     each distinct).</returns>
        /// <remarks>NumPy returns its hash set's iteration order for the integer and complex dtypes; this returns them
        ///     sorted (the same set: the documented divergence of <c>np.unique_values</c>).</remarks>
        internal static NDArray uniqueValuesFast(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Boolean: return UniqueHashSortedInt<bool>(a, wantCounts: false).values;
                case NPTypeCode.Byte: return UniqueHashSortedInt<byte>(a, wantCounts: false).values;
                case NPTypeCode.SByte: return UniqueHashSortedInt<sbyte>(a, wantCounts: false).values;
                case NPTypeCode.Int16: return UniqueHashSortedInt<short>(a, wantCounts: false).values;
                case NPTypeCode.UInt16: return UniqueHashSortedInt<ushort>(a, wantCounts: false).values;
                case NPTypeCode.Int32: return UniqueHashSortedInt<int>(a, wantCounts: false).values;
                case NPTypeCode.UInt32: return UniqueHashSortedInt<uint>(a, wantCounts: false).values;
                case NPTypeCode.Int64: return UniqueHashSortedInt<long>(a, wantCounts: false).values;
                case NPTypeCode.UInt64: return UniqueHashSortedInt<ulong>(a, wantCounts: false).values;
                case NPTypeCode.Char: return UniqueHashSortedInt<char>(a, wantCounts: false).values;
                case NPTypeCode.Complex: return UniqueHashSortedComplex(a, wantCounts: false).values;
                // Half/Single/Double take the SORT path (NumPy's own choice — floats are NOT in its
                // hash map): NumPy sorts them with SIMD vqsort, and a hash regresses ~2x on all-unique
                // input (insert N + scalar-sort N) with no cheap way to detect that cardinality. Decimal
                // needs value-equality (1.0m != 1.00m by bits), so it sorts too.
                default: return UniqueCompute(a, false, false, false, axis: null, equal_nan: false)[0];
            }
        }

        /// <summary>
        ///     Sorted unique values + counts for <c>np.unique_counts</c>. Integer-family dtypes
        ///     take the hash path; everything else routes through the sort path (equal_nan=False).
        /// </summary>
        /// <param name="a">The numeric operand (any layout; only read).</param>
        /// <returns>[values, counts]: the sorted distinct values and their int64 occurrence counts.</returns>
        internal static NDArray[] uniqueCountsFast(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Boolean: { var (v, c) = UniqueHashSortedInt<bool>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Byte: { var (v, c) = UniqueHashSortedInt<byte>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.SByte: { var (v, c) = UniqueHashSortedInt<sbyte>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int16: { var (v, c) = UniqueHashSortedInt<short>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt16: { var (v, c) = UniqueHashSortedInt<ushort>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int32: { var (v, c) = UniqueHashSortedInt<int>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt32: { var (v, c) = UniqueHashSortedInt<uint>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int64: { var (v, c) = UniqueHashSortedInt<long>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt64: { var (v, c) = UniqueHashSortedInt<ulong>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Char: { var (v, c) = UniqueHashSortedInt<char>(a, wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Complex: { var (v, c) = UniqueHashSortedComplex(a, wantCounts: true); return new[] { v, c }; }
                // Half/Single/Double/Decimal take the SORT path — see uniqueValuesFast.
                default: return UniqueCompute(a, false, false, true, axis: null, equal_nan: false);
            }
        }

        /// <summary>
        ///     Open-addressing (linear-probe) dedup of an integer-family array, returning the SORTED
        ///     distinct values and — when <paramref name="wantCounts"/> — the aligned occurrence counts
        ///     (int64). The table grows on demand (load factor ≤ 0.5) so memory stays O(unique), not O(n).
        /// </summary>
        /// <typeparam name="T">The integer-family element type (bool and char included).</typeparam>
        /// <param name="a">The operand (any layout; only read).</param>
        /// <param name="wantCounts">Also count each distinct value's occurrences.</param>
        /// <returns>The sorted values and, when <paramref name="wantCounts"/>, the aligned int64 counts (null
        ///     otherwise).</returns>
        /// <remarks>
        ///     <para>
        ///     Past 131,072 distinct values the table stops (it no longer fits a core's L2) and an LSD radix sort of the
        ///     whole extract finishes the job: the same sorted set. Past 2^28 elements the method hands the operand to
        ///     np.unique's flat path; that path's own hash routes stop at the same bound, so the two cannot recurse.
        ///     </para>
        /// </remarks>
        private static unsafe (NDArray values, NDArray counts) UniqueHashSortedInt<T>(NDArray a, bool wantCounts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long n = a.size;
            if (n == 0)
            {
                var emptyV = new NDArray(new ArraySlice<T>(new UnmanagedMemoryBlock<T>(0)), Shape.Vector(0));
                var emptyC = wantCounts
                    ? new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0))
                    : null;
                return (emptyV, emptyC);
            }

            // The table is int-indexed; for arrays beyond this bound fall back to the sort path
            // (correct, and n this large means the whole-array sort is not the bottleneck anyway).
            if (n > (1L << 28))
            {
                var r = UniqueCompute(a, false, false, wantCounts, axis: null, equal_nan: false);
                return (r[0], wantCounts ? r[1] : null);
            }

            var data = ExtractKeysOnly<T>(a, n);   // resolves any layout (strided/transposed/reversed) to C-order

            // High-cardinality bail-out: once the distinct set grows past ~L2 the open-addressing
            // table becomes a cache-miss per insert, so an LSD radix sort of `data` (sequential,
            // comparison-free) overtakes it — exactly where NumPy's own std::unordered_set path
            // thrashes (int32 2M-distinct: NumPy 1540ms). This is the "cheap cardinality sample
            // cannot decide hash-vs-sort" problem solved by OBSERVING growth instead of guessing:
            // low/medium cardinality never reaches the threshold (the array's own distinct count is
            // the ceiling), so it keeps the hash; only a genuinely large distinct set bails, and the
            // wasted inserts are bounded by the threshold. The radix result is the SAME sorted set.
            const long HashBailoutThreshold = 1L << 17;   // ~131k distinct: table past L2, radix wins

            int cap = 1024;
            var table = new T[cap];
            var used = new bool[cap];
            var cnt = wantCounts ? new long[cap] : null;
            int mask = cap - 1;
            int uniqueCount = 0;

            for (int i = 0; i < n; i++)
            {
                // Grow at load factor 0.5, rehashing existing entries into a table twice the size.
                if (uniqueCount >= (cap >> 1))
                {
                    int ncap = cap << 1;
                    var nt = new T[ncap];
                    var nu = new bool[ncap];
                    var nc = wantCounts ? new long[ncap] : null;
                    int nmask = ncap - 1;
                    for (int s = 0; s < cap; s++)
                    {
                        if (!used[s]) continue;
                        int hh = (int)(HashKey(table[s]) & (ulong)nmask);
                        while (nu[hh]) hh = (hh + 1) & nmask;
                        nu[hh] = true;
                        nt[hh] = table[s];
                        if (wantCounts) nc[hh] = cnt[s];
                    }
                    table = nt; used = nu; cnt = nc; cap = ncap; mask = nmask;
                }

                T v = data[i];
                int h = (int)(HashKey(v) & (ulong)mask);
                while (used[h])
                {
                    if (table[h].Equals(v)) { if (wantCounts) cnt[h]++; goto matched; }
                    h = (h + 1) & mask;
                }
                used[h] = true;
                table[h] = v;
                if (wantCounts) cnt[h] = 1;
                uniqueCount++;
                if (uniqueCount >= HashBailoutThreshold)   // high cardinality — radix beats the hash from here
                {
                    RadixSortValues(data, (int)n);
                    return RadixDedupSortedInt<T>(data, (int)n, wantCounts);
                }
                matched: ;
            }

            // Collect the live entries, then sort into NumPy's ascending value order.
            var uv = new T[uniqueCount];
            var uc = wantCounts ? new long[uniqueCount] : null;
            int j = 0;
            for (int s = 0; s < cap; s++)
            {
                if (!used[s]) continue;
                uv[j] = table[s];
                if (wantCounts) uc[j] = cnt[s];
                j++;
            }

            if (wantCounts) System.Array.Sort(uv, uc);
            else System.Array.Sort(uv);

            var valuesArr = VectorFromManaged(uv, uniqueCount);
            var countsArr = wantCounts ? VectorFromManaged(uc, uniqueCount) : null;
            return (valuesArr, countsArr);
        }

        /// <summary>
        ///     Hashes an integer-family value by its raw bits (a bijection with the value, so it is
        ///     consistent with <see cref="IEquatable{T}"/> equality) through the splitmix64 finalizer.
        ///     The <c>SizeOf&lt;T&gt;</c> tests are JIT constants per instantiation, so exactly one
        ///     branch survives per specialized method. A full-avalanche finalizer is REQUIRED (not a
        ///     single Fibonacci multiply): float bit patterns for integer-valued doubles (1.0, 2.0,
        ///     4.0 …) carry their discriminating bits high and have ZERO low bits, so a low-bits table
        ///     index off a single multiply collapsed them all into one bucket (~500 probes/element).
        /// </summary>
        /// <typeparam name="T">The element type (1, 2, 4 or 8 bytes).</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>The 64-bit hash; the table index is its low bits.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static ulong HashKey<T>(T v) where T : unmanaged
        {
            ulong k;
            if (SysUnsafe.SizeOf<T>() == 8) k = SysUnsafe.As<T, ulong>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 4) k = SysUnsafe.As<T, uint>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 2) k = SysUnsafe.As<T, ushort>(ref v);
            else k = SysUnsafe.As<T, byte>(ref v);
            return UniqueSplitmix(k);
        }

        /// <summary>splitmix64 finalizer — avalanches every input bit into the low bits used as the table index.</summary>
        /// <param name="k">The raw key bits.</param>
        /// <returns>The mixed bits.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static ulong UniqueSplitmix(ulong k)
        {
            k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9UL;
            k = (k ^ (k >> 27)) * 0x94D049BB133111EBUL;
            return k ^ (k >> 31);
        }

        /// <summary>Wraps a managed <typeparamref name="T"/><c>[]</c> as a fresh 1-D <see cref="NDArray"/> (unmanaged copy).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="arr">The source; only its first <paramref name="count"/> elements are copied.</param>
        /// <param name="count">The element count of the result.</param>
        /// <returns>A fresh 1-D array that owns a copy of the elements (the managed array may be reused or dropped).</returns>
        private static unsafe NDArray VectorFromManaged<T>(T[] arr, long count) where T : unmanaged
        {
            var block = new UnmanagedMemoryBlock<T>(count);
            if (count > 0)
            {
                fixed (T* src = arr)
                {
                    long bytes = count * SysUnsafe.SizeOf<T>();
                    Buffer.MemoryCopy(src, block.Address, bytes, bytes);
                }
            }
            return new NDArray(new ArraySlice<T>(block), Shape.Vector(count));
        }

        // ============================================================================
        //  COMPLEX hash path under equal_nan=False.
        //
        //  NumPy hashes int + COMPLEX (both are in its _unique_hash map) and sorts floats. We match
        //  that selection: complex dedups here (NumPy's scalar lexicographic complex sort is slow, so
        //  the hash is a clean 3-6x win), while Half/Single/Double keep the sort path (NumPy sorts
        //  them with SIMD vqsort, and a hash regresses ~2x on all-unique input — insert N + scalar
        //  sort N — with no cheap way to detect that cardinality up front).
        //
        //  Two rules make the bitwise integer machinery reusable: any-NaN complex is partitioned out
        //  (each is DISTINCT under equal_nan=False — NumPy's equal_complex is false for two of them — and
        //  appended after the sorted finite set in NumPy's complex sort order, SortNaNPartsInNumPyOrder), and
        //  -0.0 is normalized to +0.0 PER COMPONENT for HASHING only — the FIRST-occurrence original
        //  bits are stored (NumPy keeps the first element's signed zero), and equality is Complex `==`
        //  (IEEE per component: -0.0 == +0.0). The finite set is lex-sorted (real, then imaginary).
        // ============================================================================

        /// <summary>The lexicographic complex order (real part, then imaginary part) the finite set is sorted by.</summary>
        private static readonly IComparer<Complex> ComplexLexComparer = Comparer<Complex>.Create((x, y) =>
        {
            int c = x.Real.CompareTo(y.Real);
            return c != 0 ? c : x.Imaginary.CompareTo(y.Imaginary);
        });

        /// <summary>
        ///     NumPy's <c>_unique_hash</c> for complex128 (<c>unique.cpp</c>'s <c>unique_numeric</c> with
        ///     <c>hash_complex</c> / <c>equal_complex</c>), then the sort of the distinct values: each equality class keeps
        ///     its FIRST element in C order (bits included, so the first of <c>+0.0</c> / <c>-0.0</c>).
        /// </summary>
        /// <param name="a">The complex128 operand (any layout; only read).</param>
        /// <param name="wantCounts">Also count each class's elements.</param>
        /// <param name="equalNan">NumPy's <c>equal_nan</c>: true makes every value with a NaN part ONE class (kept as the
        ///     first such value in C order, sorted last); false keeps each NaN distinct, appended in NumPy's complex sort
        ///     order (<see cref="SortNaNPartsInNumPyOrder"/>; equal ones in input order, where NumPy's are in its hash
        ///     table's order).</param>
        /// <returns>The sorted values and, when <paramref name="wantCounts"/>, the aligned int64 counts.</returns>
        /// <remarks>
        ///     With <paramref name="equalNan"/> the caller (np.unique's values-only path) keeps arrays above 2^28 elements on
        ///     the sort path: this method's own large-array bail-out re-enters np.unique, which would come back here.
        /// </remarks>
        private static unsafe (NDArray values, NDArray counts) UniqueHashSortedComplex(NDArray a, bool wantCounts, bool equalNan = false)
        {
            long n = a.size;
            if (n == 0)
            {
                var ev = new NDArray(new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>(0)), Shape.Vector(0));
                var ec = wantCounts ? new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0)) : null;
                return (ev, ec);
            }
            if (n > (1L << 28))
            {
                var r = UniqueCompute(a, false, false, wantCounts, axis: null, equal_nan: false);
                return (r[0], wantCounts ? r[1] : null);
            }

            var data = ExtractKeysOnly<Complex>(a, n);
            List<Complex> nans = null;
            // equal_nan: NumPy's hash_complex sends every value with a NaN part to bucket 0 and equal_complex calls two such
            // values equal, so the class keeps the first one inserted; its count is every NaN-part value.
            long nanClassCount = 0;

            int cap = 1024;
            var table = new Complex[cap];
            var used = new bool[cap];
            var cnt = wantCounts ? new long[cap] : null;
            int mask = cap - 1;
            int uniqueCount = 0;
            // One look at the cardinality a sixteenth of the way through a long line (ComplexHashCheckpointMinLength's
            // remarks): more than 90% distinct so far means the argsort route is faster for the whole line.
            long checkpoint = n >= ComplexHashCheckpointMinLength && !DisableComplexHashCheckpoint ? n / 16 : -1;

            // The insert loop runs to the checkpoint, the test runs once, and the loop resumes to the end: the hot loop
            // carries no per-element test (with one inside, the 1,000-element high-cardinality cells ran ~10 µs slower).
            long stop = checkpoint > 0 ? checkpoint : n;
            int i = 0;
            while (true)
            {
                for (; i < stop; i++)
                {
                    Complex c = data[i];
                    if (double.IsNaN(c.Real) || double.IsNaN(c.Imaginary))
                    {
                        // equal_nan: one class, the first NaN-part value kept; otherwise each NaN is distinct.
                        if (!equalNan || nans is null)
                            (nans ??= new List<Complex>()).Add(c);
                        nanClassCount++;
                        continue;
                    }
                    Complex normed = NormalizeZeros(c);                        // -0.0 -> +0.0 per component, for HASHING only

                    if (uniqueCount >= (cap >> 1))
                    {
                        int ncap = cap << 1;
                        var nt = new Complex[ncap]; var nu = new bool[ncap]; var nc = wantCounts ? new long[ncap] : null;
                        int nmask = ncap - 1;
                        for (int s = 0; s < cap; s++)
                        {
                            if (!used[s]) continue;
                            int hh = (int)(HashKeyComplex(NormalizeZeros(table[s])) & (ulong)nmask);
                            while (nu[hh]) hh = (hh + 1) & nmask;
                            nu[hh] = true; nt[hh] = table[s]; if (wantCounts) nc[hh] = cnt[s];
                        }
                        table = nt; used = nu; cnt = nc; cap = ncap; mask = nmask;
                    }

                    int h = (int)(HashKeyComplex(normed) & (ulong)mask);
                    while (used[h])
                    {
                        if (table[h] == c) { if (wantCounts) cnt[h]++; goto matched; }   // Complex ==: -0.0 == +0.0 per component
                        h = (h + 1) & mask;
                    }
                    used[h] = true; table[h] = c; if (wantCounts) cnt[h] = 1; uniqueCount++;   // store FIRST-occurrence bits
                    matched: ;
                }
                if (stop == n)
                    break;
                // At the checkpoint: more than 90% distinct so far hands the line over.
                if (uniqueCount * 10L > checkpoint * 9L)
                {
                    ComplexHashCheckpointHandOffs++;
                    // The same answer, faster from here; the extracted elements go along (no second copy of the line).
                    return ArgsortedHashAnswer(a, n, wantCounts, equalNan, data);
                }
                stop = n;
            }

            var uv = new Complex[uniqueCount]; var uc = wantCounts ? new long[uniqueCount] : null;
            int j = 0;
            for (int s = 0; s < cap; s++) { if (!used[s]) continue; uv[j] = table[s]; if (wantCounts) uc[j] = cnt[s]; j++; }
            SortDistinctComplex(uv, uc, uniqueCount);
            // NumPy sorts the whole hash result, so distinct NaN-part values come out in its complex order, not input order
            // (np.unique([1+nanj, nan+2j], equal_nan=False) and the reverse input both give [1+nanj, nan+2j]).
            if (!equalNan && nans is not null && nans.Count > 1)
                SortNaNPartsInNumPyOrder(nans);

            var (values, counts) = AppendNaNsAndBuild(uv, uc, uniqueCount, nans, nans?.Count ?? 0, wantCounts);
            // The collapsed NaN class counts every value with a NaN part (AppendNaNsAndBuild counts each NaN as 1).
            if (equalNan && wantCounts && nans is not null)
                counts.SetInt64(nanClassCount, uniqueCount);
            return (values, counts);
        }

        /// <summary>Builds the value array (sorted finite set + each distinct NaN appended in the order given) and, when requested, aligned int64 counts (finite counts + 1 per NaN).</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="finite">The sorted finite values.</param>
        /// <param name="finiteCnt">Their counts (read only when <paramref name="wantCounts"/>).</param>
        /// <param name="finiteCount">How many entries of <paramref name="finite"/> are used.</param>
        /// <param name="nans">The NaN values to append, in the order they go out (null when there are none).</param>
        /// <param name="nanCount">How many of them.</param>
        /// <param name="wantCounts">Also build the counts.</param>
        /// <returns>The fresh 1-D values and, when <paramref name="wantCounts"/>, the aligned int64 counts (null
        ///     otherwise).</returns>
        private static unsafe (NDArray values, NDArray counts) AppendNaNsAndBuild<T>(
            T[] finite, long[] finiteCnt, int finiteCount, List<T> nans, int nanCount, bool wantCounts) where T : unmanaged
        {
            long total = (long)finiteCount + nanCount;
            var vblock = new UnmanagedMemoryBlock<T>(total);
            T* vp = vblock.Address;
            for (int i = 0; i < finiteCount; i++) vp[i] = finite[i];
            for (int i = 0; i < nanCount; i++) vp[finiteCount + i] = nans[i];
            var values = new NDArray(new ArraySlice<T>(vblock), Shape.Vector(total));

            NDArray counts = null;
            if (wantCounts)
            {
                var cblock = new UnmanagedMemoryBlock<long>(total);
                long* cp = cblock.Address;
                for (int i = 0; i < finiteCount; i++) cp[i] = finiteCnt[i];
                for (int i = 0; i < nanCount; i++) cp[finiteCount + i] = 1;
                counts = new NDArray(new ArraySlice<long>(cblock), Shape.Vector(total));
            }
            return (values, counts);
        }

        /// <summary>
        ///     The shortest line the complex hash path checks the cardinality of: a sixteenth of it (the checkpoint) is then
        ///     at least 2,048 elements. Below it the hash runs to the end (a short line's routes differ by microseconds).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The rule (the checkpoint test in <see cref="UniqueHashSortedComplex"/>): a sixteenth of the way through a line
        ///     of at least this many elements, more than 90% of the elements seen being distinct hands the line to the stable
        ///     argsort route (<see cref="ArgsortedHashAnswer"/>), which gives the same values and counts.
        ///     </para>
        ///     <para>
        ///     Why there: over 100K and 1M elements drawn uniformly from D values (1% NaN parts, two P-cores) the hash costs
        ///     about 15 ns an element plus about 110 ns per distinct value (an insert that misses the caches, its share of
        ///     the sort), the argsort route about 50-70 ns an element: D/n = 0.001 hash 1.9 / 21 ms, argsort 4.3 / 53 ms;
        ///     D/n = 0.2 hash 4.4 / 53 ms, argsort 5.0 / 61 ms; D/n = 0.9 hash 9.3 / 121 ms, argsort 6.0 / 73 ms. The routes
        ///     break even near D = 0.3 n. For uniformly drawn values, the share of distinct elements among the first n/16 is
        ///     (1 - e^-x) / x with x = n / (16 D): 91% for D = 0.32 n, 79% for D = 0.13 n, 97% for D = 0.9 n. A line that is
        ///     not uniform can be misjudged; both routes give the same answer, and the worse one costs at most about twice
        ///     the better one near the boundary.
        ///     </para>
        /// </remarks>
        private const long ComplexHashCheckpointMinLength = 32768;

        /// <summary>
        ///     Test and benchmark hook: true runs the complex hash path to the end whatever the cardinality (no checkpoint),
        ///     so the two routes can be compared on one line. Per thread, false by default.
        /// </summary>
        [ThreadStatic] internal static bool DisableComplexHashCheckpoint;

        /// <summary>Test hook: how many lines this thread's complex hash path handed to the argsort route at its checkpoint.</summary>
        [ThreadStatic] internal static int ComplexHashCheckpointHandOffs;

        /// <summary>
        ///     The complex hash path's answer computed from the stable argsort instead (what the hash path hands a high-
        ///     cardinality line to, and what <see cref="uniqueValuesOnlyComplex"/> computes above 2^28 elements): the
        ///     argsort statements without an index or an inverse, then, under <paramref name="equalNan"/>, the collapsed NaN
        ///     class represented by its first value in C order as NumPy's hash set keeps it.
        /// </summary>
        /// <param name="a">The complex128 operand (any layout; only read).</param>
        /// <param name="n">Its element count (positive, at most <see cref="Array.MaxLength"/>).</param>
        /// <param name="wantCounts">Also return the counts.</param>
        /// <param name="equalNan">NumPy's <c>equal_nan</c>.</param>
        /// <param name="values">The elements in C order when the caller extracted them already; null extracts them.</param>
        /// <returns>The sorted values and, when <paramref name="wantCounts"/>, the aligned int64 counts (null otherwise).</returns>
        private static (NDArray values, NDArray counts) ArgsortedHashAnswer(NDArray a, long n, bool wantCounts, bool equalNan,
            Complex[] values = null)
        {
            var r = uniqueFlatArgsorted<Complex>(a, n, false, false, wantCounts, equalNan, values);
            if (equalNan)
                KeepFirstNaNPartInCOrder(r[0], a);
            return (r[0], wantCounts ? r[1] : null);
        }

        /// <summary>
        ///     From this many distinct values on the hash path sorts them with the radix argsort
        ///     (<see cref="SortDistinctComplex"/>); below it with <see cref="ComplexLexComparer"/>. Measured on distinct
        ///     values (i9-13900K, one core, best of 9): the comparison sort 25 µs for 1,000 values, 179 µs for 2,000, 14.4 ms
        ///     for 80,000; the radix route (an NDArray copy, the argsort, the gather) 40 µs, 91 µs and 4.8 ms.
        /// </summary>
        private const int RadixDistinctComplexMin = 2048;

        /// <summary>
        ///     Sorts the hash path's distinct complex values lexicographically (real part, then imaginary part), carrying
        ///     their counts along. The values are NaN-free and pairwise unequal (the hash merged both zeros into one class),
        ///     so every correct sort gives the same order: the comparison sort for few values, the stable radix argsort
        ///     (<c>AxisSort.ArgSort</c>, whose complex order is lexicographic there) for many.
        /// </summary>
        /// <param name="uv">The distinct values; sorted in place. Its length is <paramref name="count"/>.</param>
        /// <param name="uc">Their counts, moved with them; null when no counts are wanted.</param>
        /// <param name="count">The number of distinct values.</param>
        private static unsafe void SortDistinctComplex(Complex[] uv, long[] uc, int count)
        {
            if (count < RadixDistinctComplexMin)
            {
                if (uc != null) System.Array.Sort(uv, uc, ComplexLexComparer); else System.Array.Sort(uv, ComplexLexComparer);
                return;
            }
            // The radix argsort reads an NDArray: a copy of the values. Then the gather, the counts in step.
            // np.array(uv)'s statement for a C# array: a 1-D array over a copy of it.
            using var arr = new NDArray(Unmanaged.ArraySlice.FromArray(uv, true), Shape.Vector(uv.Length));
            using var perm = AxisSort.ArgSort(arr, null, NPY_SORTKIND.NPY_STABLESORT);
            long* p = (long*)perm.Address;   // a fresh C-contiguous int64 array: its first element is at its base
            var sorted = new Complex[count];
            for (int i = 0; i < count; i++) sorted[i] = uv[p[i]];
            sorted.CopyTo(uv, 0);
            if (uc != null)
            {
                var sortedCounts = new long[count];
                for (int i = 0; i < count; i++) sortedCounts[i] = uc[p[i]];
                sortedCounts.CopyTo(uc, 0);
            }
        }

        /// <summary>
        ///     Sorts complex values that have a NaN part into NumPy's complex sort order (<c>CDOUBLE_LT</c>): <c>R+nanj</c>
        ///     by R, then <c>nan+Rj</c> by R, then <c>nan+nanj</c>. Equal values (both zeros are equal; every
        ///     <c>nan+nanj</c> is equal) keep their input order: NumPy sorts its hash table's output, so its order of equal
        ///     values is the table's iteration order, which follows the C++ library's bucket layout and is not reproduced.
        /// </summary>
        /// <param name="nans">The values, in input order (each with a NaN part); sorted in place.</param>
        private static void SortNaNPartsInNumPyOrder(List<Complex> nans)
        {
            var keyed = new NaNPartKey[nans.Count];
            for (int i = 0; i < keyed.Length; i++)
                keyed[i] = new NaNPartKey(nans[i], i);
            // A generic span sort over a struct comparer: the comparison is devirtualized, and the position in the key
            // makes the (unstable) introsort's order a total, stable one.
            keyed.AsSpan().Sort(default(NaNPartKeyComparer));
            for (int i = 0; i < keyed.Length; i++)
                nans[i] = keyed[i].Value;
        }

        /// <summary>A complex value with a NaN part and the key NumPy's complex order sorts it by.</summary>
        private readonly struct NaNPartKey
        {
            /// <summary>0 for <c>R+nanj</c>, 1 for <c>nan+Rj</c>, 2 for <c>nan+nanj</c>: the classes' order in <c>CDOUBLE_LT</c>.</summary>
            public readonly int Class;

            /// <summary>R, the part that is not NaN (0 for <c>nan+nanj</c>, whose values are all equal).</summary>
            public readonly double Key;

            /// <summary>The value's position in the input, which breaks ties.</summary>
            public readonly int Position;

            /// <summary>The value itself (its bits are kept).</summary>
            public readonly Complex Value;

            /// <summary>Classifies <paramref name="z"/>.</summary>
            /// <param name="z">A value with a NaN part.</param>
            /// <param name="position">Its position in the input.</param>
            public NaNPartKey(Complex z, int position)
            {
                bool reNaN = double.IsNaN(z.Real);
                Class = !reNaN ? 0 : double.IsNaN(z.Imaginary) ? 2 : 1;
                Key = Class == 0 ? z.Real : Class == 1 ? z.Imaginary : 0.0;
                Position = position;
                Value = z;
            }
        }

        /// <summary>Orders <see cref="NaNPartKey"/>s by class, then key (IEEE: both zeros equal), then position.</summary>
        private readonly struct NaNPartKeyComparer : IComparer<NaNPartKey>
        {
            /// <summary>Compares two keyed values.</summary>
            /// <param name="x">The first.</param>
            /// <param name="y">The second.</param>
            /// <returns>Negative, zero or positive as <paramref name="x"/> sorts before, with or after <paramref name="y"/>
            ///     (zero only for the same position).</returns>
            public int Compare(NaNPartKey x, NaNPartKey y)
            {
                if (x.Class != y.Class) return x.Class < y.Class ? -1 : 1;
                if (x.Key < y.Key) return -1;
                if (x.Key > y.Key) return 1;
                return x.Position.CompareTo(y.Position);
            }
        }

        /// <summary>Collapses signed zeros (-0.0→+0.0) per component so ±0 variants share a hash bucket.</summary>
        /// <param name="c">A complex value without NaN.</param>
        /// <returns>The value with each zero part made <c>+0.0</c> (used for hashing only; the table keeps the original
        ///     bits).</returns>
        [MethodImpl(OptimizeAndInline)]
        private static Complex NormalizeZeros(Complex c)
            => new Complex(c.Real == 0.0 ? 0.0 : c.Real, c.Imaginary == 0.0 ? 0.0 : c.Imaginary);

        /// <summary>Hashes a normalized (-0.0→+0.0, non-NaN) complex value by its two components' raw bits.</summary>
        /// <param name="v">The normalized value.</param>
        /// <returns>The 64-bit hash; the table index is its low bits.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static ulong HashKeyComplex(Complex v)
        {
            ulong r = (ulong)BitConverter.DoubleToInt64Bits(v.Real);
            ulong im = (ulong)BitConverter.DoubleToInt64Bits(v.Imaginary);
            return UniqueSplitmix((r * 0x9E3779B97F4A7C15UL) ^ (im * 0xC2B2AE3D27D4EB4FUL));
        }
    }
}
