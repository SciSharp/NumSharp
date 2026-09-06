using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp
{
    public partial class NDArray
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
        internal NDArray uniqueValuesFast()
        {
            switch (typecode)
            {
                case NPTypeCode.Boolean: return UniqueHashSortedInt<bool>(wantCounts: false).values;
                case NPTypeCode.Byte: return UniqueHashSortedInt<byte>(wantCounts: false).values;
                case NPTypeCode.SByte: return UniqueHashSortedInt<sbyte>(wantCounts: false).values;
                case NPTypeCode.Int16: return UniqueHashSortedInt<short>(wantCounts: false).values;
                case NPTypeCode.UInt16: return UniqueHashSortedInt<ushort>(wantCounts: false).values;
                case NPTypeCode.Int32: return UniqueHashSortedInt<int>(wantCounts: false).values;
                case NPTypeCode.UInt32: return UniqueHashSortedInt<uint>(wantCounts: false).values;
                case NPTypeCode.Int64: return UniqueHashSortedInt<long>(wantCounts: false).values;
                case NPTypeCode.UInt64: return UniqueHashSortedInt<ulong>(wantCounts: false).values;
                case NPTypeCode.Char: return UniqueHashSortedInt<char>(wantCounts: false).values;
                case NPTypeCode.Complex: return UniqueHashSortedComplex(wantCounts: false).values;
                // Half/Single/Double take the SORT path (NumPy's own choice — floats are NOT in its
                // hash map): NumPy sorts them with SIMD vqsort, and a hash regresses ~2x on all-unique
                // input (insert N + scalar-sort N) with no cheap way to detect that cardinality. Decimal
                // needs value-equality (1.0m != 1.00m by bits), so it sorts too.
                default: return unique(false, false, false, axis: null, equal_nan: false)[0];
            }
        }

        /// <summary>
        ///     Sorted unique values + counts for <c>np.unique_counts</c>. Integer-family dtypes
        ///     take the hash path; everything else routes through the sort path (equal_nan=False).
        /// </summary>
        internal NDArray[] uniqueCountsFast()
        {
            switch (typecode)
            {
                case NPTypeCode.Boolean: { var (v, c) = UniqueHashSortedInt<bool>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Byte: { var (v, c) = UniqueHashSortedInt<byte>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.SByte: { var (v, c) = UniqueHashSortedInt<sbyte>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int16: { var (v, c) = UniqueHashSortedInt<short>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt16: { var (v, c) = UniqueHashSortedInt<ushort>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int32: { var (v, c) = UniqueHashSortedInt<int>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt32: { var (v, c) = UniqueHashSortedInt<uint>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Int64: { var (v, c) = UniqueHashSortedInt<long>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.UInt64: { var (v, c) = UniqueHashSortedInt<ulong>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Char: { var (v, c) = UniqueHashSortedInt<char>(wantCounts: true); return new[] { v, c }; }
                case NPTypeCode.Complex: { var (v, c) = UniqueHashSortedComplex(wantCounts: true); return new[] { v, c }; }
                // Half/Single/Double/Decimal take the SORT path — see uniqueValuesFast.
                default: return unique(false, false, true, axis: null, equal_nan: false);
            }
        }

        /// <summary>
        ///     Open-addressing (linear-probe) dedup of an integer-family array, returning the SORTED
        ///     distinct values and — when <paramref name="wantCounts"/> — the aligned occurrence counts
        ///     (int64). The table grows on demand (load factor ≤ 0.5) so memory stays O(unique), not O(n).
        /// </summary>
        private unsafe (NDArray values, NDArray counts) UniqueHashSortedInt<T>(bool wantCounts)
            where T : unmanaged, IComparable<T>, IEquatable<T>
        {
            long n = size;
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
                var r = unique(false, false, wantCounts, axis: null, equal_nan: false);
                return (r[0], wantCounts ? r[1] : null);
            }

            var data = ExtractKeysOnly<T>(n);   // resolves any layout (strided/transposed/reversed) to C-order

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
        [MethodImpl(OptimizeAndInline)]
        private static ulong HashKey<T>(T v) where T : unmanaged
        {
            ulong k;
            if (SysUnsafe.SizeOf<T>() == 8) k = SysUnsafe.As<T, ulong>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 4) k = SysUnsafe.As<T, uint>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 2) k = SysUnsafe.As<T, ushort>(ref v);
            else k = SysUnsafe.As<T, byte>(ref v);
            return Splitmix(k);
        }

        /// <summary>splitmix64 finalizer — avalanches every input bit into the low bits used as the table index.</summary>
        [MethodImpl(OptimizeAndInline)]
        private static ulong Splitmix(ulong k)
        {
            k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9UL;
            k = (k ^ (k >> 27)) * 0x94D049BB133111EBUL;
            return k ^ (k >> 31);
        }

        /// <summary>Wraps a managed <typeparamref name="T"/><c>[]</c> as a fresh 1-D <see cref="NDArray"/> (unmanaged copy).</summary>
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
        //  (each is DISTINCT and appended after the sorted finite set, matching equal_nan=False), and
        //  -0.0 is normalized to +0.0 PER COMPONENT for HASHING only — the FIRST-occurrence original
        //  bits are stored (NumPy keeps the first element's signed zero), and equality is Complex `==`
        //  (IEEE per component: -0.0 == +0.0). The finite set is lex-sorted (real, then imaginary).
        // ============================================================================

        private static readonly IComparer<Complex> ComplexLexComparer = Comparer<Complex>.Create((x, y) =>
        {
            int c = x.Real.CompareTo(y.Real);
            return c != 0 ? c : x.Imaginary.CompareTo(y.Imaginary);
        });

        private unsafe (NDArray values, NDArray counts) UniqueHashSortedComplex(bool wantCounts)
        {
            long n = size;
            if (n == 0)
            {
                var ev = new NDArray(new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>(0)), Shape.Vector(0));
                var ec = wantCounts ? new NDArray(new ArraySlice<long>(new UnmanagedMemoryBlock<long>(0)), Shape.Vector(0)) : null;
                return (ev, ec);
            }
            if (n > (1L << 28))
            {
                var r = unique(false, false, wantCounts, axis: null, equal_nan: false);
                return (r[0], wantCounts ? r[1] : null);
            }

            var data = ExtractKeysOnly<Complex>(n);
            List<Complex> nans = null;

            int cap = 1024;
            var table = new Complex[cap];
            var used = new bool[cap];
            var cnt = wantCounts ? new long[cap] : null;
            int mask = cap - 1;
            int uniqueCount = 0;

            for (int i = 0; i < n; i++)
            {
                Complex c = data[i];
                if (double.IsNaN(c.Real) || double.IsNaN(c.Imaginary)) { (nans ??= new List<Complex>()).Add(c); continue; }  // any-NaN distinct
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

            var uv = new Complex[uniqueCount]; var uc = wantCounts ? new long[uniqueCount] : null;
            int j = 0;
            for (int s = 0; s < cap; s++) { if (!used[s]) continue; uv[j] = table[s]; if (wantCounts) uc[j] = cnt[s]; j++; }
            if (wantCounts) System.Array.Sort(uv, uc, ComplexLexComparer); else System.Array.Sort(uv, ComplexLexComparer);

            return AppendNaNsAndBuild(uv, uc, uniqueCount, nans, nans?.Count ?? 0, wantCounts);
        }

        /// <summary>Builds the value array (sorted finite set + each distinct NaN appended, input order) and, when requested, aligned int64 counts (finite counts + 1 per NaN).</summary>
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

        /// <summary>Collapses signed zeros (-0.0→+0.0) per component so ±0 variants share a hash bucket.</summary>
        [MethodImpl(OptimizeAndInline)]
        private static Complex NormalizeZeros(Complex c)
            => new Complex(c.Real == 0.0 ? 0.0 : c.Real, c.Imaginary == 0.0 ? 0.0 : c.Imaginary);

        /// <summary>Hashes a normalized (-0.0→+0.0, non-NaN) complex value by its two components' raw bits.</summary>
        [MethodImpl(OptimizeAndInline)]
        private static ulong HashKeyComplex(Complex v)
        {
            ulong r = (ulong)BitConverter.DoubleToInt64Bits(v.Real);
            ulong im = (ulong)BitConverter.DoubleToInt64Bits(v.Imaginary);
            return Splitmix((r * 0x9E3779B97F4A7C15UL) ^ (im * 0xC2B2AE3D27D4EB4FUL));
        }
    }
}
