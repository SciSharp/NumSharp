using System;
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
        ///     consistent with <see cref="IEquatable{T}"/> equality) through a Fibonacci multiply-xor
        ///     mix. The <c>SizeOf&lt;T&gt;</c> tests are JIT constants per instantiation, so exactly
        ///     one branch survives per specialized method.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong HashKey<T>(T v) where T : unmanaged
        {
            ulong k;
            if (SysUnsafe.SizeOf<T>() == 8) k = SysUnsafe.As<T, ulong>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 4) k = SysUnsafe.As<T, uint>(ref v);
            else if (SysUnsafe.SizeOf<T>() == 2) k = SysUnsafe.As<T, ushort>(ref v);
            else k = SysUnsafe.As<T, byte>(ref v);
            k *= 0x9E3779B97F4A7C15UL;
            return k ^ (k >> 29);
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
    }
}
