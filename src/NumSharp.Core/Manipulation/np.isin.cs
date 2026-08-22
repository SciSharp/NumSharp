using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Calculates <c>element in test_elements</c>, broadcasting over <paramref name="element"/> only.<br></br>
        ///     Returns a boolean array of the same shape as <paramref name="element"/> that is True where an
        ///     element of <paramref name="element"/> is in <paramref name="test_elements"/> and False otherwise.
        /// </summary>
        /// <param name="element">Input array.</param>
        /// <param name="test_elements">The values against which to test each value of <paramref name="element"/>.
        ///     Flattened before use.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <param name="invert">If True, the values in the returned array are inverted, as if calculating
        ///     <c>element not in test_elements</c>. Default is False.</param>
        /// <param name="kind">The algorithm to use: <c>null</c> (auto), <c>"sort"</c>, or <c>"table"</c>. Does not
        ///     affect the result, only speed/memory. <c>"table"</c> is only valid for boolean/integer arrays.</param>
        /// <returns>A boolean array with the same shape as <paramref name="element"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isin.html</remarks>
        public static NDArray isin(NDArray element, NDArray test_elements,
            bool assume_unique = false, bool invert = false, string kind = null)
        {
            if (kind != null && kind != "sort" && kind != "table")
                throw new ValueError($"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.");

            // Result is shaped like element in C-order. Build the shape from the DIMENSIONS (a fresh
            // C-contiguous shape) — reusing element.Shape would carry a view's strides and scramble the
            // contiguous membership buffer on reshape (transposed/strided/negative-stride element).
            Shape outShape = new Shape(element.shape);
            NDArray ar1 = np.ravel(element);        // C-order flatten (NumPy: np.asarray(element).ravel())
            NDArray ar2 = np.ravel(test_elements);  // test values are always flattened

            // kind='table' is only defined for boolean/integer arrays, and can overflow for wide signed ranges.
            if (kind == "table")
            {
                if (!(IsIntegerOrBool(ar1.typecode) && IsIntegerOrBool(ar2.typecode)))
                    throw new ValueError("The 'table' method is only supported for boolean or integer arrays. "
                        + "Please select 'sort' or None for kind.");
                if (ar2.size > 0 && !TableRangeIsSafe(ar2))
                    throw new InvalidOperationException(
                        "You have specified kind='table', but the range of values in `ar2` or `ar1` exceed the "
                        + "maximum integer of the datatype. Please set `kind` to None or 'sort'.");
            }

            // Empty element -> empty boolean result shaped like element (value irrelevant).
            if (ar1.size == 0)
                return np.zeros(outShape, NPTypeCode.Boolean);

            // Empty test set -> nothing is a member (all invert).
            if (ar2.size == 0)
                return np.full(outShape, invert, NPTypeCode.Boolean);

            // assume_unique is a NumPy speed hint that skips deduplication; our sort/table paths need no
            // dedup, so it changes nothing here and the result is ALWAYS the correct membership (see the
            // documented divergence for assume_unique=true on genuinely non-unique inputs).
            NDArray member = ComputeMembership(ar1, ar2);
            NDArray result = invert ? np.logical_not(member) : member;
            return result.reshape(outShape);
        }

        /// <summary>
        ///     Membership of every value in <paramref name="ar1"/> against the set of values in <paramref name="ar2"/>,
        ///     both already flattened. Comparison happens in <c>result_type(ar1, ar2)</c> — matching NumPy, whose
        ///     brute/sort methods concatenate/compare in the promoted dtype.
        /// </summary>
        private static NDArray ComputeMembership(NDArray ar1, NDArray ar2)
        {
            NPTypeCode t = np.result_type(ar1.typecode, ar2.typecode);

            // Exact-integer guard: among integer/bool pairs, ONLY uint64-with-signed promotes to Double
            // (NEP50), which loses precision past 2^53 and would report two distinct integers as equal.
            // Membership between integers must be exact (as NumPy's table method is), so route that pair
            // through the uint64 domain with a sign filter rather than compare as float64.
            if (t == NPTypeCode.Double && IsIntegerOrBool(ar1.typecode) && IsIntegerOrBool(ar2.typecode))
                return MembershipUInt64Signed(ar1, ar2);

            NDArray elemT = ar1.typecode == t ? ar1 : ar1.astype(t);
            NDArray testT = ar2.typecode == t ? ar2 : ar2.astype(t);

            // Integer counting-table fast path (NumPy's 'table' method): when the value RANGE of the
            // test set is modest relative to the operand sizes, a direct lookup gather beats the
            // O(n log m) sort/search. uint64 is excluded (its values don't fit the int64 offset math);
            // its own kind is exact via SortedSearchMembership.
            if (IsIntegerOrBool(t) && t != NPTypeCode.UInt64)
            {
                NDArray table = TryTableMembership(elemT, testT);
                if (table is not null)
                    return table;
            }

            return SortedSearchMembership(elemT, testT);
        }

        /// <summary>
        ///     NumPy's counting-table membership: build a boolean lookup over [min(test), max(test)],
        ///     then gather one bool per element. Returns null (declining) when the range exceeds
        ///     6×(|element|+|test|) — NumPy's own below-memory heuristic — leaving the sort path to run.
        ///     Both operands are the same exact integer dtype (not uint64), so int64 offset math is exact.
        /// </summary>
        private static NDArray TryTableMembership(NDArray elemT, NDArray testT)
        {
            // Booleans have no arithmetic; NumPy promotes them to uint8 for the table (the result is
            // still the bool membership mask).
            if (elemT.typecode == NPTypeCode.Boolean) elemT = elemT.astype(NPTypeCode.Byte);
            if (testT.typecode == NPTypeCode.Boolean) testT = testT.astype(NPTypeCode.Byte);

            NDArray minS = np.amin(testT);                          // 0-d, dtype of the operands
            NDArray maxS = np.amax(testT);
            long minI = Convert.ToInt64(minS.GetAtIndex(0));
            long maxI = Convert.ToInt64(maxS.GetAtIndex(0));
            ulong urange = unchecked((ulong)(maxI - minI));
            if (urange > 6UL * (ulong)(elemT.size + testT.size))
                return null;                                       // range too wide -> sort path
            long range = (long)urange;

            NDArray table = np.zeros(new Shape(range + 1), NPTypeCode.Boolean);
            table[testT - minS] = (NDArray)true;                   // scatter: mark each present value

            // Stay in the operands' own dtype (no widening of the large element array). Out-of-range
            // elements clamp to a valid slot via take(mode='clip') and are then forced False by the
            // range mask regardless of the (possibly wrapped) offset — so no explicit bounds check.
            NDArray offset = elemT - minS;
            NDArray gathered = np.take(table, offset, mode: "clip");
            NDArray inRange = (elemT >= minS) & (elemT <= maxS);
            return gathered & inRange;
        }

        /// <summary>
        ///     Core membership over two operands ALREADY in the same dtype: sort the test set, then a
        ///     searchsorted + equality probe answers membership. Duplicates are harmless (side='left' finds
        ///     the first equal slot); NaN never equals itself, so a NaN element is never a member.
        /// </summary>
        private static NDArray SortedSearchMembership(NDArray elemT, NDArray testT)
        {
            // Hash-set membership (O(n+m)) replaces the sort + n·searchsorted below for hashable
            // dtypes. That searchsorted — n binary searches over the m-element sorted test set — was
            // the isin bottleneck (1M float64: ~317ms of a ~213ms call; NumPy's in1d is itself
            // sort-based here, so hashing BEATS it). Integer bit patterns are exact; floats normalize
            // -0.0→+0.0 and exclude NaN (NaN never equals itself — never a member). Complex/Half/
            // Decimal keep the sort path (Decimal needs value-equality: 1.0m == 1.00m by value, not
            // bits). Returns null when it declines (huge test set) so the sort path runs.
            NDArray hashed = TryHashMembership(elemT, testT);
            if (hashed is not null)
                return hashed;

            NDArray u = np.sort(testT);
            long m = u.size;
            if (m == 0)
                return np.zeros(new Shape(elemT.size), NPTypeCode.Boolean);

            NDArray idx = np.searchsorted(u, elemT, "left");        // int64, shape == elemT.shape (1-D)
            NDArray clamped = np.minimum(idx, (NDArray)(m - 1));    // stay in-bounds for the gather
            NDArray gathered = np.take(u, clamped);
            NDArray inRange = idx < (NDArray)m;                     // idx == m => greater than all => not found
            return (gathered == elemT) & inRange;
        }

        /// <summary>
        ///     Open-addressing hash-set membership: build a table of <paramref name="testT"/>'s
        ///     (normalized) bit patterns, then probe every element of <paramref name="elemT"/> — O(n+m)
        ///     vs the sort path's O(m log m + n log m). Handles the hashable dtypes (integer-family +
        ///     char + float32/float64); declines (returns null) for Complex/Half/Decimal and for a test
        ///     set too large to index, leaving <see cref="SortedSearchMembership"/> to run.
        /// </summary>
        private static NDArray TryHashMembership(NDArray elemT, NDArray testT)
        {
            // elemT and testT are the SAME dtype here (both cast to result_type upstream).
            if (testT.size > (1L << 29))   // ~512M distinct-candidates: fall back rather than over-allocate the table
                return null;
            switch (elemT.typecode)
            {
                case NPTypeCode.Boolean: return HashMembership<bool>(elemT, testT, floatSem: false);
                case NPTypeCode.Byte: return HashMembership<byte>(elemT, testT, floatSem: false);
                case NPTypeCode.SByte: return HashMembership<sbyte>(elemT, testT, floatSem: false);
                case NPTypeCode.Int16: return HashMembership<short>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt16: return HashMembership<ushort>(elemT, testT, floatSem: false);
                case NPTypeCode.Char: return HashMembership<char>(elemT, testT, floatSem: false);
                case NPTypeCode.Int32: return HashMembership<int>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt32: return HashMembership<uint>(elemT, testT, floatSem: false);
                case NPTypeCode.Int64: return HashMembership<long>(elemT, testT, floatSem: false);
                case NPTypeCode.UInt64: return HashMembership<ulong>(elemT, testT, floatSem: false);
                case NPTypeCode.Single: return HashMembership<float>(elemT, testT, floatSem: true);
                case NPTypeCode.Double: return HashMembership<double>(elemT, testT, floatSem: true);
                default: return null;   // Complex / Half / Decimal → sort path
            }
        }

        private static unsafe NDArray HashMembership<T>(NDArray elemT, NDArray testT, bool floatSem)
            where T : unmanaged
        {
            // `.Address` ignores Shape.offset, so a sliced/broadcast view must be materialized first
            // (the upstream astype already produces a fresh contiguous array when the dtype changed).
            if (elemT.Shape.IsSliced || elemT.Shape.IsBroadcasted) elemT = elemT.copy();
            if (testT.Shape.IsSliced || testT.Shape.IsBroadcasted) testT = testT.copy();
            long m = testT.size, n = elemT.size;

            // Pre-size to load factor ≤ 0.5 (next pow2 ≥ 2·m) so no growth/rehash is needed.
            long want = m * 2 + 1;
            int cap = 1024;
            while (cap < want) cap <<= 1;
            int mask = cap - 1;
            var keys = new ulong[cap];
            var used = new bool[cap];

            T* tp = (T*)testT.Address;
            for (long i = 0; i < m; i++)
            {
                T v = tp[i];
                if (floatSem && IsNaNFloat(v)) continue;   // NaN is never a member
                ulong k = KeyBits(v, floatSem);
                int h = (int)(Splitmix(k) & (ulong)mask);
                while (used[h]) { if (keys[h] == k) goto inserted; h = (h + 1) & mask; }
                used[h] = true; keys[h] = k;
                inserted: ;
            }

            var result = new NDArray(NPTypeCode.Boolean, new Shape(n), false);
            bool* rp = (bool*)result.Address;
            T* ep = (T*)elemT.Address;
            for (long i = 0; i < n; i++)
            {
                T v = ep[i];
                if (floatSem && IsNaNFloat(v)) { rp[i] = false; continue; }
                ulong k = KeyBits(v, floatSem);
                int h = (int)(Splitmix(k) & (ulong)mask);
                bool found = false;
                while (used[h]) { if (keys[h] == k) { found = true; break; } h = (h + 1) & mask; }
                rp[i] = found;
            }
            return result;
        }

        /// <summary>Normalized bit key: raw zero-extended bits for integers; for floats, -0.0→+0.0
        /// collapses to 0 so signed zeros share a bucket (NaN is filtered by the caller).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong KeyBits<T>(T v, bool floatSem) where T : unmanaged
        {
            if (floatSem)
            {
                if (typeof(T) == typeof(double)) { double d = SysUnsafe.As<T, double>(ref v); return d == 0.0 ? 0UL : BitConverter.DoubleToUInt64Bits(d); }
                if (typeof(T) == typeof(float)) { float f = SysUnsafe.As<T, float>(ref v); return f == 0f ? 0UL : BitConverter.SingleToUInt32Bits(f); }
            }
            int sz = SysUnsafe.SizeOf<T>();
            if (sz == 1) return SysUnsafe.As<T, byte>(ref v);
            if (sz == 2) return SysUnsafe.As<T, ushort>(ref v);
            if (sz == 4) return SysUnsafe.As<T, uint>(ref v);
            return SysUnsafe.As<T, ulong>(ref v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsNaNFloat<T>(T v) where T : unmanaged
        {
            if (typeof(T) == typeof(double)) return double.IsNaN(SysUnsafe.As<T, double>(ref v));
            if (typeof(T) == typeof(float)) return float.IsNaN(SysUnsafe.As<T, float>(ref v));
            return false;
        }

        /// <summary>splitmix64 finalizer — avalanches every bit into the low bits used as the table index.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong Splitmix(ulong k)
        {
            k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9UL;
            k = (k ^ (k >> 27)) * 0x94D049BB133111EBUL;
            return k ^ (k >> 31);
        }

        /// <summary>
        ///     Exact membership when one operand is uint64 and the other is a signed integer. Equality can
        ///     only hold in the overlap [0, 2^63-1]: a negative signed value never equals a uint64, and a
        ///     uint64 above int64.max never equals a signed value. Reduce to a uint64 comparison and mask
        ///     the signed operand's negatives out.
        /// </summary>
        private static NDArray MembershipUInt64Signed(NDArray ar1, NDArray ar2)
        {
            if (ar1.typecode == NPTypeCode.UInt64)
            {
                // element uint64, test signed: only test's non-negative values can match anything.
                NDArray testNonNeg = ar2[ar2 >= (NDArray)0L].astype(NPTypeCode.UInt64);
                return SortedSearchMembership(ar1, testNonNeg);
            }
            // element signed, test uint64: reinterpret element as uint64 for the probe, then exclude
            // negatives (whose reinterpreted huge value could otherwise collide with a real uint64 test value).
            NDArray raw = SortedSearchMembership(ar1.astype(NPTypeCode.UInt64), ar2);
            return raw & (ar1 >= (NDArray)0L);
        }

        private static bool IsIntegerOrBool(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        // range = max(ar2) - min(ar2); safe iff it fits the positive span of ar2's dtype (NumPy's
        // ar2_range <= iinfo(ar2.dtype).max). Only signed integer test dtypes can overflow: for unsigned
        // (and bool, treated as uint8) the range is always <= the type's max by construction.
        private static bool TableRangeIsSafe(NDArray ar2)
        {
            switch (ar2.typecode)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.Int32:
                case NPTypeCode.Int64:
                    long mn = Convert.ToInt64(np.amin(ar2).GetAtIndex(0));
                    long mx = Convert.ToInt64(np.amax(ar2).GetAtIndex(0));
                    ulong range = unchecked((ulong)(mx - mn));      // two's-complement distance, exact for all signed
                    ulong typeMax = ar2.typecode switch
                    {
                        NPTypeCode.SByte => (ulong)sbyte.MaxValue,
                        NPTypeCode.Int16 => (ulong)short.MaxValue,
                        NPTypeCode.Int32 => (ulong)int.MaxValue,
                        _ => (ulong)long.MaxValue,
                    };
                    return range <= typeMax;
                default:
                    return true;    // unsigned / bool: range always fits
            }
        }
    }
}
