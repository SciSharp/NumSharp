using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Sorting;
using NumSharp.Backends.Unmanaged;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp
{
    internal static partial class UniqueEngine
    {
        // ============================================================================
        //  RADIX flat sort for np.unique's SORT path.
        //
        //  np.unique's flat pipeline (UniqueEngine.Kwargs.cs) sorts a managed key buffer
        //  (+ perm for return_index/inverse/counts) and then masks/dedups. The sort was
        //  System.Array.Sort — the .NET BCL introsort, a COMPARISON sort (~n·log n comparisons,
        //  ~50% branch-mispredict on random data). This routes it instead through the SAME
        //  LSD RadixSort core the axis-sort driver runs (Backends/Default/Sorting/RadixSort.cs):
        //  O(n·nbytes), comparison-free, and — via the single-pass histogram + trivial-pass skip —
        //  cardinality/magnitude-adaptive (small integers do 1 pass, not 4).
        //
        //  Bit-parity is by construction: these transforms are the SAME monotonic unsigned keys
        //  AxisSort uses (so ascending key order == NumPy value order, -0.0 sorts before +0.0
        //  exactly as Array.Sort's default float comparer). The callers (since the np-layer plan's
        //  np.unique family, 2026-10-28):
        //    - the integer family, char and bool with an index or an inverse (and counts past the hash
        //      path's 2^28 bound) sort keys + perm here (RadixArgSortKeysPerm, uniqueFlatSorted<T>). Radix
        //      is stable, so a run's permutation order is ascending index (first occurrence), which the
        //      emit's min-perm-within-run scan also recovers: the outputs are those of the BCL path this
        //      replaced;
        //    - float32 / float64 values alone and counts alone sort the non-NaN prefix here
        //      (RadixSortValues) after the NaN partition, and WriteSortNaNs then writes NumPy's sort NaN
        //      into the tail (NumPy sorts in place there). The -0.0-below-+0.0 key makes -0.0 the value
        //      of a run of zeros whenever the line holds one; NumPy's x86-simd-sort orders equal keys its
        //      own way and may keep +0.0, a divergence left as it is;
        //    - the integer hash path's high-cardinality bail-out sorts and dedups here (RadixSortValues +
        //      RadixDedupSortedInt, UniqueEngine.Hash.cs).
        //  The index and inverse routes of float16 / float32 / float64 / complex128, and every float16
        //  call, do not come here: they are NumPy's argsort-and-gather statements over AxisSort.ArgSort
        //  (uniqueFlatArgsorted<T>, UniqueEngine.Kwargs.cs).
        //
        //  Scope: the 12 fixed-width numeric dtypes with a monotonic key (bool/byte/sbyte/int16/
        //  uint16/char/int32/uint32/int64/uint64/single/double). Half (no Vector<Half>, needs a
        //  float compare), Decimal (16-byte value equality) and Complex (lexicographic) never sort
        //  here — the same three dtypes AxisSort's VALUE sort routes to its scalar kernel (its argsort
        //  radix-sorts Half and Complex too, with keys that fold what NumPy calls equal).
        // ============================================================================

        /// <summary>True when dtype <paramref name="tc"/> has a monotonic radix key (see class remarks).</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>True for the 12 fixed-width dtypes the radix routes serve; false for Half, Decimal, Complex and text,
        ///     which never sort through this file (Decimal keeps the BCL sort; Half and Complex take the routes the class
        ///     remarks name).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool CanRadixSortUnique(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte
                or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                or NPTypeCode.Int32 or NPTypeCode.UInt32
                or NPTypeCode.Int64 or NPTypeCode.UInt64
                or NPTypeCode.Single or NPTypeCode.Double => true,
            _ => false,
        };

        /// <summary>
        ///     Radix-sorts <paramref name="keys"/><c>[0..count)</c> ascending IN PLACE (values only,
        ///     no perm). Drop-in for <c>System.Array.Sort(keys, 0, count)</c> on a radix-able dtype
        ///     (for floats, the caller passes the non-NaN prefix length so NaN stays untouched at the
        ///     tail). Same total order as the BCL comparer for every non-NaN value, so the mask/dedup
        ///     that follows produces an identical result.
        /// </summary>
        /// <typeparam name="T">A radix-able element type (see <see cref="CanRadixSortUnique"/>).</typeparam>
        /// <param name="keys">The keys; the first <paramref name="count"/> are sorted in place (a NaN among them would
        ///     sort by its bits, so float callers pass the non-NaN prefix only).</param>
        /// <param name="count">How many keys to sort; zero or one leaves them as they are.</param>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no radix key (a bug in the caller).</exception>
        /// <remarks>The key orders <c>-0.0</c> below <c>+0.0</c> (it is the raw bits made monotonic), so the two zeros do
        ///     not stay in input order.</remarks>
        private static unsafe void RadixSortValues<T>(T[] keys, int count) where T : unmanaged
        {
            if (count <= 1) return;
            int b = RadixKeyBytes<T>();
            if (b == 8)
            {
                var kb = GC.AllocateUninitializedArray<ulong>(count);
                var kt = GC.AllocateUninitializedArray<ulong>(count);
                var hist = GC.AllocateUninitializedArray<long>(8 * 256);
                for (int i = 0; i < count; i++) kb[i] = ToKey64(keys[i]);
                fixed (ulong* pk = kb, pt = kt)
                fixed (long* ph = hist)
                {
                    ulong* r = RadixSort.SortU64(pk, pt, count, ph);
                    for (int i = 0; i < count; i++) keys[i] = FromKey64<T>(r[i]);
                }
            }
            else
            {
                var kb = GC.AllocateUninitializedArray<uint>(count);
                var kt = GC.AllocateUninitializedArray<uint>(count);
                var hist = GC.AllocateUninitializedArray<long>(4 * 256);
                for (int i = 0; i < count; i++) kb[i] = ToKey32(keys[i]);
                fixed (uint* pk = kb, pt = kt)
                fixed (long* ph = hist)
                {
                    uint* r = RadixSort.SortU32(pk, pt, count, b, ph);
                    for (int i = 0; i < count; i++) keys[i] = FromKey32<T>(r[i]);
                }
            }
        }

        /// <summary>
        ///     Radix-argsorts <paramref name="keys"/><c>[0..count)</c> and co-moves
        ///     <paramref name="perm"/><c>[0..count)</c>, both IN PLACE. Drop-in for
        ///     <c>System.Array.Sort(keys, perm, 0, count)</c> on a radix-able dtype. The sorted values
        ///     come back through the key's inverse transform (no gather — the sorted key column is
        ///     already in memory order), the sorted permutation from the co-sorted index column.
        /// </summary>
        /// <typeparam name="T">A radix-able element type (see <see cref="CanRadixSortUnique"/>).</typeparam>
        /// <param name="keys">The keys; the first <paramref name="count"/> are sorted in place.</param>
        /// <param name="perm">The permutation, reordered with the keys (stable: equal keys keep their order).</param>
        /// <param name="count">How many keys to sort; zero or one leaves both arrays as they are.</param>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no radix key (a bug in the caller).</exception>
        /// <remarks>Stable for equal KEYS, but <c>-0.0</c> and <c>+0.0</c> have different keys: <c>-0.0</c> comes first
        ///     whatever the input order.</remarks>
        private static unsafe void RadixArgSortKeysPerm<T>(T[] keys, long[] perm, int count) where T : unmanaged
        {
            if (count <= 1) return;
            int b = RadixKeyBytes<T>();
            if (b == 8)
            {
                var kb = GC.AllocateUninitializedArray<ulong>(count);
                var kt = GC.AllocateUninitializedArray<ulong>(count);
                var it = GC.AllocateUninitializedArray<long>(count);
                var hist = GC.AllocateUninitializedArray<long>(8 * 256);
                for (int i = 0; i < count; i++) kb[i] = ToKey64(keys[i]);
                fixed (ulong* pk = kb, pkt = kt)
                fixed (long* pperm = perm, pit = it)
                fixed (long* ph = hist)
                {
                    long* sp = RadixSort.ArgSortU64(pk, pkt, pperm, pit, count, ph, out ulong* sk);
                    for (int i = 0; i < count; i++) { keys[i] = FromKey64<T>(sk[i]); perm[i] = sp[i]; }
                }
            }
            else
            {
                var kb = GC.AllocateUninitializedArray<uint>(count);
                var kt = GC.AllocateUninitializedArray<uint>(count);
                var it = GC.AllocateUninitializedArray<long>(count);
                var hist = GC.AllocateUninitializedArray<long>(4 * 256);
                for (int i = 0; i < count; i++) kb[i] = ToKey32(keys[i]);
                fixed (uint* pk = kb, pkt = kt)
                fixed (long* pperm = perm, pit = it)
                fixed (long* ph = hist)
                {
                    long* sp = RadixSort.ArgSortU32(pk, pkt, pperm, pit, count, b, ph, out uint* sk);
                    for (int i = 0; i < count; i++) { keys[i] = FromKey32<T>(sk[i]); perm[i] = sp[i]; }
                }
            }
        }

        /// <summary>
        ///     Dedup-emit for the hash path's high-cardinality bail-out: <paramref name="sorted"/> is a
        ///     radix-sorted (ascending) integer-family buffer of <paramref name="n"/> elements
        ///     (<paramref name="n"/> &gt; 0). Emits the distinct values and — when
        ///     <paramref name="wantCounts"/> — the aligned int64 run-length counts, matching the sort
        ///     path's <c>EmitValuesOnly</c>/<c>EmitOutputs</c> exactly (integer bit patterns bijection
        ///     with value, so this equals the hash result AS A SET, in the same ascending order).
        /// </summary>
        /// <typeparam name="T">The integer-family element type.</typeparam>
        /// <param name="sorted">The sorted elements.</param>
        /// <param name="n">Their count; must be positive (the first element is always kept).</param>
        /// <param name="wantCounts">Also build the run-length counts.</param>
        /// <returns>The fresh 1-D distinct values and, when <paramref name="wantCounts"/>, the aligned int64 counts (null
        ///     otherwise).</returns>
        private static unsafe (NDArray values, NDArray counts) RadixDedupSortedInt<T>(T[] sorted, int n, bool wantCounts)
            where T : unmanaged, IEquatable<T>
        {
            long uc = 1;
            for (int i = 1; i < n; i++) if (!sorted[i].Equals(sorted[i - 1])) uc++;

            var vblock = new UnmanagedMemoryBlock<T>(uc);
            T* vp = vblock.Address;
            UnmanagedMemoryBlock<long> cblock = default;
            long* cp = null;
            if (wantCounts) { cblock = new UnmanagedMemoryBlock<long>(uc); cp = cblock.Address; }

            vp[0] = sorted[0];
            long vi = 1;
            long runStart = 0;
            for (int i = 1; i < n; i++)
            {
                if (!sorted[i].Equals(sorted[i - 1]))
                {
                    if (wantCounts) cp[vi - 1] = i - runStart;
                    vp[vi++] = sorted[i];
                    runStart = i;
                }
            }
            if (wantCounts) cp[vi - 1] = n - runStart;

            var values = new NDArray(new ArraySlice<T>(vblock), Shape.Vector(uc));
            var counts = wantCounts ? new NDArray(new ArraySlice<long>(cblock), Shape.Vector(uc)) : null;
            return (values, counts);
        }

        // ----- monotonic key transforms (JIT-const typeof(T) dispatch; exactly one branch survives
        //       per specialized method — same pattern as HashKey<T> in UniqueEngine.Hash.cs).
        //       These mirror AxisSort's IKey32/IKey64/FKey adapters byte-for-byte. -----

        /// <summary>Significant key bytes for the u32 core (1/2/4), or 8 to signal the u64 core.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <returns>1, 2, 4 or 8.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no monotonic radix key.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RadixKeyBytes<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(bool) || typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte)) return 1;
            if (typeof(T) == typeof(short) || typeof(T) == typeof(ushort) || typeof(T) == typeof(char)) return 2;
            if (typeof(T) == typeof(int) || typeof(T) == typeof(uint) || typeof(T) == typeof(float)) return 4;
            if (typeof(T) == typeof(long) || typeof(T) == typeof(ulong) || typeof(T) == typeof(double)) return 8;
            throw new NotSupportedException($"RadixKeyBytes: {typeof(T)} has no monotonic radix key.");
        }

        /// <summary>
        ///     The 32-bit radix key of a 1-, 2- or 4-byte value: unsigned values as they are, signed ones with the sign bit
        ///     flipped, float32 bits made monotonic (negative values inverted, positive ones with the sign bit set).
        /// </summary>
        /// <typeparam name="T">A 1-, 2- or 4-byte radix-able element type.</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>A key whose unsigned order is the value order (a NaN keys by its bits).</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no 32-bit radix key.</exception>
        [MethodImpl(OptimizeAndInline)]
        private static uint ToKey32<T>(T v) where T : unmanaged
        {
            if (typeof(T) == typeof(byte) || typeof(T) == typeof(bool)) return SysUnsafe.As<T, byte>(ref v);
            if (typeof(T) == typeof(sbyte)) { sbyte s = SysUnsafe.As<T, sbyte>(ref v); return (byte)(s ^ unchecked((sbyte)0x80)); }
            if (typeof(T) == typeof(short)) { short s = SysUnsafe.As<T, short>(ref v); return (ushort)(s ^ unchecked((short)0x8000)); }
            if (typeof(T) == typeof(ushort)) return SysUnsafe.As<T, ushort>(ref v);
            if (typeof(T) == typeof(char)) return SysUnsafe.As<T, char>(ref v);
            if (typeof(T) == typeof(int)) { int s = SysUnsafe.As<T, int>(ref v); return (uint)s ^ 0x80000000u; }
            if (typeof(T) == typeof(uint)) return SysUnsafe.As<T, uint>(ref v);
            if (typeof(T) == typeof(float)) { float s = SysUnsafe.As<T, float>(ref v); uint bch = BitConverter.SingleToUInt32Bits(s); return bch ^ ((uint)((int)bch >> 31) | 0x80000000u); }
            throw new NotSupportedException();
        }

        /// <summary>
        ///     The 64-bit radix key of an 8-byte value: uint64 as it is, int64 with the sign bit flipped, float64 bits made
        ///     monotonic.
        /// </summary>
        /// <typeparam name="T">long, ulong or double.</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>A key whose unsigned order is the value order (a NaN keys by its bits).</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no 64-bit radix key.</exception>
        [MethodImpl(OptimizeAndInline)]
        private static ulong ToKey64<T>(T v) where T : unmanaged
        {
            if (typeof(T) == typeof(long)) { long s = SysUnsafe.As<T, long>(ref v); return (ulong)s ^ 0x8000000000000000UL; }
            if (typeof(T) == typeof(ulong)) return SysUnsafe.As<T, ulong>(ref v);
            if (typeof(T) == typeof(double)) { double s = SysUnsafe.As<T, double>(ref v); ulong bch = BitConverter.DoubleToUInt64Bits(s); return bch ^ ((ulong)((long)bch >> 63) | 0x8000000000000000UL); }
            throw new NotSupportedException();
        }

        /// <summary>The inverse of <see cref="ToKey32{T}"/>: the value whose key is <paramref name="k"/>, bits exact.</summary>
        /// <typeparam name="T">A 1-, 2- or 4-byte radix-able element type.</typeparam>
        /// <param name="k">The key.</param>
        /// <returns>The value.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no 32-bit radix key.</exception>
        [MethodImpl(OptimizeAndInline)]
        private static T FromKey32<T>(uint k) where T : unmanaged
        {
            if (typeof(T) == typeof(byte) || typeof(T) == typeof(bool)) { byte x = (byte)k; return SysUnsafe.As<byte, T>(ref x); }
            if (typeof(T) == typeof(sbyte)) { sbyte x = (sbyte)((byte)k ^ 0x80); return SysUnsafe.As<sbyte, T>(ref x); }
            if (typeof(T) == typeof(short)) { short x = (short)((ushort)k ^ 0x8000); return SysUnsafe.As<short, T>(ref x); }
            if (typeof(T) == typeof(ushort)) { ushort x = (ushort)k; return SysUnsafe.As<ushort, T>(ref x); }
            if (typeof(T) == typeof(char)) { char x = (char)k; return SysUnsafe.As<char, T>(ref x); }
            if (typeof(T) == typeof(int)) { int x = (int)(k ^ 0x80000000u); return SysUnsafe.As<int, T>(ref x); }
            if (typeof(T) == typeof(uint)) { uint x = k; return SysUnsafe.As<uint, T>(ref x); }
            if (typeof(T) == typeof(float)) { uint bch = k ^ (((k >> 31) - 1) | 0x80000000u); float x = BitConverter.UInt32BitsToSingle(bch); return SysUnsafe.As<float, T>(ref x); }
            throw new NotSupportedException();
        }

        /// <summary>The inverse of <see cref="ToKey64{T}"/>: the value whose key is <paramref name="k"/>, bits exact.</summary>
        /// <typeparam name="T">long, ulong or double.</typeparam>
        /// <param name="k">The key.</param>
        /// <returns>The value.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no 64-bit radix key.</exception>
        [MethodImpl(OptimizeAndInline)]
        private static T FromKey64<T>(ulong k) where T : unmanaged
        {
            if (typeof(T) == typeof(long)) { long x = (long)(k ^ 0x8000000000000000UL); return SysUnsafe.As<long, T>(ref x); }
            if (typeof(T) == typeof(ulong)) { ulong x = k; return SysUnsafe.As<ulong, T>(ref x); }
            if (typeof(T) == typeof(double)) { ulong bch = k ^ (((k >> 63) - 1) | 0x8000000000000000UL); double x = BitConverter.UInt64BitsToDouble(bch); return SysUnsafe.As<double, T>(ref x); }
            throw new NotSupportedException();
        }
    }
}
