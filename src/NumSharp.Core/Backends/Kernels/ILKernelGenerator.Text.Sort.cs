using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Text.Sort.cs — the element loops of the fixed-width text sort (S / U)
// =============================================================================
//
// RESPONSIBILITY
//   The two loops the text sort (Backends/Default/Sorting/TextSort.cs) runs over the strings themselves; everything else
//   it does works on 64-bit keys and indices:
//   * the PREFIX KEY of every selected element: its first code units packed big-endian into a ulong, so unsigned key
//     order is the strings' order on that prefix (NumPy's STRING_LT / UNICODE_LT compare unsigned units);
//   * the GATHER: element index[j] of a strided source copied to slot j of a strided destination (a sorted line written
//     back, a line copied into a contiguous scratch buffer).
//
// THE KEY PACKING (plan D3)
//   S: the first 8 bytes. U: the first k units with b bits each, chosen by the caller from the largest unit it saw:
//   k = 8, b = 8 (every unit below 2^8, so ASCII and Latin-1 text keeps 8 characters in a key), k = 4, b = 16 (below 2^16),
//   k = 3, b = 21 (below 2^21, every Unicode code point), k = 2, b = 32 (any 32-bit value, the units above U+10FFFF a view
//   can hold). Unit t sits at bit 64 - b*(t + 1), so the first unit is the most significant: key order is lexicographic
//   order on the k units. An element narrower than k units keeps zeros below its last unit, exactly the zero padding the
//   comparison of two elements of one width reads.
//   The b = 8 form of U also ORs the units it reads into four accumulators (units 0-1, 2, 3, 4-7) and stores them through
//   its last argument: from them the caller learns whether every unit fit in 8 bits (the keys stand) or which wider
//   packing to run instead (units 0-2 for 21 bits, 0-3 for 16 bits). ASCII text thus costs one pass.
//   With 128-bit vectors the 8-unit form of it is two loads per element (TextSortUnitKey8): the units are narrowed to
//   bytes (truncating — a unit that does not fit shows in the OR, and the keys are then recomputed) and byte-swapped;
//   the OR accumulates per position in two vectors, folded into the four groups once at the end.
//
// CONTRACT
//   Element j of a key kernel is read at text + index[j] * stride, or at text + j * stride when index is null (a line's
//   first pass), and only its first Units code units are read. A gather reads element (index == null ? j : index[j]) at
//   src + that * srcStride and writes Width bytes at dst + j * dstStride; source and destination must not overlap.
//   Every read and write is unaligned-safe: an S element sits at any byte, a U view at any 4-byte boundary or worse.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     A text sort's prefix-key loop: <c>keys[j]</c> = the order-preserving 64-bit key of the element at
    ///     <c>text + index[j] * stride</c>, or <c>text + j * stride</c> when <paramref name="index"/> is null (see
    ///     <see cref="ILKernelGenerator.GetTextSortKeyKernel"/> for the packing).
    /// </summary>
    /// <param name="text">The first byte of element 0's key range (the caller adds a unit offset to key a later part).</param>
    /// <param name="stride">The byte distance between consecutive elements (any sign, 0 for a broadcast line).</param>
    /// <param name="index">The element of each key slot, or null for the identity (slot j keys element j).</param>
    /// <param name="count">The number of keys to write.</param>
    /// <param name="keys">The key slots, <paramref name="count"/> of them.</param>
    /// <param name="unitOr">Four uint slots that receive the OR of the units read at positions 0-1, 2, 3 and 4-7, for the
    ///     kernels that collect it (the 8-bit U packing); ignored (may be null) by the others.</param>
    internal unsafe delegate void TextSortKeyFunc(byte* text, long stride, long* index, long count, ulong* keys, uint* unitOr);

    /// <summary>
    ///     A text gather: copies the element at <c>src + i * srcStride</c>, i = <c>index[j]</c> (or j when
    ///     <paramref name="index"/> is null), to <c>dst + j * dstStride</c>, for every j below <paramref name="count"/>.
    /// </summary>
    /// <param name="src">Element 0 of the source.</param>
    /// <param name="srcStride">The source's byte stride (any sign).</param>
    /// <param name="index">The source element of each destination slot, or null for the identity.</param>
    /// <param name="count">The number of elements to copy.</param>
    /// <param name="dst">Slot 0 of the destination (must not overlap the source).</param>
    /// <param name="dstStride">The destination's byte stride (any sign).</param>
    internal unsafe delegate void TextGatherFunc(byte* src, long srcStride, long* index, long count, byte* dst, long dstStride);

    public static partial class ILKernelGenerator
    {
        /// <summary>The prefix-key kernels, keyed by (unit, units read, bits per unit, OR collection).</summary>
        private static readonly ConcurrentDictionary<TextSortKeyKey, TextSortKeyFunc> s_textSortKeyKernels = new();

        /// <summary>The gather kernels, keyed by the element width in bytes.</summary>
        private static readonly ConcurrentDictionary<int, TextGatherFunc> s_textGatherKernels = new();

        /// <summary><see cref="BinaryPrimitives.ReverseEndianness(ulong)"/>: one <c>bswap</c> turns an 8-byte S prefix into its key.</summary>
        private static readonly MethodInfo s_bswap64 = typeof(BinaryPrimitives).GetMethod(nameof(BinaryPrimitives.ReverseEndianness), new[] { typeof(ulong) })
            ?? throw new MissingMethodException(nameof(BinaryPrimitives), nameof(BinaryPrimitives.ReverseEndianness));

        /// <summary><see cref="BinaryPrimitives.ReverseEndianness(uint)"/>: the 4-byte head of a 4..7-byte S prefix.</summary>
        private static readonly MethodInfo s_bswap32 = typeof(BinaryPrimitives).GetMethod(nameof(BinaryPrimitives.ReverseEndianness), new[] { typeof(uint) })
            ?? throw new MissingMethodException(nameof(BinaryPrimitives), nameof(BinaryPrimitives.ReverseEndianness));

        /// <summary><see cref="BitOperations.RotateLeft(ulong, int)"/>: two U units of 32 bits from one 8-byte load.</summary>
        private static readonly MethodInfo s_rotl64 = typeof(BitOperations).GetMethod(nameof(BitOperations.RotateLeft), new[] { typeof(ulong), typeof(int) })
            ?? throw new MissingMethodException(nameof(BitOperations), nameof(BitOperations.RotateLeft));

        /// <summary>The identity of a prefix-key kernel: everything its IL depends on.</summary>
        /// <param name="Unit">The code-unit size: 1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="Units">How many leading units go into the key (1 to 64 / <paramref name="Bits"/>).</param>
        /// <param name="Bits">The bits each unit takes in the key: 8 for <c>S</c>; 8, 16, 21 or 32 for <c>U</c>.</param>
        /// <param name="CollectOr">Whether the kernel also ORs the units it reads into the four accumulators (the 8-bit
        ///     <c>U</c> packing only).</param>
        private readonly record struct TextSortKeyKey(int Unit, int Units, int Bits, bool CollectOr);

        /// <summary>
        ///     The prefix-key kernel for one packing, generated on first use and cached (see the file header for the
        ///     packing and the contract).
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="units">The leading units packed into each key: 1 to 8 for <c>S</c> and for the 8-bit <c>U</c>
        ///     packing, at most 4 / 3 / 2 for 16 / 21 / 32 bits. Fewer than the packing holds when the element (or the part
        ///     of it still to key) is narrower; the key's low bits are then zero, as the zero padding reads.</param>
        /// <param name="bits">The bits per unit: 8 for <c>S</c>; 8, 16, 21 or 32 for <c>U</c>. A unit must be below
        ///     2^<paramref name="bits"/> for the key to be order-preserving (the caller checks it with the 8-bit kernel's
        ///     ORs before running a narrower packing).</param>
        /// <param name="collectOr">Collect the OR of the units read (allowed only for <c>U</c> with 8 bits).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller runs
        ///     <see cref="TextSortKeysManaged"/> instead).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit other than 1 or 4, a bit count the unit has no packing
        ///     for, a unit count outside 1 to 64 / <paramref name="bits"/>, or OR collection asked of another packing.</exception>
        internal static TextSortKeyFunc GetTextSortKeyKernel(int unit, int units, int bits, bool collectOr)
        {
            ValidateTextSortKey(unit, units, bits, collectOr);
            if (!TextKernelsAvailable)
                return null;
            var key = new TextSortKeyKey(unit, units, bits, collectOr);
            if (s_textSortKeyKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textSortKeyKernels.GetOrAdd(key, GenerateTextSortKeyKernel(key));
            }
            catch (Exception ex)
            {
                // An emission failure must not fail the sort: the caller falls back to the managed loop.
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextSortKeyKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The gather kernel for one element width, generated on first use and cached.
        /// </summary>
        /// <param name="width">The element width in bytes (0 copies nothing).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller runs
        ///     <see cref="TextGatherManaged"/> instead).</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/> is negative.</exception>
        internal static TextGatherFunc GetTextGatherKernel(int width)
        {
            if (width < 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "An element width cannot be negative.");
            if (!TextKernelsAvailable)
                return null;
            if (s_textGatherKernels.TryGetValue(width, out var cached))
                return cached;
            try
            {
                return s_textGatherKernels.GetOrAdd(width, GenerateTextGatherKernel(width));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextGatherKernel({width}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The managed twin of a prefix-key kernel (same keys, same ORs), for hosts without dynamic code and for the
        ///     tests that hold the emitted kernels to it.
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="units">The leading units packed into each key.</param>
        /// <param name="bits">The bits per unit.</param>
        /// <param name="collectOr">Collect the OR of the units read into <paramref name="unitOr"/>.</param>
        /// <param name="text">See <see cref="TextSortKeyFunc"/>.</param>
        /// <param name="stride">See <see cref="TextSortKeyFunc"/>.</param>
        /// <param name="index">See <see cref="TextSortKeyFunc"/>.</param>
        /// <param name="count">See <see cref="TextSortKeyFunc"/>.</param>
        /// <param name="keys">See <see cref="TextSortKeyFunc"/>.</param>
        /// <param name="unitOr">See <see cref="TextSortKeyFunc"/>.</param>
        /// <exception cref="ArgumentOutOfRangeException">An invalid packing (see <see cref="GetTextSortKeyKernel"/>).</exception>
        internal static unsafe void TextSortKeysManaged(int unit, int units, int bits, bool collectOr,
                                                        byte* text, long stride, long* index, long count, ulong* keys, uint* unitOr)
        {
            ValidateTextSortKey(unit, units, bits, collectOr);
            uint or01 = 0, or2 = 0, or3 = 0, or47 = 0;
            for (long j = 0; j < count; j++)
            {
                byte* p = text + (index is null ? j : index[j]) * stride;
                ulong key = 0;
                for (int t = 0; t < units; t++)
                {
                    uint u = unit == 1 ? p[t] : Unsafe.ReadUnaligned<uint>(p + 4 * t);
                    key |= (ulong)u << (64 - bits * (t + 1));
                    if (collectOr)
                    {
                        if (t < 2) or01 |= u;
                        else if (t == 2) or2 |= u;
                        else if (t == 3) or3 |= u;
                        else or47 |= u;
                    }
                }
                keys[j] = key;
            }
            if (collectOr)
            {
                unitOr[0] = or01;
                unitOr[1] = or2;
                unitOr[2] = or3;
                unitOr[3] = or47;
            }
        }

        /// <summary>The managed twin of a gather kernel (same bytes), for hosts without dynamic code.</summary>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="src">See <see cref="TextGatherFunc"/>.</param>
        /// <param name="srcStride">See <see cref="TextGatherFunc"/>.</param>
        /// <param name="index">See <see cref="TextGatherFunc"/>.</param>
        /// <param name="count">See <see cref="TextGatherFunc"/>.</param>
        /// <param name="dst">See <see cref="TextGatherFunc"/>.</param>
        /// <param name="dstStride">See <see cref="TextGatherFunc"/>.</param>
        internal static unsafe void TextGatherManaged(int width, byte* src, long srcStride, long* index, long count, byte* dst, long dstStride)
        {
            for (long j = 0; j < count; j++)
            {
                long i = index is null ? j : index[j];
                Buffer.MemoryCopy(src + i * srcStride, dst + j * dstStride, width, width);
            }
        }

        /// <summary>Rejects a packing no prefix-key kernel has (see <see cref="GetTextSortKeyKernel"/>).</summary>
        /// <param name="unit">The code-unit size.</param>
        /// <param name="units">The units per key.</param>
        /// <param name="bits">The bits per unit.</param>
        /// <param name="collectOr">The OR collection flag.</param>
        /// <exception cref="ArgumentOutOfRangeException">The combination is not one of the packings.</exception>
        private static void ValidateTextSortKey(int unit, int units, int bits, bool collectOr)
        {
            if (unit is not (1 or 4))
                throw new ArgumentOutOfRangeException(nameof(unit), unit, "A string code unit is 1 byte (S) or 4 bytes (U).");
            bool bitsOk = unit == 1 ? bits == 8 : bits is 8 or 16 or 21 or 32;
            if (!bitsOk)
                throw new ArgumentOutOfRangeException(nameof(bits), bits, "S keys take 8 bits a unit; U keys 8, 16, 21 or 32.");
            if (units < 1 || units > 64 / bits)
                throw new ArgumentOutOfRangeException(nameof(units), units, $"A {bits}-bit packing holds 1 to {64 / bits} units.");
            if (collectOr && !(unit == 4 && bits == 8))
                throw new ArgumentOutOfRangeException(nameof(collectOr), collectOr, "Only the 8-bit U packing collects the unit OR.");
        }

        /// <summary><see cref="TextSortUnitKey8"/>, the vector body of the 8-unit, 8-bit <c>U</c> key kernel.</summary>
        private static readonly MethodInfo s_unitKey8 = typeof(ILKernelGenerator).GetMethod(nameof(TextSortUnitKey8), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(ILKernelGenerator), nameof(TextSortUnitKey8));

        /// <summary><see cref="TextSortStoreUnitOr"/>, the vector kernel's epilogue.</summary>
        private static readonly MethodInfo s_storeUnitOr = typeof(ILKernelGenerator).GetMethod(nameof(TextSortStoreUnitOr), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(ILKernelGenerator), nameof(TextSortStoreUnitOr));

        /// <summary>
        ///     The 8-bit key of the first eight units of a <c>U</c> element, from two 16-byte loads: the units narrowed to
        ///     bytes by truncation, in big-endian order. The units are ORed into the per-position accumulators, from which the
        ///     caller learns whether every unit fit in 8 bits (only then is the truncated key the key).
        /// </summary>
        /// <param name="p">The element's first unit (any alignment; the element holds at least 8 units).</param>
        /// <param name="orLo">The OR of units 0-3 so far, per position.</param>
        /// <param name="orHi">The OR of units 4-7 so far, per position.</param>
        /// <returns>The key.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ulong TextSortUnitKey8(byte* p, ref Vector128<uint> orLo, ref Vector128<uint> orHi)
        {
            var lo = Vector128.Load((uint*)p);
            var hi = Vector128.Load((uint*)(p + 16));
            orLo |= lo;
            orHi |= hi;
            // Narrow truncates (each lane keeps its low bits): units 0-7 become bytes 0-7, unit 0 in the lowest byte.
            var bytes = Vector128.Narrow(Vector128.Narrow(lo, hi), Vector128<ushort>.Zero);
            return BinaryPrimitives.ReverseEndianness(bytes.AsUInt64().ToScalar());
        }

        /// <summary>Folds the per-position ORs of the vector key kernel into the four groups the scalar kernels store.</summary>
        /// <param name="orLo">The OR of units 0-3, per position.</param>
        /// <param name="orHi">The OR of units 4-7, per position.</param>
        /// <param name="unitOr">Receives the OR of units 0-1, 2, 3 and 4-7.</param>
        private static unsafe void TextSortStoreUnitOr(Vector128<uint> orLo, Vector128<uint> orHi, uint* unitOr)
        {
            unitOr[0] = orLo.GetElement(0) | orLo.GetElement(1);
            unitOr[1] = orLo.GetElement(2);
            unitOr[2] = orLo.GetElement(3);
            unitOr[3] = orHi.GetElement(0) | orHi.GetElement(1) | orHi.GetElement(2) | orHi.GetElement(3);
        }

        /// <summary>
        ///     Emits a prefix-key kernel: <c>for (j = 0; j &lt; count; j++) { p = text + index[j] * stride; keys[j] = pack(p); }</c>,
        ///     the packing straight-line code over the key's units (emit-time constants), then the four OR accumulators stored
        ///     when the kernel collects them.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static TextSortKeyFunc GenerateTextSortKeyKernel(TextSortKeyKey key)
        {
            var dm = new DynamicMethod($"NDTextSortKey_u{key.Unit}_n{key.Units}_b{key.Bits}{(key.CollectOr ? "_or" : "")}",
                typeof(void), new[] { typeof(byte*), typeof(long), typeof(long*), typeof(long), typeof(ulong*), typeof(uint*) },
                typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var j = il.DeclareLocal(typeof(long));
            var p = il.DeclareLocal(typeof(byte*));
            // The 8-unit, 8-bit U key with OR collection is the vector body when 128-bit vectors run in hardware: per-position
            // OR accumulators (zeroed: DynamicMethod locals are initialized) instead of the four scalar groups.
            bool vector = key.Unit == 4 && key.Bits == 8 && key.Units == 8 && key.CollectOr && Vector128.IsHardwareAccelerated;
            LocalBuilder orLo = vector ? il.DeclareLocal(typeof(Vector128<uint>)) : null;
            LocalBuilder orHi = vector ? il.DeclareLocal(typeof(Vector128<uint>)) : null;
            // The OR groups the caller reads: units 0-1 (the 21- and 16-bit checks need units 0-2 / 0-3), 2, 3 and 4-7.
            LocalBuilder[] ors = key.CollectOr && !vector
                ? new[] { il.DeclareLocal(typeof(uint)), il.DeclareLocal(typeof(uint)), il.DeclareLocal(typeof(uint)), il.DeclareLocal(typeof(uint)) }
                : null;

            // One loop per addressing (chosen once per call): element index[j] at text + index[j] * stride, or — index null,
            // the first key pass of a line — element j at a running pointer that advances by the stride.
            var sequential = il.DefineLabel();
            var epilogue = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Brfalse, sequential);
            EmitTextSortKeyLoop(il, key, indexed: true, j, p, vector, orLo, orHi, ors);
            il.Emit(OpCodes.Br, epilogue);
            il.MarkLabel(sequential);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, p);
            EmitTextSortKeyLoop(il, key, indexed: false, j, p, vector, orLo, orHi, ors);
            il.MarkLabel(epilogue);

            if (vector)
            {
                il.Emit(OpCodes.Ldloc, orLo);
                il.Emit(OpCodes.Ldloc, orHi);
                il.Emit(OpCodes.Ldarg_S, (byte)5);
                il.Emit(OpCodes.Call, s_storeUnitOr);
            }
            if (ors is not null)
            {
                for (int g = 0; g < 4; g++)
                {
                    il.Emit(OpCodes.Ldarg_S, (byte)5);
                    if (g != 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, 4 * g);
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Ldloc, ors[g]);
                    il.Emit(OpCodes.Stind_I4);
                }
            }
            il.Emit(OpCodes.Ret);
            return (TextSortKeyFunc)dm.CreateDelegate(typeof(TextSortKeyFunc));
        }

        /// <summary>
        ///     Emits one key loop: <c>for (j = 0; j &lt; count; j++) keys[j] = key(element j)</c>, the element addressed
        ///     through <c>index[j]</c> or by the running pointer <paramref name="p"/>.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="key">The kernel identity.</param>
        /// <param name="indexed">True: element <c>index[j]</c> at <c>text + index[j] * stride</c>; false: <paramref name="p"/>
        ///     starts at <c>text</c> (set by the caller) and advances by the stride after each element.</param>
        /// <param name="j">The counter local.</param>
        /// <param name="p">The element pointer local.</param>
        /// <param name="vector">Whether the vector body runs (the 8-unit, 8-bit U key).</param>
        /// <param name="orLo">The vector body's OR accumulator for units 0-3, or null.</param>
        /// <param name="orHi">The vector body's OR accumulator for units 4-7, or null.</param>
        /// <param name="ors">The scalar body's four OR groups, or null.</param>
        private static void EmitTextSortKeyLoop(ILGenerator il, TextSortKeyKey key, bool indexed, LocalBuilder j, LocalBuilder p,
                                                bool vector, LocalBuilder orLo, LocalBuilder orHi, LocalBuilder[] ors)
        {
            var top = il.DefineLabel();
            var end = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, j);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, end);

            if (indexed)
            {
                // p = text + index[j] * stride
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, p);
            }

            // &keys[j], then the key, then the store.
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            if (vector)
            {
                il.Emit(OpCodes.Ldloc, p);
                il.Emit(OpCodes.Ldloca, orLo);
                il.Emit(OpCodes.Ldloca, orHi);
                il.Emit(OpCodes.Call, s_unitKey8);
            }
            else
                EmitTextSortKeyValue(il, key, p, ors);
            il.Emit(OpCodes.Stind_I8);

            if (!indexed)
            {
                il.Emit(OpCodes.Ldloc, p);
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, p);
            }
            il.Emit(OpCodes.Ldloc, j);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, j);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);
        }

        /// <summary>
        ///     Pushes the key of the element at <paramref name="p"/> (a ulong), and folds the units read into the OR
        ///     accumulators when the kernel collects them.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="key">The kernel identity.</param>
        /// <param name="p">The element pointer local.</param>
        /// <param name="ors">The four OR accumulators, or null.</param>
        /// <remarks>
        ///     Three shapes: a full 8-byte <c>S</c> prefix is ONE load and a byte swap; two 32-bit <c>U</c> units are one
        ///     8-byte load rotated by 32 (the little-endian load holds unit 1 above unit 0); every other packing ORs each
        ///     unit, zero-extended and shifted to its slot (a 4..7-byte <c>S</c> prefix starts from a swapped 4-byte head).
        /// </remarks>
        private static void EmitTextSortKeyValue(ILGenerator il, TextSortKeyKey key, LocalBuilder p, LocalBuilder[] ors)
        {
            if (key.Unit == 1 && key.Units == 8)
            {
                il.Emit(OpCodes.Ldloc, p);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Call, s_bswap64);
                return;
            }
            if (key.Unit == 4 && key.Bits == 32 && key.Units == 2)
            {
                il.Emit(OpCodes.Ldloc, p);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I4, 32);
                il.Emit(OpCodes.Call, s_rotl64);
                return;
            }

            int first = 0;
            if (key.Unit == 1 && key.Units >= 4)
            {
                // Bytes 0-3 as the key's top 32 bits: one swapped 4-byte load.
                il.Emit(OpCodes.Ldloc, p);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_U4);
                il.Emit(OpCodes.Call, s_bswap32);
                il.Emit(OpCodes.Conv_U8);
                il.Emit(OpCodes.Ldc_I4, 32);
                il.Emit(OpCodes.Shl);
                first = 4;
            }
            else
                il.Emit(OpCodes.Ldc_I8, 0L);

            for (int t = first; t < key.Units; t++)
            {
                // The unit, zero-extended (unsigned order), kept as uint for the OR groups.
                il.Emit(OpCodes.Ldloc, p);
                if (t != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, t * key.Unit);
                    il.Emit(OpCodes.Add);
                }
                if (key.Unit == 4)
                {
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Ldind_U4);
                }
                else
                    il.Emit(OpCodes.Ldind_U1);

                if (ors is not null)
                {
                    // or[group] |= u, leaving u on the stack.
                    var group = ors[t < 2 ? 0 : t == 2 ? 1 : t == 3 ? 2 : 3];
                    il.Emit(OpCodes.Dup);
                    il.Emit(OpCodes.Ldloc, group);
                    il.Emit(OpCodes.Or);
                    il.Emit(OpCodes.Stloc, group);
                }

                il.Emit(OpCodes.Conv_U8);
                int shift = 64 - key.Bits * (t + 1);
                if (shift != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, shift);
                    il.Emit(OpCodes.Shl);
                }
                il.Emit(OpCodes.Or);
            }
        }

        /// <summary>
        ///     Emits a gather kernel: one loop for an index array and one for the identity (chosen once per call), each element
        ///     copied by the chunk plan of its width (straight-line chunk moves; the last chunk may overlap the previous one,
        ///     which rewrites the same bytes).
        /// </summary>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>The compiled kernel.</returns>
        private static TextGatherFunc GenerateTextGatherKernel(int width)
        {
            var dm = new DynamicMethod($"NDTextGather_w{width}", typeof(void),
                new[] { typeof(byte*), typeof(long), typeof(long*), typeof(long), typeof(byte*), typeof(long) },
                typeof(ILKernelGenerator), skipVisibility: true);
            var il = dm.GetILGenerator();
            var plan = PlanTextRange(0, width, 1);
            var lane = typeof(byte);
            var j = il.DeclareLocal(typeof(long));
            var from = il.DeclareLocal(typeof(byte*));
            var to = il.DeclareLocal(typeof(byte*));
            var chunk = plan.Count > 0 ? TextChunkLocal(il, plan.Kind, lane) : null;

            void EmitCopy()
            {
                ForEachTextChunk(il, plan, site =>
                {
                    EmitTextChunkLoad(il, from, site, plan.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunk);
                    EmitTextChunkStore(il, to, null, 0, site, plan.Kind, lane, chunk);
                });
            }

            var identity = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Brfalse, identity);

            // Indexed: from = src + index[j] * srcStride, to = dst + j * dstStride.
            {
                var top = il.DefineLabel();
                var end = il.DefineLabel();
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, j);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Stloc, to);
                il.MarkLabel(top);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Bge, end);

                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, from);
                EmitCopy();

                il.Emit(OpCodes.Ldloc, to);
                il.Emit(OpCodes.Ldarg_S, (byte)5);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, to);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, j);
                il.Emit(OpCodes.Br, top);
                il.MarkLabel(end);
                il.Emit(OpCodes.Ret);
            }

            // Identity: both pointers advance by their strides.
            il.MarkLabel(identity);
            {
                var top = il.DefineLabel();
                var end = il.DefineLabel();
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, j);
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Stloc, from);
                il.Emit(OpCodes.Ldarg_S, (byte)4);
                il.Emit(OpCodes.Stloc, to);
                il.MarkLabel(top);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Bge, end);
                EmitCopy();
                il.Emit(OpCodes.Ldloc, from);
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, from);
                il.Emit(OpCodes.Ldloc, to);
                il.Emit(OpCodes.Ldarg_S, (byte)5);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, to);
                il.Emit(OpCodes.Ldloc, j);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, j);
                il.Emit(OpCodes.Br, top);
                il.MarkLabel(end);
                il.Emit(OpCodes.Ret);
            }
            return (TextGatherFunc)dm.CreateDelegate(typeof(TextGatherFunc));
        }
    }
}
