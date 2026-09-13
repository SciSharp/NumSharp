using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The population-count (set-bit count) kernel behind <c>np.bitwise_count</c> — a faithful port of
    ///     NumPy 2.4.2's <c>@TYPE@_bitwise_count</c> loop (numpy/_core/src/umath/loops_autovec.dispatch.c.src,
    ///     <c>*out = npy_popcount@c@(in)</c>): the number of 1-bits in the ABSOLUTE value of each element,
    ///     written as <see cref="byte"/> (NumPy's fixed <c>uint8</c> output for every integer input loop).
    ///     The contiguous whole-array route of <see cref="UnaryOp.BitwiseCount"/> delegates here via a single
    ///     IL <c>call</c> (see <c>DirectILKernelGenerator.EmitBitwiseCountContiguous</c>).
    ///
    ///     <para>
    ///     WHY a bespoke SIMD kernel instead of a per-element IL loop: the operation is
    ///     <b>dtype-agnostic by element WIDTH</b> (the only per-dtype quantities are the byte width and
    ///     signedness — <c>bitwise_count(-3) == bitwise_count(3)</c> because both count <c>|x|</c>) AND it
    ///     <b>narrows</b> every input width down to one output byte, a cross-width shape the generic
    ///     equal-width <c>Vector&lt;T&gt;</c> emitters cannot express. This mirrors <see cref="PackBits"/> /
    ///     <see cref="FiniteScan"/>, the established home for fused, dtype-agnostic byte kernels; the caller
    ///     hands a C-contiguous buffer and this kernel processes the whole array in one call.
    ///     </para>
    ///
    ///     <para>
    ///     SIMD strategy is the AVX2 nibble-lookup (Muła's algorithm), the same <c>Avx2.Shuffle</c> (256-bit
    ///     <c>vpshufb</c>) primitive <see cref="PackBits"/> relies on: a 16-entry per-nibble popcount table
    ///     gives a per-BYTE popcount, then the bytes of each element are horizontally summed to the element's
    ///     popcount — <c>vpmaddubsw</c>/<c>vpmaddwd</c> for 2/4-byte widths — before a saturation-free narrow
    ///     (values are ≤64) packs the results to bytes. Signed inputs run through <c>Vector256.Abs</c> first
    ///     (VPABSB/W/D are one instruction each; the values are then treated as unsigned bit patterns, so
    ///     <c>abs(int.MinValue)</c> keeps its single-bit pattern → popcount 1, matching NumPy's
    ///     <c>a &lt; 0 ? -a : a</c> two's-complement wrap).
    ///     </para>
    ///
    ///     <para>
    ///     <b>8-byte widths are SCALAR on purpose.</b> AVX2 has no 64-bit <c>vpabsq</c>, so
    ///     <c>Vector256.Abs(Vector256&lt;long&gt;)</c> lowers to a compare/xor/subtract sequence that costs
    ///     more than the hardware <c>POPCNT</c> the scalar <see cref="BitOperations.PopCount(ulong)"/>
    ///     compiles to (measured: scalar 4×-unrolled ~5 ms vs the nibble path 7–15 ms at 10M int64; both
    ///     memory-bandwidth-bound and at parity with NumPy). Every width also has a scalar fallback for
    ///     hosts without AVX2.
    ///     </para>
    /// </summary>
    internal static unsafe class BitwiseCountKernel
    {
        /// <summary>Per-nibble popcount table, duplicated across both 128-bit lanes so <see cref="Avx2.Shuffle"/>
        /// (a per-128-lane <c>vpshufb</c>) yields the popcount of each byte's low nibble in-lane.</summary>
        private static readonly Vector256<byte> NibbleLut = Vector256.Create(
            (byte)0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                   0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

        /// <summary>Mask isolating the low nibble of every byte.</summary>
        private static readonly Vector256<byte> LowNibble = Vector256.Create((byte)0x0F);

        /// <summary>All-ones <c>sbyte</c> multiplier: <c>vpmaddubsw(counts, ONE8)</c> sums adjacent byte pairs into 16-bit lanes.</summary>
        private static readonly Vector256<sbyte> One8 = Vector256.Create((sbyte)1);

        /// <summary>All-ones <c>short</c> multiplier: <c>vpmaddwd(pairs, ONE16)</c> sums adjacent 16-bit pairs into 32-bit lanes.</summary>
        private static readonly Vector256<short> One16 = Vector256.Create((short)1);

        /// <summary>
        ///     Writes, for each of <paramref name="n"/> contiguous <paramref name="elemSize"/>-byte elements at
        ///     <paramref name="src"/>, the popcount of its absolute value to the corresponding byte at
        ///     <paramref name="dst"/>. The buffers must not overlap and <paramref name="dst"/> holds <paramref name="n"/> bytes.
        /// </summary>
        /// <param name="src">Base pointer of the C-contiguous input (already offset to logical element 0 by the caller).</param>
        /// <param name="dst">Base pointer of the C-contiguous <c>uint8</c> output (<paramref name="n"/> bytes).</param>
        /// <param name="n">Element count.</param>
        /// <param name="elemSize">Input element width in bytes: 1, 2, 4 or 8.</param>
        /// <param name="signed">True to count <c>|x|</c> for a signed dtype (apply abs first); false for unsigned/bool/char.</param>
        /// <exception cref="NotSupportedException"><paramref name="elemSize"/> is not 1, 2, 4 or 8.</exception>
        public static void Count(byte* src, byte* dst, long n, int elemSize, bool signed)
        {
            switch (elemSize)
            {
                case 1: W1(src, dst, n, signed); break;
                case 2: W2((ushort*)src, dst, n, signed); break;
                case 4: W4((uint*)src, dst, n, signed); break;
                case 8: W8((ulong*)src, dst, n, signed); break;
                default:
                    // Every NumSharp integer/bool/char dtype is 1/2/4/8 bytes; the caller validates the dtype.
                    throw new NotSupportedException($"bitwise_count element size {elemSize} is not supported.");
            }
        }

        /// <summary>Per-byte popcount of a 32-byte vector via the nibble lookup (two <c>vpshufb</c> + one add).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> ByteCounts(Vector256<byte> v)
        {
            var lo = v & LowNibble;
            var hi = Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte() & LowNibble;
            return Avx2.Shuffle(NibbleLut, lo) + Avx2.Shuffle(NibbleLut, hi);
        }

        /// <summary>1-byte width: 32 elements per SIMD iteration (per-byte popcount is the answer).</summary>
        private static void W1(byte* sp, byte* op, long n, bool signed)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 32 <= n; i += 32)
                {
                    var v = Vector256.Load(sp + i);
                    if (signed) v = Vector256.Abs(v.AsSByte()).AsByte();   // VPABSB
                    Avx.Store(op + i, ByteCounts(v));
                }
            }
            for (; i < n; i++)
            {
                int a = signed ? (sbyte)sp[i] : sp[i];
                op[i] = (byte)BitOperations.PopCount((uint)(a < 0 ? -a : a));
            }
        }

        /// <summary>2-byte width: 32 elements per SIMD iteration; <c>vpmaddubsw</c> folds each element's two bytes.</summary>
        private static void W2(ushort* sp, byte* op, long n, bool signed)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 32 <= n; i += 32)
                {
                    var v0 = Vector256.Load(sp + i);
                    var v1 = Vector256.Load(sp + i + 16);
                    if (signed) { v0 = Vector256.Abs(v0.AsInt16()).AsUInt16(); v1 = Vector256.Abs(v1.AsInt16()).AsUInt16(); }  // VPABSW
                    var p0 = Avx2.MultiplyAddAdjacent(ByteCounts(v0.AsByte()), One8);  // 16 shorts = popcount per element
                    var p1 = Avx2.MultiplyAddAdjacent(ByteCounts(v1.AsByte()), One8);
                    Avx.Store(op + i, Vector256.Narrow(p0, p1).AsByte());
                }
            }
            for (; i < n; i++)
            {
                int a = signed ? (short)sp[i] : sp[i];
                op[i] = (byte)BitOperations.PopCount((uint)(a < 0 ? -a : a));
            }
        }

        /// <summary>4-byte width: 32 elements per SIMD iteration; <c>vpmaddubsw</c> then <c>vpmaddwd</c> fold each element's four bytes.</summary>
        private static void W4(uint* sp, byte* op, long n, bool signed)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 32 <= n; i += 32)
                {
                    var q0 = Quad(sp + i, signed);
                    var q1 = Quad(sp + i + 8, signed);
                    var q2 = Quad(sp + i + 16, signed);
                    var q3 = Quad(sp + i + 24, signed);
                    var s0 = Vector256.Narrow(q0, q1);
                    var s1 = Vector256.Narrow(q2, q3);
                    Avx.Store(op + i, Vector256.Narrow(s0, s1).AsByte());
                }
            }
            for (; i < n; i++)
            {
                long a = signed ? (int)sp[i] : sp[i];
                op[i] = (byte)BitOperations.PopCount((ulong)(a < 0 ? -a : a));
            }
        }

        /// <summary>Popcount of eight 4-byte elements → <c>Vector256&lt;int&gt;</c> (one lane per element).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> Quad(uint* p, bool signed)
        {
            var v = Vector256.Load(p);
            if (signed) v = Vector256.Abs(v.AsInt32()).AsUInt32();   // VPABSD
            var pair = Avx2.MultiplyAddAdjacent(ByteCounts(v.AsByte()), One8);
            return Avx2.MultiplyAddAdjacent(pair, One16);
        }

        /// <summary>
        ///     8-byte width: SCALAR hardware POPCNT (see class remarks — no AVX2 VPABSQ, so
        ///     <see cref="BitOperations.PopCount(ulong)"/> beats the nibble path here). 4×-unrolled with
        ///     independent lanes so the CPU dispatches four POPCNTs per iteration.
        /// </summary>
        private static void W8(ulong* sp, byte* op, long n, bool signed)
        {
            long i = 0;
            if (signed)
            {
                var lp = (long*)sp;
                for (; i + 4 <= n; i += 4)
                {
                    long a0 = lp[i], a1 = lp[i + 1], a2 = lp[i + 2], a3 = lp[i + 3];
                    op[i]     = (byte)BitOperations.PopCount((ulong)(a0 < 0 ? -a0 : a0));
                    op[i + 1] = (byte)BitOperations.PopCount((ulong)(a1 < 0 ? -a1 : a1));
                    op[i + 2] = (byte)BitOperations.PopCount((ulong)(a2 < 0 ? -a2 : a2));
                    op[i + 3] = (byte)BitOperations.PopCount((ulong)(a3 < 0 ? -a3 : a3));
                }
                for (; i < n; i++) { long a = lp[i]; op[i] = (byte)BitOperations.PopCount((ulong)(a < 0 ? -a : a)); }
            }
            else
            {
                for (; i + 4 <= n; i += 4)
                {
                    op[i]     = (byte)BitOperations.PopCount(sp[i]);
                    op[i + 1] = (byte)BitOperations.PopCount(sp[i + 1]);
                    op[i + 2] = (byte)BitOperations.PopCount(sp[i + 2]);
                    op[i + 3] = (byte)BitOperations.PopCount(sp[i + 3]);
                }
                for (; i < n; i++) op[i] = (byte)BitOperations.PopCount(sp[i]);
            }
        }
    }
}
