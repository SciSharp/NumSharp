using System;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Text.Strip.cs — the rstrip comparison body (numpy.char's comparisons, compare_chararrays)
// =============================================================================
//
// NUMPY'S SEMANTICS (string_ufuncs.cpp string_comparison_loop<rstrip=true>, string_buffer.h Buffer::rstrip / strcmp)
//   Each element first loses its trailing run of "strip units": NUL and the six ASCII whitespace characters NumPy's
//   NumPyOS_ascii_isspace accepts (' ', '\t', '\n', '\v', '\f', '\r'), for S and U alike — so '\x1c', '\x85', '\xa0' and
//   U+3000 stay. The stripped strings then compare as the plain loop compares any two strings: the common prefix unit by
//   unit (unsigned), and when that is equal, a nonzero unit in the longer one's remaining tail makes it the greater.
//   An embedded NUL still counts ('a\0b' > 'a'); a leading space stays (' a' != 'a').
//
// THE BIT-MASK BODY (every element of at most 64 code units, TextStripMaxLanes)
//   A stripped element is the element with its stripped tail zeroed, and a zeroed tail is invisible to the zero-padded
//   compare. So with K = the kept length in units (one past the last unit that is not a strip unit):
//       strip bits  S = one bit per unit that is a strip unit          (Equals 0 | Equals ' ' | (unit - '\t') <=u 4)
//       kept length K = 64 - lzcnt(~S & fullMask)                      (0 when every unit strips)
//       equal bits  E = one bit per common-prefix unit where A == B
//       d = tzcnt(~E)                                                   (the first differing unit; >= the common width
//                                                                        when none, since E has no bit beyond it)
//   the answer is: if d < min(KA, KB) the units at d decide (unsigned), else KA versus KB (the longer kept string has a
//   nonzero last unit — a strip unit includes NUL — so it is the greater). Every chunk of the element plan feeds the three
//   masks from ONE load per operand; a broadcast operand's kept length and chunks are hoisted out of the loop.
//
// THE WIDE BODY (more than 64 code units)
//   Two managed helpers per element: TextKeptBytes (a backward vector scan for the last kept unit) and TextCompareKept
//   (CommonPrefixLength over the kept bytes); a broadcast operand's kept length is computed once per call.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        ///     The most code units an element may have for the bit-mask rstrip body: one bit per unit in a <c>ulong</c>.
        ///     Wider elements (an <c>S</c> past 64 bytes, a <c>U</c> past 64 characters) take the wide body.
        /// </summary>
        private const int TextStripMaxLanes = 64;

        // BitOperations.LeadingZeroCount(ulong) — the last kept unit of a strip-bit mask — is s_lzcnt64, shared with the
        // concatenation kernels (ILKernelGenerator.Text.Concat.cs).

        /// <summary>The wide body's per-element kept-length helper (<see cref="TextKeptBytes"/>).</summary>
        private static readonly MethodInfo s_textKeptBytes = typeof(ILKernelGenerator).GetMethod(nameof(TextKeptBytes), BindingFlags.NonPublic | BindingFlags.Static)
                                                             ?? throw new MissingMethodException(nameof(ILKernelGenerator), nameof(TextKeptBytes));

        /// <summary>The wide body's per-element compare helper (<see cref="TextCompareKept"/>).</summary>
        private static readonly MethodInfo s_textCompareKept = typeof(ILKernelGenerator).GetMethod(nameof(TextCompareKept), BindingFlags.NonPublic | BindingFlags.Static)
                                                               ?? throw new MissingMethodException(nameof(ILKernelGenerator), nameof(TextCompareKept));

        /// <summary>
        ///     Emits the rstrip comparison element loop over a frame whose pointers are (A, B, out) — or (A, B, out, mask) for
        ///     the masked form: the bit-mask body for elements of at most <see cref="TextStripMaxLanes"/> units, the wide body
        ///     beyond (see the file header). Called by <see cref="EmitTextCompareLoop"/> for a key whose
        ///     <see cref="TextCompareKey.Rstrip"/> is set, so every form (1-D, broadcast stages, rows, masked) reaches it.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="key">The kernel identity (its <c>Rstrip</c> is true).</param>
        /// <param name="frame">The frame (pointers and strides of A, B, out, and the mask when <paramref name="mask"/> is
        ///     given).</param>
        /// <param name="hoistA">A is broadcast in this loop: its kept length (and, for the bit-mask body, its unrolled
        ///     chunks) are computed once before the loop, its pointer never advances.</param>
        /// <param name="hoistB">B is broadcast in this loop.</param>
        /// <param name="mask">The masked form's mask pointer (the frame's fourth pointer), or null for the plain store.</param>
        /// <remarks>
        ///     Inside a rows kernel the hoisting runs once per ROW (the loop is emitted inside the row loop), which is what a
        ///     column-broadcast operand needs: its element changes from row to row.
        /// </remarks>
        private static void EmitTextStripCompareLoop(ILGenerator il, TextCompareKey key, TextFrame frame, bool hoistA, bool hoistB,
                                                     LocalBuilder mask)
        {
            int unit = key.Unit;
            if (Math.Max(key.WidthA, key.WidthB) / unit > TextStripMaxLanes)
            {
                EmitTextStripWideLoop(il, key, frame, hoistA, hoistB, mask);
                return;
            }

            int wA = key.WidthA, wB = key.WidthB;
            int nA = wA / unit, nB = wB / unit;
            var prefix = PlanTextRange(0, Math.Min(wA, wB), unit);
            var tail = PlanTextRange(Math.Min(wA, wB), Math.Abs(wA - wB), unit);
            // Unit lanes: one mask bit per code unit (a byte lane for S, a uint lane for U).
            var lane = unit == 4 ? typeof(uint) : typeof(byte);
            bool tailIsA = wA > wB;

            var a = new TextStripOperand(frame.Pointers[0]);
            var b = new TextStripOperand(frame.Pointers[1]);
            var prefixType = TextStripVectorType(prefix.Kind, lane);
            var tailType = TextStripVectorType(tail.Kind, lane);
            // Scratch vectors: the two prefix chunks, the subtracted copy the strip test reads twice, the tail chunk.
            var ca = il.DeclareLocal(prefixType);
            var cb = il.DeclareLocal(prefixType);
            var xPrefix = il.DeclareLocal(prefixType);
            LocalBuilder ct = tail.Count > 0 ? il.DeclareLocal(tailType) : null;
            LocalBuilder xTail = tail.Count > 0 ? il.DeclareLocal(tailType) : null;

            if (hoistA)
                a.Hoist(il, prefix, tailIsA ? tail : default, nA, unit, lane, xPrefix, xTail);
            if (hoistB)
                b.Hoist(il, prefix, tailIsA ? default : tail, nB, unit, lane, xPrefix, xTail);

            var eq = il.DeclareLocal(typeof(ulong));
            LocalBuilder sA = hoistA ? null : il.DeclareLocal(typeof(ulong));
            LocalBuilder sB = hoistB ? null : il.DeclareLocal(typeof(ulong));
            var keptA = hoistA ? a.Kept : il.DeclareLocal(typeof(int));
            var keptB = hoistB ? b.Kept : il.DeclareLocal(typeof(int));
            var d = il.DeclareLocal(typeof(int));
            var offset = il.DeclareLocal(typeof(int));
            var result = il.DeclareLocal(typeof(int));
            LocalBuilder old = mask is null ? null : il.DeclareLocal(typeof(int));

            var advance = new bool[frame.Pointers.Length];
            advance[0] = !hoistA;
            advance[1] = !hoistB;
            for (int k = 2; k < advance.Length; k++)
                advance[k] = true;

            EmitTextElementLoop(il, frame, advance, body: () =>
            {
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, eq);
                if (sA is not null)
                {
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Stloc, sA);
                }
                if (sB is not null)
                {
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Stloc, sB);
                }

                // The common prefix: ONE load per operand feeds the equal bits and both strip masks.
                ForEachTextChunk(il, prefix, site =>
                {
                    a.PushChunk(il, site, prefix.Kind, lane);
                    il.Emit(OpCodes.Stloc, ca);
                    b.PushChunk(il, site, prefix.Kind, lane);
                    il.Emit(OpCodes.Stloc, cb);

                    int bits = TextStripVectorBits(prefix.Kind);
                    il.Emit(OpCodes.Ldloc, ca);
                    il.Emit(OpCodes.Ldloc, cb);
                    il.Emit(OpCodes.Call, VectorMethodCache.Equals(bits, lane));
                    il.Emit(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(bits, lane));
                    il.Emit(OpCodes.Conv_U8);
                    EmitTextLaneBitsPlace(il, prefix.Kind, site, unit);
                    EmitTextOrInto(il, eq);

                    if (sA is not null)
                    {
                        EmitTextStripBits(il, ca, xPrefix, prefix.Kind, lane, site, unit);
                        EmitTextOrInto(il, sA);
                    }
                    if (sB is not null)
                    {
                        EmitTextStripBits(il, cb, xPrefix, prefix.Kind, lane, site, unit);
                        EmitTextOrInto(il, sB);
                    }
                });

                // The longer operand's tail only feeds its strip mask (the prefix already holds every unit both share).
                var tailMask = tailIsA ? sA : sB;
                var tailOwner = tailIsA ? a : b;
                if (tail.Count > 0 && tailMask is not null)
                {
                    ForEachTextChunk(il, tail, site =>
                    {
                        tailOwner.PushChunkFromMemory(il, site, tail.Kind, lane);
                        il.Emit(OpCodes.Stloc, ct);
                        EmitTextStripBits(il, ct, xTail, tail.Kind, lane, site, unit);
                        EmitTextOrInto(il, tailMask);
                    });
                }

                if (sA is not null)
                {
                    EmitTextKeptLanes(il, sA, nA);
                    il.Emit(OpCodes.Stloc, keptA);
                }
                if (sB is not null)
                {
                    EmitTextKeptLanes(il, sB, nB);
                    il.Emit(OpCodes.Stloc, keptB);
                }

                // d = the first unit where the elements differ (>= the common width when none does).
                il.Emit(OpCodes.Ldloc, eq);
                il.Emit(OpCodes.Not);
                il.Emit(OpCodes.Call, s_tzcnt64);
                il.Emit(OpCodes.Stloc, d);

                if (key.Op is TextCompareOp.Equal or TextCompareOp.NotEqual)
                {
                    // equal = KA == KB && d >= KA: the kept strings have one length and no difference inside it.
                    il.Emit(OpCodes.Ldloc, keptA);
                    il.Emit(OpCodes.Ldloc, keptB);
                    il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.Ldloc, d);
                    il.Emit(OpCodes.Ldloc, keptA);
                    il.Emit(OpCodes.Clt);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.And);
                    if (key.Op == TextCompareOp.NotEqual)
                    {
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                    }
                    il.Emit(OpCodes.Stloc, result);
                }
                else
                {
                    var noDiff = il.DefineLabel();
                    var done = il.DefineLabel();
                    // A difference counts only inside BOTH kept strings; beyond the shorter one the kept lengths decide.
                    il.Emit(OpCodes.Ldloc, d);
                    il.Emit(OpCodes.Ldloc, keptA);
                    il.Emit(OpCodes.Bge, noDiff);
                    il.Emit(OpCodes.Ldloc, d);
                    il.Emit(OpCodes.Ldloc, keptB);
                    il.Emit(OpCodes.Bge, noDiff);
                    // The units at d differ (unsigned order decides, as in the plain loop).
                    il.Emit(OpCodes.Ldloc, d);
                    if (unit == 4)
                    {
                        il.Emit(OpCodes.Ldc_I4_2);
                        il.Emit(OpCodes.Shl);
                    }
                    il.Emit(OpCodes.Stloc, offset);
                    EmitTextUnitLoad(il, a.Pointer, offset, unit);
                    EmitTextUnitLoad(il, b.Pointer, offset, unit);
                    il.Emit(OpCodes.Clt_Un);
                    il.Emit(OpCodes.Stloc, result);
                    il.Emit(OpCodes.Br, done);

                    // Equal kept prefixes: the longer kept string is the greater (its last kept unit is nonzero).
                    il.MarkLabel(noDiff);
                    il.Emit(OpCodes.Ldloc, keptA);
                    il.Emit(OpCodes.Ldloc, keptB);
                    if (key.Op == TextCompareOp.Less)
                        il.Emit(OpCodes.Clt);
                    else
                    {
                        il.Emit(OpCodes.Cgt);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                    }
                    il.Emit(OpCodes.Stloc, result);
                    il.MarkLabel(done);
                }

                EmitTextResultStore(il, frame, mask, result, old);
            });
        }

        /// <summary>
        ///     Emits the wide rstrip body (elements past <see cref="TextStripMaxLanes"/> units): per element the kept byte
        ///     length of each operand (<see cref="TextKeptBytes"/>; a broadcast operand's once per call) and the compare of
        ///     the kept bytes (<see cref="TextCompareKept"/>), then the op's answer from the -1 / 0 / 1 order.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="key">The kernel identity.</param>
        /// <param name="frame">The frame (A, B, out, and the mask when <paramref name="mask"/> is given).</param>
        /// <param name="hoistA">A is broadcast in this loop (its kept length is computed before the loop).</param>
        /// <param name="hoistB">B is broadcast in this loop.</param>
        /// <param name="mask">The masked form's mask pointer, or null.</param>
        /// <remarks>
        ///     A helper call per element is the cheap part here: a wide element costs NumPy a backward scalar scan of each
        ///     operand and a scalar compare, while <see cref="TextKeptBytes"/> scans 32 bytes per step and
        ///     <see cref="TextCompareKept"/> is the BCL's vectorized <c>CommonPrefixLength</c>.
        /// </remarks>
        private static void EmitTextStripWideLoop(ILGenerator il, TextCompareKey key, TextFrame frame, bool hoistA, bool hoistB,
                                                  LocalBuilder mask)
        {
            var keptA = il.DeclareLocal(typeof(int));
            var keptB = il.DeclareLocal(typeof(int));
            var result = il.DeclareLocal(typeof(int));
            LocalBuilder old = mask is null ? null : il.DeclareLocal(typeof(int));
            if (hoistA)
                EmitTextKeptBytesCall(il, frame.Pointers[0], key.WidthA, key.Unit, keptA);
            if (hoistB)
                EmitTextKeptBytesCall(il, frame.Pointers[1], key.WidthB, key.Unit, keptB);

            var advance = new bool[frame.Pointers.Length];
            advance[0] = !hoistA;
            advance[1] = !hoistB;
            for (int k = 2; k < advance.Length; k++)
                advance[k] = true;

            EmitTextElementLoop(il, frame, advance, body: () =>
            {
                if (!hoistA)
                    EmitTextKeptBytesCall(il, frame.Pointers[0], key.WidthA, key.Unit, keptA);
                if (!hoistB)
                    EmitTextKeptBytesCall(il, frame.Pointers[1], key.WidthB, key.Unit, keptB);
                il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                il.Emit(OpCodes.Ldloc, keptA);
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Ldloc, keptB);
                il.Emit(OpCodes.Ldc_I4, key.Unit);
                il.Emit(OpCodes.Call, s_textCompareKept);
                // The order (-1 / 0 / 1) to the op's bool.
                il.Emit(OpCodes.Ldc_I4_0);
                switch (key.Op)
                {
                    case TextCompareOp.Equal:
                        il.Emit(OpCodes.Ceq);
                        break;
                    case TextCompareOp.NotEqual:
                        il.Emit(OpCodes.Ceq);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                        break;
                    case TextCompareOp.Less:
                        il.Emit(OpCodes.Clt);
                        break;
                    default:
                        il.Emit(OpCodes.Cgt);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                        break;
                }
                il.Emit(OpCodes.Stloc, result);
                EmitTextResultStore(il, frame, mask, result, old);
            });
        }

        /// <summary>Emits <c>kept = TextKeptBytes(pointer, width, unit)</c>.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="width">The element width in bytes (an emit-time constant).</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="kept">The int local receiving the kept byte length.</param>
        private static void EmitTextKeptBytesCall(ILGenerator il, LocalBuilder pointer, int width, int unit, LocalBuilder kept)
        {
            il.Emit(OpCodes.Ldloc, pointer);
            il.Emit(OpCodes.Ldc_I4, width);
            il.Emit(OpCodes.Ldc_I4, unit);
            il.Emit(OpCodes.Call, s_textKeptBytes);
            il.Emit(OpCodes.Stloc, kept);
        }

        /// <summary>
        ///     Emits the element's result store: <c>*out = result</c>, or the masked form's branch-free blend
        ///     (<see cref="EmitTextMaskedStore"/>).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame (its third pointer is the output).</param>
        /// <param name="mask">The mask pointer local, or null for the plain store.</param>
        /// <param name="result">The int local holding 0 or 1.</param>
        /// <param name="old">The masked form's scratch int local (null when <paramref name="mask"/> is null).</param>
        private static void EmitTextResultStore(ILGenerator il, TextFrame frame, LocalBuilder mask, LocalBuilder result, LocalBuilder old)
        {
            if (mask is null)
            {
                il.Emit(OpCodes.Ldloc, frame.Pointers[2]);
                il.Emit(OpCodes.Ldloc, result);
                il.Emit(OpCodes.Stind_I1);
            }
            else
                EmitTextMaskedStore(il, frame.Pointers[2], mask, result, old);
        }

        /// <summary>
        ///     The vector type a chunk is tested as: its own Vector256 / Vector128 of unit lanes, or — for the scalar kinds
        ///     (an element narrower than 16 bytes) — a Vector128 holding the chunk's bytes in its low lanes and zeros above
        ///     (<c>Vector128.CreateScalar(ulong)</c>), whose extra lanes <see cref="EmitTextLaneBitsPlace"/> masks off.
        /// </summary>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The unit lane type (byte or uint).</param>
        /// <returns>The vector type.</returns>
        private static Type TextStripVectorType(TextChunkKind kind, Type lane)
            => VectorMethodCache.V(TextStripVectorBits(kind), lane);

        /// <summary>The vector width (bits) a chunk kind is tested at: 256 for Vector256 chunks, 128 for every other kind.</summary>
        /// <param name="kind">The chunk kind.</param>
        /// <returns>256 or 128.</returns>
        private static int TextStripVectorBits(TextChunkKind kind) => kind == TextChunkKind.Vector256 ? 256 : 128;

        /// <summary>
        ///     Pushes one chunk of the element at <paramref name="pointer"/> as its test vector
        ///     (<see cref="TextStripVectorType"/>): a vector kind loads directly; a scalar kind loads its bytes zero-extended
        ///     to a ulong and moves them into the low lanes of a zeroed Vector128.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="site">The chunk site.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The unit lane type.</param>
        private static void EmitTextStripChunkLoad(ILGenerator il, LocalBuilder pointer, TextChunkSite site, TextChunkKind kind, Type lane)
        {
            EmitTextChunkLoad(il, pointer, site, kind, lane);
            if (IsVectorChunk(kind))
                return;
            il.Emit(OpCodes.Call, VectorMethodCache.CreateScalar(128, typeof(ulong)));
            il.Emit(OpCodes.Call, VectorMethodCache.As(128, typeof(ulong), lane));
        }

        /// <summary>
        ///     Emits the strip test of the chunk vector in <paramref name="v"/> and leaves its bits, placed at the chunk's
        ///     units (<see cref="EmitTextLaneBitsPlace"/>), as a ulong: bit i set when unit i is NUL, a space, or one of
        ///     <c>'\t'</c> .. <c>'\r'</c> — NumPy's <c>*tmp == '\0' || NumPyOS_ascii_isspace(*tmp)</c>, as three lane tests:
        ///     <c>v == 0</c>, <c>v == ' '</c>, and <c>(v - '\t') &lt;=u 4</c> (an unsigned min-equals: <c>min(x, 4) == x</c>,
        ///     one instruction pair for byte and uint lanes on AVX2).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="v">The chunk vector local.</param>
        /// <param name="x">A scratch local of the same vector type (the subtracted copy, read twice).</param>
        /// <param name="kind">The chunk kind (decides the vector width and whether the high lanes are padding).</param>
        /// <param name="lane">The unit lane type.</param>
        /// <param name="site">The chunk site (its unit offset is the shift).</param>
        /// <param name="unit">1 or 4.</param>
        private static void EmitTextStripBits(ILGenerator il, LocalBuilder v, LocalBuilder x, TextChunkKind kind, Type lane,
                                              TextChunkSite site, int unit)
        {
            int bits = TextStripVectorBits(kind);
            var or = VectorMethodCache.Operator(bits, lane, "op_BitwiseOr");
            var equals = VectorMethodCache.Equals(bits, lane);
            var broadcast = VectorMethodCache.CreateBroadcast(bits, lane);

            // v == 0
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Call, VectorMethodCache.Zero(bits, lane));
            il.Emit(OpCodes.Call, equals);
            // | v == ' '
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Ldc_I4, 0x20);
            il.Emit(OpCodes.Call, broadcast);
            il.Emit(OpCodes.Call, equals);
            il.Emit(OpCodes.Call, or);
            // | (v - '\t') <=u 4, as min(x, 4) == x
            il.Emit(OpCodes.Ldloc, v);
            il.Emit(OpCodes.Ldc_I4, 0x09);
            il.Emit(OpCodes.Call, broadcast);
            il.Emit(OpCodes.Call, VectorMethodCache.Operator(bits, lane, "op_Subtraction"));
            il.Emit(OpCodes.Stloc, x);
            il.Emit(OpCodes.Ldloc, x);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Call, broadcast);
            il.Emit(OpCodes.Call, VectorMethodCache.Generic(bits, "Min", lane, 2));
            il.Emit(OpCodes.Ldloc, x);
            il.Emit(OpCodes.Call, equals);
            il.Emit(OpCodes.Call, or);

            il.Emit(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(bits, lane));
            il.Emit(OpCodes.Conv_U8);
            EmitTextLaneBitsPlace(il, kind, site, unit);
        }

        /// <summary>
        ///     Moves a chunk's lane bits (a ulong on the stack, bit i = the chunk's unit i) to the element's unit positions:
        ///     a scalar kind first drops the padding lanes above its own units (a zero lane would claim equal / strip at a
        ///     unit the chunk does not hold), then the bits shift left by the chunk's unit offset (constant, or the loop
        ///     form's offset local divided by the unit size).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="site">The chunk site.</param>
        /// <param name="unit">1 or 4.</param>
        private static void EmitTextLaneBitsPlace(ILGenerator il, TextChunkKind kind, TextChunkSite site, int unit)
        {
            if (!IsVectorChunk(kind))
            {
                int lanes = TextChunkBytes(kind) / unit;
                il.Emit(OpCodes.Ldc_I8, (long)((1UL << lanes) - 1));
                il.Emit(OpCodes.And);
            }
            if (site.VarOffset is not null)
            {
                il.Emit(OpCodes.Ldloc, site.VarOffset);
                if (unit == 4)
                {
                    il.Emit(OpCodes.Ldc_I4_2);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Shl);
            }
            else if (site.ConstOffset / unit != 0)
            {
                il.Emit(OpCodes.Ldc_I4, site.ConstOffset / unit);
                il.Emit(OpCodes.Shl);
            }
        }

        /// <summary>Emits <c>target |= (the ulong on the stack)</c>.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="target">The ulong accumulator local.</param>
        private static void EmitTextOrInto(ILGenerator il, LocalBuilder target)
        {
            il.Emit(OpCodes.Ldloc, target);
            il.Emit(OpCodes.Or);
            il.Emit(OpCodes.Stloc, target);
        }

        /// <summary>
        ///     Pushes the kept length in units of an element whose strip bits are in <paramref name="strip"/>:
        ///     <c>64 - lzcnt(~strip &amp; fullMask)</c>, one past the last unit that is not a strip unit (0 when every unit
        ///     strips; <c>lzcnt(0)</c> is 64).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="strip">The ulong local holding the element's strip bits.</param>
        /// <param name="lanes">The element's width in units (at most 64): the mask that keeps <c>~strip</c> inside the element.</param>
        private static void EmitTextKeptLanes(ILGenerator il, LocalBuilder strip, int lanes)
        {
            ulong full = lanes >= 64 ? ulong.MaxValue : (1UL << lanes) - 1;
            il.Emit(OpCodes.Ldc_I4, 64);
            il.Emit(OpCodes.Ldloc, strip);
            il.Emit(OpCodes.Not);
            il.Emit(OpCodes.Ldc_I8, (long)full);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Call, s_lzcnt64);
            il.Emit(OpCodes.Sub);
        }

        /// <summary>
        ///     One operand of the bit-mask rstrip body: its pointer local and, when it is broadcast, its kept length and its
        ///     unrolled prefix chunks (as test vectors), computed once before the loop.
        /// </summary>
        private sealed class TextStripOperand
        {
            /// <summary>The element pointer local.</summary>
            internal readonly LocalBuilder Pointer;

            /// <summary>The hoisted prefix chunk vectors by unrolled chunk index, or null (not broadcast, or a loop-form plan).</summary>
            private LocalBuilder[] _chunks;

            /// <summary>The hoisted kept length in units (an int local), or null when the operand is not broadcast.</summary>
            internal LocalBuilder Kept { get; private set; }

            /// <summary>Wraps an operand pointer.</summary>
            /// <param name="pointer">The pointer local.</param>
            internal TextStripOperand(LocalBuilder pointer) => Pointer = pointer;

            /// <summary>
            ///     Emits the hoisting of a broadcast operand (before the element loop): its strip bits over the whole
            ///     element (prefix chunks, then its tail when it owns one) into a kept length, keeping each unrolled prefix
            ///     chunk vector in a local for the equal bits.
            /// </summary>
            /// <param name="il">The generator.</param>
            /// <param name="prefix">The common-prefix plan.</param>
            /// <param name="tail">This operand's tail plan, or a default (empty) plan when the other operand owns it.</param>
            /// <param name="lanes">The operand's width in units.</param>
            /// <param name="unit">1 or 4.</param>
            /// <param name="lane">The unit lane type.</param>
            /// <param name="xPrefix">A scratch vector local of the prefix chunks' test type.</param>
            /// <param name="xTail">A scratch vector local of the tail chunks' test type (null when there is no tail).</param>
            internal void Hoist(ILGenerator il, TextChunkPlan prefix, TextChunkPlan tail, int lanes, int unit, Type lane,
                                LocalBuilder xPrefix, LocalBuilder xTail)
            {
                var strip = il.DeclareLocal(typeof(ulong));
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, strip);
                if (prefix.Count > 0 && prefix.Unrolled)
                    _chunks = new LocalBuilder[prefix.Count];
                var prefixType = TextStripVectorType(prefix.Kind, lane);
                var chunk = il.DeclareLocal(prefixType);
                ForEachTextChunk(il, prefix, site =>
                {
                    EmitTextStripChunkLoad(il, Pointer, site, prefix.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunk);
                    if (_chunks is not null && site.Index >= 0)
                    {
                        _chunks[site.Index] = il.DeclareLocal(prefixType);
                        il.Emit(OpCodes.Ldloc, chunk);
                        il.Emit(OpCodes.Stloc, _chunks[site.Index]);
                    }
                    EmitTextStripBits(il, chunk, xPrefix, prefix.Kind, lane, site, unit);
                    EmitTextOrInto(il, strip);
                });
                if (tail.Count > 0)
                {
                    var tailChunk = il.DeclareLocal(TextStripVectorType(tail.Kind, lane));
                    ForEachTextChunk(il, tail, site =>
                    {
                        EmitTextStripChunkLoad(il, Pointer, site, tail.Kind, lane);
                        il.Emit(OpCodes.Stloc, tailChunk);
                        EmitTextStripBits(il, tailChunk, xTail, tail.Kind, lane, site, unit);
                        EmitTextOrInto(il, strip);
                    });
                }
                Kept = il.DeclareLocal(typeof(int));
                EmitTextKeptLanes(il, strip, lanes);
                il.Emit(OpCodes.Stloc, Kept);
            }

            /// <summary>Pushes a prefix chunk's test vector: its hoisted local when there is one for the site, else a fresh read.</summary>
            /// <param name="il">The generator.</param>
            /// <param name="site">The chunk site.</param>
            /// <param name="kind">The prefix chunk kind.</param>
            /// <param name="lane">The unit lane type.</param>
            internal void PushChunk(ILGenerator il, TextChunkSite site, TextChunkKind kind, Type lane)
            {
                if (_chunks is not null && site.Index >= 0)
                    il.Emit(OpCodes.Ldloc, _chunks[site.Index]);
                else
                    EmitTextStripChunkLoad(il, Pointer, site, kind, lane);
            }

            /// <summary>Pushes a chunk's test vector read from the element (the tail, which is never hoisted per chunk).</summary>
            /// <param name="il">The generator.</param>
            /// <param name="site">The chunk site.</param>
            /// <param name="kind">The chunk kind.</param>
            /// <param name="lane">The unit lane type.</param>
            internal void PushChunkFromMemory(ILGenerator il, TextChunkSite site, TextChunkKind kind, Type lane)
                => EmitTextStripChunkLoad(il, Pointer, site, kind, lane);
        }

        // =============================================================================================================
        // The managed helpers (the wide body, and NDStringComparison's managed kernel on hosts without dynamic code)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>Buffer::rstrip</c> as a length: the byte length of the element at <paramref name="p"/> once its
        ///     trailing NULs and ASCII whitespace (<c>' '</c>, <c>'\t'</c>, <c>'\n'</c>, <c>'\v'</c>, <c>'\f'</c>, <c>'\r'</c>)
        ///     are removed, a whole number of units.
        /// </summary>
        /// <param name="p">The element's first byte (any alignment).</param>
        /// <param name="width">The element width in bytes (a multiple of <paramref name="unit"/>).</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>The kept length in bytes (0 when every unit strips).</returns>
        /// <remarks>
        ///     Scans from the end 32 bytes at a time (one strip test per block; the first block holding a kept unit decides),
        ///     then unit by unit — a wide element padded with NULs (a short string in a wide <c>U</c>) costs a few vector
        ///     steps instead of one step per unit.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe int TextKeptBytes(byte* p, int width, int unit)
        {
            int n = width;
            if (unit == 1)
            {
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; n >= 32; n -= 32)
                    {
                        // keep = the byte lanes that are not strip units; the highest one ends the kept string.
                        uint keep = ~TextStripLanes(Vector256.Load(p + n - 32));
                        if (keep != 0)
                            return n - BitOperations.LeadingZeroCount(keep);
                    }
                }
                for (; n > 0; n--)
                    if (!IsTextStripUnit(p[n - 1]))
                        break;
                return n;
            }
            if (Vector256.IsHardwareAccelerated)
            {
                for (; n >= 32; n -= 32)
                {
                    // Eight uint lanes: keep holds 8 bits, so lzcnt (of the 32-bit value) is 24 + the unit lanes above
                    // the last kept one; each lane is 4 bytes.
                    uint keep = ~TextStripLanes(Vector256.Load((uint*)(p + n - 32))) & 0xFFu;
                    if (keep != 0)
                        return n + 96 - 4 * BitOperations.LeadingZeroCount(keep);
                }
            }
            for (; n > 0; n -= 4)
                if (!IsTextStripUnit(Unsafe.ReadUnaligned<uint>(p + n - 4)))
                    break;
            return n;
        }

        /// <summary>
        ///     NumPy's <c>Buffer::strcmp</c> of two already-stripped elements: the first differing unit of the common kept
        ///     prefix decides (unsigned), else the longer kept string is the greater (its last kept unit is nonzero).
        /// </summary>
        /// <param name="a">The first element.</param>
        /// <param name="keptA">Its kept byte length (<see cref="TextKeptBytes"/>).</param>
        /// <param name="b">The second element.</param>
        /// <param name="keptB">Its kept byte length.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        /// <remarks>
        ///     The prefix is compared as BYTES (the BCL's vectorized <c>CommonPrefixLength</c>); for <c>U</c> the first
        ///     differing byte is rounded down to its unit, whose units then differ while every earlier unit is equal.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe int TextCompareKept(byte* a, int keptA, byte* b, int keptB, int unit)
        {
            int common = Math.Min(keptA, keptB);
            int d = new ReadOnlySpan<byte>(a, common).CommonPrefixLength(new ReadOnlySpan<byte>(b, common));
            if (d < common)
            {
                if (unit == 4)
                {
                    d &= ~3;
                    uint ua = Unsafe.ReadUnaligned<uint>(a + d), ub = Unsafe.ReadUnaligned<uint>(b + d);
                    return ua < ub ? -1 : 1;
                }
                return a[d] < b[d] ? -1 : 1;
            }
            return keptA == keptB ? 0 : keptA < keptB ? -1 : 1;
        }

        /// <summary>Whether a code unit is one NumPy's rstrip removes: NUL, a space, or <c>'\t'</c> .. <c>'\r'</c>.</summary>
        /// <param name="c">The unit (a byte of <c>S</c> or a UCS-4 unit of <c>U</c>, as an unsigned value).</param>
        /// <returns>True for the seven strip units.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsTextStripUnit(uint c) => c == 0 || c == 0x20 || c - 0x09 <= 4;

        /// <summary>The strip-unit lanes of 32 bytes: bit i set when byte i is NUL, a space or <c>'\t'</c> .. <c>'\r'</c>.</summary>
        /// <param name="v">The bytes.</param>
        /// <returns>The 32 lane bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint TextStripLanes(Vector256<byte> v)
        {
            var x = v - Vector256.Create((byte)0x09);
            var s = Vector256.Equals(v, Vector256<byte>.Zero) | Vector256.Equals(v, Vector256.Create((byte)0x20))
                    | Vector256.Equals(Vector256.Min(x, Vector256.Create((byte)4)), x);
            return s.ExtractMostSignificantBits();
        }

        /// <summary>The strip-unit lanes of eight UCS-4 units: bit i set when unit i is NUL, a space or <c>'\t'</c> .. <c>'\r'</c>.</summary>
        /// <param name="v">The units.</param>
        /// <returns>The 8 lane bits (in the low byte).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint TextStripLanes(Vector256<uint> v)
        {
            var x = v - Vector256.Create(0x09u);
            var s = Vector256.Equals(v, Vector256<uint>.Zero) | Vector256.Equals(v, Vector256.Create(0x20u))
                    | Vector256.Equals(Vector256.Min(x, Vector256.Create(4u)), x);
            return s.ExtractMostSignificantBits();
        }
    }
}
