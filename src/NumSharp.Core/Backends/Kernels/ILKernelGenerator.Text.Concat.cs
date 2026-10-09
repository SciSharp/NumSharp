using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Text.Concat.cs — the text kernels that WRITE text: concatenation, repetition, and the length
// =============================================================================
//
// NUMPY'S SEMANTICS (string_ufuncs.cpp, probed against 2.4.2)
//   A string's length is num_codepoints: the index of its last NONZERO code unit plus one — trailing NULs never count,
//   embedded ones do ('a\0' has length 1, 'a\0b' length 3; a U unit above U+10FFFF is just a nonzero unit here).
//   * string_add (np.add over (S,S) / (U,U)): out = a[:len(a)] + b[:len(b)], zero-filled to the output width, which the
//     loop's descriptor resolution fixes at width(a) + width(b) — an out= of another width receives the result through
//     the iterator's cast (truncated or zero-padded), never through this loop.
//   * string_multiply (np.multiply over (S,int64) / (int64,S), same for U): a one-unit string is FILLED min(reps, width)
//     times; a longer one is copied reps times, cut at the output width (width / len full copies plus width % len units);
//     reps < 1 or an empty string writes an empty element. reps * len past 2^63 - 1 leaves that element untouched and
//     raises OverflowError after the loop ("Overflow detected in string multiply").
//   * str_len: the length, as int64.
//
// THE KERNELS (NDInnerLoopFunc, element widths are EMIT-TIME constants as in ILKernelGenerator.Text.cs)
//   * concatenation (A, B, out of width wa + wb): copy A's chunks to out[0, wa) — the length of A falls out of the same
//     chunk values (the highest nonzero byte of the last nonzero chunk, rounded up to its unit) —, zero out[wa, wa + wb),
//     then copy B's chunks to out[len(A), len(A) + wb). At <U8 + <U8 that is two loads and three stores per element.
//   * length (text, int64 out): the same chunk scan, one store.
//   * repetition, NARROW form (text ≤ NarrowRepeatMaxWidth bytes, out ≤ NarrowRepeatMaxOutWidth bytes): the text's chunks
//     are loaded once into locals (its length from the same values), then written once per copy at offset k * len into a
//     stack scratch row that was zeroed first, and the row is copied to the output — every store a fixed-size chunk, so
//     the per-element cost does not branch on the element's length. NumPy's loop (one memcpy per copy, then a zero fill)
//     and the general form below both run variable-size copies whose size changes element to element, which the branch
//     predictor cannot follow (measured: 30-40 ns per element for <U8 x 3, against ~7 here). The row is wider than the
//     output (by one text width), so the last copy may overrun it: a cut repetition simply stores past the output width
//     and the copy-out drops it. A one-unit text is NumPy's buffer_memset: a broadcast of that unit over the whole
//     output when reps covers it, else the general helper (a partial fill).
//   * repetition, GENERAL form (wider texts / outputs): one direct call per element to RepeatElement (a C# helper: the
//     element's work is a data-dependent copy loop, doubled copies for many repetitions), with the widths as constants.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>The concatenation kernels, keyed by (unit, widths).</summary>
        private static readonly ConcurrentDictionary<TextConcatKey, NDInnerLoopFunc> s_textConcatKernels = new();

        /// <summary>The length kernels, keyed by (unit, width).</summary>
        private static readonly ConcurrentDictionary<TextLengthKey, NDInnerLoopFunc> s_textLengthKernels = new();

        /// <summary>The repetition kernels, keyed by (unit, widths, text operand slot).</summary>
        private static readonly ConcurrentDictionary<TextRepeatKey, NDInnerLoopFunc> s_textRepeatKernels = new();

        /// <summary><c>BitOperations.LeadingZeroCount(uint)</c>: the highest nonzero lane of a vector chunk's mask.</summary>
        private static readonly MethodInfo s_lzcnt32 = typeof(BitOperations).GetMethod(nameof(BitOperations.LeadingZeroCount), new[] { typeof(uint) })
                                                       ?? throw new MissingMethodException(nameof(BitOperations), "LeadingZeroCount(uint)");

        /// <summary><c>BitOperations.LeadingZeroCount(ulong)</c>: the highest nonzero byte of a scalar chunk.</summary>
        private static readonly MethodInfo s_lzcnt64 = typeof(BitOperations).GetMethod(nameof(BitOperations.LeadingZeroCount), new[] { typeof(ulong) })
                                                       ?? throw new MissingMethodException(nameof(BitOperations), "LeadingZeroCount(ulong)");

        /// <summary><see cref="RepeatElement"/>, the per-element body the repetition kernel calls.</summary>
        private static readonly MethodInfo s_repeatElement = typeof(ILKernelGenerator).GetMethod(nameof(RepeatElement), BindingFlags.NonPublic | BindingFlags.Static)
                                                             ?? throw new MissingMethodException(nameof(ILKernelGenerator), nameof(RepeatElement));

        /// <summary>
        ///     The widest text (bytes) the NARROW repetition kernel takes: at most four Vector256 chunks held in locals
        ///     (the chunk plan stays unrolled). A wider text takes the general kernel.
        /// </summary>
        internal const int NarrowRepeatMaxWidth = 128;

        /// <summary>
        ///     The widest output (bytes) the NARROW repetition kernel takes. Its scratch row is zeroed and copied out per
        ///     element, which costs about two passes over the output in L1; past this width that exceeds the general
        ///     kernel's single pass of doubling copies (whose branch misses no longer dominate a long element).
        /// </summary>
        internal const int NarrowRepeatMaxOutWidth = 512;

        /// <summary>The identity of a concatenation kernel.</summary>
        /// <param name="Unit">The code-unit size: 1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="WidthA">The first operand's element width in bytes.</param>
        /// <param name="WidthB">The second operand's element width in bytes (the output is <c>WidthA + WidthB</c>).</param>
        private readonly record struct TextConcatKey(int Unit, int WidthA, int WidthB);

        /// <summary>The identity of a length kernel.</summary>
        /// <param name="Unit">The code-unit size.</param>
        /// <param name="Width">The element width in bytes.</param>
        private readonly record struct TextLengthKey(int Unit, int Width);

        /// <summary>The identity of a repetition kernel.</summary>
        /// <param name="Unit">The code-unit size.</param>
        /// <param name="Width">The text operand's element width in bytes.</param>
        /// <param name="OutWidth">The output's element width in bytes.</param>
        /// <param name="TextIndex">The operand slot of the text (0 or 1; the int64 repeat count is the other).</param>
        private readonly record struct TextRepeatKey(int Unit, int Width, int OutWidth, int TextIndex);

        /// <summary>
        ///     The concatenation kernel for one (unit, widths): <c>dataptrs[0]</c> = A, <c>dataptrs[1]</c> = B,
        ///     <c>dataptrs[2]</c> = the output, whose elements are <c>widthA + widthB</c> bytes. Any strides (a broadcast
        ///     operand has stride 0); NumPy's <c>string_add</c> per element.
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's element width in bytes.</param>
        /// <param name="widthB">B's element width in bytes.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller takes its managed route).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, or a width that is negative or not a
        ///     whole number of units.</exception>
        internal static NDInnerLoopFunc GetTextConcatKernel(int unit, int widthA, int widthB)
        {
            ValidateTextShape(unit, widthA);
            ValidateTextShape(unit, widthB);
            if (!TextKernelsAvailable)
                return null;
            var key = new TextConcatKey(unit, widthA, widthB);
            if (s_textConcatKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textConcatKernels.GetOrAdd(key, GenerateTextConcatKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                // Emission failed on this runtime: the managed route answers.
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextConcatKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The length kernel for one (unit, width): <c>dataptrs[0]</c> = the text, <c>dataptrs[1]</c> = the int64 output
        ///     (NumPy's <c>str_len</c>: the index of the last nonzero unit plus one).
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, or a width that is negative or not a
        ///     whole number of units.</exception>
        internal static NDInnerLoopFunc GetTextLengthKernel(int unit, int width)
        {
            ValidateTextShape(unit, width);
            if (!TextKernelsAvailable)
                return null;
            var key = new TextLengthKey(unit, width);
            if (s_textLengthKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textLengthKernels.GetOrAdd(key, GenerateTextLengthKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextLengthKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The repetition kernel for one (unit, widths, operand order): <c>dataptrs[textIndex]</c> = the text,
        ///     <c>dataptrs[1 - textIndex]</c> = the int64 repeat counts, <c>dataptrs[2]</c> = the output. When an element
        ///     overflows (<c>reps * len</c> past 2^63 - 1) it is left untouched and 1 is stored into the <c>int</c> auxdata
        ///     points at, which must be non-null.
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The text element width in bytes.</param>
        /// <param name="outWidth">The output element width in bytes.</param>
        /// <param name="textIndex">The text operand's slot, 0 or 1.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An invalid unit, width or slot.</exception>
        /// <remarks>
        ///     The NARROW kernel (<see cref="IsNarrowRepeat"/>) and the GENERAL kernel write the same bytes; which one a key
        ///     gets is fixed by its widths, so the cache holds one kernel per key either way.
        /// </remarks>
        internal static NDInnerLoopFunc GetTextRepeatKernel(int unit, int width, int outWidth, int textIndex)
        {
            ValidateTextShape(unit, width);
            ValidateTextShape(unit, outWidth);
            if (textIndex is not (0 or 1))
                throw new ArgumentOutOfRangeException(nameof(textIndex), textIndex, "The text operand is slot 0 or 1.");
            if (!TextKernelsAvailable)
                return null;
            var key = new TextRepeatKey(unit, width, outWidth, textIndex);
            if (s_textRepeatKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textRepeatKernels.GetOrAdd(key, IsNarrowRepeat(key) ? GenerateTextRepeatNarrowKernel(key) : GenerateTextRepeatKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextRepeatKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        // =============================================================================================================
        // Emission
        // =============================================================================================================

        /// <summary>
        ///     Emits a concatenation kernel: per element, A's chunks are copied to the output and scanned for A's length,
        ///     the output's B part is zeroed, and B's chunks are copied at A's length.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        /// <remarks>
        ///     The order of the three writes is what makes the result right without a variable-length zero fill: after A's
        ///     copy, <c>out[len(A), wa)</c> holds A's trailing NULs; after the zero fill, <c>out[wa, wa + wb)</c> is zero; B's
        ///     copy then overwrites <c>out[len(A), len(A) + wb)</c> with B INCLUDING its own trailing NULs, so every position
        ///     past <c>len(A) + len(B)</c> is zero.
        /// </remarks>
        private static NDInnerLoopFunc GenerateTextConcatKernel(TextConcatKey key)
        {
            var dm = NewTextKernelMethod($"NDTextConcat_u{key.Unit}_{key.WidthA}_{key.WidthB}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { 0, 1, 2 });
            var planA = PlanTextRange(0, key.WidthA, key.Unit);
            var planB = PlanTextRange(0, key.WidthB, key.Unit);
            var lane = typeof(byte);
            var length = il.DeclareLocal(typeof(int));   // len(A) in BYTES, a whole number of units
            var chunkA = TextChunkLocal(il, planA.Kind, lane);
            var chunkB = TextChunkLocal(il, planB.Kind, lane);

            EmitTextElementLoop(il, frame, advance: new[] { true, true, true }, body: () =>
            {
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, length);

                // 1. A → out[0, wa), and len(A) from the same chunk values.
                ForEachTextChunk(il, planA, site =>
                {
                    EmitTextChunkLoad(il, frame.Pointers[0], site, planA.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunkA);
                    EmitTextChunkStore(il, frame.Pointers[2], null, 0, site, planA.Kind, lane, chunkA);
                    EmitTextLengthUpdate(il, chunkA, site, planA.Kind, lane, key.Unit, length);
                });

                // 2. out[wa, wa + wb) = 0 (B's chunk plan, shifted past A).
                if (planB.Count > 0)
                {
                    EmitTextZero(il, planB.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunkB);
                    ForEachTextChunk(il, planB, site =>
                        EmitTextChunkStore(il, frame.Pointers[2], null, key.WidthA, site, planB.Kind, lane, chunkB));
                }

                // 3. B → out[len(A), len(A) + wb).
                ForEachTextChunk(il, planB, site =>
                {
                    EmitTextChunkLoad(il, frame.Pointers[1], site, planB.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunkB);
                    EmitTextChunkStore(il, frame.Pointers[2], length, 0, site, planB.Kind, lane, chunkB);
                });
            });

            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>Emits a length kernel: per element, the chunk scan for the last nonzero unit, stored as int64 units.</summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateTextLengthKernel(TextLengthKey key)
        {
            var dm = NewTextKernelMethod($"NDTextLength_u{key.Unit}_{key.Width}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { 0, 1 });
            var plan = PlanTextRange(0, key.Width, key.Unit);
            var lane = typeof(byte);
            var length = il.DeclareLocal(typeof(int));
            var chunk = TextChunkLocal(il, plan.Kind, lane);

            EmitTextElementLoop(il, frame, advance: new[] { true, true }, body: () =>
            {
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, length);
                ForEachTextChunk(il, plan, site =>
                {
                    EmitTextChunkLoad(il, frame.Pointers[0], site, plan.Kind, lane);
                    il.Emit(OpCodes.Stloc, chunk);
                    EmitTextLengthUpdate(il, chunk, site, plan.Kind, lane, key.Unit, length);
                });
                // *(long*)out = len in units (bytes >> 2 for U).
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Ldloc, length);
                if (key.Unit == 4)
                {
                    il.Emit(OpCodes.Ldc_I4_2);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Stind_I8);
            });

            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>
        ///     Emits a repetition kernel: per element one direct call to <see cref="RepeatElement"/> with the widths as
        ///     constants; an overflow flag OR-ed over the call is stored into <c>*(int*)auxdata</c> at the end when set.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateTextRepeatKernel(TextRepeatKey key)
        {
            var dm = NewTextKernelMethod($"NDTextRepeat_u{key.Unit}_{key.Width}_{key.OutWidth}_t{key.TextIndex}");
            var il = dm.GetILGenerator();
            // Frame order: text, reps, out (whatever slots the caller's operand order puts them in).
            var frame = EmitTextFrame(il, new[] { key.TextIndex, 1 - key.TextIndex, 2 });
            var overflow = il.DeclareLocal(typeof(int));
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, overflow);

            EmitTextElementLoop(il, frame, advance: new[] { true, true, true }, body: () =>
            {
                // overflow |= RepeatElement(text, width, unit, *(long*)reps, out, outWidth)
                il.Emit(OpCodes.Ldloc, overflow);
                il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                il.Emit(OpCodes.Ldc_I4, key.Width);
                il.Emit(OpCodes.Ldc_I4, key.Unit);
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldloc, frame.Pointers[2]);
                il.Emit(OpCodes.Ldc_I4, key.OutWidth);
                il.Emit(OpCodes.Call, s_repeatElement);
                il.Emit(OpCodes.Or);
                il.Emit(OpCodes.Stloc, overflow);
            });

            EmitTextOverflowFlagStore(il, overflow);
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>
        ///     Whether a repetition key takes the NARROW kernel: a text of at most <see cref="NarrowRepeatMaxWidth"/> bytes
        ///     whose chunk plan is unrolled (each chunk is held in its own local across the copies), and an output of at
        ///     most <see cref="NarrowRepeatMaxOutWidth"/> bytes (the stack scratch row).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>True for the narrow kernel.</returns>
        private static bool IsNarrowRepeat(TextRepeatKey key)
            => key.Width <= NarrowRepeatMaxWidth && key.OutWidth <= NarrowRepeatMaxOutWidth && PlanTextRange(0, key.Width, key.Unit).Unrolled;

        /// <summary>
        ///     Emits the NARROW repetition kernel (see the file header): per element, the text's chunks into locals and its
        ///     length from them; an empty text or a count below 1 zero-fills the output; a one-unit text is a broadcast fill
        ///     (or, for a partial fill, the general helper); otherwise the overflow test, then the scratch row — zeroed,
        ///     written with the text once per copy at <c>k * len</c> while <c>k &lt; reps</c> and the copy starts inside the
        ///     output, and copied out at the output width.
        /// </summary>
        /// <param name="key">The kernel identity (a narrow one: <see cref="IsNarrowRepeat"/>).</param>
        /// <returns>The compiled kernel.</returns>
        /// <remarks>
        ///     <para>
        ///     <b>Why the bytes match NumPy's loop.</b> A copy stores the text's whole width — its <c>len</c> bytes of text,
        ///     then its own trailing NULs. The next copy, at <c>len</c> further, overwrites those NULs with text, so after the
        ///     last copy the row holds the repetition followed by NULs (the last copy's own, then the zeroed row). A copy that
        ///     starts inside the output but runs past it is a cut repetition's partial copy (NumPy's <c>pad</c>) — or a
        ///     whole copy whose trailing NULs reach past the output — and the copy-out drops everything past the output
        ///     width, which is why the row is one text width longer than the output.
        ///     </para>
        ///     <para>
        ///     <b>Lifetime.</b> The row is <c>localloc</c> memory of this kernel call (freed on return), sized
        ///     <c>outWidth + width + 32</c> bytes, at most 672.
        ///     </para>
        /// </remarks>
        private static NDInnerLoopFunc GenerateTextRepeatNarrowKernel(TextRepeatKey key)
        {
            var dm = NewTextKernelMethod($"NDTextRepeatNarrow_u{key.Unit}_{key.Width}_{key.OutWidth}_t{key.TextIndex}");
            var il = dm.GetILGenerator();
            // Frame order: text, reps, out (whatever slots the caller's operand order puts them in).
            var frame = EmitTextFrame(il, new[] { key.TextIndex, 1 - key.TextIndex, 2 });
            var planText = PlanTextRange(0, key.Width, key.Unit);
            var planOut = PlanTextRange(0, key.OutWidth, key.Unit);
            var lane = typeof(byte);
            int shift = key.Unit == 4 ? 2 : 0;
            long outUnits = key.OutWidth / key.Unit;

            var overflow = il.DeclareLocal(typeof(int));
            var scratch = il.DeclareLocal(typeof(byte*));
            var length = il.DeclareLocal(typeof(int));    // len in BYTES, a whole number of units
            var reps = il.DeclareLocal(typeof(long));
            var copyOffset = il.DeclareLocal(typeof(int));
            var copies = il.DeclareLocal(typeof(long));
            var textChunks = new LocalBuilder[planText.Count];
            for (int i = 0; i < textChunks.Length; i++)
                textChunks[i] = TextChunkLocal(il, planText.Kind, lane);
            var outChunk = TextChunkLocal(il, planOut.Kind, lane);
            var zero = TextChunkLocal(il, planOut.Kind, lane);

            // overflow = 0; scratch = localloc(outWidth + width + 32); zero = 0 (hoisted out of the element loop).
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, overflow);
            il.Emit(OpCodes.Ldc_I4, key.OutWidth + key.Width + 32);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, scratch);
            EmitTextZero(il, planOut.Kind, lane);
            il.Emit(OpCodes.Stloc, zero);

            EmitTextElementLoop(il, frame, advance: new[] { true, true, true }, body: () =>
            {
                var emptyElement = il.DefineLabel();
                var oneUnit = il.DefineLabel();
                var noOverflow = il.DefineLabel();
                var copyTop = il.DefineLabel();
                var copyOut = il.DefineLabel();
                var partialFill = il.DefineLabel();
                var next = il.DefineLabel();

                // The text's chunks, and len from the same values.
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, length);
                ForEachTextChunk(il, planText, site =>
                {
                    EmitTextChunkLoad(il, frame.Pointers[0], site, planText.Kind, lane);
                    il.Emit(OpCodes.Stloc, textChunks[site.Index]);
                    EmitTextLengthUpdate(il, textChunks[site.Index], site, planText.Kind, lane, key.Unit, length);
                });

                // reps = *(long*)countPointer; reps < 1 or an empty text: an empty element.
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, reps);
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Blt, emptyElement);
                il.Emit(OpCodes.Ldloc, length);
                il.Emit(OpCodes.Brfalse, emptyElement);
                il.Emit(OpCodes.Ldloc, length);
                il.Emit(OpCodes.Ldc_I4, key.Unit);
                il.Emit(OpCodes.Beq, oneUnit);

                // npy_mul_with_overflow: reps * lenUnits past 2^63 - 1 leaves the element untouched. A count that fits an
                // int cannot overflow (lenUnits < 2^31), so the division runs only for huge counts.
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Ldc_I8, (long)int.MaxValue);
                il.Emit(OpCodes.Ble, noOverflow);
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Ldc_I8, long.MaxValue);
                il.Emit(OpCodes.Ldloc, length);
                if (shift != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, shift);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Div);
                il.Emit(OpCodes.Ble, noOverflow);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Stloc, overflow);
                il.Emit(OpCodes.Br, next);
                il.MarkLabel(noOverflow);

                // The scratch row: zeroed at the output width (the copies cover the rest they touch).
                ForEachTextChunk(il, planOut, site => EmitTextChunkStore(il, scratch, null, 0, site, planOut.Kind, lane, zero));
                // for (off = 0, k = 0; k < reps && off < outWidth; off += len, k++) store the text at row + off.
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, copyOffset);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, copies);
                il.MarkLabel(copyTop);
                il.Emit(OpCodes.Ldloc, copies);
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Bge, copyOut);
                il.Emit(OpCodes.Ldloc, copyOffset);
                il.Emit(OpCodes.Ldc_I4, key.OutWidth);
                il.Emit(OpCodes.Bge, copyOut);
                ForEachTextChunk(il, planText, site =>
                    EmitTextChunkStore(il, scratch, copyOffset, 0, site, planText.Kind, lane, textChunks[site.Index]));
                il.Emit(OpCodes.Ldloc, copyOffset);
                il.Emit(OpCodes.Ldloc, length);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, copyOffset);
                il.Emit(OpCodes.Ldloc, copies);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, copies);
                il.Emit(OpCodes.Br, copyTop);
                // The row's first outWidth bytes are the element.
                il.MarkLabel(copyOut);
                ForEachTextChunk(il, planOut, site =>
                {
                    EmitTextChunkLoad(il, scratch, site, planOut.Kind, lane);
                    il.Emit(OpCodes.Stloc, outChunk);
                    EmitTextChunkStore(il, frame.Pointers[2], null, 0, site, planOut.Kind, lane, outChunk);
                });
                il.Emit(OpCodes.Br, next);

                // An empty text or a count below 1: an empty element (NumPy's buffer_fill_with_zeros_after_index(0)).
                il.MarkLabel(emptyElement);
                ForEachTextChunk(il, planOut, site => EmitTextChunkStore(il, frame.Pointers[2], null, 0, site, planOut.Kind, lane, zero));
                il.Emit(OpCodes.Br, next);

                // A one-unit text (NumPy's buffer_memset, never an overflow): reps covering the output is a broadcast of
                // the unit over the whole width; a shorter fill is the general helper's (a fill, then zeros).
                il.MarkLabel(oneUnit);
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Ldc_I8, outUnits);
                il.Emit(OpCodes.Blt, partialFill);
                EmitTextBroadcastUnit(il, frame.Pointers[0], key.Unit, planOut.Kind, lane);
                il.Emit(OpCodes.Stloc, outChunk);
                ForEachTextChunk(il, planOut, site => EmitTextChunkStore(il, frame.Pointers[2], null, 0, site, planOut.Kind, lane, outChunk));
                il.Emit(OpCodes.Br, next);
                il.MarkLabel(partialFill);
                il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                il.Emit(OpCodes.Ldc_I4, key.Width);
                il.Emit(OpCodes.Ldc_I4, key.Unit);
                il.Emit(OpCodes.Ldloc, reps);
                il.Emit(OpCodes.Ldloc, frame.Pointers[2]);
                il.Emit(OpCodes.Ldc_I4, key.OutWidth);
                il.Emit(OpCodes.Call, s_repeatElement);
                il.Emit(OpCodes.Pop);   // a fill never overflows: RepeatElement returns 0 here

                il.MarkLabel(next);
            });

            EmitTextOverflowFlagStore(il, overflow);
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>Emits <c>if (overflow != 0) *(int*)auxdata = 1</c>: the repetition kernels' report to their driver.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="overflow">The int local OR-ed over the call's elements.</param>
        private static void EmitTextOverflowFlagStore(ILGenerator il, LocalBuilder overflow)
        {
            var clean = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, overflow);
            il.Emit(OpCodes.Brfalse, clean);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Stind_I4);
            il.MarkLabel(clean);
        }

        /// <summary>
        ///     Pushes a chunk value of <paramref name="kind"/> filled with the element's first code unit: a broadcast vector
        ///     (<c>Vector{N}.Create(unit)</c> viewed as bytes), or the unit repeated across a ulong — 8 times for a byte
        ///     unit, twice for a 4-byte one, which a narrower scalar store truncates to the same pattern (a <c>U</c> plan's
        ///     chunks sit on unit boundaries, so every store starts at a whole unit).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The text element pointer local.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="kind">The chunk kind of the plan the value is stored with.</param>
        /// <param name="lane">The vector lane type of the chunk locals (byte).</param>
        private static void EmitTextBroadcastUnit(ILGenerator il, LocalBuilder pointer, int unit, TextChunkKind kind, Type lane)
        {
            il.Emit(OpCodes.Ldloc, pointer);
            if (unit == 4)
            {
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_U4);
            }
            else
                il.Emit(OpCodes.Ldind_U1);

            if (IsVectorChunk(kind))
            {
                int bits = TextChunkBytes(kind) * 8;
                if (unit == 4)
                {
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateBroadcast(bits, typeof(uint)));
                    il.Emit(OpCodes.Call, VectorMethodCache.As(bits, typeof(uint), lane));
                }
                else
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateBroadcast(bits, lane));
                return;
            }
            il.Emit(OpCodes.Conv_U8);
            il.Emit(OpCodes.Ldc_I8, unit == 4 ? 0x0000000100000001L : 0x0101010101010101L);
            il.Emit(OpCodes.Mul);
        }

        /// <summary>A local that holds one chunk value of a kind (a Vector{256,128}&lt;lane&gt; or a ulong).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The vector lane type.</param>
        /// <returns>The local.</returns>
        private static LocalBuilder TextChunkLocal(ILGenerator il, TextChunkKind kind, Type lane)
            => il.DeclareLocal(IsVectorChunk(kind) ? VectorMethodCache.V(TextChunkBytes(kind) * 8, lane) : typeof(ulong));

        /// <summary>Pushes the zero value of a chunk kind (a zero vector, or a zero ulong).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The vector lane type.</param>
        private static void EmitTextZero(ILGenerator il, TextChunkKind kind, Type lane)
        {
            if (IsVectorChunk(kind))
                il.Emit(OpCodes.Call, VectorMethodCache.Zero(TextChunkBytes(kind) * 8, lane));
            else
                il.Emit(OpCodes.Ldc_I8, 0L);
        }

        /// <summary>
        ///     Stores the chunk value held in <paramref name="value"/> at <c>pointer + [offset local] + baseOffset + the site's
        ///     offset</c>. Every store is unaligned-safe (an output element sits at any byte for <c>S</c>, at any 4-byte
        ///     boundary or worse for a <c>U</c> view).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="offset">An int local added to the address (B's position after A's length), or null.</param>
        /// <param name="baseOffset">A constant added to the address (the zero fill's start past A), usually 0.</param>
        /// <param name="site">The chunk's site within its plan.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The vector lane type.</param>
        /// <param name="value">The local holding the chunk value.</param>
        private static void EmitTextChunkStore(ILGenerator il, LocalBuilder pointer, LocalBuilder offset, int baseOffset,
                                               TextChunkSite site, TextChunkKind kind, Type lane, LocalBuilder value)
        {
            bool vector = IsVectorChunk(kind);
            // Vector{N}.Store(value, T* destination) takes the value first; the stind opcodes take the address first.
            if (vector)
                il.Emit(OpCodes.Ldloc, value);
            il.Emit(OpCodes.Ldloc, pointer);
            if (offset is not null)
            {
                il.Emit(OpCodes.Ldloc, offset);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            if (baseOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I4, baseOffset);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            if (site.VarOffset is not null || site.ConstOffset != 0)
            {
                PushTextSiteOffset(il, site);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            if (vector)
            {
                il.Emit(OpCodes.Call, VectorMethodCache.Store(TextChunkBytes(kind) * 8, lane));
                return;
            }
            il.Emit(OpCodes.Ldloc, value);
            switch (kind)
            {
                case TextChunkKind.UInt64:
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Stind_I8);
                    break;
                case TextChunkKind.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Stind_I4);
                    break;
                case TextChunkKind.UInt16:
                    il.Emit(OpCodes.Conv_U2);
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Stind_I2);
                    break;
                default:
                    il.Emit(OpCodes.Conv_U1);
                    il.Emit(OpCodes.Stind_I1);
                    break;
            }
        }

        /// <summary>
        ///     Emits <c>if (chunk != 0) length = site + ((highest nonzero byte of chunk) &gt;&gt; shift) + 1 &lt;&lt; shift</c>:
        ///     the element's length in bytes, rounded up to its code unit, as far as this chunk says. Run over the chunks in
        ///     increasing offset order, the last nonzero chunk decides, which is the element's last nonzero unit — the
        ///     overlapping last chunk of a plan included (it covers the range's end, and a nonzero byte in it is the highest).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="chunk">The local holding the chunk value.</param>
        /// <param name="site">The chunk's site.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The vector lane type (byte lanes: one mask bit per byte).</param>
        /// <param name="unit">The code-unit size (1 or 4).</param>
        /// <param name="length">The int local holding the length so far (bytes).</param>
        private static void EmitTextLengthUpdate(ILGenerator il, LocalBuilder chunk, TextChunkSite site, TextChunkKind kind,
                                                 Type lane, int unit, LocalBuilder length)
        {
            var skip = il.DefineLabel();
            var high = il.DeclareLocal(typeof(int));
            if (IsVectorChunk(kind))
            {
                // mask = one bit per NONZERO byte: ~(chunk == 0), most significant bit of each byte lane.
                int bits = TextChunkBytes(kind) * 8;
                var mask = il.DeclareLocal(typeof(uint));
                il.Emit(OpCodes.Ldloc, chunk);
                il.Emit(OpCodes.Call, VectorMethodCache.Zero(bits, lane));
                il.Emit(OpCodes.Call, VectorMethodCache.Equals(bits, lane));
                il.Emit(OpCodes.Call, VectorMethodCache.OnesComplement(bits, lane));
                il.Emit(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(bits, lane));
                il.Emit(OpCodes.Stloc, mask);
                il.Emit(OpCodes.Ldloc, mask);
                il.Emit(OpCodes.Brfalse, skip);
                // high = 31 - lzcnt(mask): the highest nonzero byte of the chunk.
                il.Emit(OpCodes.Ldc_I4, 31);
                il.Emit(OpCodes.Ldloc, mask);
                il.Emit(OpCodes.Call, s_lzcnt32);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, high);
            }
            else
            {
                // A scalar chunk is zero-extended to a ulong (little-endian: byte k at bits 8k..8k+7).
                il.Emit(OpCodes.Ldloc, chunk);
                il.Emit(OpCodes.Brfalse, skip);
                il.Emit(OpCodes.Ldc_I4, 63);
                il.Emit(OpCodes.Ldloc, chunk);
                il.Emit(OpCodes.Call, s_lzcnt64);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Ldc_I4_3);
                il.Emit(OpCodes.Shr_Un);
                il.Emit(OpCodes.Stloc, high);
            }

            // length = site + ((high >> shift) + 1) << shift   (shift = 0 for S, 2 for U).
            int shift = unit == 4 ? 2 : 0;
            PushTextSiteOffset(il, site);
            il.Emit(OpCodes.Ldloc, high);
            if (shift != 0)
            {
                il.Emit(OpCodes.Ldc_I4, shift);
                il.Emit(OpCodes.Shr_Un);
            }
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            if (shift != 0)
            {
                il.Emit(OpCodes.Ldc_I4, shift);
                il.Emit(OpCodes.Shl);
            }
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, length);
            il.MarkLabel(skip);
        }

        // =============================================================================================================
        // The repetition element
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>string_multiply</c> for one element: <paramref name="reps"/> copies of the element's text (its units
        ///     up to the last nonzero one) written into <paramref name="dst"/>, cut at <paramref name="outWidth"/>, the rest
        ///     zero-filled.
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="reps">The repeat count (NumPy converts the integer operand to int64 first).</param>
        /// <param name="dst">The output element's first byte (never overlapping <paramref name="src"/>: the caller's iterator
        ///     copies an overlapping operand first).</param>
        /// <param name="outWidth">The output element width in bytes.</param>
        /// <returns>1 when <c>reps * len</c> overflows 2^63 - 1 (the element is then left untouched, as NumPy leaves it), 0
        ///     otherwise.</returns>
        /// <remarks>
        ///     A one-unit string is a fill of <c>min(reps, outUnits)</c> units (NumPy's <c>buffer_memset</c>; it never
        ///     overflows). A longer string writes <c>outUnits / len</c> whole copies plus <c>outUnits % len</c> units when the
        ///     repetition is longer than the output; the copies DOUBLE (copy the text once, then the written prefix onto the
        ///     space after it), so k copies cost log2(k) block copies instead of k.
        /// </remarks>
        internal static unsafe int RepeatElement(byte* src, int width, int unit, long reps, byte* dst, int outWidth)
        {
            int len = LastNonzeroEnd(src, width, unit);
            if (reps < 1 || len == 0)
            {
                new Span<byte>(dst, outWidth).Clear();
                return 0;
            }

            long lenUnits = len / unit;
            long outUnits = outWidth / unit;
            if (lenUnits == 1)
            {
                // buffer_memset: the one unit min(reps, width) times, then zeros.
                int n = (int)Math.Min(reps, outUnits);
                if (unit == 1)
                    new Span<byte>(dst, n).Fill(*src);
                else
                    new Span<uint>(dst, n).Fill(Unsafe.ReadUnaligned<uint>(src));
                new Span<byte>(dst + n * unit, outWidth - n * unit).Clear();
                return 0;
            }

            // npy_mul_with_overflow_size_t(&newlen, reps, len1) || newlen > PY_SSIZE_T_MAX: reps * len1 > 2^63 - 1.
            if (reps > long.MaxValue / lenUnits)
                return 1;
            long newlen = reps * lenUnits;
            // The repetition is cut at the output: whole copies, then a partial one (pad units).
            long copies = reps, pad = 0;
            if (outUnits < newlen)
            {
                copies = outUnits / lenUnits;
                pad = outUnits % lenUnits;
            }

            long total = (copies * lenUnits + pad) * unit;   // bytes written; ≤ outWidth, so it fits an int
            if (total > 0)
            {
                // The first copy (or the partial one alone), then doubling copies of the written prefix.
                long first = Math.Min(len, total);
                Buffer.MemoryCopy(src, dst, first, first);
                long written = first;
                while (written < total)
                {
                    long chunk = Math.Min(written, total - written);
                    Buffer.MemoryCopy(dst, dst + written, chunk, chunk);
                    written += chunk;
                }
            }
            new Span<byte>(dst + total, (int)(outWidth - total)).Clear();
            return 0;
        }

        /// <summary>
        ///     The length of a string element in BYTES: the end of its last nonzero code unit (NumPy's
        ///     <c>num_codepoints</c> times the unit size). Vectorized by the span primitives.
        /// </summary>
        /// <param name="src">The element's first byte.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>0 for an element of NULs; otherwise a positive multiple of <paramref name="unit"/>.</returns>
        internal static unsafe int LastNonzeroEnd(byte* src, int width, int unit)
        {
            int last = new ReadOnlySpan<byte>(src, width).LastIndexOfAnyExcept((byte)0);
            // The unit holding the last nonzero byte ends at the next unit boundary.
            return last < 0 ? 0 : (last / unit + 1) * unit;
        }
    }
}
