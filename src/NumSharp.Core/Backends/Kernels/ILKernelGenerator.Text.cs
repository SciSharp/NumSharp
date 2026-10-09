using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Text.cs — the inner loops of the fixed-width string ufuncs (S / U)
// =============================================================================
//
// RESPONSIBILITY
//   The NDIter inner loops (NDInnerLoopFunc) that compute on fixed-width text: the six comparison ufuncs (NumPy's
//   string_comparison_loop, string_ufuncs.cpp) and a string's truth value (NumPy's STRING_nonzero / UNICODE_nonzero and
//   the string -> bool cast). One DynamicMethod per (operation, code-unit size, element widths): the widths are EMIT-TIME
//   CONSTANTS, so the element body is straight-line code over a fixed chunk plan — a <U8 element (32 bytes) is ONE
//   Vector256 load per operand.
//
// NUMPY'S SEMANTICS (string_buffer.h, Buffer<enc>::strcmp, probed against 2.4.2)
//   Two strings compare their common prefix code unit by code unit as UNSIGNED values (npy_ucs4 for U; for S the char is
//   sign-extended to npy_ucs4, which keeps the unsigned byte order: 0x80..0xFF land above 0x00..0x7F, in order); at the
//   first difference the smaller unit is the smaller string. When the prefix is equal, a NONZERO unit in the longer
//   operand's remaining tail makes it the greater; trailing NULs never count ('a' == 'a\0'), embedded ones do. That is
//   exactly a lexicographic unsigned compare of both elements zero-padded to the larger width — what the kernels compute.
//   A unit above U+10FFFF (reachable through views) is just a large unsigned value here.
//
// THE CHUNK PLAN
//   A byte range [start, start + length) is read as chunks of the largest size W that fits (32 = Vector256, 16 =
//   Vector128, 8, 4, 2, 1 bytes; never below one code unit): length / W chunks, plus one chunk ending exactly at the range
//   end when W does not divide it (an OVERLAPPING read). Overlap is harmless: equality and "any nonzero" are idempotent,
//   and the first-difference search reaches the overlapping chunk only when everything before it compared equal, so the
//   first difference it finds is the first of the range. Up to TextUnrollMaxChunks chunks are unrolled; a longer range
//   runs its full chunks in a loop (constant trip count) and the overlapping one after it.
//
// THE KERNELS
//   * comparison: (A, B, out bool). Equal / NotEqual xor-or-accumulate the prefix (branch-free) and or-accumulate the
//     longer operand's tail; Less / LessEqual search the prefix for the first differing unit (vector: Equals +
//     ExtractMostSignificantBits + TrailingZeroCount; scalar chunks: xor + TrailingZeroCount), then compare that unit
//     unsigned, or decide from the tail. Greater / GreaterEqual are Less / LessEqual with the operands swapped (the
//     kernel's A / B operand indices are emit-time constants).
//   * broadcast patterns: a stride-0 operand (a 0-d / scalar comparand, a broadcast row) is read ONCE per call — its
//     chunks hoisted into locals before the loop, its tail's truth precomputed. Each pattern is a STAGE: a DynamicMethod
//     of its own that the kernel forwards to with one call, compiled on its first call (the BroadcastPatterns rationale:
//     one method per instruction stream keeps the JIT's inline budget for the vector helpers). Elements wider than
//     TextHoistMaxBytes are not hoisted (their loop form reads by offset).
//   * truth: (text, out bool) — any nonzero unit. The cast flavor (NumPy's string -> bool cast builds a Python str per
//     element, and a unit above U+10FFFF makes PyUnicode_New raise SystemError) also ORs every U unit's "> 0x10FFFF"
//     test into a flag it stores into auxdata (int*) when set; the caller raises.
//
// CONTRACT
//   The kernels follow NDInnerLoopFunc: dataptrs / strides (BYTES) per operand, count elements per call, any strides
//   (a negative or zero stride is fine). They read exactly the element bytes [0, width) of every operand element.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The comparison a text comparison kernel computes for its (A, B) operand pair. <c>greater</c> and
    ///     <c>greater_equal</c> are not listed: they are <see cref="Less"/> / <see cref="LessEqual"/> with the operands
    ///     swapped (<see cref="ILKernelGenerator.GetTextCompareKernel"/>'s <c>aIndex</c>), which keeps the kernel count down.
    /// </summary>
    internal enum TextCompareOp : byte
    {
        /// <summary>A == B: every unit equal after zero-padding to the larger width.</summary>
        Equal,

        /// <summary>A != B.</summary>
        NotEqual,

        /// <summary>A &lt; B: the first differing unit is smaller in A (unsigned), or A is a strict zero-padded prefix.</summary>
        Less,

        /// <summary>A &lt;= B.</summary>
        LessEqual,
    }

    public static partial class ILKernelGenerator
    {
        /// <summary>
        ///     The most chunks a range is unrolled into before its full chunks become a loop: 9 covers every element up to
        ///     256 bytes with Vector256 chunks (8 full + 1 overlapping), the width cap the strings plan fixed. Wider elements
        ///     get the loop form, so the emitted IL stays small for any width.
        /// </summary>
        private const int TextUnrollMaxChunks = 9;

        /// <summary>
        ///     The widest element whose chunks a broadcast stage hoists into locals (one local per chunk). Wider broadcast
        ///     operands are read by offset each element, still from L1.
        /// </summary>
        private const int TextHoistMaxBytes = 256;

        /// <summary>The comparison kernels, keyed by (op, unit, widths, operand order).</summary>
        private static readonly ConcurrentDictionary<TextCompareKey, NDInnerLoopFunc> s_textCompareKernels = new();

        /// <summary>The truth kernels, keyed by (unit, width, cast flavor).</summary>
        private static readonly ConcurrentDictionary<TextTruthKey, NDInnerLoopFunc> s_textTruthKernels = new();

        /// <summary>The identity of a comparison kernel: everything its IL depends on.</summary>
        /// <param name="Op">The comparison of A against B.</param>
        /// <param name="Unit">The code-unit size: 1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="WidthA">A's element width in bytes.</param>
        /// <param name="WidthB">B's element width in bytes.</param>
        /// <param name="AIndex">The operand index A is read from (0 or 1; B is the other).</param>
        private readonly record struct TextCompareKey(TextCompareOp Op, int Unit, int WidthA, int WidthB, int AIndex);

        /// <summary>The identity of a truth kernel.</summary>
        /// <param name="Unit">The code-unit size: 1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="Width">The element width in bytes.</param>
        /// <param name="CheckCodePoints">The cast flavor: also flag U units above U+10FFFF.</param>
        private readonly record struct TextTruthKey(int Unit, int Width, bool CheckCodePoints);

        /// <summary>
        ///     The comparison kernel for one (op, unit, widths, operand order), generated on first use and cached. Its
        ///     operands are <c>dataptrs[aIndex]</c> = A, <c>dataptrs[1 - aIndex]</c> = B and <c>dataptrs[2]</c> = the bool
        ///     output; a <c>where=</c> mask, when present, is applied by <see cref="NDIterRef.ForEach"/> around the kernel.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's element width in bytes (its descriptor's itemsize).</param>
        /// <param name="widthB">B's element width in bytes.</param>
        /// <param name="aIndex">0 to compare operand 0 against operand 1; 1 for the swapped order (greater /
        ///     greater_equal run as less / less_equal of (x2, x1)).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller takes its managed route).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, a negative width, a width that is
        ///     not a whole number of units, or an operand index other than 0 or 1.</exception>
        internal static NDInnerLoopFunc GetTextCompareKernel(TextCompareOp op, int unit, int widthA, int widthB, int aIndex)
        {
            ValidateTextShape(unit, widthA);
            ValidateTextShape(unit, widthB);
            if (aIndex is not (0 or 1))
                throw new ArgumentOutOfRangeException(nameof(aIndex), aIndex, "A comparison reads its operands from slots 0 and 1.");
            if (!TextKernelsAvailable)
                return null;
            var key = new TextCompareKey(op, unit, widthA, widthB, aIndex);
            if (s_textCompareKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textCompareKernels.GetOrAdd(key, GenerateTextCompareKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                // Emission failed on this runtime (a missing vector API, an AOT-restricted host): the managed route answers.
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextCompareKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The truth kernel for one (unit, width, flavor): <c>out[i] = any unit of text[i] is nonzero</c>, operands
        ///     <c>dataptrs[0]</c> = text and <c>dataptrs[1]</c> = the bool output.
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="checkCodePoints">The cast flavor for <c>U</c>: the kernel also tests every unit against U+10FFFF
        ///     and, when one is above it, stores 1 into the <c>int</c> auxdata points at (which must then be non-null).
        ///     Ignored for <c>S</c> (bytes never fail NumPy's bool cast).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, or a width that is negative or not a
        ///     whole number of units.</exception>
        internal static NDInnerLoopFunc GetTextTruthKernel(int unit, int width, bool checkCodePoints)
        {
            ValidateTextShape(unit, width);
            if (!TextKernelsAvailable)
                return null;
            var key = new TextTruthKey(unit, width, checkCodePoints && unit == 4);
            if (s_textTruthKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textTruthKernels.GetOrAdd(key, GenerateTextTruthKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextTruthKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Whether the text kernels can be emitted here: dynamic code must be supported (not a NativeAOT host) and the IL
        ///     generator enabled. When false every getter returns null and the callers run their managed element loops.
        /// </summary>
        internal static bool TextKernelsAvailable
            => System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported && DirectILKernelGenerator.Enabled;

        /// <summary>Rejects a code-unit size or element width no fixed-width string descriptor can have.</summary>
        /// <param name="unit">The code-unit size.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <exception cref="ArgumentOutOfRangeException">The unit is not 1 or 4, or the width is negative or not a multiple
        ///     of the unit.</exception>
        private static void ValidateTextShape(int unit, int width)
        {
            if (unit is not (1 or 4))
                throw new ArgumentOutOfRangeException(nameof(unit), unit, "A string code unit is 1 byte (S) or 4 bytes (U).");
            if (width < 0 || width % unit != 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "An element width is a non-negative whole number of code units.");
        }

        // =============================================================================================================
        // The chunk plan
        // =============================================================================================================

        /// <summary>How one chunk of an element is read.</summary>
        private enum TextChunkKind : byte
        {
            /// <summary>32 bytes as a Vector256 (byte or uint lanes).</summary>
            Vector256,

            /// <summary>16 bytes as a Vector128.</summary>
            Vector128,

            /// <summary>8 bytes as a ulong.</summary>
            UInt64,

            /// <summary>4 bytes, zero-extended to a ulong.</summary>
            UInt32,

            /// <summary>2 bytes, zero-extended to a ulong.</summary>
            UInt16,

            /// <summary>1 byte, zero-extended to a ulong.</summary>
            Byte,
        }

        /// <summary>The byte size of a chunk kind.</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>32, 16, 8, 4, 2 or 1.</returns>
        private static int TextChunkBytes(TextChunkKind kind) => kind switch
        {
            TextChunkKind.Vector256 => 32,
            TextChunkKind.Vector128 => 16,
            TextChunkKind.UInt64 => 8,
            TextChunkKind.UInt32 => 4,
            TextChunkKind.UInt16 => 2,
            _ => 1,
        };

        /// <summary>Whether a chunk kind is a vector (its value is a Vector{128,256}, otherwise a ulong).</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>True for the two vector kinds.</returns>
        private static bool IsVectorChunk(TextChunkKind kind) => kind is TextChunkKind.Vector256 or TextChunkKind.Vector128;

        /// <summary>
        ///     A byte range of an element, planned as chunks: <see cref="FullCount"/> chunks of <see cref="Kind"/> at
        ///     <c>Start + k * size</c>, then (when <see cref="HasOverlap"/>) one more ending exactly at the range end.
        /// </summary>
        private readonly struct TextChunkPlan
        {
            /// <summary>The range's first byte, relative to the element.</summary>
            internal readonly int Start;

            /// <summary>The range's length in bytes (0 = an empty plan, no chunks).</summary>
            internal readonly int Length;

            /// <summary>How every chunk of the range is read.</summary>
            internal readonly TextChunkKind Kind;

            /// <summary>The number of non-overlapping chunks from <see cref="Start"/>.</summary>
            internal readonly int FullCount;

            /// <summary>Whether a last chunk at <c>Start + Length - size</c> covers the remainder (overlapping the previous).</summary>
            internal readonly bool HasOverlap;

            /// <summary>Plans a range (see the type summary).</summary>
            /// <param name="start">The first byte.</param>
            /// <param name="length">The length in bytes.</param>
            /// <param name="kind">The chunk kind.</param>
            internal TextChunkPlan(int start, int length, TextChunkKind kind)
            {
                Start = start;
                Length = length;
                Kind = kind;
                int size = TextChunkBytes(kind);
                FullCount = length <= 0 ? 0 : length / size;
                HasOverlap = length > 0 && length % size != 0;
            }

            /// <summary>The chunk size in bytes.</summary>
            internal int Size => TextChunkBytes(Kind);

            /// <summary>The total chunk count (full + overlap).</summary>
            internal int Count => FullCount + (HasOverlap ? 1 : 0);

            /// <summary>Whether the range is emitted chunk by chunk (constant offsets) rather than as a loop.</summary>
            internal bool Unrolled => Count <= TextUnrollMaxChunks;

            /// <summary>The offset of chunk <paramref name="i"/> (0 ≤ i &lt; <see cref="Count"/>).</summary>
            /// <param name="i">The chunk index.</param>
            /// <returns>Its first byte, relative to the element.</returns>
            internal int Offset(int i) => i < FullCount ? Start + i * Size : Start + Length - Size;
        }

        /// <summary>
        ///     Plans the byte range [<paramref name="start"/>, <paramref name="start"/> + <paramref name="length"/>): the
        ///     largest chunk that fits the range (and the host's vectors), at least one code unit.
        /// </summary>
        /// <param name="start">The first byte (a multiple of <paramref name="unit"/>).</param>
        /// <param name="length">The length in bytes (a multiple of <paramref name="unit"/>; 0 = nothing to read).</param>
        /// <param name="unit">The code-unit size; chunks never split a unit, so a U range's chunk offsets stay aligned to
        ///     4-byte units (the first-difference search maps a byte position back to its unit).</param>
        /// <returns>The plan.</returns>
        private static TextChunkPlan PlanTextRange(int start, int length, int unit)
        {
            if (length <= 0)
                return new TextChunkPlan(start, 0, TextChunkKind.Byte);
            TextChunkKind kind;
            if (length >= 32 && Vector256.IsHardwareAccelerated)
                kind = TextChunkKind.Vector256;
            else if (length >= 16 && Vector128.IsHardwareAccelerated)
                kind = TextChunkKind.Vector128;
            else if (length >= 8)
                kind = TextChunkKind.UInt64;
            else if (length >= 4)
                kind = TextChunkKind.UInt32;
            else if (length >= 2)
                kind = TextChunkKind.UInt16;
            else
                kind = TextChunkKind.Byte;
            // A U range is a multiple of 4 bytes, so every kind chosen above is at least one unit (4 <= length);
            // the 1- and 2-byte kinds can only appear for S.
            System.Diagnostics.Debug.Assert(TextChunkBytes(kind) >= unit);
            return new TextChunkPlan(start, length, kind);
        }
    }
}
