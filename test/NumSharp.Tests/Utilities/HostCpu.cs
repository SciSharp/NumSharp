using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace NumSharp.Tests
{
    /// <summary>
    /// The NaN an invalid IEEE operation (<c>0 * inf</c>, <c>inf - inf</c>, <c>0 / 0</c>) GENERATES on the CPU running the
    /// tests. Use it for every expected value whose NaN was born from such an operation (no NaN among the inputs); keep
    /// literal bits for a NaN that was PROPAGATED from an input, whose sign and payload do not depend on the CPU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sign of a generated NaN is fixed by the hardware, not chosen by NumSharp or NumPy: x86/x64 SSE/AVX return the
    /// "real indefinite" QNaN, NEGATIVE (<c>0xFFF8000000000000</c>, float16 <c>0xFE00</c>), while AArch64 returns its default
    /// NaN, POSITIVE (<c>0x7FF8000000000000</c>, float16 <c>0x7E00</c>), and so do the other non-x86 ISAs .NET runs on.
    /// NumPy's expectations in this suite were probed on win-amd64, so they carry the negative bits; NumPy's own arm64
    /// wheels produce the positive ones for the same call. Asserting the x86 bits everywhere is what turned the macOS
    /// (Apple-silicon) CI leg red on 2026-10-02 in eight tests (complex np.convolve, polymul, mapdomain, hermline, float16
    /// polyval, legacy standard_t / chisquare, Generator f).
    /// </para>
    /// <para>
    /// The architecture is read from <see cref="RuntimeInformation.ProcessArchitecture"/> rather than probed with a
    /// division, because a probe must defeat constant folding to mean anything: Roslyn folds <c>0.0 / 0.0</c> at build time
    /// into the metadata constant behind <c>double.NaN</c> (negative on every host), so a naive probe reports x86 bits on
    /// arm64. An x64 process under Rosetta 2 or another x86 emulator reports X64 and gets the x86 bits, which is what such
    /// an emulator produces. Same rule as <c>HalfArithKernelTests.DefaultQNaNHalf</c>.
    /// </para>
    /// </remarks>
    internal static class HostCpu
    {
        /// <summary>
        /// True when the process runs x86 or x64 code, whose generated NaN is NEGATIVE — the host NumPy's expectations were
        /// probed on, so these are the cases where the literal NumPy bits apply unchanged.
        /// </summary>
        public static bool IsX86 => RuntimeInformation.ProcessArchitecture is Architecture.X86 or Architecture.X64;

        /// <summary>
        /// True on an x86 host with NumPy's X86_V3 dispatch target (AVX2, FMA3, BMI1, BMI2, LZCNT): the host whose NumPy
        /// 2.4.2 SIMD loops run 32-byte vectors, so a NaN pattern shaped by their blocks (the scalar-broadcast region rule:
        /// blocks of 8 float64 / 16 float32 keep the broadcast operand's NaN) is the probed contract. Gate such a pattern on
        /// it; never use it for a value or a NaN sign, which do not depend on the vector width.
        /// </summary>
        /// <remarks>
        /// Any other host splits the blocks elsewhere, NumSharp included (it derives the block from this same feature set,
        /// <c>DirectILKernelGenerator.ComputeNumPyBroadcastSimdBlock</c>: 16-byte vectors without X86_V3, as NumPy's
        /// dispatcher would choose), and arm64's NumPy is compiled by another compiler with another operand order, so the
        /// win-amd64 patterns are no contract there. An AVX-512 host is still X86_V3: NumPy builds no AVX-512
        /// <c>loops_arithm_fp</c> target, so it runs the 32-byte loop. Same predicate as the oracle's
        /// <c>FuzzCorpusTests.NaNOperandOrderHost</c> (the <c>nan_order</c> tier).
        /// </remarks>
        public static bool IsX86V3 => IsX86 && Avx2.IsSupported && Fma.IsSupported && Bmi1.IsSupported && Bmi2.IsSupported
                                      && Lzcnt.IsSupported;

        /// <summary>
        /// The float64 bits of a generated NaN on this CPU: <c>0xFFF8000000000000</c> on x86/x64 (NumPy win-amd64's bits),
        /// <c>0x7FF8000000000000</c> elsewhere. Compare a <see cref="System.BitConverter.DoubleToUInt64Bits"/> result
        /// against it.
        /// </summary>
        public static ulong DefaultNaN64Bits => IsX86 ? 0xFFF8000000000000UL : 0x7FF8000000000000UL;

        /// <summary>
        /// The float16 bits of a generated NaN on this CPU: <c>0xFE00</c> on x86/x64, <c>0x7E00</c> elsewhere. NumPy's HALF
        /// loops compute in float32 and narrow, so this is the float32 default NaN (<c>0xFFC00000</c> / <c>0x7FC00000</c>)
        /// narrowed.
        /// </summary>
        public static ushort DefaultNaN16Bits => IsX86 ? (ushort)0xFE00 : (ushort)0x7E00;

        /// <summary>
        /// <see cref="DefaultNaN64Bits"/> as NumPy's <c>tobytes().hex()</c> spells one float64 (little-endian, lowercase):
        /// <c>000000000000f8ff</c> on x86/x64, <c>000000000000f87f</c> elsewhere. Splice it into a hex dump at exactly the
        /// 16-character slots that hold a generated NaN; a blind text replace would also rewrite propagated NaNs.
        /// </summary>
        public static string DefaultNaN64Hex => IsX86 ? "000000000000f8ff" : "000000000000f87f";

        /// <summary>
        /// <see cref="DefaultNaN16Bits"/> as NumPy's <c>tobytes().hex()</c> spells one float16 (little-endian, lowercase):
        /// <c>00fe</c> on x86/x64, <c>007e</c> elsewhere.
        /// </summary>
        public static string DefaultNaN16Hex => IsX86 ? "00fe" : "007e";

        /// <summary>
        /// The float32 bits of a generated NaN on this CPU: <c>0xFFC00000</c> on x86/x64, <c>0x7FC00000</c> elsewhere.
        /// Compare a <see cref="BitConverter.SingleToUInt32Bits"/> result against it.
        /// </summary>
        public static uint DefaultNaN32Bits => IsX86 ? 0xFFC00000U : 0x7FC00000U;

        /// <summary>
        /// <see cref="DefaultNaN32Bits"/> as NumPy's <c>tobytes().hex()</c> spells one float32 (little-endian, lowercase):
        /// <c>0000c0ff</c> on x86/x64, <c>0000c07f</c> elsewhere.
        /// </summary>
        public static string DefaultNaN32Hex => IsX86 ? "0000c0ff" : "0000c07f";

        /// <summary>
        /// <see cref="DefaultNaN64Bits"/> as a 16-digit big-endian word, the way <c>bits.ToString("x16")</c> prints it and
        /// the bit-dump expectations of the reduction tests spell an element: <c>fff8000000000000</c> on x86/x64,
        /// <c>7ff8000000000000</c> elsewhere. Not interchangeable with <see cref="DefaultNaN64Hex"/>, whose bytes run the
        /// other way.
        /// </summary>
        public static string DefaultNaN64Word => DefaultNaN64Bits.ToString("x16");

        /// <summary>
        /// <see cref="DefaultNaN32Bits"/> as an 8-digit big-endian word (<c>bits.ToString("x8")</c>): <c>ffc00000</c> on
        /// x86/x64, <c>7fc00000</c> elsewhere.
        /// </summary>
        public static string DefaultNaN32Word => DefaultNaN32Bits.ToString("x8");

        /// <summary>
        /// <see cref="DefaultNaN16Bits"/> as a 4-digit big-endian word (<c>bits.ToString("x4")</c>): <c>fe00</c> on x86/x64,
        /// <c>7e00</c> elsewhere.
        /// </summary>
        public static string DefaultNaN16Word => DefaultNaN16Bits.ToString("x4");

        /// <summary>
        /// A NumPy <c>tobytes().hex()</c> dump probed on win-amd64, with every element slot that holds x86's GENERATED NaN
        /// (sign set, payload 0) rewritten to this host's (<see cref="DefaultNaN64Hex"/> / <see cref="DefaultNaN32Hex"/> /
        /// <see cref="DefaultNaN16Hex"/>); unchanged on x86/x64.
        /// </summary>
        /// <param name="littleEndianHex">The dump, two lowercase hex digits per byte, elements in C order; a complex128
        ///     dump is two float64 slots per element.</param>
        /// <param name="itemBytes">The width of one real slot: 8 (float64, and each part of a complex128), 4 (float32) or
        ///     2 (float16).</param>
        /// <returns>The dump this host's NumPy writes for the same call.</returns>
        /// <remarks>
        /// <b>Footgun:</b> the rewrite matches bits, not history. Use it only for a dump whose operands hold no NaN with
        /// exactly x86's default bits, so that every such slot was generated by an invalid operation; a PROPAGATED input NaN
        /// that happens to carry those bits keeps them on every CPU, and this would wrongly rewrite it. A propagated NaN with
        /// any payload (<c>560400000000f8ff</c>) or the positive sign is never touched.
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="littleEndianHex"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="itemBytes"/> is not 2, 4 or 8.</exception>
        /// <exception cref="ArgumentException">The dump's length is not a whole number of
        ///     <paramref name="itemBytes"/>-byte slots, so a slot boundary would be guessed.</exception>
        public static string WithHostDefaultNaN(string littleEndianHex, int itemBytes)
        {
            ArgumentNullException.ThrowIfNull(littleEndianHex);
            string x86 = itemBytes switch
            {
                8 => "000000000000f8ff",
                4 => "0000c0ff",
                2 => "00fe",
                _ => throw new ArgumentOutOfRangeException(nameof(itemBytes), itemBytes, "a real slot is 2, 4 or 8 bytes wide"),
            };
            int slot = 2 * itemBytes;
            if (littleEndianHex.Length % slot != 0)
                throw new ArgumentException($"a dump of {littleEndianHex.Length} hex digits is not whole {itemBytes}-byte slots",
                                            nameof(littleEndianHex));
            if (IsX86)
                return littleEndianHex;

            string host = itemBytes switch { 8 => DefaultNaN64Hex, 4 => DefaultNaN32Hex, _ => DefaultNaN16Hex };
            var sb = new StringBuilder(littleEndianHex.Length);
            // Slot by slot, never a text replace: "...f8ff" could otherwise match across a slot boundary.
            for (int i = 0; i < littleEndianHex.Length; i += slot)
            {
                string s = littleEndianHex.Substring(i, slot);
                sb.Append(string.Equals(s, x86, StringComparison.Ordinal) ? host : s);
            }

            return sb.ToString();
        }
    }
}
