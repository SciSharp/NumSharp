using System.Runtime.InteropServices;

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
    }
}
