using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp.Tests
{
    /// <summary>
    /// Comparisons for expected values that were probed from NumPy 2.4.2's win-amd64 wheel AND pass
    /// through the platform C runtime's libm (atanh, acos, sin, …). They are bit-exact on the host they
    /// were probed on and only close elsewhere.
    /// </summary>
    /// <remarks>
    /// <para>
    /// On Windows, .NET's <c>Math.*</c> and NumPy's <c>npy_*</c> both call MSVC <c>ucrtbase</c>, so a
    /// NumPy-probed value is reproduced bit-for-bit. glibc and Apple's libm round some of the same
    /// inputs one ULP differently (measured on CI: glibc <c>atanh(0.5)</c>, macOS <c>acos(0.5)</c> and
    /// <c>acos(0.3)</c>, and <c>sin</c> inside <c>np.sinc</c>). That is a platform artifact, not a
    /// NumSharp-vs-NumPy defect. Asserting the Windows bits everywhere turned the Linux and macOS test
    /// jobs red.
    /// </para>
    /// <para>
    /// The differential-fuzz corpus already handles this by host-pinning such tiers
    /// (<c>FuzzCorpusTests.RunHostLibmCorpus</c>: exact on Windows, inconclusive elsewhere).
    /// These helpers give unit tests the same pin, but keep checking values off-Windows instead of
    /// skipping: exact bits on the pinned host, and elsewhere a bounded ULP distance. NaN-ness,
    /// infinities and the sign bit (including the sign of a zero) must still match exactly, because
    /// none of those are libm rounding artifacts.
    /// </para>
    /// </remarks>
    internal static class HostLibm
    {
        /// <summary>
        /// True on the host whose CRT libm the NumPy expectations were probed against (Windows /
        /// <c>ucrtbase</c>). Same condition as <c>FuzzCorpusTests.RunHostLibmCorpus</c>, so unit tests and
        /// the corpus agree on where bit-exactness is contractual.
        /// </summary>
        public static bool IsPinnedHost => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        /// <summary>
        /// The largest ULP distance accepted off the pinned host. Mainstream libms round these functions to
        /// within 1-2 ULP; the headroom covers one composition step (emath and sinc build on the ufuncs) and
        /// still rejects any real kernel error, which shows up as thousands of ULP or a wrong sign.
        /// </summary>
        public const long OffHostUlpBudget = 4;

        /// <summary>
        /// Whether <paramref name="actual"/> is an acceptable reproduction of the NumPy-probed
        /// <paramref name="expected"/> on this host.
        /// </summary>
        /// <param name="expected">The value NumPy 2.4.2 (win-amd64) produced.</param>
        /// <param name="actual">The value NumSharp produced.</param>
        /// <returns>
        /// On the pinned host: bit equality (NaN sign and payload included). Elsewhere: both NaN, or the
        /// same infinity, or same sign bit and at most <see cref="OffHostUlpBudget"/> ULP apart.
        /// </returns>
        public static bool Matches(double expected, double actual)
        {
            if (IsPinnedHost)
                return BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual);

            if (double.IsNaN(expected) || double.IsNaN(actual))
                return double.IsNaN(expected) && double.IsNaN(actual);
            if (double.IsInfinity(expected) || double.IsInfinity(actual))
                return expected.Equals(actual);
            // A sign difference (including +0 vs -0) is never a rounding artifact.
            if (double.IsNegative(expected) != double.IsNegative(actual))
                return false;
            // Same sign bit: the raw bit patterns are monotonic in magnitude and share their top bit, so
            // their difference IS the ULP distance and cannot overflow.
            long ulps = System.Math.Abs(BitConverter.DoubleToInt64Bits(expected) - BitConverter.DoubleToInt64Bits(actual));
            return ulps <= OffHostUlpBudget;
        }

        /// <summary>
        /// The complex counterpart of <see cref="Matches(double, double)"/>, applied per component.
        /// </summary>
        /// <param name="expected">The value NumPy 2.4.2 (win-amd64) produced.</param>
        /// <param name="actual">The value NumSharp produced.</param>
        /// <returns>
        /// On the pinned host: both components bit-equal. Elsewhere: each component passes
        /// <see cref="Matches(double, double)"/>, or (for a component much smaller than the number itself)
        /// has the same sign and lies within <see cref="OffHostUlpBudget"/> machine epsilons of
        /// <c>|expected|</c>. A tiny cancellation-born component is judged on the whole value's scale.
        /// </returns>
        public static bool Matches(Complex expected, Complex actual)
        {
            if (IsPinnedHost)
                return Matches(expected.Real, actual.Real) && Matches(expected.Imaginary, actual.Imaginary);

            double scale = Complex.Abs(expected);
            return ComponentMatches(expected.Real, actual.Real, scale)
                && ComponentMatches(expected.Imaginary, actual.Imaginary, scale);
        }

        /// <summary>One off-host complex component: ULP-close, or close relative to the whole number's magnitude.</summary>
        /// <param name="expected">Expected component.</param>
        /// <param name="actual">Actual component.</param>
        /// <param name="scale"><c>|expected|</c> of the complex number the component belongs to.</param>
        /// <returns>True when the component is an acceptable off-host reproduction.</returns>
        private static bool ComponentMatches(double expected, double actual, double scale)
        {
            if (Matches(expected, actual))
                return true;
            if (!double.IsFinite(expected) || !double.IsFinite(actual) || double.IsNegative(expected) != double.IsNegative(actual))
                return false;
            return System.Math.Abs(expected - actual) <= OffHostUlpBudget * MachineEpsilon * scale;
        }

        /// <summary>
        /// 2^-52, the spacing of doubles at 1.0 (NumPy's <c>finfo(float64).eps</c>). NOT
        /// <see cref="double.Epsilon"/>, which in .NET is the smallest subnormal (4.9e-324).
        /// </summary>
        private const double MachineEpsilon = 2.220446049250313e-16;

        /// <summary>
        /// Human-readable description of a mismatch, naming the host mode so a failure log says which
        /// contract was violated.
        /// </summary>
        /// <param name="what">What was compared (e.g. "element 1").</param>
        /// <param name="expected">Expected value.</param>
        /// <param name="actual">Actual value.</param>
        /// <returns>A message with both values in round-trip form and the active contract.</returns>
        public static string Describe(string what, double expected, double actual)
            => $"{what}: got {actual:R} (0x{BitConverter.DoubleToInt64Bits(actual):X16}) expected {expected:R} " +
               $"(0x{BitConverter.DoubleToInt64Bits(expected):X16}) — " +
               (IsPinnedHost ? "bit-exact on the pinned win-amd64 CRT" : $"within {OffHostUlpBudget} ULP off the pinned CRT host");

        /// <summary>
        /// Ends the running test as <b>Inconclusive</b> when this is not the pinned host, and does nothing on it. Call it
        /// right before the exact assertion of a NumPy-probed value that did NOT reproduce, when that value comes out of a
        /// libm-sensitive RANDOM stream: on the pinned host the exact assertion that follows still fails the test, and off it
        /// the mismatch is reported as "cannot be adjudicated here" instead of as a NumSharp defect.
        /// </summary>
        /// <param name="why">
        /// What did not reproduce (the case, the position, both values). It leads the Inconclusive message, followed by the
        /// host-pin explanation, so a CI log says which stream drifted and why that is not a failure.
        /// </param>
        /// <exception cref="AssertInconclusiveException">Always, when <see cref="IsPinnedHost"/> is false.</exception>
        /// <remarks>
        /// <para>
        /// A random stream is libm-sensitive twice over, so <see cref="Matches(double, double)"/>' per-value ULP budget is
        /// not enough for it. Its VALUES pass through <c>log</c>/<c>exp</c>/<c>pow</c>/<c>acos</c>, and an fmod wrap
        /// (<c>vonmises</c>) turns one ULP of an intermediate near pi into 8-16 ULP of a result near 0.25. Its DECISIONS pass
        /// through them too: one ULP can flip a rejection test, after which every later draw of the stream differs, and a
        /// SHA-256 digest of a 20,000-draw stream changes with any single bit. Neither is a NumSharp-vs-NumPy defect: NumPy
        /// itself draws that different stream on glibc or Apple's libm. Measured on the 2026-10-02 CI (PR #631, ubuntu and
        /// macOS, NumPy expectations probed on win-amd64): 20 of the 69 seed-2024 digests and 1-4 of the seed-42 tables.
        /// </para>
        /// <para>
        /// So the contract mirrors the corpus's <c>FuzzCorpusTests.RunHostLibmCorpus</c> (exact on the pinned host,
        /// Inconclusive elsewhere), refined per value: callers first accept an off-host value within
        /// <see cref="OffHostUlpBudget"/> (the plain last-bit artifact keeps the rest of the stream checked) and call this
        /// only for what remains. MSTest does not count Inconclusive as a failure, so the Linux/macOS legs stay green while the
        /// Windows leg keeps every byte contractual. The cost: a genuinely wrong stream on a non-Windows host only (an
        /// architecture-specific codegen defect) reads as Inconclusive there; the same managed code is still byte-checked on
        /// the pinned host.
        /// </para>
        /// </remarks>
        public static void InconclusiveOffPinnedHost(string why)
        {
            if (!IsPinnedHost)
                Assert.Inconclusive(why + " — this random stream passes through the platform libm (values and accept/reject " +
                                    "decisions), so its NumPy 2.4.2 bytes are contractual only on the pinned win-amd64 CRT host.");
        }
    }
}
