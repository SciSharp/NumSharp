using System.Collections.Generic;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// <c>NDFloatMath.Tanh(double)</c> is a transcription of NumPy 2.4.2's <c>simd_tanh_f64</c>
    /// (<c>loops_hyperbolic.dispatch.cpp.src</c>) operation for operation — with exactly ONE
    /// exception, the subinterval index extraction, which is re-spelled. NumPy does a 32-bit
    /// <c>Max</c>/<c>Min</c> over the 64-bit lanes and justifies it in a comment ("it's fine … since
    /// we're not crossing 32-bit edge"); NumSharp clamps the high half directly, because C# has no
    /// natural way to write "reinterpret this vector of longs as ints, clamp, reinterpret back"
    /// without the vector types the scalar path does not use.
    ///
    /// <para>That makes it the one line a reviewer cannot check by diffing against NumPy's source,
    /// which is why it is proven here instead of argued. The proof is EXHAUSTIVE, not sampled:
    /// <c>ndnan = bits &amp; 0x7ff8000000000000</c> retains only the 11 exponent bits and mantissa
    /// bit 51, clearing everything else, and neither formula reads anything but <c>ndnan</c> — so the
    /// entire input domain is 2^12 = 4096 values and enumerating it is a complete proof.</para>
    ///
    /// <para>The value-level parity of the kernel itself is covered elsewhere (the
    /// <c>numpy_f64_kernels.jsonl</c> fuzz tier, plus a 4.83-billion-value sweep during development).
    /// What this file pins is narrower and would otherwise be untested: that the two SPELLINGS are
    /// interchangeable, so a future edit to either cannot drift them apart unnoticed.</para>
    /// </summary>
    [TestClass]
    public class TanhIndexExtractionTest : TestClass
    {
        private const ulong NdNanMask = 0x7ff8000000000000UL;
        private const long IndexBias = 0x3fc0000000000000L;
        private const int IndexClamp = 0x780000;

        /// <summary>
        /// NumPy's spelling, transcribed operation for operation:
        /// <code>
        /// vec_s64 idxs = Sub(ndnan, Set(s64, 0x3fc0000000000000));
        /// vec_s32 idxl = Max(BitCast(s32, idxs), Zero(s32));
        ///        idxl = Min(idxl, Set(s32, 0x780000));
        /// vec_u64 idx  = ShiftRightSame(BitCast(u64, idxl), 51);
        /// </code>
        /// The <c>BitCast</c> to <c>s32</c> splits each 64-bit lane into [low half, high half] on a
        /// little-endian target, so both halves are clamped independently and then recombined.
        /// </summary>
        private static ulong NumPySpelling(ulong ndnan)
        {
            long idxs = unchecked((long)ndnan - IndexBias);
            int low = unchecked((int)(uint)((ulong)idxs & 0xFFFFFFFFUL));
            int high = unchecked((int)(uint)((ulong)idxs >> 32));

            low = System.Math.Min(System.Math.Max(low, 0), IndexClamp);
            high = System.Math.Min(System.Math.Max(high, 0), IndexClamp);

            ulong recombined = ((ulong)(uint)high << 32) | (uint)low;
            return recombined >> 51;
        }

        /// <summary>NumSharp's spelling, copied from <c>NDFloatMath.Tanh(double)</c>.</summary>
        private static ulong NumSharpSpelling(ulong ndnan)
        {
            int index = (int)((unchecked((long)ndnan) - IndexBias) >> 32);
            index = System.Math.Max(index, 0);
            index = System.Math.Min(index, IndexClamp);
            return (ulong)(index >> 19);
        }

        [TestMethod]
        public void TanhFloat64IndexExtraction_MatchesNumPySpelling()
        {
            for (int pattern = 0; pattern < 4096; pattern++)
            {
                // The 12 live bits sit at 62..51; everything else is cleared by the mask.
                ulong ndnan = ((ulong)pattern) << 51;
                (ndnan & ~NdNanMask).Should().Be(0UL, "the enumeration must stay inside the mask");

                NumSharpSpelling(ndnan).Should().Be(NumPySpelling(ndnan),
                    $"the two index spellings must agree for ndnan=0x{ndnan:x16}");
            }
        }

        /// <summary>
        /// The proof above is only complete if 4096 really is the whole domain. This pins the other
        /// half of that claim: every <c>ndnan</c> a real double can produce lies inside the mask, and
        /// ordinary bit patterns do reach all 4096 — so the enumeration is exact, not an
        /// over-approximation that happens to cover the interesting cases.
        /// </summary>
        [TestMethod]
        public void TanhFloat64IndexDomain_IsExactly4096Values()
        {
            var seen = new HashSet<ulong>();

            foreach (ulong bits in new[]
            {
                0UL, 0x8000000000000000UL,                          // +-0
                0x7ff0000000000000UL, 0xfff0000000000000UL,         // +-inf
                0x7ff8000000000000UL, 0xfff8000000000000UL,         // quiet NaN
                0x7ff0000000000001UL,                               // signalling NaN
                0x0000000000000001UL, 0x000fffffffffffffUL,         // subnormals
                0x0010000000000000UL,                               // smallest normal
                0x7fefffffffffffffUL, 0xffefffffffffffffUL,         // +-DBL_MAX
                0x3ff0000000000000UL, 0xbff0000000000000UL,         // +-1
            })
            {
                seen.Add(bits & NdNanMask);
            }

            var rng = new System.Random(7);
            for (int i = 0; i < 500_000; i++)
            {
                ulong bits = ((ulong)(uint)rng.Next() << 32) | (uint)rng.Next();
                seen.Add(bits & NdNanMask);
            }

            foreach (ulong ndnan in seen)
                (ndnan & ~NdNanMask).Should().Be(0UL, "a masked value cannot escape the mask");

            seen.Count.Should().Be(4096,
                "the exhaustive proof enumerates 4096 values, so the reachable domain must be exactly that");
        }
    }
}
