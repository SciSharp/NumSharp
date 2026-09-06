using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The <c>size × itemsize</c> overflow guard, and negative dimensions.
    /// </summary>
    /// <remarks>
    ///     Every expectation was probed against NumPy 2.4.2 and the messages are verbatim.
    ///
    ///     What this pins is a MEMORY-SAFETY bug, not a cosmetic one. <c>np.zeros((2^31, 2^31))</c>
    ///     of float64 has element count 2^62 — representable — but 2^62 × 8 is 2^65, which wraps to
    ///     exactly 0. The allocator was asked for zero bytes and obliged, so the array constructed,
    ///     <c>[0,0]</c> read 0, and the last element read out of bounds; <c>np.ones</c> on the same
    ///     shape wrote out of bounds and took the process down.
    ///
    ///     None of these tests allocate: every case is rejected before an allocator is reached,
    ///     which is why they carry no <c>[HighMemory]</c> attribute.
    /// </remarks>
    [TestClass]
    public class AllocationGuardTests
    {
        private const string TooBig =
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.";

        private const string Negative = "negative dimensions are not allowed";

        // AwesomeAssertions' WithMessage treats the expectation as a wildcard pattern, so the
        // backticks and '*'-free text are fine, but '.' and '`' are literal. Escape nothing;
        // just match exactly.
        private static void ShouldThrowTooBig(Action act)
            => act.Should().ThrowExactly<ValueError>().WithMessage(TooBig);

        private static void ShouldThrowNegative(Action act)
            => act.Should().ThrowExactly<ValueError>().WithMessage(Negative);

        [TestMethod]
        public void Zeros_ByteCountOverflowsButElementCountDoesNot_IsRejected()
        {
            // np.zeros((2**31, 2**31)) -> ValueError: array is too big; ...
            // size = 2**62 (fine), size*8 = 2**65 -> wraps to 0.
            ShouldThrowTooBig(() => np.zeros(new Shape(1L << 31, 1L << 31)));
        }

        [TestMethod]
        public void Zeros_ElementCountItselfOverflows_IsRejected()
        {
            // np.zeros((2**30, 2**33)) -> ValueError: array is too big; ...
            // NumSharp used to report a NEGATIVE size here (2**63 wraps to long.MinValue).
            ShouldThrowTooBig(() => np.zeros(new Shape(1L << 30, 1L << 33)));
        }

        [TestMethod]
        public void AllCreationEntryPoints_ShareTheGuard()
        {
            var big = new Shape(1L << 31, 1L << 31);
            ShouldThrowTooBig(() => np.zeros(big));
            ShouldThrowTooBig(() => np.empty(big));
            ShouldThrowTooBig(() => np.ones(big));
            ShouldThrowTooBig(() => np.full(big, 1.0d));
        }

        [TestMethod]
        public void Zeros_ExactByteCountBoundary_ForFloat64()
        {
            // 2**60 elements * 8 bytes = 2**63 -> overflows signed 64-bit. NumPy rejects.
            ShouldThrowTooBig(() => np.zeros(new Shape(1L << 60)));
        }

        [TestMethod]
        public void ItemSize_ParticipatesInTheCheck_NotJustElementCount()
        {
            // The SAME dimensions are legal at a smaller itemsize, which is the proof that the
            // quantity being checked is bytes and not elements:
            //   np.zeros((0, 2**62))          -> ValueError (8 * 2**62 = 2**65)
            //   np.zeros((0, 2**62), int8)    -> shape (0, 4611686018427387904)
            ShouldThrowTooBig(() => np.zeros(new Shape(0, 1L << 62)));

            var ok = np.zeros(new Shape(0, 1L << 62), NPTypeCode.Byte);
            ok.Shape.Dimensions.Should().Equal(0L, 1L << 62);
            ok.size.Should().Be(0);
        }

        [TestMethod]
        public void ZeroDimension_DoesNotSuppressTheOverflowCheck()
        {
            // NumPy keeps multiplying "as if" a zero dim were 1, precisely so this still reports.
            // A naive `if (size == 0) return;` short-circuit would let both of these through.
            ShouldThrowTooBig(() => np.zeros(new Shape(0, 1L << 62)));
            ShouldThrowTooBig(() => np.zeros(new Shape(1L << 62, 1L << 62, 0)));
        }

        [TestMethod]
        public void NegativeDimensions_ReportNumPyText_NotOutOfMemory()
        {
            // Before the guard these surfaced as OutOfMemoryException, because a negative count
            // reached the allocator.
            ShouldThrowNegative(() => np.zeros(new Shape(-1L)));
            ShouldThrowNegative(() => np.zeros(new Shape(2L, -3L)));
            ShouldThrowNegative(() => np.zeros(new Shape(0, -1L)));
            ShouldThrowNegative(() => np.zeros(new Shape(-1L, 0)));
        }

        [TestMethod]
        public void CheckOrder_FollowsNumPys_LeftToRightScan()
        {
            // Dimensions are scanned in order and each check fires in place, so which error you get
            // depends on which bad dimension comes first:
            //   (-1, 2**62) -> negative dimensions      (dim 0 is negative)
            //   (2**62, -1) -> array is too big         (dim 0 already overflows 8 * 2**62)
            ShouldThrowNegative(() => np.zeros(new Shape(-1L, 1L << 62)));
            ShouldThrowTooBig(() => np.zeros(new Shape(1L << 62, -1L)));
        }

        [TestMethod]
        public void OrdinaryAllocations_AreUnaffected()
        {
            np.zeros(new Shape(2, 3)).Shape.Dimensions.Should().Equal(2L, 3L);
            np.zeros(new Shape(0, 3)).Shape.Dimensions.Should().Equal(0L, 3L);
            np.ones(new Shape(4)).Shape.Dimensions.Should().Equal(4L);
            np.zeros(new Shape(Array.Empty<long>())).Shape.Dimensions.Should().BeEmpty();

            // and the values are still right
            np.ones(new Shape(3)).GetDouble(0).Should().Be(1d);
            np.zeros(new Shape(3)).GetDouble(2).Should().Be(0d);
        }

        [TestMethod]
        public void EveryDtype_IsGuarded()
        {
            // 2**62 elements overflows the byte count for every itemsize > 2.
            foreach (var tc in new[]
            {
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            })
            {
                new Action(() => np.zeros(new Shape(1L << 62), tc))
                    .Should().ThrowExactly<ValueError>($"{tc} must be size-guarded")
                    .WithMessage(TooBig);
            }
        }

        [TestMethod]
        public void ElementCountItselfWrappingToZero_IsCaughtOnEveryCreationPath()
        {
            // Two DIFFERENT wrap points, and a guard that only sees one of them misses the other:
            //   (2^31, 2^31) -> element count 2^62 is fine, the BYTE count wraps
            //   (2^62, 2^62) -> the ELEMENT COUNT itself wraps, to exactly 0
            // The second slipped past np.full / np.full_like / np.ones, which allocate from
            // shape.size directly rather than through UnmanagedStorage.Allocate — so they were
            // handed a count of 0, allocated nothing, and returned an array claiming (2^62, 2^62).
            var wrapsToZero = new Shape(new[] { 1L << 62, 1L << 62 });

            ShouldThrowTooBig(() => np.full(wrapsToZero, 1));
            ShouldThrowTooBig(() => np.full(wrapsToZero, 1.0d));
            ShouldThrowTooBig(() => np.ones(wrapsToZero));
            ShouldThrowTooBig(() => np.zeros(wrapsToZero));
            ShouldThrowTooBig(() => np.empty(wrapsToZero));

            wrapsToZero.size.Should().Be(0, "this is precisely why the guard cannot key off size");
        }

        [TestMethod]
        public void FillingCreators_StillProduceCorrectValues()
        {
            var full = np.full(new Shape(2, 3), 7);
            full.Shape.Dimensions.Should().Equal(2L, 3L);
            full.GetInt32(1, 2).Should().Be(7);

            var ones = np.ones(new Shape(2, 3));
            ones.Shape.Dimensions.Should().Equal(2L, 3L);
            ones.GetDouble(1, 2).Should().Be(1d);

            var like = np.full_like(np.ones(new Shape(2, 3)), 9);
            like.Shape.Dimensions.Should().Equal(2L, 3L);
            like.GetTypeCode.Should().Be(NPTypeCode.Double,
                "full_like preserves the source dtype unless dtype= is explicit");
            like.GetDouble(1, 2).Should().Be(9d);
        }

        [TestMethod]
        public void MemoryBlockBackstop_RejectsAWrappingCount()
        {
            // The guard at the allocator itself, reached without going through a Shape.
            // 2**61 doubles = 2**64 bytes -> wraps to 0.
            new Action(() => ArraySlice.Allocate(NPTypeCode.Double, 1L << 61))
                .Should().ThrowExactly<ValueError>().WithMessage(TooBig);

            new Action(() => ArraySlice.Allocate(NPTypeCode.Double, -1L))
                .Should().ThrowExactly<ValueError>().WithMessage(Negative);
        }
    }
}
