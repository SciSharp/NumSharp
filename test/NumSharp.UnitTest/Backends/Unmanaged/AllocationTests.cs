using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    /// <summary>
    /// Tests for large memory allocations.
    /// Marked as [HighMemory] because they allocate 1-4GB of memory and
    /// can cause OOM on CI runners (especially ubuntu with limited RAM).
    /// </summary>
    [HighMemory]
    public class AllocationTests
    {
        private const long onegb = 1_073_741_824;
        private static readonly object _lock = new Object();

        [Test]
        [HighMemory]
        [SkipOnLowMemory(8)] // Actually allocates 4GB (Int32 * 1B elements)
        public void Allocate_1GB()
        {
            lock (_lock)
            {
                var shape = new Shape((1, (int)onegb));
                shape.Should().BeShaped(1, (int)onegb).And.NotBeOfSize(1);
                new Action(() => np.empty(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Test]
        [HighMemory]
        [SkipOnLowMemory(12)] // Actually allocates 8GB (Int32 * 2B elements)
        public void Allocate_2GB()
        {
            lock (_lock)
            {
                var shape = new Shape((2, (int)onegb + 1));
                shape.Should().BeShaped(2, (int)onegb + 1).And.NotBeOfSize(2);;
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Test]
        [HighMemory]
        [SkipOnLowMemory(20)] // Actually allocates 16GB (Int32 * 4B elements)
        public void Allocate_4GB()
        {
            lock (_lock)
            {
                var shape = new Shape((4, (int)onegb + 1));
                shape.Should().BeShaped(4, (int)onegb + 1).And.NotBeOfSize(4);;
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Test]
        [HighMemory]
        [SkipOnLowMemory(50)] // Actually allocates 44GB+
        [OpenBugs]
        public void Allocate_44GB()
        {
            lock (_lock)
            {
                var shape = new Shape((44, (int)onegb + 1));
                shape.Should().BeShaped(44, (int)onegb + 1).And.NotBeOfSize(44);
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }
    }
}
