using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class AllocationTests
    {
        private const long onegb = 1_073_741_824;
        private static readonly object _lock = new Object();

        [TestMethod]
        public void Allocate_1GB()
        {
            lock (_lock)
            {
                var shape = new Shape((1, (int)onegb));
                shape.Should().BeShaped(1, (int)onegb).And.NotBeOfSize(1);
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Ignore("Fails expectedly. Int32 can not address this size any more")]
        [TestMethod]
        public void Allocate_2GB()
        {
            lock (_lock)
            {
                var shape = new Shape((2, (int)onegb + 1));
                shape.Should().BeShaped(2, (int)onegb + 1).And.NotBeOfSize(2);;
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Ignore("Fails expectedly. Int32 can not address this size any more")]
        [TestMethod]
        public void Allocate_4GB()
        {
            lock (_lock)
            {
                var shape = new Shape((4, (int)onegb + 1));
                shape.Should().BeShaped(4, (int)onegb + 1).And.NotBeOfSize(4);;
                new Action(() => np.ones(shape, NPTypeCode.Int32)).Should().NotThrow();
            }
        }

        [Ignore("Fails expectedly. Int32 can not address this size any more")]
        [TestMethod]
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
