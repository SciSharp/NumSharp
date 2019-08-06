using System;
using System.Collections;
using System.Collections.Generic;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Memory.Pooling;
using NumSharp.Unmanaged.Memory;

namespace NumSharp.UnitTest.Utilities
{
    [TestClass]
    public class StackedMemoryPoolTests
    {
        [TestMethod]
        public void TakeExceedStored()
        {
            var pool = new StackedMemoryPool(4, 10);
            pool.Available.Should().Be(10);
            var l = new List<IntPtr>();
            for (int i = 0; i < 10; i++) l.Add(pool.Take());
            pool.Available.Should().Be(0);
            var next = pool.Take();
            pool.Available.Should().Be(0);
            l.Add(next);
            next.Should().NotBe(IntPtr.Zero);

            for (int i = 0; i < 11; i++) 
                pool.Return(l[i]);

            pool.Available.Should().Be(11);
        }
    }
}
