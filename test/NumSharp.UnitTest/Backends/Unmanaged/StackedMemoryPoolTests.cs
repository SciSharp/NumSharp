using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Unmanaged.Memory;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class StackedMemoryPoolTests
    {
        [TestMethod]
        public void Case1()
        {
            var l = new Stack<IntPtr>();
            var stack = new StackedMemoryPool(12, 100);
            stack.GarbageCollectionDelay = 1000;
            for (int i = 0; i < 100; i++)
                l.Push(stack.Take());

            for (int i = 0; i < 34; i++)
                l.Push(stack.Take());

            for (int i = 0; i < 133; i++)
                stack.Return(l.Pop());

            var manual = new ManualResetEventSlim();
            stack.GCInvoked += () =>
            {
                stack.Available.Should().Be(100);
                manual.Set();
            };

            stack.Return(l.Pop());
            manual.Wait(5000).Should().BeTrue();
        }
    }
}

