using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class DeallocationTests
    {
        [TestMethod]
        public unsafe void DisposerCopiedAcrossStructCopy()
        {
            var newMem = new UnmanagedMemoryBlock<int>(5);
            var mem2 = newMem;
            Console.WriteLine(newMem);
            Assert.IsTrue(ReferenceEquals(newMem, mem2) == false);
            ReferenceEquals(mem2.GetType().GetField("_disposer", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(mem2),
                mem2.GetType().GetField("_disposer", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(newMem));
        }

        [TestMethod]
        public unsafe void GcDoesntCollectArraySliceAlone()
        {
            //this test should be churned.
            const int iterations = 100_000;
            //alocate and store
            var l = new List<ArraySlice<float>>(iterations);
            for (int i = 0; i < iterations; i++)
            {
                l.Add(inner(3));
            }

            //force GC
            GC.Collect();
            Thread.Sleep(40); //2 thread cycles
            //allocate more with different value for the chance of overriding previous memory
            for (int i = 0; i < iterations*10; i++)
            {
                inner(5);
            }

            //all stored values should be 3.
            for (int i = 0; i < iterations; i++)
            {
                l[i].All(f => f == 3f).Should().BeTrue();
            }
        }

        unsafe ArraySlice<float> inner(int val)
        {
            var mm = new UnmanagedMemoryBlock<float>(15, val);

            var addr = (IntPtr)mm.Address;
            var arr = new ArraySlice<float>(mm);
            return arr.Slice(5, 5);
        }
    }
}
