using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Unmanaged
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
            const int iterations = 10_000;
            // Epoch BEFORE the kept slices exist, so the collection below can prove it reaches every one of them.
            var since = GcQuiescence.Epoch.Capture();
            //alocate and store
            var l = new List<ArraySlice<float>>(iterations);
            for (int i = 0; i < iterations; i++)
            {
                l.Add(inner(3));
            }

            // Force the GC to decide the kept slices' fate, then let the finalizers of anything it found unreachable
            // run — a block wrongly released under a live slice frees its memory for the allocations below to
            // overwrite. CollectSince takes a young collection when that is exact (no collection ran since the
            // epoch: every slice and its disposer is still in gen 0/1) and the full one otherwise; the finalizer
            // wait is deterministic where the former Thread.Sleep(40) after a full GC.Collect() only hoped the
            // finalizer thread had run (~120 ms in a full run for the pair, the full collection marking the whole
            // test host's heap).
            GcQuiescence.CollectSince(since);
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
