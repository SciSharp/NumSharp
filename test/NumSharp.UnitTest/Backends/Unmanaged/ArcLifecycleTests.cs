using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    /// <summary>
    ///     Tests for the atomic refcounting (ARC) lifecycle on
    ///     <c>UnmanagedMemoryBlock&lt;T&gt;</c> and <see cref="NDArray"/>'s
    ///     <see cref="IDisposable"/> integration. Covers the invariants:
    ///     <list type="bullet">
    ///       <item>Atomic free on final Release (no finalizer queue dependency).</item>
    ///       <item>Idempotent Dispose — multiple calls are safe.</item>
    ///       <item>View safety — disposing a parent doesn't invalidate live views.</item>
    ///       <item>Thread-safety under concurrent AddRef/Release.</item>
    ///       <item>Stray Release self-healing.</item>
    ///       <item>Non-owning wraps never free.</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class ArcLifecycleTests
    {
        // ----- helpers ------------------------------------------------------

        private static long GetRefCount(IArraySlice slice)
        {
            var mb = slice.MemoryBlock;
            var dispField = mb.GetType()
                .GetField("_disposer", BindingFlags.NonPublic | BindingFlags.Instance);
            var disp = dispField!.GetValue(mb);
            var rcField = disp!.GetType()
                .GetField("_refCount", BindingFlags.NonPublic | BindingFlags.Instance);
            return (long)rcField!.GetValue(disp)!;
        }

        // ----- atomic release ------------------------------------------------

        [TestMethod]
        public void SingleNDArray_DisposeFreesAtomically()
        {
            var a = np.arange(100);
            var slice = a.Storage.InternalArray;
            slice.IsReleased.Should().BeFalse();
            GetRefCount(slice).Should().Be(1);

            a.Dispose();

            slice.IsReleased.Should().BeTrue();
            a.IsDisposed.Should().BeTrue();
            GetRefCount(slice).Should().Be(-1);
        }

        [TestMethod]
        public void Dispose_IsIdempotent()
        {
            var a = np.arange(100);
            a.Dispose();
            a.Dispose();
            a.Dispose();
            a.IsDisposed.Should().BeTrue();
        }

        // ----- view safety ---------------------------------------------------

        [TestMethod]
        public void ParentDispose_DoesNotInvalidateView()
        {
            var a = np.arange(20).reshape(4, 5);
            var view = a["1:3"];

            a.Dispose();

            // View must remain readable
            view.Storage.InternalArray.IsReleased.Should().BeFalse();
            long sum = 0;
            for (int i = 0; i < view.shape[0]; i++)
                for (int j = 0; j < view.shape[1]; j++)
                    sum += (int)view[i, j];
            sum.Should().Be(5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14);

            view.Dispose();
        }

        [TestMethod]
        public void ViewDispose_KeepsParentAlive()
        {
            var a = np.arange(20).reshape(4, 5);
            var view = a["1:3"];

            view.Dispose();

            // Parent must remain readable
            a.Storage.InternalArray.IsReleased.Should().BeFalse();
            ((int)a[0, 0]).Should().Be(0);
            ((int)a[3, 4]).Should().Be(19);

            a.Dispose();
        }

        // ----- finalizer safety net ------------------------------------------

        [TestMethod]
        public void Finalizer_ReleasesUnmanaged_WhenDisposeMissed()
        {
            // Helper isolates the variable so the eval-stack temp doesn't keep
            // the array rooted past the call.
            static (IArraySlice slice, WeakReference weak) MakeAndDrop()
            {
                var a = new NDArray(NPTypeCode.Double, new Shape(10_000), fillZeros: true);
                return (a.Storage.InternalArray, new WeakReference(a));
            }

            var (slice, weak) = MakeAndDrop();
            slice.IsReleased.Should().BeFalse();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            weak.IsAlive.Should().BeFalse("NDArray should be collected after GC");
            slice.IsReleased.Should().BeTrue("finalizer chain must release unmanaged buffer");
        }

        // ----- concurrency ---------------------------------------------------

        [TestMethod]
        public void ConcurrentAddRefRelease_PreservesCount()
        {
            const int threads = 32;
            const int opsPerThread = 1_000;

            var a = np.arange(1_000);
            var slice = a.Storage.InternalArray;
            int addRefFails = 0;

            var tasks = new Task[threads];
            for (int t = 0; t < threads; t++)
            {
                tasks[t] = Task.Run(() =>
                {
                    for (int i = 0; i < opsPerThread; i++)
                    {
                        if (!slice.TryAddRef()) Interlocked.Increment(ref addRefFails);
                        slice.Release();
                    }
                });
            }
            Task.WaitAll(tasks);

            addRefFails.Should().Be(0);
            GetRefCount(slice).Should().Be(1);
            slice.IsReleased.Should().BeFalse();

            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void ParallelDispose_ReleasesExactlyOnce()
        {
            const int threads = 50;
            var a = np.arange(1_000);
            var slice = a.Storage.InternalArray;

            var tasks = new Task[threads];
            for (int t = 0; t < threads; t++)
                tasks[t] = Task.Run(() => a.Dispose());
            Task.WaitAll(tasks);

            a.IsDisposed.Should().BeTrue();
            slice.IsReleased.Should().BeTrue();
            // Even with N parallel disposes, refCount drops to -1 exactly once.
            GetRefCount(slice).Should().Be(-1);
        }

        [TestMethod]
        public void RaceAddRefVsRelease_NeverCorrupts()
        {
            // Adversarial race: thread B tries to AddRef while thread A
            // brings refCount to 0 via Release. The invariant is:
            // "if TryAddRef returned true, the block is NOT released."
            const int iterations = 10_000;
            int corruptions = 0;

            for (int iter = 0; iter < iterations; iter++)
            {
                var a = np.arange(10);
                var slice = a.Storage.InternalArray;
                bool t2Result = false;

                var t1 = Task.Run(() => a.Dispose());
                var t2 = Task.Run(() => t2Result = slice.TryAddRef());
                Task.WaitAll(t1, t2);

                if (t2Result && slice.IsReleased)
                    Interlocked.Increment(ref corruptions);

                if (t2Result) slice.Release();
            }

            corruptions.Should().Be(0);
        }

        // ----- stray release self-healing ------------------------------------

        [TestMethod]
        public void StrayReleases_OnFreedBlock_SelfHeal()
        {
            var a = np.arange(10);
            var slice = a.Storage.InternalArray;
            a.Dispose();
            GetRefCount(slice).Should().Be(-1);

            for (int i = 0; i < 100; i++)
                slice.Release();

            GetRefCount(slice).Should().Be(-1, "stray Releases must self-heal");
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void TryAddRef_OnReleasedBlock_AlwaysFalse()
        {
            var a = np.arange(10);
            var slice = a.Storage.InternalArray;
            a.Dispose();

            for (int i = 0; i < 100; i++)
                slice.TryAddRef().Should().BeFalse();

            GetRefCount(slice).Should().Be(-1);
        }

        // ----- non-owning wraps ----------------------------------------------

        [TestMethod]
        public unsafe void WrapAllocation_IsImmortal()
        {
            var buf = (int*)NativeMemory.Alloc(40);
            try
            {
                var block = new UnmanagedMemoryBlock<int>(buf, 10);
                block.IsReleased.Should().BeFalse();

                block.TryAddRef().Should().BeTrue();
                block.Release();
                block.IsReleased.Should().BeFalse("wraps don't track refcount");

                for (int i = 0; i < 100; i++)
                    block.Release();
                block.IsReleased.Should().BeFalse("Wrap Release is a no-op");
            }
            finally
            {
                NativeMemory.Free(buf);
            }
        }

        [TestMethod]
        public unsafe void WrapAllocation_ConcurrentAccess_IsSafe()
        {
            var buf = (int*)NativeMemory.Alloc(40);
            try
            {
                var block = new UnmanagedMemoryBlock<int>(buf, 10);
                var tasks = new Task[20];
                for (int t = 0; t < tasks.Length; t++)
                {
                    tasks[t] = Task.Run(() =>
                    {
                        for (int i = 0; i < 1_000; i++)
                        {
                            block.TryAddRef();
                            block.Release();
                        }
                    });
                }
                Task.WaitAll(tasks);
                block.IsReleased.Should().BeFalse();
            }
            finally
            {
                NativeMemory.Free(buf);
            }
        }

        // ----- GCHandle (FromArray) ------------------------------------------

        [TestMethod]
        public void GCHandleAllocation_DisposeFreesPin_NotManagedArray()
        {
            int[] data = { 1, 2, 3, 4, 5 };
            var a = new NDArray(data);
            var slice = a.Storage.InternalArray;
            slice.IsReleased.Should().BeFalse();

            a.Dispose();

            slice.IsReleased.Should().BeTrue();
            // Managed array survives — only the pin was released.
            data[0] = 999;
            data[0].Should().Be(999);
        }

        // ----- null-storage safety -------------------------------------------

        [TestMethod]
        public void Dispose_OnTypeOnlyConstructed_NDArray_DoesNotThrow()
        {
            var a = new NDArray(typeof(int));
            a.Storage.InternalArray.Should().BeNull();

            Action act = () => a.Dispose();
            act.Should().NotThrow();
            a.IsDisposed.Should().BeTrue();
        }

        // ----- alloc churn / no memory leak ----------------------------------

        [TestMethod]
        public void AllocAndDispose_TenThousandTimes_NoMemoryAccumulation()
        {
            // After 10k explicit alloc+dispose cycles, working set must not
            // have grown by more than a few MiB (allocator's internal slack).
            using var p = System.Diagnostics.Process.GetCurrentProcess();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 10_000; i++)
            {
                var a = new NDArray(NPTypeCode.Double, new Shape(10_000), fillZeros: false);
                a.Dispose();
            }
            p.Refresh();
            long delta = (p.WorkingSet64 - start) / 1024 / 1024;
            delta.Should().BeLessThan(20,
                "10k cycles should not accumulate >20 MiB (allocator-level slack only)");
        }
    }
}
