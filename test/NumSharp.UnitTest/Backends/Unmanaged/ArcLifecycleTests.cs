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

        // ====================================================================
        //   Lessons learned — view chain, dtype coverage, creation paths,
        //   ergonomic patterns, cross-thread behavior.
        // ====================================================================

        // ----- view chains (slice of slice of slice) -------------------------

        [TestMethod]
        public void ViewChain_ThreeLevels_AllShareRefCount()
        {
            // Lesson: every view-creating operation MUST bump refcount via
            // InitializeArc. A three-level chain produces refCount = 4 (owner + 3 views).
            var a = np.arange(100);
            var slice = a.Storage.InternalArray;

            var v1 = a["10:90"];
            var v2 = v1["10:70"];
            var v3 = v2["10:50"];
            GetRefCount(slice).Should().Be(4);

            v1.Dispose();
            v2.Dispose();
            v3.Dispose();
            GetRefCount(slice).Should().Be(1);
            slice.IsReleased.Should().BeFalse();

            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void ViewChain_DisposeOrder_DoesNotMatter()
        {
            // Reverse the dispose order — refcount must still reach 0.
            var a = np.arange(100);
            var slice = a.Storage.InternalArray;
            var v1 = a["10:90"];
            var v2 = v1["10:70"];
            var v3 = v2["10:50"];

            // Dispose owner first, then views in arbitrary order
            a.Dispose();
            slice.IsReleased.Should().BeFalse("3 views still alive");

            v2.Dispose();
            slice.IsReleased.Should().BeFalse("2 views still alive");

            v3.Dispose();
            slice.IsReleased.Should().BeFalse("1 view still alive");

            v1.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        // ----- reshape: contig (view) vs non-contig (copy) -------------------

        [TestMethod]
        public void ReshapeContiguous_SharesRefCount()
        {
            // Contig reshape returns a VIEW — must share refcount.
            var a = np.arange(12);
            var slice = a.Storage.InternalArray;

            var b = a.reshape(3, 4);
            GetRefCount(slice).Should().Be(2, "reshape view must add a ref");

            a.Dispose();
            slice.IsReleased.Should().BeFalse("b still alive");

            b.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void ReshapeNonContiguous_AllocatesNewOwningBuffer()
        {
            // Non-contig reshape (e.g. of a transposed array) returns a COPY —
            // the copy has its own MemoryBlock, independent of the parent's
            // MemoryBlock. We verify the semantic, not strict refcount math
            // (reshape's non-contig path creates an unreachable intermediate
            // NDArray that Debug builds keep alive until method exit, so the
            // returned NDArray's refcount = 2, not 1).
            var raw = np.arange(12);
            var a = raw.reshape(3, 4);
            var b = a.T;                  // 4x3 transposed view
            var b_slice = b.Storage.InternalArray;

            // Reshaping the transposed view forces a materialized copy
            var c = b.reshape(12);
            var c_slice = c.Storage.InternalArray;

            object.ReferenceEquals(c_slice, b_slice).Should().BeFalse(
                "non-contig reshape must allocate a new owning buffer");

            // The semantic that matters: disposing parents must NOT affect
            // the copy. Buffers are independent.
            raw.Dispose();
            a.Dispose();
            b.Dispose();
            c_slice.IsReleased.Should().BeFalse("copy is independent of parent chain");
            b_slice.IsReleased.Should().BeTrue("parent chain released — sources gone");

            c.Dispose();
            // c_slice's refcount may still be > 0 from the orphan intermediate
            // pinned in Debug builds; force-drain to verify the copy buffer is
            // eventually reclaimable.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            c_slice.IsReleased.Should().BeTrue("copy reclaimed after Dispose + GC");
        }

        // ----- transpose / negative-stride / strided views -------------------

        [TestMethod]
        public void Transpose_ParticipatesInRefCount()
        {
            // Hold every NDArray explicitly. Chained creation
            // (np.arange(N).reshape(...)) pins the intermediate in Debug
            // builds and produces a non-zero "initial" we can't drive to -1.
            var raw = np.arange(12);
            var a = raw.reshape(3, 4);
            var slice = a.Storage.InternalArray;
            var initial = GetRefCount(slice);

            var t = a.T;
            GetRefCount(slice).Should().Be(initial + 1, "transpose view must AddRef");

            t.Dispose();
            GetRefCount(slice).Should().Be(initial);

            a.Dispose();
            raw.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void NegativeStride_ParticipatesInRefCount()
        {
            var a = np.arange(10);
            var slice = a.Storage.InternalArray;
            var rev = a["::-1"];
            GetRefCount(slice).Should().Be(2);

            rev.Dispose();
            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        // ----- explicit copy independence ------------------------------------

        [TestMethod]
        public void Copy_AllocatesIndependentBuffer()
        {
            var a = np.arange(100);
            var a_slice = a.Storage.InternalArray;

            var b = a.copy();
            var b_slice = b.Storage.InternalArray;

            object.ReferenceEquals(a_slice, b_slice).Should().BeFalse();
            GetRefCount(a_slice).Should().Be(1);
            GetRefCount(b_slice).Should().Be(1);

            a.Dispose();
            a_slice.IsReleased.Should().BeTrue();
            b_slice.IsReleased.Should().BeFalse("copy is independent");

            b.Dispose();
            b_slice.IsReleased.Should().BeTrue();
        }

        // ----- creation-path coverage ----------------------------------------

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void DtypeCoverage_AllParticipateInRefCount(NPTypeCode tc)
        {
            var a = new NDArray(tc, new Shape(100), fillZeros: true);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);
            slice.IsReleased.Should().BeFalse();

            a.Dispose();

            slice.IsReleased.Should().BeTrue();
            GetRefCount(slice).Should().Be(-1);
        }

        [TestMethod]
        public void NpZeros_OwnsRefCount()
        {
            var a = np.zeros(new Shape(50), NPTypeCode.Double);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);
            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void NpOnes_OwnsRefCount()
        {
            var a = np.ones(new Shape(50), NPTypeCode.Double);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);
            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void NpEmpty_OwnsRefCount()
        {
            var a = np.empty(new Shape(50), NPTypeCode.Double);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);
            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        [TestMethod]
        public void NpArange_OwnsRefCount()
        {
            var a = np.arange(50);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);
            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        // ----- ergonomic patterns --------------------------------------------

        [TestMethod]
        public void UsingStatement_ReleasesAtScopeExit()
        {
            // Lesson: NDArray's IDisposable lets users opt into deterministic
            // release via `using` — without forcing it on every call site.
            IArraySlice captured;
            using (var a = np.arange(1000))
            {
                captured = a.Storage.InternalArray;
                GetRefCount(captured).Should().Be(1);
                captured.IsReleased.Should().BeFalse();
            }
            captured.IsReleased.Should().BeTrue("`using` should atomically free at scope exit");
        }

        [TestMethod]
        public void UsingStatement_Nested_FreesInReverseOrder()
        {
            IArraySlice innerSlice = null!;
            IArraySlice outerSlice = null!;
            using (var outer = np.arange(100))
            {
                outerSlice = outer.Storage.InternalArray;
                using (var inner = np.arange(50))
                {
                    innerSlice = inner.Storage.InternalArray;
                    outerSlice.IsReleased.Should().BeFalse();
                    innerSlice.IsReleased.Should().BeFalse();
                }
                innerSlice.IsReleased.Should().BeTrue("inner releases first");
                outerSlice.IsReleased.Should().BeFalse("outer still alive");
            }
            outerSlice.IsReleased.Should().BeTrue();
        }

        // ----- cross-thread disposal -----------------------------------------

        [TestMethod]
        public void CrossThread_Dispose_Works()
        {
            // Allocate on one thread, dispose on another.
            // ARC is thread-safe by construction so this must Just Work.
            NDArray a = null!;
            IArraySlice slice = null!;
            var allocThread = new Thread(() =>
            {
                a = np.arange(1000);
                slice = a.Storage.InternalArray;
            });
            allocThread.Start();
            allocThread.Join();

            GetRefCount(slice).Should().Be(1);

            var disposeThread = new Thread(() => a.Dispose());
            disposeThread.Start();
            disposeThread.Join();

            slice.IsReleased.Should().BeTrue();
            GetRefCount(slice).Should().Be(-1);
        }

        [TestMethod]
        public unsafe void CrossThread_View_ReadsAndDisposes()
        {
            // Use raw address indexing — view[i] would build per-element
            // orphan NDArrays that bump refcount and prevent reaching 0.
            // Use long* because np.arange returns Int64 by default.
            var a = np.arange(100);
            var slice = a.Storage.InternalArray;
            var view = a["10:50"];

            long sum = 0;
            var t = new Thread(() =>
            {
                long* ptr = (long*)view.Storage.Address;
                for (int i = 0; i < view.shape[0]; i++)
                    sum += ptr[i];
                view.Dispose();
            });
            t.Start();
            t.Join();

            sum.Should().Be(10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 +
                            20 + 21 + 22 + 23 + 24 + 25 + 26 + 27 + 28 + 29 +
                            30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 +
                            40 + 41 + 42 + 43 + 44 + 45 + 46 + 47 + 48 + 49);
            slice.IsReleased.Should().BeFalse("parent still alive");

            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        // ----- refcount accumulation -----------------------------------------

        [TestMethod]
        public void ManyAddRefs_AccumulateCorrectly()
        {
            var a = np.arange(10);
            var slice = a.Storage.InternalArray;

            for (int i = 0; i < 1000; i++)
                slice.TryAddRef().Should().BeTrue();
            GetRefCount(slice).Should().Be(1001);

            for (int i = 0; i < 1000; i++)
                slice.Release();
            GetRefCount(slice).Should().Be(1);
            slice.IsReleased.Should().BeFalse();

            a.Dispose();
            slice.IsReleased.Should().BeTrue();
        }

        // ----- weak reference proves GC eligibility --------------------------

        [TestMethod]
        public void DisposedNDArray_BecomesCollectable()
        {
            // After Dispose, the NDArray itself is also a regular GC candidate.
            // GC.SuppressFinalize was called so it doesn't even enter the
            // finalizer queue.
            static WeakReference MakeAndDispose()
            {
                var a = np.arange(1000);
                a.Dispose();
                return new WeakReference(a);
            }

            var w = MakeAndDispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            w.IsAlive.Should().BeFalse();
        }

        [TestMethod]
        public void UndisposedNDArray_BecomesCollectable_ViaFinalizer()
        {
            // Without explicit Dispose, the NDArray still becomes collectable
            // — it just takes one extra GC cycle (finalizer + reclaim).
            static (WeakReference weak, IArraySlice slice) MakeAndDrop()
            {
                var a = new NDArray(NPTypeCode.Double, new Shape(1000), fillZeros: false);
                return (new WeakReference(a), a.Storage.InternalArray);
            }

            var (w, slice) = MakeAndDrop();

            // Two passes: first GC.Collect surfaces it to finalizer; second
            // GC reclaims after finalizer ran.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            w.IsAlive.Should().BeFalse("NDArray should be reclaimed");
            slice.IsReleased.Should().BeTrue("finalizer must release the buffer");
        }

        // ----- multi-NDArray same Storage (manual sharing) -------------------

        [TestMethod]
        public void MultipleNDArrays_SameStorageReference_EachBumpsRefCount()
        {
            // Anti-pattern: two NDArrays manually wrapping the same Storage.
            // This SHOULDN'T be common, but if it happens, each ctor must
            // independently bump refcount so disposing one doesn't break the
            // other.
            var a = np.arange(50);
            var slice = a.Storage.InternalArray;
            GetRefCount(slice).Should().Be(1);

            // Manually create another wrapper around the SAME storage
            var alias = new NDArray(a.Storage);
            GetRefCount(slice).Should().Be(2);

            a.Dispose();
            slice.IsReleased.Should().BeFalse("alias still alive");

            alias.Dispose();
            slice.IsReleased.Should().BeTrue();
        }
    }
}
