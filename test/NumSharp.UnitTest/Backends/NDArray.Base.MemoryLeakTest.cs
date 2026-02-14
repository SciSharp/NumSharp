using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    /// Memory leak tests for the .base property implementation.
    /// These tests verify that:
    /// 1. Views don't leak when original is collected
    /// 2. Originals aren't collected while views exist
    /// 3. Circular reference scenarios are handled
    /// </summary>
    public class NDArray_Base_MemoryLeakTest
    {
        #region Memory Pressure Tests

        /// <summary>
        /// Create many views, verify they don't leak when original lives.
        /// </summary>
        [Test]
        public void MemoryLeak_ManyViews_NoLeak()
        {
            var original = np.arange(1000);
            var views = new List<NDArray>();

            // Create 1000 views
            for (int i = 0; i < 1000; i++)
            {
                views.Add(original[$"{i % 500}:{(i % 500) + 100}"]);
            }

            // Force GC
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // All views should still be valid
            foreach (var view in views)
            {
                view.@base.Should().NotBeNull();
                view.@base!.Storage.Should().BeSameAs(original.Storage);
            }
        }

        /// <summary>
        /// Stress test: many nested views don't leak.
        /// </summary>
        [Test]
        public void MemoryLeak_DeepNesting_NoLeak()
        {
            var original = np.arange(10000);
            NDArray current = original;

            // Create 100 nested views
            for (int i = 0; i < 100; i++)
            {
                current = current["1:-1"];
                if (current.size < 2) break;
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Deepest view should chain to original
            current.@base!.Storage.Should().BeSameAs(original.Storage);
        }

        /// <summary>
        /// Multiple independent chains don't interfere.
        /// </summary>
        [Test]
        public void MemoryLeak_IndependentChains_NoInterference()
        {
            var chains = new List<(NDArray original, List<NDArray> views)>();

            // Create 10 independent chains
            for (int c = 0; c < 10; c++)
            {
                var original = np.arange(100);
                var views = new List<NDArray>();

                NDArray current = original;
                for (int i = 0; i < 10; i++)
                {
                    current = current["1:-1"];
                    views.Add(current);
                }

                chains.Add((original, views));
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Each chain should be independent
            foreach (var (original, views) in chains)
            {
                foreach (var view in views)
                {
                    view.@base!.Storage.Should().BeSameAs(original.Storage);
                }
            }
        }

        #endregion

        #region Lifecycle Tests

        /// <summary>
        /// View survives when original reference is dropped (but memory is kept alive).
        /// </summary>
        [Test]
        public void Lifecycle_ViewSurvivesDroppedOriginalReference()
        {
            var view = CreateViewAndDropOriginal();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // View should still work
            view.size.Should().Be(5);
            view.GetInt32(0).Should().Be(2);

            // And base should still be accessible (points to storage that's kept alive)
            view.@base.Should().NotBeNull();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private NDArray CreateViewAndDropOriginal()
        {
            var original = np.arange(10);
            return original["2:7"];
        }

        /// <summary>
        /// Multiple views survive when original reference is dropped.
        /// </summary>
        [Test]
        public void Lifecycle_MultipleViewsSurvive()
        {
            var views = CreateMultipleViewsAndDropOriginal();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // All views should still work
            foreach (var view in views)
            {
                view.@base.Should().NotBeNull();
            }

            // All views should share the same base storage
            var firstBase = views[0].@base!.Storage;
            foreach (var view in views)
            {
                view.@base!.Storage.Should().BeSameAs(firstBase);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private List<NDArray> CreateMultipleViewsAndDropOriginal()
        {
            var original = np.arange(100);
            return new List<NDArray>
            {
                original["0:10"],
                original["10:20"],
                original["20:30"],
                original.reshape(10, 10),
                original.T.reshape(100)
            };
        }

        #endregion

        #region Large Array Tests

        /// <summary>
        /// Large array views work correctly.
        /// </summary>
        [Test]
        public void LargeArray_ViewsWork()
        {
            // 10 million elements
            var large = np.arange(10_000_000);

            // Create multiple views
            var view1 = large["0:1000000"];
            var view2 = large["5000000:6000000"];
            var view3 = large.reshape(1000, 10000);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // All should chain to original
            view1.@base!.Storage.Should().BeSameAs(large.Storage);
            view2.@base!.Storage.Should().BeSameAs(large.Storage);
            view3.@base!.Storage.Should().BeSameAs(large.Storage);

            // Values should be correct
            view1.GetInt32(0).Should().Be(0);
            view2.GetInt32(0).Should().Be(5_000_000);
        }

        #endregion

        #region Mixed Operations Tests

        /// <summary>
        /// Mix of copies and views works correctly.
        /// </summary>
        [Test]
        public void MixedOperations_CopiesAndViews()
        {
            var a = np.arange(100);
            var b = a["10:90"];          // view of a
            var c = b.copy();            // copy, owns data
            var d = c["10:70"];          // view of c
            var e = d.reshape(6, 10);    // view of c
            var f = np.copy(e);          // copy, owns data
            var g = f.T;                 // view of f

            // b chains to a
            b.@base!.Storage.Should().BeSameAs(a.Storage);

            // c owns data
            c.@base.Should().BeNull();

            // d, e chain to c
            d.@base!.Storage.Should().BeSameAs(c.Storage);
            e.@base!.Storage.Should().BeSameAs(c.Storage);

            // f owns data
            f.@base.Should().BeNull();

            // g chains to f
            g.@base!.Storage.Should().BeSameAs(f.Storage);
        }

        /// <summary>
        /// Broadcast followed by slice.
        /// Note: In NumSharp, slicing a broadcast array materializes the data (copies it),
        /// so the slice does NOT chain to the original. This differs from NumPy.
        /// </summary>
        [Test]
        [Misaligned]
        public void MixedOperations_BroadcastThenSlice()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(5, 3));

            // b chains to a
            b.@base!.Storage.Should().BeSameAs(a.Storage);

            // Slicing a broadcast materializes data in NumSharp (copies)
            // The materialized data has its own base chain, not to 'a'
            var c = b["1:4, :"];
            // c has a base (it's a view of the materialized data) but not chained to a
            c.@base.Should().NotBeNull();
        }

        #endregion

        #region Finalization Safety Tests

        /// <summary>
        /// Verify safe behavior when array is collected while view exists.
        /// This test creates a scenario where the original array's CLR object
        /// is collected but the view keeps the data alive.
        /// </summary>
        [Test]
        public void Finalization_OriginalCollected_ViewStillWorks()
        {
            WeakReference<NDArray>? weakOriginal;
            NDArray view;

            CreateAndGetReferences(out weakOriginal, out view);

            // Force collection of original CLR object
            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }

            // Original CLR object should be collected
            // (but not its underlying data due to shared storage)
            weakOriginal.TryGetTarget(out _).Should().BeFalse();

            // View should still work
            view.size.Should().Be(5);
            view.GetInt32(0).Should().Be(2);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void CreateAndGetReferences(out WeakReference<NDArray> weakOriginal, out NDArray view)
        {
            var original = np.arange(10);
            weakOriginal = new WeakReference<NDArray>(original);
            view = original["2:7"];
        }

        #endregion

        #region Concurrent Access Tests

        /// <summary>
        /// Views created from multiple threads all chain correctly.
        /// </summary>
        [Test]
        public void Concurrent_MultipleThreads_ViewsChainCorrectly()
        {
            var original = np.arange(1000);
            var views = new System.Collections.Concurrent.ConcurrentBag<NDArray>();

            // Create views from multiple threads
            System.Threading.Tasks.Parallel.For(0, 100, i =>
            {
                var view = original[$"{i}:{i + 100}"];
                views.Add(view);
            });

            // All views should chain to original
            foreach (var view in views)
            {
                view.@base!.Storage.Should().BeSameAs(original.Storage);
            }
        }

        #endregion
    }
}
