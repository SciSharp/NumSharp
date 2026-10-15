using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Reference accounting and fault injection. The base class gates the bridge's own counters; these tests go one
    ///     level down, to what those counters stand for — NumSharp's ARC reference on the exported buffer
    ///     (<c>IArraySlice.IsUniquelyReferenced</c>, and NumPy's own <c>resize(refcheck)</c> that reads it), and the
    ///     pinned <c>GCHandle</c> behind an import lease (as the GC itself counts pinned objects). They also drive every
    ///     REFUSAL path — the faults a caller can inject (a wrong element type, a negative stride, an oversized copy, a
    ///     read-only span request) — and check each one leaves nothing behind.
    /// </summary>
    [TestClass]
    public class ArcAccountingTests : TensorsTestBase
    {
        /// <summary>Is the array's buffer referenced by the array alone (no view, no export handle holds it)?</summary>
        /// <param name="nd">An array that owns its buffer.</param>
        /// <returns><c>true</c> when nothing else holds a reference.</returns>
        private static bool Unique(NDArray nd) => nd.Storage.InternalArray.IsUniquelyReferenced;

        /// <summary>
        ///     Every refused export leaves the buffer exactly as referenced as before: a wrong element type (both
        ///     verbs), a negative-stride view, an oversized ToTensor and a read-only handle's <c>.Span</c> request all
        ///     throw their documented exception, the live-export count is unchanged, and once the test's own views are
        ///     disposed the owner is referenced by itself alone.
        /// </summary>
        [TestMethod]
        public void RefusedExports_LeaveNoReferenceBehind()
        {
            using NDArray owner = np.arange(12.0);
            int exports = NDArrayTensorsInterop.LiveExports;
            Unique(owner).Should().BeTrue();

            new Action(() => owner.AsTensorSpan<float>()).Should().Throw<ArgumentException>();
            new Action(() => owner.ToTensor<float>()).Should().Throw<ArgumentException>();
            Unique(owner).Should().BeTrue("a wrong-type refusal takes no reference");

            using (NDArray reversed = owner["::-1"])
                new Action(() => reversed.AsTensorSpan<double>()).Should().Throw<InvalidOperationException>();
            Unique(owner).Should().BeTrue("a negative-stride refusal takes no reference");

            using (NDArray first = owner["0"])
            using (NDArray huge = np.broadcast_to(first, new Shape(1L << 40)))
                new Action(() => huge.ToTensor<double>()).Should().Throw<NotSupportedException>();
            Unique(owner).Should().BeTrue("an oversize refusal takes no reference");

            using (NDArray window = owner["2:6"])
            {
                window.setflags(write: false);
                using var h = window.AsTensorSpan<double>();
                new Action(() => { _ = h.Span; }).Should().Throw<InvalidOperationException>();
            }
            Unique(owner).Should().BeTrue("a refused Span request does not keep the handle's reference");
            NDArrayTensorsInterop.LiveExports.Should().Be(exports);
        }

        /// <summary>
        ///     An export handle holds exactly ONE reference: the owner is shared while the handle lives, referenced by
        ///     itself alone once it is disposed, and further disposes neither release again (the owner would lose its
        ///     own reference) nor free the buffer. ToTensor, contiguous or strided, holds none once it returns.
        /// </summary>
        [TestMethod]
        public void ExportHandle_HoldsOneReference_ReleasedExactlyOnce()
        {
            using NDArray owner = np.arange(12.0);
            var h = owner.AsTensorSpan<double>();
            Unique(owner).Should().BeFalse("the handle holds a reference");
            h.Dispose();
            Unique(owner).Should().BeTrue();
            h.Dispose();
            h.Dispose();
            Unique(owner).Should().BeTrue("repeated disposes release nothing more");
            owner.Storage.InternalArray.IsReleased.Should().BeFalse("the owner's own reference must survive");
            owner.GetDouble(11).Should().Be(11.0);

            owner.ToTensor<double>();
            Unique(owner).Should().BeTrue("a contiguous ToTensor keeps no reference");
            using (NDArray strided = owner["::3"])
                strided.ToTensor<double>();
            Unique(owner).Should().BeTrue("a strided ToTensor (which densifies first) keeps no reference");
        }

        /// <summary>
        ///     NumPy's own reference check sees the export: <c>ndarray.resize(refcheck: true)</c> refuses to resize an
        ///     array while an export handle holds its buffer (resizing would free memory a live span points at) and
        ///     succeeds once the handle is disposed.
        /// </summary>
        [TestMethod]
        public void Resize_WithRefcheck_RefusesWhileExported_AndSucceedsAfter()
        {
            using NDArray owner = np.arange(6.0);
            using (var h = owner.AsTensorSpan<double>())
            {
                new Action(() => owner.resize(new Shape(12), refcheck: true))
                    .Should().Throw<IncorrectShapeException>().WithMessage("*referenced*");
                h.ReadOnlySpan[new nint[] { 5 }].Should().Be(5.0, "the refused resize left the exported memory alone");
            }
            owner.resize(new Shape(12), refcheck: true);
            owner.size.Should().Be(12);
        }

        /// <summary>
        ///     An import with nothing to share opens no lease: an empty tensor and the rank-0 <c>Tensor&lt;T&gt;.Empty</c>
        ///     come back as fresh empty arrays and the live-import count does not move.
        /// </summary>
        [TestMethod]
        public void Import_OfAnEmptyTensor_OpensNoLease()
        {
            int before = NDArrayTensorsInterop.LiveImports;
            using (NDArray e = Tensor.Create(Array.Empty<double>(), new nint[] { 2, 0, 3 }).AsNDArray())
                NDArrayTensorsInterop.LiveImports.Should().Be(before);
            using (NDArray z = Tensor<double>.Empty.AsNDArray())
                NDArrayTensorsInterop.LiveImports.Should().Be(before);
        }

        /// <summary>
        ///     One import lease is shared by the imported view and every view derived from it, and released exactly once,
        ///     by whichever of them is disposed LAST — checked for all 24 disposal orders of four sharing views.
        /// </summary>
        [TestMethod]
        public void ImportLease_ReleasedOnceByTheLastView_InEveryDisposalOrder()
        {
            foreach (int[] order in Permutations(4))
            {
                int before = NDArrayTensorsInterop.LiveImports;
                Tensor<double> t = Tensor.Create(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), new nint[] { 3, 4 });
                NDArray view = t.AsNDArray();
                NDArray[] sharing = { view, view["1:"], view.T, view["::2, 1::2"] };
                for (int k = 0; k < order.Length; k++)
                {
                    sharing[order[k]].Dispose();
                    int expected = k == order.Length - 1 ? before : before + 1;
                    NDArrayTensorsInterop.LiveImports.Should().Be(expected, $"order {string.Join(",", order)}, after disposing {k + 1} view(s)");
                }
            }
        }

        /// <summary>
        ///     The lease's pin is real and so is its release, as the GC itself counts pinned objects: with 200 live
        ///     import views the GC sees about 200 more pinned objects; once the views are disposed it sees them gone.
        ///     (A lease that kept its <c>GCHandle</c> would leave each tensor's array pinned forever, fragmenting the heap.)
        /// </summary>
        [TestMethod]
        public void ImportLeases_PinWhileLive_AndUnpinOnRelease_AsTheGcCountsThem()
        {
            const int n = 200;
            long baseline = PinnedObjectsAfterFullGc();

            var tensors = Enumerable.Range(0, n).Select(i => Tensor.Create(new double[] { i, i + 1 }, new nint[] { 2 })).ToArray();
            NDArray[] views = tensors.Select(t => t.AsNDArray()).ToArray();
            long during = PinnedObjectsAfterFullGc();

            foreach (NDArray v in views)
                v.Dispose();
            long after = PinnedObjectsAfterFullGc();
            GC.KeepAlive(tensors);

            (during - baseline).Should().BeGreaterThanOrEqualTo(n - 10, "every live lease pins its tensor's backing array");
            (after - baseline).Should().BeLessThan(10, "every released lease unpinned its array");
        }

        /// <summary>
        ///     An import view that is never disposed does not leak its lease: once the views become unreachable, the
        ///     GC's finalizers release it (the live-import count returns to its baseline). The undisposed export handle
        ///     has the same test in <see cref="LifetimeTests"/>.
        /// </summary>
        [TestMethod]
        public void UndisposedImportViews_AreReleasedByTheGc()
        {
            int before = NDArrayTensorsInterop.LiveImports;
            MakeAndDropImportViews(16);
            WaitFor(() => NDArrayTensorsInterop.LiveImports == before, 10_000)
                .Should().BeTrue("the finalizers must release the leases of views nobody disposed");
        }

        /// <summary>Imports <paramref name="count"/> tensors (each with a derived view) and drops every reference undisposed.</summary>
        /// <param name="count">How many tensors to import.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MakeAndDropImportViews(int count)
        {
            for (int i = 0; i < count; i++)
            {
                Tensor<double> t = Tensor.Create(new double[] { 1, 2, 3, 4 }, new nint[] { 2, 2 });
                NDArray view = t.AsNDArray();
                NDArray derived = view.T;
                GC.KeepAlive(derived);
            }
        }

        /// <summary>The number of objects the GC found pinned during a forced, blocking full collection.</summary>
        /// <returns>The pinned-object count of that collection.</returns>
        private static long PinnedObjectsAfterFullGc()
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
            return GC.GetGCMemoryInfo(GCKind.FullBlocking).PinnedObjectsCount;
        }

        /// <summary>Every permutation of 0..n-1.</summary>
        /// <param name="n">The number of elements.</param>
        /// <returns>The permutations.</returns>
        private static IEnumerable<int[]> Permutations(int n)
        {
            if (n == 1)
            {
                yield return new[] { 0 };
                yield break;
            }
            foreach (int[] shorter in Permutations(n - 1))
                for (int at = 0; at < n; at++)
                {
                    var p = new List<int>(shorter);
                    p.Insert(at, n - 1);
                    yield return p.ToArray();
                }
        }
    }
}
