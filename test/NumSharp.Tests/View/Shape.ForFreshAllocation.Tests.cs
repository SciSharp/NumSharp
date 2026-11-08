using System;
using System.Linq;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     Pins <see cref="Shape.ForFreshAllocation"/>, the rule every allocating entry point (<c>new NDArray(dtype, shape)</c>,
    ///     <c>np.empty</c>/<c>np.zeros</c>/<c>np.ones</c>/<c>np.full</c> of a <see cref="Shape"/>) applies before it installs a
    ///     shape over a brand-new buffer of exactly <c>size</c> elements, and <see cref="Shape.AsClean"/>, the stricter
    ///     dims-only layout the random samplers allocate their <c>size</c> with (NumPy's <c>np.empty(size)</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The bug this rule closes (catalog open item, 2026-09-24): the allocators installed a VIEW's shape as given, strides
    ///     and offset included, over a buffer that holds only <c>size</c> elements. For <c>x = arange(20)["::4"]</c> the "fresh"
    ///     array put element i at i * 4 of a 5-element buffer, so <c>default_rng(42).integers(0, 100, x.Shape)</c> returned heap
    ///     pointers and <c>np.polyval(p, x)</c> (which runs <c>np.zeros(x.Shape)</c>) crashed the process with an internal CLR
    ///     error.
    ///     </para>
    ///     <para>
    ///     The contract: a layout that already describes a fresh buffer — offset 0, buffer size equal to the element count, and
    ///     strides visiting every element of <c>[0, size)</c> once with no negative step (C, F, or a KEEPORDER permutation) — is
    ///     kept, because NumSharp allocates <c>copy(order='K')</c>, <c>astype</c>, ufunc and <c>concatenate</c> outputs that way;
    ///     every other layout is replaced by the C layout of its dimensions (NumPy's answer for a dims-only request); the result
    ///     is always writeable. NumPy has no counterpart for a shape that carries a layout, so the expected values are the
    ///     memory-safety contract itself, not a NumPy output.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ShapeForFreshAllocationTests
    {
        /// <summary>Asserts a shape is the C layout of <paramref name="dims"/> over a buffer of exactly its size, writeable.</summary>
        /// <param name="s">The shape to check.</param>
        /// <param name="dims">The expected dimensions.</param>
        private static void ShouldBeCleanC(Shape s, params long[] dims)
        {
            s.dimensions.Should().Equal(dims);
            long expect = 1;
            var strides = new long[dims.Length];
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                strides[i] = expect;
                expect *= dims[i];
            }

            s.strides.Should().Equal(strides, "a re-laid-out shape takes the C strides of its dimensions");
            s.offset.Should().Be(0, "a fresh buffer has no parent to be offset into");
            s.BufferSize.Should().Be(s.size, "the buffer holds exactly the elements, not a parent's");
            s.IsContiguous.Should().BeTrue();
            s.IsBroadcasted.Should().BeFalse();
            s.IsWriteable.Should().BeTrue("a fresh buffer is always writeable");
        }

        /// <summary>The shape of a view built by <paramref name="make"/>, the source array disposed (a Shape holds no storage).</summary>
        /// <param name="make">Builds the view.</param>
        /// <returns>The view's shape.</returns>
        private static Shape ViewShape(Func<NDArray> make)
        {
            using var view = make();
            return view.Shape;
        }

        // ---------------------------------------------------------------- layouts that are re-laid out in C order

        [TestMethod]
        public void SteppedView_IsReLaidOutInCOrder()
        {
            // The reported case: arange(20)[::4] has strides 4 over a 20-element parent.
            var s = ViewShape(() => np.arange(20)["::4"]);
            s.strides.Should().Equal(new long[] { 4 });
            ShouldBeCleanC(s.ForFreshAllocation(), 5);
        }

        [TestMethod]
        public void OffsetSteppedView_IsReLaidOutInCOrder()
        {
            var s = ViewShape(() => np.arange(20)["3::4"]);
            s.offset.Should().Be(3);
            ShouldBeCleanC(s.ForFreshAllocation(), 5);
        }

        [TestMethod]
        public void OffsetSteppedView2D_IsReLaidOutInCOrder()
        {
            var s = ViewShape(() => np.zeros(new Shape(7, 9))["1::2, 1::2"]);
            s.offset.Should().NotBe(0);
            ShouldBeCleanC(s.ForFreshAllocation(), 3, 4);
        }

        [TestMethod]
        public void ReversedView_IsReLaidOutWithPositiveStrides()
        {
            // Dense but negative: NumPy never allocates a negative stride, and the view needs an offset to stay in bounds.
            var s = ViewShape(() => np.zeros(new Shape(4, 5))["::-1"]);
            s.strides[0].Should().BeNegative();
            ShouldBeCleanC(s.ForFreshAllocation(), 4, 5);
        }

        [TestMethod]
        public void BroadcastView_IsReLaidOutAndWriteable()
        {
            // Stride 0 aliases elements and the view is read-only; the fresh array must own one slot per element.
            var s = ViewShape(() => np.broadcast_to(np.arange(3.0), new Shape(4, 3)));
            s.IsBroadcasted.Should().BeTrue();
            s.IsWriteable.Should().BeFalse();
            ShouldBeCleanC(s.ForFreshAllocation(), 4, 3);
        }

        [TestMethod]
        public void ScalarBroadcastView_IsReLaidOut()
        {
            var s = ViewShape(() => np.broadcast_to(np.zeros(new Shape(1)), new Shape(5L)));
            ShouldBeCleanC(s.ForFreshAllocation(), 5);
        }

        [TestMethod]
        public void ContiguousSliceWithLargerBuffer_GetsItsOwnBufferSize()
        {
            // C-ordered already, but the shape still reports the parent's 23-element buffer; code that trusts BufferSize on
            // a fresh array (AliasComplexLane sizes its float64 lane by it) would run past a 20-element buffer.
            var s = ViewShape(() => np.zeros(new Shape(23))[":20"].reshape(4, 5));
            s.BufferSize.Should().Be(23);
            ShouldBeCleanC(s.ForFreshAllocation(), 4, 5);
        }

        [TestMethod]
        public void ColumnSlice_IsReLaidOut()
        {
            // (3, 1) with row stride 4: a gap between rows (C-contiguity ignores the extent-1 axis, not the row stride).
            var s = ViewShape(() => np.arange(12).reshape(3, 4)[":, 1:2"]);
            ShouldBeCleanC(s.ForFreshAllocation(), 3, 1);
        }

        [TestMethod]
        public void EmptyViewWithParentOffset_IsReLaidOut()
        {
            // An empty slice of a reversed parent keeps the parent's offset (19) and buffer (20) over no elements.
            var s = ViewShape(() => np.arange(20)["::-1"]["0:0"]);
            s.offset.Should().Be(19);
            var fresh = s.ForFreshAllocation();
            fresh.dimensions.Should().Equal(new long[] { 0 });
            fresh.offset.Should().Be(0);
            fresh.BufferSize.Should().Be(0);
        }

        [TestMethod]
        public void UserBuiltStridesWithGap_AreReLaidOut()
        {
            // The public Shape(dims, strides) constructor accepts any strides: (1, 4) over 6 elements reaches index 9.
            ShouldBeCleanC(new Shape(new long[] { 2, 3 }, new long[] { 1, 4 }).ForFreshAllocation(), 2, 3);
        }

        [TestMethod]
        public void UserBuiltOverlappingStrides_AreReLaidOut()
        {
            // Two axes on one stride visit some elements twice and others never.
            ShouldBeCleanC(new Shape(new long[] { 2, 3 }, new long[] { 1, 1 }).ForFreshAllocation(), 2, 3);
        }

        [TestMethod]
        public void ContiguousLayoutWithAnOffset_IsReLaidOut()
        {
            // A real view at an offset always reports a larger buffer too; this shape claims a buffer of exactly its size,
            // so only the offset test can reject it. Installed as given, element 0 would sit at element 3 of the new buffer.
            var shifted = new Shape(new long[] { 2, 3 }, new long[] { 3, 1 }, 3, 6);
            shifted.IsContiguous.Should().BeTrue();
            ShouldBeCleanC(shifted.ForFreshAllocation(), 2, 3);
        }

        [TestMethod]
        public void ZeroDimensionalViewWithOffset_BecomesTheFreshScalar()
        {
            // Integer indexing narrows its 0-d result today; a 0-d view with an offset is still a valid Shape.
            var view0d = new Shape(Array.Empty<long>(), Array.Empty<long>(), 3, 20);
            var fresh = view0d.ForFreshAllocation();
            fresh.IsScalar.Should().BeTrue();
            fresh.offset.Should().Be(0);
            fresh.BufferSize.Should().Be(1);
        }

        // ---------------------------------------------------------------- layouts that are kept

        [TestMethod]
        public void FreshCShape_IsKeptAsIs()
        {
            var s = new Shape(3, 4);
            var fresh = s.ForFreshAllocation();
            fresh.dimensions.Should().BeSameAs(s.dimensions, "the hot path returns the shape itself, no re-layout");
            fresh.strides.Should().BeSameAs(s.strides);
        }

        [TestMethod]
        public void FreshFShape_KeepsItsLayout()
        {
            // np.empty(order='F') and the KEEPORDER callers allocate F shapes through this rule.
            var s = new Shape(new long[] { 3, 4 }, 'F');
            var fresh = s.ForFreshAllocation();
            fresh.strides.Should().Equal(new long[] { 1, 3 });
            fresh.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void DensePermutation_KeepsItsLayout()
        {
            // copy(order='K') of a 3-D transpose keeps the transposed stride order (neither C nor F).
            var perm = ViewShape(() => np.transpose(np.zeros(new Shape(3, 2, 4)), new[] { 1, 0, 2 })).KeepOrder();
            perm.IsContiguous.Should().BeFalse();
            perm.IsFContiguous.Should().BeFalse();
            var fresh = perm.ForFreshAllocation();
            fresh.strides.Should().Equal(perm.strides);
            fresh.offset.Should().Be(0);
        }

        [TestMethod]
        public void TransposeOfAWholeArray_KeepsItsDenseLayout()
        {
            // Indistinguishable from a fresh F request; memory-safe either way (callers that need NumPy's dims-only C layout,
            // such as the random samplers' size, use Shape.Clean themselves).
            var s = ViewShape(() => np.zeros(new Shape(5, 4)).T);
            var fresh = s.ForFreshAllocation();
            fresh.strides.Should().Equal(new long[] { 1, 4 });
            fresh.BufferSize.Should().Be(20);
        }

        [TestMethod]
        public void ExtentOneAxisStride_DoesNotMatter()
        {
            // NumPy's flags skip an extent-1 axis: its stride is never stepped, so it cannot address outside the buffer.
            var s = new Shape(new long[] { 1, 3 }, new long[] { -7, 1 });
            s.ForFreshAllocation().strides.Should().Equal(new long[] { -7, 1 });
        }

        [TestMethod]
        public void ReadOnlyFreshLayout_BecomesWriteable()
        {
            // mmap_r + 1 allocates under the read-only operand's shape; the new buffer is the caller's to write.
            var ro = new Shape(3, 4).WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            var fresh = ro.ForFreshAllocation();
            fresh.IsWriteable.Should().BeTrue();
            fresh.strides.Should().Equal(new long[] { 4, 1 });
        }

        [TestMethod]
        public void Uninitialized_IsReturnedUnchanged()
        {
            // The allocators raise their own error for default(Shape); the rule must not turn it into something else.
            default(Shape).ForFreshAllocation().IsEmpty.Should().BeTrue();
        }

        // ---------------------------------------------------------------- AsClean: NumPy's np.empty(size) of a dims-only size

        [TestMethod]
        public void AsClean_CLayout_IsReturnedWithoutACopy()
        {
            // The random samplers call AsClean on every sized draw: an already-clean size must cost no allocation.
            var s = new Shape(3, 4);
            var c = s.AsClean();
            c.dimensions.Should().BeSameAs(s.dimensions);
            c.strides.Should().BeSameAs(s.strides);
            Shape.Scalar.AsClean().IsScalar.Should().BeTrue();
        }

        [TestMethod]
        public void AsClean_DenseFView_IsReLaidOutInCOrder()
        {
            // Unlike ForFreshAllocation, a dense F layout is not kept: a sampler fills in memory order, and NumPy's
            // np.empty(size) of the same dimensions is C-ordered, so an F output would hold the draws transposed.
            var s = ViewShape(() => np.zeros(new Shape(5, 4)).T);
            ShouldBeCleanC(s.AsClean(), 4, 5);
        }

        [TestMethod]
        public void AsClean_EmptySteppedView_TakesTheCStridesOfItsDimensions()
        {
            // C-contiguity is set for every empty shape, so only the exact stride comparison sees the view's (0, 2).
            var s = ViewShape(() => np.zeros(new Shape(0, 6))[":, ::2"]);
            s.IsContiguous.Should().BeTrue();
            s.strides.Should().Equal(new long[] { 0, 2 });
            s.AsClean().strides.Should().Equal(new long[] { 3, 1 });
        }

        [TestMethod]
        public void AsClean_StrayStrideOnAnExtentOneAxis_IsReLaidOut()
        {
            // C-contiguity skips the extent-1 axis; NumPy's np.empty((1, 3)) reports the C strides (3, 1).
            ShouldBeCleanC(new Shape(new long[] { 1, 3 }, new long[] { -7, 1 }).AsClean(), 1, 3);
        }

        [TestMethod]
        public void AsClean_ViewBaggage_IsReLaidOut()
        {
            ShouldBeCleanC(ViewShape(() => np.arange(20)["::4"]).AsClean(), 5);
            ShouldBeCleanC(ViewShape(() => np.zeros(new Shape(23))[":20"].reshape(4, 5)).AsClean(), 4, 5);
            ShouldBeCleanC(new Shape(new long[] { 2, 3 }, new long[] { 3, 1 }, 3, 6).AsClean(), 2, 3);
            ShouldBeCleanC(ViewShape(() => np.broadcast_to(np.arange(3.0), new Shape(4, 3))).AsClean(), 4, 3);
            new Shape(Array.Empty<long>(), Array.Empty<long>(), 3, 20).AsClean().offset.Should().Be(0);
        }

        [TestMethod]
        public void AsClean_Uninitialized_IsReturnedUnchanged()
        {
            // size=None: the samplers branch on it before allocating, so AsClean must not throw on it.
            default(Shape).AsClean().IsEmpty.Should().BeTrue();
        }

        [TestMethod]
        public void EveryLayout_KeepsItsDimensionsAndElementCount()
        {
            Func<NDArray>[] views =
            {
                () => np.arange(60)["::3"], () => np.arange(60).reshape(3, 4, 5)[":, ::2, 1:4"],
                () => np.arange(60).reshape(3, 4, 5)["::-1, :, ::-2"], () => np.arange(60).reshape(3, 4, 5).T,
                () => np.broadcast_to(np.arange(5), new Shape(3, 4, 5)), () => np.arange(60).reshape(3, 4, 5)["1"],
                () => np.transpose(np.arange(60).reshape(3, 4, 5), new[] { 2, 0, 1 }),
            };
            foreach (var make in views)
            {
                var s = ViewShape(make);
                var fresh = s.ForFreshAllocation();
                fresh.dimensions.Should().Equal(s.dimensions);
                fresh.size.Should().Be(s.size);
                fresh.offset.Should().Be(0);
                fresh.BufferSize.Should().Be(fresh.size);
                fresh.IsWriteable.Should().BeTrue();
                fresh.IsBroadcasted.Should().BeFalse();
                // Every element address lies in [0, size) and no two elements share one.
                var seen = new bool[fresh.size];
                foreach (var flat in Enumerable.Range(0, (int)fresh.size))
                {
                    long off = fresh.GetOffset(fresh.GetCoordinates(flat));
                    off.Should().BeInRange(0, fresh.size - 1);
                    seen[off].Should().BeFalse();
                    seen[off] = true;
                }
            }
        }
    }
}
