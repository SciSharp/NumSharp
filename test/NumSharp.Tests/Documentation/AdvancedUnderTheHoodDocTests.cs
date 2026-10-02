using System;
using NumSharp;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples in
    ///     <c>docs/website-src/docs/advanced/under-the-hood.md</c> ("Under the hood — internals").
    ///     Each test mirrors one documented snippet and asserts the behaviour the page claims. Values
    ///     were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class AdvancedUnderTheHoodDocTests
    {
        [TestMethod]
        public void Views_AreMetadataOnly_ShareBuffer()
        {
            var a = np.arange(12).reshape(3, 4);
            var t = a.T;                                   // transpose: no data moves
            t.Shape.IsContiguous.Should().BeFalse();
            np.shares_memory(a, t).Should().BeTrue("transpose is a view over the same buffer");
        }

        [TestMethod]
        public void Strides_BytesVsElements()
        {
            var a = np.arange(24).reshape(4, 6).astype(np.float64);
            a.strides.Should().Equal(new long[] { 48, 8 }, "public nd.strides is in BYTES (NumPy parity)");
            a.Shape.Strides.Should().Equal(new long[] { 6, 1 }, "Shape.Strides is in ELEMENTS");
        }

        [TestMethod]
        public void Flags_AreO1_ContiguityTracked()
        {
            var a = np.arange(24).reshape(4, 6).astype(np.float64);
            a.Shape.IsContiguous.Should().BeTrue();
            a["1:3, ::2"].Shape.IsContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void Base_And_SharesMemory()
        {
            var owner = np.arange(10);
            (owner.@base is null).Should().BeTrue("an array that owns its data has a null base");

            var slice = owner["2:5"];
            (slice.@base is not null).Should().BeTrue("a view chains to its owner");
            np.shares_memory(owner, slice).Should().BeTrue();

            var copy = owner.copy();
            (copy.@base is null).Should().BeTrue("a copy owns its data");
        }

        [TestMethod]
        public void BroadcastView_IsReadOnly()
        {
            var bc = np.broadcast_to(np.array(new[] { 1, 2, 3 }), (4, 3));
            bc.Shape.IsBroadcasted.Should().BeTrue();
            bc.Shape.IsWriteable.Should().BeFalse();
            ((Action)(() => bc[0, 0] = 9)).Should().Throw<Exception>("broadcast views are read-only");
        }

        [TestMethod]
        public void Inspection_PublicMetadata()
        {
            var a = np.arange(24).reshape(4, 6).astype(np.float64);
            a.Shape.Dimensions.Should().Equal(new long[] { 4, 6 });
            a.Shape.Strides.Should().Equal(new long[] { 6, 1 });
            a.Shape.Offset.Should().Be(0);
            a.ndim.Should().Be(2);
            a.size.Should().Be(24);
        }
    }
}
