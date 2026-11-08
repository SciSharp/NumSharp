using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     NumSharp's integer slice (<see cref="Slice.Index(long)"/>) and an integer's notation (<c>"1"</c>) beside an advanced
    ///     entry are NumPy's integer index: an advanced entry of its own, which consumes its axis and keeps none, joins the
    ///     advanced block's broadcast and decides where the block goes. Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by replaying a 6,000-key NumPy differential of the getter and setter with every integer spelled
    ///     <c>Slice.Index(n)</c> or <c>"n"</c> (P3 batch 8, np-layer dtype plan): the routes and <c>PrepareIndex</c> read
    ///     both as slices of length one, so <c>a[Slice.Index(-1), []]</c> kept an axis ((0, 0) where NumPy gives (0,)), 1,693
    ///     of the 12,000 outcomes differed where the same keys with C# integers differed in 742.
    /// </remarks>
    [TestClass]
    public class integer_slice_entry_numpy_port_tests
    {
        /// <summary>A float64 array of <paramref name="dims"/> holding 0, 1, 2, ... in C order.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The array.</returns>
        private static NDArray Ramp(params long[] dims)
        {
            long size = 1;
            foreach (var d in dims)
                size *= d;
            return np.arange((double)size).reshape(dims);
        }

        private static readonly long[,] Ia2d = { { 0, 1 }, { 1, 0 } };

        /// <summary>The integer consumes its axis and keeps none, in both spellings.</summary>
        [TestMethod]
        public void Get_TheIntegerKeepsNoAxis()
        {
            Ramp(3, 4)[Slice.Index(-1), np.array(new long[0])].shape.Should().Equal(0L);
            Ramp(3, 4)["-1", np.array(new long[0])].shape.Should().Equal(0L);
            Ramp(3, 4)["-1", np.array(new long[] { 0, 2 })].ToArray<double>().Should().Equal(8, 10);
            Ramp(3, 4)[Slice.Index(2), np.array(new long[] { 0, 2 })].ToArray<double>().Should().Equal(8, 10);

            var r = Ramp(3, 4)[np.array(Ia2d), Slice.NewAxis, "0"];
            r.shape.Should().Equal(2L, 2L, 1L);
            r.flatten().ToArray<double>().Should().Equal(0, 4, 4, 0);
        }

        /// <summary>The integer is an advanced entry: separated from another by an ellipsis or a slice, the block goes
        ///     in front.</summary>
        [TestMethod]
        public void Get_TheIntegerIsAnAdvancedEntry()
        {
            var r = Ramp(2, 3, 4)[Slice.Index(0), Slice.Ellipsis, np.array(true)];
            r.shape.Should().Equal(1L, 3L, 4L);
            r.flatten().ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            Ramp(2, 3, 4)["0", Slice.All, np.array(new long[] { 0, 1 })].shape.Should().Equal(2L, 3L);
        }

        /// <summary>A store through the same keys writes what NumPy's writes.</summary>
        [TestMethod]
        public void Store_ThroughAnIntegerSlice()
        {
            var a = Ramp(2, 3, 4, 5);
            a[np.array(Ia2d), Slice.Index(-1)] = np.arange(80.0).reshape(2, 2, 4, 5) + 100.0;
            a.GetDouble(0, 2, 0, 0).Should().Be(160.0);
            a.GetDouble(1, 2, 3, 4).Should().Be(159.0);
            a.GetDouble(0, 0, 0, 0).Should().Be(0.0);
            a.GetDouble(1, 1, 1, 1).Should().Be(86.0);

            var b = Ramp(2, 3, 4);
            b[np.array(new long[,] { { 0 }, { 1 } }), "0", Slice.Index(0)] = np.array(new double[,] { { 50 }, { 60 } });
            b.GetDouble(0, 0, 0).Should().Be(50.0);
            b.GetDouble(1, 0, 0).Should().Be(60.0);
            b.GetDouble(0, 0, 1).Should().Be(1.0);

            var c = Ramp(3, 4);
            c.__setitem__(new object[] { "0", np.array(new long[] { 0, 2 }) }, new object[] { 1.0, 2.0 });
            c["0"].ToArray<double>().Should().Equal(1, 1, 2, 3);
        }

        /// <summary>An integer out of bounds raises NumPy's integer text (not a slice's ArgumentException).</summary>
        [TestMethod]
        public void OutOfBounds_RaisesTheIntegersText()
        {
            Action a = () => _ = Ramp(3, 4)[Slice.Index(5), np.array(new long[] { 0 })];
            a.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");
            Action b = () => _ = np.zeros(new Shape(0, 4))["-1", np.array(new long[,] { { 0 }, { 1 } })];
            b.Should().Throw<IndexError>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
        }
    }
}
