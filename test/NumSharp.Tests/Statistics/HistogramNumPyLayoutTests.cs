using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Statistics
{
    /// <summary>
    ///     The histogram family's result objects and bin specifications, as NumPy 2.4.2 returns and checks them: the caller's
    ///     edges array comes back as that same object, <c>histogramdd</c>'s H is a view of the padded counts, and a
    ///     per-dimension scalar bin meets <c>bins[i] &lt; 1</c> and <c>operator.index</c>. Probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class HistogramNumPyLayoutTests
    {
        private static NDArray X() => np.array(new[] { 1.0, 2.0, 3.0, 2.5 });
        private static NDArray Y() => np.array(new[] { 1.0, 2.0, 3.0, 0.5 });

        [TestMethod]
        public void Histogramdd_H_IsAViewOfThePaddedCounts()
        {
            // H, e = np.histogramdd([x, y], bins=2): H [[1, 0], [1, 2]], strides (32, 8), not C-contiguous, H.base (4, 4).
            var r = np.histogramdd(new[] { X(), Y() }, 2);
            r.H.ToArray<double>().Should().Equal(1.0, 0.0, 1.0, 2.0);
            r.H.strides.Should().Equal(new long[] { 32, 8 });
            r.H.flags.c_contiguous.Should().BeFalse();
            r.H.flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Histogramdd_Density_IsAFreshArray()
        {
            // The density branch divides into fresh arrays, as NumPy's `hist / dedges[i].reshape(shape)` does.
            var r = np.histogramdd(new[] { X(), Y() }, new[] { 2, 3 }, density: true);
            r.H.flags.c_contiguous.Should().BeTrue();
            r.H.flags.owndata.Should().BeTrue();
        }

        [TestMethod]
        public void ExplicitEdges_ComeBackAsTheCallersArray()
        {
            // np.histogram(x, bins=edges)[1] is edges; np.histogram_bin_edges(x, bins=edges) is edges;
            // np.histogram2d(x, y, bins=edges) returns edges for both axes.
            var edges = np.array(new[] { 0.0, 1, 2, 3 });
            ReferenceEquals(np.histogram(X(), edges).bin_edges, edges).Should().BeTrue();
            ReferenceEquals(np.histogram_bin_edges(X(), edges), edges).Should().BeTrue();
            var r2 = np.histogram2d(X(), Y(), edges);
            ReferenceEquals(r2.xedges, edges).Should().BeTrue();
            ReferenceEquals(r2.yedges, edges).Should().BeTrue();
        }

        [TestMethod]
        public void Histogramdd_ScalarBinSpecs_FollowPythonsRules()
        {
            // bins=[2.5, 3] -> TypeError (operator.index of a float); bins=[True, 2] -> a Python bool IS an int (1 bin).
            Action fl = () => np.histogramdd(new[] { X(), Y() }, new object[] { 2.5, 3 });
            fl.Should().Throw<TypeError>().WithMessage("`bins[0]` must be an integer, when a scalar");
            var r = np.histogramdd(new[] { X(), Y() }, new object[] { true, 2 });
            r.H.shape.Should().Equal(new long[] { 1, 2 });
            r.H.ToArray<double>().Should().Equal(2.0, 2.0);
            // A 0-d text bin meets `bins[i] < 1` first: 'less' has no loop for a text array and a Python int.
            Action text = () => np.histogramdd(new[] { X(), Y() }, new object[] { np.array("2"), 3 });
            text.Should().Throw<TypeError>().WithMessage(
                "ufunc 'less' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes._PyLongDType'>) -> None");
        }

        [TestMethod]
        public void Histogram2d_OneElementArrayBins_IsTheDimensionError()
        {
            // np.histogram2d(x, y, bins=np.array([3])): N = 1, so histogramdd gets a 1-entry spec for 2 dimensions.
            Action act = () => np.histogram2d(X(), Y(), np.array(new long[] { 3 }));
            act.Should().Throw<ValueError>().WithMessage("The dimension of bins must be equal to the dimension of the sample x.");
        }
    }
}
