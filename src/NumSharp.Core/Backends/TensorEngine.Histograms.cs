// The functions of numpy/lib/_histograms_impl.py (histogram, histogram_bin_edges, histogramdd) and histogram2d, NumPy's
// wrapper of histogramdd in numpy/lib/_twodim_base_impl.py, as TensorEngine members: NumPy writes them in Python,
// DefaultEngine runs those statements in NumPy's order, and an engine that derives from DefaultEngine can replace one for
// every array bound to it (docs/plans/np-layer-dtype-logic.md, section 1 "Which O", Appendix D).

using System;
using NumSharp.Backends;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.histogram_bin_edges(a, bins=10, range=None, weights=None)</c>: the bin edges
        ///     <see cref="Histogram"/> would use, from <c>_ravel_and_check_weights</c> and <c>_get_bin_edges</c>.
        /// </summary>
        /// <param name="a">The data, a plain array (flattened).</param>
        /// <param name="bins">
        ///     An <see cref="int"/> count (NumPy's default is 10; the facades pass it), an estimator name (<c>auto</c>,
        ///     <c>fd</c>, <c>doane</c>, <c>scott</c>, <c>stone</c>, <c>rice</c>, <c>sturges</c>, <c>sqrt</c>), or an
        ///     <see cref="NDArray"/> (a 0-d count or 1-D monotonic edges, returned as given).
        /// </param>
        /// <param name="range">The outer edges, or null for the data's minimum and maximum.</param>
        /// <param name="weights">Weights of <paramref name="a"/>'s shape (shape-checked only), or null.</param>
        /// <returns>The bin edges (<c>nbins + 1</c> of them); an explicit edge array comes back as that array.</returns>
        /// <exception cref="ValueError">A weights shape that differs from <paramref name="a"/>'s; an unknown estimator; a
        ///     count below 1; edges that are not 1-D or decrease (text edges by the string comparison); an invalid range;
        ///     too many bins for the range.</exception>
        /// <exception cref="TypeError">Weights with an estimator; a non-integer scalar <paramref name="bins"/>; text data
        ///     without a range (its <c>a.min()</c>), with a range and an estimator (the in-range mask's comparison), or
        ///     against the edges' dtype (<see cref="DTypePromotionError"/>: no common DType for text and Python
        ///     numbers).</exception>
        /// <remarks>
        ///     NumPy writes histogram_bin_edges in Python; DefaultEngine's override runs its statements. The facades open
        ///     the matrix boundary's plain region and convert their C# <c>bins</c> form first; an override replaces it for
        ///     every array whose engine it is (the engine of <paramref name="a"/>). New in 0.71.0 (the np-layer dtype plan,
        ///     phase PE); it was the internal class <c>HistogramEngine</c>.
        /// </remarks>
        public abstract NDArray HistogramBinEdges(NDArray a, object bins, (double, double)? range = null,
            NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.histogram(a, bins=10, range=None, density=None, weights=None)</c>: the counts (or summed
        ///     weights, or density) of the data in each bin, and the bin edges.
        /// </summary>
        /// <param name="a">The data, a plain array (flattened).</param>
        /// <param name="bins">An <see cref="int"/> count (NumPy's default is 10; the facades pass it), an estimator name, or
        ///     an <see cref="NDArray"/> (a 0-d count or 1-D monotonic edges).</param>
        /// <param name="range">The outer edges, or null for the data's minimum and maximum.</param>
        /// <param name="density">Return the probability density instead of the counts.</param>
        /// <param name="weights">Weights of <paramref name="a"/>'s shape, or null.</param>
        /// <returns>The histogram and the edges (NumPy's tuple).</returns>
        /// <exception cref="ValueError">A weights shape that differs from <paramref name="a"/>'s, and the edge errors of
        ///     <see cref="HistogramBinEdges"/>.</exception>
        /// <exception cref="TypeError">The edge errors of <see cref="HistogramBinEdges"/>; text data that NumPy's counting
        ///     step refuses (its no-loop texts).</exception>
        /// <exception cref="UnicodeDecodeError">A text search converts <c>S</c> text holding a byte that is not ASCII to a
        ///     <c>U</c> common descriptor.</exception>
        /// <exception cref="ArgumentException">Bool weights over data on explicit edges (the in-place update of the bool
        ///     total).</exception>
        /// <remarks>
        ///     NumPy writes histogram in Python; DefaultEngine's override runs its statements, the weighted float sums in
        ///     NumPy's blocks of 65,536 samples (which keeps them bit-identical to NumPy's). The facades open the matrix
        ///     boundary's plain region first; an override replaces it for every array whose engine it is. New in 0.71.0
        ///     (the np-layer dtype plan, phase PE); it was the internal class <c>HistogramEngine</c>.
        /// </remarks>
        public abstract np.HistogramResult Histogram(NDArray a, object bins, (double, double)? range = null,
            bool density = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.histogramdd(sample, bins=10, range=None, density=None, weights=None)</c>: the D-dimensional
        ///     histogram of N points.
        /// </summary>
        /// <param name="sample">The points as D coordinate columns, each 1-D of length N: the normalized form of NumPy's
        ///     <c>(N, D)</c> array or sequence of D arrays (the facades build it).</param>
        /// <param name="bins">The bin spec: an <see cref="int"/> for every dimension (NumPy's default is 10; the facades pass
        ///     it), an <see cref="int"/> array, an <see cref="NDArray"/>, an <see cref="NDArray"/> array, or an object array
        ///     of per-dimension specs.</param>
        /// <param name="range">Per-dimension outer edges, or null.</param>
        /// <param name="density">Return the probability density instead of the counts.</param>
        /// <param name="weights">Per-point weights: a 1-D array of N values, read as given, or null.</param>
        /// <returns>The histogram and its per-dimension edges (NumPy's tuple). Without density the histogram is NumPy's
        ///     VIEW of the padded counts; a 1-D edges array the caller passed is returned as that same array.</returns>
        /// <exception cref="ValueError">A dimension mismatch, a non-positive scalar bin, edges that are not a scalar or 1-D,
        ///     or non-monotonic edges; with points, 1-D weights of another length than the points.</exception>
        /// <exception cref="IncorrectShapeException">With points, weights that are not 1-D (bincount's depth
        ///     check).</exception>
        /// <exception cref="InvalidCastException">With points, complex weights (bincount's safe cast to float64).</exception>
        /// <exception cref="IndexError">An empty edge array in any dimension (NumPy's <c>edges[i][-1]</c>).</exception>
        /// <exception cref="TypeError">A scalar bin that is not an integer, or that does not compare with 1; text samples
        ///     with a scalar bin spec and no range; text edges; 1-D text weights with points.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> column beside a <c>U</c> column holding a byte that is not
        ///     ASCII (NumPy's stacking decodes it before any bin is read).</exception>
        /// <remarks>
        ///     NumPy writes histogramdd in Python; DefaultEngine's override runs its statements one dimension at a time, in
        ///     NumPy's order. An override replaces <c>np.histogramdd</c> for every array whose engine it is (the engine of
        ///     the sample's first column); <c>np.histogram2d</c> reaches DefaultEngine's histogramdd statements through
        ///     <see cref="Histogram2d"/>. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class
        ///     <c>HistogramEngine</c>.
        /// </remarks>
        public abstract np.HistogramddResult Histogramdd(NDArray[] sample, object bins, (double, double)?[] range = null,
            bool density = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.histogram2d(x, y, bins=10, range=None, density=None, weights=None)</c>: the bidimensional
        ///     histogram of two samples, NumPy's wrapper of histogramdd (<c>numpy/lib/_twodim_base_impl.py</c>).
        /// </summary>
        /// <param name="x">The x coordinates, a plain array (not null).</param>
        /// <param name="y">The y coordinates, a plain array of <paramref name="x"/>'s length (not null).</param>
        /// <param name="bins">The bin spec: an <see cref="int"/> or <see cref="NDArray"/> for both dimensions (NumPy's
        ///     default is 10; the facades pass it), or a 1- or 2-element sequence of per-dimension specs.</param>
        /// <param name="range">Per-dimension outer edges, or null.</param>
        /// <param name="density">Return the probability density instead of the counts.</param>
        /// <param name="weights">Per-point weights, or null.</param>
        /// <returns>The histogram <c>H</c> and the two edge arrays (NumPy's tuple).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="y"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="x"/> or <paramref name="y"/> is 0-d (<c>len() of unsized
        ///     object</c>), and the TypeErrors of <see cref="Histogramdd"/>.</exception>
        /// <exception cref="ValueError"><paramref name="x"/> and <paramref name="y"/> differ in length (<c>x and y must have
        ///     the same length.</c>); arrays of one length but different shapes; two 2-D arrays of one shape (<c>too many
        ///     values to unpack (expected 2)</c>); and the ValueErrors of <see cref="Histogramdd"/>.</exception>
        /// <remarks>
        ///     NumPy writes histogram2d in Python; DefaultEngine's override runs its statements (the length check, the bins
        ///     dispatch, then histogramdd's statements). An override replaces <c>np.histogram2d</c> for every array whose
        ///     engine it is (the engine of <paramref name="x"/>). New in 0.71.0 (the np-layer dtype plan, phase PE); it was
        ///     the internal class <c>HistogramEngine</c>.
        /// </remarks>
        public abstract np.Histogram2dResult Histogram2d(NDArray x, NDArray y, object bins,
            (double, double)?[] range = null, bool density = false, NDArray weights = null);
    }
}
