using NumSharp;

namespace NumSharpMLNetExample;

/// <summary>A small labelled dataset made with NumSharp's random generator, so the example needs no file.</summary>
internal static class Blobs
{
    /// <summary>
    ///     Two clouds of points in four dimensions, normally distributed around (0, 0, 0, 0) and (1.5, 1.5, 1.5, 1.5),
    ///     shuffled together and split into training and test rows. Each row holds the four features and then the label
    ///     (0 for the first cloud, 1 for the second), all float32, the layout <c>AsDataView</c> reads as five columns.
    /// </summary>
    /// <remarks>
    ///     The clouds overlap, so no classifier reaches 100%: the example's accuracy is meaningful. The same seed gives
    ///     the same rows on every machine (NumSharp's <c>RandomState</c> is NumPy's generator).
    /// </remarks>
    /// <param name="pointsPerCloud">The number of points in each cloud.</param>
    /// <param name="testRows">How many of the shuffled rows go to the test set.</param>
    /// <param name="seed">The random seed.</param>
    /// <returns>The training rows and the test rows, (rows, 5) float32 arrays the caller owns and disposes.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="testRows"/> is not below the total number of points.</exception>
    public static (NDArray Train, NDArray Test) Make(int pointsPerCloud, int testRows, int seed)
    {
        int total = 2 * pointsPerCloud;
        if (testRows < 0 || testRows >= total)
            throw new ArgumentOutOfRangeException(nameof(testRows), $"{testRows} test rows out of {total} points.");

        var random = np.random.RandomState(seed);
        using NDArray first = random.normal(0.0, 1.0, new Shape(pointsPerCloud, 4));
        using NDArray second = random.normal(1.5, 1.0, new Shape(pointsPerCloud, 4));
        using NDArray zeros = np.zeros(new Shape(pointsPerCloud, 1));
        using NDArray ones = np.ones(new Shape(pointsPerCloud, 1));
        using NDArray firstRows = np.hstack(first, zeros);
        using NDArray secondRows = np.hstack(second, ones);
        using NDArray rows = np.vstack(firstRows, secondRows);           // (total, 5) float64
        using NDArray order = random.permutation(total);
        using NDArray shuffled = rows[order];                            // fancy indexing: a shuffled copy
        using NDArray single = shuffled.astype(NPTypeCode.Single);      // ML.NET's trainers read float32 features

        NDArray train = single[$":{total - testRows}"].copy();          // copies: the caller owns them outright
        NDArray test = single[$"{total - testRows}:"].copy();
        return (train, test);
    }
}
