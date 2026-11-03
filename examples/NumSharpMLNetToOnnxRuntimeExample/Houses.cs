using NumSharp;

namespace NumSharpMLNetToOnnxRuntimeExample;

/// <summary>The dataset: houses whose price depends on their size, rooms and age, made with NumSharp's generator.</summary>
internal static class Houses
{
    /// <summary>
    ///     Make <paramref name="count"/> houses: a size of 40 to 220 m², about one room per 35 m², an age of 0 to 60 years,
    ///     and a price (in thousands) that grows with the size and the rooms and falls with the age, plus noise.
    /// </summary>
    /// <remarks>
    ///     The arithmetic runs inside an <see cref="NDScope"/>: every intermediate array is released when the method
    ///     returns, except the table it hands back through <c>scope.Returns</c>. NumSharp's generator is NumPy's, so a
    ///     seed gives the same houses on every machine.
    /// </remarks>
    /// <param name="count">How many houses (rows).</param>
    /// <param name="seed">The seed of the generator.</param>
    /// <returns>A new <c>(count, 4)</c> float32 table: size, rooms, age, price. The caller disposes it.</returns>
    public static NDArray Make(int count, int seed)
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(seed);
        NDArray size = rng.rand(count) * 180 + 40;
        NDArray rooms = np.floor(size / 35 + rng.rand(count));
        NDArray age = rng.rand(count) * 60;
        NDArray price = size * 2.1 + rooms * 15 - age * 0.8 + rng.randn(count) * 15;
        return scope.Returns(np.column_stack(new[] { size, rooms, age, price }).astype(np.float32));
    }
}
