// Gist: https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
// Pinned source: https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee/b12c675bed3be4126fd9f502f0cd9b33ec4b2186#file-timeseries_cnn-py
// Source: timeseries_cnn.py, jkleint (2016-07-22).
// No license declaration was found in the pinned gist. Independently implemented numerical core;
// Keras convolution layers, model fitting, prediction and console formatting are not ported.
using System;

namespace NumSharp.Examples.Gist;

/// <summary>Time-major lookback windows and a chronological split for a downstream regressor.</summary>
public static class TimeseriesCnn
{
    /// <summary>
    /// Preserve the gist's ranks: vector input gives X=(samples,window,1), Y=(samples,),
    /// Query=(1,window,1); matrix input gives X=(samples,window,series), Y=(samples,series).
    /// X and Query are independent copies; Y is a view of the input, exactly as in the Python routine.
    /// All NumSharp numeric dtypes and non-contiguous input are supported; there is no model dependency.
    /// </summary>
    public static WindowedSeries MakeInstances(NDArray timeseries, int windowSize)
    {
        if (timeseries is null) throw new ArgumentNullException(nameof(timeseries));
        if (timeseries.ndim != 1 && timeseries.ndim != 2)
            throw new ArgumentException("Expected a vector or a time-by-series matrix.", nameof(timeseries));
        if (windowSize <= 0 || windowSize >= timeseries.shape[0])
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window must be positive and shorter than the series.");

        int count = checked((int)(timeseries.shape[0] - windowSize));
        long series = timeseries.ndim == 1 ? 1 : timeseries.shape[1];
        var windows = new NDArray[count];
        try
        {
            for (int start = 0; start < count; start++)
                windows[start] = timeseries[$"{start}:{start + windowSize}"];
            using var stacked = np.stack(windows);
            var x = stacked.reshape(count, windowSize, series);
            var y = timeseries[$"{windowSize}:"];
            using var tail = timeseries[$"{-windowSize}:"];
            using var copy = tail.copy();
            var query = copy.reshape(1, windowSize, series);
            return new WindowedSeries(x, y, query);
        }
        finally
        {
            foreach (var window in windows) window?.Dispose();
        }
    }

    /// <summary>The input normalization used by evaluate_timeseries: a row vector becomes a column.</summary>
    public static NDArray AsTimeMajorMatrix(NDArray timeseries)
    {
        if (timeseries is null) throw new ArgumentNullException(nameof(timeseries));
        if (timeseries.ndim != 1 && timeseries.ndim != 2)
            throw new ArgumentException("Expected a vector or matrix.", nameof(timeseries));
        if (timeseries.ndim == 1) return np.expand_dims(timeseries, 1);
        return timeseries.shape[0] == 1 ? timeseries.T : timeseries.view();
    }

    /// <summary>
    /// The original train/test slice rule, including its Python -0 behavior: testSize=0 gives
    /// empty training data and the entire dataset as test data. Choose a positive size for fitting.
    /// All returned arrays are views; no sample is shuffled or leaked from the future into training.
    /// </summary>
    public static SeriesSplit Split(WindowedSeries instances, int testSize)
    {
        if (instances is null) throw new ArgumentNullException(nameof(instances));
        if (testSize < 0) throw new ArgumentOutOfRangeException(nameof(testSize));
        string stop = testSize == 0 ? "0" : $"-{testSize}";
        return new SeriesSplit(instances.X[$":{stop}"], instances.X[$"{stop}:"],
            instances.Y[$":{stop}"], instances.Y[$"{stop}:"]);
    }

    public static void Demo()
    {
        using var raw = np.arange(12);
        using var timeMajor = AsTimeMajorMatrix(raw);
        using var data = MakeInstances(timeMajor, 4);
        using var split = Split(data, 2);
        Console.WriteLine($"Time-series windows: {data.X.shape[0]} x {data.X.shape[1]} x {data.X.shape[2]}; train={split.XTrain.shape[0]}, test={split.XTest.shape[0]}; next target={data.Y.item<long>(0)}");
    }
}

public sealed class WindowedSeries : IDisposable
{
    public NDArray X { get; }
    public NDArray Y { get; }
    public NDArray Query { get; }
    internal WindowedSeries(NDArray x, NDArray y, NDArray query) => (X, Y, Query) = (x, y, query);
    public void Dispose() { X.Dispose(); Y.Dispose(); Query.Dispose(); }
}

public sealed class SeriesSplit : IDisposable
{
    public NDArray XTrain { get; }
    public NDArray XTest { get; }
    public NDArray YTrain { get; }
    public NDArray YTest { get; }
    internal SeriesSplit(NDArray xTrain, NDArray xTest, NDArray yTrain, NDArray yTest)
        => (XTrain, XTest, YTrain, YTest) = (xTrain, xTest, yTrain, yTest);
    public void Dispose() { XTrain.Dispose(); XTest.Dispose(); YTrain.Dispose(); YTest.Dispose(); }
}
