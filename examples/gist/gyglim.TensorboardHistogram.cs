// Gist: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514 — Michael Gygli, tensorboard_logging.py
// Pinned: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514/abe6172ae826c4c75fc344ba1def8b756ae9e8a7#file-tensorboard_logging-py
// Original header says "BSD License 2.0"; no full license text was supplied in that file.
// Functional numerical HistogramProto payload, not a TensorFlow/TensorBoard event-file writer.
using NumSharp;

namespace NumSharp.Examples.Gist;

public static class TensorboardHistogram
{
    public sealed record Summary(double Min, double Max, long Num, double Sum, double SumSquares,
        NDArray BucketLimits, NDArray Buckets) : IDisposable
    {
        public void Dispose() { BucketLimits.Dispose(); Buckets.Dispose(); }
    }

    // Like the gist's integer-bin call, use equal-width bins over the observed finite data range.
    // NumSharp has no np.histogram entry point, so compose its searchsorted + bincount primitives.
    public static Summary Create(NDArray values, int bins = 1000)
    {
        if (bins < 1 || bins == int.MaxValue) throw new ArgumentOutOfRangeException(nameof(bins));
        if (values.size == 0) throw new ArgumentException("Histogram summary requires at least one value.");
        if (values.typecode is NPTypeCode.Complex or NPTypeCode.Decimal or NPTypeCode.Half or NPTypeCode.Char or NPTypeCode.Boolean)
            throw new ArgumentException("Histogram summary supports integer, float32 and float64 values.");
        using var scope = NDScope.Open();
        var edgeType = values.typecode == NPTypeCode.Single ? np.float32 : np.float64;
        var data = values.astype(edgeType).ravel();
        if (!np.all(np.isfinite(data))) throw new ArgumentException("Histogram range must be finite.");
        var lower = np.min(data);
        var upper = np.max(data);
        double minimum = lower.astype(np.float64).item<double>(), maximum = upper.astype(np.float64).item<double>();
        double lo = minimum, hi = maximum;
        if (lo == hi)
        {
            // Typed endpoints retain NumPy's float32 arithmetic when widening a constant range.
            var half = np.array(.5).astype(edgeType);
            lower = lower - half;
            upper = upper + half;
            lo = lower.astype(np.float64).item<double>();
            hi = upper.astype(np.float64).item<double>();
        }
        if (!double.IsFinite(hi - lo) || hi <= lo) throw new ArgumentException("Histogram range is not representable in float64.");
        // The public scalar linspace overload widens its arguments. Compose NumPy's typed
        // arange * ((stop-start)/bins) + start instead, retaining float32 intermediates.
        var positions = np.arange((long)bins + 1).astype(edgeType);
        var delta = upper - lower;
        var divisor = np.array(bins).astype(edgeType);
        var step = delta / divisor;
        var edges = step.astype(np.float64).item<double>() == 0
            ? (positions / divisor) * delta + lower // NumPy's subnormal-step route
            : positions * step + lower;
        edges[-1] = upper;
        if (!np.all(np.diff(edges) > 0))
            throw new ArgumentException($"Too many bins for data range. Cannot create {bins} finite-sized bins.");
        var indices = np.minimum(np.searchsorted(edges, data, side: "right") - 1L, (long)bins - 1);
        var counts = np.bincount(indices, minlength: bins);
        var limits = edges["1:"].copy();
        // Summary scalar fields are doubles, but the reductions happen in the input's dtype first.
        // The dense square buffer needs NumPy's pairwise fold, not a flat SIMD accumulator:
        // at the source's 1,000-bin demonstration scale the latter differed by three float32 ULP.
        var result = new Summary(minimum, maximum, data.size, np.sum(values).astype(np.float64).item<double>(),
            np.sum(np.square(values).ravel('K'), axis: 0).astype(np.float64).item<double>(), scope.Returns(limits), scope.Returns(counts));
        return result;
    }

    public static void Demo()
    {
        using var values = np.array(new[] { 0.0, 1, 1, 2, 3, 4 });
        using var summary = Create(values, bins: 4);
        Console.WriteLine($"Histogram: n={summary.Num}, range=[{summary.Min}, {summary.Max}], sum={summary.Sum}, sumSquares={summary.SumSquares}");
        for (int i = 0; i < summary.Buckets.size; i++)
            Console.WriteLine($"  upper={summary.BucketLimits.item<double>(i)}, count={summary.Buckets.item<long>(i)}");
    }
}
