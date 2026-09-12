using System;
using System.IO;
using System.Globalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;
using NumSharp.Backends;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Examples;

[TestClass, DoNotParallelize]
public class KarpathyLstmTests
{
    private IBlasBackend? previous;
    private TensorEngine engine = null!;
    [TestInitialize] public void EnableProducts()
    {
        using var a = np.zeros(0);
        engine = a.TensorEngine; previous = engine.Blas;
        OpenBlasEngine.Enable(threads: 1);
    }
    [TestCleanup] public void RestoreProducts() => engine.Blas = previous;

    [TestMethod]
    public void Initialization_PacksBiasFirst_AndPositiveForgetBias()
    {
        using var w = BatchedLstm.Initialize(10, 4);
        CollectionAssert.AreEqual(new long[] { 15, 16 }, w.shape);
        for (int j = 0; j < 16; j++) Assert.AreEqual(j is >= 4 and < 8 ? 3.0 : 0.0, w.item<double>(0, j));
    }

    [TestMethod]
    public void OriginalSequentialVersusBatch_AllFiveClaimsAreAssertions()
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(7);
        var w = np.array(rng.randn(15, 16) / np.sqrt(np.array(14.0)));
        w["0,:"] = 0.0; w["0,4:8"] = 3.0;
        var x = rng.randn(5, 3, 10); var h0 = rng.randn(3, 4); var c0 = rng.randn(3, 4);
        using var batch = BatchedLstm.Forward(x, w, c0, h0);
        var ticks = new BatchedLstm.ForwardPass[5];
        try
        {
            NDArray cp = c0, hp = h0;
            var hcat = np.zeros((5, 3, 4));
            for (int t = 0; t < 5; t++)
            {
                ticks[t] = BatchedLstm.Forward(x[$"{t}:{t + 1},:,:"] , w, cp, hp);
                cp = ticks[t].Cell["0,:,:"]; hp = ticks[t].Hidden["0,:,:"];
                hcat[$"{t},:,:"] = hp;
            }
            Near(batch.Hidden, hcat, 0);
            var dh = rng.randn(5, 3, 4);
            using var bg = BatchedLstm.Backward(dh, batch);
            var dx = np.zeros_like(x); var dw = np.zeros_like(w);
            NDArray? dcNext = null, dhNext = null;
            var gradients = new BatchedLstm.Gradients[5];
            try
            {
                for (int t = 4; t >= 0; t--)
                {
                    gradients[t] = BatchedLstm.Backward(dh[$"{t}:{t + 1},:,:"] , ticks[t], dcNext, dhNext);
                    dcNext = gradients[t].InitialCell; dhNext = gradients[t].InitialHidden;
                    dx[$"{t},:,:"] = gradients[t].Input["0,:,:"];
                    dw[":,:"] += gradients[t].Weights;
                }
                Near(bg.Input, dx, 0); Near(bg.Weights, dw, 0);
                Near(bg.InitialCell, dcNext!, 0); Near(bg.InitialHidden, dhNext!, 0);
            }
            finally { foreach (var g in gradients) g?.Dispose(); }
        }
        finally { foreach (var tick in ticks) tick?.Dispose(); }
    }

    [TestMethod]
    public void OriginalAll414Coordinates_PassFiniteDifferenceGradientCheck()
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(7);
        var w = rng.randn(15, 16) / np.sqrt(np.array(14.0));
        w["0,:"] = 0.0; w["0,4:8"] = 3.0;
        var x = rng.randn(5, 3, 10); var h0 = rng.randn(3, 4); var c0 = rng.randn(3, 4);
        using var pass = BatchedLstm.Forward(x, w, c0, h0);
        var dh = rng.randn(5, 3, 4);
        using var gradients = BatchedLstm.Backward(dh, pass);
        var values = new[] { x, w, c0, h0 };
        var derivatives = new[] { gradients.Input, gradients.Weights, gradients.InitialCell, gradients.InitialHidden };
        int checkedCount = 0;
        double maximum = 0;
        double Loss()
        {
            using var tick = NDScope.Open();
            using var forward = BatchedLstm.Forward(x, w, c0, h0);
            return np.sum(forward.Hidden * dh).item<double>();
        }
        for (int k = 0; k < values.Length; k++)
        {
            using var flat = values[k].reshape(-1);
            for (long i = 0; i < flat.size; i++)
            {
                double saved = flat.item<double>(i), plus, minus;
                try { flat[i] = saved + 1e-5; plus = Loss(); flat[i] = saved - 1e-5; minus = Loss(); }
                finally { flat[i] = saved; }
                double numerical = (plus - minus) / 2e-5, analytic = derivatives[k].item<double>(i);
                Assert.IsTrue(double.IsFinite(numerical) && double.IsFinite(analytic), $"Nonfinite gradient at group {k}, element {i}.");
                if (System.Math.Max(System.Math.Abs(numerical), System.Math.Abs(analytic)) >= 1e-7)
                {
                    double relative = System.Math.Abs(numerical - analytic) / System.Math.Abs(numerical + analytic);
                    maximum = System.Math.Max(maximum, relative);
                    Assert.IsTrue(relative < 1e-2, $"Source warning threshold: group {k}, element {i}, relative={relative:R}");
                }
                checkedCount++;
            }
        }
        Assert.AreEqual(414, checkedCount);
        Console.WriteLine($"LSTM original all-coordinate gradcheck: {checkedCount} checked; maximum relative error={maximum:R}");
    }

    [TestMethod]
    public void ForwardCache_OwnsSnapshotsBeyondCallerScope_AndBackwardDoesNotMutateUpstream()
    {
        BatchedLstm.ForwardPass cache;
        using (var scope = NDScope.Open())
        {
            var w = BatchedLstm.Initialize(2, 3);
            var x = np.ones((2, 1, 2));
            cache = BatchedLstm.Forward(x, w);
            w[":,:"] = 1000.0;
        }
        using (cache)
        {
            using var dh = np.ones((2, 1, 3));
            using var g = BatchedLstm.Backward(dh, cache);
            Assert.IsTrue(np.all(np.isfinite(g.Input)));
            Assert.IsTrue(np.all(dh == 1.0));
            Assert.IsTrue(np.all(cache.Weights < 1000.0));
        }
    }

    [TestMethod]
    public void EmptyWrongDtypeAndIncompatibleState_AreExplicitErrors()
    {
        using var scope = NDScope.Open();
        var w = BatchedLstm.Initialize(2, 3);
        Assert.ThrowsException<ArgumentException>(() => BatchedLstm.Forward(np.zeros((0, 1, 2)), w));
        Assert.ThrowsException<ArgumentException>(() => BatchedLstm.Forward(np.zeros((2, 1, 2), dtype: np.float32), w));
        Assert.ThrowsException<ArgumentException>(() => BatchedLstm.Forward(np.zeros((2, 1, 2)), w, np.zeros((2, 3))));
    }

    [TestMethod]
    public void ActualDemo_ReportsIndependentlyObservedLossAndDerivative()
    {
        var previousWriter = Console.Out; var culture = CultureInfo.CurrentCulture;
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        try { Console.SetOut(output); CultureInfo.CurrentCulture = CultureInfo.InvariantCulture; BatchedLstm.Demo(); }
        finally { Console.SetOut(previousWriter); CultureInfo.CurrentCulture = culture; }
        Assert.AreEqual("LSTM: time=5, batch=3, hidden=4; weighted loss=4.767046778; dX[0]=-0.204665241", output.ToString().Trim());
    }

    private static void Near(NDArray a, NDArray b, double tolerance)
    {
        CollectionAssert.AreEqual(a.shape, b.shape);
        for (long i = 0; i < a.size; i++) Assert.AreEqual(a.item<double>(i), b.item<double>(i), tolerance, $"element {i}");
    }
}
