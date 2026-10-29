// Gist: https://gist.github.com/karpathy/587454dc0146a6ae21fc — Andrej Karpathy, batched LSTM.
// Pinned gist revision: 029a6e85048960fec169411f34eea304b42846a5 (gistfile1.py).
// Source SHA256: 137a213cfe6ec6eb7c73dc73c18f356dfd676e6e36bb66d2be6667ca6be7a4c4.
// No explicit license declaration in the pinned file. Independently expressed numerical equations.
using NumSharp;

namespace NumSharp.Examples.Gist.Karpathy;

/// <summary>Float64 IFOG LSTM, including reverse-mode derivatives and recurrent boundary gradients.</summary>
public static class BatchedLstm
{
    public sealed class ForwardPass : IDisposable
    {
        public NDArray Hidden { get; }
        public NDArray Cell { get; }
        public NDArray CellTanh { get; }
        public NDArray GateInputs { get; }
        public NDArray Gates { get; }
        public NDArray Concatenated { get; }
        public NDArray Weights { get; }
        public NDArray InitialCell { get; }
        public NDArray InitialHidden { get; }
        private readonly NDArray[] owned;
        internal ForwardPass(NDArray hidden, NDArray cell, NDArray cellTanh, NDArray gateInputs,
            NDArray gates, NDArray concatenated, NDArray weights, NDArray initialCell, NDArray initialHidden)
        {
            Hidden = hidden; Cell = cell; CellTanh = cellTanh; GateInputs = gateInputs;
            Gates = gates; Concatenated = concatenated; Weights = weights;
            InitialCell = initialCell; InitialHidden = initialHidden;
            owned = new[] { hidden, cell, cellTanh, gateInputs, gates, concatenated, weights, initialCell, initialHidden };
            NDScope.Detach(owned); // this cache, not a caller's temporary scope, owns its snapshots
        }
        public void Dispose() { foreach (var array in owned) array.Dispose(); }
    }

    public sealed class Gradients : IDisposable
    {
        public NDArray Input { get; }
        public NDArray Weights { get; }
        public NDArray InitialCell { get; }
        public NDArray InitialHidden { get; }
        private readonly NDArray[] owned;
        internal Gradients(NDArray input, NDArray weights, NDArray cell, NDArray hidden)
        {
            Input = input; Weights = weights; InitialCell = cell; InitialHidden = hidden;
            owned = new[] { input, weights, cell, hidden };
            NDScope.Detach(owned);
        }
        public void Dispose() { foreach (var array in owned) array.Dispose(); }
    }

    public static NDArray Initialize(int inputSize, int hiddenSize, int seed = 7, double forgetBias = 3)
    {
        if (inputSize < 1 || hiddenSize < 1) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (!double.IsFinite(forgetBias)) throw new ArgumentOutOfRangeException(nameof(forgetBias));
        using var scope = NDScope.Open();
        var weights = np.random.RandomState(seed).randn(inputSize + hiddenSize + 1, 4 * hiddenSize) /
            np.sqrt(np.array((double)(inputSize + hiddenSize)));
        weights["0,:"] = 0.0;
        weights[$"0,{hiddenSize}:{2 * hiddenSize}"] = forgetBias; // source code is POSITIVE +3, despite its stale comment
        return scope.Returns(weights);
    }

    /// <summary>Input is (time,batch,input), weights (1+input+hidden,4*hidden), states (batch,hidden).
    /// Cache arrays are independent snapshots owned by the returned disposable object.</summary>
    public static ForwardPass Forward(NDArray input, NDArray weights, NDArray? initialCell = null, NDArray? initialHidden = null)
    {
        Check(input, 3, nameof(input)); Check(weights, 2, nameof(weights));
        long n = input.shape[0], b = input.shape[1], inputs = input.shape[2], d = weights.shape[1] / 4;
        if (n < 1 || b < 1 || inputs < 1 || d < 1 || weights.shape[1] % 4 != 0 || weights.shape[0] != 1 + inputs + d)
            throw new ArgumentException("Nonempty input and compatible packed IFOG weights are required.");
        CheckState(initialCell, b, d, nameof(initialCell)); CheckState(initialHidden, b, d, nameof(initialHidden));
        using var scope = NDScope.Open();
        var w = weights.copy();
        var c0 = initialCell?.copy() ?? np.zeros((b, d), dtype: np.float64);
        var h0 = initialHidden?.copy() ?? np.zeros((b, d), dtype: np.float64);
        var hin = np.zeros((n, b, weights.shape[0]), dtype: np.float64);
        var h = np.zeros((n, b, d), dtype: np.float64);
        var c = np.zeros_like(h); var ct = np.zeros_like(h);
        var gates = np.zeros((n, b, 4 * d), dtype: np.float64);
        var activated = np.zeros_like(gates);
        for (long t = 0; t < n; t++)
        {
            using var tick = NDScope.Open();
            hin[$"{t},:,0"] = 1.0;
            hin[$"{t},:,1:{inputs + 1}"] = input[$"{t},:,:" ];
            hin[$"{t},:,{inputs + 1}:"] = t == 0 ? h0 : h[$"{t - 1},:,:" ];
            gates[$"{t},:,:"] = np.dot(hin[$"{t},:,:"], w);
            activated[$"{t},:,:{3 * d}"] = 1.0 / (1.0 + np.exp(-gates[$"{t},:,:{3 * d}"]));
            activated[$"{t},:,{3 * d}:"] = np.tanh(gates[$"{t},:,{3 * d}:"]);
            var previous = t == 0 ? c0 : c[$"{t - 1},:,:" ];
            c[$"{t},:,:"] = activated[$"{t},:,:{d}"] * activated[$"{t},:,{3 * d}:"] +
                activated[$"{t},:,{d}:{2 * d}"] * previous;
            ct[$"{t},:,:"] = np.tanh(c[$"{t},:,:" ]);
            h[$"{t},:,:"] = activated[$"{t},:,{2 * d}:{3 * d}"] * ct[$"{t},:,:" ];
        }
        return new ForwardPass(h, c, ct, gates, activated, hin, w, c0, h0);
    }

    public static Gradients Backward(NDArray hiddenGradient, ForwardPass cache,
        NDArray? finalCellGradient = null, NDArray? finalHiddenGradient = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        Check(hiddenGradient, 3, nameof(hiddenGradient));
        if (!hiddenGradient.shape.SequenceEqual(cache.Hidden.shape)) throw new ArgumentException("Hidden-gradient shape must match forward output.");
        long n = cache.Hidden.shape[0], b = cache.Hidden.shape[1], d = cache.Hidden.shape[2];
        long inputs = cache.Weights.shape[0] - d - 1;
        CheckState(finalCellGradient, b, d, nameof(finalCellGradient));
        CheckState(finalHiddenGradient, b, d, nameof(finalHiddenGradient));
        using var scope = NDScope.Open();
        var da = np.zeros_like(cache.Gates); var dg = np.zeros_like(cache.GateInputs);
        var dw = np.zeros_like(cache.Weights); var di = np.zeros_like(cache.Concatenated);
        var dc = np.zeros_like(cache.Cell); var dx = np.zeros((n, b, inputs), dtype: np.float64);
        var dh = hiddenGradient.copy(); var dc0 = np.zeros((b, d), dtype: np.float64); var dh0 = np.zeros_like(dc0);
        if (finalCellGradient is not null) dc[$"{n - 1},:,:"] += finalCellGradient;
        if (finalHiddenGradient is not null) dh[$"{n - 1},:,:"] += finalHiddenGradient;
        for (long t = n - 1; t >= 0; t--)
        {
            using var tick = NDScope.Open();
            var tanh = cache.CellTanh[$"{t},:,:" ];
            da[$"{t},:,{2 * d}:{3 * d}"] = tanh * dh[$"{t},:,:" ];
            dc[$"{t},:,:"] += (1.0 - np.square(tanh)) * (cache.Gates[$"{t},:,{2 * d}:{3 * d}"] * dh[$"{t},:,:" ]);
            var prev = t == 0 ? cache.InitialCell : cache.Cell[$"{t - 1},:,:" ];
            da[$"{t},:,{d}:{2 * d}"] = prev * dc[$"{t},:,:" ];
            var carry = cache.Gates[$"{t},:,{d}:{2 * d}"] * dc[$"{t},:,:" ];
            if (t > 0) dc[$"{t - 1},:,:"] += carry;
            else dc0[":,:"] = carry;
            da[$"{t},:,:{d}"] = cache.Gates[$"{t},:,{3 * d}:"] * dc[$"{t},:,:" ];
            da[$"{t},:,{3 * d}:"] = cache.Gates[$"{t},:,:{d}"] * dc[$"{t},:,:" ];
            dg[$"{t},:,{3 * d}:"] = (1.0 - np.square(cache.Gates[$"{t},:,{3 * d}:"])) * da[$"{t},:,{3 * d}:"];
            var y = cache.Gates[$"{t},:,:{3 * d}"];
            dg[$"{t},:,:{3 * d}"] = (y * (1.0 - y)) * da[$"{t},:,:{3 * d}"];
            dw[":,:"] += np.dot(cache.Concatenated[$"{t},:,:"].T, dg[$"{t},:,:"]);
            di[$"{t},:,:"] = np.dot(dg[$"{t},:,:"], cache.Weights.T);
            dx[$"{t},:,:"] = di[$"{t},:,1:{inputs + 1}"];
            if (t > 0) dh[$"{t - 1},:,:"] += di[$"{t},:,{inputs + 1}:"];
            else dh0[":,:"] += di[$"{t},:,{inputs + 1}:"];
        }
        return new Gradients(dx, dw, dc0, dh0);
    }

    private static void Check(NDArray array, int rank, string name)
    {
        ArgumentNullException.ThrowIfNull(array, name);
        if (array.ndim != rank || array.typecode != NPTypeCode.Double) throw new ArgumentException($"{name} must be a rank-{rank} float64 array.");
    }
    private static void CheckState(NDArray? state, long b, long d, string name)
    {
        if (state is null) return;
        Check(state, 2, name);
        if (state.shape[0] != b || state.shape[1] != d) throw new ArgumentException($"{name} must have shape ({b},{d}).");
    }

    public static void Demo()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(7);
        var w = random.randn(15, 16) / np.sqrt(np.array(14.0));
        w["0,:"] = 0.0; w["0,4:8"] = 3.0;
        var x = random.randn(5, 3, 10); var h0 = random.randn(3, 4); var c0 = random.randn(3, 4);
        using var pass = Forward(x, w, c0, h0);
        var upstream = random.randn(5, 3, 4);
        using var gradient = Backward(upstream, pass);
        Console.WriteLine($"LSTM: time=5, batch=3, hidden=4; weighted loss={np.sum(pass.Hidden * upstream).item<double>():F9}; dX[0]={gradient.Input.item<double>(0):F9}");
    }
}
