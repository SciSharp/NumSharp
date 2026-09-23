// Gist: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 — Andrej Karpathy, microgpt.py.
// Immutable source file: https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/f42a9440467b4620bd6c178777e7a0209ef064d2/microgpt.py
// The raw-file revision is verified; the gist HEAD API was unavailable. No source license was declared.
// Independently expressed NumSharp transformer with manual reverse-mode derivatives and Adam.
// Original scalar Python autograd is replaced by matrix arithmetic; see coverage-microgpt.json.
using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Examples.Gist.Karpathy;

public static class MicroGpt
{
    public static readonly string[] TinyNames = { "anna", "emma", "ava", "mia", "ella", "eve", "ada", "bob", "otto", "noah" };

    /// <summary>Character vocabulary in Unicode code-point order, plus the shared start/end token.</summary>
    public sealed class Tokenizer
    {
        private readonly Dictionary<int, int> _encode;
        private readonly string[] _characters;
        public int Bos => _characters.Length;
        public int Size => Bos + 1;
        public Tokenizer(IEnumerable<string> documents)
        {
            ArgumentNullException.ThrowIfNull(documents);
            var codes = documents.SelectMany(x => x.EnumerateRunes().Select(r => r.Value)).Distinct().OrderBy(x => x).ToArray();
            if (codes.Length == 0) throw new ArgumentException("At least one character is required.", nameof(documents));
            _encode = codes.Select((code, id) => (code, id)).ToDictionary(x => x.code, x => x.id);
            _characters = codes.Select(char.ConvertFromUtf32).ToArray();
        }
        public int[] Encode(string text) => new[] { Bos }.Concat(text.EnumerateRunes().Select(r =>
            _encode.TryGetValue(r.Value, out int token) ? token : throw new ArgumentException("Character is absent from the vocabulary.", nameof(text))))
            .Append(Bos).ToArray();
        public string Decode(IEnumerable<int> tokens) => string.Concat(tokens.Select(t =>
            t == Bos ? "" : t >= 0 && t < Bos ? _characters[t] : throw new ArgumentOutOfRangeException(nameof(tokens))));
    }

    /// <summary>One complete float64 GPT: causal attention, residual MLP, full backpropagation and Adam.</summary>
    public sealed class Model : IDisposable
    {
        private readonly Dictionary<string, NDArray> _weights = new();
        private readonly Dictionary<string, NDArray> _first = new(), _second = new();
        private bool _disposed;
        public int VocabularySize { get; }
        public int EmbeddingSize { get; }
        public int HeadCount { get; }
        public int BlockSize { get; }
        public int LayerCount { get; }
        public int OptimizerStep { get; private set; }
        // Borrowed model buffers. Callers may inspect/assign values (e.g. gradient checking), not dispose them.
        public IReadOnlyDictionary<string, NDArray> Parameters => _weights;
        public IReadOnlyDictionary<string, NDArray> FirstMoments => _first;
        public IReadOnlyDictionary<string, NDArray> SecondMoments => _second;
        public long ParameterCount => _weights.Values.Sum(x => x.size);

        public Model(int vocabularySize, int embeddingSize = 16, int headCount = 4,
            int blockSize = 16, int layerCount = 1, int seed = 42)
        {
            if (vocabularySize < 2 || embeddingSize < 1 || headCount < 1 || embeddingSize % headCount != 0 || blockSize < 1 || layerCount < 1)
                throw new ArgumentException("Require vocabulary>=2, positive dimensions and heads dividing embedding width.");
            (VocabularySize, EmbeddingSize, HeadCount, BlockSize, LayerCount) = (vocabularySize, embeddingSize, headCount, blockSize, layerCount);
            var random = np.random.RandomState(seed); // NumPy RNG, not Python random.gauss's stream.
            using var initialization = NDScope.Open();
            void Add(string name, int rows, int columns)
            {
                var w = random.randn(rows, columns) * .08;
                var m = np.zeros_like(w);
                var v = np.zeros_like(w);
                _weights.Add(name, w); _first.Add(name, m); _second.Add(name, v);
                NDScope.Detach(w); NDScope.Detach(m); NDScope.Detach(v);
            }
            try
            {
                Add("wte", vocabularySize, embeddingSize); Add("wpe", blockSize, embeddingSize); Add("lm_head", vocabularySize, embeddingSize);
                for (int layer = 0; layer < layerCount; layer++)
                {
                    foreach (string name in new[] { "attn_wq", "attn_wk", "attn_wv", "attn_wo" }) Add($"layer{layer}.{name}", embeddingSize, embeddingSize);
                    Add($"layer{layer}.mlp_fc1", 4 * embeddingSize, embeddingSize);
                    Add($"layer{layer}.mlp_fc2", embeddingSize, 4 * embeddingSize);
                }
            }
            catch { Dispose(); throw; }
        }

        /// <summary>All prefix logits, T x vocabulary. Causal masking prevents later tokens affecting earlier logits.</summary>
        public NDArray Forward(IReadOnlyList<int> inputs)
        {
            ValidateTokens(inputs);
            using var scope = NDScope.Open();
            return scope.Returns(ForwardCore(inputs).Logits);
        }

        /// <summary>Mean next-token cross entropy over one complete bounded document.</summary>
        public double Loss(IReadOnlyList<int> inputs, IReadOnlyList<int> targets)
        {
            ValidatePair(inputs, targets);
            using var scope = NDScope.Open();
            return CrossEntropy(Softmax(ForwardCore(inputs).Logits), targets);
        }

        public GradientBatch LossAndGradients(IReadOnlyList<int> inputs, IReadOnlyList<int> targets)
        {
            ValidatePair(inputs, targets);
            using var scope = NDScope.Open();
            var tape = ForwardCore(inputs);
            var probabilities = Softmax(tape.Logits);
            double loss = CrossEntropy(probabilities, targets);
            var gradients = _weights.ToDictionary(x => x.Key, x => np.zeros_like(x.Value));
            var dLogits = probabilities.copy();
            for (int row = 0; row < targets.Count; row++) dLogits[row, targets[row]] = dLogits.item<double>(row, targets[row]) - 1;
            dLogits = dLogits / inputs.Count;
            gradients["lm_head"] = np.matmul(dLogits.T, tape.Final);
            var dx = np.matmul(dLogits, _weights["lm_head"]);

            for (int layer = LayerCount - 1; layer >= 0; layer--)
            {
                string prefix = $"layer{layer}.";
                var c = tape.Layers[layer];
                gradients[prefix + "mlp_fc2"] = np.matmul(dx.T, c.Hidden);
                var dHidden = np.matmul(dx, _weights[prefix + "mlp_fc2"]);
                dHidden = dHidden * (c.PreHidden > 0.0);
                gradients[prefix + "mlp_fc1"] = np.matmul(dHidden.T, c.MlpNorm);
                var dNorm2 = np.matmul(dHidden, _weights[prefix + "mlp_fc1"]);
                var dResidual = dx + RmsBackward(dNorm2, c.AfterAttention, c.MlpScale);

                gradients[prefix + "attn_wo"] = np.matmul(dResidual.T, c.AttentionOutput);
                var dAttention = np.matmul(dResidual, _weights[prefix + "attn_wo"]);
                var dq = np.zeros_like(c.Q); var dk = np.zeros_like(c.K); var dv = np.zeros_like(c.V);
                int width = EmbeddingSize / HeadCount;
                double rootWidth = Math.Sqrt(width);
                for (int head = 0; head < HeadCount; head++)
                {
                    string slice = $":,{head * width}:{(head + 1) * width}";
                    var q = c.Q[slice]; var k = c.K[slice]; var v = c.V[slice];
                    var dHead = dAttention[slice]; var attention = c.Attention[head];
                    var dProbabilities = np.matmul(dHead, v.T);
                    var weighted = attention * dProbabilities;
                    var dScores = attention * (dProbabilities - np.sum(weighted, axis: 1, keepdims: true));
                    dq[slice] = np.matmul(dScores, k) / rootWidth;
                    dk[slice] = np.matmul(dScores.T, q) / rootWidth;
                    dv[slice] = np.matmul(attention.T, dHead);
                }
                gradients[prefix + "attn_wq"] = np.matmul(dq.T, c.AttentionNorm);
                gradients[prefix + "attn_wk"] = np.matmul(dk.T, c.AttentionNorm);
                gradients[prefix + "attn_wv"] = np.matmul(dv.T, c.AttentionNorm);
                var dNorm1 = np.matmul(dq, _weights[prefix + "attn_wq"]) + np.matmul(dk, _weights[prefix + "attn_wk"]);
                dNorm1 = dNorm1 + np.matmul(dv, _weights[prefix + "attn_wv"]);
                dx = dResidual + RmsBackward(dNorm1, c.Input, c.AttentionScale);
            }
            var dEmbedding = RmsBackward(dx, tape.Embedding, tape.EmbeddingScale);
            for (int row = 0; row < inputs.Count; row++)
            {
                using var contribution = dEmbedding[$"{row},:"];
                using var current = gradients["wte"][$"{inputs[row]},:"];
                np.add(current, contribution, @out: current);
                gradients["wpe"][$"{row},:"] = contribution;
            }
            foreach (var value in gradients.Values) scope.Returns(value);
            return new GradientBatch(loss, scope.Returns(tape.Logits), gradients);
        }

        /// <summary>Adam update, with source beta1=.85, beta2=.99, epsilon=1e-8 and bias correction.</summary>
        public void ApplyAdam(GradientBatch batch, double learningRate, double beta1 = .85, double beta2 = .99, double epsilon = 1e-8)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(batch);
            if (!double.IsFinite(learningRate) || learningRate < 0 || !double.IsFinite(beta1) || beta1 < 0 || beta1 >= 1 || !double.IsFinite(beta2) || beta2 < 0 || beta2 >= 1 || epsilon <= 0 || !double.IsFinite(epsilon))
                throw new ArgumentOutOfRangeException(nameof(learningRate));
            if (batch.Parameters.Count != _weights.Count) throw new ArgumentException("Gradient parameter set differs from the model.", nameof(batch));
            foreach (var (name, w) in _weights)
                if (!batch.Parameters.TryGetValue(name, out var g) || g.typecode != NPTypeCode.Double || !g.shape.SequenceEqual(w.shape))
                    throw new ArgumentException("Gradient shape/dtype differs from the model.", nameof(batch));
            using var scope = NDScope.Open();
            int step = OptimizerStep + 1;
            foreach (var (name, w) in _weights)
            {
                var g = batch.Parameters[name];
                var m = beta1 * _first[name] + (1 - beta1) * g;
                var v = beta2 * _second[name] + (1 - beta2) * np.square(g);
                var mHat = m / (1 - Math.Pow(beta1, step));
                var vHat = v / (1 - Math.Pow(beta2, step));
                var update = learningRate * mHat / (np.sqrt(vHat) + epsilon);
                _first[name][":"] = m; _second[name][":"] = v;
                np.subtract(w, update, @out: w);
            }
            OptimizerStep = step;
        }

        /// <summary>
        /// Train on a bounded local dataset; includes BOS at both ends and truncates at the context size.
        /// Dataset order is supplied by the caller (no hidden download/shuffle). Returns the per-step losses.
        /// </summary>
        public NDArray Train(Tokenizer tokenizer, IReadOnlyList<string> documents, int steps = 1000,
            double learningRate = .01, Action<int, double>? progress = null)
        {
            if (tokenizer.Size != VocabularySize || documents.Count == 0 || steps < 1) throw new ArgumentException("Require compatible vocabulary, documents and positive steps.");
            using var scope = NDScope.Open();
            var losses = np.empty(steps, dtype: np.float64);
            for (int step = 0; step < steps; step++)
            {
                int[] tokens = tokenizer.Encode(documents[step % documents.Count]);
                int length = Math.Min(BlockSize, tokens.Length - 1);
                using var batch = LossAndGradients(tokens.Take(length).ToArray(), tokens.Skip(1).Take(length).ToArray());
                losses[step] = batch.Loss;
                ApplyAdam(batch, learningRate * (1 - (double)step / steps));
                progress?.Invoke(step, batch.Loss);
            }
            return scope.Returns(losses);
        }

        /// <summary>
        /// Categorical sampling with temperature and BOS termination. Prefix recomputation replaces
        /// the scalar source's key/value cache; it is the same causal function, not a stub predictor.
        /// NumPy's seeded uniform stream replaces Python random.choices, so samples are not source-RNG pins.
        /// </summary>
        public string[] Generate(Tokenizer tokenizer, int samples = 20, double temperature = .5, int seed = 42)
        {
            if (tokenizer.Size != VocabularySize || samples < 1 || !double.IsFinite(temperature) || temperature <= 0)
                throw new ArgumentException("Require compatible vocabulary, positive sample count and temperature.");
            var random = np.random.RandomState(seed);
            var output = new string[samples];
            for (int sample = 0; sample < samples; sample++)
            {
                var prefix = new List<int> { tokenizer.Bos };
                var text = new List<int>();
                for (int position = 0; position < BlockSize; position++)
                {
                    using var scope = NDScope.Open();
                    var logits = Forward(prefix);
                    var last = logits[$"{position}:{position + 1},:"] / temperature;
                    var probabilities = Softmax(last);
                    double draw = random.rand().item<double>();
                    double cumulative = 0;
                    int token = VocabularySize - 1;
                    for (int i = 0; i < VocabularySize; i++)
                    {
                        cumulative += probabilities.item<double>(0, i);
                        if (draw < cumulative) { token = i; break; }
                    }
                    if (token == tokenizer.Bos) break;
                    text.Add(token); prefix.Add(token);
                }
                output[sample] = tokenizer.Decode(text);
            }
            return output;
        }

        private Tape ForwardCore(IReadOnlyList<int> inputs)
        {
            int count = inputs.Count;
            var embedding = np.empty((count, EmbeddingSize), dtype: np.float64);
            for (int row = 0; row < count; row++) embedding[$"{row},:"] = _weights["wte"][$"{inputs[row]},:"] + _weights["wpe"][$"{row},:"];
            var x = Rms(embedding, out var embeddingScale);
            var layers = new LayerTape[LayerCount];
            for (int layer = 0; layer < LayerCount; layer++)
            {
                string prefix = $"layer{layer}.";
                var input = x;
                var norm = Rms(x, out var attentionScale);
                var q = np.matmul(norm, _weights[prefix + "attn_wq"].T);
                var k = np.matmul(norm, _weights[prefix + "attn_wk"].T);
                var v = np.matmul(norm, _weights[prefix + "attn_wv"].T);
                var attentionOutput = np.empty((count, EmbeddingSize), dtype: np.float64);
                var weights = new NDArray[HeadCount];
                int width = EmbeddingSize / HeadCount;
                for (int head = 0; head < HeadCount; head++)
                {
                    string slice = $":,{head * width}:{(head + 1) * width}";
                    var scores = np.matmul(q[slice], k[slice].T) / Math.Sqrt(width);
                    for (int row = 0; row < count; row++)
                        if (row + 1 < count) scores[$"{row},{row + 1}:"] = double.NegativeInfinity;
                    weights[head] = Softmax(scores);
                    attentionOutput[slice] = np.matmul(weights[head], v[slice]);
                }
                var afterAttention = np.matmul(attentionOutput, _weights[prefix + "attn_wo"].T) + input;
                var mlpNorm = Rms(afterAttention, out var mlpScale);
                var preHidden = np.matmul(mlpNorm, _weights[prefix + "mlp_fc1"].T);
                var hidden = np.maximum(preHidden, 0.0);
                x = np.matmul(hidden, _weights[prefix + "mlp_fc2"].T) + afterAttention;
                layers[layer] = new LayerTape(input, norm, attentionScale, q, k, v, weights, attentionOutput, afterAttention, mlpNorm, mlpScale, preHidden, hidden);
            }
            return new Tape(embedding, embeddingScale, layers, x, np.matmul(x, _weights["lm_head"].T));
        }

        private static NDArray Rms(NDArray x, out NDArray scale)
        {
            scale = np.power(np.mean(np.square(x), axis: 1, keepdims: true) + 1e-5, -.5);
            return x * scale;
        }
        private static NDArray RmsBackward(NDArray dy, NDArray x, NDArray scale)
        {
            var projection = np.sum(dy * x, axis: 1, keepdims: true);
            return dy * scale - x * ((scale * scale * scale) / x.shape[1]) * projection;
        }
        private static NDArray Softmax(NDArray logits)
        {
            var exp = np.exp(logits - np.max(logits, axis: 1, keepdims: true));
            return exp / np.sum(exp, axis: 1, keepdims: true);
        }
        private static double CrossEntropy(NDArray probabilities, IReadOnlyList<int> targets)
        {
            var logs = np.log(probabilities);
            var selected = np.empty(targets.Count, dtype: np.float64);
            for (int row = 0; row < targets.Count; row++) selected[row] = logs.item<double>(row, targets[row]);
            return -np.mean(selected, axis: 0).item<double>();
        }
        private void ValidateTokens(IReadOnlyList<int> tokens)
        {
            ThrowIfDisposed();
            if (tokens is null || tokens.Count == 0 || tokens.Count > BlockSize || tokens.Any(t => t < 0 || t >= VocabularySize))
                throw new ArgumentException("Tokens must be a nonempty, vocabulary-valid sequence no longer than the context.", nameof(tokens));
        }
        private void ValidatePair(IReadOnlyList<int> inputs, IReadOnlyList<int> targets)
        {
            ValidateTokens(inputs); ValidateTokens(targets);
            if (inputs.Count != targets.Count) throw new ArgumentException("Inputs and targets must have equal length.");
        }
        private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(nameof(Model)); }
        public void Dispose()
        {
            if (_disposed) return;
            foreach (var w in _weights.Values) w.Dispose();
            foreach (var m in _first.Values) m.Dispose();
            foreach (var v in _second.Values) v.Dispose();
            _disposed = true;
        }
        private sealed record Tape(NDArray Embedding, NDArray EmbeddingScale, LayerTape[] Layers, NDArray Final, NDArray Logits);
        private sealed record LayerTape(NDArray Input, NDArray AttentionNorm, NDArray AttentionScale, NDArray Q, NDArray K,
            NDArray V, NDArray[] Attention, NDArray AttentionOutput, NDArray AfterAttention, NDArray MlpNorm,
            NDArray MlpScale, NDArray PreHidden, NDArray Hidden);
    }

    public sealed class GradientBatch : IDisposable
    {
        public double Loss { get; }
        public NDArray Logits { get; }
        public IReadOnlyDictionary<string, NDArray> Parameters { get; }
        internal GradientBatch(double loss, NDArray logits, Dictionary<string, NDArray> parameters)
        {
            (Loss, Logits, Parameters) = (loss, logits, parameters);
            // An owning result object must outlive a caller's temporary array scope.
            NDScope.Detach(logits);
            foreach (var gradient in parameters.Values) NDScope.Detach(gradient);
        }
        public void Dispose() { Logits.Dispose(); foreach (var g in Parameters.Values) g.Dispose(); }
    }

    public static void Demo()
    {
        var tokenizer = new Tokenizer(TinyNames);
        using var model = new Model(tokenizer.Size);
        using var losses = model.Train(tokenizer, TinyNames); // source default: 1000 updates
        Console.WriteLine($"MicroGPT: {TinyNames.Length} local names, vocabulary={tokenizer.Size}, parameters={model.ParameterCount}, trained steps={model.OptimizerStep}");
        Console.WriteLine($"First loss={losses.item<double>(0):F4}; last loss={losses.item<double>(losses.size - 1):F4}");
        var samples = model.Generate(tokenizer);
        for (int i = 0; i < samples.Length; i++) Console.WriteLine($"sample {i + 1:D2}: {samples[i]}");
    }
}
