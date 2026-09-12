// Andrej Karpathy's minimal character RNN: https://gist.github.com/karpathy/d4dee566867f8291f086
// Immutable raw file: https://gist.githubusercontent.com/karpathy/d4dee566867f8291f086/raw/45350b64ab738fa1ca2bf170969494339eab5633/min-char-rnn.py
// Raw-file revision 45350b64ab738fa1ca2bf170969494339eab5633 is NOT a verified gist HEAD.
// Fetched source SHA256: 98a0eaa61092f9541e883f3e8b6c02dae28b5658dbc5f692ffe0881be37a390a.
// The source header declares BSD License. Independently expressed C# numerical implementation.

namespace NumSharp.Examples.Gist.Karpathy;

/// <summary>A complete float64 vanilla RNN: forward pass, BPTT, clipped gradients and Adagrad.</summary>
public static class MinimalCharacterRnn
{
    /// <summary>Sorted UTF-16 characters make the original set-based vocabulary reproducible.</summary>
    public sealed class Vocabulary
    {
        private readonly char[] _characters;
        private readonly Dictionary<char, int> _indices;
        public int Count => _characters.Length;
        public string Characters => new(_characters);
        public Vocabulary(string text)
        {
            ArgumentException.ThrowIfNullOrEmpty(text);
            _characters = text.Distinct().OrderBy(c => c).ToArray();
            _indices = _characters.Select((c, i) => (c, i)).ToDictionary(pair => pair.c, pair => pair.i);
        }
        public int[] Encode(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            return text.Select(c => _indices.TryGetValue(c, out var index) ? index
                : throw new ArgumentException($"Character U+{(int)c:X4} is outside this vocabulary.", nameof(text))).ToArray();
        }
        public string Decode(IEnumerable<int> tokens)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            return new string(tokens.Select(i => i >= 0 && i < Count ? _characters[i]
                : throw new ArgumentOutOfRangeException(nameof(tokens))).ToArray());
        }
    }

    /// <summary>Owns five arrays in source order: Wxh, Whh, Why, bh, by. Properties are borrowed.</summary>
    public sealed class ParameterSet : IDisposable
    {
        public NDArray Wxh { get; }
        public NDArray Whh { get; }
        public NDArray Why { get; }
        public NDArray Bh { get; }
        public NDArray By { get; }
        public IReadOnlyList<NDArray> Arrays { get; }
        internal ParameterSet(NDArray wxh, NDArray whh, NDArray why, NDArray bh, NDArray by)
        {
            Wxh = wxh; Whh = whh; Why = why; Bh = bh; By = by;
            Arrays = Array.AsReadOnly(new[] { wxh, whh, why, bh, by });
            foreach (var array in Arrays) NDScope.Detach(array);
        }
        public ParameterSet Copy() => new(Wxh.copy(), Whh.copy(), Why.copy(), Bh.copy(), By.copy());
        internal static ParameterSet ZerosLike(ParameterSet p) => new(np.zeros_like(p.Wxh), np.zeros_like(p.Whh),
            np.zeros_like(p.Why), np.zeros_like(p.Bh), np.zeros_like(p.By));
        public void Dispose() { foreach (var array in Arrays) array.Dispose(); }
    }

    /// <summary>Owns the five gradients, final H×1 state, and time-major state/probability traces.</summary>
    public sealed class LossResult : IDisposable
    {
        public double Loss { get; }
        public ParameterSet Gradients { get; }
        public NDArray LastHidden { get; }
        public NDArray HiddenStates { get; }
        public NDArray Probabilities { get; }
        internal LossResult(double loss, ParameterSet gradients, NDArray lastHidden, NDArray hidden, NDArray probabilities)
        {
            Loss = loss; Gradients = gradients; LastHidden = lastHidden; HiddenStates = hidden; Probabilities = probabilities;
            NDScope.Detach(lastHidden); NDScope.Detach(hidden); NDScope.Detach(probabilities);
        }
        public void Dispose() { Gradients.Dispose(); LastHidden.Dispose(); HiddenStates.Dispose(); Probabilities.Dispose(); }
    }

    public sealed record TrainingStep(int Iteration, int Position, bool ResetHidden, double Loss, double SmoothLoss);
    public sealed record SampleRecord(int Iteration, int[] Tokens);
    public sealed class TrainingResult : IDisposable
    {
        public IReadOnlyList<TrainingStep> Steps { get; }
        public IReadOnlyList<SampleRecord> Samples { get; }
        public NDArray LastHidden { get; }
        internal TrainingResult(List<TrainingStep> steps, List<SampleRecord> samples, NDArray hidden)
        { Steps = steps.AsReadOnly(); Samples = samples.AsReadOnly(); LastHidden = hidden; NDScope.Detach(hidden); }
        public void Dispose() => LastHidden.Dispose();
    }

    /// <summary>Owns its parameters, optimizer accumulators and independent seeded RandomState.</summary>
    public sealed class Model : IDisposable
    {
        private readonly NumPyRandom _random;
        private bool _disposed;
        public int HiddenSize { get; }
        public int VocabularySize { get; }
        public ParameterSet Parameters { get; }
        public ParameterSet AdagradMemory { get; }

        public Model(int vocabularySize, int hiddenSize = 100, int seed = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabularySize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
            VocabularySize = vocabularySize; HiddenSize = hiddenSize;
            _random = np.random.RandomState(seed);
            using var scope = NDScope.Open();
            Parameters = new ParameterSet(_random.randn(hiddenSize, vocabularySize) * .01,
                _random.randn(hiddenSize, hiddenSize) * .01, _random.randn(vocabularySize, hiddenSize) * .01,
                np.zeros((hiddenSize, 1), dtype: np.float64), np.zeros((vocabularySize, 1), dtype: np.float64));
            AdagradMemory = ParameterSet.ZerosLike(Parameters);
        }

        private void ValidateHidden(NDArray hidden)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(hidden);
            if (hidden.typecode != NPTypeCode.Double || !hidden.shape.SequenceEqual(new long[] { HiddenSize, 1 }))
                throw new ArgumentException("Hidden state must be float64 with shape (hiddenSize,1).", nameof(hidden));
        }
        private void ValidateTokens(IReadOnlyList<int> tokens, string name)
        {
            ArgumentNullException.ThrowIfNull(tokens, name);
            if (tokens.Any(token => token < 0 || token >= VocabularySize)) throw new ArgumentOutOfRangeException(name);
        }
        private NDArray OneHot(int token)
        {
            var x = np.zeros((VocabularySize, 1), dtype: np.float64);
            x[token, 0] = 1.0;
            return x;
        }
        private static NDArray ProbabilitiesOf(NDArray logits)
        {
            using var scope = NDScope.Open();
            var numerator = np.exp(logits);
            // Explicit axis keeps NumPy's pairwise reduction order for the 1-D buffer.
            var denominator = np.sum(numerator.ravel('K'), axis: 0);
            double total = denominator.item<double>();
            if (!double.IsFinite(total) || total <= 0)
                throw new ArithmeticException("The source softmax overflowed or underflowed; reduce the learning rate or parameter magnitudes.");
            return scope.Returns(numerator / denominator);
        }

        /// <summary>No parameters are mutated. Set clipGradients=false for finite-difference checks.</summary>
        public LossResult LossAndGradients(IReadOnlyList<int> inputs, IReadOnlyList<int> targets,
            NDArray initialHidden, bool clipGradients = true)
        {
            ValidateHidden(initialHidden); ValidateTokens(inputs, nameof(inputs)); ValidateTokens(targets, nameof(targets));
            if (inputs.Count == 0 || inputs.Count != targets.Count)
                throw new ArgumentException("Equal, nonempty input and next-character target sequences are required.");
            using var scope = NDScope.Open();
            int count = inputs.Count;
            var states = new NDArray[count + 1]; states[0] = initialHidden.copy();
            var encoded = new NDArray[count]; var probabilities = new NDArray[count];
            var stateTrace = np.empty((count, HiddenSize), dtype: np.float64);
            var probabilityTrace = np.empty((count, VocabularySize), dtype: np.float64);
            double loss = 0;
            for (int t = 0; t < count; t++)
            {
                using var step = NDScope.Open();
                encoded[t] = step.Returns(OneHot(inputs[t]));
                states[t + 1] = step.Returns(np.tanh(np.dot(Parameters.Wxh, encoded[t]) +
                    np.dot(Parameters.Whh, states[t]) + Parameters.Bh));
                probabilities[t] = step.Returns(ProbabilitiesOf(np.dot(Parameters.Why, states[t + 1]) + Parameters.By));
                loss -= np.log(np.array(probabilities[t].item<double>(targets[t], 0))).item<double>();
                stateTrace[$"{t},:"] = states[t + 1].reshape(-1);
                probabilityTrace[$"{t},:"] = probabilities[t].reshape(-1);
            }
            var gradients = ParameterSet.ZerosLike(Parameters);
            try
            {
                var futureGradient = np.zeros((HiddenSize, 1), dtype: np.float64);
                for (int t = count - 1; t >= 0; t--)
                {
                    using var step = NDScope.Open();
                    var outputGradient = probabilities[t].copy();
                    outputGradient[targets[t], 0] = outputGradient.item<double>(targets[t], 0) - 1.0;
                    np.add(gradients.Why, np.dot(outputGradient, states[t + 1].T), @out: gradients.Why);
                    np.add(gradients.By, outputGradient, @out: gradients.By);
                    var hiddenGradient = np.dot(Parameters.Why.T, outputGradient) + futureGradient;
                    var preactivationGradient = (1.0 - states[t + 1] * states[t + 1]) * hiddenGradient;
                    np.add(gradients.Bh, preactivationGradient, @out: gradients.Bh);
                    np.add(gradients.Wxh, np.dot(preactivationGradient, encoded[t].T), @out: gradients.Wxh);
                    np.add(gradients.Whh, np.dot(preactivationGradient, states[t].T), @out: gradients.Whh);
                    var previous = futureGradient;
                    futureGradient = step.Returns(np.dot(Parameters.Whh.T, preactivationGradient));
                    previous.Dispose();
                }
                if (clipGradients) ClipGradients(gradients);
                return new LossResult(loss, gradients, states[count].copy(), stateTrace, probabilityTrace);
            }
            catch { gradients.Dispose(); throw; }
        }

        public void ApplyAdagrad(ParameterSet gradients, double learningRate = .1)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(gradients);
            if (!double.IsFinite(learningRate) || learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
            for (int i = 0; i < 5; i++)
                if (gradients.Arrays[i].typecode != NPTypeCode.Double || !gradients.Arrays[i].shape.SequenceEqual(Parameters.Arrays[i].shape))
                    throw new ArgumentException("Gradient shapes and float64 dtype must match the parameters.", nameof(gradients));
            using var scope = NDScope.Open();
            for (int i = 0; i < 5; i++)
            {
                var parameter = Parameters.Arrays[i]; var gradient = gradients.Arrays[i]; var memory = AdagradMemory.Arrays[i];
                np.add(memory, gradient * gradient, @out: memory);
                np.add(parameter, -learningRate * gradient / np.sqrt(memory + 1e-8), @out: parameter);
            }
        }

        public int[] Sample(NDArray initialHidden, int seedToken, int count, NumPyRandom? random = null)
        {
            ValidateHidden(initialHidden);
            if (seedToken < 0 || seedToken >= VocabularySize) throw new ArgumentOutOfRangeException(nameof(seedToken));
            ArgumentOutOfRangeException.ThrowIfNegative(count);
            random ??= _random;
            var sampled = new int[count];
            using var scope = NDScope.Open();
            var hidden = initialHidden.copy();
            int token = seedToken;
            for (int t = 0; t < count; t++)
            {
                using var step = NDScope.Open();
                var next = step.Returns(np.tanh(np.dot(Parameters.Wxh, OneHot(token)) + np.dot(Parameters.Whh, hidden) + Parameters.Bh));
                hidden.Dispose(); hidden = next;
                var probability = ProbabilitiesOf(np.dot(Parameters.Why, hidden) + Parameters.By);
                var weights = new double[VocabularySize];
                for (int i = 0; i < weights.Length; i++) weights[i] = probability.item<double>(i, 0);
                token = checked((int)random.choice(VocabularySize, p: weights).item<long>());
                sampled[t] = token;
            }
            return sampled;
        }

        /// <summary>
        /// Bounded version of the source's sequence scan: reset at its original boundary,
        /// sample BEFORE an update, then loss/BPTT, smoothing and Adagrad. Sampling consumes
        /// this model's RNG unless supplied separately to Sample; training itself is deterministic.
        /// </summary>
        public TrainingResult Train(IReadOnlyList<int> corpus, int iterations, int sequenceLength = 25,
            double learningRate = .1, int sampleEvery = 100, int sampleLength = 200)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ValidateTokens(corpus, nameof(corpus));
            ArgumentOutOfRangeException.ThrowIfNegative(iterations);
            if (sequenceLength <= 0 || corpus.Count <= sequenceLength) throw new ArgumentOutOfRangeException(nameof(sequenceLength));
            if (!double.IsFinite(learningRate) || learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
            ArgumentOutOfRangeException.ThrowIfNegative(sampleEvery); ArgumentOutOfRangeException.ThrowIfNegative(sampleLength);
            using var scope = NDScope.Open();
            var hidden = np.zeros((HiddenSize, 1), dtype: np.float64);
            var records = new List<TrainingStep>(); var samples = new List<SampleRecord>();
            double smoothLoss = -np.log(np.array(1.0 / VocabularySize)).item<double>() * sequenceLength;
            int position = 0;
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                using var step = NDScope.Open();
                bool reset = iteration == 0 || position + sequenceLength + 1 >= corpus.Count;
                if (reset) { hidden.Dispose(); hidden = step.Returns(np.zeros((HiddenSize, 1), dtype: np.float64)); position = 0; }
                var inputs = Enumerable.Range(position, sequenceLength).Select(i => corpus[i]).ToArray();
                var targets = Enumerable.Range(position + 1, sequenceLength).Select(i => corpus[i]).ToArray();
                if (sampleEvery > 0 && iteration % sampleEvery == 0)
                    samples.Add(new SampleRecord(iteration, Sample(hidden, inputs[0], sampleLength)));
                using var result = LossAndGradients(inputs, targets, hidden);
                smoothLoss = smoothLoss * .999 + result.Loss * .001;
                records.Add(new TrainingStep(iteration, position, reset, result.Loss, smoothLoss));
                ApplyAdagrad(result.Gradients, learningRate);
                hidden.Dispose(); hidden = step.Returns(result.LastHidden.copy());
                position += sequenceLength;
            }
            return new TrainingResult(records, samples, hidden);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true; Parameters.Dispose(); AdagradMemory.Dispose();
        }
    }

    public static void ClipGradients(ParameterSet gradients, double limit = 5.0)
    {
        ArgumentNullException.ThrowIfNull(gradients);
        if (!double.IsFinite(limit) || limit <= 0) throw new ArgumentOutOfRangeException(nameof(limit));
        foreach (var gradient in gradients.Arrays) np.clip(gradient, -limit, limit, @out: gradient);
    }

    public static void Demo()
    {
        string corpus = string.Concat(Enumerable.Repeat("hello world\n", 40));
        var vocabulary = new Vocabulary(corpus);
        using var model = new Model(vocabulary.Count, hiddenSize: 16, seed: 1);
        using var training = model.Train(vocabulary.Encode(corpus), 40, sampleEvery: 20, sampleLength: 40);
        Console.WriteLine($"RNN: characters={corpus.Length}, vocabulary={vocabulary.Count}, hidden=16, updates={training.Steps.Count}");
        Console.WriteLine($"Loss: initial={training.Steps[0].Loss:G17}, final={training.Steps[^1].Loss:G17}");
        foreach (var sample in training.Samples)
            Console.WriteLine($"Sample {sample.Iteration}: {vocabulary.Decode(sample.Tokens).Replace("\n", "\\n")}");
    }
}
