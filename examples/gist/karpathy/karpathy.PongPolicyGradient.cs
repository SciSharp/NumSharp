// Gist: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
// Pinned revision: 06d092624118444f7350a22653e8ba9d1c6e63d6; source: pg-pong.py.
// Andrej Karpathy; SHA256 cf764d11a0ebebb46d02c482c5a9c7c31081960d24b7c332757362735ad71a14.
// No license declaration appears in the pinned file. Independently expressed numerical
// implementation; no original prose/docstrings are reproduced. The demo is a bounded
// synthetic rollout, not an Atari environment or evidence of learning to play Pong.
using System;
using System.Collections.Generic;

namespace NumSharp.Examples.Gist.Karpathy;

public static class PongPolicyGradient
{
    public const int InputSize = 80 * 80;
    public const int HiddenUnits = 200;

    /// <summary>Owns independent float64 weights. Public arrays are borrowed until model disposal.</summary>
    public sealed class Model : IDisposable
    {
        public NDArray W1 { get; }
        public NDArray W2 { get; }
        public int Inputs => checked((int)W1.shape[1]);
        public int Hidden => checked((int)W1.shape[0]);

        public Model(NDArray w1, NDArray w2)
        {
            RequireDouble(w1, nameof(w1));
            RequireDouble(w2, nameof(w2));
            if (w1.ndim != 2 || w1.shape[0] < 1 || w1.shape[1] < 1 || w2.ndim != 1 || w2.size != w1.shape[0])
                throw new ArgumentException("Expected W1=(hidden,input) and W2=(hidden,).");
            using var scope = NDScope.Open();
            W1 = w1.copy(); W2 = w2.copy();
            NDScope.Detach(W1); NDScope.Detach(W2);
        }

        public void Dispose() { W1.Dispose(); W2.Dispose(); }
    }

    public static Model Initialize(NumPyRandom random, int inputs = InputSize, int hidden = HiddenUnits)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (inputs < 1 || hidden < 1) throw new ArgumentOutOfRangeException(nameof(inputs));
        using var scope = NDScope.Open();
        var w1 = random.randn(hidden, inputs) / np.sqrt(np.array((double)inputs));
        var w2 = random.randn(hidden) / np.sqrt(np.array((double)hidden));
        return new Model(w1, w2);
    }

    /// <summary>
    /// Crop rows35..194, take every second row/column and red channel, erase144/109,
    /// binarize, and return an owning float64 vector. Like the source, this MODIFIES
    /// the selected red-channel pixels in the supplied uint8 frame. Copy first if needed.
    /// </summary>
    public static NDArray PreprocessFrame(NDArray frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (frame.typecode != NPTypeCode.Byte || frame.ndim != 3 || frame.shape[0] != 210 || frame.shape[1] != 160 || frame.shape[2] != 3)
            throw new ArgumentException("Expected a 210x160x3 uint8 RGB frame.", nameof(frame));
        using var scope = NDScope.Open();
        var pixels = frame["35:195:2,::2,0"];
        pixels[pixels == 144] = (byte)0;
        pixels[pixels == 109] = (byte)0;
        pixels[pixels != 0] = (byte)1;
        return scope.Returns(pixels.astype(NPTypeCode.Double).ravel());
    }

    public static NDArray FrameDifference(NDArray current, NDArray? previous)
    {
        RequireVector(current, nameof(current));
        using var scope = NDScope.Open();
        if (previous is null) return scope.Returns(np.zeros_like(current));
        RequireVector(previous, nameof(previous));
        if (current.size != previous.size) throw new ArgumentException("Frames must have matching sizes.");
        return scope.Returns(current - previous);
    }

    public static double Sigmoid(double logit)
    {
        using var scope = NDScope.Open();
        return (1.0 / (1.0 + np.exp(np.array(-logit)))).item<double>();
    }

    public static int SampleAction(double probabilityOfUp, double uniformDraw)
    {
        RequireProbability(probabilityOfUp);
        if (!double.IsFinite(uniformDraw) || uniformDraw < 0 || uniformDraw >= 1)
            throw new ArgumentOutOfRangeException(nameof(uniformDraw));
        return uniformDraw < probabilityOfUp ? 2 : 3;
    }

    public static double ActionLogGradient(int action, double probabilityOfUp)
    {
        RequireProbability(probabilityOfUp);
        if (action != 2 && action != 3) throw new ArgumentOutOfRangeException(nameof(action));
        return (action == 2 ? 1.0 : 0.0) - probabilityOfUp;
    }

    public static (double Probability, NDArray Hidden) Forward(Model model, NDArray input)
    {
        ArgumentNullException.ThrowIfNull(model);
        RequireVector(input, nameof(input));
        if (input.size != model.Inputs) throw new ArgumentException("Input width does not match W1.", nameof(input));
        using var scope = NDScope.Open();
        var hidden = np.dot(model.W1, input);
        hidden[hidden < 0.0] = 0.0;
        double probability = Sigmoid(np.dot(model.W2, hidden).item<double>());
        return (probability, scope.Returns(hidden));
    }

    /// <summary>Reset backward reward accumulation at EVERY nonzero point reward; keep (T) or (T,1) shape.</summary>
    public static NDArray DiscountRewards(NDArray rewards, double gamma = .99)
    {
        RequireColumnOrVector(rewards, nameof(rewards));
        if (!double.IsFinite(gamma) || gamma < 0 || gamma > 1) throw new ArgumentOutOfRangeException(nameof(gamma));
        using var scope = NDScope.Open();
        var discounted = np.zeros_like(rewards);
        var flat = discounted.reshape(-1);
        double running = 0;
        for (long t = rewards.size - 1; t >= 0; t--)
        {
            double reward = rewards.item<double>(t);
            if (reward != 0) running = 0;
            running = running * gamma + reward;
            flat[t] = running;
        }
        return scope.Returns(discounted);
    }

    /// <summary>
    /// Center first, then compute std of that centered array, in the source's order.
    /// A zero/nonfinite std throws instead of allowing NaN advantages to poison weights.
    /// </summary>
    public static NDArray StandardizeReturns(NDArray discounted)
    {
        RequireColumnOrVector(discounted, nameof(discounted));
        if (discounted.size == 0) throw new ArgumentException("An episode must contain returns.", nameof(discounted));
        using var scope = NDScope.Open();
        var flat = discounted.reshape(-1);
        var centered = flat - np.mean(flat, axis: 0);
        // NumPy std recomputes the (usually near-zero) mean after the in-place centering.
        // Axis0 selects its pairwise summation order for these contiguous 1-D quantities.
        var residual = centered - np.mean(centered, axis: 0);
        double variance = np.sum(np.square(residual), axis: 0).item<double>() / residual.size;
        double deviation = np.sqrt(np.array(variance)).item<double>();
        if (!(deviation > 0) || !double.IsFinite(deviation))
            throw new ArgumentException("Cannot normalize zero-variance or nonfinite returns.", nameof(discounted));
        return scope.Returns((centered / deviation).reshape(discounted.shape));
    }

    public static (NDArray W1, NDArray W2) Backward(Model model, NDArray inputs, NDArray hidden, NDArray weightedLogGradients)
    {
        ValidateEpisode(model, inputs, hidden, weightedLogGradients);
        using var scope = NDScope.Open();
        var column = weightedLogGradients.reshape(-1, 1);
        var dw2 = np.dot(hidden.T, column).ravel();
        var dh = np.outer(column, model.W2);
        dh[hidden <= 0.0] = 0.0;
        var dw1 = np.dot(dh.T, inputs);
        return scope.Returns((dw1, dw2));
    }

    /// <summary>One ascent update; modifies weights/cache and clears the supplied accumulated gradient.</summary>
    public static void RmsPropUpdate(NDArray weights, NDArray gradient, NDArray cache,
        double learningRate = 1e-4, double decayRate = .99, double epsilon = 1e-5)
    {
        RequireDouble(weights, nameof(weights)); RequireDouble(gradient, nameof(gradient)); RequireDouble(cache, nameof(cache));
        if (!weights.shape.AsSpan().SequenceEqual(gradient.shape) || !weights.shape.AsSpan().SequenceEqual(cache.shape))
            throw new ArgumentException("Weights, gradient and cache shapes must match.");
        ValidateOptimizer(learningRate, decayRate, epsilon);
        using var scope = NDScope.Open();
        var updatedCache = decayRate * cache + (1 - decayRate) * np.square(gradient);
        var update = learningRate * gradient / (np.sqrt(updatedCache) + epsilon);
        np.copyto(cache, updatedCache);
        np.add(weights, update, @out: weights);
        gradient.fill(0.0);
    }

    /// <summary>Owns optimizer state, borrows Model, and applies updates after a configurable episode batch.</summary>
    public sealed class Trainer : IDisposable
    {
        public Model Model { get; }
        public NDArray GradientW1 { get; }
        public NDArray GradientW2 { get; }
        public NDArray CacheW1 { get; }
        public NDArray CacheW2 { get; }
        public int BatchSize { get; }
        public double LearningRate { get; }
        public double Gamma { get; }
        public double DecayRate { get; }
        public int Episodes { get; private set; }
        public int Updates { get; private set; }
        public double? RunningReward { get; private set; }
        public double LastEpisodeReward { get; private set; }

        public Trainer(Model model, int batchSize = 10, double learningRate = 1e-4, double gamma = .99, double decayRate = .99)
        {
            ArgumentNullException.ThrowIfNull(model);
            if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (!double.IsFinite(gamma) || gamma < 0 || gamma > 1) throw new ArgumentOutOfRangeException(nameof(gamma));
            ValidateOptimizer(learningRate, decayRate, 1e-5);
            Model = model; BatchSize = batchSize; LearningRate = learningRate; Gamma = gamma; DecayRate = decayRate;
            using var scope = NDScope.Open();
            GradientW1 = np.zeros_like(model.W1); GradientW2 = np.zeros_like(model.W2);
            CacheW1 = np.zeros_like(model.W1); CacheW2 = np.zeros_like(model.W2);
            NDScope.Detach(GradientW1); NDScope.Detach(GradientW2); NDScope.Detach(CacheW1); NDScope.Detach(CacheW2);
        }

        public void Accumulate(NDArray dw1, NDArray dw2)
        {
            RequireDouble(dw1, nameof(dw1)); RequireDouble(dw2, nameof(dw2));
            if (!dw1.shape.AsSpan().SequenceEqual(Model.W1.shape) || !dw2.shape.AsSpan().SequenceEqual(Model.W2.shape))
                throw new ArgumentException("Gradient shapes must match model weights.");
            np.add(GradientW1, dw1, @out: GradientW1);
            np.add(GradientW2, dw2, @out: GradientW2);
        }

        public bool TrainEpisode(NDArray inputs, NDArray hidden, NDArray actionLogGradients, NDArray rewards)
        {
            ValidateEpisode(Model, inputs, hidden, actionLogGradients);
            RequireColumnOrVector(rewards, nameof(rewards));
            if (rewards.size != inputs.shape[0]) throw new ArgumentException("One reward per time step is required.");
            using var scope = NDScope.Open();
            var advantages = StandardizeReturns(DiscountRewards(rewards, Gamma)).reshape(-1, 1);
            var weighted = actionLogGradients.reshape(-1, 1) * advantages;
            var (dw1, dw2) = Backward(Model, inputs, hidden, weighted);
            Accumulate(dw1, dw2);
            Episodes++;
            bool update = Episodes % BatchSize == 0;
            if (update)
            {
                RmsPropUpdate(Model.W1, GradientW1, CacheW1, LearningRate, DecayRate);
                RmsPropUpdate(Model.W2, GradientW2, CacheW2, LearningRate, DecayRate);
                Updates++;
            }
            double sum = 0;
            for (long i = 0; i < rewards.size; i++) sum += rewards.item<double>(i);
            LastEpisodeReward = sum;
            RunningReward = RunningReward is null ? sum : RunningReward.Value * .99 + sum * .01;
            return update;
        }

        public void Dispose() { GradientW1.Dispose(); GradientW2.Dispose(); CacheW1.Dispose(); CacheW2.Dispose(); }
    }

    public readonly record struct TrainingSummary(int Episodes, int Updates, double RunningReward, double ParameterChangeL2, double LastProbability);

    /// <summary>Full-size network on synthetic moving-pixel observations and invented point rewards; never starts Gym.</summary>
    public static TrainingSummary RunSyntheticTraining(int episodes = 2, int stepsPerEpisode = 6, int seed = 0)
    {
        if (episodes < 1 || stepsPerEpisode < 2) throw new ArgumentOutOfRangeException(nameof(episodes));
        var random = np.random.RandomState(seed);
        using var model = Initialize(random);
        using var trainer = new Trainer(model, batchSize: 2);
        using var initial = model.W1.copy();
        double lastProbability = .5;
        for (int episode = 0; episode < episodes; episode++)
        {
            using var scope = NDScope.Open();
            var xs = new List<NDArray>(); var hs = new List<NDArray>();
            var logGradients = new double[stepsPerEpisode]; var rewards = new double[stepsPerEpisode];
            NDArray? previous = null;
            for (int t = 0; t < stepsPerEpisode; t++)
            {
                using var frame = np.full(new Shape(210, 160, 3), (byte)144, dtype: np.uint8);
                frame[35 + 2 * ((t * 3 + episode) % 80), 2 * ((t * 7 + 5) % 80), 0] = (byte)255;
                var current = PreprocessFrame(frame);
                var input = FrameDifference(current, previous);
                previous?.Dispose(); previous = current;
                var (probability, hidden) = Forward(model, input);
                lastProbability = probability;
                int action = SampleAction(probability, random.uniform().item<double>());
                xs.Add(input); hs.Add(hidden);
                logGradients[t] = ActionLogGradient(action, probability);
                // Deliberately fabricated terminal point reward, not an Atari score.
                if (t == stepsPerEpisode - 1) rewards[t] = action == (episode % 2 == 0 ? 2 : 3) ? 1 : -1;
            }
            trainer.TrainEpisode(np.vstack(xs.ToArray()), np.vstack(hs.ToArray()),
                np.array(logGradients).reshape(-1, 1), np.array(rewards).reshape(-1, 1));
        }
        using var difference = model.W1 - initial;
        using var norm = np.linalg.norm(difference);
        return new TrainingSummary(trainer.Episodes, trainer.Updates, trainer.RunningReward ?? 0, norm.item<double>(), lastProbability);
    }

    public static void Demo()
    {
        var run = RunSyntheticTraining();
        Console.WriteLine($"Synthetic Pong-policy rollout (not Atari): H={HiddenUnits}, D={InputSize}, episodes={run.Episodes}, updates={run.Updates}, ||delta W1||={run.ParameterChangeL2:G6}, last p(up)={run.LastProbability:F6}");
    }

    private static void ValidateEpisode(Model model, NDArray inputs, NDArray hidden, NDArray logGradients)
    {
        ArgumentNullException.ThrowIfNull(model);
        RequireDouble(inputs, nameof(inputs)); RequireDouble(hidden, nameof(hidden)); RequireColumnOrVector(logGradients, nameof(logGradients));
        if (inputs.ndim != 2 || hidden.ndim != 2 || inputs.shape[0] < 1 || inputs.shape[1] != model.Inputs ||
            hidden.shape[0] != inputs.shape[0] || hidden.shape[1] != model.Hidden || logGradients.size != inputs.shape[0])
            throw new ArgumentException("Expected inputs=(T,D), hidden=(T,H), log-gradients=(T) or (T,1).");
    }

    private static void RequireDouble(NDArray value, string name)
    {
        ArgumentNullException.ThrowIfNull(value, name);
        if (value.typecode != NPTypeCode.Double) throw new ArgumentException("Expected float64.", name);
    }

    private static void RequireVector(NDArray value, string name)
    {
        RequireDouble(value, name);
        if (value.ndim != 1) throw new ArgumentException("Expected a vector.", name);
    }

    private static void RequireColumnOrVector(NDArray value, string name)
    {
        RequireDouble(value, name);
        if (value.ndim != 1 && !(value.ndim == 2 && value.shape[1] == 1))
            throw new ArgumentException("Expected a vector or a single column.", name);
    }

    private static void RequireProbability(double value)
    {
        if (!double.IsFinite(value) || value < 0 || value > 1) throw new ArgumentOutOfRangeException(nameof(value));
    }

    private static void ValidateOptimizer(double learningRate, double decayRate, double epsilon)
    {
        if (!double.IsFinite(learningRate) || learningRate <= 0 || !double.IsFinite(decayRate) || decayRate < 0 || decayRate >= 1 ||
            !double.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
    }
}
