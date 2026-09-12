// Gist: https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d — Andrej Karpathy, nes.py
// Pinned: https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d/2d87bda1fe2a1711eac65c65a2d215f6cfb799e3#file-nes-py
// No explicit license was found in the pinned gist. Independent implementation of its numerical algorithm;
// original source SHA256: 4144833dab064335631040b233d19aea2392f1e460a2cae1491a5def5d912f3b.
using NumSharp;

namespace NumSharp.Examples.Gist;

public static class NaturalEvolutionStrategies
{
    public static double QuadraticReward(NDArray weights, NDArray solution)
    {
        using var scope = NDScope.Open();
        if (weights.ndim != 1 || !weights.shape.SequenceEqual(solution.shape) ||
            weights.typecode != NPTypeCode.Double || solution.typecode != NPTypeCode.Double)
            throw new ArgumentException("Weights and solution must be matching float64 vectors.");
        return -np.sum(np.square(solution - weights)).item<double>();
    }

    public static NDArray Rewards(NDArray weights, NDArray noise, Func<NDArray, double> objective, double sigma = .1)
    {
        Validate(weights, noise, sigma);
        ArgumentNullException.ThrowIfNull(objective);
        using var scope = NDScope.Open();
        var result = np.zeros(noise.shape[0], dtype: np.float64);
        for (long row = 0; row < noise.shape[0]; row++)
        {
            using var iteration = NDScope.Open();
            result[row] = objective(weights + sigma * noise[$"{row},:"]);
        }
        return scope.Returns(result);
    }

    public static NDArray Step(NDArray weights, NDArray noise, NDArray rewards, double sigma = .1, double alpha = .001)
    {
        Validate(weights, noise, sigma);
        if (rewards.ndim != 1 || rewards.size != noise.shape[0] || rewards.typecode != NPTypeCode.Double)
            throw new ArgumentException("One float64 reward per noise row is required.");
        if (!double.IsFinite(alpha) || alpha <= 0) throw new ArgumentOutOfRangeException(nameof(alpha));
        using var scope = NDScope.Open();
        // The vector's explicit axis selects NumPy-style pairwise folding rather than the
        // flat SIMD accumulator. Match NumPy's two-pass population variance composition:
        // the old final point agreed by chance while 118 intermediate weight/reward scalars differed.
        var mean = np.mean(rewards, axis: 0);
        var centered = rewards - mean;
        double deviation = np.sqrt(np.mean(np.square(centered), axis: 0)).item<double>();
        if (!double.IsFinite(deviation) || deviation == 0)
            throw new ArgumentException("Reward standard deviation must be finite and nonzero.");
        var advantages = centered / deviation;
        return scope.Returns(weights + alpha / (noise.shape[0] * sigma) * np.dot(noise.T, advantages));
    }

    public static NDArray OptimizeQuadratic(NDArray solution, int iterations = 300, int population = 50,
        double sigma = .1, double alpha = .001, int seed = 0,
        Action<int, NDArray, double>? progress = null)
    {
        if (solution.ndim != 1 || solution.size == 0 || solution.typecode != NPTypeCode.Double)
            throw new ArgumentException("A nonempty float64 solution vector is required.");
        if (iterations < 0 || population < 2) throw new ArgumentOutOfRangeException(nameof(iterations));
        if (!double.IsFinite(sigma) || sigma <= 0) throw new ArgumentOutOfRangeException(nameof(sigma));
        if (!double.IsFinite(alpha) || alpha <= 0) throw new ArgumentOutOfRangeException(nameof(alpha));
        var random = np.random.RandomState(seed);
        NDArray weights = random.randn(solution.size);
        try
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                // Report the pre-update state, matching the gist's printed checkpoints.
                // This array is borrowed: observers must not mutate it or retain it without copying.
                progress?.Invoke(iteration, weights, QuadraticReward(weights, solution));
                using var noise = random.randn(population, solution.size);
                using var rewards = Rewards(weights, noise, trial => QuadraticReward(trial, solution), sigma);
                var updated = Step(weights, noise, rewards, sigma, alpha);
                weights.Dispose();
                weights = updated;
            }
            progress?.Invoke(iterations, weights, QuadraticReward(weights, solution));
            return weights;
        }
        catch { weights.Dispose(); throw; }
    }

    private static void Validate(NDArray weights, NDArray noise, double sigma)
    {
        if (weights.ndim != 1 || noise.ndim != 2 || noise.shape[1] != weights.size || noise.shape[0] < 2)
            throw new ArgumentException("Noise must have shape (population >= 2, weights.size).");
        if (weights.typecode != NPTypeCode.Double || noise.typecode != NPTypeCode.Double)
            throw new ArgumentException("This example uses float64 weights and perturbations.");
        if (!double.IsFinite(sigma) || sigma <= 0) throw new ArgumentOutOfRangeException(nameof(sigma));
    }

    public static void Demo()
    {
        using var solution = np.array(new[] { .5, .1, -.3 });
        using var fitted = OptimizeQuadratic(solution);
        Console.WriteLine($"NES: solution=[{fitted.item<double>(0):F6}, {fitted.item<double>(1):F6}, {fitted.item<double>(2):F6}], reward={QuadraticReward(fitted, solution):G8}");
    }
}
