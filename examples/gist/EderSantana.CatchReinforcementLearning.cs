// Gist: https://gist.github.com/EderSantana/c7222daa328f0e885093 — Eder Santana, qlearn.py
// Pinned: https://gist.github.com/EderSantana/c7222daa328f0e885093/90a8dafa550b581abbe2d578eb10acec0fe3dcd3#file-qlearn-py
// The pinned CATCH_Keras_RL.md declares MIT; see SOURCES.md for provenance.
// Functional NumSharp environment and replay-target core. A prediction delegate replaces Keras;
// neural-network training, model serialization and graphical playback are deliberately outside this port.
using NumSharp;

namespace NumSharp.Examples.Gist;

public static class CatchReinforcementLearning
{
    /// <summary>The original epsilon-greedy NumPy action selector; the caller owns the RNG.</summary>
    public static int SelectAction(NDArray predictedQ, double epsilon, NumPyRandom random)
    {
        ArgumentNullException.ThrowIfNull(predictedQ);
        ArgumentNullException.ThrowIfNull(random);
        if (predictedQ.ndim != 2 || predictedQ.shape[0] != 1 || predictedQ.shape[1] == 0)
            throw new ArgumentException("Predicted Q values must have shape (1, actions).", nameof(predictedQ));
        if (!double.IsFinite(epsilon) || epsilon < 0 || epsilon > 1)
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        using var scope = NDScope.Open();
        return random.rand().item<double>() <= epsilon
            ? random.randint(0, predictedQ.shape[1]).item<int>()
            : checked((int)np.argmax(predictedQ["0,:"]));
    }

    public sealed class CatchGame : IDisposable
    {
        private NDArray _state;
        public int GridSize { get; }
        public bool IsOver => _state.item<long>(0, 0) == GridSize - 1;

        public CatchGame(int gridSize = 10, int seed = 0)
        {
            if (gridSize < 4) throw new ArgumentOutOfRangeException(nameof(gridSize));
            GridSize = gridSize;
            _state = np.zeros((1, 3), dtype: np.int64);
            NDScope.Detach(_state);
            Reset(np.random.RandomState(seed));
        }

        public void Reset(NumPyRandom random)
        {
            ArgumentNullException.ThrowIfNull(random);
            using var scope = NDScope.Open();
            // Explicit scalar extraction repairs the original mixed scalar/size-one-array reset,
            // which NumPy 2 rejects as an inhomogeneous np.asarray input.
            Reset(random.randint(0, GridSize - 1).item<int>(), random.randint(1, GridSize - 2).item<int>());
        }

        public void Reset(int fruitColumn, int basketColumn)
        {
            if (fruitColumn < 0 || fruitColumn >= GridSize) throw new ArgumentOutOfRangeException(nameof(fruitColumn));
            if (basketColumn < 1 || basketColumn >= GridSize) throw new ArgumentOutOfRangeException(nameof(basketColumn));
            _state[0, 0] = 0L;
            _state[0, 1] = (long)fruitColumn;
            _state[0, 2] = (long)basketColumn;
        }

        public NDArray State() => _state.copy();

        public NDArray Observe()
        {
            using var scope = NDScope.Open();
            var image = np.zeros((GridSize, GridSize), dtype: np.float64);
            long row = _state.item<long>(0, 0), column = _state.item<long>(0, 1), basket = _state.item<long>(0, 2);
            image[row, column] = 1.0;
            // NumPy truncates the slice at the right border: a basket at GridSize-1 has width two.
            image[$"-1,{basket - 1}:{basket + 2}"] = 1.0;
            return scope.Returns(image.reshape(1, -1));
        }

        public (NDArray Observation, int Reward, bool GameOver) Act(int action)
        {
            if (action < 0 || action > 2) throw new ArgumentOutOfRangeException(nameof(action));
            if (IsOver) throw new InvalidOperationException("Reset a completed game before acting again.");
            long basket = Math.Clamp(_state.item<long>(0, 2) + action - 1, 1, GridSize - 1);
            _state[0, 0] = _state.item<long>(0, 0) + 1;
            _state[0, 2] = basket;
            int reward = IsOver ? (Math.Abs(_state.item<long>(0, 1) - basket) <= 1 ? 1 : -1) : 0;
            return (Observe(), reward, IsOver);
        }

        public void Dispose() => _state.Dispose();
    }

    public sealed class ReplayMemory : IDisposable
    {
        private sealed record Transition(NDArray State, int Action, double Reward, NDArray NextState, bool Terminal);
        private readonly List<Transition> _memory = new();
        public int Capacity { get; }
        public double Discount { get; }
        public int Count => _memory.Count;

        public ReplayMemory(int capacity = 100, double discount = .9)
        {
            if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (!double.IsFinite(discount) || discount < 0 || discount > 1) throw new ArgumentOutOfRangeException(nameof(discount));
            Capacity = capacity;
            Discount = discount;
        }

        public void Remember(NDArray state, int action, double reward, NDArray nextState, bool terminal)
        {
            if (state.ndim != 2 || state.shape[0] != 1 || !state.shape.SequenceEqual(nextState.shape))
                throw new ArgumentException("States must have identical (1, features) shapes.");
            if (action < 0 || !double.IsFinite(reward)) throw new ArgumentException("Action and reward must be valid.");
            if (_memory.Count > 0 && state.shape[1] != _memory[0].State.shape[1])
                throw new ArgumentException("Every transition must use the same feature count.");
            // Copies belong to this persistent memory, not to a caller's per-step temporary scope.
            // Keep both tracked until construction succeeds, then detach the committed state.
            using var copyScope = NDScope.Open();
            var saved = state.copy();
            var savedNext = nextState.copy();
            _memory.Add(new Transition(saved, action, reward, savedNext, terminal));
            NDScope.Detach(saved);
            NDScope.Detach(savedNext);
            if (_memory.Count > Capacity)
            {
                _memory[0].State.Dispose();
                _memory[0].NextState.Dispose();
                _memory.RemoveAt(0);
            }
        }

        public (NDArray Inputs, NDArray Targets) GetBatch(Func<NDArray, NDArray> predict, int actions,
            int batchSize = 10, NumPyRandom? random = null)
        {
            if (_memory.Count == 0) throw new InvalidOperationException("Replay memory is empty.");
            if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize));
            using var samples = (random ?? np.random).randint(0, _memory.Count, new Shape(Math.Min(Count, batchSize)));
            var indices = new int[samples.size];
            for (int i = 0; i < indices.Length; i++) indices[i] = samples.item<int>(i);
            return GetBatchAt(predict, actions, indices);
        }

        // Explicit sampled indices expose the numerical core for deterministic Python/C# parity.
        // Predict must return an owning array and must not mutate or retain the borrowed input.
        public (NDArray Inputs, NDArray Targets) GetBatchAt(Func<NDArray, NDArray> predict, int actions, params int[] indices)
        {
            ArgumentNullException.ThrowIfNull(predict);
            if (_memory.Count == 0) throw new InvalidOperationException("Replay memory is empty.");
            if (actions < 1 || indices.Length < 1) throw new ArgumentException("A nonempty batch and positive action count are required.");
            using var scope = NDScope.Open();
            var inputs = np.zeros((indices.Length, _memory[0].State.shape[1]), dtype: np.float64);
            var targets = np.zeros((indices.Length, actions), dtype: np.float64);
            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i] < 0 || indices[i] >= Count) throw new ArgumentOutOfRangeException(nameof(indices));
                var item = _memory[indices[i]];
                if (item.Action >= actions) throw new ArgumentException("Recorded action exceeds predictor width.");
                using var current = predict(item.State);
                using var next = predict(item.NextState);
                if (!current.shape.SequenceEqual(new long[] { 1, actions }) || !next.shape.SequenceEqual(current.shape))
                    throw new ArgumentException("Predict must return (1, actions).");
                inputs[$"{i}:{i + 1},:"] = item.State;
                targets[$"{i},:"] = current["0,:"];
                double nextBest = np.max(next).item<double>();
                targets[i, item.Action] = item.Terminal ? item.Reward : item.Reward + Discount * nextBest;
            }
            return scope.Returns((inputs, targets));
        }

        public void Dispose()
        {
            foreach (var item in _memory) { item.State.Dispose(); item.NextState.Dispose(); }
            _memory.Clear();
        }
    }

    public static void Demo()
    {
        using var game = new CatchGame(5);
        game.Reset(2, 2);
        using var replay = new ReplayMemory();
        int lastReward = 0;
        while (!game.IsOver)
        {
            using var before = game.Observe();
            var step = game.Act(1);
            using var after = step.Observation;
            replay.Remember(before, 1, step.Reward, after, step.GameOver);
            lastReward = step.Reward;
        }
        var batch = replay.GetBatchAt(_ => np.array(new double[,] { { .25, .5, -.25 } }), 3, 0, 3);
        using var inputs = batch.Inputs;
        using var targets = batch.Targets;
        Console.WriteLine($"Catch: reward={lastReward}, transitions={replay.Count}, terminal target={targets.item<double>(1, 1)}");
    }
}
