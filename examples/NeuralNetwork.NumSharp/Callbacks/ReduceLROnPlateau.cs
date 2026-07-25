using System;
using System.Collections.Generic;

namespace NeuralNetwork.NumSharp.Callbacks
{
    /// <summary>
    /// Cuts the learning rate when a monitored metric stops improving — a port of
    /// <c>keras.callbacks.ReduceLROnPlateau</c>.
    ///
    /// <para>The callback writes <see cref="Optimizers.BaseOptimizer.LearningRate"/>,
    /// the optimizer's BASE rate. That is legitimate and is the scheduler's job:
    /// <c>SGD</c>/<c>Adam</c> derive each step's rate as
    /// <c>lr0 / (1 + decay·t)</c> computed fresh from this field, so replacing it
    /// changes the schedule from here on without reintroducing the compounding
    /// bug that the per-step in-place multiply used to cause.</para>
    ///
    /// <para>Keras's loop, reproduced exactly:</para>
    /// <list type="bullet">
    ///   <item>The improvement test carries <c>min_delta</c> INSIDE the
    ///     comparison — <c>current &lt; best - min_delta</c> when minimizing,
    ///     <c>current &gt; best + min_delta</c> when maximizing.</item>
    ///   <item>While in cooldown the wait counter is held at 0 and the cooldown
    ///     counter ticks down; a plateau cannot be detected during cooldown.</item>
    ///   <item>A reduction is skipped entirely once the rate is already at or
    ///     below <see cref="MinLr"/>, and the new rate is floored at it.</item>
    /// </list>
    /// </summary>
    public class ReduceLROnPlateau : BaseCallback
    {
        public string Monitor { get; }

        /// <summary>New rate = old rate × this. Must be in (0, 1).</summary>
        public float Factor { get; }

        /// <summary>Epochs with no improvement before the rate is cut.</summary>
        public int Patience { get; }

        /// <summary>Epochs to wait after a cut before resuming plateau detection.</summary>
        public int Cooldown { get; }

        /// <summary>Lower bound on the learning rate.</summary>
        public float MinLr { get; }

        /// <summary>Threshold for measuring a new optimum.</summary>
        public float MinDelta { get; }

        /// <summary>"auto" (default), "min" or "max".</summary>
        public string Mode { get; }

        /// <summary>0 = silent, 1 = one line per reduction.</summary>
        public int Verbose { get; }

        /// <summary>Best monitored value seen so far.</summary>
        public float Best { get; private set; }

        /// <summary>Epochs since the best value.</summary>
        public int Wait { get; private set; }

        /// <summary>Epochs of cooldown remaining.</summary>
        public int CooldownCounter { get; private set; }

        /// <summary>How many times the rate has been cut this run.</summary>
        public int ReductionCount { get; private set; }

        private readonly bool _maximize;

        public ReduceLROnPlateau(string monitor = "val_loss", float factor = 0.1f, int patience = 10,
                                 int cooldown = 0, float minLr = 0f, float minDelta = 1e-4f,
                                 string mode = "auto", int verbose = 0)
        {
            if (factor <= 0f || factor >= 1f)
                throw new ArgumentOutOfRangeException(nameof(factor), "ReduceLROnPlateau does not support a factor outside (0, 1).");
            if (patience < 0) throw new ArgumentOutOfRangeException(nameof(patience));
            if (cooldown < 0) throw new ArgumentOutOfRangeException(nameof(cooldown));

            Monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
            Factor = factor;
            Patience = patience;
            Cooldown = cooldown;
            MinLr = minLr;
            MinDelta = Math.Abs(minDelta);
            Mode = mode;
            Verbose = verbose;

            _maximize = ResolveMaximize(monitor, mode);
        }

        public override void OnTrainBegin() => Reset();

        private void Reset()
        {
            Best = _maximize ? float.NegativeInfinity : float.PositiveInfinity;
            CooldownCounter = 0;
            Wait = 0;
            ReductionCount = 0;
        }

        private bool InCooldown => CooldownCounter > 0;

        public override void OnEpochEnd(int epoch, IDictionary<string, float> logs)
        {
            // Keras publishes the current rate into the logs so CSVLogger and
            // friends record it; do the same before the early return.
            if (logs != null)
                logs["learning_rate"] = Context.Optimizer.LearningRate;

            if (!TryGetMonitorValue(logs, Monitor, out float current))
                return;

            if (InCooldown)
            {
                CooldownCounter--;
                Wait = 0;
            }

            if (IsImprovement(current))
            {
                Best = current;
                Wait = 0;
                return;
            }

            if (InCooldown)
                return;

            Wait++;
            if (Wait < Patience)
                return;

            float oldLr = Context.Optimizer.LearningRate;
            if (oldLr > MinLr)
            {
                float newLr = Math.Max(oldLr * Factor, MinLr);
                Context.Optimizer.LearningRate = newLr;
                ReductionCount++;

                if (Verbose > 0)
                    Console.WriteLine($"Epoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {newLr}.");

                CooldownCounter = Cooldown;
                Wait = 0;

                if (logs != null)
                    logs["learning_rate"] = newLr;
            }
        }

        /// <summary>
        /// Keras's <c>monitor_op</c>: <c>current &lt; best - min_delta</c> when
        /// minimizing, <c>current &gt; best + min_delta</c> when maximizing. Note
        /// the delta sits on the REFERENCE side here, unlike EarlyStopping's
        /// <c>current - min_delta</c> form — the two callbacks genuinely differ in
        /// upstream, and the results diverge on infinite references.
        /// </summary>
        private bool IsImprovement(float current)
            => _maximize ? current > Best + MinDelta
                         : current < Best - MinDelta;
    }
}
