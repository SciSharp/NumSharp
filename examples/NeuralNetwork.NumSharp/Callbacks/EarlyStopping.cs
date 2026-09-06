using System;
using System.Collections.Generic;
using NeuralNetwork.NumSharp.Serialization;
using NumSharp;

namespace NeuralNetwork.NumSharp.Callbacks
{
    /// <summary>
    /// Stops training once a monitored metric has stopped improving — a port of
    /// <c>keras.callbacks.EarlyStopping</c>.
    ///
    /// <para>Three details of Keras's loop are easy to get wrong and are
    /// reproduced deliberately:</para>
    /// <list type="number">
    ///   <item><b><c>MinDelta</c> is signed by the direction.</b> Keras stores
    ///     <c>abs(min_delta)</c> and then NEGATES it in "min" mode, so the single
    ///     comparison <c>current - min_delta ⋛ best</c> means "improved by at
    ///     least min_delta" for both directions. Applying an unsigned min_delta
    ///     to a minimized metric inverts the test and early-stops immediately.</item>
    ///   <item><b><c>wait</c> increments BEFORE the improvement check</b>, and is
    ///     reset to 0 only on an improvement — so <c>patience</c> counts epochs
    ///     since the best, inclusive.</item>
    ///   <item><b>The stop test carries an <c>epoch &gt; 0</c> guard</b>, so
    ///     <c>patience: 0</c> can never stop on the very first epoch.</item>
    /// </list>
    ///
    /// <para><see cref="RestoreBestWeights"/> uses
    /// <see cref="ModelWeights.Capture"/>/<see cref="ModelWeights.Restore"/> —
    /// full deep copies, so an optimizer step after the snapshot cannot disturb
    /// it. Like Keras, weights are restored only when the callback actually fires
    /// the stop; a run that finishes its epochs normally keeps its final
    /// weights.</para>
    /// </summary>
    public class EarlyStopping : BaseCallback
    {
        public string Monitor { get; }

        /// <summary>Minimum change that counts as an improvement (always given unsigned).</summary>
        public float MinDelta { get; }

        /// <summary>Epochs with no improvement after which training stops.</summary>
        public int Patience { get; }

        /// <summary>"auto" (default), "min" or "max".</summary>
        public string Mode { get; }

        /// <summary>
        /// A value the metric must also beat before <c>wait</c> resets. Null
        /// disables. Matches Keras: an improvement over the previous best that
        /// still misses the baseline updates <c>best</c> but does NOT reset the
        /// patience counter.
        /// </summary>
        public float? Baseline { get; }

        /// <summary>Roll the model back to the best epoch's weights when stopping.</summary>
        public bool RestoreBestWeights { get; }

        /// <summary>Epochs to run before the monitor is consulted at all.</summary>
        public int StartFromEpoch { get; }

        /// <summary>0 = silent, 1 = announce the stop and any restore.</summary>
        public int Verbose { get; }

        /// <summary>0-based epoch at which training stopped, or -1 if it ran to completion.</summary>
        public int StoppedEpoch { get; private set; } = -1;

        /// <summary>0-based epoch that produced <see cref="Best"/>.</summary>
        public int BestEpoch { get; private set; }

        /// <summary>Best monitored value seen so far.</summary>
        public float Best { get; private set; }

        /// <summary>Epochs elapsed since the best value (Keras's <c>wait</c>).</summary>
        public int Wait { get; private set; }

        private readonly bool _maximize;
        private readonly float _signedMinDelta;
        private Dictionary<string, NDArray> _bestWeights;

        public EarlyStopping(string monitor = "val_loss", float minDelta = 0f, int patience = 0,
                             string mode = "auto", float? baseline = null, bool restoreBestWeights = false,
                             int startFromEpoch = 0, int verbose = 0)
        {
            if (patience < 0) throw new ArgumentOutOfRangeException(nameof(patience));
            if (startFromEpoch < 0) throw new ArgumentOutOfRangeException(nameof(startFromEpoch));

            Monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
            MinDelta = Math.Abs(minDelta);
            Patience = patience;
            Mode = mode;
            Baseline = baseline;
            RestoreBestWeights = restoreBestWeights;
            StartFromEpoch = startFromEpoch;
            Verbose = verbose;

            _maximize = ResolveMaximize(monitor, mode);
            // Keras: min_delta *= 1 when maximizing, *= -1 when minimizing.
            _signedMinDelta = _maximize ? MinDelta : -MinDelta;
        }

        public override void OnTrainBegin()
        {
            Wait = 0;
            StoppedEpoch = -1;
            Best = _maximize ? float.NegativeInfinity : float.PositiveInfinity;
            BestEpoch = 0;
            _bestWeights = null;
        }

        public override void OnEpochEnd(int epoch, IDictionary<string, float> logs)
        {
            if (!TryGetMonitorValue(logs, Monitor, out float current) || epoch < StartFromEpoch)
                return;

            // Keras seeds best_weights on the first observed epoch so a stop can
            // always restore SOMETHING, even if no epoch ever improves.
            if (RestoreBestWeights && _bestWeights == null)
                _bestWeights = ModelWeights.Capture(Context.Layers);

            Wait++;

            if (IsImprovement(current, Best))
            {
                Best = current;
                BestEpoch = epoch;
                if (RestoreBestWeights)
                    _bestWeights = ModelWeights.Capture(Context.Layers);

                // Only restart the patience clock if we also cleared the baseline.
                if (!Baseline.HasValue || IsImprovement(current, Baseline.Value))
                    Wait = 0;
                return;
            }

            if (Wait >= Patience && epoch > 0)
            {
                StoppedEpoch = epoch;
                Context.StopTraining = true;

                if (RestoreBestWeights && _bestWeights != null)
                {
                    if (Verbose > 0)
                        Console.WriteLine($"Restoring model weights from the end of the best epoch: {BestEpoch + 1}.");
                    ModelWeights.Restore(Context.Layers, _bestWeights);
                }
            }
        }

        public override void OnTrainEnd()
        {
            if (StoppedEpoch >= 0 && Verbose > 0)
                Console.WriteLine($"Epoch {StoppedEpoch + 1}: early stopping");
        }

        /// <summary>
        /// Keras's <c>_is_improvement</c>: <c>monitor_op(value - min_delta, reference)</c>
        /// where the op is &lt; when minimizing and &gt; when maximizing, and
        /// <c>min_delta</c> already carries the direction's sign.
        /// </summary>
        private bool IsImprovement(float value, float reference)
            => _maximize ? value - _signedMinDelta > reference
                         : value - _signedMinDelta < reference;
    }
}
