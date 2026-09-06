using System;
using System.Collections.Generic;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.Optimizers;

namespace NeuralNetwork.NumSharp.Callbacks
{
    /// <summary>
    /// What a callback is allowed to see and touch during a run — the stand-in
    /// for Keras's <c>self.model</c>. The trainer hands one of these to every
    /// callback before <see cref="BaseCallback.OnTrainBegin"/>.
    ///
    /// <para><see cref="StopTraining"/> is the Keras <c>model.stop_training</c>
    /// flag: any callback may set it from <c>OnEpochEnd</c> and the trainer
    /// breaks out of the epoch loop after the current epoch finishes.</para>
    /// </summary>
    public sealed class TrainingContext
    {
        public TrainingContext(IReadOnlyList<BaseLayer> layers, BaseOptimizer optimizer,
                               int epochs, int batchSize, int stepsPerEpoch, bool hasValidation)
        {
            Layers = layers;
            Optimizer = optimizer;
            Epochs = epochs;
            BatchSize = batchSize;
            StepsPerEpoch = stepsPerEpoch;
            HasValidation = hasValidation;
        }

        /// <summary>The model being trained, in forward order.</summary>
        public IReadOnlyList<BaseLayer> Layers { get; }

        /// <summary>The optimizer — writable, so ReduceLROnPlateau can move the LR.</summary>
        public BaseOptimizer Optimizer { get; }

        /// <summary>Total epochs requested (not the number that will actually run).</summary>
        public int Epochs { get; }

        public int BatchSize { get; }

        /// <summary>Batches per epoch, including a partial final batch.</summary>
        public int StepsPerEpoch { get; }

        /// <summary>Whether <c>val_loss</c> / <c>val_acc</c> will appear in the logs.</summary>
        public bool HasValidation { get; }

        /// <summary>
        /// Set by a callback to end training after the current epoch
        /// (Keras <c>model.stop_training</c>).
        /// </summary>
        public bool StopTraining { get; set; }
    }

    /// <summary>
    /// Keras <c>keras.callbacks.Callback</c> analog. Subclasses override only the
    /// hooks they need; every hook has a no-op default.
    ///
    /// <para><b>Log keys</b> follow Keras convention and are what the monitoring
    /// callbacks match by name:</para>
    /// <list type="bullet">
    ///   <item><c>loss</c>, <c>acc</c> — training metrics, averaged over the epoch</item>
    ///   <item><c>val_loss</c>, <c>val_acc</c> — validation metrics (absent when
    ///         no validation set is configured)</item>
    ///   <item><c>learning_rate</c> — the optimizer's base rate at epoch end</item>
    /// </list>
    ///
    /// <para><b>Epoch and batch indices are 0-based</b>, as in Keras; the trainer's
    /// console output prints them 1-based. A callback that formats an epoch for
    /// humans should add 1 (all of the built-ins below do).</para>
    /// </summary>
    public abstract class BaseCallback
    {
        /// <summary>
        /// The run this callback is attached to. Null until the trainer calls
        /// <see cref="SetContext"/>.
        /// </summary>
        public TrainingContext Context { get; protected set; }

        /// <summary>
        /// Attaches the callback to a run (Keras <c>set_model</c>). Called by the
        /// trainer before <see cref="OnTrainBegin"/>; override to capture extra
        /// state, but call base.
        /// </summary>
        public virtual void SetContext(TrainingContext context) => Context = context;

        /// <summary>Once, before the first epoch.</summary>
        public virtual void OnTrainBegin() { }

        /// <summary>
        /// Once, after the last epoch — including when training was cut short by
        /// <see cref="TrainingContext.StopTraining"/>. Callbacks holding OS
        /// resources release them here.
        /// </summary>
        public virtual void OnTrainEnd() { }

        /// <summary>Start of each epoch, before any batch runs.</summary>
        /// <param name="epoch">0-based epoch index.</param>
        public virtual void OnEpochBegin(int epoch) { }

        /// <summary>
        /// End of each epoch, after validation has been scored — so
        /// <paramref name="logs"/> already carries <c>val_loss</c>/<c>val_acc</c>
        /// when a validation set is configured.
        /// </summary>
        /// <param name="epoch">0-based epoch index.</param>
        /// <param name="logs">Metric name → value for this epoch.</param>
        public virtual void OnEpochEnd(int epoch, IDictionary<string, float> logs) { }

        /// <summary>
        /// End of each batch. <paramref name="logs"/> carries this batch's
        /// <c>loss</c> and <c>acc</c> only — validation is an epoch-level concept.
        /// </summary>
        /// <param name="batch">0-based batch index within the epoch.</param>
        public virtual void OnBatchEnd(int batch, IDictionary<string, float> logs) { }

        // =================================================================
        // Shared helpers for the monitoring callbacks
        // =================================================================

        /// <summary>
        /// Reads <paramref name="monitor"/> out of the logs, returning false when
        /// it is absent. Keras warns and skips in that case rather than throwing —
        /// a callback watching <c>val_loss</c> on a run with no validation data
        /// must not take the run down with it.
        /// </summary>
        protected static bool TryGetMonitorValue(IDictionary<string, float> logs, string monitor, out float value)
        {
            value = 0f;
            return logs != null && logs.TryGetValue(monitor, out value);
        }

        /// <summary>
        /// Keras's <c>mode="auto"</c> resolution: a metric whose name ends in
        /// <c>acc</c> / <c>accuracy</c> / <c>auc</c> is maximized, everything else
        /// is minimized. Returns true when larger is better.
        /// </summary>
        protected static bool ResolveMaximize(string monitor, string mode)
        {
            switch (mode?.Trim().ToLowerInvariant())
            {
                case "min": return false;
                case "max": return true;
                case null:
                case "":
                case "auto":
                    break;
                default:
                    throw new ArgumentException($"mode must be 'auto', 'min' or 'max' — got '{mode}'.", nameof(mode));
            }

            string m = monitor ?? "";
            return m.EndsWith("acc", StringComparison.OrdinalIgnoreCase)
                || m.EndsWith("accuracy", StringComparison.OrdinalIgnoreCase)
                || m.EndsWith("auc", StringComparison.OrdinalIgnoreCase);
        }
    }
}
