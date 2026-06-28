using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Binary accuracy metric. Expects sigmoid probabilities as preds and
    /// 0/1 labels, both the same shape. Rounds each prediction to 0 or 1
    /// (threshold 0.5) and returns the fraction of elements matching the
    /// labels.
    ///
    /// Class name retains the original misspelling ("BinaryAccuacy") for
    /// backward compatibility.
    /// </summary>
    public class BinaryAccuacy : BaseMetric
    {
        public BinaryAccuacy() : base("binary_accuracy") { }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            // Clip first to guarantee preds are in [0, 1], then round — preds
            // fed directly from a sigmoid will already be in range, but a raw
            // logit or a probability that slipped slightly out of bounds would
            // otherwise round incorrectly.
            NDArray rounded = np.round_(np.clip(preds, (NDArray)0f, (NDArray)1f));
            NDArray matches = (rounded == labels).astype(NPTypeCode.Single);
            return np.mean(matches);
        }
    }
}
