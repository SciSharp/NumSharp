using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Binary cross-entropy loss. Expects probabilities (post-sigmoid) as
    /// preds and 0/1 labels, both the same shape. Works for single-label
    /// binary (batch,) and for multi-label (batch, features) tensors —
    /// the loss is mean-over-all-elements, matching Keras convention.
    ///
    /// Forward:
    ///   clipped = clip(preds, eps, 1-eps)
    ///   L = mean(-(labels * log(clipped) + (1 - labels) * log(1 - clipped)))
    ///
    /// Backward:
    ///   dL/dpreds = (clipped - labels) / (clipped * (1 - clipped)) / N
    /// where N = total number of elements in preds (so the /N cancels
    /// against the mean reduction in forward).
    /// </summary>
    public class BinaryCrossEntropy : BaseCost
    {
        public BinaryCrossEntropy() : base("binary_crossentropy") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray clipped = np.clip(preds, (NDArray)Epsilon, (NDArray)(1f - Epsilon));
            NDArray one     = (NDArray)1f;
            return np.mean(-(labels * np.log(clipped) + (one - labels) * np.log(one - clipped)));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray clipped = np.clip(preds, (NDArray)Epsilon, (NDArray)(1f - Epsilon));
            NDArray one     = (NDArray)1f;
            float invSize   = 1f / preds.size;
            return (clipped - labels) / (clipped * (one - clipped)) * invSize;
        }
    }
}
