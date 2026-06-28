using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Multi-class accuracy metric. Expects probabilities / logits as preds
    /// of shape (batch, numClasses) and one-hot labels of the same shape.
    /// Computes argmax-per-row on both, counts matches, returns a scalar
    /// NDArray of the fraction correct in [0, 1].
    ///
    /// Class name retains the original misspelling ("Accuacy") for backward
    /// compatibility with any existing callers.
    /// </summary>
    public class Accuacy : BaseMetric
    {
        public Accuacy() : base("accuracy") { }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            NDArray predIdx  = np.argmax(preds,  axis: 1);
            NDArray labelIdx = np.argmax(labels, axis: 1);
            NDArray matches  = (predIdx == labelIdx).astype(NPTypeCode.Single);
            return np.mean(matches);
        }
    }
}
