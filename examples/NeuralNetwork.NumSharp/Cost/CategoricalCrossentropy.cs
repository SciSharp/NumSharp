using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Categorical cross-entropy loss for multi-class classification.
    /// Expects probabilities (post-softmax) as preds and a one-hot encoded
    /// labels matrix, both shape (batch, numClasses).
    ///
    /// Forward:  L = -sum(labels * log(clip(preds, eps, 1-eps))) / batch
    /// Backward: dL/dpreds = -labels / clip(preds, eps, 1-eps) / batch
    ///
    /// Clipping protects against log(0) when softmax saturates. If you're
    /// chaining Softmax + CategoricalCrossentropy in training, prefer the
    /// combined <see cref="NeuralNetwork.NumSharp.MnistMlp.SoftmaxCrossEntropy"/>
    /// — it differentiates through both at once and yields the cleaner,
    /// numerically better backward  (softmax - labels) / batch.
    /// </summary>
    public class CategoricalCrossentropy : BaseCost
    {
        public CategoricalCrossentropy() : base("categorical_crossentropy") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray clipped = np.clip(preds, (NDArray)Epsilon, (NDArray)(1f - Epsilon));
            int batch = (int)preds.shape[0];
            return -np.sum(labels * np.log(clipped)) / (float)batch;
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray clipped = np.clip(preds, (NDArray)Epsilon, (NDArray)(1f - Epsilon));
            int batch = (int)preds.shape[0];
            return -labels / clipped / (float)batch;
        }
    }
}
