using System;
using NeuralNetwork.NumSharp.Cost;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Combined softmax + categorical cross-entropy loss.
    ///
    /// The two operations are mathematically separable but numerically hostile
    /// when split: softmax of large logits saturates, and log(softmax) of a
    /// saturated probability underflows to -inf. Fused, the two stages reuse
    /// the same max-subtracted exponentials and cancel cleanly to the stable
    /// backward form  grad = (softmax - labels) / batch  without ever computing
    /// log(softmax) directly on the critical path.
    ///
    /// Expected inputs:
    ///   preds  — raw logits (batch, numClasses) float32 (output of the final
    ///            FullyConnectedFused layer with FusedActivation.None).
    ///   labels — one-hot encoded targets (batch, numClasses) float32.
    ///
    /// Forward returns a scalar NDArray containing the mean per-sample loss;
    /// Backward returns d(loss)/d(logits) with shape (batch, numClasses).
    ///
    /// NOT thread-safe: caches the softmax output between Forward and Backward
    /// calls on a single instance. Matches the existing NeuralNetwork.NumSharp
    /// pattern (BaseLayer and BaseCost both carry mutable state between calls).
    /// </summary>
    public class SoftmaxCrossEntropy : BaseCost
    {
        private NDArray _softmaxCache;

        public SoftmaxCrossEntropy() : base("softmax_crossentropy") { }

        /// <summary>
        /// Computes softmax(logits) row-wise, then cross-entropy against labels.
        /// Returns a scalar NDArray containing the mean-per-sample loss.
        /// Caches the softmax output for reuse in Backward.
        /// </summary>
        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray softmax = ComputeSoftmax(preds);
            _softmaxCache = softmax;

            // Loss = -mean(sum(labels * log(softmax), axis=1))
            // Clip softmax into [eps, 1] before log to avoid -infinity.
            NDArray clipped = np.maximum(softmax, (NDArray)Epsilon);
            NDArray logProbs = np.log(clipped);
            NDArray perSample = np.sum(labels * logProbs, axis: 1);  // (batch,)
            return -np.mean(perSample);
        }

        /// <summary>
        /// Returns d(loss)/d(logits) = (softmax - labels) / batch.
        /// Relies on the softmax cached by the most recent Forward call.
        /// </summary>
        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            if (_softmaxCache is null)
                throw new InvalidOperationException(
                    "SoftmaxCrossEntropy.Backward called before Forward; softmax cache is empty.");

            int batch = (int)preds.shape[0];
            NDArray grad = (_softmaxCache - labels) * (1f / batch);
            return grad;
        }

        // =================================================================
        // Helpers
        // =================================================================

        /// <summary>
        /// Row-wise numerically stable softmax: subtract per-row max, exponentiate,
        /// divide by per-row sum. Produces float32 output matching the input dtype.
        /// </summary>
        private static NDArray ComputeSoftmax(NDArray logits)
        {
            // max(logits, axis=1, keepdims=true) → shape (batch, 1). Subtracting
            // broadcasts across the class dim.
            NDArray rowMax = logits.max(axis: 1, keepdims: true);
            NDArray shifted = logits - rowMax;
            NDArray exps = np.exp(shifted);
            NDArray rowSum = np.sum(exps, axis: 1, keepdims: true);
            return exps / rowSum;
        }

        /// <summary>
        /// Builds a (N, numClasses) one-hot float32 matrix from a (N,) integer
        /// label vector. Supports Byte, Int32, Int64 label dtypes — the three
        /// that MnistLoader and np.argmax produce in this project.
        /// </summary>
        public static NDArray OneHot(NDArray labels, int numClasses)
        {
            int n = (int)labels.shape[0];
            NDArray one_hot = np.zeros(new Shape(n, numClasses), NPTypeCode.Single);
            NPTypeCode lt = labels.typecode;
            unsafe
            {
                float* dst = (float*)one_hot.Address;
                for (int i = 0; i < n; i++)
                {
                    int label = lt switch
                    {
                        NPTypeCode.Byte  => labels.GetByte(i),
                        NPTypeCode.Int32 => labels.GetInt32(i),
                        NPTypeCode.Int64 => (int)labels.GetInt64(i),
                        _ => throw new NotSupportedException(
                            $"OneHot doesn't support label dtype {lt}."),
                    };
                    if ((uint)label >= (uint)numClasses)
                        throw new ArgumentOutOfRangeException(nameof(labels),
                            $"label at index {i} = {label} is outside [0,{numClasses}).");
                    dst[i * numClasses + label] = 1f;
                }
            }
            return one_hot;
        }
    }
}
