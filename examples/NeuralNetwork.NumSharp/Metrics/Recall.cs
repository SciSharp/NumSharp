using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Binary recall (Keras semantics): TP / (TP + FN), the fraction of actual
    /// positives that were predicted positive. Expects sigmoid probabilities
    /// as preds and 0/1 labels of the same shape; predictions are binarized at
    /// <see cref="Threshold"/> (p &gt; threshold → positive). Returns 0 when
    /// there are no actual positives (Keras convention).
    /// </summary>
    public class Recall : BaseMetric
    {
        public float Threshold { get; }

        public Recall(float threshold = 0.5f) : base("recall")
        {
            Threshold = threshold;
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            NDArray predPos = (preds > Threshold).astype(NPTypeCode.Single);
            NDArray tp = np.sum(predPos * labels);
            NDArray fn = np.sum(((NDArray)1f - predPos) * labels);
            float tpv = (float)tp, fnv = (float)fn;
            float denominator = tpv + fnv;
            return NDArray.Scalar(denominator > 0f ? tpv / denominator : 0f);
        }
    }
}
