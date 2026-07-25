using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Binary precision (Keras semantics): TP / (TP + FP), the fraction of
    /// positive predictions that are actually positive. Expects sigmoid
    /// probabilities as preds and 0/1 labels of the same shape; predictions
    /// are binarized at <see cref="Threshold"/> (p &gt; threshold → positive,
    /// matching Keras's strict inequality). Returns 0 when nothing was
    /// predicted positive (Keras convention, avoids 0/0).
    /// </summary>
    public class Precision : BaseMetric
    {
        public float Threshold { get; }

        public Precision(float threshold = 0.5f) : base("precision")
        {
            Threshold = threshold;
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            NDArray predPos = (preds > Threshold).astype(NPTypeCode.Single);
            NDArray tp = np.sum(predPos * labels);
            NDArray fp = np.sum(predPos * ((NDArray)1f - labels));
            float tpv = (float)tp, fpv = (float)fp;
            float denominator = tpv + fpv;
            return NDArray.Scalar(denominator > 0f ? tpv / denominator : 0f);
        }
    }
}
