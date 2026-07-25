using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>How multi-class F1 folds per-class scores into one number.</summary>
    public enum F1Average
    {
        /// <summary>Binary F1 on thresholded sigmoid outputs (sklearn average='binary').</summary>
        Binary,

        /// <summary>
        /// Unweighted mean of per-class F1 over one-hot / argmax multi-class
        /// outputs (sklearn average='macro'). Classes absent from both preds
        /// and labels contribute an F1 of 0, like sklearn.
        /// </summary>
        Macro,
    }

    /// <summary>
    /// F1 score: the harmonic mean of precision and recall,
    ///   F1 = 2 * P * R / (P + R)   (0 when P + R == 0).
    ///
    /// Binary mode expects sigmoid probabilities + 0/1 labels (same shape) and
    /// binarizes at <see cref="Threshold"/>. Macro mode expects
    /// (batch, numClasses) probabilities + one-hot labels, argmaxes both, and
    /// averages per-class F1 across all numClasses classes.
    /// </summary>
    public class F1Score : BaseMetric
    {
        public F1Average Average { get; }
        public float Threshold { get; }

        public F1Score(F1Average average = F1Average.Binary, float threshold = 0.5f) : base("f1_score")
        {
            Average = average;
            Threshold = threshold;
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            return Average == F1Average.Binary
                ? BinaryF1(preds, labels)
                : MacroF1(preds, labels);
        }

        private NDArray BinaryF1(NDArray preds, NDArray labels)
        {
            NDArray predPos = (preds > Threshold).astype(NPTypeCode.Single);
            float tp = (float)np.sum(predPos * labels);
            float fp = (float)np.sum(predPos * ((NDArray)1f - labels));
            float fn = (float)np.sum(((NDArray)1f - predPos) * labels);
            return NDArray.Scalar(F1(tp, fp, fn));
        }

        private static NDArray MacroF1(NDArray preds, NDArray labels)
        {
            int batch = (int)preds.shape[0];
            int classes = (int)preds.shape[1];
            NDArray predIdx = np.argmax(preds, axis: 1);
            NDArray labelIdx = np.argmax(labels, axis: 1);

            var tp = new int[classes];
            var fp = new int[classes];
            var fn = new int[classes];
            for (int i = 0; i < batch; i++)
            {
                int p = (int)predIdx.GetInt64(i);
                int l = (int)labelIdx.GetInt64(i);
                if (p == l)
                    tp[p]++;
                else
                {
                    fp[p]++;
                    fn[l]++;
                }
            }

            float sum = 0f;
            for (int c = 0; c < classes; c++)
                sum += F1(tp[c], fp[c], fn[c]);
            return NDArray.Scalar(sum / classes);
        }

        private static float F1(float tp, float fp, float fn)
        {
            float denominator = 2f * tp + fp + fn;
            return denominator > 0f ? 2f * tp / denominator : 0f;
        }
    }
}
