using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Top-K categorical accuracy (Keras semantics): the fraction of samples
    /// whose true class ranks within the K highest-scoring predictions.
    /// Expects (batch, numClasses) probabilities / logits and one-hot labels.
    ///
    /// Tie handling follows tf.math.in_top_k: the true class is "in the top K"
    /// when FEWER than K classes score strictly higher — classes tied with the
    /// true class's score never push it out.
    /// </summary>
    public class TopKCategoricalAccuracy : BaseMetric
    {
        public int K { get; }

        public TopKCategoricalAccuracy(int k = 5) : base("top_k_categorical_accuracy")
        {
            if (k < 1) throw new ArgumentOutOfRangeException(nameof(k));
            K = k;
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            int batch = (int)preds.shape[0];
            int classes = (int)preds.shape[1];
            NDArray labelIdx = np.argmax(labels, axis: 1);

            int correct = 0;
            for (int i = 0; i < batch; i++)
            {
                int li = (int)labelIdx.GetInt64(i);
                float trueScore = preds.GetSingle(i, li);

                int strictlyHigher = 0;
                for (int j = 0; j < classes && strictlyHigher < K; j++)
                    if (preds.GetSingle(i, j) > trueScore)
                        strictlyHigher++;

                if (strictlyHigher < K)
                    correct++;
            }

            return NDArray.Scalar((float)correct / batch);
        }
    }
}
