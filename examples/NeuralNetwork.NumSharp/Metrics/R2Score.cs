using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    /// <summary>
    /// Coefficient of determination (sklearn r2_score semantics):
    ///
    ///   R^2 = 1 - SS_res / SS_tot
    ///   SS_res = sum((labels - preds)^2)
    ///   SS_tot = sum((labels - mean(labels))^2)
    ///
    /// Degenerate constant-labels case follows sklearn: R^2 = 1 when the
    /// predictions are also perfect (SS_res == 0), else 0.
    /// A score of 1 is a perfect fit, 0 matches predicting the label mean,
    /// negative is worse than the mean predictor.
    /// </summary>
    public class R2Score : BaseMetric
    {
        public R2Score() : base("r2_score") { }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            NDArray residual = labels - preds;
            float ssRes = (float)np.sum(residual * residual);

            NDArray centered = labels - np.mean(labels);
            float ssTot = (float)np.sum(centered * centered);

            if (ssTot == 0f)
                return NDArray.Scalar(ssRes == 0f ? 1f : 0f);
            return NDArray.Scalar(1f - ssRes / ssTot);
        }
    }
}
