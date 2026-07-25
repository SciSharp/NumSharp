using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Kullback-Leibler divergence loss (Keras semantics). Both operands are
    /// probability distributions over the last axis, shape (batch, numClasses):
    ///
    ///   L = mean_i( sum_j( yt_ij * log(yt_ij / yp_ij) ) )
    ///
    /// with both yt and yp clipped into [eps, 1] before the ratio, exactly as
    /// Keras does — so zero entries in y_true contribute eps*log(eps/yp)
    /// (vanishingly small) instead of NaN.
    ///
    /// Backward (treating y_true as constant):
    ///   dL/dyp_ij = -yt_ij / clip(yp_ij) / batch
    /// (zero-gradient inside the clipped range boundaries is ignored, the
    /// standard simplification).
    /// </summary>
    public class KLDivergence : BaseCost
    {
        public KLDivergence() : base("kl_divergence") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray yt = np.clip(labels, (NDArray)Epsilon, (NDArray)1f);
            NDArray yp = np.clip(preds, (NDArray)Epsilon, (NDArray)1f);
            NDArray perSample = np.sum(yt * np.log(yt / yp), axis: 1);   // (batch,)
            return np.mean(perSample);
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray yt = np.clip(labels, (NDArray)Epsilon, (NDArray)1f);
            NDArray yp = np.clip(preds, (NDArray)Epsilon, (NDArray)1f);
            int batch = (int)preds.shape[0];
            return -yt / yp / (float)batch;
        }
    }
}
