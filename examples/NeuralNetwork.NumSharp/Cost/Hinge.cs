using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Hinge loss (Keras semantics) for maximum-margin classification:
    ///
    ///   L = mean( max(1 - y_true * y_pred, 0) )
    ///
    /// y_true is expected in {-1, +1}. Like Keras, binary {0, 1} labels are
    /// detected (every element is exactly 0 or 1) and converted to
    /// {-1, +1} via 2*y - 1 before the margin.
    ///
    /// Backward, with margin_ij = 1 - yt_ij * yp_ij and N = preds.size:
    ///   dL/dyp_ij = -yt_ij / N   where margin_ij &gt; 0
    ///                0           otherwise
    /// </summary>
    public class Hinge : BaseCost
    {
        public Hinge() : base("hinge") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray yt = MaybeConvertLabels(labels);
            NDArray margin = (NDArray)1f - yt * preds;
            return np.mean(np.maximum(margin, (NDArray)0f));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray yt = MaybeConvertLabels(labels);
            NDArray margin = (NDArray)1f - yt * preds;
            float invSize = 1f / preds.size;
            return np.where(margin > 0, -yt * invSize, (NDArray)0f);
        }

        /// <summary>
        /// Keras's _maybe_convert_labels: if EVERY label is exactly 0 or 1,
        /// map to {-1, +1}; otherwise pass through unchanged.
        /// </summary>
        private static NDArray MaybeConvertLabels(NDArray labels)
        {
            NDArray isZero = labels == (NDArray)0f;
            NDArray isOne = labels == (NDArray)1f;
            bool allBinary = np.all(isZero | isOne);
            return allBinary ? labels * 2f - 1f : labels;
        }
    }
}
