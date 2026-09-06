using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Huber loss (Keras semantics) — quadratic near zero, linear in the
    /// tails; the standard robust regression loss.
    ///
    /// Per element, with e = preds - labels:
    ///   |e| &lt;= delta :  0.5 * e^2
    ///   |e| &gt;  delta :  delta * (|e| - 0.5 * delta)
    /// Loss is the mean over ALL elements (Keras mean-over-last-axis then
    /// mean-over-batch collapses to that for dense tensors).
    ///
    /// Backward:
    ///   dL/de = e               for |e| &lt;= delta
    ///           delta * sign(e) for |e| &gt;  delta
    /// scaled by 1/N with N = preds.size, cancelling the forward mean.
    /// </summary>
    public class Huber : BaseCost
    {
        public float Delta { get; }

        public Huber(float delta = 1.0f) : base("huber")
        {
            Delta = delta;
        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray e = preds - labels;
            NDArray absE = np.abs(e);
            NDArray quadratic = e * e * 0.5f;
            NDArray linear = (absE - 0.5f * Delta) * Delta;
            return np.mean(np.where(absE <= Delta, quadratic, linear));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray e = preds - labels;
            NDArray absE = np.abs(e);
            float invSize = 1f / preds.size;
            return np.where(absE <= Delta, e, np.sign(e) * Delta) * invSize;
        }
    }
}
