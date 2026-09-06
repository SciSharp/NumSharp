using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Cost
{
    /// <summary>
    /// Log-cosh loss (Keras semantics): L = mean( log(cosh(preds - labels)) ).
    /// Behaves like L2 near zero and like L1 (minus log 2) in the tails, and is
    /// twice differentiable everywhere — a smooth Huber.
    ///
    /// Forward uses the overflow-safe identity
    ///   log(cosh(e)) = |e| + log1p(exp(-2|e|)) - log(2)
    /// so cosh never overflows for large |e| (cosh(90f) already exceeds
    /// float32 range).
    ///
    /// Backward: d log(cosh(e))/de = tanh(e), scaled by 1/N (N = preds.size).
    /// </summary>
    public class LogCosh : BaseCost
    {
        private const float Ln2 = 0.6931471805599453f;

        public LogCosh() : base("log_cosh") { }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            NDArray e = preds - labels;
            NDArray absE = np.abs(e);
            NDArray perElement = absE + np.log1p(np.exp(absE * -2f)) - Ln2;
            return np.mean(perElement);
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            NDArray e = preds - labels;
            float invSize = 1f / preds.size;
            return np.tanh(e) * invSize;
        }
    }
}
