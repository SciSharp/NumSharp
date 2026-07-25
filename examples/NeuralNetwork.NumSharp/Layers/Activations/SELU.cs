using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Scaled exponential linear unit (Klambauer et al., 2017) — the
    /// self-normalizing activation:
    ///
    ///   y = lambda * x                      for x &gt; 0
    ///   y = lambda * alpha * (exp(x) - 1)   for x &lt;= 0
    ///
    /// with the fixed constants lambda ≈ 1.0507, alpha ≈ 1.6733 chosen so
    /// activations converge to zero mean / unit variance (use with
    /// LecunNormal init).
    ///
    /// Backward reuses the cached output for the negative branch:
    ///   d/dx lambda*alpha*(exp(x)-1) = lambda*alpha*exp(x) = y + lambda*alpha
    ///   dL/dx = dL/dy * (lambda for x &gt; 0, y + lambda*alpha otherwise).
    /// </summary>
    public class SELU : BaseActivation
    {
        public const float LambdaConst = 1.0507009873554805f;
        public const float AlphaConst  = 1.6732632423543772f;

        public SELU() : base("selu") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.where(x > 0, x * LambdaConst, (np.exp(x) - 1f) * (LambdaConst * AlphaConst));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * np.where(Input > 0, (NDArray)LambdaConst, Output + LambdaConst * AlphaConst);
        }
    }
}
