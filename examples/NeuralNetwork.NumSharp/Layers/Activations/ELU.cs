using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Exponential linear unit (Clevert et al., 2016):
    ///   y = x                     for x &gt; 0
    ///   y = alpha * (exp(x) - 1)  for x &lt;= 0
    ///
    /// alpha = 1.0 is the Keras default. Backward reuses the cached output for
    /// the negative branch: d/dx alpha*(exp(x)-1) = alpha*exp(x) = y + alpha,
    /// so
    ///   dL/dx = dL/dy * (1 for x &gt; 0, y + alpha otherwise).
    ///
    /// Note np.where evaluates both branches eagerly — exp(x) of large positive
    /// x overflows to +inf on the branch that is then discarded, which is
    /// harmless (the selected branch is x itself).
    /// </summary>
    public class ELU : BaseActivation
    {
        public float Alpha { get; }

        public ELU(float alpha = 1.0f) : base("elu")
        {
            Alpha = alpha;
        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.where(x > 0, x, (np.exp(x) - 1f) * Alpha);
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * np.where(Input > 0, (NDArray)1f, Output + Alpha);
        }
    }
}
