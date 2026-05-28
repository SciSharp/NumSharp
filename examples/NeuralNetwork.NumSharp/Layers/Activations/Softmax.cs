using System;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    /// <summary>
    /// Row-wise softmax activation. Forward is numerically stable
    /// (subtracts the per-row max before exponentiating so large logits
    /// don't overflow). Backward applies the correct Jacobian-vector
    /// product for softmax:
    ///
    ///   dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
    ///
    /// When softmax is followed by categorical cross-entropy, the
    /// combined backward simplifies to (s - labels) / batch — prefer
    /// <see cref="NeuralNetwork.NumSharp.MnistMlp.SoftmaxCrossEntropy"/>
    /// there for better numerical behavior and fewer ops. This class is
    /// the right choice when softmax probabilities are consumed by
    /// something other than CE (e.g., a custom loss, a secondary head).
    /// </summary>
    public class Softmax : BaseActivation
    {
        public Softmax() : base("softmax") { }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            // Numerically stable row-wise softmax: subtract per-row max,
            // exponentiate, divide by per-row sum.
            NDArray rowMax = x.max(axis: 1, keepdims: true);
            NDArray shifted = x - rowMax;
            NDArray exps = np.exp(shifted);
            NDArray rowSum = np.sum(exps, axis: 1, keepdims: true);
            Output = exps / rowSum;
        }

        public override void Backward(NDArray grad)
        {
            // Jacobian-vector product for softmax:
            //   dL/dx = softmax * (grad - sum(grad * softmax, axis=1, keepdims))
            //
            // Row-wise: each row's gradient is the row's softmax output
            // times (row's grad minus the dot product of row's grad with
            // row's softmax). This is what falls out of the full Jacobian
            // ds_i/dx_j = s_i (δ_ij − s_j) when you multiply by grad.
            NDArray dotPerRow = np.sum(grad * Output, axis: 1, keepdims: true);  // (batch, 1)
            InputGrad = Output * (grad - dotPerRow);
        }
    }
}
