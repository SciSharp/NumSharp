using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    public class ReLU : BaseActivation
    {
        public ReLU() : base("relu")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            // max(x, 0) — NOT (x > 0) * x: the multiply form turns relu(-inf)
            // into 0 * -inf = NaN, where Keras/JAX (maximum-based) return 0.
            // NaN inputs still propagate through maximum, matching Keras.
            Output = np.maximum(x, (NDArray)0f);
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * (NDArray)(Input > 0);
        }
    }
}
