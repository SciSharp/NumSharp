using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Activations
{
    public class Sigmoid : BaseActivation
    {
        public Sigmoid() : base("sigmoid")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            //ToDo: np.exp
            //Output = 1 / (1 + Exp(-x));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * Output * (1 - Output);
        }
    }
}
