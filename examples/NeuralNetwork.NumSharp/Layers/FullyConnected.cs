using NeuralNetwork.NumSharp.Activations;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Fully connected layer
    /// </summary>
    public class FullyConnected: BaseLayer
    {
        /// <summary>
        /// Number of incoming input features
        /// </summary>
        public int InputDim { get; set; }

        /// <summary>
        /// Number of neurons for this layers
        /// </summary>
        public int OutNeurons { get; set; }

        /// <summary>
        /// Non Linear Activation function for this layer of neurons. All neurons will have the same function
        /// </summary>
        public BaseActivation Activation { get; set; }

        /// <summary>
        /// Constructor with in and out parametes
        /// </summary>
        /// <param name="in">Number of incoming input features</param>
        /// <param name="out">Number of neurons for this layers</param>
        public FullyConnected(int input_dim, int output_neurons, string act = "") : base("fc")
        {
            Parameters["w"] = np.random.normal(0.5, 1, input_dim, output_neurons);
            InputDim = input_dim;
            OutNeurons = output_neurons;

            Activation = BaseActivation.Get(act);
        }

        /// <summary>
        /// Forward the input data by performing calculation across all the neurons, store it in the Output to be accessible by next layer.
        /// </summary>
        /// <param name="x"></param>
        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = np.dot(x, Parameters["w"]);

            if(Activation!=null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public override void Backward(NDArray grad)
        {
            if(Activation != null)
            {
                Activation.Backward(grad);
                grad = Activation.InputGrad;
            }

            InputGrad = np.dot(grad, Parameters["w"].transpose());
            Grads["w"] = np.dot(Input.transpose(), grad);
        }
    }
}
