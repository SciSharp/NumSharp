using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NeuralNetwork.NumSharp.Layers
{
    /// <summary>
    /// Base class for the layers with predefined variables and functions
    /// </summary>
    public abstract class BaseLayer
    {
        /// <summary>
        /// Name of the layer
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Input for the layer
        /// </summary>
        public NDArray Input { get; set; }

        /// <summary>
        /// Output after forwarding the input across the neurons
        /// </summary>
        public NDArray Output { get; set; }

        /// <summary>
        /// Trainable parameters list, eg, weight, bias
        /// </summary>
        public Dictionary<string, NDArray> Parameters { get; set; }

        /// <summary>
        /// Gradient of the Input
        /// </summary>
        public NDArray InputGrad { get; set; }

        /// <summary>
        /// List of all parameters gradients calculated during back propagation.
        /// </summary>
        public Dictionary<string, NDArray> Grads { get; set; }

        /// <summary>
        /// Base layer instance
        /// </summary>
        /// <param name="name"></param>
        public BaseLayer(string name)
        {
            Name = name + Util.GetNext();
            Parameters = new Dictionary<string, NDArray>();
            Grads = new Dictionary<string, NDArray>();
        }

        /// <summary>
        /// Virtual forward method to perform calculation and move the input to next layer
        /// </summary>
        /// <param name="x"></param>
        public virtual void Forward(NDArray x)
        {
            Input = x;
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public virtual void Backward(NDArray grad)
        {
            
        }

        public void PrintParams(bool printGrads = true)
        {
            foreach (var item in Parameters)
            {
                Console.WriteLine(item.Value.ToString());
                if(printGrads && Grads.ContainsKey(item.Key))
                {
                    Console.WriteLine(Grads[item.Key].ToString());
                }
            }
        }
    }
}
