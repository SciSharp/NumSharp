using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.NumSharp.Layers;

namespace NeuralNetwork.NumSharp.Activations
{
    public class BaseActivation : BaseLayer
    {
        public BaseActivation(string name) : base(name)
        {

        }

        public static BaseActivation Get(string name)
        {
            BaseActivation baseActivation = null;
            switch (name)
            {
                case "relu":
                    baseActivation = new ReLU();
                    break;
                case "sigmoid":
                    baseActivation = new Sigmoid();
                    break;
                default:
                    break;
            }

            return baseActivation;
        }
    }
}
