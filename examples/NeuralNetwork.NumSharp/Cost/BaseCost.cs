using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.NumSharp.Cost
{
    public abstract class BaseCost
    {
        public float Epsilon = 1e-7f;

        public string Name { get; set; }

        public BaseCost(string name)
        {
            Name = name;
        }

        public abstract NDArray Forward(NDArray preds, NDArray labels);

        public abstract NDArray Backward(NDArray preds, NDArray labels);
    }
}
