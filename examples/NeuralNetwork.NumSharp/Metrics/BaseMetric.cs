using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    public abstract class BaseMetric
    {
        public string Name { get; set; }

        public BaseMetric(string name)
        {
            Name = name;
        }

        public abstract NDArray Calculate(NDArray preds, NDArray labels);
    }
}
