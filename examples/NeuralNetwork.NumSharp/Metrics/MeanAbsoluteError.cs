using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    public class MeanAbsoluteError : BaseMetric
    {
        public MeanAbsoluteError() : base("mean_absolute_error")
        {

        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            var error = preds - labels;
            return np.mean(np.abs(error));
        }
    }
}
