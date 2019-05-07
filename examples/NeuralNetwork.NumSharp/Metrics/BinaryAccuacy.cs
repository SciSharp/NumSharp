using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    public class BinaryAccuacy : BaseMetric
    {
        public BinaryAccuacy() : base("binary_accurary")
        {
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            //ToDo: np.round and np.clip
            //var output = Round(Clip(preds, 0, 1));
            //return np.mean(output == labels);
            return null;
        }
    }
}
