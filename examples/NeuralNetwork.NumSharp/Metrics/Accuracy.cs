using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NeuralNetwork.NumSharp.Metrics
{
    public class Accuacy : BaseMetric
    {
        public Accuacy() : base("accurary")
        {
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            var pred_idx = np.argmax(preds);
            var label_idx = np.argmax(labels);

            return np.mean(pred_idx == label_idx);
        }
    }
}
