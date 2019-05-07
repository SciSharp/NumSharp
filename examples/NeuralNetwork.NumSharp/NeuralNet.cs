using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Diagnostics;
using NeuralNetwork.NumSharp.Metrics;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Optimizers;
using NeuralNetwork.NumSharp.Layers;
using NumSharp;

namespace NeuralNetwork.NumSharp
{
    /// <summary>
    /// Sequential model builder with train and predict
    /// </summary>
    public class NeuralNet
    {
        public event EventHandler<EpochEndEventArgs> EpochEnd;

        /// <summary>
        /// Layers which the model will contain
        /// </summary>
        public List<BaseLayer> Layers { get; set; }

        /// <summary>
        /// The optimizer instance used during training
        /// </summary>
        public BaseOptimizer Optimizer { get; set; }

        /// <summary>
        /// The cost instance for the training
        /// </summary>
        public BaseCost Cost { get; set; }

        /// <summary>
        /// The metric instance for the training
        /// </summary>
        public BaseMetric Metric { get; set; }

        /// <summary>
        /// Training losses for all the iterations
        /// </summary>
        public List<float> TrainingLoss { get; set; }

        /// <summary>
        /// Training metrices for all the iterations
        /// </summary>
        public List<float> TrainingMetrics { get; set; }

        /// <summary>
        /// Create instance of the neural net with parameters
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="cost"></param>
        /// <param name="metric"></param>
        public NeuralNet(BaseOptimizer optimizer, BaseCost cost, BaseMetric metric = null)
        {
            Layers = new List<BaseLayer>();
            TrainingLoss = new List<float>();
            TrainingMetrics = new List<float>();

            this.Optimizer = optimizer != null ? optimizer : throw new Exception("Need optimizer");
            this.Cost = cost != null ? cost : throw new Exception("Need cost");
            Metric = metric;
        }

        /// <summary>
        /// Helper method to stack layer
        /// </summary>
        /// <param name="layer"></param>
        public void Add(BaseLayer layer)
        {
            Layers.Add(layer);
        }

        /// <summary>
        /// Train the model with training dataset, for certain number of iterations and using batch size
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="numIterations"></param>
        /// <param name="batchSize"></param>
        public void Train(NDArray x, NDArray y, int numIterations, int batchSize)
        {
            //Initialise bacch loss and metric list for temporary holding of result
            List<float> batchLoss = new List<float>();
            List<float> batchMetrics = new List<float>();

            Stopwatch sw = new Stopwatch();

            //Loop through till the end of specified iterations
            for (int i = 1; i <= numIterations; i++)
            {
                sw.Start();

                //Initialize local variables
                int currentIndex = 0;
                batchLoss.Clear();
                batchMetrics.Clear();

                //Loop untill the data is exhauted for every batch selected
                while (true)
                {
                    //Get the batch data based on the specified batch size
                    var xtrain = x[currentIndex, currentIndex + batchSize];
                    var ytrain = y[currentIndex, currentIndex + batchSize];

                    if (xtrain == null)
                        break;

                    //Run forward for all the layers to predict the value for the training set
                    var ypred = Forward(xtrain);

                    //Find the loss/cost value for the prediction wrt expected result
                    var costVal = Cost.Forward(ypred, ytrain);
                    batchLoss.AddRange(costVal.Data<float>());

                    //Find the metric value for the prediction wrt expected result
                    if (Metric != null)
                    {
                        var metric = Metric.Calculate(ypred, ytrain);
                        batchMetrics.AddRange(metric.Data<float>());
                    }

                    //Get the gradient of the cost function which is the passed to the layers during back-propagation
                    var grad = Cost.Backward(ypred, ytrain);

                    //Run back-propagation accross all the layers
                    Backward(grad);

                    //Now time to update the neural network weights using the specified optimizer function
                    foreach (var layer in Layers)
                    {
                        Optimizer.Update(i, layer);
                    }

                    currentIndex = currentIndex + batchSize; ;
                }

                sw.Stop();
                //Collect the result and fire the event
                float batchLossAvg = (float)Math.Round(batchLoss.Average(), 2);

                float batchMetricAvg = Metric != null ? (float)Math.Round(batchMetrics.Average(), 2) : 0;

                TrainingLoss.Add(batchLossAvg);

                if(batchMetrics.Count > 0)
                    TrainingMetrics.Add(batchMetricAvg);

                EpochEndEventArgs eventArgs = new EpochEndEventArgs(i, batchLossAvg, batchMetricAvg, sw.ElapsedMilliseconds);
                EpochEnd?.Invoke(i, eventArgs);
                sw.Reset();
            }
        }

        /// <summary>
        /// Prediction method
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public NDArray Predict(NDArray x)
        {
            return Forward(x);
        }

        /// <summary>
        /// Internal method to execute forward method accross all the layers
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private NDArray Forward(NDArray x)
        {
            BaseLayer lastLayer = null;

            foreach (var layer in Layers)
            {
                if (lastLayer == null)
                    layer.Forward(x);
                else
                    layer.Forward(lastLayer.Output);

                lastLayer = layer;
            }

            return lastLayer.Output;
        }

        /// <summary>
        /// Internal method to execute back-propagation method accross all the layers
        /// </summary>
        /// <param name="gradOutput"></param>
        private void Backward(NDArray gradOutput)
        {
            var curGradOutput = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var layer = Layers[i];

                layer.Backward(curGradOutput);
                curGradOutput = layer.InputGrad;
            }
        }
    }

    public class EpochEndEventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchEndEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        /// <param name="batch">The current batch number.</param>
        /// <param name="loss">The loss value for the batch.</param>
        /// <param name="metric">The metric value for the batch.</param>
        public EpochEndEventArgs(
            int epoch,
            float loss,
            float metric,
            long duration)
        {
            Epoch = epoch;
            Loss = loss;
            Metric = metric;
            Duration = duration;
        }

        /// <summary>
        /// Gets the current epoch number.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }

        /// <summary>
        /// Gets the loss value for this batch.
        /// </summary>
        /// <value>
        /// The loss.
        /// </value>
        public float Loss { get; }

        /// <summary>
        /// Gets the metric value for this batch.
        /// </summary>
        /// <value>
        /// The metric.
        /// </value>
        public float Metric { get; }

        /// <summary>
        /// Time taken in ms per iteration
        /// </summary>
        public long Duration { get; }
    }
}
