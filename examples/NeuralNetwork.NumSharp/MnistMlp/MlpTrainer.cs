using System;
using System.Collections.Generic;
using System.Diagnostics;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.Optimizers;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Training + evaluation loop for a classification MLP built on top of the
    /// NeuralNetwork.NumSharp BaseLayer / BaseCost / BaseOptimizer abstractions.
    ///
    /// Why not use NeuralNet.Train? The built-in loop uses
    /// <c>x[currentIndex, currentIndex + batchSize]</c> which is 2-index
    /// integer indexing in NumSharp (selects a single element), not slicing —
    /// the loop silently reads the wrong data. This trainer uses the correct
    /// <c>x[$"{start}:{end}"]</c> string-slice form and skips the broken
    /// abstraction entirely.
    ///
    /// Flow per epoch:
    ///   for b in batches:
    ///     forward  through layers (x -> y)
    ///     loss     = cost.Forward(y, y_true_onehot)
    ///     grad     = cost.Backward(y, y_true_onehot)
    ///     backward through layers in reverse (grad -> ...)
    ///     optimizer.Update(iter, each layer)
    ///
    /// Batches are taken in order (no per-epoch shuffle). MNIST's training set
    /// is pre-shuffled by the distributor, so this gives a reasonable but not
    /// ideal signal for SGD — adequate for demonstrating fusion + convergence.
    /// </summary>
    public static class MlpTrainer
    {
        public readonly record struct TrainResult(
            int Epochs,
            List<float> EpochLoss,
            List<float> EpochTrainAcc,
            float FinalTestAcc,
            long TotalMs);

        public static TrainResult Train(
            List<BaseLayer> layers,
            BaseCost cost,
            BaseOptimizer optimizer,
            NDArray trainX, NDArray trainYLabels,
            NDArray testX,  NDArray testYLabels,
            int epochs,
            int batchSize,
            int numClasses)
        {
            NDArray trainYOneHot = SoftmaxCrossEntropy.OneHot(trainYLabels, numClasses);

            int trainN = (int)trainX.shape[0];
            int numBatches = trainN / batchSize;
            int iteration = 0;

            var epochLosses = new List<float>();
            var epochTrainAccs = new List<float>();

            Console.WriteLine($"  Training: {numBatches} batches/epoch x {epochs} epochs, batch_size={batchSize}");

            var totalSw = Stopwatch.StartNew();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var epochSw = Stopwatch.StartNew();
                float epochLossSum = 0f;
                int   epochCorrect = 0;
                int   epochCount   = 0;

                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end   = start + batchSize;

                    NDArray xBatch      = trainX[$"{start}:{end}"];
                    NDArray yBatch      = trainYOneHot[$"{start}:{end}"];
                    NDArray yLabelBatch = trainYLabels[$"{start}:{end}"];

                    // --- forward ---
                    NDArray act = xBatch;
                    foreach (var layer in layers)
                    {
                        layer.Forward(act);
                        act = layer.Output;
                    }

                    // --- loss + accuracy ---
                    NDArray lossVal = cost.Forward(act, yBatch);
                    epochLossSum += (float)lossVal;

                    NDArray predIdx = np.argmax(act, axis: 1);
                    epochCorrect += CountMatches(predIdx, yLabelBatch);
                    epochCount   += batchSize;

                    // --- backward ---
                    NDArray grad = cost.Backward(act, yBatch);
                    for (int i = layers.Count - 1; i >= 0; i--)
                    {
                        layers[i].Backward(grad);
                        grad = layers[i].InputGrad;
                    }

                    // --- optimizer step ---
                    iteration++;
                    foreach (var layer in layers)
                        optimizer.Update(iteration, layer);
                }

                float avgLoss = epochLossSum / numBatches;
                float trainAcc = (float)epochCorrect / epochCount;
                epochLosses.Add(avgLoss);
                epochTrainAccs.Add(trainAcc);
                epochSw.Stop();

                Console.WriteLine($"  Epoch {epoch + 1,2}/{epochs}  loss={avgLoss:F4}  train_acc={trainAcc * 100:F2}%  " +
                                  $"({epochSw.ElapsedMilliseconds} ms, total {totalSw.ElapsedMilliseconds / 1000.0:F1} s)");
            }
            totalSw.Stop();

            // --- test-set evaluation ---
            float testAcc = Evaluate(layers, testX, testYLabels, batchSize);
            Console.WriteLine($"  Final test accuracy: {testAcc * 100:F2}%");

            return new TrainResult(epochs, epochLosses, epochTrainAccs, testAcc, totalSw.ElapsedMilliseconds);
        }

        /// <summary>
        /// Runs the layer stack forward over the full dataset in batches,
        /// taking argmax per row and counting matches against integer labels.
        /// Uses the same batch size as training so batches divide evenly.
        /// </summary>
        public static float Evaluate(List<BaseLayer> layers, NDArray x, NDArray yLabels, int batchSize)
        {
            int n = (int)x.shape[0];
            int numBatches = n / batchSize;
            int correct = 0;

            for (int b = 0; b < numBatches; b++)
            {
                int start = b * batchSize;
                int end   = start + batchSize;
                NDArray xBatch = x[$"{start}:{end}"];
                NDArray yBatch = yLabels[$"{start}:{end}"];

                NDArray act = xBatch;
                foreach (var layer in layers)
                {
                    layer.Forward(act);
                    act = layer.Output;
                }

                NDArray predIdx = np.argmax(act, axis: 1);
                correct += CountMatches(predIdx, yBatch);
            }

            return (float)correct / (numBatches * batchSize);
        }

        /// <summary>
        /// Compares predicted class indices (Int64 from np.argmax) against
        /// label bytes (from MnistLoader). Returns the count of matches.
        /// </summary>
        private static int CountMatches(NDArray predIdx, NDArray labels)
        {
            int n = (int)predIdx.shape[0];
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (predIdx.GetInt64(i) == labels.GetByte(i))
                    correct++;
            return correct;
        }
    }
}
