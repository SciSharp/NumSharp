using System;
using System.Collections.Generic;
using System.Diagnostics;
using NeuralNetwork.NumSharp.Callbacks;
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
    ///   callbacks.OnEpochBegin
    ///   (optionally) reshuffle the sample order
    ///   for b in batches:
    ///     forward  through layers (x -> y)
    ///     loss     = cost.Forward(y, y_true_onehot)
    ///     grad     = cost.Backward(y, y_true_onehot)
    ///     backward through layers in reverse (grad -> ...)
    ///     optimizer.ApplyGlobalClipNorm, then optimizer.Update per layer
    ///     callbacks.OnBatchEnd
    ///   score the validation set (if any)
    ///   callbacks.OnEpochEnd  -> may set StopTraining
    ///
    /// <para><b>Keras parity notes.</b> Shuffling draws from <c>np.random</c>
    /// (MT19937), so a seeded run is reproducible end to end.
    /// <c>validationSplit</c> takes the LAST fraction of the data and does so
    /// BEFORE any shuffling, exactly as Keras does — which means the split is
    /// deterministic and a caller whose data is ordered by class must shuffle it
    /// themselves first. Every batch is trained on, including a final partial
    /// one; epoch loss and accuracy are averaged over SAMPLES, so a short last
    /// batch is weighted correctly rather than counting as a whole batch.</para>
    /// </summary>
    public static class MlpTrainer
    {
        public readonly record struct TrainResult(
            int Epochs,
            List<float> EpochLoss,
            List<float> EpochTrainAcc,
            List<(int Epoch, float TestAcc)> TestEvals,
            float FinalTestAcc,
            long TotalMs,
            List<float> EpochValLoss,
            List<float> EpochValAcc,
            int EpochsRun,
            bool StoppedEarly);

        /// <summary>A validation set: inputs plus INTEGER labels (not one-hot).</summary>
        public readonly record struct ValidationSet(NDArray X, NDArray YLabels);

        public static TrainResult Train(
            List<BaseLayer> layers,
            BaseCost cost,
            BaseOptimizer optimizer,
            NDArray trainX, NDArray trainYLabels,
            NDArray testX,  NDArray testYLabels,
            int epochs,
            int batchSize,
            int numClasses,
            bool shuffle = true,
            float validationSplit = 0f,
            ValidationSet? validationData = null,
            IReadOnlyList<BaseCallback> callbacks = null,
            int verbose = 1)
        {
            if (layers == null || layers.Count == 0) throw new ArgumentException("at least one layer is required", nameof(layers));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (epochs < 0) throw new ArgumentOutOfRangeException(nameof(epochs));
            if (validationSplit < 0f || validationSplit >= 1f)
                throw new ArgumentOutOfRangeException(nameof(validationSplit), "validationSplit must be in [0, 1).");

            callbacks ??= Array.Empty<BaseCallback>();

            // ---- validation split: the LAST fraction, taken before shuffling ----
            NDArray valX = null, valYLabels = null;
            int fullN = (int)trainX.shape[0];

            if (validationData.HasValue)
            {
                valX = validationData.Value.X;
                valYLabels = validationData.Value.YLabels;
            }
            else if (validationSplit > 0f)
            {
                int splitAt = (int)(fullN * (1f - validationSplit));
                if (splitAt <= 0 || splitAt >= fullN)
                    throw new ArgumentOutOfRangeException(nameof(validationSplit),
                        $"validationSplit={validationSplit} leaves {splitAt} of {fullN} samples for training.");

                valX       = trainX[$"{splitAt}:{fullN}"];
                valYLabels = trainYLabels[$"{splitAt}:{fullN}"];
                trainX     = trainX[$"0:{splitAt}"];
                trainYLabels = trainYLabels[$"0:{splitAt}"];
            }

            NDArray trainYOneHot = SoftmaxCrossEntropy.OneHot(trainYLabels, numClasses);
            NDArray valYOneHot = valX is not null ? SoftmaxCrossEntropy.OneHot(valYLabels, numClasses) : null;

            int trainN = (int)trainX.shape[0];
            // Ceiling division — the final batch may be partial and IS trained on.
            int numBatches = (trainN + batchSize - 1) / batchSize;
            int iteration = 0;

            // Evaluate the test set every min(5, epochs) epochs. For short runs
            // (epochs ≤ 5) this means every epoch; for longer runs it's every 5.
            // The final epoch always gets a test eval regardless of cadence.
            bool hasTest = testX is not null && testYLabels is not null && testX.size > 0;
            int evalEvery = Math.Max(1, Math.Min(5, epochs));

            var epochLosses = new List<float>();
            var epochTrainAccs = new List<float>();
            var epochValLosses = new List<float>();
            var epochValAccs = new List<float>();
            var testEvals = new List<(int Epoch, float TestAcc)>();

            var context = new TrainingContext(layers, optimizer, epochs, batchSize, numBatches, valX is not null);
            foreach (var cb in callbacks)
                cb.SetContext(context);

            if (verbose > 0)
            {
                Console.WriteLine($"  Training: {numBatches} batches/epoch x {epochs} epochs, batch_size={batchSize}" +
                                  (shuffle ? ", shuffled" : ", in order"));
                if (valX is not null)
                    Console.WriteLine($"  Validation: {valX.shape[0]} samples" +
                                      (validationData.HasValue ? " (explicit validation_data)" : $" (validation_split={validationSplit})"));
                if (hasTest)
                    Console.WriteLine($"  Test evaluation every {evalEvery} epoch(s).");
            }

            int epochsRun = 0;
            var totalSw = Stopwatch.StartNew();

            try
            {
                foreach (var cb in callbacks)
                    cb.OnTrainBegin();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    foreach (var cb in callbacks)
                        cb.OnEpochBegin(epoch);

                    var epochSw = Stopwatch.StartNew();
                    double epochLossSum = 0.0;   // sample-weighted
                    int epochCorrect = 0;
                    int epochCount = 0;

                    // One permutation per epoch, consumed batch by batch. Gathering
                    // per batch rather than materializing a shuffled copy of the
                    // whole set keeps peak memory at one batch instead of a second
                    // copy of the training data.
                    NDArray perm = shuffle ? np.random.permutation(trainN) : null;

                    for (int b = 0; b < numBatches; b++)
                    {
                        int start = b * batchSize;
                        int end = Math.Min(start + batchSize, trainN);
                        int count = end - start;

                        NDArray xBatch, yBatch, yLabelBatch;
                        if (shuffle)
                        {
                            // A raw index array as the SOLE index is FANCY indexing in
                            // NumSharp (row selection) — the copy that produces is what
                            // we want here.
                            NDArray idx = perm[$"{start}:{end}"];
                            xBatch      = trainX[idx];
                            yBatch      = trainYOneHot[idx];
                            yLabelBatch = trainYLabels[idx];
                        }
                        else
                        {
                            xBatch      = trainX[$"{start}:{end}"];
                            yBatch      = trainYOneHot[$"{start}:{end}"];
                            yLabelBatch = trainYLabels[$"{start}:{end}"];
                        }

                        // --- forward ---
                        NDArray act = xBatch;
                        foreach (var layer in layers)
                        {
                            layer.Forward(act);
                            act = layer.Output;
                        }

                        // --- loss + accuracy ---
                        float batchLoss = (float)cost.Forward(act, yBatch);
                        epochLossSum += (double)batchLoss * count;

                        NDArray predIdx = np.argmax(act, axis: 1);
                        int batchCorrect = CountMatches(predIdx, yLabelBatch);
                        epochCorrect += batchCorrect;
                        epochCount += count;

                        // --- backward ---
                        NDArray grad = cost.Backward(act, yBatch);
                        for (int i = layers.Count - 1; i >= 0; i--)
                        {
                            layers[i].Backward(grad);
                            grad = layers[i].InputGrad;
                        }

                        // --- optimizer step ---
                        // Model-wide clipping must see every layer's gradients at
                        // once, so it runs before the per-layer updates.
                        optimizer.ApplyGlobalClipNorm(layers);
                        iteration++;
                        foreach (var layer in layers)
                            optimizer.Update(iteration, layer);

                        if (callbacks.Count > 0)
                        {
                            var batchLogs = new Dictionary<string, float>
                            {
                                ["loss"] = batchLoss,
                                ["acc"] = count > 0 ? (float)batchCorrect / count : 0f,
                            };
                            foreach (var cb in callbacks)
                                cb.OnBatchEnd(b, batchLogs);
                        }

                        if (verbose >= 2)
                            Console.WriteLine($"    batch {b + 1,4}/{numBatches}  loss={batchLoss:F4}  " +
                                              $"acc={(count > 0 ? (float)batchCorrect / count : 0f) * 100:F2}%  (n={count})");
                    }

                    float avgLoss = epochCount > 0 ? (float)(epochLossSum / epochCount) : 0f;
                    float trainAcc = epochCount > 0 ? (float)epochCorrect / epochCount : 0f;
                    epochLosses.Add(avgLoss);
                    epochTrainAccs.Add(trainAcc);

                    var logs = new Dictionary<string, float>
                    {
                        ["loss"] = avgLoss,
                        ["acc"] = trainAcc,
                    };

                    // --- validation ---
                    string valCol = "";
                    if (valX is not null)
                    {
                        var (valLoss, valAcc) = EvaluateFull(layers, cost, valX, valYOneHot, valYLabels, batchSize);
                        epochValLosses.Add(valLoss);
                        epochValAccs.Add(valAcc);
                        logs["val_loss"] = valLoss;
                        logs["val_acc"] = valAcc;
                        valCol = $"  val_loss={valLoss:F4}  val_acc={valAcc * 100:F2}%";
                    }

                    logs["learning_rate"] = optimizer.LearningRate;
                    epochSw.Stop();

                    // Periodic test evaluation. The final epoch is always evaluated
                    // regardless of cadence so the caller always gets a finalTestAcc.
                    bool lastEpoch = epoch == epochs - 1 || context.StopTraining;
                    bool doEval = hasTest && (((epoch + 1) % evalEvery == 0) || lastEpoch);
                    string evalCol = hasTest ? "                    " : "";  // same width as "  test_acc=99.99%"
                    if (doEval)
                    {
                        float testAcc = Evaluate(layers, testX, testYLabels, batchSize);
                        testEvals.Add((epoch + 1, testAcc));
                        evalCol = $"  test_acc={testAcc * 100:F2}%  ";
                    }

                    if (verbose > 0)
                        Console.WriteLine($"  Epoch {epoch + 1,3}/{epochs}  loss={avgLoss:F4}  train_acc={trainAcc * 100:F2}%{valCol}{evalCol}" +
                                          $"({epochSw.ElapsedMilliseconds} ms, total {totalSw.ElapsedMilliseconds / 1000.0:F1} s)");

                    foreach (var cb in callbacks)
                        cb.OnEpochEnd(epoch, logs);

                    epochsRun = epoch + 1;

                    if (context.StopTraining)
                    {
                        // A callback may have rolled the weights back (EarlyStopping
                        // with RestoreBestWeights), so the final test score has to be
                        // taken AFTER the callbacks ran, not from the cached value.
                        if (hasTest)
                        {
                            float restoredAcc = Evaluate(layers, testX, testYLabels, batchSize);
                            testEvals.Add((epoch + 1, restoredAcc));
                        }
                        break;
                    }
                }
            }
            finally
            {
                // Runs even if the loop threw: CSVLogger has a file handle open.
                foreach (var cb in callbacks)
                    cb.OnTrainEnd();
            }

            totalSw.Stop();

            float finalTestAcc = testEvals.Count > 0 ? testEvals[^1].TestAcc : 0f;
            if (verbose > 0 && hasTest)
                Console.WriteLine($"  Final test accuracy: {finalTestAcc * 100:F2}%");

            return new TrainResult(epochs, epochLosses, epochTrainAccs, testEvals, finalTestAcc,
                                   totalSw.ElapsedMilliseconds, epochValLosses, epochValAccs,
                                   epochsRun, context.StopTraining);
        }

        /// <summary>
        /// Runs the layer stack forward over the full dataset in batches,
        /// taking argmax per row and counting matches against integer labels.
        /// The final batch may be smaller than <paramref name="batchSize"/> —
        /// every sample is scored (the old floor-division silently dropped the
        /// remainder: 1000 test samples at batch 128 scored only 896).
        /// </summary>
        public static float Evaluate(List<BaseLayer> layers, NDArray x, NDArray yLabels, int batchSize)
        {
            int n = (int)x.shape[0];
            if (n == 0)
                return 0f;
            int correct = 0;

            for (int start = 0; start < n; start += batchSize)
            {
                int end = Math.Min(start + batchSize, n);
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

            return (float)correct / n;
        }

        /// <summary>
        /// Scores loss AND accuracy in one forward pass over the data — what a
        /// per-epoch validation report needs. Both are sample-weighted, so a
        /// partial final batch contributes proportionally.
        /// </summary>
        public static (float Loss, float Accuracy) EvaluateFull(
            IReadOnlyList<BaseLayer> layers, BaseCost cost,
            NDArray x, NDArray yOneHot, NDArray yLabels, int batchSize)
        {
            int n = (int)x.shape[0];
            if (n == 0)
                return (0f, 0f);

            double lossSum = 0.0;
            int correct = 0;

            for (int start = 0; start < n; start += batchSize)
            {
                int end = Math.Min(start + batchSize, n);
                int count = end - start;

                NDArray act = x[$"{start}:{end}"];
                foreach (var layer in layers)
                {
                    layer.Forward(act);
                    act = layer.Output;
                }

                lossSum += (double)(float)cost.Forward(act, yOneHot[$"{start}:{end}"]) * count;
                correct += CountMatches(np.argmax(act, axis: 1), yLabels[$"{start}:{end}"]);
            }

            return ((float)(lossSum / n), (float)correct / n);
        }

        /// <summary>
        /// Compares predicted class indices (Int64 from np.argmax) against
        /// integer labels. The label array's dtype depends on where it came from
        /// (MnistLoader yields Byte, a fancy-index gather preserves it, a
        /// hand-built array is usually Int32), so the comparison goes through a
        /// widening read rather than assuming one width.
        /// </summary>
        private static int CountMatches(NDArray predIdx, NDArray labels)
        {
            int n = (int)predIdx.shape[0];
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (predIdx.GetInt64(i) == LabelAt(labels, i))
                    correct++;
            return correct;
        }

        private static long LabelAt(NDArray labels, int i)
        {
            switch (labels.typecode)
            {
                case NPTypeCode.Byte:   return labels.GetByte(i);
                case NPTypeCode.Int32:  return labels.GetInt32(i);
                case NPTypeCode.Int64:  return labels.GetInt64(i);
                case NPTypeCode.Int16:  return labels.GetInt16(i);
                case NPTypeCode.UInt16: return labels.GetUInt16(i);
                case NPTypeCode.UInt32: return labels.GetUInt32(i);
                case NPTypeCode.SByte:  return labels.GetSByte(i);
                default:                return (long)Convert.ToDouble(labels.GetValue(i));
            }
        }
    }
}
