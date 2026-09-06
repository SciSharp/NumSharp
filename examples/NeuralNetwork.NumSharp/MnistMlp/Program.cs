using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NeuralNetwork.NumSharp.Callbacks;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.Optimizers;
using NeuralNetwork.NumSharp.Regularizers;
using NeuralNetwork.NumSharp.Serialization;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Entry point for the MNIST MLP experiment. Runs:
    ///   1. Data load — real IDX files if present, otherwise deterministic
    ///      synthetic tensors (~10% accuracy at best; swap in real data to
    ///      train for real).
    ///   2. Fusion probe — correctness + perf of the fused bias+ReLU kernel
    ///      (hand-rolled NDIter AND the productized np.evaluate) against the
    ///      naive np.add + np.maximum composition, at TWO sizes: the training
    ///      shape (small — iterator setup dominates, the unfused whole-array
    ///      SIMD kernels win) and a large tensor (fusion's single memory pass
    ///      wins). Both are reported honestly; see the CLAUDE.md perf notes.
    ///   3. Training — 2-layer MLP (784 -> 128 ReLU -> 10) with Adam +
    ///      SoftmaxCrossEntropy loss. Per-epoch loss / accuracy, plus final
    ///      test-set accuracy.
    ///   4. Instrumentation — IL kernel-cache delta and NDExpr delegate-slot
    ///      count, showing the fused kernels are compiled exactly once and
    ///      reused across every forward/backward pass.
    /// </summary>
    public static class Program
    {
        private const int InputDim  = MnistLoader.ImageSize; // 784
        private const int HiddenDim = 128;
        private const int OutputDim = 10;

        private const int BatchSize = 128;
        private const int Epochs    = 100;

        public static int Main(string[] args)
        {
            Console.WriteLine("=== MNIST 2-Layer MLP (NDIter-fused forward & backward) ===");
            Console.WriteLine($"  Architecture : {InputDim} -> {HiddenDim} ReLU -> {OutputDim} logits  (float32)");
            Console.WriteLine($"  Batch size   : {BatchSize}");
            Console.WriteLine($"  Epochs       : {Epochs}");
            Console.WriteLine();

            // ---- 1. Load data ----
            string dataDir = FindDataDir();
            var (trainX, trainY, testX, testY, isSynthetic) =
                MnistLoader.LoadFullDataset(dataDir,
                    syntheticTrain: 6_000,  // 10x smaller than real MNIST — keeps synthetic runs fast
                    syntheticTest:  1_000,
                    seed: 42);

            Console.WriteLine(isSynthetic
                ? $"Data: SYNTHETIC — drop real IDX files into '{dataDir}' for genuine MNIST training"
                : $"Data: REAL MNIST loaded from {dataDir}");
            Console.WriteLine($"  train = ({trainX.shape[0]}, {trainX.shape[1]}) {trainX.dtype.Name}   labels ({trainY.shape[0]},) {trainY.dtype.Name}");
            Console.WriteLine($"  test  = ({testX.shape[0]}, {testX.shape[1]}) {testX.dtype.Name}   labels ({testY.shape[0]},) {testY.dtype.Name}");
            Console.WriteLine();

            // ---- 2. Fusion probe: correctness + abbreviated perf ----
            int cacheBefore = GeneratedDelegates.InnerLoopCount;
            RunFusionProbe(trainX, trainY);

            // ---- 3. Build model and train ----
            np.random.seed(1337);

            var layers = new List<BaseLayer>
            {
                new FullyConnectedFused(InputDim,  HiddenDim, FusedActivation.ReLU),
                new FullyConnectedFused(HiddenDim, OutputDim, FusedActivation.None),
            };
            var cost      = new SoftmaxCrossEntropy();
            var optimizer = new Adam(lr: 0.001f, beta_1: 0.9f, beta_2: 0.999f);

            Console.WriteLine("Training:");
            var result = MlpTrainer.Train(
                layers, cost, optimizer,
                trainX, trainY, testX, testY,
                epochs:    Epochs,
                batchSize: BatchSize,
                numClasses: OutputDim);
            Console.WriteLine($"  Total training time: {result.TotalMs / 1000.0:F1} s");
            Console.WriteLine();

            // ---- 4. P1 showcase: serialization + callbacks ----
            RunP1Showcase(layers, trainX, trainY, testX, testY, result.FinalTestAcc);

            // ---- 5. P4 showcase: regularization + normalization layers ----
            RunP4Showcase(trainX, trainY, testX, testY);

            // ---- 6. Instrumentation ----
            int cacheAfter = GeneratedDelegates.InnerLoopCount;
            Console.WriteLine("Kernel / delegate instrumentation:");
            Console.WriteLine($"  IL kernel cache entries : {cacheBefore} -> {cacheAfter} (delta {cacheAfter - cacheBefore})");
            Console.WriteLine($"  NDExpr delegate slots  : {DelegateSlots.RegisteredCount}");
            Console.WriteLine("  (Cache delta is a small constant: one kernel per unique expression + dtype");
            Console.WriteLine("   combination. Compiled once, hit on every subsequent forward/backward pass.)");

            return 0;
        }

        // =====================================================================
        // P1 showcase — weight/architecture round-trip and the callback stack.
        //
        // Deliberately separate from the headline run above, which stays a plain
        // 100-epoch fit so its convergence numbers remain comparable across
        // commits. Everything here is fast (a 12-epoch fit on the same data).
        // =====================================================================

        private static void RunP1Showcase(List<BaseLayer> trained, NDArray trainX, NDArray trainY,
                                          NDArray testX, NDArray testY, float trainedAcc)
        {
            Console.WriteLine("P1 — serialization round-trip:");

            string dir = Path.Combine(Path.GetTempPath(), "numsharp_nn_demo");
            Directory.CreateDirectory(dir);
            string weightsPath = Path.Combine(dir, "mlp.npz");
            string archPath = Path.Combine(dir, "mlp.json");

            // Weights -> .npz (a genuine NumPy archive), architecture -> JSON.
            ModelWeights.Save(trained, weightsPath);
            ModelArchitecture.Save(trained, archPath);
            Console.WriteLine($"  weights      : {weightsPath} ({new FileInfo(weightsPath).Length:N0} bytes)");
            Console.WriteLine($"  architecture : {archPath} ({new FileInfo(archPath).Length:N0} bytes)");

            // Rebuild from the architecture alone — freshly (and differently)
            // initialized — then restore the trained values into it.
            np.random.seed(4242);
            var reloaded = ModelArchitecture.Load(archPath);
            float untrainedAcc = MlpTrainer.Evaluate(reloaded, testX, testY, BatchSize);
            ModelWeights.Load(reloaded, weightsPath);
            float reloadedAcc = MlpTrainer.Evaluate(reloaded, testX, testY, BatchSize);

            Console.WriteLine($"  rebuilt from JSON, random weights : test_acc={untrainedAcc * 100:F2}%");
            Console.WriteLine($"  after loading the .npz            : test_acc={reloadedAcc * 100:F2}%  " +
                              $"(original {trainedAcc * 100:F2}%)  ->  " +
                              $"{(Math.Abs(reloadedAcc - trainedAcc) < 1e-6f ? "PASS" : "FAIL")}");
            Console.WriteLine();

            // ---- callbacks on a short, fresh run ----
            Console.WriteLine("P1 — callbacks (validation split, early stopping, LR plateau, CSV log, checkpoints):");

            np.random.seed(2024);
            var fresh = new List<BaseLayer>
            {
                new FullyConnectedFused(InputDim,  HiddenDim, FusedActivation.ReLU),
                new FullyConnectedFused(HiddenDim, OutputDim, FusedActivation.None),
            };

            // Gradient clipping is an optimizer property; clipnorm is Keras's
            // PER-PARAMETER norm (see BaseOptimizer for the global variant).
            var optimizer = new Adam(lr: 0.001f) { ClipNorm = 1.0f };

            string csvPath = Path.Combine(dir, "training_log.csv");
            var early = new EarlyStopping("val_loss", patience: 3, restoreBestWeights: true, verbose: 1);
            var plateau = new ReduceLROnPlateau("val_loss", factor: 0.5f, patience: 2, verbose: 1);
            var checkpoint = new ModelCheckpoint(Path.Combine(dir, "best.npz"), "val_loss",
                                                 saveBestOnly: true, verbose: 1);
            var csv = new CSVLogger(csvPath);

            var shortRun = MlpTrainer.Train(
                fresh, new SoftmaxCrossEntropy(), optimizer,
                trainX, trainY, testX, testY,
                epochs: 12, batchSize: BatchSize, numClasses: OutputDim,
                shuffle: true,
                validationSplit: 0.1f,
                callbacks: new List<BaseCallback> { early, plateau, checkpoint, csv },
                verbose: 1);

            Console.WriteLine($"  epochs run       : {shortRun.EpochsRun}/12" +
                              (shortRun.StoppedEarly ? $"  (EarlyStopping fired at epoch {early.StoppedEpoch + 1})" : "  (ran to completion)"));
            Console.WriteLine($"  best val_loss    : {early.Best:F5} at epoch {early.BestEpoch + 1}");
            Console.WriteLine($"  LR reductions    : {plateau.ReductionCount}  (final lr {optimizer.LearningRate})");
            Console.WriteLine($"  checkpoints kept : {checkpoint.SaveCount} (best-only) -> {checkpoint.LastSavedPath}");
            Console.WriteLine($"  csv log          : {csvPath}");
            foreach (string line in File.ReadLines(csvPath).Take(4))
                Console.WriteLine($"      {line}");
            Console.WriteLine();
        }

        // =====================================================================
        // P4 showcase — Dropout + BatchNormalization in a real stack, and the
        // train/eval mode switch that both of them read.
        // =====================================================================

        private static void RunP4Showcase(NDArray trainX, NDArray trainY, NDArray testX, NDArray testY)
        {
            Console.WriteLine("P4 — regularization & normalization layers:");

            np.random.seed(90210);
            var model = new List<BaseLayer>
            {
                new FullyConnected(InputDim, HiddenDim, "relu"),
                new BatchNormalization(HiddenDim),
                new Dropout(0.2f),
                new FullyConnected(HiddenDim, OutputDim),
            };

            // An L2 penalty on the first layer's kernel, applied by the trainer.
            model[0].Regularizers["w"] = new L2(1e-4f);

            var run = MlpTrainer.Train(
                model, new SoftmaxCrossEntropy(), new Adam(lr: 0.001f),
                trainX, trainY, testX, testY,
                epochs: 8, batchSize: BatchSize, numClasses: OutputDim,
                shuffle: true, validationSplit: 0.1f, verbose: 0);

            Console.WriteLine($"  stack        : Dense(relu) -> BatchNorm -> Dropout(0.2) -> Dense   (+ L2 1e-4 on the kernel)");
            Console.WriteLine($"  8 epochs     : loss {run.EpochLoss[0]:F4} -> {run.EpochLoss[^1]:F4}, " +
                              $"val_acc {run.EpochValAcc[0] * 100:F2}% -> {run.EpochValAcc[^1] * 100:F2}%, " +
                              $"test_acc {run.FinalTestAcc * 100:F2}%");

            // The mode flag is observable: Dropout randomizes and BatchNorm uses
            // batch statistics while Training is true, and neither does in
            // inference — so the SAME input gives different outputs.
            NDArray probe = trainX[$"0:{BatchSize}"];

            NDArray Run(bool training)
            {
                NDArray act = probe;
                foreach (var layer in model)
                {
                    layer.Training = training;
                    layer.Forward(act);
                    act = layer.Output;
                }
                return act;
            }

            NDArray evalA = Run(false);
            NDArray evalB = Run(false);
            NDArray trainA = Run(true);

            Console.WriteLine($"  Training=false is deterministic : max|eval1-eval2| = {MaxAbsDiff(evalA, evalB):g4}");
            Console.WriteLine($"  Training=true  differs          : max|train-eval|  = {MaxAbsDiff(trainA, evalA):g4}");
            Console.WriteLine($"  BatchNorm running mean[0..3]    : " +
                              string.Join(", ", Enumerable.Range(0, 4)
                                  .Select(i => model[1].NonTrainable["moving_mean"].GetSingle(i).ToString("F4"))));
            Console.WriteLine("  (Running statistics are non-trainable state: they ride in the .npz");
            Console.WriteLine("   checkpoint but never reach the optimizer.)");
            Console.WriteLine();
        }

        // =====================================================================
        // Fusion probe — quick correctness + speedup snapshot on one batch.
        // =====================================================================

        private static void RunFusionProbe(NDArray trainX, NDArray trainY)
        {
            Console.WriteLine("Fusion probe (bias+ReLU post-matmul; fused NDIter vs np.evaluate vs naive):");

            NDArray W = np.random.normal(0.0, Math.Sqrt(2.0 / InputDim), new Shape(InputDim, HiddenDim))
                               .astype(NPTypeCode.Single);
            NDArray b = np.zeros(new Shape(HiddenDim), NPTypeCode.Single);
            NDArray x = trainX[$"0:{BatchSize}"];

            NDArray fused = FusedMlp.Forward(x, W, b,
                              np.zeros(new Shape(HiddenDim, OutputDim), NPTypeCode.Single),
                              np.zeros(new Shape(OutputDim), NPTypeCode.Single));
            NDArray naive = NaiveMlp.Forward(x, W, b,
                              np.zeros(new Shape(HiddenDim, OutputDim), NPTypeCode.Single),
                              np.zeros(new Shape(OutputDim), NPTypeCode.Single));

            double maxDiff = MaxAbsDiff(fused, naive);
            Console.WriteLine($"  correctness  : max |fused - naive| = {maxDiff:g4}  ->  {(maxDiff < 1e-5 ? "PASS" : "FAIL")}");

            // np.evaluate (the productized fusion API) must agree bit-for-bit too.
            NDArray preact = np.dot(x, W);
            NDArray viaEvaluate = np.empty_like(preact);
            np.evaluate(NDExpr.Max(NDExpr.Arr(preact) + NDExpr.Arr(b), NDExpr.Const(0f)), @out: viaEvaluate);
            NDArray viaNaive = np.maximum(np.add(preact, b), (NDArray)0f);
            double evalDiff = MaxAbsDiff(viaEvaluate, viaNaive);
            Console.WriteLine($"  np.evaluate  : max |evaluate - naive| = {evalDiff:g4}  ->  {(evalDiff < 1e-5 ? "PASS" : "FAIL")}");

            // Two sizes, reported honestly: at the training shape (128x128 = 16K
            // elements) iterator setup dominates and the unfused whole-array SIMD
            // kernels win; on a large tensor the fused single memory pass wins.
            ProbeSize("training shape 128x128", preact, b, passes: 200, warmup: 500);

            NDArray bigPreact = np.random.normal(0, 1, new Shape(2048, 2048)).astype(NPTypeCode.Single);
            NDArray bigBias   = np.zeros(new Shape(2048), NPTypeCode.Single);
            ProbeSize("large tensor 2048x2048", bigPreact, bigBias, passes: 20, warmup: 5);
            Console.WriteLine();
        }

        private enum ProbePath
        {
            FusedIter,  // hand-rolled NDIterRef + ExecuteExpression (this demo's kernels)
            Evaluate,   // np.evaluate — the productized fusion API in core
            Naive,      // np.add + np.maximum, two whole-array SIMD kernels + an intermediate
        }

        private static void ProbeSize(string label, NDArray preact, NDArray bias, int passes, int warmup)
        {
            // Warm ALL paths up-front. At the small size 500 iterations covers
            // first-time IL emission + .NET's tiered JIT promotion to the
            // optimized tier; the large size reuses the already-hot kernels and
            // only needs to touch its memory once or twice.
            for (int i = 0; i < warmup; i++)
            {
                TimeProbe(preact, bias, 1, ProbePath.FusedIter);
                TimeProbe(preact, bias, 1, ProbePath.Evaluate);
                TimeProbe(preact, bias, 1, ProbePath.Naive);
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            double fusedMs = TimeProbe(preact, bias, passes, ProbePath.FusedIter);
            double evalMs  = TimeProbe(preact, bias, passes, ProbePath.Evaluate);
            double naiveMs = TimeProbe(preact, bias, passes, ProbePath.Naive);
            Console.WriteLine($"  {label,-22} : fusedIter {fusedMs:F3} ms | np.evaluate {evalMs:F3} ms | naive {naiveMs:F3} ms");
            Console.WriteLine($"  {"",-22}   naive/fusedIter {naiveMs / fusedMs:F2}x   naive/np.evaluate {naiveMs / evalMs:F2}x   (>1 = fusion faster)");
        }

        private static double TimeProbe(NDArray preact, NDArray bias, int passes, ProbePath path)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < passes; i++)
            {
                switch (path)
                {
                    case ProbePath.FusedIter:
                    {
                        NDArray h = np.empty_like(preact);
                        FusePostMatmulBiasRelu(preact, bias, h);
                        break;
                    }
                    case ProbePath.Evaluate:
                    {
                        NDArray h = np.empty_like(preact);
                        np.evaluate(NDExpr.Max(NDExpr.Arr(preact) + NDExpr.Arr(bias), NDExpr.Const(0f)), @out: h);
                        break;
                    }
                    case ProbePath.Naive:
                        _ = np.maximum(np.add(preact, bias), (NDArray)0f);
                        break;
                }
            }
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds / passes;
        }

        private static void FusePostMatmulBiasRelu(NDArray preact, NDArray bias, NDArray output)
        {
            using var iter = NDIterRef.MultiNew(
                nop: 3,
                op: new[] { preact, bias, output },
                flags:   NDIterGlobalFlags.EXTERNAL_LOOP,
                order:   NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_NO_CASTING,
                opFlags: new[]
                {
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.WRITEONLY,
                });

            var expr = NDExpr.Max(NDExpr.Input(0) + NDExpr.Input(1), NDExpr.Const(0f));
            iter.ExecuteExpression(expr,
                new[] { NPTypeCode.Single, NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "program_probe_bias_relu_f32");
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private static string FindDataDir()
        {
            string[] candidates =
            {
                Path.Combine(AppContext.BaseDirectory, "data"),
                Path.Combine(Directory.GetCurrentDirectory(), "data"),
                Path.Combine(Directory.GetCurrentDirectory(), "examples", "NeuralNetwork.NumSharp", "data"),
            };
            foreach (var c in candidates)
                if (Directory.Exists(c)) return c;
            return candidates[0];
        }

        private static double MaxAbsDiff(NDArray a, NDArray b)
        {
            int rows = (int)a.shape[0];
            int cols = (int)a.shape[1];
            double max = 0.0;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    double d = Math.Abs(a.GetSingle(i, j) - b.GetSingle(i, j));
                    if (d > max) max = d;
                }
            return max;
        }
    }
}
