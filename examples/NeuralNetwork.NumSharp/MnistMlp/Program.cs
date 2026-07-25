using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.Optimizers;
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

            // ---- 4. Instrumentation ----
            int cacheAfter = GeneratedDelegates.InnerLoopCount;
            Console.WriteLine("Kernel / delegate instrumentation:");
            Console.WriteLine($"  IL kernel cache entries : {cacheBefore} -> {cacheAfter} (delta {cacheAfter - cacheBefore})");
            Console.WriteLine($"  NDExpr delegate slots  : {DelegateSlots.RegisteredCount}");
            Console.WriteLine("  (Cache delta is a small constant: one kernel per unique expression + dtype");
            Console.WriteLine("   combination. Compiled once, hit on every subsequent forward/backward pass.)");

            return 0;
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
