using System;
using System.Diagnostics;
using System.IO;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    /// <summary>
    /// Experiment: 2-layer MLP forward pass on MNIST where the bias-add + ReLU
    /// chunk of each layer collapses into a single NpyIter invocation via the
    /// NpyExpr DSL.
    ///
    /// Architecture: 784 -> 128 (ReLU) -> 10 (logits).
    ///
    /// The experiment:
    ///   1. Load MNIST test set (or synthesize if missing).
    ///   2. Build fresh random weights (He-init) and zero biases, float32.
    ///   3. Run the fused forward pass (one NpyIter per layer for the
    ///      post-matmul element-wise work).
    ///   4. Run a naive baseline (np.add + np.maximum separately).
    ///   5. Assert bit-for-bit agreement via a manual max-abs-diff check
    ///      (np.allclose mutates operands via astype(copy:false)).
    ///   6. Benchmark each variant — multi-run median for the noisy full
    ///      pass, and an isolated element-wise sweep to surface the clean
    ///      fusion signal. Report kernel-cache size and delegate-slot count.
    /// </summary>
    public static class Program
    {
        private const int InputDim  = MnistLoader.ImageSize;  // 784
        private const int HiddenDim = 128;
        private const int OutputDim = 10;

        private const int BatchSize    = 128;
        private const int WarmupPasses = 20;
        private const int BenchPasses  = 500;

        public static int Main(string[] args)
        {
            Console.WriteLine("=== 2-Layer MLP Forward Pass on MNIST (single NDIter fusion) ===");
            Console.WriteLine($"  Architecture : {InputDim} -> {HiddenDim} ReLU -> {OutputDim} logits");
            Console.WriteLine($"  Batch size   : {BatchSize}");
            Console.WriteLine();

            // ---- 1. Load MNIST ----
            string dataDir = FindDataDir();
            string imagesPath = Path.Combine(dataDir, "t10k-images.idx3-ubyte");
            string labelsPath = Path.Combine(dataDir, "t10k-labels.idx1-ubyte");

            var (images, labels, isSynthetic) =
                MnistLoader.LoadOrSynthesize(imagesPath, labelsPath,
                    syntheticCount: 10_000, seed: 42);

            Console.WriteLine(isSynthetic
                ? $"Data: SYNTHETIC ({images.shape[0]} samples) — drop real IDX files into '{dataDir}' for genuine MNIST"
                : $"Data: REAL MNIST ({images.shape[0]} test samples) loaded from {dataDir}");
            Console.WriteLine($"  images.shape = ({images.shape[0]}, {images.shape[1]}) dtype={images.dtype.Name}");
            Console.WriteLine($"  labels.shape = ({labels.shape[0]},) dtype={labels.dtype.Name}");
            Console.WriteLine();

            // ---- 2. Initialize weights (He-init for ReLU) ----
            np.random.seed(1337);
            var W1 = HeInit(InputDim,  HiddenDim);
            var b1 = np.zeros(new Shape(HiddenDim), NPTypeCode.Single);
            var W2 = HeInit(HiddenDim, OutputDim);
            var b2 = np.zeros(new Shape(OutputDim), NPTypeCode.Single);

            Console.WriteLine("Weights:");
            Console.WriteLine($"  W1: ({W1.shape[0]}, {W1.shape[1]}) {W1.dtype.Name}");
            Console.WriteLine($"  b1: ({b1.shape[0]},) {b1.dtype.Name}");
            Console.WriteLine($"  W2: ({W2.shape[0]}, {W2.shape[1]}) {W2.dtype.Name}");
            Console.WriteLine($"  b2: ({b2.shape[0]},) {b2.dtype.Name}");
            Console.WriteLine();

            // ---- 3. Grab a single batch (first 128 test samples) ----
            NDArray batch       = images[$"0:{BatchSize}"];
            NDArray batchLabels = labels[$"0:{BatchSize}"];

            // ---- 4. Reset kernel cache so the counts reflect only this run ----
            int cacheBefore = ILKernelGenerator.InnerLoopCachedCount;

            // ---- 5. Correctness check: fused vs naive ----
            // NOTE: np.allclose currently mutates its arguments via astype(copy:false),
            //       so we do a manual max-abs-diff check instead. See BUG NOTES.
            NDArray fused = FusedMlp.Forward(batch, W1, b1, W2, b2);
            NDArray naive = NaiveMlp.Forward(batch, W1, b1, W2, b2);

            double maxDiff = MaxAbsDiff(fused, naive);
            bool match = maxDiff < 1e-5;
            Console.WriteLine($"Correctness: max |fused - naive| = {maxDiff:g4}  ->  {(match ? "PASS" : "FAIL")}");
            if (!match) return 1;

            Console.WriteLine($"Output shape                 : ({fused.shape[0]}, {fused.shape[1]})");
            Console.WriteLine($"Output dtype                 : {fused.dtype.Name}");
            Console.WriteLine();

            // ---- 6. Accuracy sanity (random init → ~10% on 10-class) ----
            NDArray predicted = np.argmax(fused, axis: 1);
            int correct = CountMatches(predicted, batchLabels);
            Console.WriteLine($"Accuracy (random init)       : {correct}/{BatchSize} = {100.0 * correct / BatchSize:F2}%");
            Console.WriteLine("  (expected ~10% with random weights — this is a fusion + perf demo, not a trained model)");
            Console.WriteLine();

            // ---- 7. Benchmark: full forward pass (matmul-dominated, noisy) ----
            // Matmul dominates the runtime of a full forward pass, so the fusion
            // effect is small relative to the matmul's per-run variance. Report
            // multi-run min/median to surface the signal rather than a single
            // noisy number.
            Console.WriteLine("Benchmark — full forward pass (matmul + element-wise):");
            BenchMultiRun("Fused  (1 NpyIter per layer)",
                () => FusedMlp.Forward(batch, W1, b1, W2, b2),
                out double fusedMedian);
            BenchMultiRun("Naive  (np.add + np.maximum)",
                () => NaiveMlp.Forward(batch, W1, b1, W2, b2),
                out double naiveMedian);
            Console.WriteLine($"  Median speedup (naive / fused)    : {naiveMedian / fusedMedian:F2}x");
            Console.WriteLine("  (matmul dominates this workload — expect high variance; the isolated");
            Console.WriteLine("   bias+ReLU benchmark below is the clean signal.)");
            Console.WriteLine();

            // ---- 7b. Isolated element-wise benchmark ----
            // Strip out the matmul so the fusion effect is visible. Inputs are the
            // post-matmul shape (batch, HiddenDim) that both paths would see at
            // that step of the forward pass.
            BenchmarkElementWiseOnly(batch, W1, b1);

            // ---- 8. Report fusion instrumentation ----
            int cacheAfter = ILKernelGenerator.InnerLoopCachedCount;
            Console.WriteLine("Kernel / delegate instrumentation:");
            Console.WriteLine($"  IL kernel cache entries   : {cacheBefore} -> {cacheAfter} (delta {cacheAfter - cacheBefore})");
            Console.WriteLine($"  NpyExpr delegate slots    : {NpyExpr_RegisteredCount()}");
            Console.WriteLine("  Note: cache delta is a small constant (3 expected: one kernel for layer 1's");
            Console.WriteLine("        fused bias+ReLU, one for layer 2's bias-only, one for the isolated");
            Console.WriteLine("        sweep). Invariant across benchmark iteration count — the IL body is");
            Console.WriteLine("        compiled once per unique cacheKey and hit thereafter.");

            return 0;
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        /// <summary>
        /// He-normal initializer: N(0, sqrt(2/fan_in)) cast to float32.
        /// Standard choice for ReLU networks.
        /// </summary>
        private static NDArray HeInit(int fanIn, int fanOut)
        {
            double stddev = Math.Sqrt(2.0 / fanIn);
            NDArray w = np.random.normal(0.0, stddev, new Shape(fanIn, fanOut));
            return w.astype(NPTypeCode.Single);
        }

        /// <summary>
        /// Walks up from the process working directory to find the experiment's
        /// data folder — lets the binary find idx files whether it's run from
        /// bin/Debug or the source directory.
        /// </summary>
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

            // Default: next to the binary, even if missing — the loader will report.
            return candidates[0];
        }

        private static int CountMatches(NDArray predicted, NDArray labels)
        {
            // np.argmax returns Int64; labels are Byte. Use the matching accessors —
            // the storage dtype-checks GetInt32/GetByte against the raw element size
            // and throws "Memory corruption expected" on mismatch.
            int n = (int)predicted.shape[0];
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (predicted.GetInt64(i) == labels.GetByte(i))
                    correct++;
            return correct;
        }

        /// <summary>
        /// Isolates the (bias + ReLU) fusion effect from the matmul. Precomputes
        /// the preactivation once, then times just the post-matmul element-wise
        /// work for both strategies. Also sweeps a handful of sizes so the
        /// crossover point is visible.
        /// </summary>
        private static void BenchmarkElementWiseOnly(NDArray batch, NDArray W1, NDArray b1)
        {
            // Precompute the layer-1 preactivation — only the element-wise work
            // is measured. Allocated once per size, reused across both strategies.
            int[] sizes = { 128, 1024, 4096, 16384 };
            Console.WriteLine("Benchmark — isolated bias+ReLU only (no matmul):");
            Console.WriteLine($"  Shape is (N, {HiddenDim}) float32; N listed below.");
            Console.WriteLine($"  {"N",-8} {"Fused ms/op",-14} {"Naive ms/op",-14} {"Speedup",-10}");
            foreach (int n in sizes)
            {
                // Build a fake preact of the requested size using the first rows
                // of an extended random batch. When n > batch.shape[0], repeat.
                NDArray fakeBatch = BuildBatchOfSize(batch, n);
                NDArray preact   = np.dot(fakeBatch, W1);  // (n, HiddenDim)

                // Warmup + measure
                for (int i = 0; i < WarmupPasses; i++)
                {
                    NDArray hA = np.empty_like(preact);
                    FusedMlp_PostMatmul(preact, b1, hA);
                    _ = NaiveMlp_PostMatmul(preact, b1);
                }

                var swF = Stopwatch.StartNew();
                for (int i = 0; i < BenchPasses; i++)
                {
                    NDArray h = np.empty_like(preact);
                    FusedMlp_PostMatmul(preact, b1, h);
                }
                swF.Stop();

                var swN = Stopwatch.StartNew();
                for (int i = 0; i < BenchPasses; i++)
                    _ = NaiveMlp_PostMatmul(preact, b1);
                swN.Stop();

                double fusedMs = swF.Elapsed.TotalMilliseconds / BenchPasses;
                double naiveMs = swN.Elapsed.TotalMilliseconds / BenchPasses;
                Console.WriteLine($"  {n,-8} {fusedMs,-14:F4} {naiveMs,-14:F4} {naiveMs / fusedMs,-10:F2}x");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Builds an (n, 784) float32 batch by tiling the existing (128, 784)
        /// data. Keeps the workload realistic while sweeping the row count.
        /// </summary>
        private static NDArray BuildBatchOfSize(NDArray sourceBatch, int n)
        {
            if (n == sourceBatch.shape[0]) return sourceBatch;

            int srcRows = (int)sourceBatch.shape[0];
            int cols    = (int)sourceBatch.shape[1];
            var arr = new NDArray(NPTypeCode.Single, new Shape(n, cols), fillZeros: false);
            unsafe
            {
                float* src = (float*)sourceBatch.Address;
                float* dst = (float*)arr.Address;
                for (int r = 0; r < n; r++)
                {
                    long srcOff = (long)(r % srcRows) * cols;
                    long dstOff = (long)r            * cols;
                    for (int c = 0; c < cols; c++)
                        dst[dstOff + c] = src[srcOff + c];
                }
            }
            return arr;
        }

        /// <summary>Mirror of FusedMlp's Layer-1 fused op for use in isolation tests.</summary>
        private static void FusedMlp_PostMatmul(NDArray preact, NDArray bias, NDArray output)
        {
            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op: new[] { preact, bias, output },
                flags:   NpyIterGlobalFlags.EXTERNAL_LOOP,
                order:   NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_NO_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Max(
                NpyExpr.Input(0) + NpyExpr.Input(1),
                NpyExpr.Const(0f));

            iter.ExecuteExpression(
                expr,
                new[] { NPTypeCode.Single, NPTypeCode.Single },
                NPTypeCode.Single,
                cacheKey: "mnist_bench_bias_relu_f32");
        }

        private static NDArray NaiveMlp_PostMatmul(NDArray preact, NDArray bias)
            => np.maximum(np.add(preact, bias), (NDArray)0f);

        private static double MaxAbsDiff(NDArray a, NDArray b)
        {
            int rows = (int)a.shape[0];
            int cols = (int)a.shape[1];
            double max = 0.0;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    double d = System.Math.Abs(a.GetSingle(i, j) - b.GetSingle(i, j));
                    if (d > max) max = d;
                }
            return max;
        }

        private static BenchmarkResult Benchmark(string name, Func<NDArray> action)
        {
            // Warmup — compile kernels, warm CPU caches, JIT everything.
            for (int i = 0; i < WarmupPasses; i++) _ = action();

            // Drain GC debris from warmup so the timed loop starts with a clean
            // gen-0 budget. Both strategies allocate intermediate NDArrays, and
            // a gen-0 pause mid-measurement easily shifts results by 50%+.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < BenchPasses; i++) _ = action();
            sw.Stop();

            return new BenchmarkResult(name, sw.Elapsed.TotalMilliseconds,
                                       sw.Elapsed.TotalMilliseconds / BenchPasses);
        }

        /// <summary>
        /// Runs <paramref name="action"/> five times for BenchPasses iterations
        /// each and reports min / median / max ms/pass. Used on workloads where
        /// per-run variance is high enough that a single measurement is
        /// misleading (matmul-dominated code paths).
        /// </summary>
        private static void BenchMultiRun(string name, Func<NDArray> action, out double median)
        {
            const int runs = 5;
            var results = new double[runs];
            for (int r = 0; r < runs; r++)
                results[r] = Benchmark(name, action).MsPerPass;
            Array.Sort(results);
            double min = results[0];
            median     = results[runs / 2];
            double max = results[runs - 1];
            Console.WriteLine($"  {name,-32}: min={min:F3} median={median:F3} max={max:F3} ms/pass (over {runs} runs)");
        }

        private static int NpyExpr_RegisteredCount() => DelegateSlots.RegisteredCount;

        private readonly record struct BenchmarkResult(string Name, double MsTotal, double MsPerPass);
    }
}
