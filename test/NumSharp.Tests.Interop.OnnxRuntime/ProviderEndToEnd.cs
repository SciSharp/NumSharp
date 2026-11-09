using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     One execution provider end to end through the bridge: <c>matmul_const_f32</c> (X [N,4] times a constant W [4,3])
    ///     runs in a session with the provider appended, and its output is imported every way the bridge offers —
    ///     <c>AsNDArray</c> (a zero-copy lease), <c>ToNDArray</c> (a copy) and <c>session.Run(…, zeroCopyOutputs: true)</c> —
    ///     with the values checked against X @ W computed here.
    /// </summary>
    /// <remarks>
    ///     <para>No test framework in here: the desktop suite (<c>ProviderEndToEndTests</c>) and the Android instrumentation
    ///     host (<c>NumSharp.Tests.Interop.OnnxRuntime.Android</c>, which compiles this file) both call it. A failure throws
    ///     <see cref="InvalidOperationException"/> naming the provider and the route.</para>
    ///     <para>Each provider names the allocator its outputs must come from: a provider that silently left the MatMul to
    ///     the CPU provider would hand back <c>Cpu</c> memory and pass vacuously, so XNNPACK's run fails unless the output
    ///     is <c>XnnpackExecutionProvider</c> memory — the memory the bridge used to refuse.</para>
    /// </remarks>
    internal static class ProviderEndToEnd
    {
        /// <summary>
        ///     The CPU-side providers this probe can append, each with the allocator name its MatMul output lives in
        ///     (<c>null</c>: whichever the runtime chooses; NNAPI keeps no allocator of its own and may leave the node to
        ///     the CPU provider) and how to append it.
        /// </summary>
        /// <remarks>
        ///     XNNPACK gets ORT's documented arrangement: its own thread pool, and one intra-op thread for the session so
        ///     the two pools do not compete.
        /// </remarks>
        public static readonly IReadOnlyList<(string Provider, string Allocator, Action<SessionOptions> Append)> Known = new (string, string, Action<SessionOptions>)[]
        {
            ("CPUExecutionProvider", "Cpu", _ => { }),
            ("XnnpackExecutionProvider", "XnnpackExecutionProvider", options =>
            {
                options.IntraOpNumThreads = 1;
                options.AppendExecutionProvider("XNNPACK", new Dictionary<string, string> { ["intra_op_num_threads"] = "2" });
            }),
            ("NnapiExecutionProvider", null, options => options.AppendExecutionProvider_Nnapi()),
        };

        /// <summary>
        ///     The providers of <see cref="Known"/> the loaded ONNX Runtime reports as available.
        /// </summary>
        /// <returns>Their names, in <see cref="Known"/>'s order.</returns>
        public static string[] Available()
        {
            var available = new HashSet<string>(OrtEnv.Instance().GetAvailableProviders());
            return Known.Where(k => available.Contains(k.Provider)).Select(k => k.Provider).ToArray();
        }

        /// <summary>
        ///     GPU providers whose OWN memory the bridge must keep refusing: the provider, the allocator name to bind the
        ///     output to, the names ORT may report for that memory, and how to append the provider. ORT copies a graph
        ///     output back to the CPU unless an <see cref="OrtIoBinding"/> binds it to the device, so
        ///     <see cref="RunRefused"/> binds it there.
        /// </summary>
        /// <remarks>
        ///     WebGPU's buffer allocator is <c>WebGPU_Buf</c> since ONNX Runtime 1.25 (a 10-character name, for the
        ///     small-string buffer on the web) and was <c>WebGPU_Buffer</c> before; <c>CreateMemoryInfo</c> still accepts the
        ///     long name and normalizes it, so binding with it works on every version.
        /// </remarks>
        public static readonly IReadOnlyList<(string Provider, string BindName, string[] DeviceNames, Action<SessionOptions> Append)> DeviceMemoryProviders = new (string, string, string[], Action<SessionOptions>)[]
        {
            ("WebGpuExecutionProvider", "WebGPU_Buffer", new[] { "WebGPU_Buf", "WebGPU_Buffer" }, options => options.AppendExecutionProvider("WebGPU")),
        };

        /// <summary>
        ///     Runs <c>matmul_const_f32</c> through a GPU provider with its output bound to the provider's own device memory,
        ///     and checks that the bridge refuses to read it (<c>AsNDArray</c> and <c>ToNDArray</c>: an
        ///     <see cref="InvalidOperationException"/> naming the memory) instead of dereferencing a device pointer.
        /// </summary>
        /// <param name="model">The bytes of <c>matmul_const_f32.onnx</c>.</param>
        /// <param name="provider">One of <see cref="DeviceMemoryProviders"/>' provider names.</param>
        /// <returns>A one-line report: the provider, the memory the output was bound to, the refusals.</returns>
        /// <exception cref="ArgumentException"><paramref name="provider"/> is not in <see cref="DeviceMemoryProviders"/>.</exception>
        /// <exception cref="InvalidOperationException">The bridge read the device memory, or the output was not in it.</exception>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime could not append the provider, describe its memory or run.</exception>
        public static string RunRefused(byte[] model, string provider)
        {
            var known = DeviceMemoryProviders.FirstOrDefault(k => k.Provider == provider);
            if (known.Provider is null)
                throw new ArgumentException($"unknown device provider '{provider}'", nameof(provider));

            using var options = new SessionOptions();
            known.Append(options);
            using var session = new InferenceSession(model, options);
            using NDArray ints = np.arange(8);
            using NDArray flat = ints.astype(NPTypeCode.Single);
            using NDArray input = flat.reshape(2, 4);
            using OrtTensor feed = input.AsOrtValue();
            using var device = new OrtMemoryInfo(known.BindName, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
            using var binding = session.CreateIoBinding();
            using var runOptions = new RunOptions();
            binding.BindInput("X", feed.Value);
            binding.BindOutputToDevice("Y", device);
            session.RunWithBinding(runOptions, binding);
            using var outputs = binding.GetOutputValues();
            string allocator;
            using (OrtMemoryInfo info = outputs[0].GetTensorMemoryInfo())
                allocator = info.Name;
            if (!known.DeviceNames.Contains(allocator))
                throw new InvalidOperationException($"{provider}: the bound output is in '{allocator}' memory, not {string.Join(" / ", known.DeviceNames)}: nothing device-resident to refuse");
            foreach (var (route, import) in new (string, Func<NDArray>)[] { ("AsNDArray", () => outputs[0].AsNDArray()), ("ToNDArray", () => outputs[0].ToNDArray()) })
            {
                try
                {
                    using NDArray read = import();
                    throw new InvalidOperationException($"{provider} {route}: the bridge read '{allocator}' device memory instead of refusing it");
                }
                catch (InvalidOperationException e) when (e.Message.Contains("cannot address"))
                {
                    // The refusal: the bridge named the memory and did not touch it.
                }
            }
            return $"{provider}: output bound to '{allocator}' memory; AsNDArray and ToNDArray refuse it";
        }

        /// <summary>
        ///     Runs <c>matmul_const_f32</c> through <paramref name="provider"/> and imports its output every way, checking the
        ///     values and the output's allocator.
        /// </summary>
        /// <param name="model">The bytes of <c>matmul_const_f32.onnx</c>.</param>
        /// <param name="provider">One of <see cref="Known"/>'s provider names.</param>
        /// <returns>A one-line report: the provider, the allocator the output came from, the routes checked.</returns>
        /// <exception cref="ArgumentException"><paramref name="provider"/> is not in <see cref="Known"/>.</exception>
        /// <exception cref="InvalidOperationException">A route gave wrong values, the output came from another allocator
        /// than the provider's, or the bridge refused the output.</exception>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime could not append the provider or run the model.</exception>
        public static unsafe string Run(byte[] model, string provider)
        {
            var known = Known.FirstOrDefault(k => k.Provider == provider);
            if (known.Provider is null)
                throw new ArgumentException($"unknown provider '{provider}'", nameof(provider));

            const int rows = 5;
            var x = new float[rows * 4];
            for (int i = 0; i < x.Length; i++) x[i] = i * 0.5f - 3f;
            var expected = new float[rows * 3];
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < 3; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < 4; k++) sum += x[r * 4 + k] * (k * 3 + j + 1);   // W[k][j] = k*3 + j + 1
                    expected[r * 3 + j] = sum;
                }

            using var options = new SessionOptions();
            known.Append(options);
            using var session = new InferenceSession(model, options);
            using NDArray flat = np.array(x);
            using NDArray input = flat.reshape(rows, 4);

            // Route 1: ORT's own Run, then AsNDArray and ToNDArray over the very output tensor.
            string allocator;
            using (OrtTensor feed = input.AsOrtValue())
            using (var runOptions = new RunOptions())
            using (var outputs = session.Run(runOptions, new[] { "X" }, new[] { feed.Value }, new[] { "Y" }))
            {
                using (OrtMemoryInfo info = outputs[0].GetTensorMemoryInfo())
                    allocator = info.Name;
                if (known.Allocator is not null && allocator != known.Allocator)
                    throw new InvalidOperationException($"{provider}: the output came from '{allocator}' memory, not '{known.Allocator}': the provider did not run the MatMul, so this run proves nothing about its memory");
                using (NDArray view = outputs[0].AsNDArray())
                    Compare(provider, "AsNDArray", view, expected, rows);
                using (NDArray copy = outputs[0].ToNDArray())
                    Compare(provider, "ToNDArray", copy, expected, rows);
            }

            // Route 2: the bridge's own Run with zero-copy outputs.
            var results = session.Run(new Dictionary<string, NDArray> { ["X"] = input }, zeroCopyOutputs: true);
            try
            {
                Compare(provider, "Run(zeroCopyOutputs)", results["Y"], expected, rows);
            }
            finally
            {
                foreach (var result in results.Values)
                    result.Dispose();
            }

            if (NDArrayOnnxInterop.LiveImports != 0)
                throw new InvalidOperationException($"{provider}: {NDArrayOnnxInterop.LiveImports} import lease(s) still live after every view was disposed");
            return $"{provider}: output in '{allocator}' memory; AsNDArray, ToNDArray and Run(zeroCopyOutputs) equal X @ W";
        }

        /// <summary>
        ///     Checks an imported output against the expected product, exactly (the products of these small
        ///     binary fractions are exact in float32 whatever the summation order).
        /// </summary>
        /// <param name="provider">The provider, for the message.</param>
        /// <param name="route">The import route, for the message.</param>
        /// <param name="actual">The imported output.</param>
        /// <param name="expected">X @ W, row-major.</param>
        /// <param name="rows">The number of rows.</param>
        /// <exception cref="InvalidOperationException">The shape or any value differs.</exception>
        private static void Compare(string provider, string route, NDArray actual, float[] expected, int rows)
        {
            if (actual.ndim != 2 || actual.shape[0] != rows || actual.shape[1] != 3)
                throw new InvalidOperationException($"{provider} {route}: shape ({string.Join(", ", actual.shape)}), expected ({rows}, 3)");
            float[] values = actual.ToArray<float>();
            for (int i = 0; i < expected.Length; i++)
                if (values[i] != expected[i])
                    throw new InvalidOperationException($"{provider} {route}: element {i} is {values[i]:R}, expected {expected[i]:R}");
        }
    }
}
