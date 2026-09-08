using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     The ergonomic tier: run an <see cref="InferenceSession"/> straight on <see cref="NDArray"/>s. Inputs are
    ///     fed <b>zero-copy</b> (<see cref="NDArrayOnnxInterop.AsOrtValue"/>), outputs come back as owning
    ///     NumSharp arrays (<see cref="NDArrayOnnxInterop.ToNDArray(OrtValue)"/> — or, on request, as zero-copy
    ///     views that own their ORT buffer), and every <see cref="OrtValue"/> / <see cref="OrtTensor"/> lifetime is
    ///     handled inside the call — no ORT plumbing at the call site.
    ///
    ///     <para>Each call reads the session's metadata to stay safe and terse: with a single model input /
    ///     output the names default (<see cref="Run(InferenceSession, NDArray, RunOptions, string)"/>); an input
    ///     whose dtype differs from the model's declared dtype is <b>cast to it</b> when the cast is allowed under
    ///     <paramref name="casting"/> — NumPy's rule words, default <c>"safe"</c>, so an int32 array into an
    ///     int64 BERT input or a float32 array into a float64 input just works, while a lossy float64 → float16
    ///     needs an explicit <c>casting: "same_kind"</c> (or an <c>astype</c> at the call site); an input whose
    ///     declared rank or fixed extents disagree with the array is rejected with a message naming the input,
    ///     before ORT ever sees it.</para>
    /// </summary>
    public static class InferenceSessionExtensions
    {
        /// <summary>
        ///     Run a single-input, single-output model on one array: the input and output names come from the
        ///     session's metadata. Returns a fresh owning <see cref="NDArray"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">The model has more than one input or output (use the named overloads).</exception>
        public static NDArray Run(this InferenceSession session, NDArray input, RunOptions runOptions = null, string casting = "safe")
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            string inputName = SingleName(session.InputNames, "input");
            string outputName = SingleName(session.OutputNames, "output");
            return Run(session, inputName, input, outputName, runOptions, casting);
        }

        /// <summary>Run with one named input and one named output. Returns a fresh owning <see cref="NDArray"/>.</summary>
        public static NDArray Run(this InferenceSession session, string inputName, NDArray input, string outputName, RunOptions runOptions = null, string casting = "safe")
        {
            if (inputName is null)
                throw new ArgumentNullException(nameof(inputName));
            if (outputName is null)
                throw new ArgumentNullException(nameof(outputName));
            var results = Run(session, new Dictionary<string, NDArray>(1) { { inputName, input } }, new[] { outputName }, runOptions, casting);
            return results[outputName];
        }

        /// <summary>
        ///     Run with named inputs; returns the requested outputs (all of the model's outputs when
        ///     <paramref name="outputNames"/> is <c>null</c>) keyed by name.
        /// </summary>
        /// <param name="session">The session.</param>
        /// <param name="inputs">Model input name → array. Each array is fed zero-copy when C-contiguous and of the declared dtype; otherwise a cast / C-order copy is made for the duration of the run.</param>
        /// <param name="outputNames">Outputs to fetch; <c>null</c> = <see cref="InferenceSession.OutputNames"/>.</param>
        /// <param name="runOptions">Optional run options (a default instance is created when <c>null</c>).</param>
        /// <param name="casting">NumPy casting rule for dtype coercion of inputs: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c> (default), <c>"same_kind"</c>, <c>"unsafe"</c>.</param>
        /// <param name="zeroCopyOutputs">
        ///     <c>false</c> (default): each output is COPIED into an owning NDArray and the ORT values are disposed
        ///     before returning. <c>true</c>: each output is a zero-copy VIEW over ORT's buffer that OWNS its
        ///     OrtValue — ORT's memory is freed when the array (and every slice derived from it) is disposed or
        ///     collected; use it for large outputs you consume in place.
        /// </param>
        public static IReadOnlyDictionary<string, NDArray> Run(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyCollection<string> outputNames = null, RunOptions runOptions = null, string casting = "safe", bool zeroCopyOutputs = false)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            outputNames ??= session.OutputNames;

            var names = new string[inputs.Count];
            var handles = new OrtTensor[inputs.Count];
            var values = new OrtValue[inputs.Count];
            try
            {
                int i = 0;
                foreach (KeyValuePair<string, NDArray> input in inputs)
                {
                    handles[i] = PrepareInput(session, input.Key, input.Value, casting);
                    names[i] = input.Key;
                    values[i] = handles[i].Value;
                    i++;
                }

                using RunOptions owned = runOptions is null ? new RunOptions() : null;
                IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions ?? owned, names, values, outputNames);
                return zeroCopyOutputs ? ViewOutputs(outputs, outputNames) : CopyOutputs(outputs, outputNames);
            }
            finally
            {
                foreach (OrtTensor handle in handles)
                    handle?.Dispose();
            }
        }

        /// <summary>
        ///     Run into PRE-ALLOCATED outputs: each output array is handed to ORT zero-copy and ORT writes the
        ///     result straight into NumSharp's memory — no output allocation, no copy. Zero-copy in both
        ///     directions, the idiom for a hot inference loop that reuses its buffers. An output array must be
        ///     C-contiguous, writeable, of the model's declared output dtype (outputs are written in place and
        ///     cannot be cast) and of the exact shape the run produces (ORT itself checks the shape).
        /// </summary>
        public static void Run(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs, IReadOnlyDictionary<string, NDArray> outputs,
            RunOptions runOptions = null, string casting = "safe")
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            if (outputs is null)
                throw new ArgumentNullException(nameof(outputs));

            var inNames = new string[inputs.Count];
            var inHandles = new OrtTensor[inputs.Count];
            var inValues = new OrtValue[inputs.Count];
            var outNames = new string[outputs.Count];
            var outHandles = new OrtTensor[outputs.Count];
            var outValues = new OrtValue[outputs.Count];
            try
            {
                int i = 0;
                foreach (KeyValuePair<string, NDArray> input in inputs)
                {
                    inHandles[i] = PrepareInput(session, input.Key, input.Value, casting);
                    inNames[i] = input.Key;
                    inValues[i] = inHandles[i].Value;
                    i++;
                }

                int j = 0;
                foreach (KeyValuePair<string, NDArray> output in outputs)
                {
                    outHandles[j] = PrepareOutput(session, output.Key, output.Value);
                    outNames[j] = output.Key;
                    outValues[j] = outHandles[j].Value;
                    j++;
                }

                using RunOptions owned = runOptions is null ? new RunOptions() : null;
                session.Run(runOptions ?? owned, inNames, inValues, outNames, outValues);
            }
            finally
            {
                foreach (OrtTensor handle in inHandles)
                    handle?.Dispose();
                foreach (OrtTensor handle in outHandles)
                    handle?.Dispose();
            }
        }

        // ---- output collection ------------------------------------------------------------------------

        /// <summary>Copy every output into an owning NDArray, then dispose the ORT values (the collection owns them).</summary>
        private static IReadOnlyDictionary<string, NDArray> CopyOutputs(IDisposableReadOnlyCollection<OrtValue> outputs, IReadOnlyCollection<string> outputNames)
        {
            var result = new Dictionary<string, NDArray>(outputs.Count);
            try
            {
                using (outputs)
                {
                    int j = 0;
                    foreach (string name in outputNames)
                        result[name] = outputs[j++].ToNDArray();
                }

                return result;
            }
            catch
            {
                foreach (NDArray made in result.Values)
                    made.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Wrap every output as a zero-copy view that OWNS its OrtValue. The collection wrapper ORT
        ///     returned is deliberately NOT disposed — that would free the buffers under the views; it is a
        ///     plain list with no finalizer, and each OrtValue is now released by the view's lease instead.
        /// </summary>
        private static IReadOnlyDictionary<string, NDArray> ViewOutputs(IDisposableReadOnlyCollection<OrtValue> outputs, IReadOnlyCollection<string> outputNames)
        {
            var result = new Dictionary<string, NDArray>(outputs.Count);
            int j = 0;
            try
            {
                foreach (string name in outputNames)
                {
                    result[name] = outputs[j].AsNDArray(ownsValue: true);
                    j++;
                }

                return result;
            }
            catch
            {
                // views already made own their values and release them; the rest are still ours to free
                foreach (NDArray made in result.Values)
                    made.Dispose();
                for (int k = j; k < outputs.Count; k++)
                    outputs[k].Dispose();
                throw;
            }
        }

        // ---- input / output preparation ----------------------------------------------------------

        /// <summary>
        ///     Resolve one model input to a zero-copy <see cref="OrtTensor"/>: look the name up, coerce the dtype
        ///     to the declared one under <paramref name="casting"/>, validate the declared shape, and materialize
        ///     a C-order copy only when the array is not C-contiguous. Any temporary this creates is owned by the
        ///     returned handle and disposed with it.
        /// </summary>
        internal static OrtTensor PrepareInput(InferenceSession session, string name, NDArray array, string casting)
        {
            if (name is null)
                throw new ArgumentNullException(nameof(name));
            if (array is null)
                throw new ArgumentNullException(nameof(array), $"input '{name}' is null.");
            if (!session.InputMetadata.TryGetValue(name, out NodeMetadata meta))
                throw new ArgumentException($"the model has no input named '{name}'; its inputs are [{string.Join(", ", session.InputNames)}].", nameof(name));
            if (!meta.IsTensor)
                throw new NotSupportedException($"input '{name}' is {meta.OnnxValueType}, not a tensor; feed it with the OrtValue API (OrtValue.CreateSequence / CreateMap).");
            if (!NDArrayOnnxInterop.TryFromTensorElementType(meta.ElementDataType, out NPTypeCode want))
                throw new NotSupportedException($"input '{name}' is declared {meta.ElementDataType}: {NDArrayOnnxInterop.UnsupportedImportMessage(meta.ElementDataType)}");

            NDArray fed = array;
            bool owned = false;
            try
            {
                NPTypeCode have = array.typecode;
                // Two conversions the dtype map itself performs — no cast decision needed here: Char crosses as
                // UInt16 (UTF-16 code units), Decimal as a float64 temporary the handle owns (AsOrtValue's rule).
                bool matches = have == want
                               || (have == NPTypeCode.Char && want == NPTypeCode.UInt16)
                               || (have == NPTypeCode.Decimal && want == NPTypeCode.Double);
                if (!matches)
                {
                    if (!np.can_cast(have, want, casting))
                        throw new InvalidCastException(
                            $"input '{name}' holds {have} but the model declares {want} ({meta.ElementDataType}); the cast is not allowed under " +
                            $"casting='{casting}'. Convert explicitly — nd.astype(NPTypeCode.{want}) — or relax the rule (casting: \"same_kind\" / \"unsafe\").");
                    fed = array.astype(want);
                    owned = true;
                }

                ValidateShape(name, "input", meta, fed.Shape);

                if (!fed.Shape.IsContiguous)
                {
                    // AsOrtValue shares only C-contiguous memory; a view is copied in logical order for the run.
                    NDArray dense = fed.copy();
                    if (owned)
                        fed.Dispose();
                    fed = dense;
                    owned = true;
                }

                OrtTensor handle = NDArrayOnnxInterop.AsOrtValueCore(fed, ownsSource: owned);
                owned = false;   // the handle owns the temporary now
                return handle;
            }
            finally
            {
                if (owned)
                    fed.Dispose();
            }
        }

        /// <summary>
        ///     Resolve one PRE-ALLOCATED model output to a zero-copy <see cref="OrtTensor"/> ORT writes into:
        ///     the array must be writeable, C-contiguous and of the declared output dtype (no cast is possible —
        ///     ORT writes the bytes in place); a declared fixed shape is checked up front.
        /// </summary>
        internal static OrtTensor PrepareOutput(InferenceSession session, string name, NDArray array)
        {
            if (name is null)
                throw new ArgumentNullException(nameof(name));
            if (array is null)
                throw new ArgumentNullException(nameof(array), $"output '{name}' is null.");
            if (!session.OutputMetadata.TryGetValue(name, out NodeMetadata meta))
                throw new ArgumentException($"the model has no output named '{name}'; its outputs are [{string.Join(", ", session.OutputNames)}].", nameof(name));
            if (!meta.IsTensor)
                throw new NotSupportedException($"output '{name}' is {meta.OnnxValueType}, not a tensor; it cannot be written into an NDArray.");
            if (!NDArrayOnnxInterop.TryToTensorElementType(array.typecode, out TensorElementType have) || have != meta.ElementDataType)
                throw new InvalidOperationException(
                    $"output '{name}' is declared {meta.ElementDataType} but the pre-allocated array is {array.typecode}; outputs are written in place " +
                    "and cannot be cast — allocate the array with the model's dtype.");
            if (!array.Shape.IsWriteable)
                throw new InvalidOperationException($"output '{name}': the pre-allocated array is read-only (a broadcast or read-only view); ORT must be able to write it.");
            if (!array.Shape.IsContiguous)
                throw new InvalidOperationException($"output '{name}': {NDArrayOnnxInterop.NotCContiguousMessage}");
            ValidateShape(name, "output", meta, array.Shape);
            return NDArrayOnnxInterop.AsOrtValueCore(array, ownsSource: false);
        }

        /// <summary>
        ///     Compare an array's shape with the model's declared shape: a declared rank must match, and every
        ///     FIXED extent must match; symbolic dims (<c>-1</c>, e.g. <c>'batch'</c>) are free. A declaration
        ///     without dimensions is either a scalar or an unknown-rank tensor — the two are indistinguishable in
        ///     <see cref="NodeMetadata"/> — so it is left to ORT's own check at run time.
        /// </summary>
        internal static void ValidateShape(string name, string role, NodeMetadata meta, Shape shape)
        {
            int[] declared = meta.Dimensions;
            if (declared is null || declared.Length == 0)
                return;
            if (declared.Length != shape.NDim)
                throw new ArgumentException($"{role} '{name}' expects rank {declared.Length} {DescribeDims(meta)}, but the array has shape {shape} (rank {shape.NDim}).");
            long[] dims = shape.Dimensions;
            for (int i = 0; i < declared.Length; i++)
            {
                if (declared[i] >= 0 && declared[i] != dims[i])
                    throw new ArgumentException($"{role} '{name}' expects shape {DescribeDims(meta)}, but the array has shape {shape}: dimension {i} must be {declared[i]}, not {dims[i]}.");
            }
        }

        /// <summary>The declared shape as NumPy-ish text, symbolic dims by name: <c>(1, 3, 'H', 'W')</c>.</summary>
        internal static string DescribeDims(NodeMetadata meta)
        {
            int[] dims = meta.Dimensions;
            string[] symbolic = meta.SymbolicDimensions;
            var sb = new StringBuilder("(");
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                if (dims[i] >= 0)
                    sb.Append(dims[i]);
                else
                    sb.Append('\'').Append(symbolic is not null && i < symbolic.Length && !string.IsNullOrEmpty(symbolic[i]) ? symbolic[i] : "?").Append('\'');
            }
            return sb.Append(')').ToString();
        }

        private static string SingleName(IReadOnlyList<string> names, string role)
        {
            if (names.Count == 1)
                return names[0];
            throw new InvalidOperationException(
                $"the model has {names.Count} {role}s ([{string.Join(", ", names)}]); name them explicitly — Run(session, inputName, input, outputName) " +
                "or Run(session, inputs: Dictionary<string, NDArray>, outputNames).");
        }
    }
}
