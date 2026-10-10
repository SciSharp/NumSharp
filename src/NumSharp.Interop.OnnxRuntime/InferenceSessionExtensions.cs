using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends.Iteration;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     The ergonomic tier: run an <see cref="InferenceSession"/> straight on <see cref="NDArray"/>s. Inputs are
    ///     fed <b>zero-copy</b> when ONNX Runtime can read them in place (<see cref="NDArrayOnnxInterop.AsOrtValue"/>),
    ///     outputs come back as owning NumSharp arrays (<see cref="NDArrayOnnxInterop.ToNDArray(OrtValue)"/> — or, on
    ///     request, as zero-copy views that own their ORT buffer, or written into arrays the caller pre-allocated),
    ///     and every <see cref="OrtValue"/> / <see cref="OrtTensor"/> lifetime is handled inside the call — no ORT
    ///     plumbing at the call site.
    ///
    ///     <para><b>What each call checks, from the session's metadata.</b> With a single model input / output the
    ///     names default (<see cref="Run(InferenceSession, NDArray, RunOptions, string, OnnxCodecMode)"/>). An input
    ///     name is looked up among the model's inputs and then among its overridable initializers. An input declared
    ///     <c>optional(tensor)</c> takes a plain array. An input whose dtype differs from the declared one is <b>cast
    ///     to it</b> when the cast is allowed under <c>casting</c> — NumPy's rule words, default <c>"safe"</c>, so an
    ///     int32 array into an int64 BERT input or a float32 array into a float64 input just works, while a lossy
    ///     float64 → float16 or a Decimal → float64 needs an explicit <c>casting: "same_kind"</c> (or an
    ///     <c>astype</c> at the call site). An input whose declared rank or fixed extents disagree with the array is
    ///     rejected with a message naming the input, before ORT ever sees it.</para>
    ///
    ///     <para><b>The input mode.</b> Every form takes an <see cref="OnnxCodecMode"/> for its INPUTS.
    ///     <see cref="OnnxCodecMode.Auto"/> (the default) shares an input that is C-contiguous and of the declared
    ///     dtype and feeds a temporary copy for any other (a cast, a C-order copy of a view).
    ///     <see cref="OnnxCodecMode.View"/> throws for an input that would need such a copy, so a hot loop can prove
    ///     that it copies nothing. <see cref="OnnxCodecMode.Copy"/> feeds a copy of every input.</para>
    /// </summary>
    public static class InferenceSessionExtensions
    {
        /// <summary>
        ///     Run a single-input, single-output model on one array: the input and output names come from the
        ///     session's metadata. Returns a fresh owning <see cref="NDArray"/>.
        /// </summary>
        /// <param name="session">The session; its model must declare exactly one input and one output.</param>
        /// <param name="input">The array to feed; any layout, and any dtype that casts to the declared one under <paramref name="casting"/>.</param>
        /// <param name="runOptions">Optional run options (a default instance is created for the call when <c>null</c>).</param>
        /// <param name="casting">NumPy casting rule for the input's dtype coercion: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c> (default), <c>"same_kind"</c>, <c>"unsafe"</c>.</param>
        /// <param name="mode">How the input is fed (view, copy, or view-when-possible); see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A new owning C-contiguous array holding the model's output.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="session"/> or <paramref name="input"/> is null.</exception>
        /// <exception cref="InvalidOperationException">The model has more than one input or output (use the named overloads), or <paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the input needs a copy.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">The input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="ArgumentException">The input's shape contradicts the declared one.</exception>
        /// <exception cref="NotSupportedException">The input or the output is not a tensor of a dtype NumSharp has.</exception>
        /// <exception cref="ObjectDisposedException">The input's buffer has already been released.</exception>
        public static NDArray Run(this InferenceSession session, NDArray input, RunOptions runOptions = null, string casting = "safe",
            OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            string inputName = SingleName(session.InputNames, "input");
            string outputName = SingleName(session.OutputNames, "output");
            return Run(session, inputName, input, outputName, runOptions, casting, mode);
        }

        /// <summary>Run with one named input and one named output. Returns a fresh owning <see cref="NDArray"/>.</summary>
        /// <param name="session">The session.</param>
        /// <param name="inputName">The model input (or overridable initializer) to feed.</param>
        /// <param name="input">The array to feed; any layout, and any dtype that casts to the declared one under <paramref name="casting"/>.</param>
        /// <param name="outputName">The model output to fetch.</param>
        /// <param name="runOptions">Optional run options (a default instance is created for the call when <c>null</c>).</param>
        /// <param name="casting">NumPy casting rule for the input's dtype coercion (default <c>"safe"</c>).</param>
        /// <param name="mode">How the input is fed; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A new owning C-contiguous array holding the named output.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="session"/>, <paramref name="inputName"/>, <paramref name="input"/> or <paramref name="outputName"/> is null.</exception>
        /// <exception cref="ArgumentException">The model has no such input, or the input's shape contradicts the declared one.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">The input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the input needs a copy.</exception>
        /// <exception cref="NotSupportedException">The input or the output is not a tensor of a dtype NumSharp has.</exception>
        /// <exception cref="ObjectDisposedException">The input's buffer has already been released.</exception>
        public static NDArray Run(this InferenceSession session, string inputName, NDArray input, string outputName, RunOptions runOptions = null,
            string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (inputName is null)
                throw new ArgumentNullException(nameof(inputName));
            if (outputName is null)
                throw new ArgumentNullException(nameof(outputName));
            var results = Run(session, new Dictionary<string, NDArray>(1) { { inputName, input } }, new[] { outputName }, runOptions, casting, zeroCopyOutputs: false, mode);
            return results[outputName];
        }

        /// <summary>
        ///     Run with named inputs; returns the requested outputs (all of the model's outputs when
        ///     <paramref name="outputNames"/> is <c>null</c>) keyed by name.
        /// </summary>
        /// <param name="session">The session.</param>
        /// <param name="inputs">Model input name → array. Each array is fed zero-copy when it is C-contiguous and of the declared dtype; otherwise a cast / C-order copy is made for the duration of the run (or refused, under <see cref="OnnxCodecMode.View"/>).</param>
        /// <param name="outputNames">Outputs to fetch; <c>null</c> = <see cref="InferenceSession.OutputNames"/>.</param>
        /// <param name="runOptions">Optional run options (a default instance is created when <c>null</c>).</param>
        /// <param name="casting">NumPy casting rule for dtype coercion of inputs: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c> (default), <c>"same_kind"</c>, <c>"unsafe"</c>. Validated on every call, also when no input needs a cast.</param>
        /// <param name="zeroCopyOutputs">
        ///     <c>false</c> (default): each output is COPIED into an owning NDArray and the ORT values are disposed
        ///     before returning. <c>true</c>: each output is a zero-copy VIEW over ORT's buffer that OWNS its
        ///     OrtValue — ORT's memory is freed when the array (and every slice derived from it) is disposed or
        ///     collected; use it for large outputs you consume in place.
        /// </param>
        /// <param name="mode">How the INPUTS are fed (view, copy, or view-when-possible); see <see cref="OnnxCodecMode"/>. It does not affect the outputs, which <paramref name="zeroCopyOutputs"/> decides.</param>
        /// <returns>The fetched outputs by name; the caller disposes each array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="session"/> or <paramref name="inputs"/> is null, or an input name or array in it is null.</exception>
        /// <exception cref="ArgumentException">The model has no input of a given name, or an input's shape contradicts the declared one.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">An input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and an input needs a copy; or an output lives in device memory.</exception>
        /// <exception cref="NotSupportedException">An input or an output is not a tensor of a dtype NumSharp has.</exception>
        /// <exception cref="ObjectDisposedException">An input's buffer has already been released.</exception>
        public static IReadOnlyDictionary<string, NDArray> Run(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyCollection<string> outputNames = null, RunOptions runOptions = null, string casting = "safe", bool zeroCopyOutputs = false,
            OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            // The two arguments that do not depend on the inputs are validated before any input is touched, and on
            // EVERY call: the rule name used to be checked only when some input needed a cast, so a misspelled rule
            // went unnoticed until the first input of another dtype.
            NPY_CASTING rule = DTypeCasting.ParseCasting(casting);
            NDArrayOnnxInterop.EnsureDefined(mode);
            outputNames ??= session.OutputNames;

            var names = new string[inputs.Count];
            var handles = new OrtTensor[inputs.Count];
            var values = new OrtValue[inputs.Count];
            try
            {
                int i = 0;
                foreach (KeyValuePair<string, NDArray> input in inputs)
                {
                    handles[i] = PrepareInput(session, input.Key, input.Value, rule, casting, mode);
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
        /// <remarks>
        ///     <para><b>Overlap.</b> ORT may write an output while it still reads an input, so an output that shares memory
        ///     with an input ORT reads in place gives wrong values (a 64×64 MatMul written into its own input: 3,072 of
        ///     4,096 elements wrong). The call compares the memory of every output with the memory of every fed input:
        ///     under <see cref="OnnxCodecMode.Auto"/> and <see cref="OnnxCodecMode.Copy"/> an overlapped input is fed
        ///     as a copy, so the result is the one computed from the inputs' values before the run (NumPy's rule for
        ///     <c>out=</c>); under <see cref="OnnxCodecMode.View"/> the call throws instead of copying. Two outputs
        ///     that overlap each other are refused in every mode: no copy can make that well defined.</para>
        ///     <para>It was an overload of <c>Run</c>. It has its own name because <c>session.Run(inputs, null)</c> was
        ///     ambiguous between the two dictionary forms (CS0121), and because it returns nothing.</para>
        /// </remarks>
        /// <param name="session">The session.</param>
        /// <param name="inputs">Model input name → array, fed as in <see cref="Run(InferenceSession, IReadOnlyDictionary{string, NDArray}, IReadOnlyCollection{string}, RunOptions, string, bool, OnnxCodecMode)"/>.</param>
        /// <param name="outputs">Model output name → the array ORT writes it into.</param>
        /// <param name="runOptions">Optional run options (a default instance is created when <c>null</c>).</param>
        /// <param name="casting">NumPy casting rule for dtype coercion of INPUTS (default <c>"safe"</c>); outputs are never cast.</param>
        /// <param name="mode">How the INPUTS are fed; see <see cref="OnnxCodecMode"/>. The outputs are always shared.</param>
        /// <exception cref="ArgumentNullException"><paramref name="session"/>, <paramref name="inputs"/> or <paramref name="outputs"/> is null, or a name or array in them is null.</exception>
        /// <exception cref="ArgumentException">The model has no input or output of a given name; a shape contradicts the declared one; or two outputs overlap in memory.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">An input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="InvalidOperationException">
        ///     An output array is read-only, not C-contiguous or not of the declared dtype; or <paramref name="mode"/> is
        ///     <see cref="OnnxCodecMode.View"/> and an input needs a copy (a cast, a layout, or an overlap with an output).
        /// </exception>
        /// <exception cref="NotSupportedException">An input or an output is not a tensor of a dtype NumSharp has.</exception>
        /// <exception cref="ObjectDisposedException">An input's or an output's buffer has already been released.</exception>
        public static void RunInto(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs, IReadOnlyDictionary<string, NDArray> outputs,
            RunOptions runOptions = null, string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            if (outputs is null)
                throw new ArgumentNullException(nameof(outputs));
            NPY_CASTING rule = DTypeCasting.ParseCasting(casting);
            NDArrayOnnxInterop.EnsureDefined(mode);

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
                    inHandles[i] = PrepareInput(session, input.Key, input.Value, rule, casting, mode);
                    inNames[i] = input.Key;
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

                // Decided on the memory ORT will really read and write (the handles' ranges), so an input that is
                // already fed as a temporary copy never counts as an overlap.
                SeparateInputsFromOutputs(inNames, inHandles, outNames, outHandles, mode);
                for (int k = 0; k < inHandles.Length; k++)
                    inValues[k] = inHandles[k].Value;

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

        /// <summary>
        ///     Makes a pre-allocated-output run well defined when arrays share memory: refuses two outputs that overlap, and
        ///     replaces every input ORT would read in place from memory an output is written to with a copy of that input
        ///     (or refuses it, under <see cref="OnnxCodecMode.View"/>).
        /// </summary>
        /// <remarks>
        ///     Every range here is one contiguous run of bytes (an output is C-contiguous, and an input is fed either from
        ///     its own C-contiguous memory or from a temporary), so an interval test is exact: it reports no false overlap
        ///     for views that interleave. An empty tensor occupies no byte and overlaps nothing.
        /// </remarks>
        /// <param name="inNames">The input names, for messages.</param>
        /// <param name="inHandles">The prepared inputs; an overlapped one is disposed and replaced in place by a handle over a copy.</param>
        /// <param name="outNames">The output names, for messages.</param>
        /// <param name="outHandles">The prepared outputs (always views of the caller's arrays).</param>
        /// <param name="mode">The input mode of the run.</param>
        /// <exception cref="ArgumentException">Two outputs overlap in memory.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and an input overlaps an output.</exception>
        /// <exception cref="ObjectDisposedException">An overlapped input's buffer was released before it could be copied.</exception>
        private static void SeparateInputsFromOutputs(string[] inNames, OrtTensor[] inHandles, string[] outNames, OrtTensor[] outHandles, OnnxCodecMode mode)
        {
            for (int j = 0; j < outHandles.Length; j++)
            {
                for (int k = 0; k < j; k++)
                {
                    if (Overlap(outHandles[j], outHandles[k]))
                        throw new ArgumentException(
                            $"outputs '{outNames[k]}' and '{outNames[j]}' overlap in memory: ONNX Runtime would write both results into the same bytes. " +
                            "Give each output its own array.");
                }

                for (int i = 0; i < inHandles.Length; i++)
                {
                    if (!Overlap(inHandles[i], outHandles[j]))
                        continue;
                    if (mode == OnnxCodecMode.View)
                        throw new InvalidOperationException(
                            $"RunInto(mode: OnnxCodecMode.View): input '{inNames[i]}' and output '{outNames[j]}' overlap in memory, and ONNX Runtime may write the " +
                            "output while it still reads the input. Pass OnnxCodecMode.Auto (the input is then fed as a copy) or give the output its own array.");

                    // The handle reads the caller's own memory (a temporary cannot overlap a caller's array). Feed a copy
                    // of that array instead; the new handle owns the copy. The old handle is disposed only once the copy
                    // exists, so a failure leaves every slot holding a handle the caller's finally can dispose.
                    OrtTensor copy = NDArrayOnnxInterop.AsOrtValueCore(inHandles[i].Source, ownsSource: false, OnnxCodecMode.Copy);
                    inHandles[i].Dispose();
                    inHandles[i] = copy;
                }
            }
        }

        /// <summary>Do the byte ranges two tensors occupy intersect? An empty tensor never does.</summary>
        /// <param name="a">One tensor.</param>
        /// <param name="b">The other tensor.</param>
        /// <returns><c>true</c> when at least one byte belongs to both.</returns>
        private static unsafe bool Overlap(OrtTensor a, OrtTensor b)
        {
            if (a.ByteLength == 0 || b.ByteLength == 0)
                return false;
            byte* aStart = a.Data, bStart = b.Data;
            return aStart < bStart + b.ByteLength && bStart < aStart + a.ByteLength;
        }

        // ---- output collection ------------------------------------------------------------------------

        /// <summary>Copy every output into an owning NDArray, then dispose the ORT values (the collection owns them).</summary>
        /// <param name="outputs">The run's outputs, in the order of <paramref name="outputNames"/>; disposed here.</param>
        /// <param name="outputNames">The fetched names.</param>
        /// <returns>The copies by name. A name fetched more than once keeps its last copy.</returns>
        /// <exception cref="NotSupportedException">An output is not a dense tensor of a dtype NumSharp has (the copies already made are disposed).</exception>
        /// <exception cref="InvalidOperationException">An output lives in device memory the CPU cannot address.</exception>
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
        /// <param name="outputs">The run's outputs, in the order of <paramref name="outputNames"/>.</param>
        /// <param name="outputNames">The fetched names.</param>
        /// <returns>The owning views by name.</returns>
        /// <exception cref="NotSupportedException">An output is not a dense tensor of a dtype NumSharp has (the views already made and the remaining values are disposed).</exception>
        /// <exception cref="InvalidOperationException">An output lives in device memory the CPU cannot address.</exception>
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
        ///     The declaration a fed array is checked against: the model input of that name or, failing that, the
        ///     overridable initializer of that name (an initializer the model also lists as a graph input, so a caller may
        ///     replace its value for a run); an <c>optional(tensor)</c> declaration is unwrapped to its tensor declaration.
        /// </summary>
        /// <remarks>
        ///     Both used to be refused: an overridable initializer with "the model has no input named …" (only
        ///     <see cref="InferenceSession.InputMetadata"/> was searched) and an optional input with a hint to use the
        ///     sequence / map factories, which do not make an optional. ONNX Runtime takes a plain tensor for both.
        /// </remarks>
        /// <param name="session">The session.</param>
        /// <param name="name">The name the caller feeds.</param>
        /// <returns>The tensor declaration to validate against (dtype, rank, extents), or a non-tensor declaration for the caller to refuse.</returns>
        /// <exception cref="ArgumentException">The model has neither an input nor an overridable initializer named <paramref name="name"/>.</exception>
        internal static NodeMetadata InputDeclaration(InferenceSession session, string name)
        {
            if (!session.InputMetadata.TryGetValue(name, out NodeMetadata meta) &&
                !session.OverridableInitializerMetadata.TryGetValue(name, out meta))
            {
                string overridable = session.OverridableInitializerMetadata.Count == 0
                    ? ""
                    : $" and its overridable initializers are [{string.Join(", ", session.OverridableInitializerMetadata.Keys)}]";
                throw new ArgumentException($"the model has no input named '{name}'; its inputs are [{string.Join(", ", session.InputNames)}]{overridable}.", nameof(name));
            }

            // optional(tensor): ONNX Runtime accepts the tensor itself where an optional is declared, so the element
            // declaration is the one that describes what to feed.
            if (meta.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
                meta = meta.AsOptionalMetadata().ElementMeta;
            return meta;
        }

        /// <summary>
        ///     Resolve one model input to an <see cref="OrtTensor"/> ONNX Runtime reads: look the name up, coerce the
        ///     dtype to the declared one under the casting rule, validate the declared shape, and feed the array's own
        ///     memory when ORT can read it in place, else a temporary. Any temporary this creates is owned by the
        ///     returned handle and disposed with it.
        /// </summary>
        /// <remarks>
        ///     <para><b>The dtype rule.</b> Equal dtypes need no rule. Otherwise NumSharp's <c>np.can_cast</c> decides
        ///     under <paramref name="rule"/> — for every pair, including the two the bridge used to convert without asking:
        ///     Char into a UInt16 input (allowed from <c>"safe"</c>; it is a reinterpretation of the same 16 bits, so it
        ///     stays zero-copy) and Decimal into a float64 input (allowed from <c>"same_kind"</c>: it loses digits, so the
        ///     default <c>"safe"</c> refuses it).</para>
        ///     <para><b>Order.</b> Everything the name, the dtypes, the shape and the mode decide is decided before the array
        ///     is read, so a refused input costs no conversion. Every read of the caller's array — the cast, the C-order
        ///     copy, the share — then happens under a reference on its buffer (<see cref="NDArrayOnnxInterop.ConvertPinned"/>,
        ///     <see cref="NDArrayOnnxInterop.CopyPinned"/>, the pin inside <see cref="NDArrayOnnxInterop.AsOrtValueCore"/>),
        ///     so an input whose buffer has been released is refused instead of being cast or copied from freed memory.</para>
        /// </remarks>
        /// <param name="session">The session whose metadata declares the input.</param>
        /// <param name="name">The model input's (or overridable initializer's) name.</param>
        /// <param name="array">The array to feed; any layout and any dtype castable to the declared one.</param>
        /// <param name="rule">The parsed casting rule a dtype coercion must satisfy.</param>
        /// <param name="casting">The rule as the caller wrote it, for messages.</param>
        /// <param name="mode">View, copy, or view-when-possible (already validated).</param>
        /// <returns>A handle over the memory ORT reads; it owns any temporary it was given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="array"/> is null.</exception>
        /// <exception cref="ArgumentException">The model has no input named <paramref name="name"/>, or the array's shape contradicts the declared one.</exception>
        /// <exception cref="NotSupportedException">The input is not a tensor, or its declared element type has no NumSharp dtype.</exception>
        /// <exception cref="InvalidCastException">The array's dtype does not cast to the declared one under <paramref name="rule"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array needs a conversion or a C-order copy.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static OrtTensor PrepareInput(InferenceSession session, string name, NDArray array, NPY_CASTING rule, string casting, OnnxCodecMode mode)
        {
            if (name is null)
                throw new ArgumentNullException(nameof(name));
            if (array is null)
                throw new ArgumentNullException(nameof(array), $"input '{name}' is null.");
            NodeMetadata meta = InputDeclaration(session, name);
            if (!meta.IsTensor)
                throw new NotSupportedException($"input '{name}' is {meta.OnnxValueType}, not a tensor; feed it with the OrtValue API (OrtValue.CreateSequence / CreateMap).");
            if (!NDArrayOnnxInterop.TryFromTensorElementType(meta.ElementDataType, out DType wantDtype))
                throw new NotSupportedException($"input '{name}' is declared {meta.ElementDataType}: {NDArrayOnnxInterop.UnsupportedImportMessage(meta.ElementDataType)}");
            NPTypeCode want = wantDtype.GetTypeCode();
            NPTypeCode have = array.typecode;

            if (have != want && !np.can_cast(have, want, rule))
                throw new InvalidCastException(
                    $"input '{name}' holds {have} but the model declares {want} ({meta.ElementDataType}); the cast is not allowed under " +
                    $"casting='{casting}'. Convert explicitly — nd.astype(NPTypeCode.{want}) — or relax the rule (casting: \"same_kind\" / \"unsafe\").");
            ValidateShape(name, "input", meta, array.Shape);

            // Char → UInt16 is the one allowed "cast" that changes no bit: ORT reads the UTF-16 code units in place.
            bool needsConversion = have != want && !(have == NPTypeCode.Char && want == NPTypeCode.UInt16);
            if (mode == OnnxCodecMode.View && (needsConversion || !array.Shape.IsContiguous))
                throw new InvalidOperationException(
                    $"Run(mode: OnnxCodecMode.View): input '{name}' cannot be fed as a view: " +
                    (needsConversion
                        ? $"it holds {have} and the model declares {want}, and a conversion is a copy."
                        : "it is not C-contiguous, and ONNX Runtime tensors carry no strides.") +
                    " Pass OnnxCodecMode.Auto, or feed an array of the declared dtype that is C-contiguous.");

            NDArray fed = array;
            bool owned = false;
            try
            {
                if (needsConversion)
                {
                    // The cast is the first read of the caller's array, so it runs under a reference on its buffer: a
                    // bare astype read a RELEASED buffer silently, and the model then ran on whatever the pool had put
                    // in the freed memory (the share below pins only the fresh cast, which is always alive).
                    fed = NDArrayOnnxInterop.ConvertPinned(array, want);
                    owned = true;
                }

                if (!fed.Shape.IsContiguous)
                {
                    // ORT reads only C-contiguous memory; a view is copied in logical order for the run. The copy reads
                    // the caller's array when no cast was made, so it too checks the buffer is alive first. (astype keeps
                    // the memory order, so a cast of a transposed array still needs this pass.)
                    NDArray dense = NDArrayOnnxInterop.CopyPinned(fed);
                    if (owned)
                        fed.Dispose();
                    fed = dense;
                    owned = true;
                }

                // fed is C-contiguous and crosses bit for bit now. A temporary is shared as it is (the handle owns it): a
                // second copy of it would serve nothing, also under Copy. The caller's own array is shared, or copied when
                // the caller asked for a copy of every input. View for the share: the core must not copy silently, so a
                // broken precondition here throws instead of feeding an unintended temporary.
                OrtTensor handle = NDArrayOnnxInterop.AsOrtValueCore(fed, ownsSource: owned, owned || mode != OnnxCodecMode.Copy ? OnnxCodecMode.View : OnnxCodecMode.Copy);
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
        /// <param name="session">The session whose metadata declares the output.</param>
        /// <param name="name">The model output's name.</param>
        /// <param name="array">The array ORT writes the output into.</param>
        /// <returns>A handle that shares the array's memory with ORT.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="array"/> is null.</exception>
        /// <exception cref="ArgumentException">The model has no output named <paramref name="name"/>, or the array's shape contradicts the declared one.</exception>
        /// <exception cref="NotSupportedException">The output is not a tensor.</exception>
        /// <exception cref="InvalidOperationException">The array is not of the declared dtype, is read-only, or is not C-contiguous.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
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
            // View, never Auto: ORT must write the caller's own memory, and a copy would absorb the result silently.
            return NDArrayOnnxInterop.AsOrtValueCore(array, ownsSource: false, OnnxCodecMode.View);
        }

        /// <summary>
        ///     Compare an array's shape with the model's declared shape: a declared rank must match, and every
        ///     FIXED extent must match; symbolic dims (<c>-1</c>, e.g. <c>'batch'</c>) are free. A declaration
        ///     without dimensions is either a scalar or an unknown-rank tensor — the two are indistinguishable in
        ///     <see cref="NodeMetadata"/> — so it is left to ORT's own check at run time.
        /// </summary>
        /// <param name="name">The input's or output's name, for the message.</param>
        /// <param name="role">"input" or "output", for the message.</param>
        /// <param name="meta">The tensor declaration.</param>
        /// <param name="shape">The array's shape.</param>
        /// <exception cref="ArgumentException">The rank or a fixed extent differs.</exception>
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
        /// <param name="meta">The tensor declaration.</param>
        /// <returns>The text.</returns>
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

        /// <summary>The model's only input or output name, for the single-array overload.</summary>
        /// <param name="names">The model's input or output names.</param>
        /// <param name="role">"input" or "output", for the message.</param>
        /// <returns>The single name.</returns>
        /// <exception cref="InvalidOperationException">The model declares more or fewer than one.</exception>
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
