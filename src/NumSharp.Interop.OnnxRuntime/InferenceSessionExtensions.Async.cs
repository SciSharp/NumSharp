using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends.Iteration;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class InferenceSessionExtensions
    {
        // ===========================  asynchronous runs  ===========================
        //
        // ONNX Runtime's managed InferenceSession.RunAsync schedules the run on the session's intra-op thread pool and
        // completes a task from a native callback: no .NET thread waits for the run. The two forms here build on it
        // under four facts, each measured (docs/plans/onnxruntime-rework.md, section 3.6):
        //
        //  1. Before 1.30.0 the managed code handed its name and value arrays to the native run without pinning them,
        //     and the native side keeps the pointers until the run ends (microsoft/onnxruntime#32015): the garbage
        //     collector could move them under it. So both forms refuse an older assembly.
        //  2. ORT completes its task ON ITS OWN INTRA-OP THREAD (its completion source runs continuations inline). Code
        //     that continues there runs inside the session's thread pool: session.Dispose() called there never
        //     returned (measured on 1.31.0). So the task a caller gets here completes its continuations
        //     asynchronously, off that thread.
        //  3. ORT awaits its completion source without ConfigureAwait(false), so it captures the caller's
        //     SynchronizationContext: a caller that blocked a thread which owns a context waited forever for ORT's own
        //     continuation (measured: Wait(2 s) returned false, then true once the context was pumped by hand). So
        //     the context, and a caller's task scheduler, are hidden from ORT for the call.
        //  4. The session needs 2 or more intra-op threads. With 1, ORT refuses and its task is faulted.

        /// <summary>The first ONNX Runtime version whose managed <c>RunAsync</c> pins the arrays it hands to the native run.</summary>
        private static readonly Version RunAsyncSafeFrom = new Version(1, 30);

        /// <summary>
        ///     The version the loaded managed ONNX Runtime assembly reports, read once. An assembly that reports none
        ///     gives 0.0.0.0 (the 1.16.0 package does), which counts as older than every release.
        /// </summary>
        private static readonly Version LoadedOrtVersion = typeof(InferenceSession).Assembly.GetName().Version ?? new Version(0, 0);

        /// <summary>
        ///     Can <see cref="RunIntoAsync"/> and <see cref="RunAsync"/> run on the ONNX Runtime that is loaded? They
        ///     need its managed assembly at version 1.30.0 or newer, the first that keeps its argument arrays pinned
        ///     while the native run reads them. On an older one both forms throw <see cref="NotSupportedException"/>,
        ///     so a caller that must work on both tests this property and calls the synchronous forms otherwise.
        /// </summary>
        /// <remarks>
        ///     The managed assembly is what is tested, because the defect was in the managed code. An assembly that
        ///     reports no version (0.0.0.0) counts as older, also when it is a private build of a newer source.
        /// </remarks>
        public static bool IsRunAsyncSupported => LoadedOrtVersion >= RunAsyncSafeFrom;

        /// <summary>
        ///     Run into PRE-ALLOCATED outputs without blocking a thread: ONNX Runtime executes the model on the
        ///     session's own intra-op thread pool and writes each result straight into the caller's array. The
        ///     returned task completes when the run has ended. Zero-copy in both directions, like
        ///     <see cref="RunInto"/>, of which this is the asynchronous form: the same inputs, the same output
        ///     rules, the same overlap rule.
        /// </summary>
        /// <remarks>
        ///     <para><b>What it needs.</b> ONNX Runtime 1.30.0 or newer (<see cref="IsRunAsyncSupported"/>), and a session
        ///     with 2 or more intra-op threads (<c>SessionOptions.IntraOpNumThreads</c> is not 1): with a single
        ///     thread ONNX Runtime refuses an asynchronous run, and the task is faulted with its message.</para>
        ///     <para><b>While the run is in flight.</b> ONNX Runtime reads every input that is fed in place, and writes
        ///     every output, on its own threads. So until the task completes: do not write an input array (the model
        ///     would compute on a mix of old and new values), and do not read an output array. Pass
        ///     <see cref="OnnxCodecMode.Copy"/> to feed a copy of every input taken during this call; the caller's
        ///     arrays may then change at once. DISPOSING an input or an output array before the task completes is
        ///     safe: the run holds its own reference on every buffer until it ends.</para>
        ///     <para><b>Where the task completes.</b> ONNX Runtime completes its own task on an intra-op thread of the
        ///     session. The task returned here never runs a caller's continuation on that thread: after an
        ///     <c>await</c>, code runs where an awaited task normally resumes it (the caller's synchronization
        ///     context, or the thread pool). This matters: code that continues on the session's own thread and
        ///     disposes the session never returns. The bridge needs no synchronization context for its own work, so a
        ///     thread that blocks on the task is released when the run ends.</para>
        ///     <para><b>Errors.</b> Everything this method can check is checked before the run starts and throws from
        ///     the call itself, not through the task: a null argument, an unknown name, a dtype that does not cast, a
        ///     shape that contradicts the model, an input that would need a copy under
        ///     <see cref="OnnxCodecMode.View"/>. A failure inside ONNX Runtime faults the task with ORT's
        ///     <c>OnnxRuntimeException</c>. In both cases every handle and temporary of the run is released.</para>
        ///     <para><b>Cancellation.</b> Pass a <see cref="RunOptions"/> and set its <c>Terminate</c> property from
        ///     any thread: ONNX Runtime stops the run and the task is faulted. The <paramref name="runOptions"/> object
        ///     and the session must stay alive, and not be disposed, until the task completes.</para>
        /// </remarks>
        /// <param name="session">The session; it must have 2 or more intra-op threads.</param>
        /// <param name="inputs">Model input name → array, fed as in <see cref="RunInto"/>.</param>
        /// <param name="outputs">Model output name → the array ONNX Runtime writes it into: C-contiguous, writeable, of the declared dtype and of the shape the run produces.</param>
        /// <param name="runOptions">Optional run options; <c>null</c> runs with ONNX Runtime's defaults. Keep it alive until the task completes.</param>
        /// <param name="casting">NumPy casting rule for dtype coercion of INPUTS (default <c>"safe"</c>); outputs are never cast.</param>
        /// <param name="mode">How the INPUTS are fed; see <see cref="OnnxCodecMode"/>. The outputs are always shared.</param>
        /// <returns>
        ///     A task that completes when the run has ended and every output array holds its result. It is faulted with
        ///     ONNX Runtime's exception when the run fails (a wrong output shape, a terminated run, a session with one
        ///     intra-op thread).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="session"/>, <paramref name="inputs"/> or <paramref name="outputs"/> is null, or a name or array in them is null.</exception>
        /// <exception cref="NotSupportedException">The loaded ONNX Runtime is older than 1.30.0; or an input or an output is not a tensor of a dtype NumSharp has.</exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="outputs"/> is empty; the model has no input or output of a given name; a shape contradicts
        ///     the declared one; two outputs overlap in memory; or the array for an output that is a fed input does not
        ///     have that input's shape.
        /// </exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">An input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="InvalidOperationException">
        ///     An output array is read-only, not C-contiguous, not of the declared dtype or text; or <paramref name="mode"/>
        ///     is <see cref="OnnxCodecMode.View"/> and an input needs a copy (a cast, a layout, or an overlap with an output).
        /// </exception>
        /// <exception cref="ObjectDisposedException">An input's or an output's buffer has already been released.</exception>
        public static Task RunIntoAsync(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyDictionary<string, NDArray> outputs, RunOptions runOptions = null, string casting = "safe",
            OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            if (outputs is null)
                throw new ArgumentNullException(nameof(outputs));
            EnsureRunAsyncSupported(nameof(RunIntoAsync), nameof(RunInto));
            NPY_CASTING rule = DTypeCasting.ParseCasting(casting);
            NDArrayOnnxInterop.EnsureDefined(mode);
            if (outputs.Count == 0)
                throw new ArgumentException("no output was given: a run needs at least one output to write.", nameof(outputs));

            var run = new PendingRun(session, runOptions, inputs.Count, outputs.Count, allocated: null);
            try
            {
                run.PrepareInputs(inputs, rule, casting, mode);
                int j = 0;
                foreach (KeyValuePair<string, NDArray> output in outputs)
                {
                    run.OutHandles[j] = PrepareOutput(session, output.Key, output.Value);
                    run.OutNames[j] = output.Key;
                    j++;
                }

                // The same rule as the synchronous form, on the memory ORT will really read and write. It matters more
                // here: the caller has its thread back while ORT still reads the inputs.
                SeparateInputsFromOutputs(run.InNames, run.InHandles, run.OutNames, run.OutHandles, mode, nameof(RunIntoAsync));
                return run.Start();
            }
            catch
            {
                run.Abort();   // nothing was started (Abort does nothing once ORT has the run)
                throw;
            }
        }

        /// <summary>
        ///     Run with named inputs without blocking a thread, and get the fetched outputs as new owning arrays. The
        ///     asynchronous form of <see cref="Run(InferenceSession, IReadOnlyDictionary{string, NDArray}, IReadOnlyCollection{string}, RunOptions, string, bool, OnnxCodecMode)"/>,
        ///     with one limit that ONNX Runtime sets: an asynchronous run writes into memory that exists before it
        ///     starts. So this method allocates every output itself, and can do that only for an output whose shape is
        ///     known before the run.
        /// </summary>
        /// <remarks>
        ///     <para><b>Which outputs it can allocate.</b> A tensor of a numeric dtype whose declared shape is fixed (every
        ///     dimension a number), and an output that is a fed input passed through, which takes the fed input's
        ///     shape. For any other output the method throws <see cref="NotSupportedException"/> before it reads an
        ///     input, and the message says what to do:</para>
        ///     <para>- a dimension the model fixes only at run time (<c>'batch'</c>): allocate the array yourself, with
        ///     the shape you expect, and call <see cref="RunIntoAsync"/>;</para>
        ///     <para>- a declaration without dimensions: it is a scalar or a tensor of unknown rank, and ONNX Runtime's
        ///     metadata does not tell which, so the method does not guess; <see cref="RunIntoAsync"/> takes a 0-d
        ///     array;</para>
        ///     <para>- a string tensor, a sequence or a map: ONNX Runtime creates these itself, so only the synchronous
        ///     forms return them.</para>
        ///     <para>A name that appears more than once in <paramref name="outputNames"/> is fetched once.</para>
        ///     <para>Everything <see cref="RunIntoAsync"/> says about the run in flight, the thread the task completes
        ///     on, errors and cancellation holds here too. When the task is faulted, the arrays this method allocated
        ///     are disposed.</para>
        /// </remarks>
        /// <param name="session">The session; it must have 2 or more intra-op threads.</param>
        /// <param name="inputs">Model input name → array, fed as in the synchronous form.</param>
        /// <param name="outputNames">Outputs to fetch; <c>null</c> = <see cref="InferenceSession.OutputNames"/>.</param>
        /// <param name="runOptions">Optional run options; <c>null</c> runs with ONNX Runtime's defaults. Keep it alive until the task completes.</param>
        /// <param name="casting">NumPy casting rule for dtype coercion of inputs (default <c>"safe"</c>).</param>
        /// <param name="mode">How the INPUTS are fed; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>
        ///     A task whose result holds the fetched outputs by name: new owning C-contiguous arrays the caller disposes.
        ///     It is faulted with ONNX Runtime's exception when the run fails.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="session"/> or <paramref name="inputs"/> is null, or a name or array is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     The loaded ONNX Runtime is older than 1.30.0; an output cannot be allocated before the run (see the
        ///     remarks); or an input is not a tensor of a dtype NumSharp has.
        /// </exception>
        /// <exception cref="ArgumentException"><paramref name="outputNames"/> is empty; the model has no input or output of a given name; or an input's shape contradicts the declared one.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five rule names.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidCastException">An input's dtype does not cast to the declared one under <paramref name="casting"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and an input needs a copy.</exception>
        /// <exception cref="ObjectDisposedException">An input's buffer has already been released.</exception>
        public static Task<IReadOnlyDictionary<string, NDArray>> RunAsync(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyCollection<string> outputNames = null, RunOptions runOptions = null, string casting = "safe",
            OnnxCodecMode mode = OnnxCodecMode.Auto)
        {
            if (session is null)
                throw new ArgumentNullException(nameof(session));
            if (inputs is null)
                throw new ArgumentNullException(nameof(inputs));
            EnsureRunAsyncSupported(nameof(RunAsync), nameof(Run));
            NPY_CASTING rule = DTypeCasting.ParseCasting(casting);
            NDArrayOnnxInterop.EnsureDefined(mode);

            // Everything the model's declarations decide is decided here, before an input is read: an output that
            // cannot be allocated costs no cast and no copy.
            string[] names = DistinctOutputNames(outputNames ?? session.OutputNames);
            var declared = new NodeMetadata[names.Length];
            for (int j = 0; j < names.Length; j++)
                declared[j] = AllocatableOutput(session, names[j], isFed: inputs.ContainsKey(names[j]));

            var allocated = new Dictionary<string, NDArray>(names.Length);
            var run = new PendingRun(session, runOptions, inputs.Count, names.Length, allocated);
            try
            {
                run.PrepareInputs(inputs, rule, casting, mode);
                for (int j = 0; j < names.Length; j++)
                {
                    // A fed input passed through has the fed tensor's dtype and shape, whatever the model declares for
                    // the dimensions; every other output has the fixed shape its declaration states.
                    OrtTensor fed = FedTensor(names[j], run.InNames, run.InHandles);
                    NDArray array = fed is null
                        ? new NDArray(NDArrayOnnxInterop.FromTensorElementType(declared[j].ElementDataType), DeclaredShape(declared[j]), fillZeros: false)
                        : new NDArray(NDArrayOnnxInterop.FromTensorElementType(fed.ElementType), NDArrayOnnxInterop.ShapeFrom(fed.ShapeNoCopy), fillZeros: false);
                    allocated.Add(names[j], array);   // from here on the run disposes it when it fails
                    run.OutHandles[j] = PrepareOutput(session, names[j], array);
                    run.OutNames[j] = names[j];
                }

                // No overlap pass: every output is a new array, which shares memory with nothing.
                return run.Start();
            }
            catch
            {
                run.Abort();
                throw;
            }
        }

        // ---- the version gate ----------------------------------------------------------------------

        /// <summary>
        ///     Refuses an asynchronous run on an ONNX Runtime whose managed <c>RunAsync</c> does not pin its arguments.
        ///     The refusal is a property of the loaded assembly, so it comes before any other check of the arguments but
        ///     the null checks.
        /// </summary>
        /// <param name="verb">The asynchronous verb that was called (named in the message).</param>
        /// <param name="synchronousVerb">The synchronous verb that does the same work on any version.</param>
        /// <exception cref="NotSupportedException">The loaded managed assembly reports a version older than 1.30.0.</exception>
        private static void EnsureRunAsyncSupported(string verb, string synchronousVerb)
        {
            if (IsRunAsyncSupported)
                return;
            // The native version is only for the message: the 1.16.0 assembly reports 0.0.0.0, which alone tells a
            // reader nothing. A failure to read it must not replace the refusal.
            string native;
            try
            {
                native = OrtEnv.Instance().GetVersionString();
            }
            catch (Exception)
            {
                native = "unknown";
            }

            throw new NotSupportedException(
                $"{verb}: ONNX Runtime's asynchronous run is memory-safe only from version 1.30.0. An older managed assembly hands its name and value " +
                "arrays to the native run without pinning them, and the native side reads them until the run ends, so the garbage collector can move " +
                "them under it (microsoft/onnxruntime#32015). The loaded Microsoft.ML.OnnxRuntime assembly reports version " +
                $"{LoadedOrtVersion} (native library: {native}); an assembly that reports no version counts as older. " +
                $"Call the synchronous {synchronousVerb}, or reference ONNX Runtime 1.30.0 or newer. " +
                $"{nameof(InferenceSessionExtensions)}.{nameof(IsRunAsyncSupported)} tells which applies.");
        }

        // ---- outputs RunAsync allocates ------------------------------------------------------------

        /// <summary>The names to fetch, each once, in the order given.</summary>
        /// <param name="outputNames">The requested names (the caller's, or the model's own list).</param>
        /// <returns>The distinct names.</returns>
        /// <exception cref="ArgumentNullException">A name is null.</exception>
        /// <exception cref="ArgumentException">No name was given: ONNX Runtime needs at least one output to fetch.</exception>
        private static string[] DistinctOutputNames(IReadOnlyCollection<string> outputNames)
        {
            var names = new List<string>(outputNames.Count);
            foreach (string name in outputNames)
            {
                if (name is null)
                    throw new ArgumentNullException(nameof(outputNames), "an output name is null.");
                // A model has a handful of outputs: a linear scan needs no set.
                if (!names.Contains(name))
                    names.Add(name);
            }

            if (names.Count == 0)
                throw new ArgumentException("no output name was given: a run needs at least one output to fetch.", nameof(outputNames));
            return names.ToArray();
        }

        /// <summary>
        ///     The declaration of an output <see cref="RunAsync"/> is asked to fetch, after the check that the method can
        ///     allocate it before the run: a tensor, of a numeric dtype NumSharp has, and either a fed input passed
        ///     through (its shape comes from what is fed) or declared with a fixed shape.
        /// </summary>
        /// <param name="session">The session whose metadata declares the output.</param>
        /// <param name="name">The output's name.</param>
        /// <param name="isFed"><c>true</c> when the caller feeds an input of this name, so the output is that input passed through.</param>
        /// <returns>The output's tensor declaration.</returns>
        /// <exception cref="ArgumentException">The model has no output named <paramref name="name"/>.</exception>
        /// <exception cref="NotSupportedException">The output is not a numeric tensor NumSharp has a dtype for, or its shape is not known before the run.</exception>
        private static NodeMetadata AllocatableOutput(InferenceSession session, string name, bool isFed)
        {
            if (!session.OutputMetadata.TryGetValue(name, out NodeMetadata meta))
                throw new ArgumentException($"the model has no output named '{name}'; its outputs are [{string.Join(", ", session.OutputNames)}].", nameof(name));
            if (!meta.IsTensor)
                throw new NotSupportedException(
                    $"RunAsync: output '{name}' is {meta.OnnxValueType}, not a tensor. An asynchronous run writes into memory that exists before it starts, " +
                    "and ONNX Runtime creates a sequence or a map itself. Fetch this output with the synchronous OrtValue API (session.Run, then " +
                    "ToNDArrays / ToMap / ToMaps), or fetch only the tensor outputs here.");
            if (meta.ElementDataType == TensorElementType.String)
                throw new NotSupportedException(
                    $"RunAsync: output '{name}' is a string tensor. ONNX Runtime writes its strings into objects of its own, never into NumSharp memory, " +
                    "so no array can be allocated for it before the run. Fetch it with the synchronous Run, which returns it as a text array.");
            if (!NDArrayOnnxInterop.TryFromTensorElementType(meta.ElementDataType, out _))
                throw new NotSupportedException($"RunAsync: output '{name}' is declared {meta.ElementDataType}: {NDArrayOnnxInterop.UnsupportedImportMessage(meta.ElementDataType)}");
            if (isFed)
                return meta;   // its shape is the fed input's, known once the input is prepared

            int[] dims = meta.Dimensions;
            if (dims is null || dims.Length == 0)
                throw new NotSupportedException(
                    $"RunAsync: output '{name}' is declared without dimensions. That is a scalar or a tensor of unknown rank, and ONNX Runtime's " +
                    "metadata does not tell which, so its array cannot be allocated before the run. Allocate it yourself (a 0-d array for a scalar) " +
                    "and call RunIntoAsync.");
            foreach (int d in dims)
            {
                if (d < 0)
                    throw new NotSupportedException(
                        $"RunAsync: output '{name}' is declared with the shape {DescribeDims(meta)}, which has a dimension the model fixes only at run " +
                        "time, so its array cannot be allocated before the run. Allocate it yourself, with the shape this run produces, and call RunIntoAsync.");
            }

            return meta;
        }

        /// <summary>The fixed shape a declaration states, as a NumSharp shape.</summary>
        /// <param name="meta">A declaration <see cref="AllocatableOutput"/> accepted as fixed.</param>
        /// <returns>The shape (ONNX Runtime reports the dimensions as 32-bit numbers).</returns>
        private static Shape DeclaredShape(NodeMetadata meta) =>
            NDArrayOnnxInterop.ShapeFrom(Array.ConvertAll(meta.Dimensions, d => (long)d));

        // ---- one run in flight ---------------------------------------------------------------------

        /// <summary>
        ///     One asynchronous run: the handles ONNX Runtime reads and writes, and the task the caller waits on. It is
        ///     the object that keeps every buffer of the run alive while no caller's frame does, and the one place that
        ///     releases them, exactly once, when the run ends or cannot start.
        /// </summary>
        /// <remarks>
        ///     <para><b>What keeps it alive.</b> ONNX Runtime roots its callback state with a GC handle for the length of
        ///     the run; that state references ORT's task, whose continuation is <see cref="Finish"/> on this object. So
        ///     the handles here cannot be finalized under a run in flight, also when the caller drops the task.</para>
        ///     <para><b>Order at the end.</b> The handles are released first, then the task completes. A caller that
        ///     continues after the task therefore sees its arrays free of the run's references (an
        ///     <c>ndarray.resize</c> with its reference check works at once).</para>
        /// </remarks>
        private sealed class PendingRun
        {
            /// <summary>The names fed, in the order of <see cref="InHandles"/>.</summary>
            internal readonly string[] InNames;

            /// <summary>The prepared inputs; a slot is null until its input is prepared.</summary>
            internal readonly OrtTensor[] InHandles;

            /// <summary>The names fetched, in the order of <see cref="OutHandles"/>.</summary>
            internal readonly string[] OutNames;

            /// <summary>The prepared outputs; a slot is null until its output is prepared.</summary>
            internal readonly OrtTensor[] OutHandles;

            private readonly InferenceSession _session;
            private readonly RunOptions _runOptions;

            // RunAsync: the arrays this run allocated, which are the task's result and are disposed when the run fails.
            // RunIntoAsync: null (the arrays are the caller's).
            private readonly Dictionary<string, NDArray> _allocated;

            // RunContinuationsAsynchronously is the point of this source: Finish runs on ORT's intra-op thread, and a
            // caller's continuation must not (see fact 2 at the top of the file). A thread that BLOCKS on the task is
            // still released at once: a blocking wait is not a continuation.
            private readonly TaskCompletionSource<IReadOnlyDictionary<string, NDArray>> _completion =
                new TaskCompletionSource<IReadOnlyDictionary<string, NDArray>>(TaskCreationOptions.RunContinuationsAsynchronously);

            private OrtTensor[] _passedThrough;
            private int _started;

            /// <summary>Creates the run's empty slots.</summary>
            /// <param name="session">The session that runs.</param>
            /// <param name="runOptions">The caller's run options, or null.</param>
            /// <param name="inputCount">How many inputs are fed.</param>
            /// <param name="outputCount">How many outputs are fetched.</param>
            /// <param name="allocated">The dictionary the caller fills with the arrays it allocates for the outputs (the task's result), or null when the arrays are the caller's.</param>
            internal PendingRun(InferenceSession session, RunOptions runOptions, int inputCount, int outputCount, Dictionary<string, NDArray> allocated)
            {
                _session = session;
                _runOptions = runOptions;
                _allocated = allocated;
                InNames = new string[inputCount];
                InHandles = new OrtTensor[inputCount];
                OutNames = new string[outputCount];
                OutHandles = new OrtTensor[outputCount];
            }

            /// <summary>Prepares every input as the synchronous forms do (name, dtype rule, declared shape, mode).</summary>
            /// <param name="inputs">Model input name → array.</param>
            /// <param name="rule">The parsed casting rule.</param>
            /// <param name="casting">The rule as the caller wrote it, for messages.</param>
            /// <param name="mode">The input mode.</param>
            /// <exception cref="ArgumentNullException">A name or an array is null.</exception>
            /// <exception cref="ArgumentException">The model has no input of a given name, or a shape contradicts the declared one.</exception>
            /// <exception cref="NotSupportedException">An input is not a tensor of a dtype NumSharp has.</exception>
            /// <exception cref="InvalidCastException">A dtype does not cast under the rule.</exception>
            /// <exception cref="InvalidOperationException">The mode is <see cref="OnnxCodecMode.View"/> and an input needs a copy.</exception>
            /// <exception cref="ObjectDisposedException">An input's buffer has already been released.</exception>
            internal void PrepareInputs(IReadOnlyDictionary<string, NDArray> inputs, NPY_CASTING rule, string casting, OnnxCodecMode mode)
            {
                int i = 0;
                foreach (KeyValuePair<string, NDArray> input in inputs)
                {
                    InHandles[i] = PrepareInput(_session, input.Key, input.Value, rule, casting, mode);
                    InNames[i] = input.Key;
                    i++;
                }
            }

            /// <summary>
            ///     Hands the prepared run to ONNX Runtime and returns the task the caller waits on. Once ORT has the run,
            ///     this object owns the release of everything; <see cref="Abort"/> then does nothing.
            /// </summary>
            /// <returns>The caller's task.</returns>
            /// <exception cref="ArgumentException">The array for an output that is a fed input does not have that input's shape.</exception>
            /// <exception cref="InvalidOperationException">Such an array does not have the fed input's element type.</exception>
            internal Task<IReadOnlyDictionary<string, NDArray>> Start()
            {
                // After every handle is final (the overlap pass may have replaced an input handle), before the run.
                _passedThrough = PassThroughSources(InNames, InHandles, OutNames, OutHandles);
                var inValues = new OrtValue[InHandles.Length];
                for (int i = 0; i < inValues.Length; i++)
                    inValues[i] = InHandles[i].Value;
                var outValues = new OrtValue[OutHandles.Length];
                for (int j = 0; j < outValues.Length; j++)
                    outValues[j] = OutHandles[j].Value;

                Task<IReadOnlyCollection<OrtValue>> native = StartNative(_session, _runOptions, InNames, inValues, OutNames, outValues);
                // ORT has the run (or holds the failure in the task). From here a release from the caller's frame would
                // free memory ORT may be reading, so only Finish releases.
                Volatile.Write(ref _started, 1);
                // ExecuteSynchronously: Finish runs where ORT completes its task, with no thread-pool hop of its own. It
                // only releases handles and completes the source, whose continuations are the ones that hop.
                native.ContinueWith(static (task, state) => ((PendingRun)state).Finish(task), this, CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
                return _completion.Task;
            }

            /// <summary>
            ///     Calls ORT's <c>RunAsync</c> so that its internal <c>await</c> can capture nothing of the caller: no
            ///     synchronization context and no task scheduler.
            /// </summary>
            /// <remarks>
            ///     ORT awaits its completion source without <c>ConfigureAwait(false)</c>. An <c>await</c> captures the
            ///     current synchronization context, or else the current task scheduler when it is not the default one,
            ///     and resumes there. With a caller's context captured, ORT's task (and so this run's cleanup) would wait
            ///     for that context to run it: a caller that blocks the context's thread on the task would wait forever.
            ///     The context can be hidden for the length of the call. A task scheduler cannot, so from inside a task
            ///     of another scheduler the call is made from the default one, which costs one thread-pool hop.
            /// </remarks>
            /// <param name="session">The session.</param>
            /// <param name="runOptions">The run options, or null.</param>
            /// <param name="inNames">The input names.</param>
            /// <param name="inValues">The input values.</param>
            /// <param name="outNames">The output names.</param>
            /// <param name="outValues">The pre-allocated output values.</param>
            /// <returns>ORT's task for the run.</returns>
            private static Task<IReadOnlyCollection<OrtValue>> StartNative(InferenceSession session, RunOptions runOptions, string[] inNames,
                OrtValue[] inValues, string[] outNames, OrtValue[] outValues)
            {
                if (TaskScheduler.Current != TaskScheduler.Default)
                {
                    return Task.Factory.StartNew(
                        () => session.RunAsync(runOptions, inNames, inValues, outNames, outValues),
                        CancellationToken.None, TaskCreationOptions.DenyChildAttach, TaskScheduler.Default).Unwrap();
                }

                SynchronizationContext context = SynchronizationContext.Current;
                if (context is null)
                    return session.RunAsync(runOptions, inNames, inValues, outNames, outValues);
                SynchronizationContext.SetSynchronizationContext(null);
                try
                {
                    return session.RunAsync(runOptions, inNames, inValues, outNames, outValues);
                }
                finally
                {
                    SynchronizationContext.SetSynchronizationContext(context);
                }
            }

            /// <summary>
            ///     Releases a run that did not start: every handle prepared so far and every array allocated so far. It
            ///     does nothing once ORT has the run, because ORT may be reading the memory.
            /// </summary>
            internal void Abort()
            {
                if (Volatile.Read(ref _started) != 0)
                    return;
                DisposeHandles();
                DisposeAllocated();
            }

            /// <summary>
            ///     The end of the run, on the thread where ORT completes its task (an intra-op thread of the session, or
            ///     the caller's own thread when ORT refused the run at once). Completes the pass-through outputs, releases
            ///     every handle, then completes the caller's task. It must not throw: an exception here would be lost in
            ///     a continuation nobody observes, and the caller's task would never complete.
            /// </summary>
            /// <param name="native">ORT's completed task.</param>
            private void Finish(Task<IReadOnlyCollection<OrtValue>> native)
            {
                bool succeeded = !native.IsFaulted && !native.IsCanceled;
                // Each step of the cleanup is tried on its own: a failure in one must not skip the next, and none of
                // them may leave the caller's task incomplete. None is expected to fail.
                Exception cleanupFailure = null;
                try
                {
                    if (succeeded)
                        CopyPassThrough(_passedThrough, OutHandles);   // the part of the run ORT does not do
                }
                catch (Exception e)
                {
                    cleanupFailure = e;
                }

                try
                {
                    DisposeHandles();
                }
                catch (Exception e)
                {
                    cleanupFailure ??= e;
                }

                if (!succeeded || cleanupFailure is not null)
                {
                    try
                    {
                        DisposeAllocated();   // a failed run returns nothing, so nobody else can dispose these
                    }
                    catch (Exception e)
                    {
                        cleanupFailure ??= e;
                    }
                }

                // ORT 1.30 and newer keep the session and the options referenced for the run; this keeps the rule
                // independent of that.
                GC.KeepAlive(_session);
                GC.KeepAlive(_runOptions);

                // Everything is released: now the caller may see the end of the run. Reading Exception marks ORT's
                // failure as observed; it travels on in the caller's task, with its own stack.
                if (native.IsFaulted)
                    _completion.TrySetException(native.Exception.InnerExceptions);
                else if (native.IsCanceled)
                    _completion.TrySetCanceled();
                else if (cleanupFailure is not null)
                    _completion.TrySetException(cleanupFailure);
                else
                    _completion.TrySetResult(_allocated);
            }

            /// <summary>Disposes every prepared handle; an unprepared slot is null. A handle's dispose is idempotent.</summary>
            private void DisposeHandles()
            {
                foreach (OrtTensor handle in InHandles)
                    handle?.Dispose();
                foreach (OrtTensor handle in OutHandles)
                    handle?.Dispose();
            }

            /// <summary>Disposes the arrays this run allocated for its outputs (none for a run into the caller's arrays).</summary>
            private void DisposeAllocated()
            {
                if (_allocated is null)
                    return;
                foreach (NDArray array in _allocated.Values)
                    array.Dispose();
            }
        }
    }
}
