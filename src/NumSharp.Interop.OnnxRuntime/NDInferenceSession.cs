using System;
using System.Threading;
using Microsoft.ML.OnnxRuntime;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     An <see cref="InferenceSession"/> that is safe to dispose while other threads use it: its <c>Dispose</c> releases
    ///     the native session EXACTLY ONCE, however many threads call it at once, and only after every run this package's
    ///     verbs started on it has returned (<see cref="InferenceSessionExtensions"/>' <c>Run</c>, <c>RunInto</c>,
    ///     <c>RunAsync</c> and <c>RunIntoAsync</c>). A run started after the <c>Dispose</c> began throws
    ///     <see cref="ObjectDisposedException"/>. Everything else is ONNX Runtime's session, unchanged: an
    ///     <see cref="NDInferenceSession"/> IS an <see cref="InferenceSession"/>, so it goes wherever one goes.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why a subclass.</b> <see cref="InferenceSession"/> is ONNX Runtime's class, and its <c>Dispose</c> takes
    ///     no lock and calls nothing a caller could hook before it frees what a run reads: it frees the UTF-8 name buffers
    ///     every run passes to the native session first, then the options, then the native session. Measured 2026-11-07
    ///     (ORT 1.31.1, plain <see cref="InferenceSession"/>): two threads disposing one session corrupted the heap
    ///     (<c>0xC0000374</c>), and a <c>Dispose</c> during this package's <c>Run</c> was an access violation in
    ///     <c>RunImpl</c>. <c>Dispose(bool)</c> is the one virtual member, so only a subclass can make the release wait and
    ///     happen once. The same races on this type: 0 crashes in 800 each, one release per session.</para>
    ///     <para><b>What it does not cover.</b> ONNX Runtime's own run methods (its <c>Run</c>, <c>RunWithBinding</c>,
    ///     <c>RunWithBoundResults</c> and <c>RunAsync</c> overloads) are not virtual, so a call to them takes no lease:
    ///     dispose the session only after those have returned. A plain <see cref="InferenceSession"/> keeps ONNX Runtime's
    ///     rule for every run, this package's included (dispose it once, after every run returned).</para>
    ///     <para><b>Where the deferred release runs.</b> On the disposing thread when no run is in flight. Otherwise on the
    ///     thread the last synchronous run returns on, or on the .NET thread pool after an asynchronous run - never on ONNX
    ///     Runtime's intra-op thread, where an asynchronous run completes and where releasing the session never returns
    ///     (measured on 1.31.0; see <c>InferenceSessionExtensions.Async.cs</c>).</para>
    ///     <para><b>Constructors.</b> ONNX Runtime's eight, the same at the floor (1.16.0) and the current release (1.31.1).
    ///     A constructor that throws leaves nothing to dispose: ONNX Runtime's own constructor releases what it made.</para>
    /// </remarks>
    public sealed class NDInferenceSession : InferenceSession
    {
        /// <summary>Native sessions released by <see cref="ReleaseOnce"/> since the process started. A diagnostic for the tests: one per disposed session.</summary>
        private static long _releases;

        /// <summary>This package's runs in flight on the session; the release waits for 0.</summary>
        private int _runs;

        /// <summary>1 once a <c>Dispose</c> began: new runs are refused from then on.</summary>
        private int _closing;

        /// <summary>1 once the native session was released: the once-guard of <see cref="ReleaseOnce"/>.</summary>
        private int _released;

        /// <summary>Opens the model at <paramref name="modelPath"/> with ONNX Runtime's default options.</summary>
        /// <param name="modelPath">The <c>.onnx</c> file.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model (ONNX Runtime's own exceptions, unchanged).</exception>
        public NDInferenceSession(string modelPath) : base(modelPath) { }

        /// <summary>Opens the model at <paramref name="modelPath"/>, sharing pre-packed weights through <paramref name="prepackedWeightsContainer"/>.</summary>
        /// <param name="modelPath">The <c>.onnx</c> file.</param>
        /// <param name="prepackedWeightsContainer">The container sessions of the same model share pre-packed weights through.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model.</exception>
        public NDInferenceSession(string modelPath, PrePackedWeightsContainer prepackedWeightsContainer) : base(modelPath, prepackedWeightsContainer) { }

        /// <summary>Opens the model at <paramref name="modelPath"/> with <paramref name="options"/>.</summary>
        /// <param name="modelPath">The <c>.onnx</c> file.</param>
        /// <param name="options">The session options (execution providers, threads, optimization level, ...).</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model or apply the options.</exception>
        public NDInferenceSession(string modelPath, SessionOptions options) : base(modelPath, options) { }

        /// <summary>Opens the model at <paramref name="modelPath"/> with <paramref name="options"/>, sharing pre-packed weights.</summary>
        /// <param name="modelPath">The <c>.onnx</c> file.</param>
        /// <param name="options">The session options.</param>
        /// <param name="prepackedWeightsContainer">The container sessions of the same model share pre-packed weights through.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model or apply the options.</exception>
        public NDInferenceSession(string modelPath, SessionOptions options, PrePackedWeightsContainer prepackedWeightsContainer)
            : base(modelPath, options, prepackedWeightsContainer) { }

        /// <summary>Opens the model serialized in <paramref name="model"/> with ONNX Runtime's default options.</summary>
        /// <param name="model">The model's bytes.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model.</exception>
        public NDInferenceSession(byte[] model) : base(model) { }

        /// <summary>Opens the model serialized in <paramref name="model"/>, sharing pre-packed weights.</summary>
        /// <param name="model">The model's bytes.</param>
        /// <param name="prepackedWeightsContainer">The container sessions of the same model share pre-packed weights through.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model.</exception>
        public NDInferenceSession(byte[] model, PrePackedWeightsContainer prepackedWeightsContainer) : base(model, prepackedWeightsContainer) { }

        /// <summary>Opens the model serialized in <paramref name="model"/> with <paramref name="options"/>.</summary>
        /// <param name="model">The model's bytes.</param>
        /// <param name="options">The session options.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model or apply the options.</exception>
        public NDInferenceSession(byte[] model, SessionOptions options) : base(model, options) { }

        /// <summary>Opens the model serialized in <paramref name="model"/> with <paramref name="options"/>, sharing pre-packed weights.</summary>
        /// <param name="model">The model's bytes.</param>
        /// <param name="options">The session options.</param>
        /// <param name="prepackedWeightsContainer">The container sessions of the same model share pre-packed weights through.</param>
        /// <exception cref="OnnxRuntimeException">ONNX Runtime cannot load the model or apply the options.</exception>
        public NDInferenceSession(byte[] model, SessionOptions options, PrePackedWeightsContainer prepackedWeightsContainer)
            : base(model, options, prepackedWeightsContainer) { }

        /// <summary>
        ///     <c>true</c> once a <c>Dispose</c> began. This package's verbs then refuse new runs; the runs already in flight
        ///     finish, and the last of them releases the native session (which may still be alive when this turns true).
        /// </summary>
        public bool IsDisposed => Volatile.Read(ref _closing) != 0;

        /// <summary>The native sessions this type released since the process started (one per disposed session). A diagnostic for the tests.</summary>
        internal static long Releases => Interlocked.Read(ref _releases);

        /// <summary>
        ///     Opens a run: fails once a <c>Dispose</c> began. The increment comes before the read of the closing flag and
        ///     <see cref="Dispose(bool)"/> sets the flag before it reads the count, both with full fences, so either the run
        ///     sees the flag or the dispose sees the run - never neither. A run refused here may still have been seen by the
        ///     dispose, which then deferred the release to it: the refusal gives the run back like an ended one, which makes
        ///     that release.
        /// </summary>
        /// <returns><c>true</c> when the run may proceed; it must call <see cref="ExitRun"/> exactly once when it is done.</returns>
        internal bool TryEnterRun()
        {
            Interlocked.Increment(ref _runs);
            if (Volatile.Read(ref _closing) == 0)
                return true;
            ExitRun(onOrtThread: false);
            return false;
        }

        /// <summary>
        ///     Ends a run opened by <see cref="TryEnterRun"/>: the last run after a <c>Dispose</c> began releases the native
        ///     session - here, or on the .NET thread pool when <paramref name="onOrtThread"/> says the caller is the end of an
        ///     asynchronous run, which may be ONNX Runtime's own intra-op thread.
        /// </summary>
        /// <param name="onOrtThread"><c>true</c> at the end of an asynchronous run (the release must not run on ORT's thread).</param>
        internal void ExitRun(bool onOrtThread)
        {
            if (Interlocked.Decrement(ref _runs) != 0 || Volatile.Read(ref _closing) == 0)
                return;
            if (onOrtThread)
                ThreadPool.UnsafeQueueUserWorkItem(static session => session.ReleaseOnce(), this, preferLocal: false);
            else
                ReleaseOnce();
        }

        /// <summary>
        ///     Releases the session once its runs are done: the first <c>Dispose</c> refuses new runs and releases at once
        ///     when none is in flight, otherwise the last run releases; every other <c>Dispose</c> returns at once. The
        ///     finalizer (<paramref name="disposing"/> <c>false</c>) releases as ONNX Runtime does: no run can be in flight
        ///     then, because a run keeps its session reachable.
        /// </summary>
        /// <param name="disposing"><c>true</c> from <c>Dispose()</c>; <c>false</c> from the finalizer.</param>
        protected override void Dispose(bool disposing)
        {
            if (!disposing)
            {
                base.Dispose(false);
                return;
            }
            if (Interlocked.Exchange(ref _closing, 1) != 0)
                return;   // a racing (or repeated) Dispose: the first one decides, exactly once
            if (Volatile.Read(ref _runs) == 0)
                ReleaseOnce();
        }

        /// <summary>
        ///     Runs ONNX Runtime's own <c>Dispose(true)</c> once: the names, the options and the native session are released
        ///     by ORT's code, from whichever of the disposing thread, the last run and a thread-pool item gets here first.
        /// </summary>
        private void ReleaseOnce()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            base.Dispose(true);
            Interlocked.Increment(ref _releases);
        }
    }

    /// <summary>
    ///     A run's lease on its session: for an <see cref="NDInferenceSession"/>, the run counted until it ends (so its
    ///     <c>Dispose</c> waits for the run); for any other session, ONNX Runtime's own disposal flag checked once, as every
    ///     run did before (<see cref="OrtDisposalProbe"/>). A struct, so a <c>using</c> costs no allocation.
    /// </summary>
    internal readonly struct SessionLease : IDisposable
    {
        /// <summary>The leased session, or null for a session that cannot be leased (a plain <see cref="InferenceSession"/>).</summary>
        private readonly NDInferenceSession _session;

        /// <summary>Wraps a run already counted on <paramref name="session"/>.</summary>
        /// <param name="session">The session whose run count this lease holds.</param>
        private SessionLease(NDInferenceSession session) => _session = session;

        /// <summary>
        ///     Opens the lease of a run of <paramref name="verb"/> on <paramref name="session"/>, before anything of the run
        ///     reaches native code.
        /// </summary>
        /// <param name="session">The session (not null).</param>
        /// <param name="verb">The public verb, named in a refusal.</param>
        /// <returns>The lease; dispose it when the run's last native call returned (an asynchronous run: when it completed).</returns>
        /// <exception cref="ObjectDisposedException">The session has been disposed (or, for an <see cref="NDInferenceSession"/>, is being disposed).</exception>
        internal static SessionLease Enter(InferenceSession session, string verb)
        {
            if (session is NDInferenceSession leased)
            {
                if (!leased.TryEnterRun())
                    throw Disposed(verb);
                return new SessionLease(leased);
            }
            OrtDisposalProbe.ThrowIfDisposed(session, verb);
            return default;
        }

        /// <summary>
        ///     The early refusal of a verb that validates its arguments before it opens its lease (the asynchronous runs): an
        ///     <see cref="NDInferenceSession"/> whose <c>Dispose</c> began, or any other session ONNX Runtime has disposed. It
        ///     guards nothing by itself - the lease opened later is what keeps the session alive.
        /// </summary>
        /// <param name="session">The session (not null).</param>
        /// <param name="verb">The public verb, named in the refusal.</param>
        /// <exception cref="ObjectDisposedException">The session has been disposed (or, for an <see cref="NDInferenceSession"/>, is being disposed).</exception>
        internal static void ThrowIfDisposed(InferenceSession session, string verb)
        {
            if (session is NDInferenceSession leased)
            {
                if (leased.IsDisposed)
                    throw Disposed(verb);
                return;
            }
            OrtDisposalProbe.ThrowIfDisposed(session, verb);
        }

        /// <summary>The refusal of a run on an <see cref="NDInferenceSession"/> whose <c>Dispose</c> began.</summary>
        /// <param name="verb">The public verb.</param>
        /// <returns>The exception to throw.</returns>
        private static ObjectDisposedException Disposed(string verb) =>
            new ObjectDisposedException(nameof(NDInferenceSession),
                $"{verb}: the NDInferenceSession has been disposed. The runs already in flight finish, and the native session is " +
                "released after the last of them; a new run is refused. Create a new session, or keep this one until its runs are done.");

        /// <summary>Ends a synchronous run: the last one after a <c>Dispose</c> began releases the session on this thread.</summary>
        public void Dispose() => _session?.ExitRun(onOrtThread: false);

        /// <summary>Ends an asynchronous run, possibly on ONNX Runtime's intra-op thread: a release it makes runs on the thread pool.</summary>
        internal void ExitFromOrtThread() => _session?.ExitRun(onOrtThread: true);
    }
}
