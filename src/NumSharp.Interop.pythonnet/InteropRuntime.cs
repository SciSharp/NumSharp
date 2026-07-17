using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop
{
    /// <summary>
    ///     Process-wide interop session state: cached Python modules, the lifetime registries that tie
    ///     NumSharp buffers to Python objects (and vice versa), and the <see cref="PythonEngine"/>
    ///     shutdown coordination that keeps teardown crash-free.
    ///
    ///     <para><b>Lock ordering invariant (deadlock freedom):</b> the GIL is ALWAYS acquired before
    ///     the drain gate (<see cref="_drainGate"/>), never after — no thread ever waits for the GIL
    ///     while holding the gate, so an app thread that keeps the GIL long-term (e.g. never calls
    ///     <c>PythonEngine.BeginAllowThreads()</c>) can still drain inline. Release hooks that may run
    ///     on threads already holding the GIL — or on finalizer threads — never block at all:
    ///     <see cref="ImportLease.Release"/> only enqueues (lock-free) and
    ///     <see cref="ExportKeeper.Release"/> touches CLR state only.</para>
    /// </summary>
    internal static class InteropRuntime
    {
        // ---- session wiring ------------------------------------------------------------------

        private static readonly object _sessionGate = new object();
        private static int _sessionLive; // 1 once wired to the CURRENT engine session

        private static PyObject _numpy, _ctypes, _builtins, _weakref;

        /// <summary>
        ///     Set by <see cref="PythonConvert.RegisterCodec()"/>; reset at engine shutdown because
        ///     pythonnet's <c>PyObjectConversions.Reset()</c> clears all registered codecs there.
        /// </summary>
        internal static int CodecRegistered;

        /// <summary>
        ///     Throws a crisp error when the embedded engine is not running, and (re-)wires the
        ///     per-session state (shutdown handler, module cache) on first use of a session.
        /// </summary>
        internal static void EnsureEngine()
        {
            if (!PythonEngine.IsInitialized)
                throw new InvalidOperationException(
                    "Python engine is not initialized. Call PythonEngine.Initialize() first " +
                    "(after setting Runtime.PythonDLL or the PYTHONNET_PYDLL environment variable).");

            if (Volatile.Read(ref _sessionLive) == 1)
                return;

            lock (_sessionGate)
            {
                if (Volatile.Read(ref _sessionLive) == 1)
                    return;

                // pythonnet pops shutdown handlers while executing them, so this must be re-added
                // for every engine session (Initialize -> Shutdown cycle).
                PythonEngine.AddShutdownHandler(OnEngineShutdown);
                Volatile.Write(ref _engineTorndown, 0);
                Volatile.Write(ref _sessionLive, 1);
            }
        }

        /// <summary>Cached <c>numpy</c> module (per engine session). Call under the GIL.</summary>
        internal static PyObject Numpy => GetModule(ref _numpy, "numpy");

        /// <summary>Cached <c>ctypes</c> module (per engine session). Call under the GIL.</summary>
        internal static PyObject Ctypes => GetModule(ref _ctypes, "ctypes");

        /// <summary>Cached <c>builtins</c> module (per engine session). Call under the GIL.</summary>
        internal static PyObject Builtins => GetModule(ref _builtins, "builtins");

        /// <summary>Cached <c>weakref</c> module (per engine session). Call under the GIL.</summary>
        internal static PyObject Weakref => GetModule(ref _weakref, "weakref");

        private static PyObject GetModule(ref PyObject cache, string name)
        {
            var module = Volatile.Read(ref cache);
            if (module is not null)
                return module;
            // Py.Import is idempotent (CPython caches modules); racing threads may both import,
            // only one wrapper wins the cache. The loser is dropped to pythonnet's safe finalizer.
            module = Py.Import(name);
            var winner = Interlocked.CompareExchange(ref cache, module, null);
            return winner ?? module;
        }

        // ---- live-conversion registries (lock-free) --------------------------------------------

        private static readonly ConcurrentDictionary<ExportKeeper, byte> _exports = new();
        private static readonly ConcurrentDictionary<ImportLease, byte> _imports = new();
        private static int _liveExports, _liveImports;

        /// <summary>Number of NumSharp buffers currently shared with (rooted by) live Python objects.</summary>
        internal static int LiveExports => Volatile.Read(ref _liveExports);

        /// <summary>Number of Python buffers currently leased by live NumSharp views.</summary>
        internal static int LiveImports => Volatile.Read(ref _liveImports);

        internal static void TrackExport(ExportKeeper keeper)
        {
            _exports.TryAdd(keeper, 0);
            Interlocked.Increment(ref _liveExports);
        }

        internal static void OnExportReleased(ExportKeeper keeper)
        {
            if (_exports.TryRemove(keeper, out _))
                Interlocked.Decrement(ref _liveExports);
        }

        internal static void TrackImport(ImportLease lease)
        {
            _imports.TryAdd(lease, 0);
            Interlocked.Increment(ref _liveImports);
        }

        internal static void OnImportReleased(ImportLease lease)
        {
            if (_imports.TryRemove(lease, out _))
                Interlocked.Decrement(ref _liveImports);
        }

        // ---- deferred Python-side disposal ------------------------------------------------------
        //
        // ImportLease.Release is invoked by NumSharp's memory-block Disposer — potentially on the
        // finalizer thread, or on a user thread that already holds the GIL. Acquiring the GIL there
        // could deadlock against a concurrent shutdown drain (gate->GIL vs GIL->gate), so Release
        // only ENQUEUES; the actual PyBuffer_Release / decref runs here, on a thread that starts
        // out holding neither the gate nor the GIL.

        private static readonly ConcurrentQueue<ImportLease> _pendingDisposals = new();
        private static readonly object _drainGate = new object();
        private static int _drainScheduled;
        private static int _engineTorndown;

        /// <summary>Queue a lease whose NumSharp-side references are gone for Python-side disposal.</summary>
        internal static void QueueDisposal(ImportLease lease)
        {
            _pendingDisposals.Enqueue(lease);
            if (Interlocked.Exchange(ref _drainScheduled, 1) == 0)
                ThreadPool.QueueUserWorkItem(static _ => DrainPending());
        }

        /// <summary>
        ///     Dispose all queued leases. Also called at the start of every conversion so pending
        ///     buffer locks are released promptly even without ThreadPool progress.
        ///
        ///     <para><b>Ordering:</b> GIL first, gate second — never the reverse. Waiting for the GIL
        ///     while holding the gate would deadlock the whole interop against an app thread that
        ///     holds the GIL long-term (e.g. a main thread that never calls
        ///     <see cref="PythonEngine.BeginAllowThreads"/>): the inline drains those conversions run
        ///     would block on the gate forever.</para>
        /// </summary>
        internal static void DrainPending()
        {
            Interlocked.Exchange(ref _drainScheduled, 0);
            if (_pendingDisposals.IsEmpty)
                return;

            if (Volatile.Read(ref _engineTorndown) == 1 || !PythonEngine.IsInitialized)
            {
                lock (_drainGate)
                {
                    while (_pendingDisposals.TryDequeue(out var lease))
                        lease.DisposeAfterEngineDeath();
                }

                return;
            }

            // A narrow race remains here if the app calls PythonEngine.Shutdown concurrently with
            // this GIL acquisition — inherent to pythonnet embedding (PyGILState_Ensure on a dying
            // interpreter). The shutdown handler minimizes it by draining everything itself first.
            using (Py.GIL())
            lock (_drainGate)
            {
                if (Volatile.Read(ref _engineTorndown) == 1)
                {
                    while (_pendingDisposals.TryDequeue(out var lease))
                        lease.DisposeAfterEngineDeath();
                    return;
                }

                while (_pendingDisposals.TryDequeue(out var lease))
                    lease.DisposeUnderGil();
            }
        }

        /// <summary>
        ///     Runs inside <see cref="PythonEngine.Shutdown"/>, BEFORE the interpreter is finalized
        ///     (pythonnet executes shutdown handlers first), so the GIL is still available here.
        ///
        ///     Every outstanding import lease — queued or still referenced by live NDArrays — must have
        ///     its <c>PyBuffer</c> disposed NOW: pythonnet 3.0.x's <c>~PyBuffer()</c> throws when the
        ///     runtime is down, which would crash the finalizer thread later. NDArray views over Python
        ///     memory are invalid after engine shutdown (the interpreter that owned the memory is gone);
        ///     that is documented on <see cref="PythonConvert.ToNDArrayView(PyObject, bool)"/>.
        ///
        ///     Export keepers are NOT force-released here: Python <c>atexit</c>/GC still runs during
        ///     <c>Py_Finalize</c> and may touch exported arrays; their <c>weakref.finalize</c> callbacks
        ///     fire during finalization and release the NumSharp buffer refs then (CLR-only work, safe
        ///     at any point). A callback that never fires leaks the buffer — safe by construction.
        /// </summary>
        private static void OnEngineShutdown()
        {
            using (Py.GIL())          // GIL before gate — same ordering as DrainPending
            lock (_drainGate)
            {
                Volatile.Write(ref _engineTorndown, 1);

                while (_pendingDisposals.TryDequeue(out var lease))
                    lease.DisposeUnderGil();

                foreach (var lease in _imports.Keys)
                    lease.ForceReleaseUnderGil();

                DisposeModule(ref _numpy);
                DisposeModule(ref _ctypes);
                DisposeModule(ref _builtins);
                DisposeModule(ref _weakref);
            }

            // pythonnet clears all registered codecs during shutdown (PyObjectConversions.Reset),
            // so a new engine session must register ours again.
            Volatile.Write(ref CodecRegistered, 0);
            Volatile.Write(ref _sessionLive, 0);
        }

        private static void DisposeModule(ref PyObject cache)
        {
            var module = Interlocked.Exchange(ref cache, null);
            module?.Dispose();
        }
    }

    /// <summary>
    ///     Roots one NumSharp buffer for the lifetime of the Python objects viewing it.
    ///
    ///     <para>Created by the export path with its own ARC reference on the source buffer
    ///     (<see cref="IArraySlice.TryAddRef"/>), so the unmanaged memory survives even if every C#
    ///     <see cref="NDArray"/> referencing it is disposed or collected. The Python side owns the
    ///     release: a <c>weakref.finalize</c> registered on the deepest base object of the exported
    ///     numpy array (the ctypes buffer every derived numpy view chains to) invokes
    ///     <see cref="Release"/> when the LAST Python-side view is collected — including at
    ///     interpreter exit, since <c>weakref.finalize</c> guarantees an atexit pass.</para>
    ///
    ///     <para><see cref="Release"/> touches only CLR state (no GIL, no Python calls), so it is safe
    ///     from any thread at any point of the engine lifecycle, including during <c>Py_Finalize</c>.</para>
    /// </summary>
    internal sealed class ExportKeeper
    {
        private readonly NDArray _source;    // keeps the source NDArray (and its Shape/Storage) reachable
        private readonly IArraySlice _slice; // the buffer this keeper holds an ARC reference on
        private int _released;

        internal ExportKeeper(NDArray source, IArraySlice slice)
        {
            _source = source;
            _slice = slice;
        }

        /// <summary>
        ///     Idempotent. Drops this keeper's ARC reference on the NumSharp buffer; if it was the
        ///     last reference the unmanaged memory is freed synchronously on the calling thread.
        /// </summary>
        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            _slice.Release();
            InteropRuntime.OnExportReleased(this);
            GC.KeepAlive(_source);
        }
    }

    /// <summary>
    ///     Owns one Python-side buffer lease backing a NumSharp view: either a locked
    ///     <see cref="PyBuffer"/> (PEP 3118 exporters — the exporter is pinned and, for resizable
    ///     objects like <c>bytearray</c>, protected against reallocation), or a strong object
    ///     reference (<c>__array_interface__</c> path for non-contiguous numpy arrays, where CPython
    ///     offers no buffer lock but numpy's own <c>resize</c> refcheck refuses while references exist).
    ///
    ///     <para><see cref="Release"/> is wired as the <see cref="UnmanagedMemoryBlock{T}"/> dispose
    ///     hook, so it runs exactly once when the LAST NumSharp view over the memory drops its ARC
    ///     reference — via <see cref="NDArray.Dispose"/>, the NDArray finalizer, or the memory-block
    ///     Disposer finalizer. Derived views (slices of the imported view) share the same block and
    ///     therefore extend the lease automatically.</para>
    ///
    ///     <para>Release never touches Python directly (see the lock-ordering note on
    ///     <see cref="InteropRuntime"/>); actual disposal is deferred to
    ///     <see cref="InteropRuntime.DrainPending"/> or the engine-shutdown drain.</para>
    /// </summary>
    internal sealed class ImportLease
    {
        private readonly PyBuffer _buffer;  // PEP 3118 lease (null on the array-interface path)
        private readonly PyObject _holder;  // strong-ref container (null on the PyBuffer path)
        private readonly long _bytes;       // GC memory-pressure registered for this lease
        private int _released;              // NumSharp-side handoff happened (enqueue exactly once)
        private int _disposed;              // Python-side resources actually released

        internal ImportLease(PyBuffer buffer, PyObject holder, long bytes)
        {
            _buffer = buffer;
            _holder = holder;
            _bytes = bytes;
            if (_bytes > 0)
                GC.AddMemoryPressure(_bytes);
        }

        /// <summary>
        ///     NumSharp-side release hook (memory-block dispose action). Lock-free and GIL-free:
        ///     safe from finalizer threads and from threads already holding the GIL.
        /// </summary>
        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            InteropRuntime.QueueDisposal(this);
        }

        /// <summary>Actual Python-side disposal. Only called by the drains, under the GIL.</summary>
        internal void DisposeUnderGil()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            try
            {
                _buffer?.Dispose();   // PyBuffer_Release + SuppressFinalize (pythonnet)
                _holder?.Dispose();   // decref of the strong-ref container
            }
            catch
            {
                // A failed release must never take down a drain loop; the buffer object's
                // pythonnet-side state stays consistent either way.
            }
            finally
            {
                Complete();
            }
        }

        /// <summary>
        ///     Terminal path when the interpreter no longer exists: the lease cannot be released,
        ///     only defused. pythonnet 3.0.x's <c>~PyBuffer()</c> THROWS when the runtime is down
        ///     (crashing the finalizer thread), so the finalizer is suppressed and the view leaked —
        ///     the interpreter that owned the memory is gone anyway.
        /// </summary>
        internal void DisposeAfterEngineDeath()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (_buffer is not null)
                GC.SuppressFinalize(_buffer);
            // _holder: pythonnet's PyObject finalization machinery is shutdown-safe on its own.
            Complete();
        }

        /// <summary>
        ///     Engine-shutdown sweep for leases still referenced by live NDArrays: claims the
        ///     NumSharp-side handoff too, so a later <see cref="Release"/> (when those NDArrays die
        ///     after the engine) is a no-op instead of an enqueue into a dead session.
        /// </summary>
        internal void ForceReleaseUnderGil()
        {
            Interlocked.Exchange(ref _released, 1);
            DisposeUnderGil();
        }

        private void Complete()
        {
            if (_bytes > 0)
                GC.RemoveMemoryPressure(_bytes);
            InteropRuntime.OnImportReleased(this);
        }
    }
}
