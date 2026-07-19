using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
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
    internal static class PythonRuntimeInterop
    {
        // ---- session wiring ------------------------------------------------------------------

        private static readonly object _sessionGate = new object();
        private static int _sessionLive; // 1 once wired to the CURRENT engine session

        private static PyObject _numpy, _ctypes, _builtins, _weakref;

        /// <summary>
        ///     Set by <see cref="NDArrayPythonInterop.RegisterCodec()"/>; reset at engine shutdown because
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

                // A previous session may have died while Python still held exported buffers; those
                // pins belong to an interpreter that no longer exists (see the orphaned-exports
                // region). If the deferred sweep has not caught them yet — e.g. an immediate
                // Shutdown -> Initialize cycle raced its poll — release them before this session
                // starts so the leak cannot outlive the session boundary.
                ReleaseOrphanedExports();
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

        // ---- session-cached callables, attribute names & literals ------------------------------
        //
        // Every conversion used to walk dynamic dispatch (DLR call sites -> pythonnet
        // DynamicMetaObject -> GetAttr string marshaling -> Invoke) for each Python operation.
        // These caches resolve each callable / attribute-name / literal ONCE per engine session,
        // turning a conversion into direct Invoke calls on pre-resolved PyObjects. All accessors
        // must be called under the GIL (they are — only conversion bodies and the shutdown
        // handler touch them). Losing racers dispose their instance immediately (safe under the
        // GIL); winners are owned by the session and swept by OnEngineShutdown.

        private static PyObject _npEmpty, _npFrombuffer, _npArray, _npAsStrided;
        private static PyObject _ctypesCCharMul, _weakrefFinalize, _builtinsMemoryview;
        private static PyObject _trueLiteral, _falseLiteral, _strC, _strB;
        private static PyObject _nameFromAddress, _nameReshape, _nameSetflags, _nameCast, _nameTobytes,
                                _nameFormat, _nameItemsize, _nameShape, _nameCContiguous,
                                _nameArrayInterface, _nameTypestr, _nameData, _nameStrides;
        private static readonly PyObject[] _dtypeStrings = new PyObject[129]; // indexed by (int)NPTypeCode; max = Complex (128)

        /// <summary>Cached <c>numpy.empty</c>. Call under the GIL.</summary>
        internal static PyObject NpEmpty => GetCached(ref _npEmpty, static () => Numpy.GetAttr("empty"));

        /// <summary>Cached <c>numpy.frombuffer</c>. Call under the GIL.</summary>
        internal static PyObject NpFrombuffer => GetCached(ref _npFrombuffer, static () => Numpy.GetAttr("frombuffer"));

        /// <summary>Cached <c>numpy.array</c>. Call under the GIL.</summary>
        internal static PyObject NpArray => GetCached(ref _npArray, static () => Numpy.GetAttr("array"));

        /// <summary>Cached <c>numpy.lib.stride_tricks.as_strided</c>. Call under the GIL.</summary>
        internal static PyObject NpAsStrided => GetCached(ref _npAsStrided, static () =>
        {
            using var lib = Numpy.GetAttr("lib");
            using var strideTricks = lib.GetAttr("stride_tricks");
            return strideTricks.GetAttr("as_strided");
        });

        /// <summary>
        ///     Cached bound <c>ctypes.c_char.__mul__</c> — computes the sized array type
        ///     <c>(ctypes.c_char * n)</c> as a direct call (the metaclass exposes the repeat
        ///     operator as a bound <c>__mul__</c> on the type). Call under the GIL.
        /// </summary>
        internal static PyObject CCharMul => GetCached(ref _ctypesCCharMul, static () =>
        {
            using var cchar = Ctypes.GetAttr("c_char");
            return cchar.GetAttr("__mul__");
        });

        /// <summary>Cached <c>weakref.finalize</c>. Call under the GIL.</summary>
        internal static PyObject WeakrefFinalize => GetCached(ref _weakrefFinalize, static () => Weakref.GetAttr("finalize"));

        /// <summary>Cached <c>builtins.memoryview</c>. Call under the GIL.</summary>
        internal static PyObject BuiltinsMemoryview => GetCached(ref _builtinsMemoryview, static () => Builtins.GetAttr("memoryview"));

        /// <summary>Cached Python <c>True</c> (counterpart of <see cref="FalseLiteral"/> for <c>setflags</c>).</summary>
        internal static PyObject TrueLiteral => GetCached(ref _trueLiteral, static () => true.ToPython());

        /// <summary>Cached Python <c>False</c> (first positional of <c>ndarray.setflags</c> is <c>write</c>).</summary>
        internal static PyObject FalseLiteral => GetCached(ref _falseLiteral, static () => false.ToPython());

        /// <summary>Cached <c>'C'</c> (memoryview.tobytes order).</summary>
        internal static PyObject StrC => GetCached(ref _strC, static () => new PyString("C"));

        /// <summary>Cached <c>'B'</c> (memoryview.cast format).</summary>
        internal static PyObject StrB => GetCached(ref _strB, static () => new PyString("B"));

        // Attribute names / dict keys — cached PyStrings avoid the per-call UTF-8 marshal +
        // unicode allocation of the string-based GetAttr/indexer overloads.
        internal static PyObject NameFromAddress => GetCached(ref _nameFromAddress, static () => new PyString("from_address"));
        internal static PyObject NameReshape => GetCached(ref _nameReshape, static () => new PyString("reshape"));
        internal static PyObject NameSetflags => GetCached(ref _nameSetflags, static () => new PyString("setflags"));
        internal static PyObject NameCast => GetCached(ref _nameCast, static () => new PyString("cast"));
        internal static PyObject NameTobytes => GetCached(ref _nameTobytes, static () => new PyString("tobytes"));
        internal static PyObject NameFormat => GetCached(ref _nameFormat, static () => new PyString("format"));
        internal static PyObject NameItemsize => GetCached(ref _nameItemsize, static () => new PyString("itemsize"));
        internal static PyObject NameShape => GetCached(ref _nameShape, static () => new PyString("shape"));
        internal static PyObject NameCContiguous => GetCached(ref _nameCContiguous, static () => new PyString("c_contiguous"));
        internal static PyObject NameArrayInterface => GetCached(ref _nameArrayInterface, static () => new PyString("__array_interface__"));
        internal static PyObject NameTypestr => GetCached(ref _nameTypestr, static () => new PyString("typestr"));
        internal static PyObject NameData => GetCached(ref _nameData, static () => new PyString("data"));
        internal static PyObject NameStrides => GetCached(ref _nameStrides, static () => new PyString("strides"));

        /// <summary>
        ///     Cached PyString of <see cref="NDArrayPythonInterop.ToNumpyDtypeStr"/> for <paramref name="tc"/>
        ///     (session-owned — callers must NOT dispose it). Call under the GIL.
        /// </summary>
        internal static PyObject DtypeString(NPTypeCode tc)
        {
            ref PyObject cache = ref _dtypeStrings[(int)tc];
            var v = Volatile.Read(ref cache);
            if (v is not null)
                return v;
            return PublishCached(ref cache, new PyString(NDArrayPythonInterop.ToNumpyDtypeStr(tc)));
        }

        private static PyObject GetCached(ref PyObject cache, Func<PyObject> resolve)
        {
            var v = Volatile.Read(ref cache);
            if (v is not null)
                return v;
            return PublishCached(ref cache, resolve());
        }

        private static PyObject PublishCached(ref PyObject cache, PyObject fresh)
        {
            var winner = Interlocked.CompareExchange(ref cache, fresh, null);
            if (winner is null)
                return fresh;
            fresh.Dispose();   // racing loser — safe to decref immediately, every caller holds the GIL
            return winner;
        }

        private static void DisposeSessionCache()
        {
            DisposeModule(ref _npEmpty);
            DisposeModule(ref _npFrombuffer);
            DisposeModule(ref _npArray);
            DisposeModule(ref _npAsStrided);
            DisposeModule(ref _ctypesCCharMul);
            DisposeModule(ref _weakrefFinalize);
            DisposeModule(ref _builtinsMemoryview);
            DisposeModule(ref _trueLiteral);
            DisposeModule(ref _falseLiteral);
            DisposeModule(ref _strC);
            DisposeModule(ref _strB);
            DisposeModule(ref _nameFromAddress);
            DisposeModule(ref _nameReshape);
            DisposeModule(ref _nameSetflags);
            DisposeModule(ref _nameCast);
            DisposeModule(ref _nameTobytes);
            DisposeModule(ref _nameFormat);
            DisposeModule(ref _nameItemsize);
            DisposeModule(ref _nameShape);
            DisposeModule(ref _nameCContiguous);
            DisposeModule(ref _nameArrayInterface);
            DisposeModule(ref _nameTypestr);
            DisposeModule(ref _nameData);
            DisposeModule(ref _nameStrides);
            for (int i = 0; i < _dtypeStrings.Length; i++)
                DisposeModule(ref _dtypeStrings[i]);
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

        // ---- orphaned exports (engine death) ----------------------------------------------------
        //
        // Export pins are normally released by a Python-side weakref.finalize when the last view
        // dies. pythonnet's PythonEngine.Shutdown, however, performs NO Python atexit pass (probed
        // on 3.0.x: neither atexit.register callbacks nor weakref.finalize's guaranteed exit pass
        // execute), so the finalize callbacks of exports STILL referenced by Python when Shutdown
        // begins can never fire — without a sweep, each such pin leaks its NumSharp buffer for the
        // rest of the process.
        //
        // The shutdown handler therefore SNAPSHOTS the live keepers, and this region releases them
        // only once the engine has provably finished dying (PythonEngine.IsInitialized went false,
        // or a new session was wired — either proves the old interpreter, and every Python view it
        // held, is gone). Releasing DURING teardown would be premature: interpreter code that runs
        // while dying (module teardown, __del__ during a future pythonnet's real Py_Finalize GC
        // pass) may still read exported buffers. Release itself is CLR-only and idempotent, so a
        // finalize callback that somehow DID fire earlier makes the sweep a harmless no-op.

        private static readonly List<ExportKeeper> _orphanedExports = new();

        /// <summary>Snapshot every live export keeper for the post-shutdown sweep. Called by the
        /// shutdown handler under GIL+gate; the dying interpreter still owns views of them HERE.</summary>
        private static void CaptureOrphanedExports()
        {
            lock (_orphanedExports)
            {
                foreach (var keeper in _exports.Keys)
                    _orphanedExports.Add(keeper);
            }
        }

        /// <summary>
        ///     Background waiter scheduled by the shutdown handler: polls (with backoff) until the
        ///     engine has fully finished shutting down — <see cref="PythonEngine.IsInitialized"/>
        ///     flips false at the very END of <see cref="PythonEngine.Shutdown"/> — or a new session
        ///     was wired (an Initialize can only follow a completed Shutdown). Both prove no Python
        ///     code can ever read the exported buffers again, making the release safe.
        /// </summary>
        private static void SweepOrphanedExportsWhenEngineDead()
        {
            int delay = 1;
            while (PythonEngine.IsInitialized && Volatile.Read(ref _sessionLive) == 0)
            {
                Thread.Sleep(delay);
                if (delay < 50)
                    delay *= 2;
            }

            ReleaseOrphanedExports();
        }

        /// <summary>
        ///     Drop the ARC pins of all snapshot keepers (CLR-only; frees each buffer if the pin was
        ///     its last reference). Idempotent and thread-safe: the poller and a re-initializing
        ///     session may both call it; <see cref="ExportKeeper.Release"/> is single-shot per keeper.
        /// </summary>
        internal static void ReleaseOrphanedExports()
        {
            ExportKeeper[] orphans;
            lock (_orphanedExports)
            {
                if (_orphanedExports.Count == 0)
                    return;
                orphans = _orphanedExports.ToArray();
                _orphanedExports.Clear();
            }

            foreach (var keeper in orphans)
                keeper.Release();
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
        ///     that is documented on <see cref="NDArrayPythonInterop.ToNDArrayView(PyObject, bool, bool?)"/>.
        ///
        ///     Export keepers are NOT released here — but not for the reason one might hope. Their
        ///     <c>weakref.finalize</c> callbacks will never run: pythonnet's <c>Shutdown</c> performs
        ///     no Python atexit pass at all (probed on 3.0.x — neither <c>atexit.register</c> callbacks
        ///     nor <c>weakref.finalize</c>'s guaranteed exit pass execute), so every export still
        ///     referenced by Python would simply leak. Releasing them HERE is not safe either: the
        ///     interpreter is still alive during the handler and may yet read exported memory while it
        ///     dies. The resolution is the orphaned-exports region above: snapshot now, release the
        ///     pins as soon as the engine is provably gone.
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

                CaptureOrphanedExports();

                DisposeSessionCache();
                DisposeModule(ref _numpy);
                DisposeModule(ref _ctypes);
                DisposeModule(ref _builtins);
                DisposeModule(ref _weakref);
            }

            // pythonnet clears all registered codecs during shutdown (PyObjectConversions.Reset),
            // so a new engine session must register ours again.
            Volatile.Write(ref CodecRegistered, 0);
            Volatile.Write(ref _sessionLive, 0);

            bool anyOrphans;
            lock (_orphanedExports)
                anyOrphans = _orphanedExports.Count > 0;
            if (anyOrphans)
                ThreadPool.QueueUserWorkItem(static _ => SweepOrphanedExportsWhenEngineDead());
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
    ///     <see cref="Release"/> when the LAST Python-side view is collected. CPython's documented
    ///     at-exit finalize pass does NOT happen under embedding — pythonnet's <c>Shutdown</c> runs
    ///     no atexit (probed) — so keepers still pinned when the engine dies are swept by
    ///     <see cref="PythonRuntimeInterop"/>'s orphaned-exports path right after shutdown completes.</para>
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
            PythonRuntimeInterop.OnExportReleased(this);
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
    ///     <see cref="PythonRuntimeInterop"/>); actual disposal is deferred to
    ///     <see cref="PythonRuntimeInterop.DrainPending"/> or the engine-shutdown drain.</para>
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
            PythonRuntimeInterop.QueueDisposal(this);
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
            PythonRuntimeInterop.OnImportReleased(this);
        }
    }
}
