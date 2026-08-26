using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     Ambient reclamation scope for transient <see cref="NDArray"/> intermediates — the
    ///     library's standard way to make a composition method eagerly return its temporaries'
    ///     pooled buffers instead of waiting on the finalizer (see <c>DISPOSAL-GUIDELINES.md</c>).
    ///     Every <see cref="NDArray"/> constructed on the current thread while a scope is open is
    ///     tracked by it; disposing the scope disposes every tracked array that was not yielded
    ///     via <see cref="Returns{T}(T)"/>. Tracked disposal is ordinary ARC release (the buffer
    ///     frees only at refcount 0), so releasing a base whose view was yielded never corrupts —
    ///     the same safety as a hand-written <c>Dispose</c>, with the bookkeeping automated.
    /// </summary>
    /// <remarks>
    ///     <para><b>Usage.</b> Open at the top of a boundary method, keep the original body, and
    ///     route every egress through <see cref="Returns{T}(T)"/>:</para>
    ///     <code>
    ///     using var scope = NDScope.Open();
    ///     var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
    ///     var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
    ///     return scope.Returns((b1 &amp; b2).MakeGeneric&lt;bool&gt;());
    ///     </code>
    ///     <para><b>Ownership rules become structural.</b> Inputs were constructed BEFORE the
    ///     scope opened, so they are never tracked — a passthrough (<c>ravel()</c>/
    ///     <c>atleast_2d()</c> returning its operand, a caller-supplied <c>@out</c>) needs no
    ///     guard, and <see cref="Returns{T}(T)"/> on such an array is a provable no-op (rule R2
    ///     for free). The return value is the one egress and is yielded (rule R1); an
    ///     <c>out</c>-parameter egress is written as <c>result = scope.Returns(temp);</c>.
    ///     Everything else — however deep the helper-call tree below the scope — is reclaimed,
    ///     on exception paths included.</para>
    ///     <para><b>Nesting.</b> Scopes nest per thread; <see cref="Returns{T}(T)"/> re-tracks the
    ///     yielded array into the parent scope, so an enclosing scope still reclaims an inner
    ///     call's result if the caller drops it.</para>
    ///     <para><b>Threading.</b> The current scope is <c>[ThreadStatic]</c>: a scope is opened,
    ///     used and disposed on ONE thread (asserted in debug builds; a HAND-WRITTEN scope must not
    ///     span <c>await</c> — the <c>[NDScoped]</c> weaver's state-machine seam is the one thing
    ///     that may carry a scope across suspensions, because it uninstalls the scope before each
    ///     continuation is scheduled and re-installs it on the resuming thread via
    ///     <see cref="OpenOrResume"/>/<see cref="Suspend"/>). Arrays constructed on other
    ///     threads (parallel kernel workers) see no scope and fall back to the finalizer
    ///     backstop — safe, just not eagerly reclaimed; a parallel region that wants eager
    ///     reclamation opens its own scope inside each worker body.</para>
    ///     <para><b>Granularity.</b> Scope a CALL, not a caller loop: temps a scope holds are all
    ///     alive simultaneously, and batch-disposing thousands of same-size buffers overflows
    ///     their pool bucket (the excess is freed, not pooled). Hot loops should still
    ///     <c>using</c> the results they receive — the two-audience contract is unchanged.</para>
    /// </remarks>
    public sealed class NDScope : IDisposable
    {
        [ThreadStatic] private static NDScope t_current;
        [ThreadStatic] private static NDScope t_pool;   // single-slot per-thread free list

        /// <summary>Tracked-list capacity above which a pooled scope trims before reuse.</summary>
        private const int TrimCapacity = 512;

        private NDScope _parent;
        private List<NDArray> _tracked;
        private bool _disposed;
        private int _threadId;

        // Set by ReturnsTask/ReturnsValueTask when the returned task is INCOMPLETE: disposal is
        // handed to a completion continuation instead of the method's finally, so tracked temps a
        // still-running awaited callee holds are not reclaimed under it. Read only by
        // CloseUnlessDeferred on the OPENING thread (written earlier on that same thread — the
        // completion continuation never writes it), so the flag is race-free by construction.
        private bool _deferred;

        private NDScope()
        {
            _tracked = new List<NDArray>(16);
        }

        /// <summary>The innermost open scope on the current thread, or <c>null</c>.</summary>
        internal static NDScope Current => t_current;

        /// <summary>Opens a scope on the current thread; nests (the previous scope resumes on dispose).</summary>
        public static NDScope Open()
        {
            var s = t_pool;
            if (s is null)
                s = new NDScope();
            else
            {
                t_pool = null;
                s._disposed = false;
            }

            s._threadId = Environment.CurrentManagedThreadId;
            s._deferred = false;
            s._parent = t_current;
            t_current = s;
            return s;
        }

        /// <summary>
        ///     Constructor hook (called from <c>NDArray.InitializeArc</c>, the single funnel every
        ///     concrete ctor passes): tracks a freshly constructed array into the innermost open
        ///     scope, if any. No scope open — no cost beyond one thread-static read.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Track(NDArray nd)
        {
            var s = t_current;
            if (s is null)
                return;
            nd.TrackingScope = s;
            nd.TrackingIndex = s._tracked.Count;
            s._tracked.Add(nd);
        }

        /// <summary>
        ///     Marks <paramref name="nd"/> as this scope's yielded result (rule R1: never dispose
        ///     what you return): unregisters it here and re-tracks it into the parent scope, so an
        ///     enclosing scope still reclaims it if the caller drops it. An array this scope never
        ///     tracked — an input passthrough, a caller-owned <c>@out</c> — passes through as a
        ///     no-op, which is what makes "wrap every egress" a safe blanket rule. Also the egress
        ///     call for <c>out</c>-parameter assignments: <c>result = scope.Returns(temp);</c>.
        /// </summary>
        /// <returns><paramref name="nd"/>, with its static type preserved.</returns>
        public T Returns<T>(T nd) where T : NDArray
        {
            Debug.Assert(_threadId == Environment.CurrentManagedThreadId,
                "NDScope.Returns must run on the thread that opened the scope.");
            if (nd is null || !ReferenceEquals(nd.TrackingScope, this))
                return nd;   // not ours (never tracked, or owned elsewhere): pass through

            _tracked[nd.TrackingIndex] = null;   // O(1) unregister; Dispose skips holes
            var parent = _parent;
            if (parent is null)
            {
                nd.TrackingScope = null;
            }
            else
            {
                nd.TrackingScope = parent;
                nd.TrackingIndex = parent._tracked.Count;
                parent._tracked.Add(nd);
            }

            return nd;
        }

        /// <summary>Yields every element of a tuple-style result (nonzero, meshgrid, split, …).</summary>
        public T[] Returns<T>(T[] nds) where T : NDArray
        {
            if (nds is not null)
                for (int i = 0; i < nds.Length; i++)
                    Returns(nds[i]);
            return nds;
        }

        // ---- ValueTuple egress (the shape NumPy's factorisations and modf/average/polydiv return) ----
        // A method returning `(NDArray, NDArray[, …])` has more than one egress, so the weaver (and a
        // hand-written scope) yields EACH component. A null component — an omitted `returned`/`compute_uv`
        // output — is a safe no-op. The tuple is returned unchanged (its references are re-parented, not
        // rewritten), so `return scope.Returns((q, r));` reads exactly like the single-array form. The
        // [NDScoped] weaver targets these overloads by generic arity (see ScopeWeaver.ResolveRefs).

        /// <summary>Yields both components of a two-array tuple result (e.g. <c>modf</c>, <c>polydiv</c>, <c>qr</c>, <c>eig</c>, <c>slogdet</c>, <c>average(returned)</c>).</summary>
        public (T1, T2) Returns<T1, T2>((T1, T2) tuple)
            where T1 : NDArray
            where T2 : NDArray
        {
            Returns(tuple.Item1);
            Returns(tuple.Item2);
            return tuple;
        }

        /// <summary>Yields all three components of a three-array tuple result (e.g. <c>svd</c>).</summary>
        public (T1, T2, T3) Returns<T1, T2, T3>((T1, T2, T3) tuple)
            where T1 : NDArray
            where T2 : NDArray
            where T3 : NDArray
        {
            Returns(tuple.Item1);
            Returns(tuple.Item2);
            Returns(tuple.Item3);
            return tuple;
        }

        /// <summary>Yields all four components of a four-array tuple result (e.g. <c>lstsq</c>).</summary>
        public (T1, T2, T3, T4) Returns<T1, T2, T3, T4>((T1, T2, T3, T4) tuple)
            where T1 : NDArray
            where T2 : NDArray
            where T3 : NDArray
            where T4 : NDArray
        {
            Returns(tuple.Item1);
            Returns(tuple.Item2);
            Returns(tuple.Item3);
            Returns(tuple.Item4);
            return tuple;
        }

        /// <summary>
        ///     Yields every <see cref="NDArray"/> in a tuple result of ANY arity (up to 8) and ANY mix —
        ///     the general <see cref="ITuple"/> egress that also covers a reference-type <see cref="System.Tuple"/>.
        ///     Components that are NOT an <see cref="NDArray"/> (a scalar, a count, …) are skipped; a null
        ///     tuple is a no-op. This is the weaver's egress for a tuple the strongly-typed overloads above
        ///     don't cover (arity 5–8, a non-NDArray component, or a boxed/reference tuple). It indexes
        ///     through <see cref="ITuple"/>, so value-type components box — negligible on the once-per-call
        ///     return path; hand-scope a hot all-NDArray tuple through the typed overloads to avoid it.
        /// </summary>
        public ITuple Returns(ITuple tuple)
        {
            if (tuple is not null)
                for (int i = 0; i < tuple.Length; i++)
                    if (tuple[i] is NDArray nd)
                        Returns(nd);
            return tuple;
        }

        // ---- Lower-layer buffer egress (a returned IArraySlice / UnmanagedStorage NOT wrapped in an NDArray) ----
        //
        // The scope's reclamation unit is the NDArray, which owns the ONE counted ARC reference on its
        // buffer; a bare IArraySlice / UnmanagedStorage (e.g. from GetData(), or an intermediate NDArray's
        // Storage) is an UNCOUNTED alias (see the ARC contract: ~NDArray ABANDONS, only deterministic
        // Dispose/NDScope Release eagerly frees at refcount 0, "asserting no alias outlives"). So a boundary
        // method that RETURNS a bare buffer would have it freed out from under the caller the moment the
        // scope Releases the intermediate NDArray that shares it. Yielding it here takes a counted reference
        // (TryAddRef) so the scope's Release can no longer reach 0 — the buffer survives — and that reference
        // is deliberately abandoned (never Released), so the block's finalizer reclaims it on unreachability,
        // the same non-deterministic backstop a bare buffer already relies on. A null buffer is a no-op.

        /// <summary>Protects a returned bare <see cref="IArraySlice"/> from this scope's reclamation of an NDArray that shares its buffer.</summary>
        public IArraySlice Returns(IArraySlice slice)
        {
            slice?.TryAddRef();
            return slice;
        }

        /// <summary>Protects a returned bare <see cref="UnmanagedStorage"/> from this scope's reclamation of an NDArray that shares its buffer.</summary>
        public UnmanagedStorage Returns(UnmanagedStorage storage)
        {
            storage?.InternalArray?.TryAddRef();
            return storage;
        }

        /// <summary>
        ///     Permanently removes <paramref name="nd"/> from whatever scope tracks it WITHOUT
        ///     re-tracking into a parent — for arrays that must outlive every scope (an array
        ///     being cached into a static / long-lived field from inside a scoped call). Must run
        ///     on the scope's owning thread (it always does: detachment happens at the caching
        ///     site, on the constructing thread). No-op for untracked arrays.
        /// </summary>
        public static void Detach(NDArray nd)
        {
            var s = nd?.TrackingScope;
            if (s is null)
                return;
            Debug.Assert(s._threadId == Environment.CurrentManagedThreadId,
                "NDScope.Detach must run on the thread that owns the array's scope.");
            s._tracked[nd.TrackingIndex] = null;
            nd.TrackingScope = null;
        }

        /// <summary>
        ///     Adopts <paramref name="nd"/> into the CURRENT (innermost) scope, so it is reclaimed
        ///     at that scope's exit unless yielded — the inverse of <see cref="Detach"/>. For an
        ///     array the scope did not construct: one a caller received before opening its scope
        ///     (the hot-loop pattern — attach instead of a per-result <c>using</c>), or one built
        ///     on another thread and handed over. Moves the array if some other scope tracked it
        ///     (the old registration is cleared first — an array is owned by at most ONE scope).
        ///     No-op when no scope is open or the array is already tracked here. Must run on the
        ///     thread that owns both the current scope and any previous tracking scope.
        /// </summary>
        public static void Attach(NDArray nd)
        {
            var s = t_current;
            if (nd is null || s is null || ReferenceEquals(nd.TrackingScope, s))
                return;
            var old = nd.TrackingScope;
            if (old is not null)
            {
                Debug.Assert(old._threadId == Environment.CurrentManagedThreadId,
                    "NDScope.Attach must move an array on the thread that owns its current scope.");
                old._tracked[nd.TrackingIndex] = null;   // clear the stale slot: one owner only
            }
            nd.TrackingScope = s;
            nd.TrackingIndex = s._tracked.Count;
            s._tracked.Add(nd);
        }

        /// <summary>Disposes every tracked non-yielded array and reinstates the parent scope.</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Debug.Assert(_threadId == Environment.CurrentManagedThreadId,
                "NDScope must be disposed on the thread that opened it.");

            // Unlink from the thread's scope stack. In-order disposal — the weaver's
            // try/finally and every `using` — is the fast path: this IS the innermost
            // scope, so pop to the parent (new constructions during the sweep, of which
            // there are none in Dispose paths, then track into the parent, never into us).
            // A hand-managed scope disposed OUT OF ORDER (a public IDisposable can be) is
            // spliced out of the middle of the chain instead, so t_current is never left
            // pointing at — or through — a disposed scope. A SUSPENDED scope (async
            // state-machine seam) was already unlinked, so the splice walk finds nothing
            // and the chain is untouched.
            UnlinkFromChain();
            _parent = null;

            var list = _tracked;
            for (int i = 0; i < list.Count; i++)
            {
                var nd = list[i];
                if (nd is null)
                    continue;          // yielded via Returns / Detach
                nd.TrackingScope = null;
                nd.Dispose();
            }

            list.Clear();
            if (list.Capacity > TrimCapacity)
                _tracked = new List<NDArray>(16);   // don't let one giant call pin a huge list

            if (t_pool is null)
                t_pool = this;
        }

        /// <summary>
        ///     Removes this scope from the CURRENT thread's scope chain (innermost pop, or an
        ///     out-of-order splice), leaving <c>t_current</c> never pointing at — or through — it.
        ///     Shared by <see cref="Dispose"/> and <see cref="Suspend"/>; a scope that is not in
        ///     this thread's chain at all (already suspended) is a no-op walk.
        /// </summary>
        private void UnlinkFromChain()
        {
            if (ReferenceEquals(t_current, this))
            {
                t_current = _parent;
            }
            else
            {
                for (var s = t_current; s != null; s = s._parent)
                {
                    if (ReferenceEquals(s._parent, this))
                    {
                        s._parent = _parent;
                        break;
                    }
                }
            }
        }

        // ---- State-machine seam (the [NDScoped] weaver's async/iterator egress) --------------------
        //
        // An async method compiles to a state machine whose MoveNext runs once per synchronous
        // SEGMENT, each possibly on a different thread — so a thread-static scope cannot simply span
        // the method. The weaver therefore gives the state machine ONE scope for the WHOLE logical
        // invocation, held in a weaver-added field (the "slot"): OpenOrResume installs it at every
        // MoveNext entry, Suspend UNINSTALLS it right before the builder schedules a continuation
        // (after that call the continuation may already be running on another thread — suspending in
        // the finally would race it, which is why the seam is a pre-schedule call), and
        // DisposeSlot/ExitIterator reclaim at completion. Tracked temps thus stay ALIVE across an
        // await — an in-flight awaited callee may still be using them — and are reclaimed when the
        // async method (not the segment) finishes. These members are PUBLIC because woven consumer
        // assemblies call them cross-assembly (the same reason INDArrayCarrier is public); they are
        // weaver infrastructure, not an API to call by hand.

        /// <summary>
        ///     State-machine prologue (weaver seam): installs the invocation's scope on the current
        ///     thread — opening a fresh one into <paramref name="slot"/> on the first MoveNext, and
        ///     RE-INSTALLING the suspended one (re-stamping its owning thread; segments of one state
        ///     machine are sequenced, so the handoff is race-free) on every resumption.
        /// </summary>
        public static NDScope OpenOrResume(ref NDScope slot)
        {
            var s = slot;
            if (s is null || s._disposed)
            {
                s = Open();
                slot = s;
            }
            else
                Resume(s);

            return s;
        }

        private static void Resume(NDScope s)
        {
            s._threadId = Environment.CurrentManagedThreadId;
            s._parent = t_current;
            t_current = s;
        }

        /// <summary>
        ///     Uninstalls <paramref name="scope"/> from the current thread WITHOUT disposing it
        ///     (weaver seam): emitted immediately BEFORE the builder's
        ///     <c>Await[Unsafe]OnCompleted</c> — once that call is made the continuation may already
        ///     be resuming on another thread, so the scope must be off this thread's chain first.
        ///     Everything tracked stays alive until the state machine completes. Also the deferral
        ///     seam of <see cref="ReturnsTask{TResult}(Task{TResult})"/>. No-op for null/disposed.
        /// </summary>
        public static void Suspend(NDScope scope)
        {
            if (scope is null || scope._disposed)
                return;
            Debug.Assert(scope._threadId == Environment.CurrentManagedThreadId,
                "NDScope.Suspend must run on the thread the scope is installed on.");
            scope.UnlinkFromChain();
            scope._parent = null;
        }

        /// <summary>
        ///     State-machine completion (weaver seam): disposes the slot's scope and clears the slot
        ///     — the async SetResult/SetException exit, the iterator's final <c>false</c>, and the
        ///     iterator enumerator's <c>Dispose()</c> (mid-iteration abandonment, which may land on
        ///     a different thread than the last MoveNext; that call is sequenced after it, so the
        ///     thread stamp is re-taken rather than asserted). No-op for an empty slot.
        /// </summary>
        public static void DisposeSlot(ref NDScope slot)
        {
            var s = slot;
            if (s is null)
                return;
            slot = null;
            s._threadId = Environment.CurrentManagedThreadId;
            s.Dispose();
        }

        /// <summary>
        ///     Iterator-yield exit (weaver seam): a <c>MoveNext</c> that produced a value
        ///     (<paramref name="hasMore"/>) suspends the invocation scope for the next resumption; a
        ///     finished one disposes it. For async iterators the weaver emits this BEFORE the
        ///     promise's <c>SetResult(hasMore)</c> — the consumer can re-enter MoveNext the instant
        ///     that signal lands, so the scope must already be off this thread.
        /// </summary>
        public static void ExitIterator(ref NDScope slot, bool hasMore)
        {
            if (hasMore)
                Suspend(slot);
            else
                DisposeSlot(ref slot);
        }

        /// <summary>
        ///     Task-returning-method epilogue (weaver seam): the finally of a woven method whose
        ///     return is <c>Task</c>-like — disposes unless <see cref="ReturnsTask{TResult}(Task{TResult})"/>/
        ///     <see cref="ReturnsValueTask{TResult}(ValueTask{TResult})"/> deferred disposal to the
        ///     task's completion (the exception path never defers, so a throw still reclaims eagerly).
        /// </summary>
        public static void CloseUnlessDeferred(NDScope scope)
        {
            if (scope is not null && !scope._deferred)
                scope.Dispose();
        }

        // ---- Task-shaped egress (a NON-async method returning Task/ValueTask) ----------------------
        //
        // A synchronous method returning a task has TWO reclamation moments to reconcile: its own
        // return (where the scope would normally dispose) and the task's completion (until which an
        // in-flight callee may still be USING tracked temps passed to it — `return BarAsync(t)`).
        // A task already completed at the return is the synchronous case: yield its result now, let
        // the finally dispose. An INCOMPLETE task defers: the scope is suspended off the thread and
        // a completion continuation yields the result (so the caller never receives a reclaimed
        // array) and only then disposes — after the in-flight work is done with the temps. The
        // continuation is the only writer after the method returns, so there is no concurrent access
        // to the tracked list; a task that never completes pins the scope until the task itself is
        // collectable, whereupon the arrays fall to the ordinary finalizer backstop.

        /// <summary>
        ///     Yields a task-shaped result (weaver seam for <c>[NDScoped]</c> NON-async methods
        ///     returning <see cref="Task{TResult}"/>): a completed task's result is yielded
        ///     immediately; an incomplete one defers BOTH the yield and this scope's disposal to the
        ///     task's completion. Returns the task unchanged.
        /// </summary>
        public Task<TResult> ReturnsTask<TResult>(Task<TResult> task)
        {
            if (task is null)
                return null;
            if (task.IsCompleted)
            {
                if (task.Status == TaskStatus.RanToCompletion)
                    YieldBoxed(task.Result);
                return task;
            }

            _deferred = true;
            Suspend(this);
            task.ContinueWith(static (t, state) =>
            {
                var scope = (NDScope)state;
                scope._threadId = Environment.CurrentManagedThreadId;
                if (t.Status == TaskStatus.RanToCompletion)
                    scope.YieldBoxed(t.Result);
                scope.Dispose();
            }, this, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            return task;
        }

        /// <summary>
        ///     Task egress without a DECLARED result (weaver seam for <c>[NDScoped]</c> NON-async
        ///     methods returning bare <see cref="Task"/>): defers this scope's disposal to the
        ///     task's completion when it is still running — the in-flight work may hold tracked
        ///     temps. The runtime task may still be a <see cref="Task{TResult}"/> UP-CAST to
        ///     <c>Task</c> (<c>Task&lt;T&gt; : Task</c> is implicit, so
        ///     <c>[NDScoped] Task M() =&gt; ComputeAsync();</c> compiles) — a caller can recover
        ///     that result by casting back, so a carried result is sniffed and yielded rather than
        ///     handed back disposed.
        /// </summary>
        public Task ReturnsTask(Task task)
        {
            if (task is null)
                return null;
            if (task.IsCompleted)
            {
                if (task.Status == TaskStatus.RanToCompletion)
                    YieldCarriedResult(task);
                return task;
            }

            _deferred = true;
            Suspend(this);
            task.ContinueWith(static (t, state) =>
            {
                var scope = (NDScope)state;
                scope._threadId = Environment.CurrentManagedThreadId;
                if (t.Status == TaskStatus.RanToCompletion)
                    scope.YieldCarriedResult(t);
                scope.Dispose();
            }, this, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            return task;
        }

        /// <summary>
        ///     Yields the result hiding inside a <see cref="Task{TResult}"/> that reached this
        ///     scope up-cast to plain <see cref="Task"/>. The result type is erased at the
        ///     call-site, so it is recovered reflectively — acceptable because this runs at most
        ///     once per plain-<c>Task</c>-declared call, and only for a generic runtime task (an
        ///     <c>async Task</c> method's internal <c>Task&lt;VoidTaskResult&gt;</c> passes through
        ///     <see cref="YieldBoxed"/> with nothing to match).
        /// </summary>
        private void YieldCarriedResult(Task task)
        {
            var t = task.GetType();
            if (!t.IsGenericType)
                return;
            var result = t.GetProperty(nameof(Task<object>.Result))?.GetValue(task);
            if (result is not null)
                YieldBoxed(result);
        }

        /// <summary>
        ///     <see cref="ValueTask{TResult}"/> egress (weaver seam). A ValueTask is single-consumption
        ///     — observing its result or registering a continuation would corrupt the caller's own
        ///     await — so the scope first <see cref="ValueTask{TResult}.Preserve"/>s it and RETURNS
        ///     THE PRESERVED task (multi-observable by contract; a plain-value or Task-backed
        ///     ValueTask preserves to itself at no cost). The preserved form is then handled exactly
        ///     like <see cref="ReturnsTask{TResult}(Task{TResult})"/>.
        /// </summary>
        public ValueTask<TResult> ReturnsValueTask<TResult>(ValueTask<TResult> task)
        {
            var preserved = task.Preserve();
            if (preserved.IsCompleted)
            {
                if (preserved.IsCompletedSuccessfully)
                    YieldBoxed(preserved.Result);
                return preserved;
            }

            ReturnsTask<TResult>(preserved.AsTask());
            return preserved;
        }

        /// <summary>Bare <see cref="ValueTask"/> egress (weaver seam) — the resultless twin of <see cref="ReturnsValueTask{TResult}(ValueTask{TResult})"/>.</summary>
        public ValueTask ReturnsValueTask(ValueTask task)
        {
            var preserved = task.Preserve();
            if (!preserved.IsCompleted)
                ReturnsTask(preserved.AsTask());
            return preserved;
        }

        /// <summary>
        ///     Late-bound egress dispatch for a task's completed result, whose static type is erased
        ///     by the time it is observable: routes to the same <c>Returns</c>/<c>YieldTo</c> family
        ///     the weaver emits for direct returns. Boxing is once-per-call on a completion path.
        /// </summary>
        private void YieldBoxed(object value)
        {
            switch (value)
            {
                case null:
                    break;
                case NDArray nd:
                    Returns(nd);
                    break;
                case NDArray[] arr:
                    Returns(arr);
                    break;
                case INDArrayCarrier carrier:
                    carrier.YieldTo(this);
                    break;
                case ITuple tuple:
                    Returns(tuple);
                    break;
                case IArraySlice slice:
                    Returns(slice);
                    break;
                case UnmanagedStorage storage:
                    Returns(storage);
                    break;
            }
        }
    }

    /// <summary>
    ///     Opt-in seam a tuple-standin result struct (<see cref="np.UniqueResult"/>,
    ///     <see cref="np.MeshgridResult"/>, <see cref="PolyfitResult"/>, …) implements so the
    ///     <see cref="NDScopedAttribute"/> weaver can weave a boundary method that RETURNS it:
    ///     <see cref="YieldTo"/> hands every <see cref="NDArray"/> the struct carries back to the ambient
    ///     scope (via <see cref="NDScope.Returns{T}(T)"/>), so the method's temporaries are reclaimed
    ///     while its results survive.
    /// </summary>
    /// <remarks>
    ///     Implemented EXPLICITLY (the member stays off the struct's public API); the weaver calls it
    ///     through a boxing-free constrained call at each return. The members are yielded from INSIDE the
    ///     struct rather than decomposed by the weaver because a struct's own method can read its private
    ///     fields (auto-property backing fields, <c>_grids</c>, …) while the enclosing type's woven method
    ///     cannot — the CLR grants nested→enclosing private access, not the reverse. A carrier struct
    ///     WITHOUT this interface reports build error NDW003 and must be hand-scoped.
    ///     PUBLIC because the opt-in is consumer-facing too: a project that installs the
    ///     <c>NumSharp.Weaver</c> package implements it on its own result structs so its
    ///     <see cref="NDScopedAttribute"/> methods can return them (and the woven cross-assembly
    ///     <c>YieldTo</c> call must pass the CLR's accessibility check).
    /// </remarks>
    public interface INDArrayCarrier
    {
        /// <summary>Yields every <see cref="NDArray"/> this carrier holds into <paramref name="scope"/>.</summary>
        void YieldTo(NDScope scope);
    }
}
