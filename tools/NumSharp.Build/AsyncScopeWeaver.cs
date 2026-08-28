using Mono.Cecil;

namespace NumSharp.Build;

/// <summary>
///     The <c>[NDScopedAsync]</c> weaver: ASYNC methods (<c>Task</c>/<c>Task&lt;T&gt;</c>/
///     <c>ValueTask</c>/<c>ValueTask&lt;T&gt;</c>/<c>void</c>, any <c>[AsyncMethodBuilder]</c>
///     task-like), ASYNC ITERATORS (<c>IAsyncEnumerable&lt;T&gt;</c>), and NON-ASYNC methods
///     returning <c>Task</c>/<c>ValueTask</c>[<c>&lt;T&gt;</c>]. State machines weave through the
///     base's <see cref="ScopeWeaver.WeaveStateMachineTarget"/> (one invocation scope suspended
///     across every <c>await</c>/yield); a non-async Task-returning body weaves through
///     <see cref="ScopeWeaver.WeaveMethod"/> with the deferral egress
///     (<c>ReturnsTask</c>/<c>ReturnsValueTask</c> + <c>CloseUnlessDeferred</c>).
///     <para>
///     Plain synchronous methods and SYNCHRONOUS iterators belong to <see cref="SyncScopeWeaver"/>;
///     placing <c>[NDScopedAsync]</c> on one of them is a build ERROR (NDW010) that names the right
///     attribute.
///     </para>
/// </summary>
internal sealed class AsyncScopeWeaver : ScopeWeaver
{
    private const string Label = "[NDScopedAsync]";

    public AsyncScopeWeaver(Refs refs, bool verbose, TextWriter stdout, TextWriter stderr)
        : base(refs, verbose, stdout, stderr)
    {
    }

    internal override WeaveOutcome WeaveTarget(MethodDefinition method)
    {
        var smKind = GetStateMachineKind(method, out var smType);

        // A SYNCHRONOUS iterator (IEnumerable/IEnumerator yield) is not asynchronous — it stays on
        // [NDScoped]. Refuse it here so the split has one home per shape.
        if (smKind == StateMachineKind.Iterator)
        {
            _stderr.WriteLine(
                $"NumSharp.Build : error NDW010: {Label} method '{method.FullName}' is a synchronous iterator " +
                "(IEnumerable/IEnumerator yield) — mark it [NDScoped] instead (" + Label + " weaves async methods, " +
                "async iterators and non-async Task/ValueTask returns)");
            return WeaveOutcome.Error;
        }

        // [NDScopedExit] parameters (a retained argument the caller's scope must not dispose) weave
        // regardless of the async shape — the detach lands in the visible method (the compiler stub for
        // an async/iterator method, which runs synchronously on the caller's thread at the call).
        if (ValidateExitParams(method, Label, _refs, _stderr) == ValidationOutcome.Error)
            return WeaveOutcome.Error;

        if (smKind is StateMachineKind.Async or StateMachineKind.AsyncIterator)
        {
            InjectParameterDetaches(method, _refs);
            return FromStateMachineOutcome(
                WeaveStateMachineTarget(method, smType, smKind, Label, _refs, _stderr), method, smKind);
        }

        // A plain (non-state-machine) method under [NDScopedAsync] must return a Task/ValueTask —
        // that is the deferral egress. Validate the body first (no-body/ref-egress/unsupported carrier/
        // idempotence/tail-call), then gate the shape.
        switch (Validate(method, Label, _stderr))
        {
            case ValidationOutcome.AlreadyScoped:
                if (_verbose)
                    _stdout.WriteLine($"NumSharp.Build: skip (already opens an NDScope): {method.FullName}");
                return WeaveOutcome.Skipped;
            case ValidationOutcome.Error:
                return WeaveOutcome.Error;
        }

        if (Classify(method.ReturnType) != RetKind.TaskLike)
        {
            _stderr.WriteLine(
                $"NumSharp.Build : error NDW010: {Label} method '{method.FullName}' is a plain synchronous method " +
                $"returning '{method.ReturnType.FullName}' — mark it [NDScoped] instead (" + Label + " weaves async " +
                "methods, async iterators and non-async Task/ValueTask returns)");
            return WeaveOutcome.Error;
        }

        // The deferral seam is newer than the synchronous surface — demand it only now that a
        // Task-shaped target actually needs it (an older referenced NumSharp still weaves every
        // synchronous [NDScoped] target unchanged).
        if (!_refs.HasAsyncSurface)
        {
            _stderr.WriteLine(
                $"NumSharp.Build : error NDW008: {Label} method '{method.FullName}' returns a Task/ValueTask, but the " +
                "referenced NumSharp does not carry the async scope seam (NDScope.ReturnsTask/CloseUnlessDeferred) — " +
                "update the NumSharp package");
            return WeaveOutcome.Error;
        }

        InjectParameterDetaches(method, _refs);
        WeaveMethod(method, _refs);
        if (_verbose)
            _stdout.WriteLine($"NumSharp.Build: woven: {method.FullName}");
        return WeaveOutcome.Woven;
    }
}
