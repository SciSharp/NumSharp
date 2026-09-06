using Mono.Cecil;

namespace NumSharp.Build;

/// <summary>
///     The <c>[NDScoped]</c> weaver: SYNCHRONOUS boundary methods and SYNCHRONOUS iterators
///     (<c>IEnumerable&lt;T&gt;</c>/<c>IEnumerator&lt;T&gt;</c> + <c>yield return</c>). A plain body
///     is woven through <see cref="ScopeWeaver.WeaveMethod"/>; a synchronous iterator — which still
///     compiles to a state machine, and so shares the invocation-scope seam — through the base's
///     <see cref="ScopeWeaver.WeaveStateMachineTarget"/>.
///     <para>
///     Async methods, async iterators and non-async <c>Task</c>/<c>ValueTask</c> returns suspend
///     across <c>await</c> (or defer disposal to a task's completion) and belong to
///     <see cref="AsyncScopeWeaver"/>; placing <c>[NDScoped]</c> on one of them is a build ERROR
///     (NDW009) that names the right attribute rather than a silent unwoven ship.
///     </para>
///     <para>
///     A target may carry the attribute itself or INHERIT it from the virtual/abstract/interface
///     declaration it overrides or implements (<see cref="ScopeInheritance"/>); every diagnostic and
///     log line names the inherited declaration so a rejection on an override reads back to its source.
///     </para>
/// </summary>
internal sealed class SyncScopeWeaver : ScopeWeaver
{
    private const string Label = "[NDScoped]";

    public SyncScopeWeaver(Refs refs, bool verbose, TextWriter stdout, TextWriter stderr)
        : base(refs, verbose, stdout, stderr)
    {
    }

    internal override WeaveOutcome WeaveTarget(MethodDefinition method)
    {
        var label = Label + ScopeInheritance.Provenance(_refs.Inheritance.EffectiveScope(method));
        var smKind = GetStateMachineKind(method, out var smType);

        // Async / async-iterator methods suspend across `await` and need the deferral seam — that is
        // [NDScopedAsync]'s job. Refuse loudly rather than ship them unwoven under the wrong attribute.
        if (smKind is StateMachineKind.Async or StateMachineKind.AsyncIterator)
        {
            _stderr.WriteLine(
                $"NumSharp.Build : error NDW009: {label} method '{method.FullName}' is an {StateMachineLabel(smKind)} " +
                "method — mark it [NDScopedAsync] instead (" + Label + " weaves synchronous methods and synchronous " +
                "iterators; [NDScopedAsync] weaves async methods, async iterators and non-async Task/ValueTask returns)");
            return WeaveOutcome.Error;
        }

        // [NDScopedExit] parameters (a retained argument the caller's scope must not dispose) weave
        // regardless of the method's own shape — the detach lands in the visible method (this body, or
        // the stub for an iterator). Validate them up front so a bad parameter fails the same for every
        // shape.
        if (ValidateExitParams(method, label, _refs, _stderr) == ValidationOutcome.Error)
            return WeaveOutcome.Error;

        // A SYNCHRONOUS iterator compiles to a state machine too, but it is not asynchronous — it is
        // woven here through the shared invocation-scope transform (Suspend/DisposeSlot/ExitIterator).
        // Any [NDScopedExit] parameters detach in the stub (this method), independently of MoveNext.
        if (smKind == StateMachineKind.Iterator)
        {
            InjectParameterDetaches(method, _refs);
            return FromStateMachineOutcome(
                WeaveStateMachineTarget(method, smType, smKind, label, _refs, _stderr), method, smKind);
        }

        // A NON-async method returning Task/ValueTask gets the deferral egress — also [NDScopedAsync].
        if (Classify(method.ReturnType) == RetKind.TaskLike)
        {
            _stderr.WriteLine(
                $"NumSharp.Build : error NDW009: {label} method '{method.FullName}' returns a Task/ValueTask — " +
                "mark it [NDScopedAsync] instead (the deferral egress that protects operands an in-flight callee still " +
                "holds lives on [NDScopedAsync])");
            return WeaveOutcome.Error;
        }

        switch (Validate(method, label, _stderr))
        {
            case ValidationOutcome.AlreadyScoped:
                // A hand-scoped body is left untouched — including its [NDScopedExit] parameters, which
                // the hand author detaches by hand (the same hands-off rule the scope idempotence keeps).
                if (_verbose)
                    _stdout.WriteLine($"NumSharp.Build: skip (already opens an NDScope): {method.FullName}");
                return WeaveOutcome.Skipped;
            case ValidationOutcome.Error:
                return WeaveOutcome.Error;
        }

        // Inject the parameter detaches BEFORE the scope weave so they land inside the woven try, at the
        // very top of the body (a detach targets the argument's own tracking scope — the caller's — not
        // this method's fresh scope, which never tracked an input).
        InjectParameterDetaches(method, _refs);
        WeaveMethod(method, _refs);
        if (_verbose)
            _stdout.WriteLine($"NumSharp.Build: woven{ScopeInheritance.Provenance(_refs.Inheritance.EffectiveScope(method))}: {method.FullName}");
        return WeaveOutcome.Woven;
    }
}
