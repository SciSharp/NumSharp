using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

namespace NumSharp.Build;

internal readonly record struct WeaveResult(int Woven, int Skipped, int Errors, bool SymbolsWritten, bool Wrote);

/// <summary>
///     The transform. Per <c>[NDScoped]</c> method:
///     <list type="number">
///     <item>prologue OUTSIDE the protected region: <c>scope = NDScope.Open()</c> into a fresh local;</item>
///     <item>the whole ORIGINAL body becomes the try of a try/finally whose finally is
///           <c>scope.Dispose()</c> — the CLR forbids <c>ret</c> inside a protected region, so every
///           original return is forced through a rewritable seam;</item>
///     <item>each original <c>ret</c> is rewritten IN PLACE (the ret instruction object is mutated into
///           the first replacement instruction, so branches that targeted it stay valid): the return
///           value lands in a local, then its NDArray content is yielded through <c>scope.Returns</c> —
///           a bare <c>NDArray</c> / <c>NDArray[]</c> directly; a <c>ValueTuple</c> of NDArrays through
///           the matching tuple overload; a result-struct carrier (<c>UniqueResult</c>,
///           <c>MeshgridResult</c>, <c>PolyfitResult</c>, …) by yielding each of its NDArray/NDArray[]
///           fields in place — plus each <c>out NDArray</c> parameter's FINAL value (success path only —
///           on a throw the finally reclaims, and out contents are undefined after a throw anyway), then
///           <c>leave</c> to a single epilogue (<c>[ldloc ret;] ret</c>) after the handler.</item>
///     </list>
///     Existing try/using blocks in the body nest inside the new outer handler; the handler-table
///     append order (nested first) is exactly what <c>ExceptionHandlers.Add</c> produces.
///     <para>
///     Async and iterator methods weave through their compiler STATE MACHINES instead (the
///     attributed method is a stub) — one scope per logical invocation held in a weaver-added
///     state-machine field, suspended before every continuation schedule and resumed at each
///     MoveNext; see the "state machines" section below. A NON-async method returning
///     Task/ValueTask gets the deferral egress (<c>ReturnsTask</c>/<c>ReturnsValueTask</c> +
///     <c>CloseUnlessDeferred</c> in the finally).
///     </para>
///     <para>
///     This is the ABSTRACT BASE holding that shared machinery; the two concrete weavers are
///     <see cref="SyncScopeWeaver"/> (the <c>[NDScoped]</c> attribute — synchronous bodies AND
///     synchronous iterators) and <see cref="AsyncScopeWeaver"/> (the <c>[NDScopedAsync]</c>
///     attribute — async methods, async iterators, and non-async <c>Task</c>/<c>ValueTask</c>
///     returns). <see cref="WeaveAssembly"/> collects each attribute's targets, routes them to the
///     matching weaver, and reports the wrong attribute (NDW009/NDW010) as a build error. The two
///     shapes that need the async seam — a synchronous iterator (<c>[NDScoped]</c>) and everything
///     under <c>[NDScopedAsync]</c> — both drive the SAME base state-machine/deferral transforms,
///     which is why that machinery lives here rather than in either derived weaver.
///     </para>
/// </summary>
internal abstract class ScopeWeaver
{
    internal const string SyncAttributeFullName = "NumSharp.NDScopedAttribute";
    internal const string AsyncAttributeFullName = "NumSharp.NDScopedAsyncAttribute";
    internal const string ExitAttributeFullName = "NumSharp.NDScopedExitAttribute";
    private const string NDScopeFullName = "NumSharp.NDScope";
    private const string NDArrayFullName = "NumSharp.NDArray";
    private const string CarrierInterfaceFullName = "NumSharp.INDArrayCarrier";
    private const string ITupleFullName = "System.Runtime.CompilerServices.ITuple";
    private const string ValueTuplePrefix = "System.ValueTuple`";
    private const string TuplePrefix = "System.Tuple`";
    private const string IArraySliceFullName = "NumSharp.Backends.Unmanaged.IArraySlice";
    private const string UnmanagedStorageFullName = "NumSharp.Backends.UnmanagedStorage";
    private const string TaskFullName = "System.Threading.Tasks.Task";
    private const string ValueTaskFullName = "System.Threading.Tasks.ValueTask";
    private const string TaskOfTFullName = "System.Threading.Tasks.Task`1";
    private const string ValueTaskOfTFullName = "System.Threading.Tasks.ValueTask`1";

    // Compiler state-machine shapes. The routing attribute names are the C# language contract; the
    // three GENERATED member names below are Roslyn's stable generated-name scheme (the debugger and
    // EnC parse them, so they are compatibility-pinned upstream). A state machine that does not
    // carry them — another compiler, a future rename — fails LOUDLY with NDW004, never mis-weaves.
    private const string AsyncSmAttrFullName = "System.Runtime.CompilerServices.AsyncStateMachineAttribute";
    private const string IteratorSmAttrFullName = "System.Runtime.CompilerServices.IteratorStateMachineAttribute";
    private const string AsyncIteratorSmAttrFullName = "System.Runtime.CompilerServices.AsyncIteratorStateMachineAttribute";
    private const string BuilderFieldName = "<>t__builder";
    private const string CurrentFieldName = "<>2__current";
    private const string PromiseElementFullName = "System.Threading.Tasks.Sources.ManualResetValueTaskSourceCore`1";

    /// <summary>
    ///     The weaver-added state-machine field holding the invocation's scope across suspensions.
    ///     Unspeakable (angle brackets) so it can never collide with a user identifier; its presence
    ///     is NOT the idempotence signal (that is the <c>OpenOrResume</c> call in MoveNext).
    /// </summary>
    private const string SlotFieldName = "<>ndscope";

    /// <summary>
    ///     The NDScope surface the transform emits calls to. Every member is held as a reference
    ///     ALREADY IMPORTED into the module being woven: for the self-weave (NumSharp.Core, where
    ///     NDScope is in-module) <c>ImportReference</c> is an identity pass-through, and for a
    ///     consumer weave it mints the cross-assembly <c>MemberRef</c>s targeting the referenced
    ///     NumSharp assembly.
    /// </summary>
    internal sealed class Refs
    {
        public TypeReference NDScope;
        public MethodReference Open;         // static NDScope Open()
        public MethodReference Dispose;      // void Dispose()
        public MethodReference ReturnsOne;   // T Returns<T>(T)     where T : NDArray
        public MethodReference ReturnsMany;  // T[] Returns<T>(T[]) where T : NDArray
        public MethodReference ReturnsTuple2; // (T1,T2) Returns<T1,T2>((T1,T2))
        public MethodReference ReturnsTuple3; // (T1,T2,T3) Returns<T1,T2,T3>((T1,T2,T3))
        public MethodReference ReturnsTuple4; // (T1,T2,T3,T4) Returns<T1,T2,T3,T4>((T1,T2,T3,T4))
        public MethodReference ReturnsITuple; // ITuple Returns(ITuple) — any arity/mix, boxed
        public MethodReference ReturnsSlice;   // IArraySlice Returns(IArraySlice) — counted-ref protection
        public MethodReference ReturnsStorage; // UnmanagedStorage Returns(UnmanagedStorage)
        public MethodReference CarrierYieldTo; // void INDArrayCarrier.YieldTo(NDScope)

        // ---- The async / state-machine seam (NumSharp versions that ship it) --------------------
        // Resolved LENIENTLY: an older referenced NumSharp without these members must still weave
        // every synchronous target — the members are demanded (error NDW008) only when a target
        // actually needs them. All new NAMES on the NDScope side, deliberately: an OLD weaver keys
        // the `Returns` family by generic arity alone, so a new `Returns<T>(Task<T>)` OVERLOAD
        // would silently re-bind its ReturnsOne — distinct names make old-weaver+new-NumSharp safe.
        public MethodReference OpenOrResume;        // static NDScope OpenOrResume(ref NDScope)
        public MethodReference Suspend;             // static void Suspend(NDScope)
        public MethodReference DisposeSlot;         // static void DisposeSlot(ref NDScope)
        public MethodReference ExitIterator;        // static void ExitIterator(ref NDScope, bool)
        public MethodReference CloseUnlessDeferred; // static void CloseUnlessDeferred(NDScope)
        public MethodReference ReturnsTaskOfT;      // Task<T> ReturnsTask<T>(Task<T>)
        public MethodReference ReturnsTaskPlain;    // Task ReturnsTask(Task)
        public MethodReference ReturnsValueTaskOfT; // ValueTask<T> ReturnsValueTask<T>(ValueTask<T>)
        public MethodReference ReturnsValueTaskPlain; // ValueTask ReturnsValueTask(ValueTask)

        // ---- The [NDScopedExit] parameter seam (retained-argument detach) ------------------------
        // Resolved LENIENTLY like the async seam: Detach(NDArray) is old surface, but the array/tuple
        // overloads are newer, so a target with an [NDScopedExit] parameter demands the full trio
        // (NDW008) only when it actually exists.
        public MethodReference DetachOne;   // static void Detach(NDArray)
        public MethodReference DetachMany;  // static void Detach(NDArray[])
        public MethodReference DetachTuple; // static void Detach(ITuple)

        /// <summary>Whether the referenced NumSharp carries the whole <c>Detach</c> overload set the parameter-detach weave emits.</summary>
        public bool HasExitSurface => DetachOne != null && DetachMany != null && DetachTuple != null;

        /// <summary>Whether the referenced NumSharp carries the whole async/state-machine seam.</summary>
        public bool HasAsyncSurface =>
            OpenOrResume != null && Suspend != null && DisposeSlot != null && ExitIterator != null &&
            CloseUnlessDeferred != null && ReturnsTaskOfT != null && ReturnsTaskPlain != null &&
            ReturnsValueTaskOfT != null && ReturnsValueTaskPlain != null;

        /// <summary>The <c>Returns</c> tuple overload for the given arity (2..4), or null if unsupported.</summary>
        public MethodReference ReturnsTuple(int arity) => arity switch
        {
            2 => ReturnsTuple2,
            3 => ReturnsTuple3,
            4 => ReturnsTuple4,
            _ => null
        };
    }

    internal enum RetKind
    {
        Void,
        Scalar,           // primitives / enums / decimal / Half / Complex / string — scope only
        NDArrayLike,      // NDArray or a subclass (NDArray<T>) — Returns<T>(T)
        NDArrayLikeArray, // NDArray-like[] — Returns<T>(T[])
        NDArrayTuple,     // ValueTuple<..> of 2..4 NDArray-likes — Returns<T..>((T..)) overload (no box)
        Tuple,            // any OTHER ValueTuple/Tuple (arity 5..8, a non-NDArray component, Tuple<>) — Returns(ITuple)
        Carrier,          // in-module result struct — INDArrayCarrier.YieldTo(scope)
        Storage,          // bare IArraySlice / UnmanagedStorage — Returns(slice/storage) counted-ref protection
        TaskLike          // NON-async Task/ValueTask[<T>] return — ReturnsTask/ReturnsValueTask (defer-capable)
    }

    /// <summary>Which compiler state machine a scoped method compiled into, if any.</summary>
    internal enum StateMachineKind
    {
        None,
        Async,        // Task / Task<T> / ValueTask / ValueTask<T> / void, any [AsyncMethodBuilder] task-like
        Iterator,     // IEnumerable[<T>] / IEnumerator[<T>] (yield return)
        AsyncIterator // IAsyncEnumerable<T> (await + yield return)
    }

    /// <summary>The disposition of a single woven target, tallied by <see cref="WeaveAssembly"/>.</summary>
    internal enum WeaveOutcome
    {
        Woven,
        Skipped, // already scoped by hand / a previous pass — idempotence
        Error
    }

    // ---- instance state ------------------------------------------------------------------------
    // One weaver instance per attribute (SyncScopeWeaver / AsyncScopeWeaver). The base holds the
    // resolved seam and the output sinks; the shared transforms below still take them as explicit
    // parameters (unchanged from the single-class weaver), so a derived WeaveTarget threads these.
    internal readonly Refs _refs;
    internal readonly bool _verbose;
    internal readonly TextWriter _stdout;
    internal readonly TextWriter _stderr;

    internal ScopeWeaver(Refs refs, bool verbose, TextWriter stdout, TextWriter stderr)
    {
        _refs = refs;
        _verbose = verbose;
        _stdout = stdout;
        _stderr = stderr;
    }

    /// <summary>
    ///     Weaves ONE method this weaver's attribute marked, per its own sync/async policy — the one
    ///     member the two concrete weavers differ in. <see cref="SyncScopeWeaver"/> owns synchronous
    ///     bodies and synchronous iterators; <see cref="AsyncScopeWeaver"/> owns async methods, async
    ///     iterators and non-async Task/ValueTask returns; each rejects the OTHER's shapes with a
    ///     wrong-attribute build error (NDW009/NDW010). Both delegate to the shared base transforms
    ///     (<see cref="WeaveMethod"/>, <see cref="WeaveStateMachineTarget"/>).
    /// </summary>
    internal abstract WeaveOutcome WeaveTarget(MethodDefinition method);

    /// <summary>The human-readable kind name for a state machine, used in the wrong-attribute diagnostics.</summary>
    internal static string StateMachineLabel(StateMachineKind kind) => kind switch
    {
        StateMachineKind.Async => "async",
        StateMachineKind.Iterator => "synchronous iterator",
        StateMachineKind.AsyncIterator => "async-iterator",
        _ => "state-machine"
    };

    /// <summary>Maps a state-machine weave's <see cref="ValidationOutcome"/> to the tallied <see cref="WeaveOutcome"/>, with the shared verbose logging (both weavers drive state machines).</summary>
    internal WeaveOutcome FromStateMachineOutcome(ValidationOutcome outcome, MethodDefinition stub, StateMachineKind kind)
    {
        switch (outcome)
        {
            case ValidationOutcome.AlreadyScoped:
                if (_verbose)
                    _stdout.WriteLine($"NumSharp.Build: skip (state machine already woven): {stub.FullName}");
                return WeaveOutcome.Skipped;
            case ValidationOutcome.Error:
                return WeaveOutcome.Error;
            default:
                if (_verbose)
                    _stdout.WriteLine($"NumSharp.Build: woven ({kind} state machine): {stub.FullName}");
                return WeaveOutcome.Woven;
        }
    }

    public static WeaveResult WeaveAssembly(string assemblyPath, byte[] snkBlob, IReadOnlyList<string> referencePaths,
                                            bool verbose, TextWriter stdout, TextWriter stderr)
    {
        // The resolver serves BOTH weave shapes: the self-weave (NumSharp.Core — everything
        // in-module, resolver only probes the assembly's own directory) and the consumer weave
        // (the NumSharp.Build package — NDScope/NDArray live in the REFERENCED NumSharp assembly,
        // located through the compile's own reference list passed via --refs).
        using var resolver = new WeaverAssemblyResolver(
            Path.GetDirectoryName(Path.GetFullPath(assemblyPath)), referencePaths ?? Array.Empty<string>());

        var readerParameters = new ReaderParameters
        {
            ReadSymbols = true,
            SymbolReaderProvider = new DefaultSymbolReaderProvider(throwIfNoSymbol: false),
            InMemory = true, // fully load, release the file handle: we write back to the same path
            AssemblyResolver = resolver,
        };

        using var assembly = AssemblyDefinition.ReadAssembly(assemblyPath, readerParameters);
        var module = assembly.MainModule;
        bool hasSymbols = module.HasSymbols;

        int woven = 0, skipped = 0, errors = 0;

        // Targets FIRST (a pure name match, no resolution): a consumer assembly with no scoped
        // methods exits without resolving anything and — crucially — without rewriting the file, so
        // its compile-time signature, determinism id and timestamps stay untouched. The two
        // attributes are collected apart — [NDScoped] (synchronous bodies + synchronous iterators)
        // is SyncScopeWeaver's, [NDScopedAsync] (async methods, async iterators, non-async
        // Task/ValueTask returns) is AsyncScopeWeaver's.
        var syncTargets = CollectTargets(module, SyncAttributeFullName, "[NDScoped]", stderr, ref errors);
        var asyncTargets = CollectTargets(module, AsyncAttributeFullName, "[NDScopedAsync]", stderr, ref errors);

        // Methods with an [NDScopedExit] PARAMETER — an orthogonal, param-level concern (a retained
        // argument the caller's scope must not dispose). A method may carry these with OR without a
        // method-level scope attribute; the scope weavers handle the ones that overlap, and a dedicated
        // pass below covers the rest.
        var exitTargets = CollectExitTargets(module);

        // A method carrying BOTH attributes has no single scoping model — fail loudly and drop it
        // from either pass rather than weave it twice.
        if (syncTargets.Count > 0 && asyncTargets.Count > 0)
        {
            var both = new HashSet<MethodDefinition>(syncTargets);
            both.IntersectWith(asyncTargets);
            foreach (var m in both)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW011: method '{m.FullName}' carries BOTH [NDScoped] and " +
                    "[NDScopedAsync] — a method has exactly one scoping model; keep only the attribute that matches it " +
                    "([NDScoped] for synchronous methods and synchronous iterators, [NDScopedAsync] for async methods, " +
                    "async iterators and non-async Task/ValueTask returns)");
                errors++;
            }

            if (both.Count > 0)
            {
                syncTargets.RemoveAll(both.Contains);
                asyncTargets.RemoveAll(both.Contains);
                exitTargets.RemoveAll(both.Contains);
            }
        }

        if (syncTargets.Count == 0 && asyncTargets.Count == 0 && exitTargets.Count == 0)
        {
            if (errors > 0)
                return new WeaveResult(0, 0, errors, false, false);
            stdout.WriteLine($"NumSharp.Build: {Path.GetFileName(assemblyPath)} — no [NDScoped]/[NDScopedAsync]/[NDScopedExit] methods; nothing to do");
            return new WeaveResult(0, 0, 0, false, false);
        }

        var refs = ResolveRefs(module);
        if (refs is null)
        {
            // Attributed methods with no reachable NDScope is a CONFIGURATION error, not a no-op:
            // silently skipping would ship those methods unwoven while the build reads green.
            stderr.WriteLine(
                $"NumSharp.Build : error NDW001: {syncTargets.Count + asyncTargets.Count} [NDScoped]/[NDScopedAsync] " +
                $"method(s) found but type '{NDScopeFullName}' could not be resolved from " +
                $"'{Path.GetFileName(assemblyPath)}' or its references — ensure the project references NumSharp (the " +
                "attributes and the scope live there) and that the invocation passed --refs with the compile's " +
                "reference list");
            return new WeaveResult(0, 0, errors + 1, false, false);
        }

        // One weaver instance per attribute; each owns its sync/async policy and delegates to the
        // shared base transforms (synchronous bodies AND state machines are woven by base methods).
        var syncWeaver = new SyncScopeWeaver(refs, verbose, stdout, stderr);
        var asyncWeaver = new AsyncScopeWeaver(refs, verbose, stdout, stderr);
        ProcessTargets(syncTargets, syncWeaver, ref woven, ref skipped, ref errors);
        ProcessTargets(asyncTargets, asyncWeaver, ref woven, ref skipped, ref errors);

        // [NDScopedExit] parameters on a scope target were already woven inside its WeaveTarget
        // (inject-before-scope-weave). The rest — methods that merely RETAIN an NDArray argument with no
        // scope model of their own — get the standalone per-parameter detach here.
        var scopeTargets = new HashSet<MethodDefinition>(syncTargets);
        foreach (var m in asyncTargets)
            scopeTargets.Add(m);
        var exitOnly = new List<MethodDefinition>();
        foreach (var m in exitTargets)
            if (!scopeTargets.Contains(m))
                exitOnly.Add(m);
        ProcessExitOnly(exitOnly, refs, verbose, stdout, stderr, ref woven, ref skipped, ref errors);

        if (errors > 0)
            return new WeaveResult(woven, skipped, errors, false, false);

        // Every target already scoped by hand → the module was not modified; skip the rewrite so
        // the compile-time signature and determinism id survive (a write would invalidate both for
        // zero IL change).
        if (woven == 0)
            return new WeaveResult(0, skipped, 0, false, false);

        var writerParameters = new WriterParameters
        {
            WriteSymbols = hasSymbols,
        };
        if (snkBlob != null)
            writerParameters.StrongNameKeyBlob = snkBlob;

        assembly.Write(assemblyPath, writerParameters);
        return new WeaveResult(woven, skipped, 0, hasSymbols, true);
    }

    /// <summary>Weaves each target through <paramref name="weaver"/> (its attribute's policy) and tallies the outcomes.</summary>
    private static void ProcessTargets(List<MethodDefinition> targets, ScopeWeaver weaver,
                                       ref int woven, ref int skipped, ref int errors)
    {
        foreach (var method in targets)
        {
            switch (weaver.WeaveTarget(method))
            {
                case WeaveOutcome.Woven:
                    woven++;
                    break;
                case WeaveOutcome.Skipped:
                    skipped++;
                    break;
                case WeaveOutcome.Error:
                    errors++;
                    break;
            }
        }
    }

    /// <summary>
    ///     Injects the per-parameter <c>Detach</c> egress into methods that carry an
    ///     <c>[NDScopedExit]</c> parameter but NO method-level scope attribute — a method that merely
    ///     retains an NDArray argument. (Scope targets that also carry <c>[NDScopedExit]</c> parameters
    ///     were handled inside their own <see cref="WeaveTarget"/>, inject-before-scope-weave.) The
    ///     injection is idempotent (a body already calling <c>NDScope.Detach</c> is left alone), so a
    ///     re-weave without a fresh marker is a no-op.
    /// </summary>
    private static void ProcessExitOnly(List<MethodDefinition> exitOnly, Refs refs, bool verbose,
                                        TextWriter stdout, TextWriter stderr,
                                        ref int woven, ref int skipped, ref int errors)
    {
        foreach (var m in exitOnly)
        {
            if (ValidateExitParams(m, "[NDScopedExit]", refs, stderr) == ValidationOutcome.Error)
            {
                errors++;
                continue;
            }

            if (InjectParameterDetaches(m, refs))
            {
                woven++;
                if (verbose)
                    stdout.WriteLine($"NumSharp.Build: woven ([NDScopedExit] parameters): {m.FullName}");
            }
            else
            {
                skipped++;
                if (verbose)
                    stdout.WriteLine($"NumSharp.Build: skip (already detaches): {m.FullName}");
            }
        }
    }

    // ----------------------------------------------------------------- target collection

    private static List<MethodDefinition> CollectTargets(ModuleDefinition module, string attributeName, string label,
                                                         TextWriter stderr, ref int errors)
    {
        var targets = new List<MethodDefinition>();
        foreach (var type in AllTypes(module))
        {
            foreach (var method in type.Methods)
                if (HasAttribute(method.CustomAttributes, attributeName))
                    targets.Add(method);

            foreach (var property in type.Properties)
            {
                if (!HasAttribute(property.CustomAttributes, attributeName))
                    continue;
                if (property.GetMethod is null)
                {
                    stderr.WriteLine(
                        $"NumSharp.Build : error NDW006: {label} on property '{property.FullName}' " +
                        "requires a getter (put the attribute on the accessor for setter-only properties)");
                    errors++;
                    continue;
                }

                if (!targets.Contains(property.GetMethod))
                    targets.Add(property.GetMethod);
            }
        }

        return targets;
    }

    private static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
    {
        var stack = new Stack<TypeDefinition>(module.Types);
        while (stack.Count > 0)
        {
            var t = stack.Pop();
            yield return t;
            foreach (var nested in t.NestedTypes)
                stack.Push(nested);
        }
    }

    private static bool HasAttribute(ICollection<CustomAttribute> attributes, string attributeName)
    {
        foreach (var a in attributes)
            if (a.AttributeType.FullName == attributeName)
                return true;
        return false;
    }

    /// <summary>
    ///     Collects every method that has at least one <c>[NDScopedExit]</c> PARAMETER and a body to
    ///     weave. A body-less declaration (abstract/interface) carries the attribute purely as a
    ///     contract for implementers — there is nothing to inject there, so it is silently not a target
    ///     (the CONCRETE override must re-declare the attribute to be woven, exactly as a method-level
    ///     scope attribute must sit on the concrete method).
    /// </summary>
    private static List<MethodDefinition> CollectExitTargets(ModuleDefinition module)
    {
        var targets = new List<MethodDefinition>();
        foreach (var type in AllTypes(module))
            foreach (var method in type.Methods)
                if (method.HasBody && HasExitParam(method))
                    targets.Add(method);
        return targets;
    }

    private static bool HasExitParam(MethodDefinition m)
    {
        foreach (var p in m.Parameters)
            if (HasAttribute(p.CustomAttributes, ExitAttributeFullName))
                return true;
        return false;
    }

    // ----------------------------------------------------------------- validation

    internal enum ValidationOutcome
    {
        Ok,
        AlreadyScoped,
        Error
    }

    internal static ValidationOutcome Validate(MethodDefinition m, string label, TextWriter stderr)
    {
        if (!m.HasBody)
        {
            stderr.WriteLine($"NumSharp.Build : error NDW005: {label} method '{m.FullName}' has no body (abstract/extern)");
            return ValidationOutcome.Error;
        }

        foreach (var p in m.Parameters)
        {
            if (p.ParameterType is ByReferenceType brt && IsNDArrayCarrying(brt.ElementType) && !p.IsOut)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW002: {label} method '{m.FullName}' has 'ref {brt.ElementType.Name}' " +
                    $"parameter '{p.Name}' — a hidden egress the weaver cannot see; scope this method by hand");
                return ValidationOutcome.Error;
            }
        }

        if (Classify(m.ReturnType) is null)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW003: {label} method '{m.FullName}' returns '{m.ReturnType.FullName}' — " +
                "an unsupported carrier the weaver cannot see every NDArray through (a bespoke reference type, a " +
                "collection, or a result struct that does NOT implement INDArrayCarrier), so its NDArray members would " +
                "be reclaimed and handed to the caller disposed; add INDArrayCarrier to the struct, or scope this " +
                "method by hand. (NDArray, NDArray[], any ValueTuple/Tuple of NDArrays, INDArrayCarrier result " +
                "structs, bare IArraySlice/UnmanagedStorage, and Task/ValueTask of any of these ARE woven.)");
            return ValidationOutcome.Error;
        }

        // Idempotence: a body that already opens a scope (hand-written, or woven by a previous
        // pass over the same assembly) is left alone.
        foreach (var instr in m.Body.Instructions)
        {
            if (instr.OpCode.FlowControl == FlowControl.Call &&
                instr.Operand is MethodReference mr &&
                mr.Name == "Open" &&
                mr.DeclaringType.FullName == NDScopeFullName)
                return ValidationOutcome.AlreadyScoped;
        }

        // Tail-call prefixes would leave nothing on the stack for the stloc rewrite. The C#
        // compiler does not emit them; refuse loudly rather than mis-weave if one ever appears.
        foreach (var instr in m.Body.Instructions)
        {
            if (instr.OpCode == OpCodes.Tail)
            {
                stderr.WriteLine($"NumSharp.Build : error NDW007: {label} method '{m.FullName}' contains a tail-call");
                return ValidationOutcome.Error;
            }
        }

        return ValidationOutcome.Ok;
    }

    // ----------------------------------------------------------------- type classification

    private static Refs ResolveRefs(ModuleDefinition module)
    {
        // In-module first (the self-weave — NumSharp.Core carries NDScope itself), then the
        // assembly references (the consumer weave — the NumSharp.Build package's target hands the
        // reference list via --refs). Every located member is imported into the module being woven:
        // an identity pass-through in-module, a cross-assembly MemberRef otherwise.
        var scopeType = module.GetType(NDScopeFullName) ?? FindExternalType(module, NDScopeFullName);
        if (scopeType is null)
            return null;

        var refs = new Refs { NDScope = module.ImportReference(scopeType) };
        foreach (var m in scopeType.Methods)
        {
            switch (m.Name)
            {
                case "Open" when m.IsStatic && !m.HasParameters:
                    refs.Open = module.ImportReference(m);
                    break;
                case "Dispose" when !m.IsStatic && !m.HasParameters:
                    refs.Dispose = module.ImportReference(m);
                    break;
                // The Returns family is keyed by GENERIC arity, not by inspecting the parameter shape:
                // Returns<T>(T) and Returns<T>(T[]) both have one type parameter (split on ArrayType);
                // the typed tuple overloads have 2/3/4 type parameters; Returns(ITuple) has none.
                case "Returns" when m.Parameters.Count == 1:
                    switch (m.GenericParameters.Count)
                    {
                        case 0 when m.Parameters[0].ParameterType.FullName == ITupleFullName:
                            refs.ReturnsITuple = module.ImportReference(m);
                            break;
                        case 0 when m.Parameters[0].ParameterType.FullName == IArraySliceFullName:
                            refs.ReturnsSlice = module.ImportReference(m);
                            break;
                        case 0 when m.Parameters[0].ParameterType.FullName == UnmanagedStorageFullName:
                            refs.ReturnsStorage = module.ImportReference(m);
                            break;
                        case 1 when m.Parameters[0].ParameterType is ArrayType:
                            refs.ReturnsMany = module.ImportReference(m);
                            break;
                        case 1:
                            refs.ReturnsOne = module.ImportReference(m);
                            break;
                        case 2:
                            refs.ReturnsTuple2 = module.ImportReference(m);
                            break;
                        case 3:
                            refs.ReturnsTuple3 = module.ImportReference(m);
                            break;
                        case 4:
                            refs.ReturnsTuple4 = module.ImportReference(m);
                            break;
                    }

                    break;

                // The async/state-machine seam. Distinct NAMES (never `Returns` overloads) so the
                // arity-keyed binding above can never confuse them — and resolved without any
                // presence requirement: an older NumSharp simply leaves HasAsyncSurface false and
                // only an async/task-shaped target demands it (NDW008).
                case "OpenOrResume" when m.IsStatic && m.Parameters.Count == 1:
                    refs.OpenOrResume = module.ImportReference(m);
                    break;
                case "Suspend" when m.IsStatic && m.Parameters.Count == 1:
                    refs.Suspend = module.ImportReference(m);
                    break;
                case "DisposeSlot" when m.IsStatic && m.Parameters.Count == 1:
                    refs.DisposeSlot = module.ImportReference(m);
                    break;
                case "ExitIterator" when m.IsStatic && m.Parameters.Count == 2:
                    refs.ExitIterator = module.ImportReference(m);
                    break;
                case "CloseUnlessDeferred" when m.IsStatic && m.Parameters.Count == 1:
                    refs.CloseUnlessDeferred = module.ImportReference(m);
                    break;
                case "ReturnsTask" when !m.IsStatic && m.Parameters.Count == 1:
                    if (m.GenericParameters.Count == 1)
                        refs.ReturnsTaskOfT = module.ImportReference(m);
                    else
                        refs.ReturnsTaskPlain = module.ImportReference(m);
                    break;
                case "ReturnsValueTask" when !m.IsStatic && m.Parameters.Count == 1:
                    if (m.GenericParameters.Count == 1)
                        refs.ReturnsValueTaskOfT = module.ImportReference(m);
                    else
                        refs.ReturnsValueTaskPlain = module.ImportReference(m);
                    break;

                // The [NDScopedExit] detach seam — three static Detach(...) overloads keyed by the
                // parameter shape (bare NDArray / NDArray[] / ITuple). Optional surface: Detach(NDArray)
                // is old, the array/tuple overloads are newer, all demanded only when an [NDScopedExit]
                // parameter needs them (NDW008).
                case "Detach" when m.IsStatic && m.Parameters.Count == 1:
                    var detachParam = m.Parameters[0].ParameterType;
                    if (detachParam.FullName == NDArrayFullName)
                        refs.DetachOne = module.ImportReference(m);
                    else if (detachParam is ArrayType)
                        refs.DetachMany = module.ImportReference(m);
                    else if (detachParam.FullName == ITupleFullName)
                        refs.DetachTuple = module.ImportReference(m);
                    break;
            }
        }

        // The carrier seam: INDArrayCarrier.YieldTo(NDScope) — the boundary a result struct exposes so
        // the weaver can hand every NDArray it holds back through the scope (a struct's own method can
        // reach its private fields; the enclosing type's woven method cannot). It lives beside NDScope,
        // so when the scope came from a REFERENCE the interface is looked up in that same module first.
        var carrierType = module.GetType(CarrierInterfaceFullName)
                          ?? scopeType.Module.GetType(CarrierInterfaceFullName)
                          ?? FindExternalType(module, CarrierInterfaceFullName);
        var carrierYieldTo = carrierType?.Methods.FirstOrDefault(
            m => m.Name == "YieldTo" && m.Parameters.Count == 1 &&
                 m.Parameters[0].ParameterType.FullName == NDScopeFullName);
        refs.CarrierYieldTo = carrierYieldTo is null ? null : module.ImportReference(carrierYieldTo);

        if (refs.Open is null || refs.Dispose is null || refs.ReturnsOne is null || refs.ReturnsMany is null ||
            refs.ReturnsTuple2 is null || refs.ReturnsTuple3 is null || refs.ReturnsTuple4 is null ||
            refs.ReturnsITuple is null || refs.ReturnsSlice is null || refs.ReturnsStorage is null ||
            refs.CarrierYieldTo is null)
            throw new InvalidOperationException(
                $"{NDScopeFullName}/{CarrierInterfaceFullName} is missing one of Open/Dispose/Returns<T>(T)/" +
                "Returns<T>(T[])/Returns<T1,T2>/Returns<T1,T2,T3>/Returns<T1,T2,T3,T4>/Returns(ITuple)/" +
                "Returns(IArraySlice)/Returns(UnmanagedStorage)/YieldTo(NDScope) — weaver and library out of sync");

        return refs;
    }

    /// <summary>
    ///     Locates a type through the module's ASSEMBLY REFERENCES — the consumer-weave path, where
    ///     NDScope/INDArrayCarrier live in the referenced NumSharp assembly rather than the module
    ///     being woven. NumSharp-named references are tried first (the overwhelmingly common case);
    ///     the full reference list is the fallback for a merged/renamed host. Type forwarders are
    ///     honoured so a future facade split cannot silently break the lookup.
    /// </summary>
    private static TypeDefinition FindExternalType(ModuleDefinition module, string fullName)
    {
        foreach (var numSharpNamedPass in new[] { true, false })
        {
            foreach (var reference in module.AssemblyReferences)
            {
                if (reference.Name.StartsWith("NumSharp", StringComparison.OrdinalIgnoreCase) != numSharpNamedPass)
                    continue;

                AssemblyDefinition referenced;
                try
                {
                    referenced = module.AssemblyResolver.Resolve(reference);
                }
                catch
                {
                    continue; // an unresolvable reference just cannot be the host
                }

                var direct = referenced?.MainModule.GetType(fullName);
                if (direct != null)
                    return direct;

                if (referenced is null)
                    continue;
                foreach (var exported in referenced.MainModule.ExportedTypes)
                {
                    if (exported.FullName != fullName)
                        continue;
                    try
                    {
                        return exported.Resolve();
                    }
                    catch
                    {
                        // forwarder target unresolvable — keep scanning
                    }
                }
            }
        }

        return null;
    }

    /// <summary>
    ///     Framework namespaces that can never derive from NDArray or opt into INDArrayCarrier —
    ///     skipped before any resolution so the classifier does not open BCL reference assemblies
    ///     for answers it already knows.
    /// </summary>
    private static bool IsFrameworkType(TypeReference t)
    {
        var name = t.FullName;
        return name.StartsWith("System.", StringComparison.Ordinal)
               || name.StartsWith("Microsoft.", StringComparison.Ordinal);
    }

    /// <summary>NDArray-like, or a rank-1 array of NDArray-like — the two shapes Returns can yield.</summary>
    private static bool IsNDArrayCarrying(TypeReference t)
        => IsNDArrayLike(t) || (t is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType));

    /// <summary>
    ///     NDArray or any subclass (NDArray&lt;T&gt; open or instantiated). Resolution runs through the
    ///     module's <see cref="WeaverAssemblyResolver"/>, so the base-type chain is chased across the
    ///     assembly boundary too — a consumer weave classifies <c>NumSharp.Generic.NDArray&lt;T&gt;</c>
    ///     (or the consumer's own subclass) exactly as the self-weave does in-module.
    /// </summary>
    private static bool IsNDArrayLike(TypeReference t)
    {
        if (t is null || t.IsByReference || t.IsPointer || t is ArrayType || t.IsGenericParameter)
            return false;
        if (t.FullName == NDArrayFullName)
            return true;
        if (IsFrameworkType(t))
            return false;

        TypeDefinition def;
        try
        {
            def = t.Resolve();
        }
        catch
        {
            return false;
        }

        while (def != null)
        {
            if (def.FullName == NDArrayFullName)
                return true;
            var baseRef = def.BaseType;
            if (baseRef is null || baseRef.FullName == "System.Object")
                return false;
            try
            {
                def = baseRef.Resolve();
            }
            catch
            {
                return false;
            }
        }

        return false;
    }

    /// <summary>Null = rejected (unsupported carrier / hidden egress): the caller reports NDW003.</summary>
    internal static RetKind? Classify(TypeReference ret)
    {
        if (ret.MetadataType == MetadataType.Void)
            return RetKind.Void;

        if (IsNDArrayLike(ret))
            return RetKind.NDArrayLike;

        if (ret is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType))
            return RetKind.NDArrayLikeArray;

        // A NON-async method returning a task shape: weavable when the task's result type is itself
        // weavable (or there is none) — the egress call yields a completed result immediately and
        // DEFERS the scope's disposal to an incomplete task's completion, because the in-flight
        // callee may still be using tracked temps handed to it. An unsupported result type falls
        // through to NDW003 exactly like a direct return of it would.
        if (IsTaskLike(ret, out var taskResult, out _))
            return taskResult is null || Classify(taskResult) is not null ? RetKind.TaskLike : null;

        if (IsScalar(ret))
            return RetKind.Scalar;

        // A small all-NDArray ValueTuple (svd/qr/eig/lstsq/modf/average/polydiv) takes the strongly-typed
        // Returns overload (no box); any OTHER tuple — arity 5..8, a non-NDArray component, or a
        // reference-type Tuple — takes the general Returns(ITuple). A result-struct carrier
        // (UniqueResult, MeshgridResult, PolyfitResult, …) yields through its own INDArrayCarrier.YieldTo.
        if (TryGetNDArrayTuple(ret, out _))
            return RetKind.NDArrayTuple;

        if (IsGeneralTuple(ret, out _))
            return RetKind.Tuple;

        if (ImplementsCarrierInterface(ret))
            return RetKind.Carrier;

        // A bare lower-layer buffer (IArraySlice / UnmanagedStorage), NOT wrapped in an NDArray — the
        // return is given a counted reference so the scope's Release of intermediate NDArrays can't free it.
        if (ret.FullName is IArraySliceFullName or UnmanagedStorageFullName)
            return RetKind.Storage;

        return null;
    }

    /// <summary>
    ///     The kind of detach an <c>[NDScopedExit]</c> parameter needs, or null when the parameter type
    ///     is unsupported (NDW014). Only the NDArray-CARRYING by-value shapes are detachable: a bare
    ///     NDArray, an NDArray[] , or a ValueTuple/Tuple of NDArrays. A <c>ref</c>/<c>out</c>/<c>in</c>
    ///     parameter, a scalar, a bare buffer (never scope-tracked), or a result-struct carrier is
    ///     rejected — detach those by hand.
    /// </summary>
    private static RetKind? ClassifyExitParam(TypeReference t)
    {
        if (t is ByReferenceType)
            return null; // ref/out/in — a hidden aliasing egress, not a retained by-value argument

        if (IsNDArrayLike(t))
            return RetKind.NDArrayLike;

        if (t is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType))
            return RetKind.NDArrayLikeArray;

        if (TryGetNDArrayTuple(t, out _))
            return RetKind.NDArrayTuple;

        if (IsGeneralTuple(t, out _))
            return RetKind.Tuple;

        return null;
    }

    /// <summary>
    ///     Validates every <c>[NDScopedExit]</c> parameter of <paramref name="m"/>: each must be a
    ///     supported by-value NDArray-carrying shape (else NDW014), and when any is present the
    ///     referenced NumSharp must carry the <c>Detach</c> overload set (else NDW008). Returns
    ///     <see cref="ValidationOutcome.Ok"/> when there are no exit parameters at all.
    /// </summary>
    internal static ValidationOutcome ValidateExitParams(MethodDefinition m, string label, Refs refs, TextWriter stderr)
    {
        bool any = false;
        foreach (var p in m.Parameters)
        {
            if (!HasAttribute(p.CustomAttributes, ExitAttributeFullName))
                continue;
            any = true;

            if (ClassifyExitParam(p.ParameterType) is null)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW014: [NDScopedExit] on parameter '{p.Name}' of '{m.FullName}' has " +
                    $"type '{p.ParameterType.FullName}' — the attribute marks an NDArray-carrying BY-VALUE parameter " +
                    "the callee retains (NDArray, NDArray[], or a ValueTuple/Tuple of NDArrays), so the caller's scope " +
                    "will not dispose it; it is unsupported on this type (a ref/out/in parameter, a scalar, a bare " +
                    "IArraySlice/UnmanagedStorage, or an INDArrayCarrier result struct) — detach those by hand with " +
                    "NDScope.Detach");
                return ValidationOutcome.Error;
            }
        }

        if (any && !refs.HasExitSurface)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW008: {label} method '{m.FullName}' has an [NDScopedExit] parameter, but the " +
                "referenced NumSharp does not carry the NDScope.Detach(NDArray[])/Detach(ITuple) overloads the detach " +
                "weave emits — update the NumSharp package");
            return ValidationOutcome.Error;
        }

        return ValidationOutcome.Ok;
    }

    /// <summary>
    ///     True for the four task shapes a synchronous method can return — <c>Task</c>,
    ///     <c>Task&lt;T&gt;</c>, <c>ValueTask</c>, <c>ValueTask&lt;T&gt;</c> (exact types; a custom
    ///     task-like return is an unsupported carrier). <paramref name="resultType"/> is the
    ///     <c>T</c>, or null for the resultless shapes. Detected by name, like the tuples.
    /// </summary>
    private static bool IsTaskLike(TypeReference t, out TypeReference resultType, out bool isValueTask)
    {
        resultType = null;
        isValueTask = false;
        if (t is GenericInstanceType git && git.GenericArguments.Count == 1)
        {
            switch (git.ElementType.FullName)
            {
                case TaskOfTFullName:
                    resultType = git.GenericArguments[0];
                    return true;
                case ValueTaskOfTFullName:
                    resultType = git.GenericArguments[0];
                    isValueTask = true;
                    return true;
            }

            return false;
        }

        switch (t.FullName)
        {
            case TaskFullName:
                return true;
            case ValueTaskFullName:
                isValueTask = true;
                return true;
        }

        return false;
    }

    /// <summary>
    ///     True for ANY <see cref="System.ValueTuple"/> or <see cref="System.Tuple"/> instantiation — the
    ///     types that implement <c>ITuple</c>, whatever the arity or component mix. <paramref name="isValueType"/>
    ///     distinguishes a ValueTuple (a value type — must be boxed to pass as ITuple) from a reference-type
    ///     Tuple. Detected by name so no external assembly resolver is needed.
    /// </summary>
    private static bool IsGeneralTuple(TypeReference t, out bool isValueType)
    {
        isValueType = false;
        if (t is not GenericInstanceType git)
            return false;

        var name = git.ElementType.FullName;
        if (name.StartsWith(ValueTuplePrefix, StringComparison.Ordinal))
        {
            isValueType = true;
            return true;
        }

        return name.StartsWith(TuplePrefix, StringComparison.Ordinal);
    }

    /// <summary>A value that holds no NDArray — primitives, enums, string, decimal/Half/Complex, native ints.</summary>
    private static bool IsScalar(TypeReference t)
    {
        switch (t.MetadataType)
        {
            case MetadataType.Boolean:
            case MetadataType.Char:
            case MetadataType.SByte:
            case MetadataType.Byte:
            case MetadataType.Int16:
            case MetadataType.UInt16:
            case MetadataType.Int32:
            case MetadataType.UInt32:
            case MetadataType.Int64:
            case MetadataType.UInt64:
            case MetadataType.Single:
            case MetadataType.Double:
            case MetadataType.String:
            case MetadataType.IntPtr:
            case MetadataType.UIntPtr:
                return true;
        }

        switch (t.FullName)
        {
            case "System.Decimal":
            case "System.Half":
            case "System.Numerics.Complex":
                return true;
        }

        // Any resolvable enum is a scalar — NumSharp's (NPTypeCode), the consumer's own, or a
        // framework one; a type that cannot be resolved is not PROVABLY scalar and falls through
        // to NDW003 rather than being silently trusted.
        try
        {
            if (t.Resolve() is { IsEnum: true })
                return true;
        }
        catch
        {
            // not resolvable here — treat as non-scalar
        }

        return false;
    }

    /// <summary>
    ///     True for <c>System.ValueTuple&lt;…&gt;</c> of arity 2..4 whose every component is NDArray-like
    ///     — the shapes NumPy's factorisations and <c>modf</c>/<c>average</c>/<c>polydiv</c> return, yielded
    ///     through a <c>Returns</c> tuple overload. A mixed tuple, a 1-tuple, or arity &gt; 4 (no overload)
    ///     is declined and — being a <c>System.*</c> type — falls through to NDW003.
    /// </summary>
    private static bool TryGetNDArrayTuple(TypeReference t, out GenericInstanceType tuple)
    {
        tuple = null;
        if (t is not GenericInstanceType git ||
            !git.ElementType.FullName.StartsWith(ValueTuplePrefix, StringComparison.Ordinal))
            return false;

        int arity = git.GenericArguments.Count;
        if (arity < 2 || arity > 4)
            return false;

        foreach (var arg in git.GenericArguments)
            if (!IsNDArrayLike(arg))
                return false;

        tuple = git;
        return true;
    }

    /// <summary>
    ///     True for a result-struct carrier that opts into weaving by implementing
    ///     <c>NumSharp.INDArrayCarrier</c> — its <c>YieldTo(NDScope)</c> hands every NDArray it holds back
    ///     through the scope. The opt-in is what lets the weaver reach members behind PRIVATE fields
    ///     (auto-property backing fields, <c>_grids</c>, …): a struct's own method can read them, but the
    ///     enclosing type's woven method cannot (the CLR grants nested→enclosing private access, not the
    ///     reverse). A struct without the interface reports NDW003 and is hand-scoped. The interface is
    ///     public, so a CONSUMER's own result structs opt in the same way NumSharp's do.
    /// </summary>
    private static bool ImplementsCarrierInterface(TypeReference t)
    {
        if (t is null || t.IsByReference || t.IsPointer || t is ArrayType || t.IsGenericParameter)
            return false;
        if (IsFrameworkType(t))
            return false; // ValueTuples/Tuples are handled above; no framework struct is a carrier

        TypeDefinition def;
        try
        {
            def = t.Resolve();
        }
        catch
        {
            return false;
        }

        if (def is null || !def.IsValueType || def.IsEnum)
            return false;

        foreach (var i in def.Interfaces)
            if (i.InterfaceType.FullName == CarrierInterfaceFullName)
                return true;

        return false;
    }

    // ----------------------------------------------------------------- the transform

    /// <summary>
    ///     Prepends <c>NDScope.Detach(&lt;param&gt;)</c> for each <c>[NDScopedExit]</c> parameter to the
    ///     START of <paramref name="m"/>'s body — the "callee detaches its own retained argument" seam.
    ///     Because <see cref="NDScope"/> is <c>[ThreadStatic]</c> and the callee runs within the
    ///     caller's ambient scope, this reaches into that scope with no call-site plumbing. For a plain
    ///     <c>[NDScoped]</c> method it is called BEFORE the scope weave, so the detaches land inside the
    ///     woven try (before the original body); for an async/iterator method it lands in the compiler
    ///     STUB — which runs synchronously on the caller's thread at the call — independently of the
    ///     MoveNext scope weave. Idempotent: a body already calling <c>NDScope.Detach</c> (a prior pass,
    ///     or a hand-written detach) is left alone; returns whether anything was injected.
    /// </summary>
    internal static bool InjectParameterDetaches(MethodDefinition m, Refs refs)
    {
        var body = m.Body;

        // Idempotence — a Detach already present means a previous pass wove this, or the author
        // hand-detaches; either way, hands off (mirrors the scope weave's Open/OpenOrResume check).
        foreach (var instr in body.Instructions)
            if (instr.OpCode.FlowControl == FlowControl.Call &&
                instr.Operand is MethodReference mr &&
                mr.Name == "Detach" &&
                mr.DeclaringType.FullName == NDScopeFullName)
                return false;

        var il = body.GetILProcessor();

        // Insert everything immediately BEFORE the original first instruction, in parameter order.
        // Nothing branches across this point (it is the very top), so relative branch distances are
        // unchanged and no macro juggling is needed; the calls are all static, so no scope local is
        // required (Detach reaches the argument's own tracking scope).
        var anchor = body.Instructions[0];
        bool injected = false;
        foreach (var p in m.Parameters)
        {
            if (!HasAttribute(p.CustomAttributes, ExitAttributeFullName))
                continue;

            var kind = ClassifyExitParam(p.ParameterType);
            if (kind is null)
                continue; // already reported by ValidateExitParams; never reached on a validated method

            switch (kind.Value)
            {
                case RetKind.NDArrayLike:
                    // NDScope.Detach((NDArray)p) — an NDArray<T> upcasts to NDArray with no conversion.
                    il.InsertBefore(anchor, il.Create(OpCodes.Ldarg, p));
                    il.InsertBefore(anchor, il.Create(OpCodes.Call, refs.DetachOne));
                    break;

                case RetKind.NDArrayLikeArray:
                    // NDScope.Detach((NDArray[])p) — NDArray<T>[] passes by reference-array covariance.
                    il.InsertBefore(anchor, il.Create(OpCodes.Ldarg, p));
                    il.InsertBefore(anchor, il.Create(OpCodes.Call, refs.DetachMany));
                    break;

                case RetKind.NDArrayTuple:
                case RetKind.Tuple:
                    // NDScope.Detach((ITuple)p) — a ValueTuple boxes; a reference Tuple passes straight
                    // through. Mirrors the Returns(ITuple) emission; Detach returns void, so no pop.
                    il.InsertBefore(anchor, il.Create(OpCodes.Ldarg, p));
                    IsGeneralTuple(p.ParameterType, out var isValueTuple);
                    if (kind.Value == RetKind.NDArrayTuple || isValueTuple)
                        il.InsertBefore(anchor, il.Create(OpCodes.Box, p.ParameterType));
                    il.InsertBefore(anchor, il.Create(OpCodes.Call, refs.DetachTuple));
                    break;
            }

            injected = true;
        }

        return injected;
    }

    internal static void WeaveMethod(MethodDefinition m, Refs refs)
    {
        var body = m.Body;
        body.SimplifyMacros(); // long-form branches/ldloc so inserted code cannot overflow short offsets

        var il = body.GetILProcessor();
        var retKind = Classify(m.ReturnType)!.Value;

        var scopeVar = new VariableDefinition(refs.NDScope);
        body.Variables.Add(scopeVar);
        VariableDefinition retVar = null;
        if (retKind != RetKind.Void)
        {
            retVar = new VariableDefinition(m.ReturnType);
            body.Variables.Add(retVar);
        }

        body.InitLocals = true;

        var outNdParams = new List<ParameterDefinition>();
        foreach (var p in m.Parameters)
            if (p.IsOut && p.ParameterType is ByReferenceType brt && IsNDArrayCarrying(brt.ElementType))
                outNdParams.Add(p);

        // -- prologue (OUTSIDE the protected region, like C#'s `using var scope = NDScope.Open();`)
        var tryStart = body.Instructions[0];
        il.InsertBefore(tryStart, il.Create(OpCodes.Call, refs.Open));
        il.InsertBefore(tryStart, il.Create(OpCodes.Stloc, scopeVar));

        // -- epilogue (after the handler): [ldloc retVar;] ret
        Instruction epilogueStart;
        var finalRet = il.Create(OpCodes.Ret);
        if (retVar != null)
        {
            epilogueStart = il.Create(OpCodes.Ldloc, retVar);
            il.Append(epilogueStart);
            il.Append(finalRet);
        }
        else
        {
            epilogueStart = finalRet;
            il.Append(finalRet);
        }

        // -- finally handler body: scope.Dispose(); endfinally  (between body and epilogue).
        //    A Task-shaped return closes through CloseUnlessDeferred instead: ReturnsTask may have
        //    handed disposal to the task's completion, while the exception path (which never
        //    reaches ReturnsTask) still disposes eagerly right here.
        var finallyStart = il.Create(OpCodes.Ldloc, scopeVar);
        il.InsertBefore(epilogueStart, finallyStart);
        il.InsertBefore(epilogueStart, retKind == RetKind.TaskLike
            ? il.Create(OpCodes.Call, refs.CloseUnlessDeferred)
            : il.Create(OpCodes.Callvirt, refs.Dispose));
        il.InsertBefore(epilogueStart, il.Create(OpCodes.Endfinally));

        // -- rewrite every ORIGINAL ret (all precede finallyStart; our finalRet is after it)
        var originalRets = new List<Instruction>();
        for (var instr = tryStart; instr != null && instr != finallyStart; instr = instr.Next)
            if (instr.OpCode == OpCodes.Ret)
                originalRets.Add(instr);

        foreach (var ret in originalRets)
        {
            // Mutate the ret instruction itself into the FIRST replacement instruction so every
            // branch (and exception-handler boundary) that referenced it remains correct.
            Instruction cursor;
            if (retVar != null)
            {
                ret.OpCode = OpCodes.Stloc; // return value (on stack) -> retVar
                ret.Operand = retVar;
                cursor = ret;

                switch (retKind)
                {
                    case RetKind.NDArrayLike:
                    case RetKind.NDArrayLikeArray:
                        // scope.Returns(retVar) — the value is re-parented and re-stored (same reference).
                        var returnsRef = InstantiateReturns(m.ReturnType, retKind, refs);
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, returnsRef));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, retVar));
                        break;

                    case RetKind.NDArrayTuple:
                        // scope.Returns((a, b[, …])) — the matching-arity overload yields each component;
                        // the tuple's references are unchanged, so re-storing it is a no-op keep for shape.
                        TryGetNDArrayTuple(m.ReturnType, out var tupleType);
                        var tupleRef = InstantiateReturnsTuple(tupleType, refs);
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, tupleRef));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, retVar));
                        break;

                    case RetKind.Tuple:
                        // scope.Returns((ITuple)retVar) — yields each NDArray component of any arity/mix.
                        // A ValueTuple boxes to ITuple; a reference Tuple passes straight through. The tuple's
                        // NDArray references are re-parented in place, so the (unboxed) retVar is returned as-is.
                        IsGeneralTuple(m.ReturnType, out var tupleIsValueType);
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                        if (tupleIsValueType)
                            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Box, m.ReturnType));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, refs.ReturnsITuple));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Pop));
                        break;

                    case RetKind.Carrier:
                        // retVar.YieldTo(scope) via constrained callvirt (no box): the struct's own method
                        // re-parents each NDArray it holds. The struct value is unchanged, returned as-is.
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloca, retVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Constrained, m.ReturnType));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, refs.CarrierYieldTo));
                        break;

                    case RetKind.Storage:
                        // scope.Returns(retVar) — takes a counted reference on the bare buffer so the scope's
                        // Release of an intermediate NDArray sharing it cannot free it (same reference, re-stored).
                        var storageRef = m.ReturnType.FullName == UnmanagedStorageFullName
                            ? refs.ReturnsStorage
                            : refs.ReturnsSlice;
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, storageRef));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, retVar));
                        break;

                    case RetKind.TaskLike:
                        // scope.ReturnsTask(retVar) / ReturnsValueTask(retVar) — a completed task's
                        // result is yielded now; an incomplete one defers disposal to completion. The
                        // re-store matters: an incomplete ValueTask comes back PRESERVED (the
                        // multi-observable form is what makes observing it legal at all).
                        var taskRef = InstantiateReturnsTaskLike(m.ReturnType, refs);
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, taskRef));
                        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, retVar));
                        break;
                }
            }
            else if (outNdParams.Count > 0)
            {
                ret.OpCode = OpCodes.Ldloc; // becomes the first out-escape's `ldloc scope`
                ret.Operand = scopeVar;
                cursor = ret;
                cursor = EmitOutEscapeTail(il, cursor, outNdParams[0], refs);
                for (int i = 1; i < outNdParams.Count; i++)
                {
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                    cursor = EmitOutEscapeTail(il, cursor, outNdParams[i], refs);
                }

                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Leave, epilogueStart));
                continue;
            }
            else
            {
                ret.OpCode = OpCodes.Leave;
                ret.Operand = epilogueStart;
                continue;
            }

            foreach (var p in outNdParams)
            {
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = EmitOutEscapeTail(il, cursor, p, refs);
            }

            InsertAfter(il, cursor, il.Create(OpCodes.Leave, epilogueStart));
        }

        // -- the outer handler; appended last, so nested (pre-existing) handlers stay first in
        //    the table, which is the innermost-first order the CLR requires.
        body.ExceptionHandlers.Add(new ExceptionHandler(ExceptionHandlerType.Finally)
        {
            TryStart = tryStart,
            TryEnd = finallyStart,
            HandlerStart = finallyStart,
            HandlerEnd = epilogueStart,
        });

        body.OptimizeMacros();
    }

    /// <summary>Emits `ldarg p; ldind.ref; callvirt Returns&lt;elem&gt;; pop` — the caller has already put `scope` on the stack.</summary>
    private static Instruction EmitOutEscapeTail(ILProcessor il, Instruction cursor, ParameterDefinition p, Refs refs)
    {
        var elem = ((ByReferenceType)p.ParameterType).ElementType;
        GenericInstanceMethod returnsRef;
        if (elem is ArrayType arr)
        {
            returnsRef = new GenericInstanceMethod(refs.ReturnsMany); // out NDArray[]-style tuple slot
            returnsRef.GenericArguments.Add(arr.ElementType);
        }
        else
        {
            returnsRef = new GenericInstanceMethod(refs.ReturnsOne);
            returnsRef.GenericArguments.Add(elem);
        }

        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldarg, p));
        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldind_Ref));
        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, returnsRef));
        cursor = InsertAfter(il, cursor, il.Create(OpCodes.Pop));
        return cursor;
    }

    private static MethodReference InstantiateReturns(TypeReference returnType, RetKind kind, Refs refs)
    {
        if (kind == RetKind.NDArrayLike)
        {
            var g = new GenericInstanceMethod(refs.ReturnsOne);
            g.GenericArguments.Add(returnType);
            return g;
        }

        var elem = ((ArrayType)returnType).ElementType;
        var many = new GenericInstanceMethod(refs.ReturnsMany);
        many.GenericArguments.Add(elem);
        return many;
    }

    /// <summary><c>Returns&lt;T1,…&gt;</c> instantiated for a ValueTuple return, one type argument per component.</summary>
    private static MethodReference InstantiateReturnsTuple(GenericInstanceType tuple, Refs refs)
    {
        var g = new GenericInstanceMethod(refs.ReturnsTuple(tuple.GenericArguments.Count));
        foreach (var arg in tuple.GenericArguments)
            g.GenericArguments.Add(arg);
        return g;
    }

    /// <summary><c>ReturnsTask</c>/<c>ReturnsValueTask</c> for the method's exact task shape (generic ones instantiated with the result type).</summary>
    private static MethodReference InstantiateReturnsTaskLike(TypeReference returnType, Refs refs)
    {
        IsTaskLike(returnType, out var resultType, out var isValueTask);
        if (resultType is null)
            return isValueTask ? refs.ReturnsValueTaskPlain : refs.ReturnsTaskPlain;

        var g = new GenericInstanceMethod(isValueTask ? refs.ReturnsValueTaskOfT : refs.ReturnsTaskOfT);
        g.GenericArguments.Add(resultType);
        return g;
    }

    // ----------------------------------------------------------------- state machines (async / iterators)
    //
    // An async or iterator method's visible body is a stub; the real code — and every egress — is the
    // compiler-generated state machine's MoveNext, which runs once per synchronous SEGMENT, each
    // possibly on a different thread. The weave gives the state machine ONE scope for the whole
    // logical invocation, held in a weaver-added field (the slot):
    //
    //   MoveNext prologue     scope = NDScope.OpenOrResume(ref this.<>ndscope)   [open or re-install]
    //   before Await*OnCompleted   suspended = true; NDScope.Suspend(scope)
    //       — BEFORE the schedule call, not in the finally: once the builder has the continuation it
    //         may already be resuming on another thread, and unlinking a scope two threads can see
    //         is a race. `suspended` is a LOCAL for the same reason: the finally must not read scope
    //         state a concurrent resumption mutates.
    //   before builder.SetResult(result)   result routed through Returns/YieldTo (survives disposal)
    //   before promise.SetResult(hasMore)  suspended = hasMore; NDScope.ExitIterator(ref slot, hasMore)
    //       — async iterators: the consumer can re-enter MoveNext the instant the promise is
    //         signalled, so the yield-suspension must also precede its signal.
    //   every stfld <>2__current           the yielded element routed through Returns/YieldTo — the
    //         consumer owns yielded elements (they must survive the enumerator's completion sweep).
    //   finally                async: if (!suspended) NDScope.DisposeSlot(ref slot)
    //                          iterator: if (retVar) NDScope.Suspend(scope) else DisposeSlot(ref slot)
    //       — a sync iterator's suspension IS `return true`, and its caller only regains control
    //         after MoveNext returns, so the finally decision is single-threaded there.
    //   iterator Dispose()     NDScope.DisposeSlot(ref slot) before each ret — mid-iteration
    //         abandonment (`break` out of foreach) reclaims deterministically.
    //
    // Tracked temps therefore stay ALIVE across awaits — an in-flight awaited callee may still be
    // using arrays handed to it — and are reclaimed when the INVOCATION completes (SetResult /
    // SetException / final MoveNext / enumerator Dispose), not when a segment ends. Hoisted locals
    // and parameters need no special handling: parameters were constructed before the scope opened
    // (never tracked), and hoisted locals stay correctly tracked because the scope now spans the
    // whole invocation.

    /// <summary>Identifies the state machine an attributed method compiled into (attribute-driven, so a plain method is None).</summary>
    internal static StateMachineKind GetStateMachineKind(MethodDefinition m, out TypeDefinition smType)
    {
        smType = null;
        foreach (var a in m.CustomAttributes)
        {
            StateMachineKind kind;
            switch (a.AttributeType.FullName)
            {
                case AsyncIteratorSmAttrFullName:
                    kind = StateMachineKind.AsyncIterator;
                    break;
                case AsyncSmAttrFullName:
                    kind = StateMachineKind.Async;
                    break;
                case IteratorSmAttrFullName:
                    kind = StateMachineKind.Iterator;
                    break;
                default:
                    continue;
            }

            if (a.ConstructorArguments.Count == 1 && a.ConstructorArguments[0].Value is TypeReference tr)
            {
                try
                {
                    smType = tr.Resolve();
                }
                catch
                {
                    smType = null;
                }
            }

            return kind;
        }

        return StateMachineKind.None;
    }

    /// <summary>
    ///     Validates and weaves one state-machine target. Every rejection is reported BEFORE any IL
    ///     mutation of consequence (and a failed run never writes the assembly, so a partial
    ///     mutation cannot ship regardless).
    /// </summary>
    internal static ValidationOutcome WeaveStateMachineTarget(MethodDefinition stub, TypeDefinition sm,
                                                            StateMachineKind kind, string label, Refs refs, TextWriter stderr)
    {
        if (sm is null)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW004: {label} method '{stub.FullName}' is an async/iterator method whose " +
                "state-machine type could not be resolved from its StateMachineAttribute");
            return ValidationOutcome.Error;
        }

        var moveNext = FindStateMachineMethod(sm, "MoveNext");
        if (moveNext is null || !moveNext.HasBody)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW004: {label} method '{stub.FullName}' has an unrecognized state-machine " +
                $"shape ('{sm.FullName}' has no MoveNext body) — only C#-compiled state machines are weavable");
            return ValidationOutcome.Error;
        }

        // Idempotence: a MoveNext that already installs the invocation scope (a previous pass over
        // the same assembly) is left alone. A hand-written NDScope.Open INSIDE a segment does NOT
        // skip — that inner scope nests fine under the invocation scope and cannot replace it (no
        // hand-written code can span the suspension seam).
        foreach (var instr in moveNext.Body.Instructions)
        {
            if (instr.OpCode.FlowControl == FlowControl.Call &&
                instr.Operand is MethodReference already &&
                already.Name == "OpenOrResume" &&
                already.DeclaringType.FullName == NDScopeFullName)
                return ValidationOutcome.AlreadyScoped;
        }

        if (!refs.HasAsyncSurface)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW008: {label} async/iterator method '{stub.FullName}' needs the async " +
                "scope seam (NDScope.OpenOrResume/Suspend/DisposeSlot/ExitIterator), which the referenced NumSharp " +
                "does not carry — update the NumSharp package");
            return ValidationOutcome.Error;
        }

        foreach (var instr in moveNext.Body.Instructions)
        {
            if (instr.OpCode == OpCodes.Tail)
            {
                stderr.WriteLine($"NumSharp.Build : error NDW007: {label} method '{stub.FullName}' contains a tail-call");
                return ValidationOutcome.Error;
            }
        }

        // -- shape discovery (all name pins are Roslyn's stable generated-name scheme; a miss is a
        //    loud NDW004, never a silent mis-weave)
        FieldDefinition builderField = null;
        if (kind is StateMachineKind.Async or StateMachineKind.AsyncIterator)
        {
            foreach (var f in sm.Fields)
                if (f.Name == BuilderFieldName)
                {
                    builderField = f;
                    break;
                }

            if (builderField is null)
                foreach (var f in sm.Fields)
                    if (f.FieldType.GetElementType().FullName.Contains("MethodBuilder"))
                    {
                        builderField = f;
                        break;
                    }

            if (builderField is null)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW004: {label} method '{stub.FullName}' has an unrecognized state-machine " +
                    $"shape ('{sm.FullName}' carries no method-builder field)");
                return ValidationOutcome.Error;
            }
        }

        // completionSites = builder/promise calls that COMPLETE the invocation without a result to
        // yield (resultless SetResult, every SetException): the scope is disposed immediately BEFORE
        // each — not in the finally — because the completion signal can run the CALLER's
        // continuation inline on this thread, and that caller code must not execute under (and
        // track into) a scope that is about to die.
        var awaitSites = new List<Instruction>();
        var resultSites = new List<Instruction>();
        var completionSites = new List<Instruction>();
        var promiseSites = new List<Instruction>();
        var currentSites = new List<Instruction>();
        foreach (var instr in moveNext.Body.Instructions)
        {
            if (instr.OpCode.FlowControl == FlowControl.Call && instr.Operand is MethodReference mr)
            {
                if (builderField != null &&
                    mr.Name is "AwaitUnsafeOnCompleted" or "AwaitOnCompleted" &&
                    SameElementType(mr.DeclaringType, builderField.FieldType))
                    awaitSites.Add(instr);
                else if (kind == StateMachineKind.Async &&
                         mr.Name == "SetResult" && mr.Parameters.Count == 1 &&
                         SameElementType(mr.DeclaringType, builderField.FieldType))
                    resultSites.Add(instr);
                else if (kind == StateMachineKind.Async &&
                         mr.Name == "SetResult" && mr.Parameters.Count == 0 &&
                         SameElementType(mr.DeclaringType, builderField.FieldType))
                    completionSites.Add(instr);
                else if (kind == StateMachineKind.Async &&
                         mr.Name == "SetException" && mr.Parameters.Count == 1 &&
                         SameElementType(mr.DeclaringType, builderField.FieldType))
                    completionSites.Add(instr);
                else if (kind == StateMachineKind.AsyncIterator &&
                         mr.Name == "SetResult" && mr.Parameters.Count == 1 &&
                         mr.DeclaringType.GetElementType().FullName == PromiseElementFullName)
                    promiseSites.Add(instr);
                else if (kind == StateMachineKind.AsyncIterator &&
                         mr.Name == "SetException" && mr.Parameters.Count == 1 &&
                         mr.DeclaringType.GetElementType().FullName == PromiseElementFullName)
                    completionSites.Add(instr);
            }
            else if (kind is StateMachineKind.Iterator or StateMachineKind.AsyncIterator &&
                     instr.OpCode == OpCodes.Stfld && instr.Operand is FieldReference fr &&
                     fr.Name == CurrentFieldName &&
                     fr.DeclaringType.GetElementType().FullName == sm.FullName)
                currentSites.Add(instr);
        }

        // -- egress-type classification: the async RESULT (SetResult's argument) / the iterator
        //    ELEMENT (<>2__current). The same vocabulary direct returns use; an unsupported type is
        //    the same NDW003 it would be there. A nested task result is refused: its completion
        //    outlives the state machine's own scope lifecycle, so nothing sound can be emitted.
        var resultKind = RetKind.Void;
        TypeReference resultType = null;
        if (resultSites.Count > 0)
        {
            resultType = ConcreteParamType((MethodReference)resultSites[0].Operand, 0);
            var k = Classify(resultType);
            if (k is null or RetKind.TaskLike)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW003: {label} async method '{stub.FullName}' produces a result of type " +
                    $"'{resultType.FullName}' — an unsupported carrier the weaver cannot see every NDArray through; " +
                    "return a supported shape (NDArray, NDArray[], a tuple of NDArrays, an INDArrayCarrier struct, " +
                    "or a scalar) or drop the attribute");
                return ValidationOutcome.Error;
            }

            resultKind = k.Value;
        }

        var currentKind = RetKind.Void;
        TypeReference currentType = null;
        if (kind is StateMachineKind.Iterator or StateMachineKind.AsyncIterator)
        {
            FieldDefinition currentField = null;
            foreach (var f in sm.Fields)
                if (f.Name == CurrentFieldName)
                {
                    currentField = f;
                    break;
                }

            if (currentField is null)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW004: {label} method '{stub.FullName}' has an unrecognized state-machine " +
                    $"shape ('{sm.FullName}' carries no <>2__current field)");
                return ValidationOutcome.Error;
            }

            currentType = currentField.FieldType;
            var k = Classify(currentType);
            if (k is null or RetKind.TaskLike)
            {
                stderr.WriteLine(
                    $"NumSharp.Build : error NDW003: {label} iterator method '{stub.FullName}' yields elements of type " +
                    $"'{currentType.FullName}' — an unsupported carrier the weaver cannot see every NDArray through; " +
                    "yield a supported shape (NDArray, NDArray[], a tuple of NDArrays, an INDArrayCarrier struct, " +
                    "or a scalar) or drop the attribute");
                return ValidationOutcome.Error;
            }

            currentKind = k.Value;
        }

        if (kind == StateMachineKind.AsyncIterator && promiseSites.Count == 0)
        {
            stderr.WriteLine(
                $"NumSharp.Build : error NDW004: {label} method '{stub.FullName}' has an unrecognized state-machine " +
                $"shape ('{sm.FullName}' signals no ManualResetValueTaskSourceCore promise)");
            return ValidationOutcome.Error;
        }

        // A result the weave has nothing to yield for still completes the invocation — its
        // SetResult joins the plain pre-completion disposal sites.
        if (resultSites.Count > 0 && resultKind is RetKind.Void or RetKind.Scalar)
        {
            completionSites.AddRange(resultSites);
            resultSites.Clear();
        }

        WeaveStateMachineMoveNext(moveNext, sm, kind, refs, awaitSites,
            resultSites, resultKind, resultType, completionSites, promiseSites, currentSites, currentKind, currentType);

        // A sync iterator abandoned mid-iteration exits through the enumerator's Dispose (foreach
        // always calls it), which is therefore the reclamation point for a consumer that breaks out
        // early. Async iterators reach completion through MoveNext itself (DisposeAsync drives it).
        if (kind == StateMachineKind.Iterator)
        {
            var dispose = FindStateMachineMethod(sm, "Dispose");
            if (dispose is { HasBody: true })
                WeaveIteratorDispose(dispose, sm, refs);
        }

        return ValidationOutcome.Ok;
    }

    private static void WeaveStateMachineMoveNext(MethodDefinition moveNext, TypeDefinition sm, StateMachineKind kind,
                                                  Refs refs, List<Instruction> awaitSites,
                                                  List<Instruction> resultSites, RetKind resultKind, TypeReference resultType,
                                                  List<Instruction> completionSites, List<Instruction> promiseSites,
                                                  List<Instruction> currentSites, RetKind currentKind, TypeReference currentType)
    {
        var body = moveNext.Body;
        body.SimplifyMacros();
        var il = body.GetILProcessor();
        var module = moveNext.Module;

        var slotRef = MakeSelfFieldRef(GetOrAddSlotField(sm, refs), sm);

        var scopeVar = new VariableDefinition(refs.NDScope);
        body.Variables.Add(scopeVar);
        bool isIterator = kind == StateMachineKind.Iterator;
        VariableDefinition suspendedVar = null;
        VariableDefinition retVar = null;
        if (isIterator)
        {
            retVar = new VariableDefinition(moveNext.ReturnType);
            body.Variables.Add(retVar);
        }
        else
        {
            suspendedVar = new VariableDefinition(module.TypeSystem.Boolean);
            body.Variables.Add(suspendedVar);
        }

        body.InitLocals = true; // suspendedVar/retVar must read false on paths that never assign them

        // -- prologue (OUTSIDE the protected region): scope = NDScope.OpenOrResume(ref this.<>ndscope)
        var tryStart = body.Instructions[0];
        il.InsertBefore(tryStart, il.Create(OpCodes.Ldarg, body.ThisParameter));
        il.InsertBefore(tryStart, il.Create(OpCodes.Ldflda, slotRef));
        il.InsertBefore(tryStart, il.Create(OpCodes.Call, refs.OpenOrResume));
        il.InsertBefore(tryStart, il.Create(OpCodes.Stloc, scopeVar));

        // -- epilogue (after the handler): [ldloc retVar;] ret
        Instruction epilogueStart;
        var finalRet = il.Create(OpCodes.Ret);
        if (retVar != null)
        {
            epilogueStart = il.Create(OpCodes.Ldloc, retVar);
            il.Append(epilogueStart);
            il.Append(finalRet);
        }
        else
        {
            epilogueStart = finalRet;
            il.Append(finalRet);
        }

        // -- finally handler body (between body and epilogue)
        Instruction finallyStart;
        var endFinally = il.Create(OpCodes.Endfinally);
        if (isIterator)
        {
            // if (retVar) Suspend(scope); else DisposeSlot(ref slot); — `return true` IS the
            // suspension, and the consumer only regains control after MoveNext returns, so deciding
            // here is single-threaded. An exception path never stored retVar: InitLocals reads
            // false, which correctly disposes (a faulted iterator does not resume).
            finallyStart = il.Create(OpCodes.Ldloc, retVar);
            var disposePath = il.Create(OpCodes.Ldarg, body.ThisParameter);
            il.InsertBefore(epilogueStart, finallyStart);
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Brfalse, disposePath));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Ldloc, scopeVar));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Call, refs.Suspend));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Br, endFinally));
            il.InsertBefore(epilogueStart, disposePath);
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Ldflda, slotRef));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Call, refs.DisposeSlot));
            il.InsertBefore(epilogueStart, endFinally);
        }
        else
        {
            // if (!suspended) DisposeSlot(ref slot); — a suspending exit already unlinked the scope
            // BEFORE the schedule/signal call, and the resumption (possibly already running on
            // another thread) owns it now; reading only the LOCAL here is what keeps this race-free.
            finallyStart = il.Create(OpCodes.Ldloc, suspendedVar);
            il.InsertBefore(epilogueStart, finallyStart);
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Brtrue, endFinally));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Ldarg, body.ThisParameter));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Ldflda, slotRef));
            il.InsertBefore(epilogueStart, il.Create(OpCodes.Call, refs.DisposeSlot));
            il.InsertBefore(epilogueStart, endFinally);
        }

        // -- rewrite every ORIGINAL ret (all precede finallyStart)
        var originalRets = new List<Instruction>();
        for (var instr = tryStart; instr != null && instr != finallyStart; instr = instr.Next)
            if (instr.OpCode == OpCodes.Ret)
                originalRets.Add(instr);

        foreach (var ret in originalRets)
        {
            if (retVar != null)
            {
                ret.OpCode = OpCodes.Stloc;
                ret.Operand = retVar;
                InsertAfter(il, ret, il.Create(OpCodes.Leave, epilogueStart));
            }
            else
            {
                ret.OpCode = OpCodes.Leave;
                ret.Operand = epilogueStart;
            }
        }

        // -- pre-schedule suspension: suspended = true; NDScope.Suspend(scope); <original call>
        //    (mutate-in-place so a branch targeting the schedule call cannot skip the suspension)
        foreach (var site in awaitSites)
        {
            var original = il.Create(site.OpCode, (MethodReference)site.Operand);
            site.OpCode = OpCodes.Ldc_I4;
            site.Operand = 1;
            var cursor = site;
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, suspendedVar));
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Call, refs.Suspend));
            InsertAfter(il, cursor, original);
        }

        // -- async result egress: builder.SetResult(scope-yielded result) — with the scope disposed
        //    BEFORE the signal (yield first, so the result survives the sweep): the signal can run
        //    the caller's continuation inline on this very thread, and that code must not execute
        //    under a scope that is about to die. The finally still runs — DisposeSlot on the now
        //    empty slot is a no-op.
        if (resultSites.Count > 0)
        {
            var tmpVal = new VariableDefinition(resultType);
            body.Variables.Add(tmpVal);
            foreach (var site in resultSites)
            {
                var original = il.Create(site.OpCode, (MethodReference)site.Operand);
                site.OpCode = OpCodes.Stloc;
                site.Operand = tmpVal;
                var cursor = EmitYieldValueOnStack(il, site, scopeVar, tmpVal, resultKind, resultType, refs);
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldarg, body.ThisParameter));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldflda, slotRef));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Call, refs.DisposeSlot));
                InsertAfter(il, cursor, original);
            }
        }

        // -- resultless completions (SetResult() / every SetException): dispose before the signal.
        //    The three inserted instructions are stack-neutral, so the call's pending arguments
        //    (builder&/promise&, the exception) ride undisturbed beneath them.
        foreach (var site in completionSites)
        {
            var original = il.Create(site.OpCode, (MethodReference)site.Operand);
            site.OpCode = OpCodes.Ldarg;
            site.Operand = body.ThisParameter;
            var cursor = site;
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldflda, slotRef));
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Call, refs.DisposeSlot));
            InsertAfter(il, cursor, original);
        }

        // -- async-iterator yield boundary: suspended = hasMore; ExitIterator(ref slot, hasMore);
        //    promise.SetResult(hasMore) — the scope must be off this thread BEFORE the consumer is
        //    signalled (it can re-enter MoveNext, even inline, the moment the promise completes).
        if (promiseSites.Count > 0)
        {
            var tmpBool = new VariableDefinition(module.TypeSystem.Boolean);
            body.Variables.Add(tmpBool);
            foreach (var site in promiseSites)
            {
                var original = il.Create(site.OpCode, (MethodReference)site.Operand);
                site.OpCode = OpCodes.Stloc;
                site.Operand = tmpBool;
                var cursor = site;
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, tmpBool));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, suspendedVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldarg, body.ThisParameter));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldflda, slotRef));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, tmpBool));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Call, refs.ExitIterator));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, tmpBool));
                InsertAfter(il, cursor, original);
            }
        }

        // -- yielded-element egress: this.<>2__current = scope-yielded value (the consumer owns it)
        if (currentSites.Count > 0 && currentKind is not (RetKind.Void or RetKind.Scalar))
        {
            var tmpCur = new VariableDefinition(currentType);
            body.Variables.Add(tmpCur);
            foreach (var site in currentSites)
            {
                var original = il.Create(OpCodes.Stfld, (FieldReference)site.Operand);
                site.OpCode = OpCodes.Stloc;
                site.Operand = tmpCur;
                var cursor = EmitYieldValueOnStack(il, site, scopeVar, tmpCur, currentKind, currentType, refs);
                InsertAfter(il, cursor, original);
            }
        }

        // -- the outer handler; appended last so pre-existing (nested) handlers stay first
        body.ExceptionHandlers.Add(new ExceptionHandler(ExceptionHandlerType.Finally)
        {
            TryStart = tryStart,
            TryEnd = finallyStart,
            HandlerStart = finallyStart,
            HandlerEnd = epilogueStart,
        });

        body.OptimizeMacros();
    }

    /// <summary>
    ///     The sync-iterator abandonment seam: <c>foreach</c> calls the enumerator's
    ///     <c>Dispose()</c> whether iteration finished or broke out early, so
    ///     <c>NDScope.DisposeSlot(ref slot)</c> before each of its returns reclaims a suspended
    ///     invocation scope deterministically (an already-completed one left the slot null — no-op).
    /// </summary>
    private static void WeaveIteratorDispose(MethodDefinition dispose, TypeDefinition sm, Refs refs)
    {
        var body = dispose.Body;
        body.SimplifyMacros();
        var il = body.GetILProcessor();
        var slotRef = MakeSelfFieldRef(GetOrAddSlotField(sm, refs), sm);

        var rets = new List<Instruction>();
        foreach (var instr in body.Instructions)
            if (instr.OpCode == OpCodes.Ret)
                rets.Add(instr);

        foreach (var ret in rets)
        {
            ret.OpCode = OpCodes.Ldarg;
            ret.Operand = body.ThisParameter;
            var cursor = ret;
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldflda, slotRef));
            cursor = InsertAfter(il, cursor, il.Create(OpCodes.Call, refs.DisposeSlot));
            InsertAfter(il, cursor, il.Create(OpCodes.Ret));
        }

        body.OptimizeMacros();
    }

    /// <summary>
    ///     Emits the scope egress for <paramref name="valueVar"/> leaving the (unchanged) value on
    ///     the evaluation stack — the shared tail of the async-result and yielded-element seams,
    ///     mirroring instruction-for-instruction what the return-path rewrite emits per kind.
    /// </summary>
    private static Instruction EmitYieldValueOnStack(ILProcessor il, Instruction cursor, VariableDefinition scopeVar,
                                                     VariableDefinition valueVar, RetKind kind, TypeReference valueType, Refs refs)
    {
        switch (kind)
        {
            case RetKind.NDArrayLike:
            case RetKind.NDArrayLikeArray:
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, InstantiateReturns(valueType, kind, refs)));
                break;

            case RetKind.NDArrayTuple:
                TryGetNDArrayTuple(valueType, out var tupleType);
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, InstantiateReturnsTuple(tupleType, refs)));
                break;

            case RetKind.Tuple:
                IsGeneralTuple(valueType, out var tupleIsValueType);
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                if (tupleIsValueType)
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Box, valueType));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, refs.ReturnsITuple));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Pop));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                break;

            case RetKind.Carrier:
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloca, valueVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Constrained, valueType));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, refs.CarrierYieldTo));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                break;

            case RetKind.Storage:
                var storageRef = valueType.FullName == UnmanagedStorageFullName ? refs.ReturnsStorage : refs.ReturnsSlice;
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, valueVar));
                cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, storageRef));
                break;
        }

        return cursor;
    }

    /// <summary>Locates a parameterless state-machine method by simple name, tolerating explicit-interface spellings.</summary>
    private static MethodDefinition FindStateMachineMethod(TypeDefinition sm, string simpleName)
    {
        foreach (var m in sm.Methods)
            if (m.Parameters.Count == 0 &&
                (m.Name == simpleName || m.Name.EndsWith("." + simpleName, StringComparison.Ordinal)))
                return m;
        return null;
    }

    /// <summary>The weaver-added scope slot on the state machine (added once; idempotent by name).</summary>
    private static FieldDefinition GetOrAddSlotField(TypeDefinition sm, Refs refs)
    {
        foreach (var f in sm.Fields)
            if (f.Name == SlotFieldName)
                return f;

        var field = new FieldDefinition(SlotFieldName, FieldAttributes.Private, refs.NDScope);
        sm.Fields.Add(field);
        return field;
    }

    /// <summary>
    ///     A field reference usable INSIDE the state machine's own methods: for a generic state
    ///     machine (a generic async method / iterator, or one in a generic type) field tokens must
    ///     target the type instantiated with its OWN generic parameters, exactly as the compiler
    ///     emits its own field accesses.
    /// </summary>
    private static FieldReference MakeSelfFieldRef(FieldDefinition field, TypeDefinition sm)
    {
        if (!sm.HasGenericParameters)
            return field;

        var self = new GenericInstanceType(sm);
        foreach (var gp in sm.GenericParameters)
            self.GenericArguments.Add(gp);
        return new FieldReference(field.Name, field.FieldType, self);
    }

    /// <summary>
    ///     A call-site parameter's CONCRETE type: <c>AsyncTaskMethodBuilder&lt;T&gt;.SetResult(!0)</c>
    ///     declares its parameter as the builder's generic parameter — the call site's generic
    ///     instantiation carries the actual result type.
    /// </summary>
    private static TypeReference ConcreteParamType(MethodReference mr, int index)
    {
        var p = mr.Parameters[index].ParameterType;
        if (p is GenericParameter gp && gp.Type == GenericParameterType.Type && mr.DeclaringType is GenericInstanceType git)
            return git.GenericArguments[gp.Position];
        return p;
    }

    private static bool SameElementType(TypeReference a, TypeReference b)
        => a.GetElementType().FullName == b.GetElementType().FullName;

    private static Instruction InsertAfter(ILProcessor il, Instruction anchor, Instruction instruction)
    {
        il.InsertAfter(anchor, instruction);
        return instruction;
    }
}
