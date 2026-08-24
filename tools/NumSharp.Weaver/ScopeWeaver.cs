using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

namespace NumSharp.Weaver;

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
/// </summary>
internal static class ScopeWeaver
{
    private const string AttributeFullName = "NumSharp.NDScopedAttribute";
    private const string NDScopeFullName = "NumSharp.NDScope";
    private const string NDArrayFullName = "NumSharp.NDArray";
    private const string CarrierInterfaceFullName = "NumSharp.INDArrayCarrier";
    private const string ITupleFullName = "System.Runtime.CompilerServices.ITuple";
    private const string ValueTuplePrefix = "System.ValueTuple`";
    private const string TuplePrefix = "System.Tuple`";
    private const string IArraySliceFullName = "NumSharp.Backends.Unmanaged.IArraySlice";
    private const string UnmanagedStorageFullName = "NumSharp.Backends.UnmanagedStorage";

    /// <summary>
    ///     The NDScope surface the transform emits calls to. Every member is held as a reference
    ///     ALREADY IMPORTED into the module being woven: for the self-weave (NumSharp.Core, where
    ///     NDScope is in-module) <c>ImportReference</c> is an identity pass-through, and for a
    ///     consumer weave it mints the cross-assembly <c>MemberRef</c>s targeting the referenced
    ///     NumSharp assembly.
    /// </summary>
    private sealed class Refs
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

        /// <summary>The <c>Returns</c> tuple overload for the given arity (2..4), or null if unsupported.</summary>
        public MethodReference ReturnsTuple(int arity) => arity switch
        {
            2 => ReturnsTuple2,
            3 => ReturnsTuple3,
            4 => ReturnsTuple4,
            _ => null
        };
    }

    private enum RetKind
    {
        Void,
        Scalar,           // primitives / enums / decimal / Half / Complex / string — scope only
        NDArrayLike,      // NDArray or a subclass (NDArray<T>) — Returns<T>(T)
        NDArrayLikeArray, // NDArray-like[] — Returns<T>(T[])
        NDArrayTuple,     // ValueTuple<..> of 2..4 NDArray-likes — Returns<T..>((T..)) overload (no box)
        Tuple,            // any OTHER ValueTuple/Tuple (arity 5..8, a non-NDArray component, Tuple<>) — Returns(ITuple)
        Carrier,          // in-module result struct — INDArrayCarrier.YieldTo(scope)
        Storage           // bare IArraySlice / UnmanagedStorage — Returns(slice/storage) counted-ref protection
    }

    public static WeaveResult WeaveAssembly(string assemblyPath, byte[] snkBlob, IReadOnlyList<string> referencePaths,
                                            bool verbose, TextWriter stdout, TextWriter stderr)
    {
        // The resolver serves BOTH weave shapes: the self-weave (NumSharp.Core — everything
        // in-module, resolver only probes the assembly's own directory) and the consumer weave
        // (the NumSharp.Weaver package — NDScope/NDArray live in the REFERENCED NumSharp assembly,
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

        // Targets FIRST (a pure name match, no resolution): a consumer assembly with no [NDScoped]
        // methods exits without resolving anything and — crucially — without rewriting the file,
        // so its compile-time signature, determinism id and timestamps stay untouched.
        var targets = CollectTargets(module, stderr, ref errors);
        if (targets.Count == 0)
        {
            if (errors > 0)
                return new WeaveResult(0, 0, errors, false, false);
            stdout.WriteLine($"NumSharp.Weaver: {Path.GetFileName(assemblyPath)} — no [NDScoped] methods; nothing to do");
            return new WeaveResult(0, 0, 0, false, false);
        }

        var refs = ResolveRefs(module);
        if (refs is null)
        {
            // Attributed methods with no reachable NDScope is a CONFIGURATION error, not a no-op:
            // silently skipping would ship those methods unwoven while the build reads green.
            stderr.WriteLine(
                $"NumSharp.Weaver : error NDW001: {targets.Count} [NDScoped] method(s) found but type '{NDScopeFullName}' " +
                $"could not be resolved from '{Path.GetFileName(assemblyPath)}' or its references — ensure the project " +
                "references NumSharp (the attribute and the scope live there) and that the invocation passed --refs " +
                "with the compile's reference list");
            return new WeaveResult(0, 0, errors + 1, false, false);
        }

        foreach (var method in targets)
        {
            switch (Validate(method, stderr))
            {
                case ValidationOutcome.AlreadyScoped:
                    skipped++;
                    if (verbose)
                        stdout.WriteLine($"NumSharp.Weaver: skip (already opens an NDScope): {method.FullName}");
                    continue;
                case ValidationOutcome.Error:
                    errors++;
                    continue;
            }

            WeaveMethod(method, refs);
            woven++;
            if (verbose)
                stdout.WriteLine($"NumSharp.Weaver: woven: {method.FullName}");
        }

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

    // ----------------------------------------------------------------- target collection

    private static List<MethodDefinition> CollectTargets(ModuleDefinition module, TextWriter stderr, ref int errors)
    {
        var targets = new List<MethodDefinition>();
        foreach (var type in AllTypes(module))
        {
            foreach (var method in type.Methods)
                if (HasScopedAttribute(method.CustomAttributes))
                    targets.Add(method);

            foreach (var property in type.Properties)
            {
                if (!HasScopedAttribute(property.CustomAttributes))
                    continue;
                if (property.GetMethod is null)
                {
                    stderr.WriteLine(
                        $"NumSharp.Weaver : error NDW006: [NDScoped] on property '{property.FullName}' " +
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

    private static bool HasScopedAttribute(ICollection<CustomAttribute> attributes)
    {
        foreach (var a in attributes)
            if (a.AttributeType.FullName == AttributeFullName)
                return true;
        return false;
    }

    // ----------------------------------------------------------------- validation

    private enum ValidationOutcome
    {
        Ok,
        AlreadyScoped,
        Error
    }

    private static ValidationOutcome Validate(MethodDefinition m, TextWriter stderr)
    {
        if (!m.HasBody)
        {
            stderr.WriteLine($"NumSharp.Weaver : error NDW005: [NDScoped] method '{m.FullName}' has no body (abstract/extern)");
            return ValidationOutcome.Error;
        }

        // Iterators / async compile to state machines: the visible body is a stub whose real
        // egress lives in MoveNext — a scope here would close before the first element/await.
        foreach (var a in m.CustomAttributes)
        {
            var n = a.AttributeType.FullName;
            if (n is "System.Runtime.CompilerServices.AsyncStateMachineAttribute"
                or "System.Runtime.CompilerServices.IteratorStateMachineAttribute")
            {
                stderr.WriteLine($"NumSharp.Weaver : error NDW004: [NDScoped] cannot weave iterator/async method '{m.FullName}'");
                return ValidationOutcome.Error;
            }
        }

        foreach (var p in m.Parameters)
        {
            if (p.ParameterType is ByReferenceType brt && IsNDArrayCarrying(brt.ElementType) && !p.IsOut)
            {
                stderr.WriteLine(
                    $"NumSharp.Weaver : error NDW002: [NDScoped] method '{m.FullName}' has 'ref {brt.ElementType.Name}' " +
                    $"parameter '{p.Name}' — a hidden egress the weaver cannot see; scope this method by hand");
                return ValidationOutcome.Error;
            }
        }

        if (Classify(m.ReturnType) is null)
        {
            stderr.WriteLine(
                $"NumSharp.Weaver : error NDW003: [NDScoped] method '{m.FullName}' returns '{m.ReturnType.FullName}' — " +
                "an unsupported carrier the weaver cannot see every NDArray through (a bespoke reference type, a " +
                "collection, or a result struct that does NOT implement INDArrayCarrier), so its NDArray members would " +
                "be reclaimed and handed to the caller disposed; add INDArrayCarrier to the struct, or scope this " +
                "method by hand. (NDArray, NDArray[], any ValueTuple/Tuple of NDArrays, INDArrayCarrier result " +
                "structs, and bare IArraySlice/UnmanagedStorage ARE woven.)");
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
                stderr.WriteLine($"NumSharp.Weaver : error NDW007: [NDScoped] method '{m.FullName}' contains a tail-call");
                return ValidationOutcome.Error;
            }
        }

        return ValidationOutcome.Ok;
    }

    // ----------------------------------------------------------------- type classification

    private static Refs ResolveRefs(ModuleDefinition module)
    {
        // In-module first (the self-weave — NumSharp.Core carries NDScope itself), then the
        // assembly references (the consumer weave — the NumSharp.Weaver package's target hands the
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
    private static RetKind? Classify(TypeReference ret)
    {
        if (ret.MetadataType == MetadataType.Void)
            return RetKind.Void;

        if (IsNDArrayLike(ret))
            return RetKind.NDArrayLike;

        if (ret is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType))
            return RetKind.NDArrayLikeArray;

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

    private static void WeaveMethod(MethodDefinition m, Refs refs)
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

        // -- finally handler body: scope.Dispose(); endfinally  (between body and epilogue)
        var finallyStart = il.Create(OpCodes.Ldloc, scopeVar);
        il.InsertBefore(epilogueStart, finallyStart);
        il.InsertBefore(epilogueStart, il.Create(OpCodes.Callvirt, refs.Dispose));
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

    private static Instruction InsertAfter(ILProcessor il, Instruction anchor, Instruction instruction)
    {
        il.InsertAfter(anchor, instruction);
        return instruction;
    }
}
