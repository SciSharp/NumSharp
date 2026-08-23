using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

namespace NumSharp.Weaver;

internal readonly record struct WeaveResult(int Woven, int Skipped, int Errors, bool SymbolsWritten);

/// <summary>
///     The transform. Per <c>[NDScoped]</c> method:
///     <list type="number">
///     <item>prologue OUTSIDE the protected region: <c>scope = NDScope.Open()</c> into a fresh local;</item>
///     <item>the whole ORIGINAL body becomes the try of a try/finally whose finally is
///           <c>scope.Dispose()</c> — the CLR forbids <c>ret</c> inside a protected region, so every
///           original return is forced through a rewritable seam;</item>
///     <item>each original <c>ret</c> is rewritten IN PLACE (the ret instruction object is mutated into
///           the first replacement instruction, so branches that targeted it stay valid): the return
///           value lands in a local, NDArray-like values are routed through <c>scope.Returns(value)</c>,
///           each <c>out NDArray</c> parameter's FINAL value is yielded (success path only — on a throw
///           the finally reclaims, and out contents are undefined after a throw anyway), then
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

    private sealed class Refs
    {
        public TypeDefinition NDScope;
        public MethodDefinition Open;        // static NDScope Open()
        public MethodDefinition Dispose;     // void Dispose()
        public MethodDefinition ReturnsOne;  // T Returns<T>(T)     where T : NDArray
        public MethodDefinition ReturnsMany; // T[] Returns<T>(T[]) where T : NDArray
    }

    private enum RetKind
    {
        Void,
        Scalar,          // primitives / enums / decimal / Half / Complex / string — scope only
        NDArrayLike,     // NDArray or a subclass (NDArray<T>) — Returns<T>(T)
        NDArrayLikeArray // NDArray-like[] — Returns<T>(T[])
    }

    public static WeaveResult WeaveAssembly(string assemblyPath, byte[] snkBlob, bool verbose, TextWriter stdout, TextWriter stderr)
    {
        var readerParameters = new ReaderParameters
        {
            ReadSymbols = true,
            SymbolReaderProvider = new DefaultSymbolReaderProvider(throwIfNoSymbol: false),
            InMemory = true, // fully load, release the file handle: we write back to the same path
        };

        using var assembly = AssemblyDefinition.ReadAssembly(assemblyPath, readerParameters);
        var module = assembly.MainModule;
        bool hasSymbols = module.HasSymbols;

        var refs = ResolveRefs(module);
        if (refs is null)
        {
            // No NDScope in this module: nothing can be woven. Attributed methods would be a
            // configuration error, but without the scope type there is nothing to look for either.
            stdout.WriteLine($"NumSharp.Weaver: {Path.GetFileName(assemblyPath)} — no {NDScopeFullName} type; nothing to do");
            return new WeaveResult(0, 0, 0, false);
        }

        int woven = 0, skipped = 0, errors = 0;

        foreach (var method in CollectTargets(module, stderr, ref errors))
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
            return new WeaveResult(woven, skipped, errors, false);

        var writerParameters = new WriterParameters
        {
            WriteSymbols = hasSymbols,
        };
        if (snkBlob != null)
            writerParameters.StrongNameKeyBlob = snkBlob;

        assembly.Write(assemblyPath, writerParameters);
        return new WeaveResult(woven, skipped, 0, hasSymbols);
    }

    // ----------------------------------------------------------------- target collection

    private static IEnumerable<MethodDefinition> CollectTargets(ModuleDefinition module, TextWriter stderr, ref int errors)
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
                "a carrier type the weaver cannot yield through scope.Returns (its NDArray members would be " +
                "reclaimed and handed to the caller disposed); scope this method by hand and yield each member");
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
        var scopeType = module.GetType(NDScopeFullName);
        if (scopeType is null)
            return null;

        var refs = new Refs { NDScope = scopeType };
        foreach (var m in scopeType.Methods)
        {
            switch (m.Name)
            {
                case "Open" when m.IsStatic && !m.HasParameters:
                    refs.Open = m;
                    break;
                case "Dispose" when !m.IsStatic && !m.HasParameters:
                    refs.Dispose = m;
                    break;
                case "Returns" when m.HasGenericParameters && m.Parameters.Count == 1:
                    if (m.Parameters[0].ParameterType is ArrayType)
                        refs.ReturnsMany = m;
                    else
                        refs.ReturnsOne = m;
                    break;
            }
        }

        if (refs.Open is null || refs.Dispose is null || refs.ReturnsOne is null || refs.ReturnsMany is null)
            throw new InvalidOperationException(
                $"{NDScopeFullName} is missing one of Open/Dispose/Returns<T>(T)/Returns<T>(T[]) — weaver and library out of sync");

        return refs;
    }

    /// <summary>NDArray-like, or a rank-1 array of NDArray-like — the two shapes Returns can yield.</summary>
    private static bool IsNDArrayCarrying(TypeReference t)
        => IsNDArrayLike(t) || (t is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType));

    /// <summary>NDArray or any subclass (NDArray&lt;T&gt; open or instantiated). Never resolves outside the NumSharp module.</summary>
    private static bool IsNDArrayLike(TypeReference t)
    {
        if (t is null || t.IsByReference || t.IsPointer || t is ArrayType || t.IsGenericParameter)
            return false;
        if (t.FullName == NDArrayFullName)
            return true;
        if (!t.FullName.StartsWith("NumSharp", StringComparison.Ordinal))
            return false; // never chase external types (no assembly resolver is configured)

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
            if (baseRef is null || !baseRef.FullName.StartsWith("NumSharp", StringComparison.Ordinal))
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

    /// <summary>Null = rejected (carrier struct / unsupported): the caller reports NDW003.</summary>
    private static RetKind? Classify(TypeReference ret)
    {
        if (ret.MetadataType == MetadataType.Void)
            return RetKind.Void;

        if (IsNDArrayLike(ret))
            return RetKind.NDArrayLike;

        if (ret is ArrayType { Rank: 1 } arr && IsNDArrayLike(arr.ElementType))
            return RetKind.NDArrayLikeArray;

        switch (ret.MetadataType)
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
                return RetKind.Scalar;
        }

        switch (ret.FullName)
        {
            case "System.Decimal":
            case "System.Half":
            case "System.Numerics.Complex":
                return RetKind.Scalar;
        }

        // In-module enums (NPTypeCode etc.) are scalars; anything else — value tuples, result
        // structs, object, collections — is a carrier the weaver must not guess about.
        if (ret.FullName.StartsWith("NumSharp", StringComparison.Ordinal))
        {
            try
            {
                if (ret.Resolve() is { IsEnum: true })
                    return RetKind.Scalar;
            }
            catch
            {
                // fall through to rejection
            }
        }

        return null;
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

                if (retKind is RetKind.NDArrayLike or RetKind.NDArrayLikeArray)
                {
                    var returnsRef = InstantiateReturns(m.ReturnType, retKind, refs);
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, scopeVar));
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Ldloc, retVar));
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Callvirt, returnsRef));
                    cursor = InsertAfter(il, cursor, il.Create(OpCodes.Stloc, retVar));
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

    private static Instruction InsertAfter(ILProcessor il, Instruction anchor, Instruction instruction)
    {
        il.InsertAfter(anchor, instruction);
        return instruction;
    }
}
