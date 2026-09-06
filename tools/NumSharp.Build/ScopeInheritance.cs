using Mono.Cecil;

namespace NumSharp.Build;

/// <summary>The scope-family attributes a declaration can carry, as flags (a declaration carrying two is an NDW011 error).</summary>
[Flags]
internal enum ScopeKind
{
    None = 0,
    Sync = 1,    // [NDScoped]
    Async = 2,   // [NDScopedAsync]
    Covered = 4, // [NDScopedCovered] — never a weave target; on an override it OPTS OUT of an inherited scope
}

/// <summary>
///     Declaration-site inheritance of the scope attributes. <c>[NDScoped]</c> / <c>[NDScopedAsync]</c>
///     (and the <c>[NDScopedExit]</c> parameter attribute) placed on a <b>virtual</b>, <b>abstract</b> or
///     <b>interface</b> member is a CONTRACT for every override and implementation: an override that
///     carries no scope-family attribute of its own inherits the nearest attributed declaration up its
///     override / implementation graph and is woven exactly as if it carried the attribute itself. An
///     explicit attribute on the override always wins (<c>[NDScopedCovered]</c> is the opt-out), and a
///     body-less attributed declaration (abstract / interface) is the contract, not a weave target.
///     <para>
///     The graph walked, from the method outward — mirroring how the CLR resolves the virtual slot:
///     the explicit <c>.override</c> entries (MethodImpl table: explicit interface implementations,
///     C# 9 covariant returns), the implicit interface implementations of the declaring type (a
///     public virtual method matching an interface member's name and signature — the base interfaces
///     of a listed interface included), and the class base chain (a reuse-slot virtual matched by name
///     and signature, climbing past bases that do not declare the slot). Generic bases are matched
///     under their instantiation (<c>Derived : Base&lt;int&gt;</c> matches <c>Base&lt;T&gt;.M(T)</c>
///     against <c>M(int)</c>), the walk crosses assembly boundaries through the module's
///     <see cref="WeaverAssemblyResolver"/> (a consumer overriding a NumSharp — or any referenced
///     library's — scoped virtual inherits it), and framework types are never climbed into (they cannot
///     carry NumSharp attributes). A property-level attribute on a base property reaches its getter
///     override, matching <see cref="ScopeWeaver.CollectTargets"/>'s getter rule.
///     </para>
///     <para>
///     One instance per woven module (held on <see cref="ScopeWeaver.Refs"/>); results are memoized
///     per <see cref="MethodDefinition"/>, so the collection pass and the per-target validation share
///     one walk. The compile-time analyzer (<c>NumSharp.Build.Analyzer</c>, <c>ScopeInheritance</c>
///     there) walks the SAME graph over Roslyn symbols so both layers classify identically.
///     </para>
/// </summary>
internal sealed class ScopeInheritance
{
    /// <summary>A bound on the override/implementation depth walked — valid metadata never gets close; it only guards against a corrupt cyclic base chain.</summary>
    private const int MaxDepth = 64;

    private const string CoveredAttributeFullName = "NumSharp.NDScopedCoveredAttribute";

    /// <summary>
    ///     The declaration a method's scoping comes from: the method itself (<see cref="Inherited"/>
    ///     false) or the nearest attributed declaration up its override/implementation graph.
    /// </summary>
    internal readonly struct Declaration
    {
        public readonly MethodDefinition Source;
        public readonly ScopeKind Kinds;
        public readonly bool Inherited;

        public Declaration(MethodDefinition source, ScopeKind kinds, bool inherited)
        {
            Source = source;
            Kinds = kinds;
            Inherited = inherited;
        }
    }

    private readonly Dictionary<MethodDefinition, Declaration?> _scope = new();
    private readonly Dictionary<MethodDefinition, int[]> _exit = new();
    private readonly Dictionary<MethodDefinition, MethodDefinition> _exitSource = new();

    // ----------------------------------------------------------------- public surface

    /// <summary>
    ///     The scope-family attribute(s) in effect for <paramref name="m"/>: its own (any of
    ///     <c>[NDScoped]</c>/<c>[NDScopedAsync]</c>/<c>[NDScopedCovered]</c>, a getter's property-level
    ///     attribute included), else the nearest attributed declaration it overrides or implements,
    ///     else null.
    /// </summary>
    public Declaration? EffectiveScope(MethodDefinition m)
    {
        if (_scope.TryGetValue(m, out var cached))
            return cached;

        Declaration? result = null;
        var own = OwnScopeKinds(m);
        if (own != ScopeKind.None)
            result = new Declaration(m, own, inherited: false);
        else
        {
            var source = FindNearest(m, static def => OwnScopeKinds(def) != ScopeKind.None);
            if (source != null)
                result = new Declaration(source, OwnScopeKinds(source), inherited: true);
        }

        _scope[m] = result;
        return result;
    }

    /// <summary>
    ///     The parameter positions that are <c>[NDScopedExit]</c> for <paramref name="m"/>: its own
    ///     attributed parameters when it declares any, else — by position — those of the nearest
    ///     declaration it overrides or implements that declares some, else empty.
    /// </summary>
    public int[] EffectiveExitParameters(MethodDefinition m)
    {
        if (_exit.TryGetValue(m, out var cached))
            return cached;

        var own = OwnExitParameters(m);
        if (own.Length == 0)
        {
            var source = FindNearest(m, static def => OwnExitParameters(def).Length > 0);
            if (source != null)
            {
                // Positions transfer only where the override kept the arity — a candidate is matched
                // on name AND signature, so this is always the case; clamp anyway rather than index past.
                own = OwnExitParameters(source).Where(i => i < m.Parameters.Count).ToArray();
                _exitSource[m] = source;
            }
        }

        _exit[m] = own;
        return own;
    }

    /// <summary>The declaration <paramref name="m"/>'s <c>[NDScopedExit]</c> parameters were inherited from, or null when they are its own (or there are none).</summary>
    public MethodDefinition ExitParametersSource(MethodDefinition m)
    {
        EffectiveExitParameters(m);
        return _exitSource.TryGetValue(m, out var source) ? source : null;
    }

    /// <summary>The diagnostic/log suffix naming an inherited declaration — empty for a method's own attribute.</summary>
    public static string Provenance(Declaration? declaration)
        => declaration is { Inherited: true } d ? $" (inherited from {d.Source.DeclaringType.FullName}::{d.Source.Name})" : "";

    /// <summary>The scope-family attributes <paramref name="m"/> itself carries — on the method, or (for a getter) on its property.</summary>
    public static ScopeKind OwnScopeKinds(MethodDefinition m)
    {
        var kinds = KindsOf(m.CustomAttributes);
        if (kinds != ScopeKind.None)
            return kinds;

        // A property-level attribute resolves to the GETTER (CollectTargets' rule; a setter carries
        // nothing from it — NDW006 covers the setter-only case at the declaration).
        if (m.IsGetter)
        {
            foreach (var p in m.DeclaringType.Properties)
                if (ReferenceEquals(p.GetMethod, m))
                    return KindsOf(p.CustomAttributes);
        }

        return ScopeKind.None;
    }

    // ----------------------------------------------------------------- the walk

    private static ScopeKind KindsOf(ICollection<CustomAttribute> attributes)
    {
        var kinds = ScopeKind.None;
        foreach (var a in attributes)
        {
            switch (a.AttributeType.FullName)
            {
                case ScopeWeaver.SyncAttributeFullName:
                    kinds |= ScopeKind.Sync;
                    break;
                case ScopeWeaver.AsyncAttributeFullName:
                    kinds |= ScopeKind.Async;
                    break;
                case CoveredAttributeFullName:
                    kinds |= ScopeKind.Covered;
                    break;
            }
        }

        return kinds;
    }

    private static int[] OwnExitParameters(MethodDefinition m)
    {
        List<int> indexes = null;
        for (int i = 0; i < m.Parameters.Count; i++)
        {
            foreach (var a in m.Parameters[i].CustomAttributes)
            {
                if (a.AttributeType.FullName != ScopeWeaver.ExitAttributeFullName)
                    continue;
                (indexes ??= new List<int>()).Add(i);
                break;
            }
        }

        return indexes is null ? Array.Empty<int>() : indexes.ToArray();
    }

    /// <summary>
    ///     The generic instantiation a base type / interface was referenced with, as seen from the
    ///     type below it: <see cref="Args"/> are that reference's generic arguments (expressed in the
    ///     lower type's own terms), <see cref="Outer"/> the context those terms live in. A type-level
    ///     generic parameter at some level resolves through its level's map, then the result is
    ///     compared in the next level down — so <c>Derived : Base&lt;int&gt;</c>, <c>Base&lt;T&gt; :
    ///     Root&lt;T&gt;</c> compares <c>Root&lt;T&gt;.M(T)</c> against the derived <c>M(int)</c>.
    /// </summary>
    private sealed class GenericContext
    {
        public readonly IList<TypeReference> Args;
        public readonly GenericContext Outer;

        public GenericContext(IList<TypeReference> args, GenericContext outer)
        {
            Args = args;
            Outer = outer;
        }
    }

    private static GenericContext ContextOf(TypeReference reference, GenericContext outer)
        => reference is GenericInstanceType git ? new GenericContext(git.GenericArguments, outer) : null;

    /// <summary>
    ///     The nearest method up <paramref name="m"/>'s override/implementation graph for which
    ///     <paramref name="carries"/> holds — <paramref name="m"/> itself excluded — or null.
    /// </summary>
    private static MethodDefinition FindNearest(MethodDefinition m, Func<MethodDefinition, bool> carries)
    {
        // Only a virtual instance method can override or implement anything; everything else is its
        // own declaration. (The C# compiler marks implicit interface implementations virtual too.)
        if (m.IsStatic || !m.IsVirtual)
            return null;

        var visited = new HashSet<MethodDefinition> { m };
        return Visit(m, null, m, carries, visited, 0);
    }

    /// <summary>
    ///     Checks everything <paramref name="node"/> reaches in one step — its explicit overrides, the
    ///     implicit interface implementations at its level, its class base chain — recursing into each
    ///     base declaration found. <paramref name="signature"/> is the ORIGINAL method (an override
    ///     keeps its base's signature modulo generic instantiation, so it is the comparand at every level).
    /// </summary>
    private static MethodDefinition Visit(MethodDefinition node, GenericContext nodeContext, MethodDefinition signature,
                                          Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited, int depth)
    {
        if (depth > MaxDepth)
            return null;

        // 1. Explicit .override entries (the MethodImpl table): explicit interface implementations,
        //    covariant-return overrides, and anything else the compiler pinned to a specific slot.
        foreach (var overrideRef in node.Overrides)
        {
            var def = SafeResolve(overrideRef);
            if (def is null || !visited.Add(def))
                continue;
            if (carries(def))
                return def;
            var found = Visit(def, ContextOf(overrideRef.DeclaringType, nodeContext), signature, carries, visited, depth + 1);
            if (found != null)
                return found;
        }

        if (node.IsStatic || !node.IsVirtual)
            return null;

        // An interface member neither implements a base interface's member nor overrides a class
        // slot (a same-named member in a base interface is a HIDING, not an implementation).
        if (node.DeclaringType.IsInterface)
            return null;

        // 2. Implicit interface implementations at this level: a PUBLIC virtual method whose name and
        //    signature match a member of an interface the declaring type lists (or inherits through
        //    that interface's own bases) implements it.
        if (node.IsPublic)
        {
            var found = MatchInterfaces(node.DeclaringType, nodeContext, signature, carries, visited, depth + 1);
            if (found != null)
                return found;
        }

        // 3. The class base chain — a reuse-slot virtual overrides the nearest base declaring the same
        //    slot. A newslot virtual starts a fresh slot and overrides nothing in the base chain.
        if (node.IsNewSlot)
            return null;

        var context = nodeContext;
        var baseRef = node.DeclaringType.BaseType;
        while (baseRef != null && !ScopeWeaver.IsFrameworkType(baseRef))
        {
            var baseDef = SafeResolve(baseRef);
            if (baseDef is null)
                return null;

            var baseContext = ContextOf(baseRef, context);
            var candidate = FindMatching(baseDef, signature, baseContext, requireVirtual: true);
            if (candidate != null)
            {
                if (!visited.Add(candidate))
                    return null;
                if (carries(candidate))
                    return candidate;
                // The base declaration continues the walk from ITS level (its own overrides,
                // interfaces and bases).
                return Visit(candidate, baseContext, signature, carries, visited, depth + 1);
            }

            // This base does not declare the slot; it may still LIST an interface the slot implements
            // (the CLR maps an interface listed on a base onto the most derived matching method).
            if (node.IsPublic)
            {
                var found = MatchInterfaces(baseDef, baseContext, signature, carries, visited, depth + 1);
                if (found != null)
                    return found;
            }

            context = baseContext;
            baseRef = baseDef.BaseType;
        }

        return null;
    }

    private static MethodDefinition MatchInterfaces(TypeDefinition type, GenericContext typeContext, MethodDefinition signature,
                                                    Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited, int depth)
    {
        foreach (var impl in type.Interfaces)
        {
            var found = MatchInterface(impl.InterfaceType, typeContext, signature, carries, visited, depth);
            if (found != null)
                return found;
        }

        return null;
    }

    private static MethodDefinition MatchInterface(TypeReference interfaceRef, GenericContext outer, MethodDefinition signature,
                                                   Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited, int depth)
    {
        if (depth > MaxDepth || ScopeWeaver.IsFrameworkType(interfaceRef))
            return null;

        var def = SafeResolve(interfaceRef);
        if (def is null)
            return null;

        var context = ContextOf(interfaceRef, outer);
        var candidate = FindMatching(def, signature, context, requireVirtual: false);
        if (candidate != null && visited.Add(candidate) && carries(candidate))
            return candidate;

        // Listing an interface implements its base interfaces' members too.
        foreach (var sub in def.Interfaces)
        {
            var found = MatchInterface(sub.InterfaceType, context, signature, carries, visited, depth + 1);
            if (found != null)
                return found;
        }

        return null;
    }

    /// <summary>
    ///     The method of <paramref name="type"/> that declares <paramref name="signature"/>'s slot:
    ///     same name, instance-ness, generic arity and parameter count, and parameter types equal
    ///     under <paramref name="context"/>'s instantiation. The return type is deliberately NOT
    ///     compared (a C# covariant-return override differs there and reaches its base through the
    ///     explicit <c>.override</c> entry anyway).
    /// </summary>
    private static MethodDefinition FindMatching(TypeDefinition type, MethodDefinition signature, GenericContext context, bool requireVirtual)
    {
        foreach (var candidate in type.Methods)
        {
            if (candidate.IsStatic || candidate.Name != signature.Name)
                continue;
            if (requireVirtual && !candidate.IsVirtual)
                continue;
            if (candidate.GenericParameters.Count != signature.GenericParameters.Count ||
                candidate.Parameters.Count != signature.Parameters.Count)
                continue;

            bool same = true;
            for (int i = 0; i < candidate.Parameters.Count && same; i++)
                same = SameType(signature.Parameters[i].ParameterType, candidate.Parameters[i].ParameterType, context, 0);
            if (same)
                return candidate;
        }

        return null;
    }

    /// <summary>
    ///     Structural type equality with the candidate side (<paramref name="b"/>) read under its
    ///     declaring type's instantiation: a type-level generic parameter of the candidate resolves
    ///     through <paramref name="context"/> to the argument the lower level supplied, and is then
    ///     compared in THAT level's context; a method-level generic parameter matches by position.
    /// </summary>
    private static bool SameType(TypeReference a, TypeReference b, GenericContext context, int depth)
    {
        if (depth > MaxDepth || a is null || b is null)
            return false;

        if (b is GenericParameter gpB)
        {
            if (gpB.Type == GenericParameterType.Type)
            {
                if (context?.Args != null && gpB.Position < context.Args.Count)
                    return SameType(a, context.Args[gpB.Position], context.Outer, depth + 1);
                return a is GenericParameter gpA && gpA.Type == GenericParameterType.Type && gpA.Position == gpB.Position;
            }

            return a is GenericParameter gpM && gpM.Type == GenericParameterType.Method && gpM.Position == gpB.Position;
        }

        if (a is GenericParameter)
            return false;

        switch (b)
        {
            case ArrayType arrB:
                return a is ArrayType arrA && arrA.Rank == arrB.Rank && SameType(arrA.ElementType, arrB.ElementType, context, depth + 1);
            case ByReferenceType refB:
                return a is ByReferenceType refA && SameType(refA.ElementType, refB.ElementType, context, depth + 1);
            case PointerType ptrB:
                return a is PointerType ptrA && SameType(ptrA.ElementType, ptrB.ElementType, context, depth + 1);
            case PinnedType pinB:
                return a is PinnedType pinA && SameType(pinA.ElementType, pinB.ElementType, context, depth + 1);
            case RequiredModifierType reqB:
                return a is RequiredModifierType reqA && reqA.ModifierType.FullName == reqB.ModifierType.FullName &&
                       SameType(reqA.ElementType, reqB.ElementType, context, depth + 1);
            case OptionalModifierType optB:
                return a is OptionalModifierType optA && optA.ModifierType.FullName == optB.ModifierType.FullName &&
                       SameType(optA.ElementType, optB.ElementType, context, depth + 1);
            case GenericInstanceType gitB:
            {
                if (a is not GenericInstanceType gitA || gitA.GenericArguments.Count != gitB.GenericArguments.Count ||
                    gitA.ElementType.FullName != gitB.ElementType.FullName)
                    return false;
                for (int i = 0; i < gitA.GenericArguments.Count; i++)
                    if (!SameType(gitA.GenericArguments[i], gitB.GenericArguments[i], context, depth + 1))
                        return false;
                return true;
            }
            default:
                return a is not GenericInstanceType && a is not TypeSpecification && b is not TypeSpecification &&
                       a.FullName == b.FullName;
        }
    }

    private static TypeDefinition SafeResolve(TypeReference reference)
    {
        try
        {
            return reference?.Resolve();
        }
        catch
        {
            return null; // an unresolvable base/interface simply ends that branch of the walk
        }
    }

    private static MethodDefinition SafeResolve(MethodReference reference)
    {
        try
        {
            return reference?.Resolve();
        }
        catch
        {
            return null;
        }
    }
}
