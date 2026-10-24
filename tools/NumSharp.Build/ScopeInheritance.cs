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
    ///     <paramref name="carries"/> holds — <paramref name="m"/> itself excluded — or null. Runs for
    ///     every virtual method of every type the weaver loads (twice: the scope and the exit walks),
    ///     so its cost is linear in the hierarchy it climbs: each method and each interface
    ///     instantiation is examined once per walk.
    /// </summary>
    /// <param name="m">The method whose declarations are searched.</param>
    /// <param name="carries">The test that makes a declaration the answer.</param>
    /// <returns>The nearest declaration passing <paramref name="carries"/>, or null.</returns>
    private static MethodDefinition FindNearest(MethodDefinition m, Func<MethodDefinition, bool> carries)
    {
        // Only a virtual instance method can override or implement anything; everything else is its
        // own declaration. (The C# compiler marks implicit interface implementations virtual too.)
        if (m.IsStatic || !m.IsVirtual)
            return null;

        var visited = new HashSet<MethodDefinition> { m };
        var searched = new Dictionary<string, int>(StringComparer.Ordinal);
        return Visit(m, null, m, carries, visited, searched, 0);
    }

    /// <summary>
    ///     Checks everything <paramref name="node"/> reaches in one step — its explicit overrides, the
    ///     implicit interface implementations at its level, its class base chain — recursing into each
    ///     base declaration found. <paramref name="signature"/> is the ORIGINAL method (an override
    ///     keeps its base's signature modulo generic instantiation, so it is the comparand at every level).
    /// </summary>
    /// <param name="node">The declaration examined at this step.</param>
    /// <param name="nodeContext">The instantiation context of <paramref name="node"/>'s declaring type (null at the method's own level).</param>
    /// <param name="signature">The original method, the comparand at every level.</param>
    /// <param name="carries">The test that makes a declaration the answer.</param>
    /// <param name="visited">The declarations already examined in this walk (each is tested once).</param>
    /// <param name="searched">The interface instantiations already searched in this walk, with the shallowest depth each was searched at (see <see cref="MatchInterface"/>).</param>
    /// <param name="depth">The walk's depth, bounded by <see cref="MaxDepth"/> (a guard against corrupt cyclic metadata).</param>
    /// <returns>The first declaration passing <paramref name="carries"/>, or null.</returns>
    private static MethodDefinition Visit(MethodDefinition node, GenericContext nodeContext, MethodDefinition signature,
                                          Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited,
                                          Dictionary<string, int> searched, int depth)
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
            var found = Visit(def, ContextOf(overrideRef.DeclaringType, nodeContext), signature, carries, visited, searched, depth + 1);
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
            var found = MatchInterfaces(node.DeclaringType, nodeContext, signature, carries, visited, searched, depth + 1);
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
                return Visit(candidate, baseContext, signature, carries, visited, searched, depth + 1);
            }

            // This base does not declare the slot; it may still LIST an interface the slot implements
            // (the CLR maps an interface listed on a base onto the most derived matching method).
            if (node.IsPublic)
            {
                var found = MatchInterfaces(baseDef, baseContext, signature, carries, visited, searched, depth + 1);
                if (found != null)
                    return found;
            }

            context = baseContext;
            baseRef = baseDef.BaseType;
        }

        return null;
    }

    /// <summary>The first declaration passing <paramref name="carries"/> among the interfaces <paramref name="type"/> lists (and their bases).</summary>
    /// <param name="type">The type whose listed interfaces are searched.</param>
    /// <param name="typeContext">The instantiation context of <paramref name="type"/>.</param>
    /// <param name="signature">The original method, the comparand.</param>
    /// <param name="carries">The test that makes a declaration the answer.</param>
    /// <param name="visited">The declarations already examined in this walk.</param>
    /// <param name="searched">The interface instantiations already searched in this walk.</param>
    /// <param name="depth">The walk's depth.</param>
    /// <returns>The first match, or null.</returns>
    private static MethodDefinition MatchInterfaces(TypeDefinition type, GenericContext typeContext, MethodDefinition signature,
                                                    Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited,
                                                    Dictionary<string, int> searched, int depth)
    {
        foreach (var impl in type.Interfaces)
        {
            var found = MatchInterface(impl.InterfaceType, typeContext, signature, carries, visited, searched, depth);
            if (found != null)
                return found;
        }

        return null;
    }

    /// <summary>
    ///     The first declaration passing <paramref name="carries"/> in one listed interface — its member
    ///     matching <paramref name="signature"/>'s slot, then its base interfaces, depth first.
    /// </summary>
    /// <param name="interfaceRef">The interface as the listing type references it.</param>
    /// <param name="outer">The instantiation context of the listing type.</param>
    /// <param name="signature">The original method, the comparand.</param>
    /// <param name="carries">The test that makes a declaration the answer.</param>
    /// <param name="visited">The declarations already examined in this walk.</param>
    /// <param name="searched">The interface instantiations already searched in this walk, with the shallowest depth each was searched at; updated here.</param>
    /// <param name="depth">The walk's depth.</param>
    /// <returns>The first match, or null.</returns>
    private static MethodDefinition MatchInterface(TypeReference interfaceRef, GenericContext outer, MethodDefinition signature,
                                                   Func<MethodDefinition, bool> carries, HashSet<MethodDefinition> visited,
                                                   Dictionary<string, int> searched, int depth)
    {
        if (depth > MaxDepth || ScopeWeaver.IsFrameworkType(interfaceRef))
            return null;

        // The C# compiler lists an interface's WHOLE base-interface closure on it (and on every type
        // implementing it), so the recursion below reaches each base interface once per path through
        // the hierarchy: 2^k searches for a chain of k interfaces, and a class implementing a 26-deep
        // chain took a minute to weave. An interface searched before under the same instantiation,
        // at the same or a shallower depth, found nothing then (the walk would have ended) and finds
        // nothing now: every candidate it holds is already in `visited`. Skipping it leaves the first
        // match, and so the result, unchanged.
        var key = InstantiationKey(interfaceRef, outer);
        if (key != null)
        {
            if (searched.TryGetValue(key, out int searchedAt) && searchedAt <= depth)
                return null;
            searched[key] = depth;
        }

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
            var found = MatchInterface(sub.InterfaceType, context, signature, carries, visited, searched, depth + 1);
            if (found != null)
                return found;
        }

        return null;
    }

    /// <summary>
    ///     The identity <see cref="MatchInterface"/> deduplicates on: the interface's full name with every
    ///     generic argument read through <paramref name="outer"/> down to the terms of the method's own
    ///     declaring type. Two references reaching one interface DEFINITION under different
    ///     instantiations (<c>IFoo&lt;int&gt;</c> and <c>IFoo&lt;string&gt;</c> through two generic bases)
    ///     get different keys and are both searched — their members compare differently against the
    ///     signature — while the same instantiation reached along two paths gets one key.
    /// </summary>
    /// <param name="interfaceRef">The interface as the type listing it references it.</param>
    /// <param name="outer">The instantiation context of the listing type (null at the method's own level).</param>
    /// <returns>The key, or null when the type is too deep to key (the caller then searches without deduplicating — never wrong, only slower).</returns>
    private static string InstantiationKey(TypeReference interfaceRef, GenericContext outer)
    {
        var sb = new System.Text.StringBuilder();
        return AppendInstantiation(sb, interfaceRef, outer, 0) ? sb.ToString() : null;
    }

    /// <summary>Appends <paramref name="t"/>'s instantiation key, resolving type-level generic parameters through <paramref name="context"/>.</summary>
    /// <param name="sb">The key being built.</param>
    /// <param name="t">The type to append.</param>
    /// <param name="context">The context <paramref name="t"/>'s type-level generic parameters resolve in.</param>
    /// <param name="depth">Recursion depth, bounded by <see cref="MaxDepth"/>.</param>
    /// <returns>False when the type nests deeper than <see cref="MaxDepth"/> (no key).</returns>
    private static bool AppendInstantiation(System.Text.StringBuilder sb, TypeReference t, GenericContext context, int depth)
    {
        if (depth > MaxDepth || t is null)
            return false;

        switch (t)
        {
            case GenericParameter gp when gp.Type == GenericParameterType.Type:
                // A parameter of the level that wrote this reference: the argument the level below
                // supplied for it, read in THAT level's context; at the method's own level it stays a
                // parameter (by position — the declaring type's own parameters are one set).
                if (context?.Args != null && gp.Position < context.Args.Count)
                    return AppendInstantiation(sb, context.Args[gp.Position], context.Outer, depth + 1);
                sb.Append('!').Append(gp.Position);
                return true;
            case GenericParameter gp:
                sb.Append("!!").Append(gp.Position);
                return true;
            case GenericInstanceType git:
                sb.Append(git.ElementType.FullName).Append('<');
                for (int i = 0; i < git.GenericArguments.Count; i++)
                {
                    if (i > 0)
                        sb.Append(',');
                    if (!AppendInstantiation(sb, git.GenericArguments[i], context, depth + 1))
                        return false;
                }
                sb.Append('>');
                return true;
            case ArrayType arr:
                if (!AppendInstantiation(sb, arr.ElementType, context, depth + 1))
                    return false;
                sb.Append('[').Append(arr.Rank).Append(']');
                return true;
            case IModifierType mod when t is TypeSpecification modSpec:
                // modreq/modopt: the modifier is part of the type's identity.
                if (!AppendInstantiation(sb, modSpec.ElementType, context, depth + 1))
                    return false;
                sb.Append(t.MetadataType == MetadataType.RequiredModifier ? " modreq(" : " modopt(")
                  .Append(mod.ModifierType.FullName).Append(')');
                return true;
            case TypeSpecification spec when spec is ByReferenceType || spec is PointerType || spec is PinnedType:
                if (!AppendInstantiation(sb, spec.ElementType, context, depth + 1))
                    return false;
                sb.Append('|').Append(t.MetadataType);
                return true;
            default:
                sb.Append(t.FullName);
                return true;
        }
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
