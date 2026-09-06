using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;

namespace NumSharp.Build.Analyzer
{
    /// <summary>The scope-family attributes a declaration can carry, as flags (a declaration carrying two is the NDW011 error).</summary>
    [System.Flags]
    internal enum ScopeKind
    {
        None = 0,
        Sync = 1,    // [NDScoped]
        Async = 2,   // [NDScopedAsync]
        Covered = 4, // [NDScopedCovered] — never a weave target; on an override it OPTS OUT of an inherited scope
    }

    /// <summary>
    ///     Where a method's scoping comes from: the method itself (<see cref="Inherited"/> false — the
    ///     attribute sits on it, or on its property for a getter) or the nearest attributed declaration
    ///     up its override/implementation graph.
    /// </summary>
    internal readonly struct ScopeDeclaration
    {
        /// <summary>The symbol carrying the attribute: the method, or the property when the attribute is property-level.</summary>
        public readonly ISymbol Source;
        public readonly ScopeKind Kinds;
        public readonly bool Inherited;

        public ScopeDeclaration(ISymbol source, ScopeKind kinds, bool inherited)
        {
            Source = source;
            Kinds = kinds;
            Inherited = inherited;
        }
    }

    /// <summary>
    ///     Declaration-site inheritance of the scope attributes, over Roslyn symbols — the analyzer's
    ///     twin of the weaver's <c>ScopeInheritance</c> (tools/NumSharp.Build), walking the SAME graph so
    ///     both layers classify a method identically: <c>[NDScoped]</c> / <c>[NDScopedAsync]</c> /
    ///     <c>[NDScopedCovered]</c> on a virtual, abstract or interface member is a contract every
    ///     override and implementation inherits unless it carries a scope-family attribute of its own —
    ///     and <c>[NDScopedExit]</c> on a declaration's parameters is inherited BY POSITION by every
    ///     override and implementation that marks no parameter of its own (the weaver's
    ///     <c>EffectiveExitParameters</c> rule).
    ///     <para>
    ///     From the method outward: its explicit interface implementations, the implicit interface
    ///     implementations at its level (the interface member this type maps onto this very method —
    ///     <see cref="ITypeSymbol.FindImplementationForInterfaceMember"/>, the CLR's own rule), and the
    ///     overridden method — recursing into each base declaration so an interface listed on a base,
    ///     or an attribute two levels up, is still found. A property-level attribute reaches its getter
    ///     (the weaver's getter rule); generic bases and interfaces are matched through Roslyn's
    ///     substituted symbols, and declarations in referenced assemblies (a consumer overriding a
    ///     NumSharp member) are metadata symbols and walk the same way. The walk is ONE routine
    ///     (<see cref="FindNearest"/>) parameterised by what counts as a declaration — the scope
    ///     attributes for <see cref="Inherited"/>, an attributed parameter for
    ///     <see cref="InheritedExitSource"/> — exactly as the weaver's <c>FindNearest</c> is.
    ///     </para>
    /// </summary>
    internal static class ScopeInheritance
    {
        private const int MaxDepth = 64;

        /// <summary>The scope-family attributes placed directly on <paramref name="symbol"/>.</summary>
        public static ScopeKind KindsOf(ISymbol symbol, KnownTypes k)
        {
            var kinds = ScopeKind.None;
            if (symbol == null)
                return kinds;
            foreach (var a in symbol.GetAttributes())
            {
                var cls = a.AttributeClass;
                if (cls == null)
                    continue;
                if (k.SyncAttr != null && SymbolEqualityComparer.Default.Equals(cls, k.SyncAttr))
                    kinds |= ScopeKind.Sync;
                else if (k.AsyncAttr != null && SymbolEqualityComparer.Default.Equals(cls, k.AsyncAttr))
                    kinds |= ScopeKind.Async;
                else if (k.CoveredAttr != null && SymbolEqualityComparer.Default.Equals(cls, k.CoveredAttr))
                    kinds |= ScopeKind.Covered;
            }

            return kinds;
        }

        /// <summary>
        ///     The scope-family attributes <paramref name="m"/> itself carries — on the method, or (for a
        ///     getter) on its property; <paramref name="carrier"/> is whichever symbol carries them.
        /// </summary>
        public static ScopeKind OwnKinds(IMethodSymbol m, KnownTypes k, out ISymbol carrier)
        {
            carrier = m;
            var kinds = KindsOf(m, k);
            if (kinds != ScopeKind.None)
                return kinds;

            // A property-level attribute resolves to the GETTER (the weaver's CollectTargets rule; the
            // setter carries nothing from it).
            if (m.MethodKind == MethodKind.PropertyGet && m.AssociatedSymbol is IPropertySymbol p)
            {
                kinds = KindsOf(p, k);
                if (kinds != ScopeKind.None)
                    carrier = p;
            }

            return kinds;
        }

        /// <summary>The attribute(s) in effect for <paramref name="m"/>: its own, else the nearest inherited declaration's, else null.</summary>
        public static ScopeDeclaration? Effective(IMethodSymbol m, KnownTypes k)
        {
            var own = OwnKinds(m, k, out var carrier);
            if (own != ScopeKind.None)
                return new ScopeDeclaration(carrier, own, inherited: false);
            return Inherited(m, k);
        }

        /// <summary>
        ///     The nearest declaration up <paramref name="m"/>'s override/implementation graph carrying a
        ///     scope-family attribute — <paramref name="m"/> itself excluded — or null.
        /// </summary>
        public static ScopeDeclaration? Inherited(IMethodSymbol m, KnownTypes k)
        {
            var source = FindNearest(m, k, HasOwnScope);
            if (source == null)
                return null;
            var kinds = OwnKinds(source, k, out var carrier);
            return new ScopeDeclaration(carrier, kinds, inherited: true);
        }

        private static bool HasOwnScope(IMethodSymbol d, KnownTypes k) => OwnKinds(d, k, out _) != ScopeKind.None;

        // ----------------------------------------------------------------- [NDScopedExit]

        /// <summary>
        ///     The positions of <paramref name="m"/>'s OWN <c>[NDScopedExit]</c> parameters (the weaver's
        ///     <c>OwnExitParameters</c>): empty when it marks none, or when the referenced NumSharp predates
        ///     the attribute. A property setter's <c>value</c> (<c>[param: NDScopedExit]</c>) counts.
        /// </summary>
        public static int[] OwnExitParameters(IMethodSymbol m, KnownTypes k)
        {
            if (m == null || k.ExitAttr == null || m.Parameters.Length == 0)
                return Array.Empty<int>();

            List<int> indexes = null;
            for (int i = 0; i < m.Parameters.Length; i++)
            {
                foreach (var a in m.Parameters[i].GetAttributes())
                {
                    if (a.AttributeClass == null || !SymbolEqualityComparer.Default.Equals(a.AttributeClass, k.ExitAttr))
                        continue;
                    (indexes ?? (indexes = new List<int>())).Add(i);
                    break;
                }
            }

            return indexes == null ? Array.Empty<int>() : indexes.ToArray();
        }

        /// <summary>True when <paramref name="m"/> itself marks at least one parameter <c>[NDScopedExit]</c>.</summary>
        public static bool HasOwnExitParameter(IMethodSymbol m, KnownTypes k) => OwnExitParameters(m, k).Length > 0;

        /// <summary>
        ///     The nearest declaration up <paramref name="m"/>'s override/implementation graph that marks a
        ///     parameter <c>[NDScopedExit]</c> — <paramref name="m"/> itself excluded — or null. Only
        ///     meaningful when <paramref name="m"/> marks none of its own: an override that marks ANY
        ///     parameter uses only its own marks (the weaver's <c>EffectiveExitParameters</c>).
        /// </summary>
        public static IMethodSymbol InheritedExitSource(IMethodSymbol m, KnownTypes k)
            => k.ExitAttr == null ? null : FindNearest(m, k, HasOwnExitParameter);

        // ----------------------------------------------------------------- the walk

        /// <summary>
        ///     The nearest declaration up <paramref name="m"/>'s override/implementation graph for which
        ///     <paramref name="match"/> holds — <paramref name="m"/> itself excluded — or null.
        /// </summary>
        private static IMethodSymbol FindNearest(IMethodSymbol m, KnownTypes k, Func<IMethodSymbol, KnownTypes, bool> match)
        {
            if (m == null || m.IsStatic)
                return null;
            // Only an override or an interface implementation can inherit; the CLR-level hint for an
            // implicit interface implementation is `virtual`, which Roslyn reports as IsVirtual on the
            // source symbol ONLY when spelled so — so instead every non-static instance method is
            // considered, and FindImplementationForInterfaceMember decides.
            var visited = new HashSet<ISymbol>(SymbolEqualityComparer.Default) { m.OriginalDefinition };
            return Visit(m, k, visited, 0, match);
        }

        private static IMethodSymbol Visit(IMethodSymbol node, KnownTypes k, HashSet<ISymbol> visited, int depth,
            Func<IMethodSymbol, KnownTypes, bool> match)
        {
            if (depth > MaxDepth)
                return null;

            // 1. Explicit interface implementations — an interface member is a leaf (it neither
            //    overrides a class slot nor implements a base interface's member).
            foreach (var e in node.ExplicitInterfaceImplementations)
            {
                var d = e.OriginalDefinition;
                if (!visited.Add(d))
                    continue;
                if (match(d, k))
                    return d;
            }

            if (node.IsStatic)
                return null;

            // 2. Implicit interface implementations at this level: the interface members THIS type maps
            //    onto this very method (a public member whose name and signature match).
            var type = node.ContainingType;
            if (type != null && type.TypeKind != TypeKind.Interface && node.DeclaredAccessibility == Accessibility.Public)
            {
                foreach (var iface in type.AllInterfaces)
                {
                    foreach (var member in iface.GetMembers(node.Name))
                    {
                        if (!(member is IMethodSymbol im))
                            continue;
                        var impl = type.FindImplementationForInterfaceMember(im) as IMethodSymbol;
                        if (impl == null || !SymbolEqualityComparer.Default.Equals(impl.OriginalDefinition, node.OriginalDefinition))
                            continue;
                        var d = im.OriginalDefinition;
                        if (!visited.Add(d))
                            continue;
                        if (match(d, k))
                            return d;
                    }
                }
            }

            // 3. The overridden method — the base declaration continues the walk from ITS level.
            if (node.IsOverride && node.OverriddenMethod != null)
            {
                var b = node.OverriddenMethod.OriginalDefinition;
                if (!visited.Add(b))
                    return null;
                if (match(b, k))
                    return b;
                return Visit(b, k, visited, depth + 1, match);
            }

            return null;
        }
    }
}
