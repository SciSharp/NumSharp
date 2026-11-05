using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     The per-compilation index of the NAMES through which NumSharp can reach code - the analyzers' pay-for-play
    ///     gate. It is built once per compilation from syntax alone (type and member declarations, never a method body)
    ///     plus the type names of the referenced assemblies that can name NDArray, and answers three questions cheaply
    ///     and conservatively:
    ///     <list type="bullet">
    ///         <item><see cref="BodyMayInvolveNumSharp"/> - can a method body produce an owned NDArray value at all
    ///         (NDW012 walks a body's operations only when it can: creating the operation tree of every body costs more
    ///         than everything else the analyzer does);</item>
    ///         <item><see cref="SourceTypeMayHold"/> - can a source type store NDArrays (NDW016/NDW017 and the
    ///         ownership model bind a type's members only when it can);</item>
    ///         <item><see cref="MayInheritScope"/> - can a method inherit a scope-family attribute or an
    ///         <c>[NDScopedExit]</c> parameter (the inheritance walks run only when it can).</item>
    ///     </list>
    ///     <para>
    ///     <b>Why names are enough.</b> An owned value enters a method body only through a symbol whose signature
    ///     involves an owned type: a call or an operator returning one, a constructor of one, a field, property, event or
    ///     local of such a type, an inherited member of a type that derives from one. Every such symbol is reached by
    ///     NAME - a token in the body, in the method's own declaration, or the declaration of the type the body belongs
    ///     to (members of the containing type and of its bases are reachable without a token, so a relevant containing
    ///     type makes every body of it a candidate). So a body none of whose tokens is a relevant name, inside a type
    ///     that is not relevant, cannot hold an owned value. The relevant names are a least fixed point over the
    ///     declarations: the seeds (NDArray, the carrier interface and every type name of a referenced assembly that can
    ///     name NDArray), then every member whose declared type mentions a relevant name, every type whose base list,
    ///     constraints, primary-constructor parameters or any member signature mention one, and every alias whose
    ///     target does. Names stand for every symbol of that name, so a collision (a local named <c>sum</c> in a project
    ///     where <c>sum</c> also returns an NDArray) only makes a body a candidate - the index never removes a body or a
    ///     type that the full analysis would report on, it only skips work whose answer is known to be empty.
    ///     </para>
    ///     <para>
    ///     Thread-safety: built eagerly (the constructor finishes before any analyzer callback reads it); the sets are
    ///     then read-only, and the two lazy memos are concurrent dictionaries of pure answers.
    ///     </para>
    /// </summary>
    internal sealed class NumSharpNameIndex
    {
        /// <summary>The attribute names (with and without the Attribute suffix) that make a declaration a scope contract.</summary>
        private static readonly HashSet<string> ScopeAttributeNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "NDScoped", "NDScopedAttribute", "NDScopedAsync", "NDScopedAsyncAttribute",
            "NDScopedCovered", "NDScopedCoveredAttribute",
        };

        /// <summary>The parameter attribute names (with and without the suffix) that make a parameter a retained-argument contract.</summary>
        private static readonly HashSet<string> ExitAttributeNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "NDScopedExit", "NDScopedExitAttribute",
        };

        /// <summary>N: the names a body must mention (or its containing type must be relevant) for it to possibly produce an owned value.</summary>
        private readonly HashSet<string> _bodyNames;

        /// <summary>Source type names whose declaration involves a relevant name: every body of such a type is a candidate.</summary>
        private readonly HashSet<string> _relevantTypeNames;

        /// <summary>Source type names that may store NDArrays (a least fixed point over storage declarations).</summary>
        private readonly HashSet<string> _holdingTypeNames;

        /// <summary>Names of inheritable members (interface members, virtual / abstract / override members) that carry a scope-family attribute or an [NDScopedExit] parameter; getters appear as get_X.</summary>
        private readonly HashSet<string> _scopedDeclarationNames;

        /// <summary>The compilation's own assembly (a source type is never a metadata ancestor).</summary>
        private readonly IAssemblySymbol _sourceAssembly;

        /// <summary>NumSharp's assembly (null when NumSharp is not referenced; the index then answers false everywhere).</summary>
        private readonly IAssemblySymbol _numSharp;

        /// <summary>Per assembly: whether it is NumSharp or references it, so its types can name NDArray and carry NumSharp attributes.</summary>
        private readonly ConcurrentDictionary<IAssemblySymbol, bool> _numSharpAware =
            new ConcurrentDictionary<IAssemblySymbol, bool>(SymbolEqualityComparer.Default);

        /// <summary>Per type: whether its base chain or interfaces include a type from a NumSharp-aware METADATA assembly.</summary>
        private readonly ConcurrentDictionary<INamedTypeSymbol, bool> _metadataAncestor =
            new ConcurrentDictionary<INamedTypeSymbol, bool>(SymbolEqualityComparer.Default);

        /// <summary>Builds the index for one compilation; see the type summary for what it answers and why.</summary>
        /// <param name="compilation">The compilation (its syntax trees and its references are read).</param>
        /// <param name="known">The NumSharp symbols the compilation resolved (NDArray must be non-null).</param>
        /// <exception cref="ArgumentNullException"><paramref name="compilation"/> or <paramref name="known"/> is null.</exception>
        public NumSharpNameIndex(Compilation compilation, KnownTypes known)
        {
            if (compilation == null)
                throw new ArgumentNullException(nameof(compilation));
            if (known == null)
                throw new ArgumentNullException(nameof(known));

            _sourceAssembly = compilation.Assembly;
            _numSharp = known.NumSharpAssembly;

            var declarations = new List<TypeDeclarationFacts>();
            var aliases = new List<AliasFacts>();
            var staticImports = new HashSet<string>(StringComparer.Ordinal);
            _scopedDeclarationNames = new HashSet<string>(StringComparer.Ordinal);
            var scopeAttributeNames = new HashSet<string>(ScopeAttributeNames, StringComparer.Ordinal);
            var exitAttributeNames = new HashSet<string>(ExitAttributeNames, StringComparer.Ordinal);

            // Aliases of the scope attributes come first: a [S] whose alias targets NDScopedAttribute must read as the
            // attribute when the declarations are scanned below.
            foreach (var tree in compilation.SyntaxTrees)
                CollectAttributeAliases(tree.GetRoot(), scopeAttributeNames, exitAttributeNames);

            foreach (var tree in compilation.SyntaxTrees)
                CollectDeclarations(tree.GetRoot(), declarations, aliases, staticImports, scopeAttributeNames, exitAttributeNames);

            // The seeds: the owned base vocabulary by name, plus every type name of a referenced assembly that can name
            // NDArray (over-approximated: not only the types that hold or return arrays), the extension methods such an
            // assembly exports (callable without naming their type) and the static members of its types a `using static`
            // imports (callable unqualified).
            var seeds = new HashSet<string>(StringComparer.Ordinal) { known.NDArray.Name };
            if (known.INDArrayCarrier != null)
                seeds.Add(known.INDArrayCarrier.Name);
            foreach (var assembly in compilation.SourceModule.ReferencedAssemblySymbols)
                if (!SymbolEqualityComparer.Default.Equals(assembly, _sourceAssembly) && IsNumSharpAware(assembly))
                    AddMetadataNames(assembly, seeds, staticImports);

            _holdingTypeNames = HoldingFixedPoint(seeds, declarations, aliases);
            (_bodyNames, _relevantTypeNames) = RelevanceFixedPoint(seeds, declarations, aliases);
        }

        // ------------------------------------------------------------------ the three questions

        /// <summary>
        ///     Whether the body of <paramref name="method"/> can produce an owned value: its containing type is relevant,
        ///     or a token of its declaration (signature and body - for an accessor, the whole property, indexer or event
        ///     declaration) is a relevant name. False means NDW012 has nothing to report there and the body's operation
        ///     tree need not be created.
        /// </summary>
        /// <param name="method">The method whose operation block is about to be analyzed.</param>
        /// <returns>True when the body must be analyzed.</returns>
        public bool BodyMayInvolveNumSharp(IMethodSymbol method)
        {
            if (method == null)
                return false;
            if (_numSharp == null)
                return false;

            for (var t = method.ContainingType; t != null; t = t.ContainingType)
                if (_relevantTypeNames.Contains(t.Name))
                    return true;

            var references = method.DeclaringSyntaxReferences;
            if (references.Length == 0)
                return true; // nothing to read: analyze rather than guess

            foreach (var reference in references)
            {
                var node = MemberDeclarationOf(reference.GetSyntax());
                foreach (var token in node.DescendantTokens())
                    if (token.IsKind(SyntaxKind.IdentifierToken) && _bodyNames.Contains(token.ValueText))
                        return true;
            }

            return false;
        }

        /// <summary>
        ///     Whether a value of <paramref name="type"/> - a SOURCE type, generic or not - can hold NDArrays through
        ///     its own declaration (its storing members, its base, its constraints). False for a source type means its
        ///     members never need binding: the type stores nothing that reaches an array. A constructed generic of a
        ///     non-holding definition can still hold through its type arguments; the ownership model follows those.
        /// </summary>
        /// <param name="type">A type declared in this compilation.</param>
        /// <returns>True when the declaration may store NDArrays.</returns>
        public bool SourceTypeMayHold(INamedTypeSymbol type)
            => type != null && _numSharp != null && _holdingTypeNames.Contains(type.Name);

        /// <summary>
        ///     Whether <paramref name="method"/> can inherit a scope-family attribute or an <c>[NDScopedExit]</c>
        ///     parameter from a declaration it overrides or implements. An override and an implicit implementation keep
        ///     the declaration's name, an explicit implementation names it, and a getter inherits from the getter of a
        ///     base property; so the method can inherit only when one of those names is a scoped inheritable declaration
        ///     of the compilation, or when its type derives from or implements a type of a NumSharp-aware METADATA
        ///     assembly (whose declarations the index does not read).
        /// </summary>
        /// <param name="method">The method about to be walked.</param>
        /// <returns>False when the inheritance walk would certainly find nothing.</returns>
        public bool MayInheritScope(IMethodSymbol method)
        {
            if (method == null || method.IsStatic || _numSharp == null)
                return false;

            if (_scopedDeclarationNames.Contains(method.Name))
                return true;
            foreach (var e in method.ExplicitInterfaceImplementations)
                if (_scopedDeclarationNames.Contains(e.Name))
                    return true;
            if (method.AssociatedSymbol is IPropertySymbol p && _scopedDeclarationNames.Contains(p.Name))
                return true;

            var type = method.ContainingType;
            return type != null && HasNumSharpMetadataAncestor(type);
        }

        /// <summary>Whether types of <paramref name="assembly"/> can name NDArray (it is NumSharp, or it references NumSharp). Memoized.</summary>
        /// <param name="assembly">The assembly asked about; null answers false.</param>
        /// <returns>True for a NumSharp-aware assembly.</returns>
        public bool IsNumSharpAware(IAssemblySymbol assembly)
        {
            if (assembly == null || _numSharp == null)
                return false;
            return _numSharpAware.GetOrAdd(assembly, a =>
            {
                if (SymbolEqualityComparer.Default.Equals(a, _numSharp))
                    return true;
                foreach (var module in a.Modules)
                    foreach (var referenced in module.ReferencedAssemblySymbols)
                        if (SymbolEqualityComparer.Default.Equals(referenced, _numSharp))
                            return true;
                return false;
            });
        }

        /// <summary>Whether <paramref name="type"/>'s base chain or interfaces include a type of a NumSharp-aware metadata assembly. Memoized per type.</summary>
        /// <param name="type">A source type.</param>
        /// <returns>True when a declaration the index did not read could be inherited.</returns>
        private bool HasNumSharpMetadataAncestor(INamedTypeSymbol type)
            => _metadataAncestor.GetOrAdd(type, t =>
            {
                for (var b = t.BaseType; b != null; b = b.BaseType)
                    if (IsNumSharpMetadataType(b))
                        return true;
                foreach (var i in t.AllInterfaces)
                    if (IsNumSharpMetadataType(i))
                        return true;
                return false;
            });

        /// <summary>A type declared in a referenced (not this) assembly that can name NDArray.</summary>
        /// <param name="t">The type.</param>
        /// <returns>True for a NumSharp-aware metadata type.</returns>
        private bool IsNumSharpMetadataType(INamedTypeSymbol t)
        {
            var assembly = t.ContainingAssembly;
            return assembly != null && !SymbolEqualityComparer.Default.Equals(assembly, _sourceAssembly) && IsNumSharpAware(assembly);
        }

        // ------------------------------------------------------------------ the fixed points

        /// <summary>
        ///     The source type names that may store NDArrays: the least fixed point of "a type whose storage
        ///     declarations mention a holding name holds", seeded with the owned vocabulary and the metadata type names,
        ///     computed with a worklist over a reverse index (linear in the number of name mentions).
        /// </summary>
        /// <param name="seeds">The names that hold to begin with.</param>
        /// <param name="declarations">The source type declarations.</param>
        /// <param name="aliases">The using aliases (an alias of a holding type holds).</param>
        /// <returns>The holding names (seeds included).</returns>
        private static HashSet<string> HoldingFixedPoint(HashSet<string> seeds, List<TypeDeclarationFacts> declarations, List<AliasFacts> aliases)
        {
            var holding = new HashSet<string>(seeds, StringComparer.Ordinal);
            var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            void Depend(string mentioned, string dependent)
            {
                if (!dependents.TryGetValue(mentioned, out var list))
                    dependents.Add(mentioned, list = new List<string>());
                list.Add(dependent);
            }

            foreach (var d in declarations)
                foreach (var n in d.StorageNames)
                    Depend(n, d.Name);
            foreach (var a in aliases)
                foreach (var n in a.TargetNames)
                    Depend(n, a.Name);

            var work = new Stack<string>(holding);
            while (work.Count > 0)
            {
                if (!dependents.TryGetValue(work.Pop(), out var list))
                    continue;
                foreach (var dependent in list)
                    if (holding.Add(dependent))
                        work.Push(dependent);
            }

            return holding;
        }

        /// <summary>
        ///     The relevant names N and the relevant source type names: the least fixed point of "a member whose declared
        ///     type mentions a relevant name is relevant (its name joins N), a type whose declaration (base list,
        ///     constraints, primary-constructor parameters, any member signature) mentions a relevant name is relevant
        ///     (its name joins N), an alias whose target mentions one is relevant", by worklist over a reverse index.
        /// </summary>
        /// <param name="seeds">The names that are relevant to begin with.</param>
        /// <param name="declarations">The source type declarations.</param>
        /// <param name="aliases">The using aliases.</param>
        /// <returns>N, and the relevant source type names.</returns>
        private static (HashSet<string> Names, HashSet<string> Types) RelevanceFixedPoint(
            HashSet<string> seeds, List<TypeDeclarationFacts> declarations, List<AliasFacts> aliases)
        {
            var names = new HashSet<string>(seeds, StringComparer.Ordinal);
            var types = new HashSet<string>(StringComparer.Ordinal);

            // mentioned name -> (name that becomes relevant, whether that name is a type)
            var dependents = new Dictionary<string, List<(string Name, bool IsType)>>(StringComparer.Ordinal);
            void Depend(string mentioned, string dependent, bool isType)
            {
                if (!dependents.TryGetValue(mentioned, out var list))
                    dependents.Add(mentioned, list = new List<(string, bool)>());
                list.Add((dependent, isType));
            }

            foreach (var d in declarations)
            {
                foreach (var n in d.DeclarationNames)
                    Depend(n, d.Name, isType: true);
                foreach (var m in d.Members)
                    foreach (var n in m.SignatureNames)
                    {
                        // A member whose signature mentions a relevant name makes its own name relevant, and its type.
                        foreach (var memberName in m.Names)
                            Depend(n, memberName, isType: false);
                        Depend(n, d.Name, isType: true);
                    }
            }

            foreach (var a in aliases)
                foreach (var n in a.TargetNames)
                    Depend(n, a.Name, isType: false);

            var work = new Stack<string>(names);
            while (work.Count > 0)
            {
                if (!dependents.TryGetValue(work.Pop(), out var list))
                    continue;
                foreach (var (dependent, isType) in list)
                {
                    if (isType)
                        types.Add(dependent);
                    if (names.Add(dependent))
                        work.Push(dependent);
                }
            }

            return (names, types);
        }

        // ------------------------------------------------------------------ metadata

        /// <summary>
        ///     Adds the names through which code can reach a NumSharp-aware assembly without having named NumSharp
        ///     already: every type name the compilation can see (a public type, or an internal one when the assembly
        ///     grants this compilation access), nested types included; and the extension methods and statically imported
        ///     members whose type involves a NumSharp-aware type, an extension method only when its receiver does not.
        ///     Those two are callable without naming their declaring type, but one that returns, say, an int
        ///     (<c>int Count(this ReadOnlySpan&lt;Char8&gt;)</c>) cannot introduce an owned value, and one whose receiver
        ///     is already a NumSharp value adds no route the receiver's own name did not; seeding either would make every
        ///     LINQ call of the same name (<c>Count()</c>, <c>Sum()</c>) a candidate.
        /// </summary>
        /// <param name="assembly">A NumSharp-aware referenced assembly.</param>
        /// <param name="seeds">Receives the names.</param>
        /// <param name="staticImports">The last identifiers of the compilation's <c>using static</c> targets.</param>
        private void AddMetadataNames(IAssemblySymbol assembly, HashSet<string> seeds, HashSet<string> staticImports)
        {
            bool seesInternals = assembly.GivesAccessTo(_sourceAssembly);
            var stack = new Stack<INamespaceOrTypeSymbol>();
            stack.Push(assembly.GlobalNamespace);
            while (stack.Count > 0)
            {
                var current = stack.Pop();
                if (current is INamespaceSymbol n)
                {
                    foreach (var child in n.GetNamespaceMembers())
                        stack.Push(child);
                    foreach (var type in n.GetTypeMembers())
                        stack.Push(type);
                    continue;
                }

                var t = (INamedTypeSymbol)current;
                if (!IsVisible(t, seesInternals))
                    continue; // code that cannot name the type cannot reach it, nor anything nested in it, by name
                seeds.Add(t.Name);
                foreach (var nested in t.GetTypeMembers())
                    stack.Push(nested);

                bool imported = staticImports.Contains(t.Name);
                if (!imported && !t.MightContainExtensionMethods)
                    continue;
                foreach (var member in t.GetMembers())
                {
                    if (!member.IsStatic || !IsVisible(member, seesInternals))
                        continue;
                    bool extension = member is IMethodSymbol m && m.IsExtensionMethod;
                    if (!imported && !extension)
                        continue;
                    if (!InvolvesNumSharp(TypeOf(member)))
                        continue; // returns nothing of NumSharp: cannot introduce an owned value
                    // A receiver of a NumSharp type is a value the body already reached through a relevant name, so
                    // the call adds no route of its own; NumSharp's own extension methods on NDArray (Sum, Count, ...)
                    // would otherwise make every LINQ call of the same name a candidate.
                    if (extension && InvolvesNumSharp(((IMethodSymbol)member).Parameters[0].Type))
                        continue;
                    seeds.Add(member.Name);
                }
            }
        }

        /// <summary>Whether a metadata symbol can be named by this compilation: public (or protected, reachable from a derived type), or internal with access granted.</summary>
        /// <param name="symbol">A type or member.</param>
        /// <param name="seesInternals">Whether the declaring assembly grants this compilation internal access.</param>
        /// <returns>True when the compilation can name it.</returns>
        private static bool IsVisible(ISymbol symbol, bool seesInternals)
        {
            switch (symbol.DeclaredAccessibility)
            {
                case Accessibility.Public:
                case Accessibility.Protected:
                case Accessibility.ProtectedOrInternal:
                    return true;
                case Accessibility.Internal:
                case Accessibility.ProtectedAndInternal:
                    return seesInternals;
                default:
                    return false;
            }
        }

        /// <summary>The type a member gives its caller: a method's return type, a property's, field's or event's type.</summary>
        /// <param name="member">The member.</param>
        /// <returns>The type, or null for a member kind that has none.</returns>
        private static ITypeSymbol TypeOf(ISymbol member)
        {
            switch (member)
            {
                case IMethodSymbol m:
                    return m.ReturnType;
                case IPropertySymbol p:
                    return p.Type;
                case IFieldSymbol f:
                    return f.Type;
                case IEventSymbol e:
                    return e.Type;
                default:
                    return null;
            }
        }

        /// <summary>
        ///     Whether a type involves a type of a NumSharp-aware assembly - itself, an element type, a type argument or a
        ///     tuple element, recursively. A type parameter does not count: its instantiation comes from a type argument
        ///     or an inferred argument, both of which a body names.
        /// </summary>
        /// <param name="type">The type; null answers false.</param>
        /// <returns>True when a value of it can be (or carry) something of NumSharp.</returns>
        private bool InvolvesNumSharp(ITypeSymbol type)
        {
            for (int guard = 0; guard < 32 && type != null; guard++)
            {
                switch (type)
                {
                    case IArrayTypeSymbol arr:
                        type = arr.ElementType;
                        continue;
                    case IPointerTypeSymbol ptr:
                        type = ptr.PointedAtType;
                        continue;
                    case INamedTypeSymbol named:
                        if (IsNumSharpAware(named.ContainingAssembly))
                            return true;
                        foreach (var arg in named.TypeArguments)
                            if (InvolvesNumSharp(arg))
                                return true;
                        return false;
                    default:
                        return false;
                }
            }

            return false;
        }

        // ------------------------------------------------------------------ syntax

        /// <summary>One source type declaration, as the fixed points read it.</summary>
        private sealed class TypeDeclarationFacts
        {
            /// <summary>The type's simple name (a partial type contributes one entry per declaration; all share the name).</summary>
            public string Name;

            /// <summary>Names in its storage declarations (field and property types, record / primary-constructor parameters, base list, constraints).</summary>
            public readonly HashSet<string> StorageNames = new HashSet<string>(StringComparer.Ordinal);

            /// <summary>Names whose relevance makes the whole type relevant (base list, constraints, primary-constructor parameters).</summary>
            public readonly HashSet<string> DeclarationNames = new HashSet<string>(StringComparer.Ordinal);

            /// <summary>The members, with the names their signatures mention.</summary>
            public readonly List<MemberFacts> Members = new List<MemberFacts>();
        }

        /// <summary>One member declaration: the names a body uses to reach it, and the names its signature mentions.</summary>
        private sealed class MemberFacts
        {
            /// <summary>The names a body uses to reach the member (a field declaration declares several).</summary>
            public readonly List<string> Names = new List<string>();

            /// <summary>The names in its declared type, return type, or (for an operator, a conversion, a delegate) its parameters.</summary>
            public readonly HashSet<string> SignatureNames = new HashSet<string>(StringComparer.Ordinal);
        }

        /// <summary>One <c>using X = target;</c> alias.</summary>
        private sealed class AliasFacts
        {
            /// <summary>The alias name.</summary>
            public string Name;

            /// <summary>The names the target mentions.</summary>
            public readonly HashSet<string> TargetNames = new HashSet<string>(StringComparer.Ordinal);
        }

        /// <summary>
        ///     Adds the aliases of the scope attributes to the attribute-name sets: <c>using S = NumSharp.NDScopedAttribute;</c>
        ///     makes <c>[S]</c> read as <c>[NDScoped]</c>.
        /// </summary>
        /// <param name="root">A syntax tree's root.</param>
        /// <param name="scopeNames">The scope-family attribute names; receives the aliases of them.</param>
        /// <param name="exitNames">The [NDScopedExit] names; receives the aliases of it.</param>
        private static void CollectAttributeAliases(SyntaxNode root, HashSet<string> scopeNames, HashSet<string> exitNames)
        {
            foreach (var u in UsingDirectives(root))
            {
                if (u.Alias == null || u.Name == null)
                    continue;
                var last = LastIdentifier(u.Name);
                if (last == null)
                    continue;
                if (ScopeAttributeNames.Contains(last))
                    scopeNames.Add(u.Alias.Name.Identifier.ValueText);
                else if (ExitAttributeNames.Contains(last))
                    exitNames.Add(u.Alias.Name.Identifier.ValueText);
            }
        }

        /// <summary>Every using directive of a tree: the compilation unit's and every namespace's (block-scoped or file-scoped).</summary>
        /// <param name="root">A syntax tree's root.</param>
        /// <returns>The directives.</returns>
        private static IEnumerable<UsingDirectiveSyntax> UsingDirectives(SyntaxNode root)
        {
            if (root is CompilationUnitSyntax cu)
                foreach (var u in cu.Usings)
                    yield return u;
            foreach (var node in root.DescendantNodes(n => n is CompilationUnitSyntax || n is BaseNamespaceDeclarationSyntax))
                if (node is BaseNamespaceDeclarationSyntax ns)
                    foreach (var u in ns.Usings)
                        yield return u;
        }

        /// <summary>
        ///     Reads one tree's declarations: every type declaration (nested ones included, never inside a body), every
        ///     delegate declaration, every alias and <c>using static</c> target, and the scoped inheritable declarations.
        /// </summary>
        /// <param name="root">A syntax tree's root.</param>
        /// <param name="declarations">Receives the type declarations.</param>
        /// <param name="aliases">Receives the aliases.</param>
        /// <param name="staticImports">Receives the last identifier of every <c>using static</c> target.</param>
        /// <param name="scopeNames">The scope-family attribute names (aliases included).</param>
        /// <param name="exitNames">The [NDScopedExit] attribute names (aliases included).</param>
        private void CollectDeclarations(SyntaxNode root, List<TypeDeclarationFacts> declarations, List<AliasFacts> aliases,
            HashSet<string> staticImports, HashSet<string> scopeNames, HashSet<string> exitNames)
        {
            foreach (var u in UsingDirectives(root))
            {
                if (u.Name == null)
                    continue;
                if (u.Alias != null)
                {
                    var alias = new AliasFacts { Name = u.Alias.Name.Identifier.ValueText };
                    AddNames(u.Name, alias.TargetNames);
                    aliases.Add(alias);
                }
                else if (!u.StaticKeyword.IsKind(SyntaxKind.None))
                {
                    var last = LastIdentifier(u.Name);
                    if (last != null)
                        staticImports.Add(last);
                }
            }

            // Declarations live at namespace level or nested in type declarations; nothing below a member body is a
            // declaration the fixed points need (local functions belong to their body's own tokens).
            foreach (var node in root.DescendantNodes(n => n is CompilationUnitSyntax || n is BaseNamespaceDeclarationSyntax))
            {
                switch (node)
                {
                    case TypeDeclarationSyntax type:
                        CollectType(type, inheritedConstraintNames: null, declarations, scopeNames, exitNames);
                        break;
                    case DelegateDeclarationSyntax del:
                        declarations.Add(DelegateFacts(del));
                        break;
                }
            }
        }

        /// <summary>
        ///     Reads one type declaration and, recursively, its nested type declarations. A nested type also carries its
        ///     enclosing types' constraint names: a member typed by an outer type parameter reaches what that parameter's
        ///     constraints name.
        /// </summary>
        /// <param name="type">The type declaration.</param>
        /// <param name="inheritedConstraintNames">The enclosing declarations' constraint names, or null at top level.</param>
        /// <param name="declarations">Receives the facts.</param>
        /// <param name="scopeNames">The scope-family attribute names.</param>
        /// <param name="exitNames">The [NDScopedExit] attribute names.</param>
        private void CollectType(TypeDeclarationSyntax type, HashSet<string> inheritedConstraintNames, List<TypeDeclarationFacts> declarations,
            HashSet<string> scopeNames, HashSet<string> exitNames)
        {
            var facts = new TypeDeclarationFacts { Name = type.Identifier.ValueText };
            bool isInterface = type is InterfaceDeclarationSyntax;

            var constraintNames = new HashSet<string>(StringComparer.Ordinal);
            if (inheritedConstraintNames != null)
                constraintNames.UnionWith(inheritedConstraintNames);
            foreach (var clause in type.ConstraintClauses)
                AddNames(clause, constraintNames);

            facts.DeclarationNames.UnionWith(constraintNames);
            facts.StorageNames.UnionWith(constraintNames);
            if (type.BaseList != null)
            {
                AddNames(type.BaseList, facts.DeclarationNames);
                AddNames(type.BaseList, facts.StorageNames);
            }

            // Record positional parameters are auto-properties; a class or struct primary constructor's parameters are
            // in scope in every member body (and captured into compiler-generated storage).
            if (type.ParameterList != null)
            {
                AddNames(type.ParameterList, facts.DeclarationNames);
                AddNames(type.ParameterList, facts.StorageNames);
            }

            foreach (var member in type.Members)
            {
                switch (member)
                {
                    case TypeDeclarationSyntax nested:
                        CollectType(nested, constraintNames, declarations, scopeNames, exitNames);
                        break;
                    case DelegateDeclarationSyntax del:
                        declarations.Add(DelegateFacts(del));
                        break;
                    case FieldDeclarationSyntax field:
                    {
                        var m = new MemberFacts();
                        AddNames(field.Declaration.Type, m.SignatureNames);
                        AddNames(field.Declaration.Type, facts.StorageNames);
                        foreach (var v in field.Declaration.Variables)
                            m.Names.Add(v.Identifier.ValueText);
                        facts.Members.Add(m);
                        break;
                    }
                    case EventFieldDeclarationSyntax ev:
                    {
                        var m = new MemberFacts();
                        AddNames(ev.Declaration.Type, m.SignatureNames);
                        foreach (var v in ev.Declaration.Variables)
                            m.Names.Add(v.Identifier.ValueText);
                        facts.Members.Add(m);
                        break;
                    }
                    case PropertyDeclarationSyntax prop:
                    {
                        var m = new MemberFacts();
                        AddNames(prop.Type, m.SignatureNames);
                        AddNames(prop.Type, facts.StorageNames);
                        m.Names.Add(prop.Identifier.ValueText);
                        facts.Members.Add(m);
                        if (IsInheritable(prop, isInterface) && HasAttribute(prop.AttributeLists, scopeNames))
                        {
                            _scopedDeclarationNames.Add(prop.Identifier.ValueText);
                            _scopedDeclarationNames.Add("get_" + prop.Identifier.ValueText);
                        }

                        break;
                    }
                    case IndexerDeclarationSyntax indexer:
                    {
                        var m = new MemberFacts();
                        AddNames(indexer.Type, m.SignatureNames);
                        facts.Members.Add(m); // reachable only through the type (this[...], x[...]): it makes the type relevant
                        if (IsInheritable(indexer, isInterface) && HasAttribute(indexer.AttributeLists, scopeNames))
                            _scopedDeclarationNames.Add("get_Item");
                        break;
                    }
                    case EventDeclarationSyntax evDecl:
                    {
                        var m = new MemberFacts();
                        AddNames(evDecl.Type, m.SignatureNames);
                        m.Names.Add(evDecl.Identifier.ValueText);
                        facts.Members.Add(m);
                        break;
                    }
                    case MethodDeclarationSyntax method:
                        CollectMethod(method, facts, isInterface, scopeNames, exitNames);
                        break;
                    case OperatorDeclarationSyntax op:
                    {
                        // Invoked by an operator token, never by name: relevant through the type only.
                        var m = new MemberFacts();
                        AddNames(op.ReturnType, m.SignatureNames);
                        AddNames(op.ParameterList, m.SignatureNames);
                        facts.Members.Add(m);
                        break;
                    }
                    case ConversionOperatorDeclarationSyntax conv:
                    {
                        var m = new MemberFacts();
                        AddNames(conv.Type, m.SignatureNames);
                        AddNames(conv.ParameterList, m.SignatureNames);
                        facts.Members.Add(m);
                        break;
                    }
                    case BaseFieldDeclarationSyntax _:
                    case BaseMethodDeclarationSyntax _:
                    case BasePropertyDeclarationSyntax _:
                    case EnumDeclarationSyntax _:
                        break; // the kinds above read everything of these that can name an owned type
                    default:
                        // A member kind this Roslyn API does not model - a C# 14 `extension(...) { ... }` block parses as one
                        // on a newer compiler. Its methods and properties are still members a body can call (extension
                        // members on a receiver that names nothing of NumSharp), so they are indexed like the type's own.
                        foreach (var inner in member.DescendantNodes(n => !(n is BlockSyntax) && !(n is ArrowExpressionClauseSyntax)))
                        {
                            if (inner is MethodDeclarationSyntax innerMethod)
                                CollectMethod(innerMethod, facts, isInterface, scopeNames, exitNames);
                            else if (inner is PropertyDeclarationSyntax innerProp)
                            {
                                var m = new MemberFacts();
                                AddNames(innerProp.Type, m.SignatureNames);
                                m.Names.Add(innerProp.Identifier.ValueText);
                                facts.Members.Add(m);
                            }
                        }

                        break;
                }
            }

            declarations.Add(facts);
        }

        /// <summary>
        ///     Reads one method declaration: its name, the names its return type and its own constraint clauses mention
        ///     (<c>T M&lt;T&gt;() where T : NDArray</c> returns an owned value without naming one in its return type),
        ///     and whether it is a scoped inheritable declaration.
        /// </summary>
        /// <param name="method">The method declaration.</param>
        /// <param name="facts">The declaring type's facts; receives the member.</param>
        /// <param name="isInterface">Whether the declaring type is an interface.</param>
        /// <param name="scopeNames">The scope-family attribute names.</param>
        /// <param name="exitNames">The [NDScopedExit] attribute names.</param>
        private void CollectMethod(MethodDeclarationSyntax method, TypeDeclarationFacts facts, bool isInterface,
            HashSet<string> scopeNames, HashSet<string> exitNames)
        {
            var m = new MemberFacts();
            AddNames(method.ReturnType, m.SignatureNames);
            foreach (var clause in method.ConstraintClauses)
                AddNames(clause, m.SignatureNames);
            m.Names.Add(method.Identifier.ValueText);
            facts.Members.Add(m);
            if (IsInheritable(method, isInterface) &&
                (HasAttribute(method.AttributeLists, scopeNames) || HasParameterAttribute(method.ParameterList, exitNames)))
                _scopedDeclarationNames.Add(method.Identifier.ValueText);
        }

        /// <summary>A delegate declaration as a "type" whose relevance comes from its return and parameter types (invoking a variable of it can return an owned value).</summary>
        /// <param name="del">The declaration.</param>
        /// <returns>Its facts.</returns>
        private static TypeDeclarationFacts DelegateFacts(DelegateDeclarationSyntax del)
        {
            var facts = new TypeDeclarationFacts { Name = del.Identifier.ValueText };
            AddNames(del.ReturnType, facts.DeclarationNames);
            AddNames(del.ParameterList, facts.DeclarationNames);
            return facts;
        }

        /// <summary>
        ///     Whether a member can be the declaration an override or implementation inherits from: an interface member,
        ///     or one marked virtual, abstract or override (an override is itself the declaration of its own overrides).
        /// </summary>
        /// <param name="member">The member declaration.</param>
        /// <param name="inInterface">Whether it is declared in an interface.</param>
        /// <returns>True for an inheritable declaration.</returns>
        private static bool IsInheritable(MemberDeclarationSyntax member, bool inInterface)
        {
            if (inInterface)
                return true;
            foreach (var modifier in member.Modifiers)
                if (modifier.IsKind(SyntaxKind.VirtualKeyword) || modifier.IsKind(SyntaxKind.AbstractKeyword) || modifier.IsKind(SyntaxKind.OverrideKeyword))
                    return true;
            return false;
        }

        /// <summary>Whether an attribute of <paramref name="lists"/> is named (by its last identifier) one of <paramref name="names"/>.</summary>
        /// <param name="lists">The attribute lists.</param>
        /// <param name="names">The attribute names (suffix forms and aliases included).</param>
        /// <returns>True when one matches.</returns>
        private static bool HasAttribute(SyntaxList<AttributeListSyntax> lists, HashSet<string> names)
        {
            foreach (var list in lists)
                foreach (var attribute in list.Attributes)
                {
                    var last = LastIdentifier(attribute.Name);
                    if (last != null && names.Contains(last))
                        return true;
                }
            return false;
        }

        /// <summary>Whether a parameter of <paramref name="parameters"/> carries an attribute named one of <paramref name="names"/>.</summary>
        /// <param name="parameters">The parameter list.</param>
        /// <param name="names">The attribute names.</param>
        /// <returns>True when one matches.</returns>
        private static bool HasParameterAttribute(ParameterListSyntax parameters, HashSet<string> names)
        {
            if (parameters == null)
                return false;
            foreach (var p in parameters.Parameters)
                if (HasAttribute(p.AttributeLists, names))
                    return true;
            return false;
        }

        /// <summary>Adds every identifier a node mentions (simple and generic names, the parts of qualified names) to <paramref name="names"/>.</summary>
        /// <param name="node">A type, base list, constraint clause or parameter list.</param>
        /// <param name="names">Receives the identifiers.</param>
        private static void AddNames(SyntaxNode node, HashSet<string> names)
        {
            if (node == null)
                return;
            foreach (var token in node.DescendantTokens())
                if (token.IsKind(SyntaxKind.IdentifierToken))
                    names.Add(token.ValueText);
        }

        /// <summary>The last identifier of a name (<c>NumSharp.NDScoped</c> → <c>NDScoped</c>, <c>global::X</c> → <c>X</c>).</summary>
        /// <param name="name">The name.</param>
        /// <returns>The identifier, or null for a name without one.</returns>
        private static string LastIdentifier(NameSyntax name)
        {
            switch (name)
            {
                case IdentifierNameSyntax id:
                    return id.Identifier.ValueText;
                case GenericNameSyntax g:
                    return g.Identifier.ValueText;
                case QualifiedNameSyntax q:
                    return LastIdentifier(q.Right);
                case AliasQualifiedNameSyntax a:
                    return LastIdentifier(a.Name);
                default:
                    return null;
            }
        }

        /// <summary>
        ///     The member declaration a method's declaring syntax belongs to: the method, constructor, operator or local
        ///     function itself; for an accessor, the whole property, indexer or event declaration (its type token is the
        ///     accessor's signature); for top-level statements, the compilation unit.
        /// </summary>
        /// <param name="node">A declaring syntax of the method.</param>
        /// <returns>The node whose tokens are scanned.</returns>
        private static SyntaxNode MemberDeclarationOf(SyntaxNode node)
        {
            for (var n = node; n != null; n = n.Parent)
            {
                switch (n)
                {
                    case BasePropertyDeclarationSyntax _:
                    case BaseMethodDeclarationSyntax _:
                    case LocalFunctionStatementSyntax _:
                    case BaseTypeDeclarationSyntax _:
                    case CompilationUnitSyntax _:
                        return n;
                }
            }

            return node;
        }
    }
}
