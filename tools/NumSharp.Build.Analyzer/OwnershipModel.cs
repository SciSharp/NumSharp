using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     The compilation-wide <b>ownership model</b> shared by the leak analyzer (NDW012) and the
    ///     holder analyzer (NDW016/NDW017): which TYPES hold NDArrays, which of those are disposable
    ///     (and therefore the "contagious" NDArray-owning values a method or a containing type must
    ///     reclaim), and which MEMBERS of a type are its holders. Resolved once per compilation and
    ///     memoized; safe under the analyzers' concurrent execution.
    ///     <para>
    ///     <b>"Holds NDArrays"</b> is structural: a type holds NDArrays when a value of it transitively
    ///     contains one — an <c>NDArray</c> (or subclass), an array of them (any rank), a tuple with an
    ///     NDArray component, an <see cref="KnownTypes.INDArrayCarrier"/> struct, a generic type
    ///     instantiated over a holding type (<c>List&lt;NDArray&gt;</c>, <c>Dictionary&lt;K, NDArray&gt;</c>,
    ///     <c>Task&lt;NDArray&gt;</c>, <c>Lazy&lt;NDArray&gt;</c>, a consumer's <c>Box&lt;NDArray&gt;</c>), a
    ///     type parameter constrained to one, or a declared class/struct whose own storing members (or
    ///     base type) hold one. Delegates, pointers, weak references, comparers/observers and every type
    ///     from an assembly that cannot even name NDArray answer false; a type or member marked
    ///     <c>[NDBorrowed]</c> answers false by the author's assertion.
    ///     </para>
    /// </summary>
    internal sealed class OwnershipModel
    {
        private readonly KnownTypes _k;
        private readonly IAssemblySymbol _sourceAssembly;
        private readonly ConcurrentDictionary<ITypeSymbol, bool> _holds =
            new ConcurrentDictionary<ITypeSymbol, bool>(SymbolEqualityComparer.Default);
        private readonly ConcurrentDictionary<IAssemblySymbol, bool> _canNameNDArray =
            new ConcurrentDictionary<IAssemblySymbol, bool>(SymbolEqualityComparer.Default);

        /// <summary>
        ///     Generic families whose type argument may name NDArray without the value HOLDING one — a
        ///     comparer compares arrays it is handed, an observer/progress receives them, a weak
        ///     reference deliberately does not keep one alive. Keyed by namespace + metadata name.
        /// </summary>
        private static readonly ImmutableHashSet<string> NonHoldingGenericFamilies = ImmutableHashSet.Create(
            "System.Collections.Generic.IComparer`1",
            "System.Collections.Generic.IEqualityComparer`1",
            "System.Collections.Generic.Comparer`1",
            "System.Collections.Generic.EqualityComparer`1",
            "System.IComparable`1",
            "System.IEquatable`1",
            "System.IObservable`1",
            "System.IObserver`1",
            "System.IProgress`1",
            "System.Progress`1",
            "System.WeakReference`1");

        public OwnershipModel(KnownTypes k, Compilation compilation)
        {
            _k = k;
            _sourceAssembly = compilation.Assembly;
        }

        public KnownTypes Known => _k;

        /// <summary>A storing member of a type — the field or (auto-)property symbol to report, and the type it stores.</summary>
        public readonly struct HolderMember
        {
            public readonly ISymbol Symbol;
            public readonly ITypeSymbol Type;

            public HolderMember(ISymbol symbol, ITypeSymbol type)
            {
                Symbol = symbol;
                Type = type;
            }

            public string Name => Symbol.Name;
            public string Kind => Symbol is IPropertySymbol ? "property" : "field";
        }

        // ------------------------------------------------------------------ members

        /// <summary>
        ///     The INSTANCE members of <paramref name="type"/> (declared on it, not inherited) that store
        ///     NDArrays and are not <c>[NDBorrowed]</c>. For a source type these are its explicit instance
        ///     fields and its auto-properties (reported as the property; a computed property is a view,
        ///     not storage, and is skipped). For a metadata type the storage is invisible, so every visible
        ///     instance field, property and indexer counts — metadata types are never REPORTED, only
        ///     consulted for whether a value of the type holds arrays.
        /// </summary>
        public List<HolderMember> HolderMembersOf(INamedTypeSymbol type)
        {
            bool cut = false;
            return HolderMembersOf(type, new HashSet<ITypeSymbol>(SymbolEqualityComparer.Default) { type }, ref cut);
        }

        private List<HolderMember> HolderMembersOf(INamedTypeSymbol type, HashSet<ITypeSymbol> visiting, ref bool cut)
        {
            var list = new List<HolderMember>();
            bool inSource = IsInSource(type);
            foreach (var m in type.GetMembers())
            {
                if (m.IsStatic)
                    continue;

                ISymbol reported;
                ITypeSymbol memberType;
                switch (m)
                {
                    case IFieldSymbol f:
                        if (f.IsConst)
                            continue;
                        if (f.AssociatedSymbol is IPropertySymbol autoProp)
                        {
                            // The backing field of an auto-property (a positional record property too):
                            // report the property — that is the name the author wrote.
                            reported = autoProp;
                            memberType = autoProp.Type;
                        }
                        else if (f.AssociatedSymbol != null || f.IsImplicitlyDeclared)
                        {
                            continue; // an event's backing field / other compiler-synthesized state
                        }
                        else
                        {
                            reported = f;
                            memberType = f.Type;
                        }
                        break;

                    case IPropertySymbol p when !inSource:
                        // Metadata: storage is invisible, so a property (indexers included) is the only
                        // evidence a value of this type carries arrays.
                        reported = p;
                        memberType = p.Type;
                        break;

                    default:
                        continue;
                }

                if (IsBorrowed(reported))
                    continue;
                if (HoldsCore(memberType, visiting, ref cut))
                    list.Add(new HolderMember(reported, memberType));
            }
            return list;
        }

        // ------------------------------------------------------------------ type predicates

        /// <summary>Does a value of this type (transitively) hold an NDArray somebody must dispose?</summary>
        public bool HoldsNDArrays(ITypeSymbol t)
        {
            bool cut = false;
            return HoldsCore(t, null, ref cut);
        }

        private bool HoldsCore(ITypeSymbol t, HashSet<ITypeSymbol> visiting, ref bool cut)
        {
            if (t == null)
                return false;
            if (_holds.TryGetValue(t, out var memo))
                return memo;

            bool root = visiting == null;
            if (root)
                visiting = new HashSet<ITypeSymbol>(SymbolEqualityComparer.Default);
            if (!visiting.Add(t))
            {
                // A cycle (a type reachable from itself): this occurrence contributes nothing; the
                // type's answer is decided by its other members. The enclosing answers are then
                // inexact and must not be memoized — only the ROOT answer is exact (a cycle back to
                // the root cannot change what the root itself holds).
                cut = true;
                return false;
            }

            bool localCut = false;
            bool result = Compute(t, visiting, ref localCut);
            visiting.Remove(t);
            if (root || !localCut)
                _holds[t] = result;
            if (localCut)
                cut = true;
            return result;
        }

        private bool Compute(ITypeSymbol t, HashSet<ITypeSymbol> visiting, ref bool cut)
        {
            switch (t)
            {
                case IArrayTypeSymbol arr:
                    return HoldsCore(arr.ElementType, visiting, ref cut);
                case IPointerTypeSymbol _:
                case IDynamicTypeSymbol _:
                    return false;
                case ITypeParameterSymbol tp:
                    foreach (var c in tp.ConstraintTypes)
                        if (HoldsCore(c, visiting, ref cut))
                            return true;
                    return false;
            }

            if (TypeHelpers.IsNDArrayLike(t, _k))
                return true;
            if (!(t is INamedTypeSymbol named))
                return false;

            switch (named.TypeKind)
            {
                case TypeKind.Delegate:
                case TypeKind.Enum:
                case TypeKind.Error:
                    return false;
            }

            if (IsBorrowed(named))
                return false;
            if (named.IsTupleType)
            {
                foreach (var e in named.TupleElements)
                    if (HoldsCore(e.Type, visiting, ref cut))
                        return true;
                return false;
            }
            if (IsCarrier(named))
                return true;
            if (named.IsGenericType && NonHoldingGenericFamilies.Contains(FamilyKey(named)))
                return false;

            // A generic instantiation over a holding type is a container of it — List<NDArray>,
            // Task<NDArray>, Lazy<NDArray>, KeyValuePair<K, NDArray>, a consumer's Box<NDArray> —
            // and so is a type NESTED in one (Dictionary<K, NDArray>.ValueCollection, List<NDArray>.Enumerator).
            for (var g = named; g != null; g = g.ContainingType)
                if (g.IsGenericType)
                    foreach (var arg in g.TypeArguments)
                        if (HoldsCore(arg, visiting, ref cut))
                            return true;

            // A declared class/struct: its own storing members, then its base. Only an assembly that
            // can name NDArray (NumSharp, this compilation, or one referencing NumSharp) can hold one;
            // a BCL or unrelated type is answered without a member scan.
            if (named.TypeKind != TypeKind.Class && named.TypeKind != TypeKind.Struct && named.TypeKind != TypeKind.Interface)
                return false;
            if (!IsInSource(named) && !CanNameNDArray(named.ContainingAssembly))
                return false;

            if (HolderMembersOf(named, visiting, ref cut).Count > 0)
                return true;
            return named.BaseType != null && HoldsCore(named.BaseType, visiting, ref cut);
        }

        /// <summary>
        ///     Implements <c>IDisposable</c> or <c>IAsyncDisposable</c>, or is a <c>ref struct</c> with the
        ///     <c>public void Dispose()</c> pattern <c>using</c> accepts.
        /// </summary>
        public bool IsDisposable(ITypeSymbol t)
        {
            if (t == null)
                return false;
            if (IsDisposableInterface(t))
                return true;
            foreach (var i in t.AllInterfaces)
                if (IsDisposableInterface(i))
                    return true;
            return t.IsRefLikeType && HasPatternDispose(t);
        }

        /// <summary>The disposable contract a type carries, for messages: "IDisposable", "IAsyncDisposable", both, or the ref-struct pattern.</summary>
        public string DisposableContractOf(ITypeSymbol t)
        {
            bool sync = false, async = false;
            foreach (var i in t.AllInterfaces)
            {
                if (Eq(i, _k.IDisposable)) sync = true;
                else if (Eq(i, _k.IAsyncDisposable)) async = true;
            }
            if (sync && async) return "IDisposable and IAsyncDisposable";
            if (sync) return "IDisposable";
            if (async) return "IAsyncDisposable";
            return "a Dispose() pattern";
        }

        /// <summary>
        ///     The CONTAGIOUS set: a disposable class / struct / ref struct that stores NDArrays (and is
        ///     not <c>[NDBorrowed]</c>). An instance of it is an NDArray-owning value — dropping one
        ///     strands the arrays it holds exactly as dropping a bare NDArray does — so the leak analyzer
        ///     tracks it like an NDArray, and a member of this type is a holder its containing type must
        ///     dispose. NDArray itself, tuples and carrier structs are the base vocabulary, not this set.
        /// </summary>
        public bool IsNDOwningDisposable(ITypeSymbol t)
        {
            if (!(t is INamedTypeSymbol named))
                return false;
            if (named.TypeKind != TypeKind.Class && named.TypeKind != TypeKind.Struct)
                return false;
            if (named.IsTupleType || TypeHelpers.IsNDArrayLike(named, _k) || IsCarrier(named) || IsBorrowed(named))
                return false;
            return IsDisposable(named) && HoldsNDArrays(named);
        }

        /// <summary>The <c>INDArrayCarrier</c> interface itself, or any type implementing it — the scope-yieldable result vocabulary, transient by contract.</summary>
        public bool IsCarrier(ITypeSymbol t)
        {
            if (t == null || _k.INDArrayCarrier == null)
                return false;
            if (Eq(t, _k.INDArrayCarrier))
                return true;
            foreach (var i in t.AllInterfaces)
                if (Eq(i, _k.INDArrayCarrier))
                    return true;
            return false;
        }

        /// <summary><c>[NDBorrowed]</c> on a member or (via its original definition) a type.</summary>
        public bool IsBorrowed(ISymbol s)
        {
            if (s == null || _k.BorrowedAttr == null)
                return false;
            var target = s is INamedTypeSymbol n ? n.OriginalDefinition : s;
            foreach (var a in target.GetAttributes())
                if (Eq(a.AttributeClass, _k.BorrowedAttr))
                    return true;
            return false;
        }

        /// <summary>A declared type — one with a source declaration in this compilation (a constructed generic answers for its definition).</summary>
        public static bool IsInSource(ISymbol s)
        {
            if (s is INamedTypeSymbol n)
                s = n.OriginalDefinition;
            return s != null && s.DeclaringSyntaxReferences.Length > 0;
        }

        // ------------------------------------------------------------------ helpers

        private bool IsDisposableInterface(ITypeSymbol t)
            => Eq(t, _k.IDisposable) || Eq(t, _k.IAsyncDisposable);

        private static bool HasPatternDispose(ITypeSymbol t)
        {
            foreach (var m in t.GetMembers("Dispose"))
                if (m is IMethodSymbol method && !method.IsStatic && method.Parameters.Length == 0
                    && method.DeclaredAccessibility == Accessibility.Public)
                    return true;
            return false;
        }

        private bool CanNameNDArray(IAssemblySymbol asm)
        {
            if (asm == null)
                return false;
            if (_canNameNDArray.TryGetValue(asm, out var memo))
                return memo;
            bool result = Eq(asm, _k.NumSharpAssembly) || Eq(asm, _sourceAssembly);
            if (!result)
                foreach (var module in asm.Modules)
                {
                    foreach (var r in module.ReferencedAssemblySymbols)
                        if (Eq(r, _k.NumSharpAssembly))
                        {
                            result = true;
                            break;
                        }
                    if (result)
                        break;
                }
            _canNameNDArray[asm] = result;
            return result;
        }

        private static string FamilyKey(INamedTypeSymbol named)
        {
            var def = named.OriginalDefinition;
            var ns = def.ContainingNamespace;
            string prefix = ns == null || ns.IsGlobalNamespace ? "" : ns.ToDisplayString() + ".";
            return prefix + def.MetadataName;
        }

        private static bool Eq(ISymbol a, ISymbol b)
            => b != null && SymbolEqualityComparer.Default.Equals(a, b);
    }
}
