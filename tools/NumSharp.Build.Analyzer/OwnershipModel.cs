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
            // A source class/struct that does not hold NDArrays as a whole has no holding member: each of
            // its storing members feeds the type's own answer. That answer is memoized, so a type unrelated
            // to NumSharp costs one lookup here instead of a walk of its member graph.
            if (IsInSource(type) && (type.TypeKind == TypeKind.Class || type.TypeKind == TypeKind.Struct)
                && !HoldsNDArrays(type))
                return new List<HolderMember>();

            // The type itself is on the walk's stack: a member that reaches arrays only by coming back to
            // this type adds nothing. Answers of the cycles through it stay pending and are not memoized.
            var walk = new Walk();
            int low = walk.Enter(type);
            return HolderMembersOf(type, walk, ref low);
        }

        private List<HolderMember> HolderMembersOf(INamedTypeSymbol type, Walk walk, ref int low)
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
                if (HoldsCore(memberType, walk, ref low))
                    list.Add(new HolderMember(reported, memberType));
            }
            return list;
        }

        // ------------------------------------------------------------------ type predicates

        /// <summary>Does a value of this type (transitively) hold an NDArray somebody must dispose?</summary>
        public bool HoldsNDArrays(ITypeSymbol t)
        {
            int low = int.MaxValue;
            return HoldsCore(t, null, ref low);
        }

        private bool HoldsCore(ITypeSymbol t, Walk walk, ref int low)
        {
            if (t == null)
                return false;
            if (_holds.TryGetValue(t, out var memo))
                return memo;

            if (walk == null)
                walk = new Walk();
            if (walk.IndexOf(t, out int seen))
            {
                // A cycle (a type reachable from itself): this occurrence contributes nothing; the type's
                // answer is decided by its other members. Every type visited in this walk and not yet
                // memoized is still on the stack, waiting for the cycle it belongs to.
                if (seen < low)
                    low = seen;
                return false;
            }

            // Tarjan's strongly connected components over the "stores a value of" graph, so each type is
            // walked once. "Holds" is the least fixed point of an OR over that graph: a true answer is
            // exact as soon as it is found, and the types of one cycle (one component) share one answer,
            // the answer of the type the walk entered the component through. Memoizing only answers that
            // involved no cycle at all made every type re-walk every path through a model whose types
            // reference each other - exponential in the number of types.
            int index = walk.Enter(t);
            int ownLow = index;
            bool result = Compute(t, walk, ref ownLow);
            if (result)
                _holds[t] = true;
            if (ownLow == index)
            {
                // t is the entry of its component: everything above it on the stack belongs to the
                // component and reaches t, so it holds exactly when t does.
                foreach (var member in walk.ExitComponent(t))
                    _holds[member] = result || _holds.TryGetValue(member, out var held) && held;
            }
            if (ownLow < low)
                low = ownLow;
            return result;
        }

        /// <summary>One <see cref="HoldsCore"/> walk: Tarjan's visit order and the stack of types whose component is still open.</summary>
        private sealed class Walk
        {
            private readonly Dictionary<ITypeSymbol, int> _index = new Dictionary<ITypeSymbol, int>(SymbolEqualityComparer.Default);
            private readonly Stack<ITypeSymbol> _stack = new Stack<ITypeSymbol>();

            /// <summary>Visits <paramref name="t"/>: gives it the next index and puts it on the stack.</summary>
            public int Enter(ITypeSymbol t)
            {
                int index = _index.Count;
                _index[t] = index;
                _stack.Push(t);
                return index;
            }

            /// <summary>Whether <paramref name="t"/> was visited in this walk (and so is still on the stack, its answer not memoized).</summary>
            public bool IndexOf(ITypeSymbol t, out int index) => _index.TryGetValue(t, out index);

            /// <summary>Pops the component entered through <paramref name="entry"/>: the entry and every type above it.</summary>
            public List<ITypeSymbol> ExitComponent(ITypeSymbol entry)
            {
                var component = new List<ITypeSymbol>();
                ITypeSymbol t;
                do
                {
                    t = _stack.Pop();
                    component.Add(t);
                } while (!SymbolEqualityComparer.Default.Equals(t, entry));
                return component;
            }
        }

        private bool Compute(ITypeSymbol t, Walk walk, ref int low)
        {
            switch (t)
            {
                case IArrayTypeSymbol arr:
                    return HoldsCore(arr.ElementType, walk, ref low);
                case IPointerTypeSymbol _:
                case IDynamicTypeSymbol _:
                    return false;
                case ITypeParameterSymbol tp:
                    foreach (var c in tp.ConstraintTypes)
                        if (HoldsCore(c, walk, ref low))
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
                    if (HoldsCore(e.Type, walk, ref low))
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
                        if (HoldsCore(arg, walk, ref low))
                            return true;

            // A declared class/struct: its own storing members, then its base. Only an assembly that
            // can name NDArray (NumSharp, this compilation, or one referencing NumSharp) can hold one;
            // a BCL or unrelated type is answered without a member scan.
            if (named.TypeKind != TypeKind.Class && named.TypeKind != TypeKind.Struct && named.TypeKind != TypeKind.Interface)
                return false;
            if (!IsInSource(named) && !CanNameNDArray(named.ContainingAssembly))
                return false;

            if (HolderMembersOf(named, walk, ref low).Count > 0)
                return true;
            return named.BaseType != null && HoldsCore(named.BaseType, walk, ref low);
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
