using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Threading;
using Microsoft.CodeAnalysis;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     The compilation-wide <b>ownership model</b> shared by the leak analyzer (NDW012) and the
    ///     holder analyzer (NDW016/NDW017): which TYPES hold NDArrays, which of those are disposable
    ///     (and therefore the "contagious" NDArray-owning values a method or a containing type must
    ///     reclaim), and which MEMBERS of a type are its holders. Resolved once per compilation and
    ///     memoized; safe under the analyzers' concurrent execution, and every answer depends only on
    ///     the compilation, never on the order the analyzer callbacks ask in.
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
    ///     <para>
    ///     <b>How it is computed.</b> The rules above define a directed graph over types — an edge
    ///     X → Y when a value of X stores (or is a container of) a value of Y — and "holds" is the
    ///     least fixed point of an OR over it: a type holds exactly when it can reach a type that holds
    ///     by itself. One iterative pass of Tarjan's strongly connected components algorithm visits each
    ///     type and each edge once and decides every component reachable from the type asked about (a
    ///     component is a set of types that all reach each other, so they share one answer). A domain
    ///     model whose types reference each other in cycles therefore costs time linear in its size,
    ///     and a deep model cannot exhaust the thread's stack (the walk keeps its own stack).
    ///     </para>
    ///     <para>
    ///     <b>Which members are holders.</b> A member of type <c>T</c> is a holder when its type holds
    ///     NDArrays WITHOUT passing back through <c>T</c>: a parent back-pointer (<c>Node _parent</c> in
    ///     <c>Node</c>), the link of a linked list (<c>Node _next</c>), or a member that reaches arrays
    ///     only through a type that comes back to <c>T</c> stores nothing <c>T</c> would have to dispose
    ///     on its own account, so it is not reported. A member type outside <c>T</c>'s component cannot
    ///     reach <c>T</c> at all, so its answer is its own; inside the component, the answer is "does
    ///     <c>T</c> dominate the member type" in a dominator tree built once per component (see
    ///     <see cref="Component"/>).
    ///     </para>
    /// </summary>
    internal sealed class OwnershipModel
    {
        private readonly KnownTypes _k;
        private readonly IAssemblySymbol _sourceAssembly;

        /// <summary>
        ///     The decided component of every type a walk has finished. Written only under
        ///     <see cref="_gate"/>, and only for whole components (every member of a component is
        ///     published before the walk that decided it releases the lock); read lock-free. A type
        ///     present here has its final answer, so a reader never sees a partial result.
        /// </summary>
        private readonly ConcurrentDictionary<ITypeSymbol, Component> _components =
            new ConcurrentDictionary<ITypeSymbol, Component>(SymbolEqualityComparer.Default);

        private readonly ConcurrentDictionary<IAssemblySymbol, bool> _canNameNDArray =
            new ConcurrentDictionary<IAssemblySymbol, bool>(SymbolEqualityComparer.Default);

        /// <summary>
        ///     The leak analyzer's per-type memo of "is a value of this type an owned NDArray carrier"
        ///     (<c>NDArrayLeakAnalyzer</c>'s <c>IsOwnedNDArrayType</c>). That test runs for EVERY operation
        ///     of EVERY method body in a project that references NumSharp — NDArray-free code included —
        ///     and costs two base-chain walks and three scans of the type's <c>AllInterfaces</c> (about thirty
        ///     generic-math interfaces for an <c>int</c>), while a compilation has only thousands of distinct
        ///     types: unmemoized it was most of NDW012's time on a large project that never names NDArray.
        ///     The answer is a pure function of the type within one compilation, so whichever callback
        ///     computes it first stores the one answer every other callback reads. Lives here because the
        ///     model is the analyzer's per-compilation state.
        /// </summary>
        internal ConcurrentDictionary<ITypeSymbol, bool> OwnedValueTypes { get; } =
            new ConcurrentDictionary<ITypeSymbol, bool>(SymbolEqualityComparer.Default);

        /// <summary>
        ///     Serializes the walks that decide new components. Two concurrent walks over overlapping
        ///     types would each assign their own <see cref="Component"/> object to the types they share,
        ///     and <see cref="HolderMembersOf"/> relies on the members of one component sharing one
        ///     object. Holding it is cheap: the Roslyn work of a walk (binding members and their types)
        ///     is done before taking it (<see cref="Prefetch"/>), every type is walked once per
        ///     compilation, and a decided type is answered from <see cref="_components"/> without it.
        /// </summary>
        private readonly object _gate = new object();

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

        /// <summary>Creates the model for one compilation; nothing is resolved until the first question.</summary>
        /// <param name="k">The NumSharp symbols the compilation resolved (never null: the analyzers bail out before creating a model when NumSharp is not referenced).</param>
        /// <param name="compilation">The compilation whose own assembly counts as able to name NDArray.</param>
        public OwnershipModel(KnownTypes k, Compilation compilation)
        {
            _k = k;
            _sourceAssembly = compilation.Assembly;
        }

        /// <summary>The NumSharp symbols this model classifies against.</summary>
        public KnownTypes Known => _k;

        /// <summary>A storing member of a type — the field or (auto-)property symbol to report, and the type it stores.</summary>
        public readonly struct HolderMember
        {
            /// <summary>The symbol a diagnostic names: the field, or the property for an auto-property's backing field.</summary>
            public readonly ISymbol Symbol;

            /// <summary>The type of the value the member stores.</summary>
            public readonly ITypeSymbol Type;

            /// <summary>Pairs a reported member with the type it stores.</summary>
            /// <param name="symbol">The field or property a diagnostic names.</param>
            /// <param name="type">The type of the stored value.</param>
            public HolderMember(ISymbol symbol, ITypeSymbol type)
            {
                Symbol = symbol;
                Type = type;
            }

            /// <summary>The member's name as the author wrote it.</summary>
            public string Name => Symbol.Name;

            /// <summary>"property" or "field", for messages.</summary>
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
        ///     <para>
        ///     A member whose type holds NDArrays only by coming back to <paramref name="type"/> is NOT a
        ///     holder: a parent back-pointer, a linked-list link, or a reference into the rest of a cycle
        ///     that reaches arrays only through <paramref name="type"/> stores nothing the type would
        ///     have to dispose on its own account. Reporting them would demand that a child dispose its
        ///     parent (NumSharp's own <c>NDScope._parent</c> is exactly that shape).
        ///     </para>
        /// </summary>
        /// <param name="type">The type whose own members are classified; null answers an empty list.</param>
        /// <returns>The holding members in declaration order (the order <see cref="INamespaceOrTypeSymbol.GetMembers()"/> returns); a fresh list the caller may keep.</returns>
        public List<HolderMember> HolderMembersOf(INamedTypeSymbol type)
        {
            var list = new List<HolderMember>();
            // A type that does not hold NDArrays has no holding member: every member type it stores is a
            // successor in the graph, so a member that held would make the type hold. This is also the
            // case of every type in a project that stores no NDArray at all: one memo lookup after the
            // first walk.
            if (type == null || !HoldsNDArrays(type))
                return list;

            var component = ComponentOf(type);
            foreach (var member in StoringMembers(type))
                if (HoldsWithout(type, component, member.Type))
                    list.Add(member);
            return list;
        }

        /// <summary>
        ///     Whether a value of <paramref name="memberType"/> holds NDArrays without passing through
        ///     <paramref name="owner"/> — the holder test for one member of <paramref name="owner"/>.
        /// </summary>
        /// <param name="owner">The type that declares the member.</param>
        /// <param name="ownerComponent">The decided component of <paramref name="owner"/>.</param>
        /// <param name="memberType">The type the member stores.</param>
        /// <returns>True when the member stores arrays the owner would have to dispose itself.</returns>
        private bool HoldsWithout(INamedTypeSymbol owner, Component ownerComponent, ITypeSymbol memberType)
        {
            // The member's own type is the owner (a self-reference, a linked-list link): it reaches
            // arrays only through the owner by definition.
            if (memberType == null || SymbolEqualityComparer.Default.Equals(memberType, owner))
                return false;
            if (!HoldsNDArrays(memberType))
                return false;

            // A member type in another component cannot reach the owner: the owner reaches it (it is a
            // member), so reaching back would put both in one component. Its answer never involved the
            // owner, so it is exact as it stands.
            var memberComponent = ComponentOf(memberType);
            if (!ReferenceEquals(memberComponent, ownerComponent))
                return true;

            // Inside the owner's component every path may lead back to the owner: the member type holds
            // without the owner unless the owner dominates it — sits on every path from it to a type that
            // holds by itself or through a type outside the component.
            return !ownerComponent.Dominates(owner, memberType);
        }

        /// <summary>
        ///     The storing members of <paramref name="type"/> as the ownership graph sees them, before any
        ///     "does it hold" test: instance, non-const, not <c>[NDBorrowed]</c>; a source type's explicit
        ///     fields and auto-properties (the backing field reported as its property), a metadata type's
        ///     visible fields and properties (indexers included).
        /// </summary>
        /// <param name="type">The declaring type; only members declared on it are produced (not inherited ones — the base type is its own edge).</param>
        /// <returns>The storing members in declaration order.</returns>
        private IEnumerable<HolderMember> StoringMembers(INamedTypeSymbol type)
        {
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
                yield return new HolderMember(reported, memberType);
            }
        }

        // ------------------------------------------------------------------ type predicates

        /// <summary>Does a value of this type (transitively) hold an NDArray somebody must dispose?</summary>
        /// <param name="t">The type asked about; null answers false.</param>
        /// <returns>True when a value of <paramref name="t"/> can contain an NDArray it owns.</returns>
        public bool HoldsNDArrays(ITypeSymbol t) => t != null && ComponentOf(t).Holds;

        /// <summary>
        ///     The decided component of <paramref name="t"/>, deciding it (and every component it reaches)
        ///     on the first question.
        /// </summary>
        /// <param name="t">A non-null type.</param>
        /// <returns>The component; the same object for every type of one strongly connected component.</returns>
        private Component ComponentOf(ITypeSymbol t)
        {
            if (_components.TryGetValue(t, out var component))
                return component;

            // The expensive part of a walk is Roslyn's: binding the members, member types and attributes
            // of every type reached. Done here, outside the lock, it runs on every analyzer thread that
            // asks at once (Roslyn's symbols are thread-safe and computed once); the walk under the lock
            // then only reads the cached edges. Holding the lock through the binding serialized it onto
            // one thread while the others waited: about 1.5x the holder analyzer's wall time on a
            // thousand-type model.
            Prefetch(t);

            lock (_gate)
            {
                // Another walk may have decided it while this thread waited for the lock.
                if (!_components.TryGetValue(t, out component))
                {
                    Resolve(t);
                    component = _components[t];
                }
            }
            return component;
        }

        /// <summary>
        ///     Expands, without the lock, every not-yet-decided type <paramref name="root"/> reaches, so the
        ///     walk under the lock finds every edge cached. A type decided meanwhile by another thread is
        ///     simply not expanded (the walk will not enter it either); a type expanded twice by two threads
        ///     at once costs only the duplicate work, because the expansion is a pure function of the type.
        /// </summary>
        /// <param name="root">The type asked about.</param>
        private void Prefetch(ITypeSymbol root)
        {
            var seen = new HashSet<ITypeSymbol>(SymbolEqualityComparer.Default) { root };
            var queue = new Queue<ITypeSymbol>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                foreach (var s in ExpansionOf(queue.Dequeue()).Successors)
                    if (s != null && !_components.ContainsKey(s) && seen.Add(s))
                        queue.Enqueue(s);
            }
        }

        /// <summary>A type's place in the ownership graph, as <see cref="Expand"/> computed it: cached per type, immutable.</summary>
        private sealed class Expansion
        {
            /// <summary>Whether the type holds NDArrays regardless of its successors (an NDArray, a carrier).</summary>
            public readonly bool HoldsItself;

            /// <summary>The types a value of this type stores: the edges out of the node.</summary>
            public readonly ITypeSymbol[] Successors;

            /// <summary>Freezes one type's expansion.</summary>
            /// <param name="holdsItself">Whether the type holds by itself.</param>
            /// <param name="successors">Its outgoing edges.</param>
            public Expansion(bool holdsItself, ITypeSymbol[] successors)
            {
                HoldsItself = holdsItself;
                Successors = successors;
            }
        }

        /// <summary>
        ///     Every expansion computed so far. Written lock-free (<see cref="Prefetch"/> runs outside
        ///     <see cref="_gate"/>); two threads expanding one type at once both compute it and the first
        ///     stored wins, which is harmless because <see cref="Expand"/> depends only on the type.
        /// </summary>
        private readonly ConcurrentDictionary<ITypeSymbol, Expansion> _expansions =
            new ConcurrentDictionary<ITypeSymbol, Expansion>(SymbolEqualityComparer.Default);

        /// <summary>The cached expansion of <paramref name="t"/>, computing it on first use.</summary>
        /// <param name="t">A non-null type.</param>
        /// <returns>The type's expansion.</returns>
        private Expansion ExpansionOf(ITypeSymbol t)
        {
            if (_expansions.TryGetValue(t, out var expansion))
                return expansion;
            var successors = new List<ITypeSymbol>();
            bool holdsItself = Expand(t, successors);
            return _expansions.GetOrAdd(t, new Expansion(holdsItself, successors.ToArray()));
        }

        /// <summary>One type on the current walk: Tarjan's bookkeeping plus the node's place in the graph.</summary>
        private sealed class WalkNode
        {
            /// <summary>The type this node stands for.</summary>
            public readonly ITypeSymbol Type;

            /// <summary>Tarjan's visit index: the order the walk entered the node.</summary>
            public readonly int Index;

            /// <summary>The node's position on the walk's stack of nodes whose component is still open.</summary>
            public readonly int StackPosition;

            /// <summary>The type's edges and own answer.</summary>
            public readonly Expansion Expansion;

            /// <summary>The smallest visit index known to be reachable from this node through open nodes (Tarjan's low-link).</summary>
            public int Low;

            /// <summary>The next successor the walk will follow.</summary>
            public int Next;

            /// <summary>Enters a node into the walk.</summary>
            /// <param name="type">The type the node stands for.</param>
            /// <param name="expansion">The type's edges and own answer.</param>
            /// <param name="index">Its visit index.</param>
            /// <param name="stackPosition">Its position on the open-node stack.</param>
            public WalkNode(ITypeSymbol type, Expansion expansion, int index, int stackPosition)
            {
                Type = type;
                Expansion = expansion;
                Index = index;
                Low = index;
                StackPosition = stackPosition;
            }
        }

        /// <summary>
        ///     Decides the component of <paramref name="root"/> and of every type it reaches that no
        ///     earlier walk decided: Tarjan's strongly connected components algorithm with an explicit
        ///     stack, so the depth of the type graph never meets the depth of the thread's stack (a model
        ///     of a few thousand types used to overflow it and take the compiler process down).
        ///     <para>
        ///     Every successor of every node is followed — no early exit on a first "holds" — because the
        ///     components must be the exact strongly connected components: <see cref="HolderMembersOf"/>
        ///     reads "outside the owner's component" as "cannot reach the owner", which a component cut
        ///     short by a skipped edge would get wrong. The cost is unchanged: each type and each edge is
        ///     still visited once per compilation, and the edges come from <see cref="ExpansionOf"/>'s
        ///     cache (filled by <see cref="Prefetch"/> before the lock was taken).
        ///     </para>
        /// </summary>
        /// <param name="root">A type not yet decided. Must be called under <see cref="_gate"/>.</param>
        private void Resolve(ITypeSymbol root)
        {
            var walked = new Dictionary<ITypeSymbol, WalkNode>(SymbolEqualityComparer.Default);
            var open = new List<WalkNode>();   // Tarjan's stack: visited, component not yet decided
            var path = new Stack<WalkNode>();  // the depth-first path the recursion would have been

            path.Push(Enter(root, walked, open));
            while (path.Count > 0)
            {
                var node = path.Peek();
                var successors = node.Expansion.Successors;
                if (node.Next < successors.Length)
                {
                    var next = successors[node.Next++];
                    if (next == null || _components.ContainsKey(next))
                        continue; // decided by this walk or an earlier one: its answer is final
                    if (walked.TryGetValue(next, out var seen))
                    {
                        // Visited and not decided, so still open: an edge back into the open part of the
                        // walk, which joins this node to that node's component.
                        if (seen.Index < node.Low)
                            node.Low = seen.Index;
                        continue;
                    }
                    path.Push(Enter(next, walked, open));
                    continue;
                }

                // Every successor followed: hand the low-link to the parent, and decide the component if
                // this node is the first one the walk entered it through.
                path.Pop();
                if (path.Count > 0 && node.Low < path.Peek().Low)
                    path.Peek().Low = node.Low;
                if (node.Low == node.Index)
                    Decide(node, open);
            }
        }

        /// <summary>Starts a node: gives it the next visit index, takes its (normally prefetched) edges, and puts it on the open stack.</summary>
        /// <param name="t">The type entered.</param>
        /// <param name="walked">The walk's visited nodes.</param>
        /// <param name="open">The walk's open-node stack.</param>
        /// <returns>The new node.</returns>
        private WalkNode Enter(ITypeSymbol t, Dictionary<ITypeSymbol, WalkNode> walked, List<WalkNode> open)
        {
            var node = new WalkNode(t, ExpansionOf(t), walked.Count, open.Count);
            walked.Add(t, node);
            open.Add(node);
            return node;
        }

        /// <summary>
        ///     Closes the component <paramref name="entry"/> heads: pops it and every node above it from
        ///     the open stack, decides whether the component holds, and publishes one
        ///     <see cref="Component"/> for all of its types.
        /// </summary>
        /// <param name="entry">The node the walk entered the component through (its low-link equals its index).</param>
        /// <param name="open">The walk's open-node stack.</param>
        private void Decide(WalkNode entry, List<WalkNode> open)
        {
            int count = open.Count - entry.StackPosition;
            var nodes = open.GetRange(entry.StackPosition, count);
            open.RemoveRange(entry.StackPosition, count);

            // A type ANCHORS the component when it holds by itself or stores a type outside the component
            // that holds. Every edge out of the component leads to a decided component (an edge to a
            // still-open node below the entry would have lowered the entry's low-link), and an edge to a
            // type of this component finds nothing in _components yet.
            var anchors = new bool[count];
            bool holds = false;
            for (int i = 0; i < count; i++)
            {
                var node = nodes[i];
                bool anchor = node.Expansion.HoldsItself;
                foreach (var s in node.Expansion.Successors)
                    if (s != null && _components.TryGetValue(s, out var outside) && outside.Holds)
                        anchor = true;
                anchors[i] = anchor;
                holds |= anchor;
            }

            // Only a holding component of two or more types can be asked "does this member hold without
            // its owner" about a member type inside it (a one-type component's only member type is the
            // owner itself), so only then is the shape kept for the dominator tree. A holding component
            // always gets an object of its own: HolderMembersOf tells "inside the owner's component" by
            // reference identity, so two holding components must never share one. Components that hold
            // nothing are never compared (HolderMembersOf only compares holding ones) and share one.
            Component component;
            if (!holds)
            {
                component = Component.NotHolding;
            }
            else if (count == 1)
            {
                component = new Component(true, null, null, null, null);
            }
            else
            {
                var types = new ITypeSymbol[count];
                var position = new Dictionary<ITypeSymbol, int>(count, SymbolEqualityComparer.Default);
                for (int i = 0; i < count; i++)
                {
                    types[i] = nodes[i].Type;
                    position[types[i]] = i;
                }

                var inside = new int[count][];
                var scratch = new List<int>();
                for (int i = 0; i < count; i++)
                {
                    scratch.Clear();
                    foreach (var s in nodes[i].Expansion.Successors)
                        if (s != null && position.TryGetValue(s, out int j))
                            scratch.Add(j);
                    inside[i] = scratch.ToArray();
                }

                component = new Component(true, types, position, inside, anchors);
            }

            foreach (var node in nodes)
                _components[node.Type] = component;
        }

        /// <summary>
        ///     The node <paramref name="t"/> is in the ownership graph: whether it holds by itself, and its
        ///     successors (the types a value of it stores). Mirrors the classification rules of the type
        ///     summary in order; a rule that answers false outright leaves the node without successors.
        /// </summary>
        /// <param name="t">The type expanded (never null).</param>
        /// <param name="successors">Receives the edges out of the node.</param>
        /// <returns>True when the type holds NDArrays whatever its successors hold.</returns>
        private bool Expand(ITypeSymbol t, List<ITypeSymbol> successors)
        {
            switch (t)
            {
                case IArrayTypeSymbol arr:
                    successors.Add(arr.ElementType);
                    return false;
                case IPointerTypeSymbol _:
                case IDynamicTypeSymbol _:
                    return false;
                case ITypeParameterSymbol tp:
                    successors.AddRange(tp.ConstraintTypes);
                    return false;
            }

            bool holdsItself = false;
            if (TypeHelpers.IsNDArrayLike(t, _k))
            {
                // NDArray, or a subclass of it. Its answer is decided, but a SOURCE subclass is a type the
                // holder analyzer asks about, and its members can lead back to it: their edges must be in
                // the graph, or a member that reaches arrays only through it would look like a holder.
                // NumSharp's own NDArray family (metadata) is never asked about, so it is not expanded.
                holdsItself = true;
                if (!IsInSource(t))
                    return true;
            }
            if (!(t is INamedTypeSymbol named))
                return holdsItself;

            switch (named.TypeKind)
            {
                case TypeKind.Delegate:
                case TypeKind.Enum:
                case TypeKind.Error:
                    return holdsItself;
            }

            if (IsBorrowed(named))
                return holdsItself;
            if (named.IsTupleType)
            {
                foreach (var e in named.TupleElements)
                    successors.Add(e.Type);
                return holdsItself;
            }
            if (IsCarrier(named))
                return true;
            if (named.IsGenericType && NonHoldingGenericFamilies.Contains(FamilyKey(named)))
                return holdsItself;

            // A generic instantiation over a holding type is a container of it — List<NDArray>,
            // Task<NDArray>, Lazy<NDArray>, KeyValuePair<K, NDArray>, a consumer's Box<NDArray> —
            // and so is a type NESTED in one (Dictionary<K, NDArray>.ValueCollection, List<NDArray>.Enumerator).
            for (var g = named; g != null; g = g.ContainingType)
                if (g.IsGenericType)
                    successors.AddRange(g.TypeArguments);

            // A declared class/struct: its own storing members, then its base. Only an assembly that
            // can name NDArray (NumSharp, this compilation, or one referencing NumSharp) can hold one;
            // a BCL or unrelated type is answered without a member scan.
            if (named.TypeKind != TypeKind.Class && named.TypeKind != TypeKind.Struct && named.TypeKind != TypeKind.Interface)
                return holdsItself;
            if (!IsInSource(named) && !CanNameNDArray(named.ContainingAssembly))
                return holdsItself;

            foreach (var member in StoringMembers(named))
                successors.Add(member.Type);
            if (named.BaseType != null)
                successors.Add(named.BaseType);
            return holdsItself;
        }

        /// <summary>
        ///     One decided strongly connected component of the ownership graph: the types in it all reach
        ///     each other, so they share one "holds" answer. A holding component of two or more types also
        ///     keeps its internal shape, from which it answers <see cref="Dominates"/> — the question
        ///     <see cref="HolderMembersOf"/> asks about a member whose type is in its owner's component.
        ///     <para>
        ///     The dominator question is asked on the REVERSED edges inside the component, rooted at a
        ///     virtual node joined to every anchor (a type that holds by itself or stores a type outside
        ///     the component that holds). A path from that root to a type is a path from the type to an
        ///     anchor read backwards, so "every path from the type to an anchor passes through the owner"
        ///     is exactly "the owner dominates the type". The tree is built with the iterative algorithm
        ///     of Cooper, Harvey and Kennedy ("A Simple, Fast Dominance Algorithm"), on the first question
        ///     and once per component, so each member is then classified in constant time.
        ///     </para>
        /// </summary>
        private sealed class Component
        {
            /// <summary>
            ///     The one object shared by every component that holds nothing. Safe to share because
            ///     component identity is only ever compared between HOLDING components (each of which owns
            ///     its object).
            /// </summary>
            public static readonly Component NotHolding = new Component(false, null, null, null, null);

            /// <summary>Whether every type of the component holds NDArrays.</summary>
            public readonly bool Holds;

            private readonly ITypeSymbol[] _types;
            private readonly Dictionary<ITypeSymbol, int> _position;
            private readonly int[][] _inside;
            private readonly bool[] _anchors;
            private readonly object _treeLock = new object();

            // The dominator tree's depth-first entry and exit numbers per type: a dominates b exactly when
            // a's interval contains b's. Null until the first Dominates question.
            private int[] _enter;
            private int[] _exit;

            /// <summary>Creates a component.</summary>
            /// <param name="holds">The component's answer.</param>
            /// <param name="types">Its types (null for a component no dominator question can reach).</param>
            /// <param name="position">Each type's index into <paramref name="types"/>.</param>
            /// <param name="inside">Per type, the indexes of its successors inside the component.</param>
            /// <param name="anchors">Per type, whether it holds by itself or through a type outside the component.</param>
            public Component(bool holds, ITypeSymbol[] types, Dictionary<ITypeSymbol, int> position, int[][] inside, bool[] anchors)
            {
                Holds = holds;
                _types = types;
                _position = position;
                _inside = inside;
                _anchors = anchors;
            }

            /// <summary>
            ///     Whether every path from <paramref name="node"/> to an anchor of this component passes
            ///     through <paramref name="gate"/> — so a value of <paramref name="node"/> holds NDArrays
            ///     only by coming back to <paramref name="gate"/>.
            /// </summary>
            /// <param name="gate">A type of this component (the owner of a member).</param>
            /// <param name="node">A type of this component (the member's type).</param>
            /// <returns>True when <paramref name="gate"/> dominates <paramref name="node"/>; a type always dominates itself.</returns>
            /// <exception cref="InvalidOperationException">Either type is not in this component, or the component keeps no shape (it does not hold, or has one type) — a caller bug, never input-dependent.</exception>
            public bool Dominates(ITypeSymbol gate, ITypeSymbol node)
            {
                if (_position == null || !_position.TryGetValue(gate, out int g) || !_position.TryGetValue(node, out int n))
                    throw new InvalidOperationException("Dominates is only defined for two types of one holding component of two or more types.");

                // Built once, by whichever analyzer thread asks first; the volatile read pairs with the
                // volatile write that publishes _enter after _exit, so a reader that sees the tree sees
                // all of it.
                var enter = Volatile.Read(ref _enter);
                if (enter == null)
                {
                    lock (_treeLock)
                    {
                        enter = Volatile.Read(ref _enter);
                        if (enter == null)
                        {
                            BuildTree();
                            enter = _enter;
                        }
                    }
                }
                return enter[g] <= enter[n] && _exit[n] <= _exit[g];
            }

            /// <summary>
            ///     Builds the dominator tree of the reversed component rooted at the virtual anchor root,
            ///     and numbers it for interval tests. Every type is reachable from the root: the component
            ///     holds, so it has an anchor, and every type of a strongly connected component reaches it.
            /// </summary>
            private void BuildTree()
            {
                int count = _types.Length;
                int root = count;

                // Reversed edges: a successor s of t (t stores s) becomes an edge s -> t, so the reversed
                // graph's successors of s are the component's predecessors of s. The root's successors
                // are the anchors.
                var reversed = new List<int>[count + 1];
                for (int i = 0; i <= count; i++)
                    reversed[i] = new List<int>();
                for (int t = 0; t < count; t++)
                {
                    foreach (int s in _inside[t])
                        reversed[s].Add(t);
                    if (_anchors[t])
                        reversed[root].Add(t);
                }

                // Postorder numbers (the root last) and reverse postorder, by an iterative depth-first walk.
                var postorder = new int[count + 1];
                var order = new List<int>(count + 1);
                var seen = new bool[count + 1];
                var stack = new Stack<(int node, int next)>();
                stack.Push((root, 0));
                seen[root] = true;
                while (stack.Count > 0)
                {
                    var (node, next) = stack.Pop();
                    if (next < reversed[node].Count)
                    {
                        stack.Push((node, next + 1));
                        int child = reversed[node][next];
                        if (!seen[child])
                        {
                            seen[child] = true;
                            stack.Push((child, 0));
                        }
                        continue;
                    }
                    postorder[node] = order.Count;
                    order.Add(node);
                }
                order.Reverse(); // reverse postorder: the root first

                // Cooper-Harvey-Kennedy: iterate "idom(b) = the common dominator of b's processed
                // predecessors" in reverse postorder to a fixed point. A predecessor of b in the reversed
                // graph is a successor of b in the component, or the root when b is an anchor.
                var idom = new int[count + 1];
                for (int i = 0; i <= count; i++)
                    idom[i] = -1;
                idom[root] = root;
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    for (int k = 1; k < order.Count; k++)
                    {
                        int b = order[k];
                        int candidate = _anchors[b] ? root : -1;
                        foreach (int p in _inside[b])
                        {
                            if (p == b || idom[p] == -1)
                                continue; // a self-loop never decides anything; an unprocessed node not yet
                            candidate = candidate == -1 ? p : Intersect(p, candidate, idom, postorder);
                        }
                        if (idom[b] != candidate)
                        {
                            idom[b] = candidate;
                            changed = true;
                        }
                    }
                }

                // Interval numbering of the dominator tree.
                var children = new List<int>[count + 1];
                for (int i = 0; i <= count; i++)
                    children[i] = new List<int>();
                for (int i = 0; i < count; i++)
                    children[idom[i]].Add(i);

                var enter = new int[count + 1];
                var exit = new int[count + 1];
                int clock = 0;
                var walk = new Stack<(int node, int next)>();
                walk.Push((root, 0));
                enter[root] = clock++;
                while (walk.Count > 0)
                {
                    var (node, next) = walk.Pop();
                    if (next < children[node].Count)
                    {
                        walk.Push((node, next + 1));
                        int child = children[node][next];
                        enter[child] = clock++;
                        walk.Push((child, 0));
                        continue;
                    }
                    exit[node] = clock++;
                }

                _exit = exit;
                Volatile.Write(ref _enter, enter); // published last: Dominates reads _enter as "the tree is ready"
            }

            /// <summary>The nearest common dominator of two nodes, climbing the partial tree by postorder number.</summary>
            /// <param name="a">A node whose dominator is known.</param>
            /// <param name="b">Another node whose dominator is known.</param>
            /// <param name="idom">The immediate dominators found so far.</param>
            /// <param name="postorder">Postorder numbers (an ancestor's is higher).</param>
            /// <returns>The common dominator.</returns>
            private static int Intersect(int a, int b, int[] idom, int[] postorder)
            {
                while (a != b)
                {
                    while (postorder[a] < postorder[b])
                        a = idom[a];
                    while (postorder[b] < postorder[a])
                        b = idom[b];
                }
                return a;
            }
        }

        /// <summary>
        ///     Implements <c>IDisposable</c> or <c>IAsyncDisposable</c>, or is a <c>ref struct</c> with the
        ///     <c>public void Dispose()</c> pattern <c>using</c> accepts.
        /// </summary>
        /// <param name="t">The type asked about; null answers false.</param>
        /// <returns>True when a value of the type can be disposed.</returns>
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
        /// <param name="t">A disposable type.</param>
        /// <returns>The contract's name as the messages spell it.</returns>
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
        /// <param name="t">The type asked about; null answers false.</param>
        /// <returns>True for a disposable owner of NDArrays.</returns>
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
        /// <param name="t">The type asked about; null answers false.</param>
        /// <returns>True for a carrier.</returns>
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
        /// <param name="s">The member or type asked about; null answers false.</param>
        /// <returns>True when the author marked it as referencing arrays owned elsewhere.</returns>
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
        /// <param name="s">The symbol asked about; null answers false.</param>
        /// <returns>True when the symbol is declared in source.</returns>
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
