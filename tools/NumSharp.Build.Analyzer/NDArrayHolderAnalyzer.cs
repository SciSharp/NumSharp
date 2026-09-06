using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     <b>NDW016 / NDW017 — NDArray ownership at the TYPE level.</b> The per-method leak analyzer
    ///     (NDW012) treats a value stored into a field or property as a legitimate escape: the storing
    ///     type now owns it. This analyzer closes that hand-off — ownership must not evaporate at a
    ///     field boundary:
    ///     <list type="bullet">
    ///         <item><b>NDW016</b> (one per type): a class or struct declares instance fields /
    ///         auto-properties that HOLD NDArrays (an <c>NDArray</c>, an <c>NDArray[]</c>, a tuple or
    ///         collection of them, an <c>INDArrayCarrier</c> struct, a generic instantiated over one, or
    ///         another type that stores them — ownership is contagious) yet implements neither
    ///         <c>IDisposable</c> nor <c>IAsyncDisposable</c>, so nothing can ever reclaim them promptly.</item>
    ///         <item><b>NDW017</b> (one per member): the type IS disposable, but this holding member is
    ///         not disposed on any path reachable from its <c>Dispose()</c> / <c>Dispose(bool)</c> /
    ///         <c>DisposeAsync()</c> / <c>DisposeAsyncCore()</c> — including same-type helpers those call,
    ///         <c>foreach</c>-and-dispose over a collection member, <c>ForEach</c> lambdas, tuple
    ///         components, deconstruction, <c>using</c>, and handing the member to another method.</item>
    ///     </list>
    ///     A member marked <c>[NDBorrowed]</c> references an array owned elsewhere and is excluded; a
    ///     type marked <c>[NDBorrowed]</c>, a static class, and an <c>INDArrayCarrier</c> (transient
    ///     result packaging a scope yields through) are exempt. Both diagnostics are WARNINGS.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NDArrayHolderAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Build";
        private const string HelpLink16 = "https://scisharp.github.io/NumSharp/docs/numsharp-build-compiler.html#ndw016";
        private const string HelpLink17 = "https://scisharp.github.io/NumSharp/docs/numsharp-build-compiler.html#ndw017";

        internal static readonly DiagnosticDescriptor Nw016 = new DiagnosticDescriptor(
            "NDW016",
            "Type stores NDArrays but is not disposable",
            "'{0}' stores NDArrays in {1} but implements neither IDisposable nor IAsyncDisposable, so the " +
            "arrays it holds can only be reclaimed by the finalizer. Implement IDisposable (or IAsyncDisposable) " +
            "and dispose them there{2}, or mark a member that only references an array owned elsewhere [NDBorrowed].",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A type that stores an NDArray (directly, or through an array, tuple, collection, " +
                         "carrier struct or another NDArray-owning type) owns its pooled buffer. Ownership " +
                         "is contagious: unless the type is disposable and disposes the member, the buffer " +
                         "is left to the finalizer — the cost [NDScoped] and the leak analyzer exist to remove.",
            helpLinkUri: HelpLink16);

        internal static readonly DiagnosticDescriptor Nw017 = new DiagnosticDescriptor(
            "NDW017",
            "NDArray-holding member is never disposed",
            "'{0}' implements {1} but never disposes '{2}' (a {3} holding NDArrays) on any path from its " +
            "Dispose/DisposeAsync — dispose it there{4}, or mark it [NDBorrowed] if it references an array owned elsewhere.",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A disposable type must dispose every NDArray-holding member it owns from its " +
                         "Dispose path (Dispose(), Dispose(bool), DisposeAsync(), DisposeAsyncCore(), or a " +
                         "same-type helper they call). A member the path never reaches is left to the finalizer.",
            helpLinkUri: HelpLink17);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw016, Nw017);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterCompilationStartAction(start =>
            {
                var known = KnownTypes.Resolve(start.Compilation);
                if (known == null || known.NDArray == null)
                    return; // NumSharp not referenced — nothing to analyze

                var model = new OwnershipModel(known, start.Compilation);
                start.RegisterSymbolStartAction(sc => AnalyzeType(sc, model), SymbolKind.NamedType);
            });
        }

        private static void AnalyzeType(SymbolStartAnalysisContext sc, OwnershipModel model)
        {
            var type = (INamedTypeSymbol)sc.Symbol;
            if (!IsCandidate(type, model))
                return;

            var holders = model.HolderMembersOf(type);
            if (holders.Count == 0)
                return;

            if (!model.IsDisposable(type))
            {
                sc.RegisterSymbolEndAction(ec => ReportNotDisposable(ec, type, holders));
                return;
            }

            var scan = new DisposeScan(type, holders, model);
            sc.RegisterOperationBlockAction(scan.Collect);
            sc.RegisterSymbolEndAction(scan.Report);
        }

        /// <summary>A source class/struct (records included) that can own: not static, not <c>[NDBorrowed]</c>, not a carrier.</summary>
        private static bool IsCandidate(INamedTypeSymbol type, OwnershipModel model)
        {
            if (type.TypeKind != TypeKind.Class && type.TypeKind != TypeKind.Struct)
                return false;
            if (type.IsStatic || type.IsImplicitlyDeclared)
                return false;
            if (!OwnershipModel.IsInSource(type))
                return false;
            if (model.IsBorrowed(type))
                return false;
            // An INDArrayCarrier is transient result packaging by contract — the scope disposes THROUGH
            // it (YieldTo); asking it to be disposable would contradict the carrier vocabulary.
            if (model.IsCarrier(type))
                return false;
            return true;
        }

        private static void ReportNotDisposable(SymbolAnalysisContext ec, INamedTypeSymbol type, List<OwnershipModel.HolderMember> holders)
        {
            string tail = type.IsRefLikeType
                ? " (for a ref struct, a public void Dispose() is the pattern)"
                : type.TypeKind == TypeKind.Struct
                    ? ", implement INDArrayCarrier if this struct is a transient result a scope yields through"
                    : "";
            ec.ReportDiagnostic(Diagnostic.Create(Nw016, Loc(type), Display(type), FormatMembers(holders), tail));
        }

        /// <summary><c>'a'</c> / <c>'a' and 'b'</c> / <c>'a', 'b' and 'c'</c> / <c>'a', 'b', 'c' and 2 more</c>.</summary>
        private static string FormatMembers(List<OwnershipModel.HolderMember> holders)
        {
            const int shown = 3;
            var parts = new List<string>();
            for (int i = 0; i < holders.Count && i < shown; i++)
                parts.Add("'" + holders[i].Name + "'");
            if (holders.Count > shown)
                parts.Add((holders.Count - shown) + " more");
            if (parts.Count == 1)
                return parts[0];
            var sb = new StringBuilder();
            for (int i = 0; i < parts.Count - 1; i++)
                sb.Append(i > 0 ? ", " : "").Append(parts[i]);
            return sb.Append(" and ").Append(parts[parts.Count - 1]).ToString();
        }

        private static Location Loc(ISymbol s) => s.Locations.Length > 0 ? s.Locations[0] : Location.None;
        private static string Display(ISymbol s) => s.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);

        // ================================================================== the Dispose-path scan

        /// <summary>What one member body contributes: the holders it disposes, and the same-type members it calls.</summary>
        private sealed class BodyFacts
        {
            public readonly HashSet<ISymbol> Disposed = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
            public readonly HashSet<IMethodSymbol> Callees = new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);

            public BodyFacts Merge(BodyFacts other)
            {
                Disposed.UnionWith(other.Disposed);
                Callees.UnionWith(other.Callees);
                return this;
            }
        }

        /// <summary>
        ///     One disposable type's worth of NDW017 analysis: every member body of the type records
        ///     which holders it disposes and which same-type members it calls; at symbol end the
        ///     dispose roots' transitive reach decides which holders are disposed.
        /// </summary>
        private sealed class DisposeScan
        {
            private readonly INamedTypeSymbol _type;
            private readonly List<OwnershipModel.HolderMember> _holders;
            private readonly HashSet<ISymbol> _holderSymbols;
            private readonly OwnershipModel _model;
            private readonly ConcurrentDictionary<IMethodSymbol, BodyFacts> _facts =
                new ConcurrentDictionary<IMethodSymbol, BodyFacts>(SymbolEqualityComparer.Default);

            public DisposeScan(INamedTypeSymbol type, List<OwnershipModel.HolderMember> holders, OwnershipModel model)
            {
                _type = type;
                _holders = holders;
                _model = model;
                _holderSymbols = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
                foreach (var h in holders)
                    _holderSymbols.Add(h.Symbol);
            }

            public void Collect(OperationBlockAnalysisContext bc)
            {
                // Only this type's own member bodies (a nested type runs its own scan). Field
                // initializers are owned by the field symbol and carry no disposal.
                if (!(bc.OwningSymbol is IMethodSymbol m) || !IsOurs(m))
                    return;
                var facts = new BodyWalk(this).Run(bc.OperationBlocks);
                _facts.AddOrUpdate(m.OriginalDefinition, facts, (_, old) => old.Merge(facts));
            }

            public void Report(SymbolAnalysisContext ec)
            {
                var roots = new List<IMethodSymbol>();
                foreach (var m in _type.GetMembers())
                    if (m is IMethodSymbol method && IsDisposeRoot(method))
                        roots.Add(method.OriginalDefinition);

                var reachable = new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);
                var queue = new Queue<IMethodSymbol>(roots);
                while (queue.Count > 0)
                {
                    var m = queue.Dequeue();
                    if (!reachable.Add(m))
                        continue;
                    if (_facts.TryGetValue(m, out var f))
                        foreach (var c in f.Callees)
                            queue.Enqueue(c);
                }

                var disposed = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
                foreach (var m in reachable)
                    if (_facts.TryGetValue(m, out var f))
                        disposed.UnionWith(f.Disposed);

                string typeName = Display(_type);
                string contract = _model.DisposableContractOf(_type);
                foreach (var h in _holders)
                {
                    if (disposed.Contains(h.Symbol))
                        continue;
                    string hint = roots.Count == 0 ? InheritedDisposeHint() : NonDisposableHolderHint(h);
                    ec.ReportDiagnostic(Diagnostic.Create(Nw017, Loc(h.Symbol), typeName, contract, h.Name, h.Kind, hint));
                }
            }

            private string InheritedDisposeHint()
            {
                for (var b = _type.BaseType; b != null; b = b.BaseType)
                    if (_model.IsDisposable(b))
                        return " (its Dispose is inherited from '" + Display(b) + "' — override it, or reimplement IDisposable here)";
                return " (this type declares no Dispose of its own)";
            }

            /// <summary>The fix is blocked one level down: the member's type stores NDArrays but cannot be disposed yet.</summary>
            private string NonDisposableHolderHint(OwnershipModel.HolderMember h)
            {
                if (h.Type is INamedTypeSymbol n
                    && (n.TypeKind == TypeKind.Class || n.TypeKind == TypeKind.Struct)
                    && !n.IsTupleType && !n.IsGenericType
                    && !TypeHelpers.IsNDArrayLike(n, _model.Known)
                    && !_model.IsCarrier(n)
                    && !_model.IsDisposable(n))
                    return " (make '" + Display(n) + "' disposable first — it stores NDArrays but is not itself disposable)";
                return "";
            }

            internal OwnershipModel Model => _model;

            internal bool IsOurs(ISymbol member)
                => member?.ContainingType != null &&
                   SymbolEqualityComparer.Default.Equals(member.ContainingType.OriginalDefinition, _type.OriginalDefinition);

            internal bool IsHolder(ISymbol s) => s != null && _holderSymbols.Contains(s);

            /// <summary>
            ///     A dispose ROOT: the type's own <c>Dispose()</c> / <c>Dispose(bool)</c> /
            ///     <c>DisposeAsync()</c> / <c>DisposeAsyncCore()</c>, by name or as an explicit interface
            ///     implementation. A finalizer is deliberately NOT a root — it is the backstop whose cost
            ///     prompt disposal exists to avoid.
            /// </summary>
            private static bool IsDisposeRoot(IMethodSymbol m)
            {
                if (m.IsStatic || m.MethodKind == MethodKind.Destructor || m.MethodKind == MethodKind.Constructor)
                    return false;
                if (m.Parameters.Length > 1)
                    return false;
                if (IsDisposeName(m.Name) || m.Name == "DisposeAsyncCore")
                    return true;
                foreach (var e in m.ExplicitInterfaceImplementations)
                    if (IsDisposeName(e.Name))
                        return true;
                return false;
            }

            private static bool IsDisposeName(string name)
                => name == "Dispose" || name == "DisposeAsync"
                   || name.EndsWith(".Dispose", System.StringComparison.Ordinal)
                   || name.EndsWith(".DisposeAsync", System.StringComparison.Ordinal);

            /// <summary>Any parameterless-or-bool <c>Dispose</c> / <c>DisposeAsync</c> / <c>Close</c> / <c>close</c> (NumSharp's iterator spelling).</summary>
            internal static bool IsDisposeLike(IMethodSymbol m)
            {
                if (m == null || m.Parameters.Length > 1)
                    return false;
                var n = m.Name;
                return IsDisposeName(n) || n == "Close" || n == "close";
            }
        }

        /// <summary>One member body's walk: derived locals first (a local/loop variable/deconstructed name that IS a holder's value), then the disposal facts.</summary>
        private sealed class BodyWalk
        {
            private readonly DisposeScan _scan;
            private readonly Dictionary<ISymbol, ISymbol> _derived = new Dictionary<ISymbol, ISymbol>(SymbolEqualityComparer.Default);

            public BodyWalk(DisposeScan scan)
            {
                _scan = scan;
            }

            public BodyFacts Run(ImmutableArray<IOperation> blocks)
            {
                var facts = new BodyFacts();
                var all = new List<IOperation>();
                foreach (var block in blocks)
                    foreach (var op in block.DescendantsAndSelf())
                        all.Add(op);

                // Derived locals, to a fixed point (a chain `var l = _list; var f = l[0];` resolves in
                // one pass in declaration order; assignments out of order need another).
                for (int round = 0; round < 4; round++)
                {
                    int before = _derived.Count;
                    foreach (var op in all)
                    {
                        switch (op)
                        {
                            case IVariableDeclaratorOperation d when d.Initializer?.Value != null:
                                Derive(d.Symbol, d.Initializer.Value);
                                break;
                            case ISimpleAssignmentOperation asg when asg.Target is ILocalReferenceOperation l:
                                Derive(l.Local, asg.Value);
                                break;
                            case IForEachLoopOperation loop:
                            {
                                var root = RootHolder(loop.Collection, allowThroughInvocation: true);
                                if (root != null)
                                    foreach (var local in LocalsOf(loop.LoopControlVariable))
                                        _derived[local] = root;
                                break;
                            }
                            case IDeconstructionAssignmentOperation dec:
                            {
                                var root = RootHolder(dec.Value, allowThroughInvocation: true);
                                if (root != null)
                                    foreach (var local in LocalsOf(dec.Target))
                                        _derived[local] = root;
                                break;
                            }
                        }
                    }
                    if (_derived.Count == before)
                        break;
                }

                foreach (var op in all)
                {
                    switch (op)
                    {
                        case IInvocationOperation inv:
                            Invocation(inv, facts);
                            break;
                        case IObjectCreationOperation oc:
                            // handing a holder to a constructor (a composite disposable, a wrapper) hands it off
                            foreach (var a in oc.Arguments)
                                HandOff(a.Value, facts);
                            break;
                        case IUsingOperation u:
                            Using(u.Resources, facts);
                            break;
                        case IUsingDeclarationOperation ud:
                            Using(ud.DeclarationGroup, facts);
                            break;
                        case IPropertyReferenceOperation pr when _scan.IsOurs(pr.Property):
                            // a same-type property read/write runs its accessor — a callee
                            if (pr.Property.GetMethod != null) facts.Callees.Add(pr.Property.GetMethod.OriginalDefinition);
                            if (pr.Property.SetMethod != null) facts.Callees.Add(pr.Property.SetMethod.OriginalDefinition);
                            break;
                        case IMethodReferenceOperation mr when _scan.IsOurs(mr.Method):
                            facts.Callees.Add(mr.Method.OriginalDefinition); // a delegate over a same-type method
                            break;
                    }
                }
                return facts;
            }

            private void Derive(ILocalSymbol local, IOperation value)
            {
                if (local == null)
                    return;
                var root = RootHolder(value, allowThroughInvocation: true);
                if (root != null)
                    _derived[local] = root;
            }

            private void Invocation(IInvocationOperation inv, BodyFacts facts)
            {
                var tm = inv.TargetMethod;
                if (tm != null && _scan.IsOurs(tm))
                    facts.Callees.Add(tm.OriginalDefinition);

                var receiver = ReceiverOf(inv);
                if (receiver != null && DisposeScan.IsDisposeLike(tm))
                {
                    var root = RootHolder(receiver, allowThroughInvocation: true);
                    if (root != null)
                        facts.Disposed.Add(root);
                }

                // A holder handed to ANY method (a helper, Array.ForEach, Parallel.ForEach,
                // Interlocked.Exchange(ref _a, null)) is assumed disposed by it.
                foreach (var a in inv.Arguments)
                    HandOff(a.Value, facts);

                // `_list.ForEach(x => x.Dispose())` and kin: a call ON the holder that runs a lambda
                // which disposes — the lambda's parameter is the element.
                if (receiver != null && HasDisposingLambdaArgument(inv))
                {
                    var root = RootHolder(receiver, allowThroughInvocation: true);
                    if (root != null)
                        facts.Disposed.Add(root);
                }
            }

            private void HandOff(IOperation value, BodyFacts facts)
            {
                // Direct hand-off only: `Free(_a)`, `Free(_list.Values)`, `Free(_pair.Item1)`,
                // `Free(ref _a)` — the argument must itself HOLD arrays (so `Log(x.size)` on a loop
                // element hands off a scalar, not the list) and must not be a call's result
                // (`Log(_a.ToString())` hands off a string computed from the holder).
                if (value == null || !_scan.Model.HoldsNDArrays(value.Type))
                    return;
                var root = RootHolder(value, allowThroughInvocation: false);
                if (root != null)
                    facts.Disposed.Add(root);
            }

            private void Using(IOperation resources, BodyFacts facts)
            {
                if (resources == null)
                    return;
                if (resources is IVariableDeclarationGroupOperation group)
                {
                    foreach (var decl in group.Declarations)
                        foreach (var d in decl.Declarators)
                            if (d.Initializer?.Value != null)
                            {
                                var root = RootHolder(d.Initializer.Value, allowThroughInvocation: true);
                                if (root != null)
                                    facts.Disposed.Add(root);
                            }
                    return;
                }
                var r = RootHolder(resources, allowThroughInvocation: true);
                if (r != null)
                    facts.Disposed.Add(r);
            }

            private static IOperation ReceiverOf(IInvocationOperation inv)
            {
                if (inv.Instance != null)
                    return inv.Instance;
                // an extension method in reduced form carries its receiver as the first argument
                if (inv.TargetMethod != null && inv.TargetMethod.IsExtensionMethod && inv.Arguments.Length > 0)
                    return inv.Arguments[0].Value;
                return null;
            }

            private static bool HasDisposingLambdaArgument(IInvocationOperation inv)
            {
                foreach (var a in inv.Arguments)
                {
                    var v = a.Value;
                    while (v is IConversionOperation c && c.Operand != null)
                        v = c.Operand;
                    if (v is IDelegateCreationOperation dc)
                        v = dc.Target;
                    if (!(v is IAnonymousFunctionOperation fn))
                        continue;
                    foreach (var d in fn.Descendants())
                        if (d is IInvocationOperation di && DisposeScan.IsDisposeLike(di.TargetMethod))
                            return true;
                }
                return false;
            }

            /// <summary>
            ///     The holder member an expression's value ROOTS at — walking down through conversions,
            ///     parentheses, <c>?.</c>, tuple/field/property reads (<c>_pair.Item1</c>, <c>_dict.Values</c>),
            ///     indexers and array elements (<c>_arr[i]</c>, <c>_list[i]</c>), locals derived from a
            ///     holder, and (when allowed) call results on it (<c>_list.First()</c>). Null when the
            ///     value is not rooted at a holder of this type.
            /// </summary>
            private ISymbol RootHolder(IOperation op, bool allowThroughInvocation)
            {
                for (int guard = 0; guard < 64 && op != null; guard++)
                {
                    switch (op)
                    {
                        case IConversionOperation c:
                            op = c.Operand;
                            continue;
                        case IParenthesizedOperation p:
                            op = p.Operand;
                            continue;
                        case IAwaitOperation aw:
                            op = aw.Operation;
                            continue;
                        case IConditionalAccessOperation ca:
                            op = ca.Operation;
                            continue;
                        case IConditionalAccessInstanceOperation cai:
                        {
                            // `_a?.Dispose()`: the invocation's instance is the placeholder for the
                            // enclosing conditional access's receiver.
                            IOperation anc = cai.Parent;
                            while (anc != null && !(anc is IConditionalAccessOperation))
                                anc = anc.Parent;
                            op = (anc as IConditionalAccessOperation)?.Operation;
                            continue;
                        }
                        case IFieldReferenceOperation f:
                            if (_scan.IsHolder(f.Field))
                                return f.Field;
                            op = f.Instance; // a tuple component / a field of a nested holder value
                            continue;
                        case IPropertyReferenceOperation pr:
                            if (_scan.IsHolder(pr.Property))
                                return pr.Property;
                            op = pr.Instance; // an indexer / .Value / .Values on a holder
                            continue;
                        case IArrayElementReferenceOperation ae:
                            op = ae.ArrayReference;
                            continue;
                        case IInvocationOperation inv:
                            if (!allowThroughInvocation)
                                return null;
                            op = ReceiverOf(inv);
                            continue;
                        case ILocalReferenceOperation l:
                            return _derived.TryGetValue(l.Local, out var viaLocal) ? viaLocal : null;
                        case IParameterReferenceOperation prm:
                            return _derived.TryGetValue(prm.Parameter, out var viaParam) ? viaParam : null;
                        default:
                            return null;
                    }
                }
                return null;
            }

            private static IEnumerable<ILocalSymbol> LocalsOf(IOperation target)
            {
                switch (target)
                {
                    case IVariableDeclaratorOperation d:
                        yield return d.Symbol;
                        break;
                    case ILocalReferenceOperation l:
                        yield return l.Local;
                        break;
                    case IDeclarationExpressionOperation de:
                        foreach (var s in LocalsOf(de.Expression))
                            yield return s;
                        break;
                    case ITupleOperation t:
                        foreach (var e in t.Elements)
                            foreach (var s in LocalsOf(e))
                                yield return s;
                        break;
                }
            }
        }
    }
}
