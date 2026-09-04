using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     <b>NDW012 — NDArray leak detection.</b> Reports a WARNING for an <c>NDArray</c> (or any
    ///     carrier of one — <c>NDArray[]</c>, a <c>ValueTuple</c>/<c>Tuple</c> of NDArrays, an
    ///     <c>INDArrayCarrier</c> result struct) that a method CREATES but never returns, assigns to
    ///     an <c>out</c>/<c>ref</c> parameter, stores, disposes, or yields to an <see cref="NDScope"/>
    ///     — so its pooled buffer is left to the finalizer instead of being reclaimed promptly (the
    ///     perf gap <c>[NDScoped]</c> exists to close; see <c>DISPOSAL-GUIDELINES.md</c>).
    ///     <para>
    ///     A method that is already <c>[NDScoped]</c> / <c>[NDScopedAsync]</c>, that opens an
    ///     <see cref="NDScope"/> by hand, or that is marked <c>[NDScopedCovered]</c> is EXEMPT: the
    ///     weaver (or the hand-written scope) reclaims every transient the first two construct, and
    ///     <c>[NDScopedCovered]</c> is the author's assertion that this helper always runs under a
    ///     caller's ambient scope (the per-method analysis has no call-graph to see that on its own).
    ///     Those are the fixes the message suggests, alongside a <c>using</c> / <c>.Dispose()</c> /
    ///     <c>scope.Returns(...)</c>.
    ///     </para>
    ///     <para>
    ///     The analysis is deliberately comprehensive ("all scenarios"): it catches a result dropped on
    ///     the floor (<c>np.add(a, b);</c>), a local that is only ever read (<c>var t = a + b; return
    ///     t.sum();</c>), AND an owned temporary handed into another NumSharp op that does not take
    ///     ownership of it (<c>return np.sum(a * b);</c> flags the <c>a * b</c>). Handing a value to a
    ///     non-NumSharp API, storing it, or returning it all count as legitimate egress and are never
    ///     flagged — the same conservative boundary that keeps false positives out.
    ///     </para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NDArrayLeakAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Build";
        private const string HelpLink = "https://scisharp.github.io/NumSharp/docs/numsharp-build-compiler.html#ndw012";

        internal static readonly DiagnosticDescriptor Nw012 = new DiagnosticDescriptor(
            "NDW012",
            "NDArray is created but never disposed, returned, or scoped (leaked to the finalizer)",
            "This {1} is never returned, assigned to an out/ref parameter, stored, disposed, or " +
            "reclaimed by an NDScope, so its pooled buffer is left to the finalizer instead of being " +
            "returned promptly. Dispose it (a 'using' declaration or '.Dispose()'), yield it through " +
            "'scope.Returns(...)', or mark the enclosing method '{0}' so the weaver reclaims it.",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Every NDArray owns an unmanaged buffer from NumSharp's pool. A transient that " +
                         "is dropped without being disposed or yielded to a scope is reclaimed only when " +
                         "its finalizer eventually runs — the exact cost the [NDScoped] weaver removes. " +
                         "Marking the method [NDScoped]/[NDScopedAsync], or disposing the value, closes the gap.",
            helpLinkUri: HelpLink);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw012);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterCompilationStartAction(start =>
            {
                var known = KnownTypes.Resolve(start.Compilation);
                if (known == null || known.NDArray == null)
                    return; // NumSharp not referenced — nothing to analyze

                // The ownership model makes NDArray-owning DISPOSABLES (a consumer's holder class,
                // NumSharp's own np.nditer iterator) owned values too: ownership is contagious.
                var model = new OwnershipModel(known, start.Compilation);
                start.RegisterOperationBlockAction(c => new BlockScan(known, model, c).Run());
            });
        }

        private enum Disposition
        {
            Leaks,     // read/dropped/fed to a non-consuming NumSharp op — never reclaimed
            Escapes,   // returned / out / ref / stored / handed to a consuming API
            Reclaimed  // using / Dispose / scope.Returns / NDScope.Attach
        }

        /// <summary>One method/accessor/constructor/local-function body's worth of leak analysis.</summary>
        private sealed class BlockScan
        {
            private readonly KnownTypes _k;
            private readonly OwnershipModel _model;
            private readonly OperationBlockAnalysisContext _ctx;

            // local → its non-write value uses; the owning locals; the ones a 'using' already reclaims;
            // and the expressions that FEED a local (skipped in the expression pass — reported via the local).
            private readonly Dictionary<ILocalSymbol, List<IOperation>> _localUses =
                new Dictionary<ILocalSymbol, List<IOperation>>(SymbolEqualityComparer.Default);
            private readonly HashSet<ILocalSymbol> _owningLocals = new HashSet<ILocalSymbol>(SymbolEqualityComparer.Default);
            private readonly HashSet<ILocalSymbol> _aliasedLocals = new HashSet<ILocalSymbol>(SymbolEqualityComparer.Default);
            private readonly HashSet<ILocalSymbol> _usingLocals = new HashSet<ILocalSymbol>(SymbolEqualityComparer.Default);
            private readonly HashSet<IOperation> _localSourceExprs = new HashSet<IOperation>();

            public BlockScan(KnownTypes k, OwnershipModel model, OperationBlockAnalysisContext ctx)
            {
                _k = k;
                _model = model;
                _ctx = ctx;
            }

            public void Run()
            {
                if (!(_ctx.OwningSymbol is IMethodSymbol method) || method.IsImplicitlyDeclared)
                    return;

                // Exempt: a method the author has already marked scope-covered — a
                // [NDScoped]/[NDScopedAsync] boundary (the weaver reclaims every transient it
                // constructs), or an [NDScopedCovered] (an analyzer-only assertion that this helper
                // always runs under a caller's ambient scope, so that scope reclaims its transients).
                if (IsScopeExemptByAttribute(method))
                    return;

                var blocks = _ctx.OperationBlocks;

                // Exempt: a body that opens an NDScope by hand (the same signal the weaver's idempotence
                // check uses). Its own scope owns reclamation, so flagging its temps would be noise.
                foreach (var block in blocks)
                    foreach (var op in block.DescendantsAndSelf())
                        if (op is IInvocationOperation open && IsNDScopeMethod(open.TargetMethod, "Open", "OpenOrResume"))
                            return;

                CollectLocals(blocks);

                // 1) owning locals whose value never escapes / is never reclaimed.
                foreach (var local in _owningLocals)
                {
                    if (_usingLocals.Contains(local))
                        continue;
                    if (DispositionOfLocal(local) == Disposition.Leaks)
                        Report(local.Locations.Length > 0 ? local.Locations[0] : Location.None, EnclosingSuggestion(method), DescribeOwned(local.Type));
                }

                // 2) owning expressions whose value is not captured by a local and never escapes.
                foreach (var block in blocks)
                {
                    foreach (var op in block.DescendantsAndSelf())
                    {
                        if (!ProducesOwnedNDArray(op) || _localSourceExprs.Contains(op))
                            continue;
                        if (DispositionOf(op) != Disposition.Leaks)
                            continue;
                        // Report only the OUTERMOST leaking expression of a chain — a nested owned
                        // sub-expression whose owning ancestor also leaks is subsumed by that ancestor.
                        if (HasLeakingOwnedAncestor(op))
                            continue;
                        Report(op.Syntax.GetLocation(), EnclosingSuggestion(method), DescribeOwned(op.Type));
                    }
                }
            }

            private void Report(Location loc, string suggestion, string kind)
                => _ctx.ReportDiagnostic(Diagnostic.Create(Nw012, loc, suggestion, kind));

            /// <summary>How the message names the leaked value: a plain NDArray (carriers included), or an NDArray-owning disposable instance.</summary>
            private string DescribeOwned(ITypeSymbol t)
            {
                if (t is IArrayTypeSymbol arr && _model.IsNDOwningDisposable(arr.ElementType))
                    return "'" + arr.ElementType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) +
                           "' array (disposables that store NDArrays)";
                if (_model.IsNDOwningDisposable(t))
                    return "'" + t.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) +
                           "' instance (a disposable that stores NDArrays)";
                return "NDArray";
            }

            // -------------------------------------------------------------- local collection

            private void CollectLocals(ImmutableArray<IOperation> blocks)
            {
                foreach (var block in blocks)
                {
                    foreach (var op in block.DescendantsAndSelf())
                    {
                        switch (op)
                        {
                            case ILocalReferenceOperation lref when !IsWriteTarget(lref) && !IsDeclarationReference(lref):
                                Uses(lref.Local).Add(lref);
                                break;

                            case IVariableDeclaratorOperation decl:
                                RegisterUsingIfAny(decl);
                                if (decl.GetVariableInitializer()?.Value is IOperation init)
                                    RegisterSource(decl.Symbol, init);
                                break;

                            case ISimpleAssignmentOperation asg when asg.Target is ILocalReferenceOperation t:
                                RegisterSource(t.Local, asg.Value);
                                break;

                            case IDeconstructionAssignmentOperation deconstruct:
                                foreach (var target in DeconstructionLocals(deconstruct.Target))
                                    RegisterSource(target, deconstruct.Value);
                                break;
                        }
                    }
                }

                // A local that is EVER assigned a plain input (a parameter / field / another local — an
                // alias, not a value this method owns) is dropped from the owning set: disposing it could
                // free something the caller owns (R2). One owning assignment with no alias assignment ⇒ owned.
                _owningLocals.ExceptWith(_aliasedLocals);
            }

            private void RegisterSource(ILocalSymbol local, IOperation source)
            {
                if (local == null || !IsOwnedNDArrayType(local.Type))
                    return;

                var value = Unwrap(source);
                _localSourceExprs.Add(source);
                _localSourceExprs.Add(value);

                if (ProducesOwnedNDArray(value))
                    _owningLocals.Add(local);
                else if (value is ILocalReferenceOperation || value is IParameterReferenceOperation ||
                         value is IFieldReferenceOperation || value is IPropertyReferenceOperation)
                    _aliasedLocals.Add(local); // aliases an input — never our value to reclaim
            }

            private void RegisterUsingIfAny(IVariableDeclaratorOperation decl)
            {
                // `using var t = ...;` / `using (var t = ...)` already reclaim the local deterministically.
                for (IOperation p = decl.Parent; p != null; p = p.Parent)
                {
                    if (p is IUsingDeclarationOperation || p is IUsingOperation)
                    {
                        _usingLocals.Add(decl.Symbol);
                        return;
                    }
                    if (p is IBlockOperation || p is IMethodBodyBaseOperation)
                        return; // a declarator's using wrapper is immediately above it, never past a block
                }
            }

            private List<IOperation> Uses(ILocalSymbol local)
            {
                if (!_localUses.TryGetValue(local, out var list))
                    _localUses[local] = list = new List<IOperation>();
                return list;
            }

            // -------------------------------------------------------------- disposition

            private Disposition DispositionOfLocal(ILocalSymbol local)
            {
                if (_usingLocals.Contains(local))
                    return Disposition.Reclaimed;
                if (!_localUses.TryGetValue(local, out var uses))
                    return Disposition.Leaks; // declared, owned, never read again — a dead transient

                foreach (var use in uses)
                {
                    var d = DispositionOf(use, insideLocal: true);
                    if (d != Disposition.Leaks)
                        return d; // any single escaping / reclaiming use saves the local
                }
                return Disposition.Leaks;
            }

            /// <summary>Classifies where a single value expression's result goes.</summary>
            private Disposition DispositionOf(IOperation value, bool insideLocal = false)
            {
                var op = value;
                for (int guard = 0; guard < 64 && op != null; guard++)
                {
                    // yield return hands the element to the consumer — an escape the operation tree
                    // does not always model as IReturnOperation.
                    if (op.Syntax?.Parent is YieldStatementSyntax)
                        return Disposition.Escapes;

                    var parent = op.Parent;
                    switch (parent)
                    {
                        case null:
                            return Disposition.Leaks;

                        // transparent wrappers — follow the value upward.
                        case IConversionOperation _:
                        case IParenthesizedOperation _:
                        case ITupleOperation _:
                            op = parent;
                            continue;

                        case IExpressionStatementOperation _:
                            return Disposition.Leaks; // result discarded

                        case IReturnOperation _:
                            return Disposition.Escapes;

                        case IAwaitOperation _:
                            return Disposition.Escapes;

                        case IUsingOperation _:
                        case IUsingDeclarationOperation _:
                            return Disposition.Reclaimed;

                        case IVariableInitializerOperation initOp:
                            return DispositionOfAssignedLocal(DeclaredLocalOf(initOp), insideLocal);

                        case ISimpleAssignmentOperation asg when ReferenceEquals(asg.Value, op):
                            if (asg.Target is ILocalReferenceOperation lt)
                                return DispositionOfAssignedLocal(lt.Local, insideLocal);
                            // `_ = a + b;` — an explicit discard drops the owned buffer on the floor,
                            // exactly like a bare expression statement. It is NOT a store (there is no
                            // target that keeps the value), so it must read as a leak, not an escape.
                            if (asg.Target is IDiscardOperation)
                                return Disposition.Leaks;
                            return Disposition.Escapes; // stored in a field / property / element
                        case IDeconstructionAssignmentOperation _:
                            return Disposition.Escapes; // the value feeds a deconstruction into locals/targets

                        // operand of an NDArray operator (a + b, -a): the operator does not take ownership.
                        case IBinaryOperation _:
                        case IUnaryOperation _:
                            return Disposition.Leaks;

                        // receiver of `x.Member` / `x.Method(...)` / `x[i]`: `.Dispose()` reclaims;
                        // otherwise FOLLOW the call/member result upward, because so much of NumSharp is
                        // fluent reinterpret/view chains — `x.MakeGeneric<T>()`, `x.reshape(...)`, `x.T`,
                        // `x[...]` — whose result SHARES x's buffer, so if that result escapes x is not a
                        // leak. (This deliberately concedes `var t = a + b; return t.sum();`, whose sum()
                        // returns a FRESH buffer: telling a view apart from a fresh result syntactically is
                        // not possible, and following the receiver is the choice that avoids flooding the
                        // far more common view chains with false positives.)
                        case IInvocationOperation recv when ReferenceEquals(recv.Instance, op):
                            if (IsDisposeCall(recv.TargetMethod))
                                return Disposition.Reclaimed;
                            op = parent;
                            continue;
                        case IMemberReferenceOperation mref when ReferenceEquals(mref.Instance, op):
                            op = parent;
                            continue;
                        case IArrayElementReferenceOperation arr when ReferenceEquals(arr.ArrayReference, op):
                            op = parent;
                            continue;

                        case IArgumentOperation argOp:
                            return DispositionOfArgument(argOp);

                        // the collection of a foreach: C# disposes the ENUMERATOR it obtains, never the
                        // enumerable itself — so a produced NDArray (`foreach (var row in a + b)`) or an
                        // NDArray-owning disposable (`foreach (var x in np.nditer(a))`) walked this way is
                        // never reclaimed. A produced NDArray[] / tuple / carrier is a managed container
                        // whose ELEMENTS the loop body owns the fate of — conservatively consumed.
                        case IForEachLoopOperation loop when ReferenceEquals(loop.Collection, op):
                            return IsNeverDisposedByForeach(Unwrap(op).Type) ? Disposition.Leaks : Disposition.Escapes;

                        default:
                            // Unknown context: treat as an escape rather than risk a false positive.
                            return Disposition.Escapes;
                    }
                }
                return Disposition.Escapes;
            }

            private Disposition DispositionOfAssignedLocal(ILocalSymbol local, bool insideLocal)
            {
                if (local == null)
                    return Disposition.Escapes;
                // Aliased into an input-tracking local, or one that itself escapes — treat as escape.
                // Avoid infinite recursion when a local's own use assigns to itself.
                if (insideLocal || _aliasedLocals.Contains(local) || !_owningLocals.Contains(local))
                    return Disposition.Escapes;
                return DispositionOfLocal(local);
            }

            private Disposition DispositionOfArgument(IArgumentOperation argOp)
            {
                // ref / out argument is a write-back egress the caller reads.
                if (argOp.Parameter != null && argOp.Parameter.RefKind != RefKind.None)
                    return Disposition.Escapes;

                var call = argOp.Parent;
                if (call is IInvocationOperation inv)
                {
                    if (IsNDScopeMethod(inv.TargetMethod, "Returns", "Attach", "Detach"))
                        return Disposition.Reclaimed;
                    // A NumSharp op that RETURNS an NDArray never disposes its NDArray inputs (rule R2),
                    // so a temp handed to it is still unreclaimed — the aggressive "all scenarios" case.
                    if (IsNumSharpNonConsumingCall(inv))
                        return Disposition.Leaks;
                    return Disposition.Escapes; // any other API is assumed to consume / observe it
                }
                // An object creation is assumed to take ownership (e.g. wrapping a slice).
                return Disposition.Escapes;
            }

            /// <summary>True if some owned-NDArray ancestor of <paramref name="op"/> also leaks (so <paramref name="op"/> is subsumed).</summary>
            private bool HasLeakingOwnedAncestor(IOperation op)
            {
                for (var a = op.Parent; a != null; a = a.Parent)
                    if (ProducesOwnedNDArray(a))
                        return DispositionOf(a) == Disposition.Leaks;
                return false;
            }

            // -------------------------------------------------------------- predicates

            private bool IsScopeExemptByAttribute(IMethodSymbol method)
            {
                ISymbol carrier = method.AssociatedSymbol ?? (ISymbol)method; // property accessor → property
                foreach (var s in new[] { method, carrier })
                    foreach (var a in s.GetAttributes())
                    {
                        var cls = a.AttributeClass;
                        if (SymbolEqualityComparer.Default.Equals(cls, _k.SyncAttr) ||
                            SymbolEqualityComparer.Default.Equals(cls, _k.AsyncAttr) ||
                            SymbolEqualityComparer.Default.Equals(cls, _k.CoveredAttr))
                            return true;
                    }
                return false;
            }

            /// <summary>
            ///     An owned NDArray carrier value — the kinds whose buffer this method must reclaim:
            ///     NDArray / NDArray[] / a tuple with one / an INDArrayCarrier struct, and — ownership
            ///     being contagious — an NDArray-owning DISPOSABLE (a consumer's holder class, NumSharp's
            ///     np.nditer iterator) or a rank-1 array of them.
            /// </summary>
            private bool IsOwnedNDArrayType(ITypeSymbol t)
            {
                if (t == null)
                    return false;
                if (TypeHelpers.IsNDArrayCarrying(t, _k))          // NDArray / NDArray[]
                    return true;
                if (t.IsTupleType && ((INamedTypeSymbol)t).TupleElements.Any(e => IsOwnedNDArrayType(e.Type)))
                    return true;
                if (ImplementsCarrier(t))                          // INDArrayCarrier result struct
                    return true;
                if (_model.IsNDOwningDisposable(t))                // a disposable that stores NDArrays
                    return true;
                if (t is IArrayTypeSymbol arr && arr.Rank == 1 && _model.IsNDOwningDisposable(arr.ElementType))
                    return true;
                return false;
            }

            /// <summary>
            ///     Whether walking a produced value with <c>foreach</c> strands it: an NDArray-like
            ///     (enumerating a temp never disposes it) or an NDArray-owning disposable whose
            ///     enumerator is not the object itself. Arrays, tuples and carriers are not.
            /// </summary>
            private bool IsNeverDisposedByForeach(ITypeSymbol t)
                => t != null && (TypeHelpers.IsNDArrayLike(t, _k) || _model.IsNDOwningDisposable(t));

            private bool ImplementsCarrier(ITypeSymbol t)
            {
                if (_k.INDArrayCarrier == null || !t.IsValueType)
                    return false;
                foreach (var i in t.AllInterfaces)
                    if (SymbolEqualityComparer.Default.Equals(i, _k.INDArrayCarrier))
                        return true;
                return false;
            }

            /// <summary>An expression that PRODUCES an owned NDArray value (a construction, call, or operator — not a view-y property/indexer read or a plain reference).</summary>
            private bool ProducesOwnedNDArray(IOperation op)
            {
                if (op == null || !IsOwnedNDArrayType(op.Type))
                    return false;
                switch (op)
                {
                    case IObjectCreationOperation _:
                    case IBinaryOperation _:
                    case IUnaryOperation _:
                        return true;
                    case IInvocationOperation inv:
                        // NDScope.Returns/Attach return an NDArray but are reclamation plumbing, not a
                        // fresh temp to flag.
                        return !IsNDScopeMethod(inv.TargetMethod, "Returns", "Attach", "Detach");

                    // A tuple literal / array creation OWNS whatever its elements own, and it holds ALL
                    // of them at once — so it is a producer iff AT LEAST ONE element is itself a
                    // producer. A tuple/array of plain inputs ((a, b), new[]{ a, b }) owns nothing, and
                    // flagging it would demand reclaiming the caller's values (R2).
                    case ITupleOperation tup:
                        foreach (var e in tup.Elements)
                            if (ProducesOwnedNDArray(Unwrap(e)))
                                return true;
                        return false;
                    case IArrayCreationOperation arr when arr.Initializer != null:
                        foreach (var e in arr.Initializer.ElementValues)
                            if (ProducesOwnedNDArray(Unwrap(e)))
                                return true;
                        return false;
                    // The C# 12 collection-expression spelling of the same thing: `NDArray[] g = [a + b];`.
                    // Matched by SYNTAX because this analyzer compiles against Roslyn 4.8 (the .NET 8 SDK
                    // floor), which has the language feature but not ICollectionExpressionOperation — a 4.8
                    // host surfaces the expression as an unrecognized operation, newer hosts as the typed
                    // one; both carry the elements as child operations. (A spread element is not a
                    // producer, so a spread-only expression stays silent.)
                    case IOperation ce when ce.Syntax is CollectionExpressionSyntax:
                        foreach (var e in ce.ChildOperations)
                            if (ProducesOwnedNDArray(Unwrap(e)))
                                return true;
                        return false;

                    // A conditional value (ternary / switch expression / coalesce) holds only ONE of its
                    // branches, so it is a producer only when EVERY branch produces — the same
                    // conservatism as the if/else local rule, where a single aliasing branch drops the
                    // local from the owning set: if any branch may be the caller's value, demanding the
                    // result be reclaimed is not safe.
                    case IConditionalOperation cond when cond.WhenFalse != null:
                        return BranchProduces(cond.WhenTrue) && BranchProduces(cond.WhenFalse);
                    case ISwitchExpressionOperation sw:
                        if (sw.Arms.Length == 0)
                            return false;
                        foreach (var arm in sw.Arms)
                            if (!BranchProduces(arm.Value))
                                return false;
                        return true;
                    case ICoalesceOperation coal:
                        return BranchProduces(coal.Value) && BranchProduces(coal.WhenNull);

                    default:
                        return false;
                }
            }

            /// <summary>
            ///     A branch of a conditional value, for the every-branch-produces rule. A <c>throw</c>
            ///     branch never completes, so it can never be the held value — it is vacuously true
            ///     (`p ? a + b : throw …` owns its result on every path that yields one).
            /// </summary>
            private bool BranchProduces(IOperation branch)
            {
                var e = Unwrap(branch);
                return e is IThrowOperation || ProducesOwnedNDArray(e);
            }

            private bool IsNumSharpNonConsumingCall(IInvocationOperation inv)
            {
                var m = inv.TargetMethod;
                return m != null
                       && _k.NumSharpAssembly != null
                       && SymbolEqualityComparer.Default.Equals(m.ContainingAssembly, _k.NumSharpAssembly)
                       && IsOwnedNDArrayType(m.ReturnType);
            }

            private bool IsNDScopeMethod(IMethodSymbol m, params string[] names)
            {
                if (m == null || _k.NDScope == null ||
                    !SymbolEqualityComparer.Default.Equals(m.ContainingType, _k.NDScope))
                    return false;
                foreach (var n in names)
                    if (m.Name == n)
                        return true;
                return false;
            }

            /// <summary>
            ///     Any parameterless <c>Dispose()</c> / <c>DisposeAsync()</c> / <c>Close()</c> (and NumSharp's
            ///     NumPy-spelled <c>close()</c> on <c>np.nditer</c>). The receiver walk only ever arrives
            ///     here FROM an owned value, so the static type the call is made through is irrelevant —
            ///     <c>((IDisposable)t).Dispose()</c> disposes the same buffer <c>t.Dispose()</c> does
            ///     (constraining to an owned containing type made exactly that spelling a false positive).
            /// </summary>
            private static bool IsDisposeCall(IMethodSymbol m)
                => m != null && m.Parameters.Length == 0 &&
                   (m.Name == "Dispose" || m.Name == "DisposeAsync" || m.Name == "Close" || m.Name == "close");

            private string EnclosingSuggestion(IMethodSymbol method)
            {
                bool async = method.IsAsync
                             || TypeHelpers.IsTaskLike(method.ReturnType, _k, out _)
                             || (method.ReturnType is INamedTypeSymbol rn && rn.OriginalDefinition != null &&
                                 rn.OriginalDefinition.ToDisplayString() == "System.Collections.Generic.IAsyncEnumerable<T>");
                return async ? "[NDScopedAsync]" : "[NDScoped]";
            }

            // -------------------------------------------------------------- small helpers

            private static IOperation Unwrap(IOperation op)
            {
                while (op is IConversionOperation c && c.Operand != null)
                    op = c.Operand;
                while (op is IParenthesizedOperation p && p.Operand != null)
                    op = p.Operand;
                return op;
            }

            private static bool IsWriteTarget(ILocalReferenceOperation lref)
            {
                switch (lref.Parent)
                {
                    case ISimpleAssignmentOperation asg:
                        return ReferenceEquals(asg.Target, lref);
                    case ICompoundAssignmentOperation comp:
                        return ReferenceEquals(comp.Target, lref);
                    case IIncrementOrDecrementOperation inc:
                        return ReferenceEquals(inc.Target, lref);
                    default:
                        return false;
                }
            }

            /// <summary>
            ///     True for a local reference that is a WRITE-SHAPED mention of the local, not a read — a
            ///     deconstruction target, whether declaring (`var (q, r) = …` → LocalRef under Tuple under
            ///     DeclarationExpression) or into EXISTING locals (`(q, r) = …` → LocalRef under the Tuple
            ///     that IS the deconstruction-assignment's Target), or an `out var x` declaration. Counting
            ///     these as uses would mask a leak: the unused side of a deconstruction would look "used"
            ///     by its own assignment.
            /// </summary>
            private static bool IsDeclarationReference(ILocalReferenceOperation lref)
            {
                IOperation child = lref;
                var p = lref.Parent;
                while (p is ITupleOperation)
                {
                    child = p;
                    p = p.Parent;
                }
                if (p is IDeclarationExpressionOperation)
                    return true;
                return p is IDeconstructionAssignmentOperation d && ReferenceEquals(d.Target, child);
            }

            private static ILocalSymbol DeclaredLocalOf(IVariableInitializerOperation initOp)
                => (initOp.Parent as IVariableDeclaratorOperation)?.Symbol;

            private static IEnumerable<ILocalSymbol> DeconstructionLocals(IOperation target)
            {
                switch (target)
                {
                    case ILocalReferenceOperation l:
                        yield return l.Local;
                        break;
                    case IDeclarationExpressionOperation d:
                        foreach (var s in DeconstructionLocals(d.Expression))
                            yield return s;
                        break;
                    case ITupleOperation t:
                        foreach (var e in t.Elements)
                            foreach (var s in DeconstructionLocals(e))
                                yield return s;
                        break;
                }
            }
        }
    }

    internal static class VariableInitializerExtensions
    {
        /// <summary>The initializer of a declarator across Roslyn versions (the property moved from the declarator to a child in newer ones).</summary>
        public static IVariableInitializerOperation GetVariableInitializer(this IVariableDeclaratorOperation decl)
            => decl?.Initializer;
    }
}
