using System;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     Compile-time gate for <c>[NDScoped]</c> / <c>[NDScopedAsync]</c>. Reports a build ERROR
    ///     when either attribute is placed on a target the IL weaver (<c>tools/NumSharp.Build</c>)
    ///     could not weave — the SAME NDW diagnostics the weaver produces post-compile, surfaced in
    ///     the editor and at CoreCompile so the mistake is caught before the weave runs (and, in the
    ///     package, the error preempts the weaver: a failed CoreCompile never reaches the weave step).
    ///     <para>
    ///     It covers every SOURCE-detectable rejection: the wrong attribute (NDW009 = async/Task under
    ///     <c>[NDScoped]</c>, NDW010 = a plain sync method / synchronous iterator under
    ///     <c>[NDScopedAsync]</c>, NDW011 = both attributes), a hidden <c>ref</c>/<c>in</c> egress over
    ///     any NDArray-carrying shape (NDW002), an unsupported carrier return (NDW003), a body-less
    ///     <c>extern</c> (NDW005), a setter-only property (NDW006), and an <c>out</c> parameter whose
    ///     NDArray-carrying shape the out-escape cannot yield (NDW015). The IL-only rejections stay
    ///     with the weaver: NDW004 (an unrecognized state-machine shape), NDW007 (a tail-call prefix),
    ///     NDW008 (the referenced NumSharp predates the async seam), NDW014 (a bad
    ///     <c>[NDScopedExit]</c> parameter).
    ///     </para>
    ///     <para>
    ///     A target may carry the attribute itself or INHERIT it (<see cref="ScopeInheritance"/>): an
    ///     override or implementation of a scoped virtual/abstract/interface member with no scope-family
    ///     attribute of its own is gated under its declaration's attribute — and an abstract/interface
    ///     declaration carrying the attribute is that contract, never NDW005. The one inherited case
    ///     the MSBuild weaver-missing scan (NDW013 in NumSharp's <c>build/NumSharp.targets</c>, a text
    ///     probe for the attribute names) cannot see — a declaration in ANOTHER assembly, so the
    ///     override carries no attribute text — is reported here as NDW013, a WARNING at the override,
    ///     when the build says the weaver is not active (<c>build_property.NumSharpBuildActive</c>,
    ///     exposed by both targets files; absent = unknown = silent).
    ///     </para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NDScopedTargetAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Build";
        private const string HelpLink = "https://scisharp.github.io/NumSharp/docs/numsharp-build-compiler.html";
        private const string SyncName = "[NDScoped]";
        private const string AsyncName = "[NDScopedAsync]";

        private static DiagnosticDescriptor Error(string id, string title, string message) =>
            new DiagnosticDescriptor(id, title, message, Category, DiagnosticSeverity.Error,
                isEnabledByDefault: true, description: null, helpLinkUri: HelpLink);

        internal static readonly DiagnosticDescriptor Nw002 = Error("NDW002",
            "Scoped method has a hidden 'ref' NDArray egress",
            "{0} method '{1}' has a 'ref {2}' parameter '{3}' — a hidden egress the weaver cannot see; scope this method by hand");

        internal static readonly DiagnosticDescriptor Nw003 = Error("NDW003",
            "Scoped method returns an unsupported carrier",
            "{0} method '{1}' returns '{2}' — an unsupported carrier the weaver cannot see every NDArray through (a bespoke reference type, a collection, or a result struct that does NOT implement INDArrayCarrier). Return NDArray, NDArray[], a tuple of NDArrays, an INDArrayCarrier struct, a bare IArraySlice/UnmanagedStorage, a scalar, or a Task/ValueTask of one of these — add INDArrayCarrier to the struct, or scope this method by hand");

        internal static readonly DiagnosticDescriptor Nw005 = Error("NDW005",
            "Scoped method has no body",
            "{0} method '{1}' has no body (extern) — remove the attribute (an abstract or interface declaration may carry it: its overrides and implementations inherit the scope)");

        internal static readonly DiagnosticDescriptor Nw006 = Error("NDW006",
            "Scoped attribute on a setter-only property",
            "{0} on property '{1}' requires a getter (put the attribute on the getter accessor for setter-only properties)");

        internal static readonly DiagnosticDescriptor Nw009 = Error("NDW009",
            "async / Task-returning method marked [NDScoped]",
            "{0} method '{1}' is {2} — mark it [NDScopedAsync] instead ([NDScoped] weaves synchronous methods and synchronous iterators; [NDScopedAsync] weaves async methods, async iterators and non-async Task/ValueTask returns)");

        internal static readonly DiagnosticDescriptor Nw010 = Error("NDW010",
            "synchronous method / iterator marked [NDScopedAsync]",
            "{0} method '{1}' is {2} — mark it [NDScoped] instead ([NDScopedAsync] weaves async methods, async iterators and non-async Task/ValueTask returns)");

        internal static readonly DiagnosticDescriptor Nw011 = Error("NDW011",
            "method carries both [NDScoped] and [NDScopedAsync]",
            "method '{0}' carries BOTH [NDScoped] and [NDScopedAsync] — a method has exactly one scoping model; keep only the attribute that matches it ([NDScoped] for synchronous methods and synchronous iterators, [NDScopedAsync] for async methods, async iterators and non-async Task/ValueTask returns)");

        internal static readonly DiagnosticDescriptor Nw015 = Error("NDW015",
            "Scoped method has an unsupported 'out' NDArray-carrying parameter",
            "{0} method '{1}' has an 'out {2}' parameter '{3}' — an NDArray-carrying shape the scope's out-escape cannot yield (supported: NDArray, NDArray[], a ValueTuple/Tuple of supported shapes, an INDArrayCarrier struct, a bare IArraySlice/UnmanagedStorage); scope this method by hand");

        /// <summary>
        ///     The inherited-target half of NDW013 (the MSBuild half, in NumSharp's <c>build/NumSharp.targets</c>,
        ///     covers attributes typed in the project itself). A WARNING, like its MSBuild twin.
        /// </summary>
        internal static readonly DiagnosticDescriptor Nw013 = new DiagnosticDescriptor("NDW013",
            "Inherited [NDScoped] target, but the weaver is not installed",
            "method '{0}' inherits {1} from '{2}' (declared in another assembly) but the NumSharp.Build package is not installed (or weaving is disabled), so the attribute is INERT here — no NDScope is injected and the method's NDArray temporaries are left to the finalizer. Install the weaver with 'dotnet add package NumSharp.Build', mark the override [NDScopedCovered] to opt out of the inherited scope, or dispose the temporaries by hand",
            Category, DiagnosticSeverity.Warning, isEnabledByDefault: true, description: null, helpLinkUri: HelpLink + "#ndw013");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw002, Nw003, Nw005, Nw006, Nw009, Nw010, Nw011, Nw013, Nw015);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterCompilationStartAction(start =>
            {
                var known = KnownTypes.Resolve(start.Compilation);
                if (known == null)
                    return; // NumSharp not referenced — nothing to analyze

                bool weaverInactive = WeaverKnownInactive(start.Options.AnalyzerConfigOptionsProvider.GlobalOptions);

                start.RegisterSymbolAction(c => AnalyzeMethod((IMethodSymbol)c.Symbol, known, weaverInactive, c.ReportDiagnostic), SymbolKind.Method);
                start.RegisterSymbolAction(c => AnalyzeProperty((IPropertySymbol)c.Symbol, known, c.ReportDiagnostic), SymbolKind.Property);
            });
        }

        /// <summary>
        ///     True only when the build DECLARED the weaver inactive: <c>NumSharpBuildActive</c> is present
        ///     (NumSharp's targets default it to 'false'; NumSharp.Build's set it to 'true' when weaving)
        ///     and not 'true', and the documented opt-out <c>NumSharpDisableWeaverMissingWarning</c> is not
        ///     set. An ABSENT property (a source-mode build importing neither targets file, NumSharp's own
        ///     compile) is unknown, and unknown stays silent.
        /// </summary>
        private static bool WeaverKnownInactive(AnalyzerConfigOptions options)
        {
            if (!options.TryGetValue("build_property.NumSharpBuildActive", out var active))
                return false;
            if (string.Equals(active?.Trim(), "true", StringComparison.OrdinalIgnoreCase))
                return false;
            if (options.TryGetValue("build_property.NumSharpDisableWeaverMissingWarning", out var disabled) &&
                string.Equals(disabled?.Trim(), "true", StringComparison.OrdinalIgnoreCase))
                return false;
            return true;
        }

        private static void AnalyzeMethod(IMethodSymbol m, KnownTypes k, bool weaverInactive, Action<Diagnostic> report)
        {
            var own = ScopeInheritance.KindsOf(m, k);
            bool sync = (own & ScopeKind.Sync) != 0;
            bool async = (own & ScopeKind.Async) != 0;
            string label = null;

            if (!sync && !async)
            {
                // No attribute on the method itself. [NDScopedCovered] is an opt-out (of an inherited
                // scope too); a getter whose PROPERTY carries the attribute is AnalyzeProperty's target;
                // otherwise an override/implementation may INHERIT its declaration's attribute — the
                // overwhelmingly common plain method bails right here, before any type work.
                if (own != ScopeKind.None)
                    return;
                if (m.MethodKind == MethodKind.PropertyGet && m.AssociatedSymbol != null &&
                    ScopeInheritance.KindsOf(m.AssociatedSymbol, k) != ScopeKind.None)
                    return;

                var inherited = ScopeInheritance.Inherited(m, k);
                if (!(inherited is ScopeDeclaration decl))
                    return;
                sync = (decl.Kinds & ScopeKind.Sync) != 0;
                async = (decl.Kinds & ScopeKind.Async) != 0;
                if (!sync && !async)
                    return; // an inherited [NDScopedCovered]: the override is covered, nothing to gate
                if (sync && async)
                    return; // the declaration's own NDW011 reports it; its inheritors add nothing

                label = (sync ? SyncName : AsyncName) + " (inherited from " + Display(decl.Source) + ")";

                // NDW013 for the one inherited shape the MSBuild text probe is blind to: the declaration
                // lives in another assembly, so this override carries no attribute text at all.
                if (weaverInactive && !m.IsAbstract &&
                    !SymbolEqualityComparer.Default.Equals(decl.Source.ContainingAssembly, m.ContainingAssembly))
                    report(Diagnostic.Create(Nw013, Loc(m), Display(m), sync ? SyncName : AsyncName, Display(decl.Source)));
            }

            bool hasBody = !m.IsAbstract && !m.IsExtern
                           && !(m.IsPartialDefinition && m.PartialImplementationPart == null);
            AnalyzeTarget(m, sync, async, m.ReturnType, m.IsAsync, m.Parameters, hasBody, m.IsAbstract, k, report, label);
        }

        private static void AnalyzeProperty(IPropertySymbol p, KnownTypes k, Action<Diagnostic> report)
        {
            bool sync = HasAttr(p, k.SyncAttr);
            bool async = HasAttr(p, k.AsyncAttr);
            if (!sync && !async)
                return;

            // A property-level attribute resolves to the getter (mirrors the weaver's CollectTargets);
            // a setter-only property has nothing to weave.
            if (p.GetMethod == null)
            {
                var label = sync && async ? SyncName + "/" + AsyncName : sync ? SyncName : AsyncName;
                report(Diagnostic.Create(Nw006, Loc(p), label, Display(p)));
                return;
            }

            // The getter is synchronous (properties cannot be async) and never has ref parameters. An
            // abstract/interface property is the contract its overrides inherit.
            AnalyzeTarget(p, sync, async, p.Type, isAsync: false, ImmutableArray<IParameterSymbol>.Empty,
                hasBody: !p.IsAbstract, isAbstract: p.IsAbstract, k, report, label: null);
        }

        private static void AnalyzeTarget(ISymbol symbol, bool sync, bool async, ITypeSymbol returnType,
            bool isAsync, ImmutableArray<IParameterSymbol> parameters, bool hasBody, bool isAbstract, KnownTypes k,
            Action<Diagnostic> report, string label)
        {
            var loc = Loc(symbol);
            var name = Display(symbol);

            // NDW011 — a method has exactly one scoping model.
            if (sync && async)
            {
                report(Diagnostic.Create(Nw011, loc, name));
                return;
            }

            label = label ?? (async ? AsyncName : SyncName);

            bool taskLike = TypeHelpers.IsTaskLike(returnType, k, out _);

            // Wrong attribute — mirrors the SyncScopeWeaver / AsyncScopeWeaver routing exactly. A
            // synchronous iterator returns a bare enumerable (not async, not Task), so it is NOT caught
            // by NDW009 (valid under [NDScoped]) and IS caught by NDW010 (invalid under [NDScopedAsync]).
            if (sync && (isAsync || taskLike))
            {
                report(Diagnostic.Create(Nw009, loc, label, name,
                    isAsync ? "an async method" : "a Task/ValueTask-returning method"));
                return;
            }

            if (async && !isAsync && !taskLike)
            {
                report(Diagnostic.Create(Nw010, loc, label, name,
                    TypeHelpers.IsEnumerableInterface(returnType, k) ? "a synchronous iterator" : "a plain synchronous method"));
                return;
            }

            // An abstract / interface declaration is the CONTRACT its overrides and implementations
            // inherit — nothing to weave there, and nothing wrong. (Its signature is still gated above
            // and below: a wrong attribute or a hidden egress is wrong for every inheritor.)
            // NDW005 — a body-less NON-abstract target (extern, an unimplemented partial) can never be woven.
            if (!hasBody && !isAbstract)
            {
                report(Diagnostic.Create(Nw005, loc, label, name));
                return;
            }

            // NDW002 — a 'ref' (or 'in') parameter over ANYTHING that carries an NDArray (a bare
            // array, a tuple with an NDArray component, a carrier struct, a collection of NDArrays,
            // a T : NDArray) is a hidden egress. NDW015 — an 'out' whose NDArray-carrying shape the
            // return rewrite cannot yield (a collection, a task, a multi-dim array) would be handed
            // to the caller with its arrays already swept. async methods and iterators cannot have
            // by-ref parameters, so these only ever match a plain method.
            foreach (var pr in parameters)
            {
                if (pr.RefKind == RefKind.None || !TypeHelpers.CarriesNDArray(pr.Type, k))
                    continue;

                if (pr.RefKind != RefKind.Out)
                {
                    report(Diagnostic.Create(Nw002, loc, label, name, pr.Type.ToDisplayString(), pr.Name));
                    return;
                }

                if (!TypeHelpers.IsSupportedOutCarrier(pr.Type, k))
                {
                    report(Diagnostic.Create(Nw015, loc, label, name, pr.Type.ToDisplayString(), pr.Name));
                    return;
                }
            }

            // NDW003 — an unsupported return carrier. A bare enumerable interface is DEFERRED to the
            // weaver: a real iterator (yield) is fine and validates its element type at IL time, and a
            // non-iterator returning IEnumerable is rare — the analyzer must not flag the valid iterator.
            // A Task/ValueTask is unwrapped once to the result it carries; a NESTED task is refused
            // (the inner task's completion outlives the scope's deferral lifecycle — weaver parity).
            if (TypeHelpers.IsEnumerableInterface(returnType, k))
                return;

            var carrier = returnType;
            if (TypeHelpers.IsTaskLike(returnType, k, out var result))
            {
                if (result == null)
                    return; // bare Task/ValueTask — no result to carry
                if (TypeHelpers.IsTaskLike(result, k, out _))
                {
                    report(Diagnostic.Create(Nw003, loc, label, name, returnType.ToDisplayString()));
                    return;
                }

                carrier = result;
            }
            else if (isAsync)
            {
                // An async method whose return is NOT one of the four exact task shapes — a custom
                // [AsyncMethodBuilder] task-like (or void). The carrier that matters is the state
                // machine's RESULT type, which only exists at IL time (SetResult's argument); the
                // weaver validates it there, so gating the task-like TYPE here would be a false
                // positive on a shape the weaver weaves. Defer.
                return;
            }

            if (!TypeHelpers.IsSupportedCarrier(carrier, k))
                report(Diagnostic.Create(Nw003, loc, label, name, returnType.ToDisplayString()));
        }

        private static bool HasAttr(ISymbol s, INamedTypeSymbol attr)
        {
            if (attr == null)
                return false;
            foreach (var a in s.GetAttributes())
                if (SymbolEqualityComparer.Default.Equals(a.AttributeClass, attr))
                    return true;
            return false;
        }

        private static Location Loc(ISymbol s) => s.Locations.Length > 0 ? s.Locations[0] : Location.None;
        private static string Display(ISymbol s) => s.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat);
    }
}
