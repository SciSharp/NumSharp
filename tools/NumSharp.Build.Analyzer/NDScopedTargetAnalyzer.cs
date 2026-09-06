using System;
using System.Collections.Immutable;
using System.Linq;
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
    ///     declaration carrying the attribute is that contract, never NDW005.
    ///     </para>
    ///     <para>
    ///     <b>NDW013 — the weaver-missing guard.</b> Every attribute this analyzer gates is INERT without
    ///     the <c>NumSharp.Build</c> package (NumSharp does not depend on it; weaving is an explicit
    ///     opt-in): the method keeps its original body, no <c>NDScope</c> is opened, no
    ///     <c>[NDScopedExit]</c> argument is detached, and the temporaries fall to the finalizer. When the
    ///     build has DECLARED the weaver inactive — <c>build_property.NumSharpBuildActive</c> present and
    ///     not <c>true</c> (NumSharp's <c>build/NumSharp.targets</c> defaults it to <c>false</c>; the
    ///     weaver's <c>build/NumSharp.Build.targets</c> sets <c>true</c> when it will run) and the opt-out
    ///     <c>build_property.NumSharpDisableWeaverMissingWarning</c> not set — this analyzer reports NDW013,
    ///     a WARNING, at every member the weaver WOULD have woven: a member carrying <c>[NDScoped]</c>/
    ///     <c>[NDScopedAsync]</c> itself, an override/implementation inheriting one (from this assembly or
    ///     another), and a method with an own or inherited <c>[NDScopedExit]</c> parameter. It is the
    ///     PRIMARY reporter (per site, IDE-visible); the MSBuild text scan in NumSharp's targets is the
    ///     fallback for a compile the analyzer is not applied to, and yields when it is. Absent property =
    ///     unknown (a source-mode build importing neither targets file) = silent. A member the gate rejects
    ///     draws its error alone — it would not have been woven either way.
    ///     </para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NDScopedTargetAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Build";
        private const string HelpLink = "https://scisharp.github.io/NumSharp/docs/numsharp-build-compiler.html";
        private const string SyncName = "[NDScoped]";
        private const string AsyncName = "[NDScopedAsync]";
        private const string ExitName = "[NDScopedExit]";

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

        private const string InstallHint =
            "Install the weaver with 'dotnet add package NumSharp.Build' (a build-time development dependency, not a runtime dependency)";

        private static DiagnosticDescriptor Warning013(string title, string message) =>
            new DiagnosticDescriptor("NDW013", title, message, Category, DiagnosticSeverity.Warning,
                isEnabledByDefault: true, description: null, helpLinkUri: HelpLink + "#ndw013");

        /// <summary>
        ///     NDW013 at a member carrying <c>[NDScoped]</c>/<c>[NDScopedAsync]</c> ITSELF (the method, or the
        ///     property for a property-level attribute). Three descriptors share the one id — same code, same
        ///     severity, same fix — because the three shapes call for three different sentences.
        /// </summary>
        internal static readonly DiagnosticDescriptor Nw013 = Warning013(
            "[NDScoped] used, but the weaver is not installed",
            "{0} on '{1}' has no effect: the NumSharp.Build package is not installed (or weaving is disabled), so the attribute is INERT — no NDScope is injected and the member's NDArray temporaries are left to the finalizer instead of being reclaimed. " + InstallHint + ", or remove the attribute and dispose the temporaries by hand");

        /// <summary>NDW013 at an override/implementation INHERITING the scope attribute from its declaration (this assembly or another).</summary>
        internal static readonly DiagnosticDescriptor Nw013Inherited = Warning013(
            "Inherited [NDScoped] target, but the weaver is not installed",
            "method '{0}' inherits {1} from '{2}' but the NumSharp.Build package is not installed (or weaving is disabled), so the attribute is INERT here — no NDScope is injected and the method's NDArray temporaries are left to the finalizer. " + InstallHint + ", mark the override [NDScopedCovered] to opt out of the inherited scope, or dispose the temporaries by hand");

        /// <summary>NDW013 at a method whose parameter carries (or inherits, by position) <c>[NDScopedExit]</c> and that draws no scope-attribute NDW013 of its own.</summary>
        internal static readonly DiagnosticDescriptor Nw013Exit = Warning013(
            "[NDScopedExit] used, but the weaver is not installed",
            "[NDScopedExit] on parameter {0} of '{1}'{2} has no effect: the NumSharp.Build package is not installed (or weaving is disabled), so the argument is NOT detached from the caller's NDScope and may be reclaimed while '{1}' still holds it. " + InstallHint + ", or detach the argument by hand with NDScope.Detach");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw002, Nw003, Nw005, Nw006, Nw009, Nw010, Nw011, Nw013, Nw013Inherited, Nw013Exit, Nw015);

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
                start.RegisterSymbolAction(c => AnalyzeProperty((IPropertySymbol)c.Symbol, known, weaverInactive, c.ReportDiagnostic), SymbolKind.Property);
            });
        }

        /// <summary>
        ///     True only when the build DECLARED the weaver inactive: <c>NumSharpBuildActive</c> is present
        ///     (NumSharp's targets default it to 'false'; NumSharp.Build's set it to 'true' when weaving)
        ///     and not 'true', and the documented opt-out <c>NumSharpDisableWeaverMissingWarning</c> is not
        ///     set. An ABSENT property (a source-mode build importing neither targets file) is unknown, and
        ///     unknown stays silent.
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
            bool hasBody = !m.IsAbstract && !m.IsExtern
                           && !(m.IsPartialDefinition && m.PartialImplementationPart == null);

            // Whether an NDW013 has already been reported for this method — one per member is enough,
            // whichever attribute earned it (the fix is the same).
            bool inertReported = false;

            var own = ScopeInheritance.KindsOf(m, k);
            bool sync = (own & ScopeKind.Sync) != 0;
            bool async = (own & ScopeKind.Async) != 0;

            if (sync || async)
            {
                bool rejected = AnalyzeTarget(m, sync, async, m.ReturnType, m.IsAsync, m.Parameters, hasBody, m.IsAbstract, k, report, label: null);

                // NDW013 (own attribute): a member with a body the weaver would have woven. An abstract /
                // interface declaration is the contract — nothing is woven THERE, its inheritors are warned
                // one by one below; a rejected target draws its error alone.
                if (weaverInactive && !rejected && hasBody && !(sync && async))
                {
                    report(Diagnostic.Create(Nw013, Loc(m), sync ? SyncName : AsyncName, Display(m)));
                    inertReported = true;
                }
            }
            else if (own == ScopeKind.None
                     && !(m.MethodKind == MethodKind.PropertyGet && m.AssociatedSymbol != null &&
                          ScopeInheritance.KindsOf(m.AssociatedSymbol, k) != ScopeKind.None))
            {
                // No attribute on the method itself. [NDScopedCovered] is an opt-out (of an inherited scope
                // too) and skips the scope gate; a getter whose PROPERTY carries the attribute is
                // AnalyzeProperty's target; otherwise an override/implementation may INHERIT its
                // declaration's attribute — the overwhelmingly common plain method finds nothing here and
                // falls through to the [NDScopedExit] check, before any type work.
                if (ScopeInheritance.Inherited(m, k) is ScopeDeclaration decl)
                {
                    sync = (decl.Kinds & ScopeKind.Sync) != 0;
                    async = (decl.Kinds & ScopeKind.Async) != 0;
                    // An inherited [NDScopedCovered]: the override is covered, nothing to gate. Both kinds:
                    // the declaration's own NDW011 reports it; its inheritors add nothing.
                    if ((sync || async) && !(sync && async))
                    {
                        var label = (sync ? SyncName : AsyncName) + " (inherited from " + Display(decl.Source) + ")";
                        bool rejected = AnalyzeTarget(m, sync, async, m.ReturnType, m.IsAsync, m.Parameters, hasBody, m.IsAbstract, k, report, label);

                        // NDW013 (inherited attribute): the shape no metadata text scan can see — this
                        // override spells no attribute of its own — whether the declaration lives in this
                        // assembly or another.
                        if (weaverInactive && !rejected && hasBody)
                        {
                            report(Diagnostic.Create(Nw013Inherited, Loc(m), Display(m), sync ? SyncName : AsyncName, Display(decl.Source)));
                            inertReported = true;
                        }
                    }
                }
            }

            // NDW013 ([NDScopedExit]): orthogonal to the scope attributes — a method may carry (or inherit,
            // by position) a retained-parameter mark with or without a scope of its own, and the weaver
            // injects its Detach either way. Reported only when no scope-attribute NDW013 already named
            // this method.
            if (weaverInactive && !inertReported && hasBody)
            {
                var ownExit = ScopeInheritance.OwnExitParameters(m, k);
                if (ownExit.Length > 0)
                {
                    report(Diagnostic.Create(Nw013Exit, Loc(m), ParameterNames(m, ownExit), Display(m), ""));
                }
                else
                {
                    var source = ScopeInheritance.InheritedExitSource(m, k);
                    if (source != null)
                    {
                        var positions = ScopeInheritance.OwnExitParameters(source, k).Where(i => i < m.Parameters.Length).ToArray();
                        report(Diagnostic.Create(Nw013Exit, Loc(m), ParameterNames(m, positions), Display(m),
                            " (inherited from '" + Display(source) + "')"));
                    }
                }
            }
        }

        private static void AnalyzeProperty(IPropertySymbol p, KnownTypes k, bool weaverInactive, Action<Diagnostic> report)
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
            bool rejected = AnalyzeTarget(p, sync, async, p.Type, isAsync: false, ImmutableArray<IParameterSymbol>.Empty,
                hasBody: !p.IsAbstract, isAbstract: p.IsAbstract, k, report, label: null);

            // NDW013 (own attribute, property-level): the getter the weaver would have woven. Reported on
            // the property, where the attribute is; the getter itself bails out of AnalyzeMethod for a
            // scoped property, so this is its only report.
            if (weaverInactive && !rejected && !p.IsAbstract && !(sync && async))
                report(Diagnostic.Create(Nw013, Loc(p), sync ? SyncName : AsyncName, Display(p)));
        }

        /// <summary>
        ///     The [NDScoped]-target gate for one member. Returns true when it REPORTED a rejection (the
        ///     member cannot be woven as declared), false when the member passed — or was deferred to the
        ///     weaver's IL-time checks, which for the purposes of NDW013 counts as weavable.
        /// </summary>
        private static bool AnalyzeTarget(ISymbol symbol, bool sync, bool async, ITypeSymbol returnType,
            bool isAsync, ImmutableArray<IParameterSymbol> parameters, bool hasBody, bool isAbstract, KnownTypes k,
            Action<Diagnostic> report, string label)
        {
            var loc = Loc(symbol);
            var name = Display(symbol);

            // NDW011 — a method has exactly one scoping model.
            if (sync && async)
            {
                report(Diagnostic.Create(Nw011, loc, name));
                return true;
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
                return true;
            }

            if (async && !isAsync && !taskLike)
            {
                report(Diagnostic.Create(Nw010, loc, label, name,
                    TypeHelpers.IsEnumerableInterface(returnType, k) ? "a synchronous iterator" : "a plain synchronous method"));
                return true;
            }

            // An abstract / interface declaration is the CONTRACT its overrides and implementations
            // inherit — nothing to weave there, and nothing wrong. (Its signature is still gated above
            // and below: a wrong attribute or a hidden egress is wrong for every inheritor.)
            // NDW005 — a body-less NON-abstract target (extern, an unimplemented partial) can never be woven.
            if (!hasBody && !isAbstract)
            {
                report(Diagnostic.Create(Nw005, loc, label, name));
                return true;
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
                    return true;
                }

                if (!TypeHelpers.IsSupportedOutCarrier(pr.Type, k))
                {
                    report(Diagnostic.Create(Nw015, loc, label, name, pr.Type.ToDisplayString(), pr.Name));
                    return true;
                }
            }

            // NDW003 — an unsupported return carrier. A bare enumerable interface is DEFERRED to the
            // weaver: a real iterator (yield) is fine and validates its element type at IL time, and a
            // non-iterator returning IEnumerable is rare — the analyzer must not flag the valid iterator.
            // A Task/ValueTask is unwrapped once to the result it carries; a NESTED task is refused
            // (the inner task's completion outlives the scope's deferral lifecycle — weaver parity).
            if (TypeHelpers.IsEnumerableInterface(returnType, k))
                return false;

            var carrier = returnType;
            if (TypeHelpers.IsTaskLike(returnType, k, out var result))
            {
                if (result == null)
                    return false; // bare Task/ValueTask — no result to carry
                if (TypeHelpers.IsTaskLike(result, k, out _))
                {
                    report(Diagnostic.Create(Nw003, loc, label, name, returnType.ToDisplayString()));
                    return true;
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
                return false;
            }

            if (!TypeHelpers.IsSupportedCarrier(carrier, k))
            {
                report(Diagnostic.Create(Nw003, loc, label, name, returnType.ToDisplayString()));
                return true;
            }

            return false;
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

        /// <summary>The quoted names of <paramref name="m"/>'s parameters at <paramref name="positions"/>, comma-separated ("'a'", "'a', 'b'").</summary>
        private static string ParameterNames(IMethodSymbol m, int[] positions)
            => string.Join(", ", positions.Where(i => i >= 0 && i < m.Parameters.Length).Select(i => "'" + m.Parameters[i].Name + "'"));

        private static Location Loc(ISymbol s) => s.Locations.Length > 0 ? s.Locations[0] : Location.None;
        private static string Display(ISymbol s) => s.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat);
    }
}
