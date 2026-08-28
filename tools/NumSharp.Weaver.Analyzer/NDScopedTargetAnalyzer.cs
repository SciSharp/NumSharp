using System;
using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace NumSharp.Weaver.Analyzer
{
    /// <summary>
    ///     Compile-time gate for <c>[NDScoped]</c> / <c>[NDScopedAsync]</c>. Reports a build ERROR
    ///     when either attribute is placed on a target the IL weaver (<c>tools/NumSharp.Weaver</c>)
    ///     could not weave — the SAME NDW diagnostics the weaver produces post-compile, surfaced in
    ///     the editor and at CoreCompile so the mistake is caught before the weave runs (and, in the
    ///     package, the error preempts the weaver: a failed CoreCompile never reaches the weave step).
    ///     <para>
    ///     It covers every SOURCE-detectable rejection: the wrong attribute (NDW009 = async/Task under
    ///     <c>[NDScoped]</c>, NDW010 = a plain sync method / synchronous iterator under
    ///     <c>[NDScopedAsync]</c>, NDW011 = both attributes), a hidden <c>ref NDArray</c> egress
    ///     (NDW002), an unsupported carrier return (NDW003), a body-less method (NDW005), and a
    ///     setter-only property (NDW006). The IL-only rejections stay with the weaver: NDW004 (an
    ///     unrecognized state-machine shape), NDW007 (a tail-call prefix), NDW008 (the referenced
    ///     NumSharp predates the async seam).
    ///     </para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NDScopedTargetAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Weaver";
        private const string HelpLink = "https://scisharp.github.io/NumSharp/docs/ndscoped.html";
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
            "{0} method '{1}' has no body (abstract/extern) — remove the attribute");

        internal static readonly DiagnosticDescriptor Nw006 = Error("NDW006",
            "Scoped attribute on a setter-only property",
            "{0} on property '{1}' requires a getter (put the attribute on the getter accessor for setter-only properties)");

        internal static readonly DiagnosticDescriptor Nw009 = Error("NDW009",
            "async / Task-returning method marked [NDScoped]",
            "[NDScoped] method '{0}' is {1} — mark it [NDScopedAsync] instead ([NDScoped] weaves synchronous methods and synchronous iterators; [NDScopedAsync] weaves async methods, async iterators and non-async Task/ValueTask returns)");

        internal static readonly DiagnosticDescriptor Nw010 = Error("NDW010",
            "synchronous method / iterator marked [NDScopedAsync]",
            "[NDScopedAsync] method '{0}' is {1} — mark it [NDScoped] instead ([NDScopedAsync] weaves async methods, async iterators and non-async Task/ValueTask returns)");

        internal static readonly DiagnosticDescriptor Nw011 = Error("NDW011",
            "method carries both [NDScoped] and [NDScopedAsync]",
            "method '{0}' carries BOTH [NDScoped] and [NDScopedAsync] — a method has exactly one scoping model; keep only the attribute that matches it ([NDScoped] for synchronous methods and synchronous iterators, [NDScopedAsync] for async methods, async iterators and non-async Task/ValueTask returns)");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw002, Nw003, Nw005, Nw006, Nw009, Nw010, Nw011);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterCompilationStartAction(start =>
            {
                var known = KnownTypes.Resolve(start.Compilation);
                if (known == null)
                    return; // NumSharp not referenced — nothing to analyze

                start.RegisterSymbolAction(c => AnalyzeMethod((IMethodSymbol)c.Symbol, known, c.ReportDiagnostic), SymbolKind.Method);
                start.RegisterSymbolAction(c => AnalyzeProperty((IPropertySymbol)c.Symbol, known, c.ReportDiagnostic), SymbolKind.Property);
            });
        }

        private static void AnalyzeMethod(IMethodSymbol m, KnownTypes k, Action<Diagnostic> report)
        {
            bool sync = HasAttr(m, k.SyncAttr);
            bool async = HasAttr(m, k.AsyncAttr);
            if (!sync && !async)
                return; // the overwhelmingly common case — bail before any type work

            bool hasBody = !m.IsAbstract && !m.IsExtern
                           && !(m.IsPartialDefinition && m.PartialImplementationPart == null);
            AnalyzeTarget(m, sync, async, m.ReturnType, m.IsAsync, m.Parameters, hasBody, k, report);
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

            // The getter is synchronous (properties cannot be async) and never has ref parameters.
            AnalyzeTarget(p, sync, async, p.Type, isAsync: false, ImmutableArray<IParameterSymbol>.Empty, hasBody: true, k, report);
        }

        private static void AnalyzeTarget(ISymbol symbol, bool sync, bool async, ITypeSymbol returnType,
            bool isAsync, ImmutableArray<IParameterSymbol> parameters, bool hasBody, KnownTypes k, Action<Diagnostic> report)
        {
            var loc = Loc(symbol);
            var name = Display(symbol);

            // NDW011 — a method has exactly one scoping model.
            if (sync && async)
            {
                report(Diagnostic.Create(Nw011, loc, name));
                return;
            }

            bool taskLike = TypeHelpers.IsTaskLike(returnType, k, out _);

            // Wrong attribute — mirrors the SyncScopeWeaver / AsyncScopeWeaver routing exactly. A
            // synchronous iterator returns a bare enumerable (not async, not Task), so it is NOT caught
            // by NDW009 (valid under [NDScoped]) and IS caught by NDW010 (invalid under [NDScopedAsync]).
            if (sync && (isAsync || taskLike))
            {
                report(Diagnostic.Create(Nw009, loc, name,
                    isAsync ? "an async method" : "a Task/ValueTask-returning method"));
                return;
            }

            if (async && !isAsync && !taskLike)
            {
                report(Diagnostic.Create(Nw010, loc, name,
                    TypeHelpers.IsEnumerableInterface(returnType, k) ? "a synchronous iterator" : "a plain synchronous method"));
                return;
            }

            string label = async ? AsyncName : SyncName;

            // NDW005 — abstract / extern (no body to weave).
            if (!hasBody)
            {
                report(Diagnostic.Create(Nw005, loc, label, name));
                return;
            }

            // NDW002 — a 'ref' (or 'in') NDArray-carrying parameter is a hidden egress. async methods
            // and iterators cannot have by-ref parameters, so this only ever matches a plain method.
            foreach (var pr in parameters)
            {
                if (pr.RefKind != RefKind.None && pr.RefKind != RefKind.Out && TypeHelpers.IsNDArrayCarrying(pr.Type, k))
                {
                    report(Diagnostic.Create(Nw002, loc, label, name, pr.Type.ToDisplayString(), pr.Name));
                    return;
                }
            }

            // NDW003 — an unsupported return carrier. A bare enumerable interface is DEFERRED to the
            // weaver: a real iterator (yield) is fine and validates its element type at IL time, and a
            // non-iterator returning IEnumerable is rare — the analyzer must not flag the valid iterator.
            // A Task/ValueTask is unwrapped once to the result it carries.
            if (TypeHelpers.IsEnumerableInterface(returnType, k))
                return;

            var carrier = returnType;
            if (TypeHelpers.IsTaskLike(returnType, k, out var result))
            {
                if (result == null)
                    return; // bare Task/ValueTask — no result to carry
                carrier = result;
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
