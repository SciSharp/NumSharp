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
    ///     The message names those fixes in the order to try them (a false positive is suppressed, a
    ///     helper every call of which already runs under a scope is marked <c>[NDScopedCovered]</c>, an
    ///     entry point that keeps no array past the call is marked <c>[NDScoped]</c>, anything else is
    ///     disposed explicitly), each with the few words that rule out its misuse; the description and
    ///     the help page carry the full conditions. A dropped disposable that stores NDArrays gets its
    ///     own text, because no scope attribute reclaims it.
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

        /// <summary>
        ///     NDW012 for a dropped <c>NDArray</c> (or a carrier of one), written in three tiers because a build
        ///     prints the message once per warning site. The MESSAGE stays one short paragraph: the rule against
        ///     adding the attribute as a reflex, with its reason, then the four fixes in the order to try them,
        ///     each with the words that rule out its misuse. The DESCRIPTION (shown on demand by an IDE, written
        ///     once into a SARIF log) carries every condition, and the help page, whose link the compiler prints
        ///     after the message, explains each one. <c>{0}</c> is the attribute that fits the enclosing method
        ///     (<c>[NDScoped]</c>, or <c>[NDScopedAsync]</c> for an async / Task-shaped method); <c>{1}</c> names
        ///     the value (<c>NDArray</c>).
        /// </summary>
        /// <remarks>
        ///     <para>The case order follows a 2026-10-16 triage of every NDW012 in NumSharp.Core's build (171
        ///     sites): 36 were not leaks (a call that returns its own out array, a callee that takes the array
        ///     over), 133 were already reclaimed by a caller's scope, where <c>[NDScopedCovered]</c> is the fix and
        ///     a nested <c>[NDScoped]</c> only adds a scope to every call, and 2 were real leaks on unscoped
        ///     paths.</para>
        ///     <para>Why the message is short: a first version carried the whole procedure (3,034 characters),
        ///     which the build repeats at every site and again in its summary (NumSharp.Core's 171 sites printed
        ///     about 1 MB of it per framework), floods an agent's context and is cut off by IDE lists. The message
        ///     keeps the words that block a use-after-free (stored, cached, captured, a holder handed out, a
        ///     possible input disposed) and a memory blow-up (a long loop under a scope); conditions whose
        ///     violation only leaves the leak in place stay in the description. The contract test caps the
        ///     message length.</para>
        ///     <para>The first sentence claims only what this analysis knows: nothing in THIS method reclaims the
        ///     array. A scope a caller opened does reclaim it (case 2), so a sentence such as "this array is never
        ///     scoped" would be false in exactly the case the analysis cannot see, and would send an agent to add a
        ///     scope. Case 2 names the signal to look for in the callers (an <c>[NDScoped*]</c> attribute or
        ///     <c>NDScope.Open()</c>) and excludes a caller that runs the method in a long loop: scopes nest, so
        ///     <c>[NDScoped]</c> under a caller's scope frees each call's arrays when the call returns, while a
        ///     covered method's arrays wait for the caller's scope to close.</para>
        ///     <para>Every condition in the description is a way the attributes would dispose an array the program
        ///     still uses, or stop reclaiming one, that neither the weaver nor this analyzer reports:
        ///     an array kept past the call, by the method or by a callee (a static cache included: the caller's
        ///     scope tracks it); a class, or a struct without <c>INDArrayCarrier</c>, that holds arrays in its own
        ///     fields and leaves as a tuple component or a bare out parameter (the weaver refuses it only as a
        ///     direct return, NDW003; proven on woven code 2026-10-16 with <c>np.ma.average</c>'s core and with
        ///     probes of all three shapes, each handing the caller disposed arrays);
        ///     a pointer into a temporary (an <c>IntPtr</c> return is weavable);
        ///     a scope held open across a long loop, which keeps every array of every pass until it closes (an
        ///     explicit <c>Dispose</c> releases the buffer, but the scope still references the object);
        ///     an array created in a lambda or task that runs on another thread or after the scope closed, which
        ///     no scope tracks (<c>NDScope</c> is <c>[ThreadStatic]</c>);
        ///     an async method marked <c>[NDScopedCovered]</c>, which rides its caller's scope only up to its first
        ///     <c>await</c>;
        ///     and a covered helper reached through a delegate, an interface or a base virtual, which a search for
        ///     direct callers misses.
        ///     A constructor can take neither attribute (both are Method/Property-only, CS0592).</para>
        ///     <para>Keep the message a single line: build logs and agents read it line by line, and RS1032 forbids
        ///     line breaks in a message.</para>
        /// </remarks>
        internal static readonly DiagnosticDescriptor Nw012 = new DiagnosticDescriptor(
            "NDW012",
            "NDArray is created but never disposed, returned, stored, or scoped (left to the finalizer)",
            "Nothing in this method disposes, returns, stores, or scopes this {1}. Do not add {0} just to silence " +
            "this: it disposes every array the call creates except what the method returns. Take the first that fits: " +
            "(1) a call hands back an array the method already holds (@out, inplace: true): suppress only this line; " +
            "(2) every caller already has a scope ([NDScoped*] or NDScope.Open()), none runs it in a long loop, and " +
            "this method is not public: [NDScopedCovered]; (3) no new array is stored, cached, captured, or handed " +
            "out in a holder, and no long loop in it creates arrays: {0}; (4) otherwise dispose it after its last " +
            "use, unless it may be an input (np.asarray(x)).",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Every NDArray owns a pooled unmanaged buffer; one that is dropped waits for the finalizer. " +
                         "This check reads one method at a time and cannot see a scope a caller opened. [NDScoped] " +
                         "and [NDScopedAsync] dispose, when the method finishes, every array created during the call " +
                         "(by the method or anything it calls, on its thread) except what it returns, yields, or " +
                         "assigns to out parameters and what NDScope.Detach detached. Case 1, not a leak: a call hands " +
                         "back an array the method already holds (@out, inplace: true; not a view or copy) or the " +
                         "callee takes the array over: suppress only that line, with a comment. Case 2: every call of " +
                         "a non-public method, counting delegate, interface, and base-virtual calls, runs on the " +
                         "calling thread while a caller's scope is open, no caller runs it in a long loop, and the " +
                         "method creates no array after an await: [NDScopedCovered], free at run time. Scopes nest: " +
                         "[NDScoped] under a caller's scope adds an inner scope that frees the method's arrays when it " +
                         "returns and passes its result to the outer scope, which costs a scope per call but is the " +
                         "right choice under a caller's long loop, where a covered method's arrays would pile up in " +
                         "the caller's scope. Case 3: no array created during the call is kept past " +
                         "it by the method or a callee (field, property, static cache, collection, caller's array, a " +
                         "lambda or task run later) unless detached (NDScope.Detach, an [NDScopedExit] parameter); only " +
                         "NDArray, NDArray[], INDArrayCarrier structs, tuples of those, or array-free values are " +
                         "returned or put in out parameters (a holder such as NDMaskedArray or poly1d comes back " +
                         "disposed, even inside a tuple); no pointer into a temporary escapes: [NDScoped] or " +
                         "[NDScopedAsync], or the same on its unscoped callers and then case 2. Neither attribute fits " +
                         "a constructor, a scope open across a long loop that creates arrays (in the method, or in a " +
                         "caller for [NDScopedCovered]), or arrays a lambda or task creates on another thread or after " +
                         "the scope closed. Case 4: dispose after the last " +
                         "use, never an array the method returns, stores, or was given, nor one that may be an input " +
                         "(np.asarray(x), x.astype(t, copy: false)).",
            helpLinkUri: HelpLink);

        /// <summary>
        ///     NDW012 for a dropped <c>NDArray</c> in a project whose build DECLARES the NumSharp.Build weaver
        ///     inactive (<c>build_property.NumSharpBuildActive</c> present and not <c>true</c>: NumSharp's targets
        ///     default it to <c>false</c>, the weaver's targets set <c>true</c> when they will weave). There
        ///     <c>[NDScoped]</c> and <c>[NDScopedAsync]</c> are inert (NDW013): the method keeps its body, no scope
        ///     opens, and an attributed CALLER opens none either. So this text does not offer the attribute as a
        ///     fix, counts only scopes opened with <c>NDScope.Open()</c> as a caller's scope, makes explicit
        ///     disposal the fix, and names installing NumSharp.Build as the way to scope a method with many
        ///     temporaries. <c>{0}</c> and <c>{1}</c> are as in <see cref="Nw012"/>.
        /// </summary>
        /// <remarks>
        ///     The best fix depends on the project: with the weaver, <c>[NDScoped]</c> ranks above explicit
        ///     disposal because it reclaims every array the method creates, including the temporaries this
        ///     analysis deliberately leaves unflagged (read only through a member call) and those on exception
        ///     paths; without it, the attribute changes nothing and explicit disposal is the best fix that works.
        ///     An ABSENT property (a build importing neither targets file) is unknown and keeps
        ///     <see cref="Nw012"/>, the same reading NDW013 gives it.
        /// </remarks>
        internal static readonly DiagnosticDescriptor Nw012NoWeaver = new DiagnosticDescriptor(
            "NDW012",
            "NDArray is created but never disposed, returned, stored, or scoped (left to the finalizer)",
            "Nothing in this method disposes, returns, stores, or scopes this {1}. {0} is inert here: this project " +
            "does not use the NumSharp.Build package. Take the first that fits: (1) a call " +
            "hands back an array the method already holds (@out, inplace: true): suppress only this line; (2) every " +
            "call already runs inside a scope opened with NDScope.Open() ([NDScoped] callers open none here), none " +
            "runs it in a long loop, and this method is not public: [NDScopedCovered]; (3) otherwise dispose it " +
            "after its last use, unless it may be an input (np.asarray(x)). For a method with many temporaries that " +
            "keeps none past the call, installing NumSharp.Build lets {0} reclaim them all.",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "This project's build declares the NumSharp.Build weaver inactive (NumSharpBuildActive is " +
                         "not true), so [NDScoped] and [NDScopedAsync] are inert (NDW013): no scope opens in the " +
                         "method or in an attributed caller, and the array waits for the finalizer. This check reads " +
                         "one method at a time and cannot see a scope a caller opened with NDScope.Open(). Case 1, " +
                         "not a leak: a call hands back an array the method already holds (@out, inplace: true; not a " +
                         "view or copy) or the callee takes the array over: suppress only that line, with a comment. " +
                         "Case 2: every call of a non-public method runs on the calling thread inside a scope a " +
                         "caller opened with NDScope.Open(), no caller runs it in a long loop, and the method creates " +
                         "no array after an await: [NDScopedCovered], free at run time. Case 3: dispose the array " +
                         "after its last use, never an array the method returns, stores, or was given, nor one that " +
                         "may be an input (np.asarray(x), x.astype(t, copy: false)). A method with many temporaries " +
                         "can instead install the weaver ('dotnet add package NumSharp.Build', a build-time " +
                         "dependency) and take [NDScoped] or [NDScopedAsync] under the conditions on the help page, " +
                         "or open a scope by hand ('using var scope = NDScope.Open();') and route every return " +
                         "through scope.Returns(...); a return that skips it hands back a disposed array.",
            helpLinkUri: HelpLink);

        /// <summary>
        ///     NDW012 for a dropped DISPOSABLE that stores NDArrays: a holder class, NumSharp's <c>np.nditer</c>
        ///     iterator, an <c>NpzFile</c>, or an array of them. It has a text of its own because the scope
        ///     attributes cannot fix it: an <c>NDScope</c> reclaims only the NDArrays created during the
        ///     call and never calls the instance's <c>Dispose</c>, so whatever else the instance owns (native
        ///     iterator state, an open file) waits for the finalizer. <c>{0}</c> names the value.
        /// </summary>
        internal static readonly DiagnosticDescriptor Nw012Holder = new DiagnosticDescriptor(
            "NDW012",
            "Disposable that stores NDArrays is created but never disposed, returned, or stored",
            "This {0} is never disposed, returned, or stored. An NDScope never calls its Dispose, so neither " +
            "[NDScoped] nor [NDScopedCovered] fixes this: dispose it after its last use ('using', Dispose(), or " +
            "Close()), or suppress only this line if the method it is passed to disposes or returns it. A foreach " +
            "disposes only its enumerator: hold the collection in a 'using'.",
            Category,
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A disposable that stores NDArrays (a holder class of your own, NumSharp's np.nditer " +
                         "iterator, an NpzFile) owns its arrays and possibly native state or an open file. An NDScope " +
                         "reclaims only the NDArrays created during the call and never calls Dispose, so the instance " +
                         "itself must be disposed or closed once it is no longer used (each element of an array of " +
                         "them). A foreach disposes the enumerator it obtains, not the collection it walks. When the " +
                         "method the holder is passed to disposes it or returns it, nothing leaks: suppress only that " +
                         "line, with a comment.",
            helpLinkUri: HelpLink);

        /// <summary>
        ///     The three NDW012 texts: one id and one severity, three messages, because the best fix differs. A
        ///     dropped NDArray with the weaver active (or its state unknown): find who reclaims it, then suppress,
        ///     cover, scope, or dispose. The same without the weaver: suppress, cover, or dispose, since the scope
        ///     attributes are inert there. A dropped holder: always dispose it.
        /// </summary>
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
            ImmutableArray.Create(Nw012, Nw012NoWeaver, Nw012Holder);

        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            // Generated code is never analyzed (no Analyze flag), and a diagnostic that lands in it is dropped by
            // CompilationFacts.Generated before it is reported: ReportDiagnostics only switches off the driver's own
            // per-diagnostic test, whose first use per tree rescans the tree on every thread (GeneratedCodeFilter).
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.ReportDiagnostics);
            context.RegisterCompilationStartAction(start =>
            {
                // Shared with the holder and target analyzers: the ownership model makes NDArray-owning
                // DISPOSABLES (a consumer's holder class, NumSharp's own np.nditer iterator) owned values too
                // (ownership is contagious), and the name index gates which bodies are walked at all.
                var facts = CompilationFacts.Get(start.Compilation);
                if (facts == null)
                    return; // NumSharp not referenced — nothing to analyze

                // Read once per compilation: the build property is global, and it decides which text
                // offers [NDScoped] (it is inert when the weaver is declared inactive).
                bool weaverInactive = WeaverDeclaredInactive(start.Options.AnalyzerConfigOptionsProvider.GlobalOptions);
                start.RegisterOperationBlockAction(c => BlockScan.Analyze(facts, c, weaverInactive));
            });
        }

        /// <summary>
        ///     True when the build DECLARES the NumSharp.Build weaver inactive: <c>NumSharpBuildActive</c> is
        ///     present (NumSharp's targets default it to <c>false</c>; the weaver's targets set <c>true</c> when
        ///     they will weave) and is not <c>true</c>. An absent property, a build importing neither targets file,
        ///     is unknown and answers false, so such a build keeps the text that offers <c>[NDScoped]</c>, the
        ///     reading NDW013 gives the same property.
        /// </summary>
        /// <remarks>
        ///     NDW013's opt-out (<c>NumSharpDisableWeaverMissingWarning</c>) does not count here, unlike in
        ///     NDW013's own check: it silences a warning about inert attributes, but the attributes stay inert, so
        ///     offering <c>[NDScoped]</c> as a fix would still be wrong.
        /// </remarks>
        /// <param name="options">The compilation's global analyzer options, where MSBuild's generated editorconfig
        ///     puts every <c>CompilerVisibleProperty</c> as <c>build_property.&lt;name&gt;</c>.</param>
        /// <returns>True when <see cref="Nw012NoWeaver"/> applies to every dropped NDArray of the compilation.</returns>
        private static bool WeaverDeclaredInactive(AnalyzerConfigOptions options)
            => options.TryGetValue("build_property.NumSharpBuildActive", out var active) &&
               !string.Equals(active?.Trim(), "true", System.StringComparison.OrdinalIgnoreCase);

        private enum Disposition
        {
            Leaks,     // read/dropped/fed to a non-consuming NumSharp op — never reclaimed
            Escapes,   // returned / out / ref / stored / handed to a consuming API
            Reclaimed  // using / Dispose / scope.Returns / NDScope.Attach
        }

        /// <summary>One method/accessor/constructor/local-function body's worth of leak analysis.</summary>
        /// <remarks>
        ///     <b>Pooled per thread.</b> A scan owns six collections, and a project analyzes one block per method, accessor
        ///     and constructor: allocated per block they were a tenth of the analyzer's time on NumSharp's own source,
        ///     mostly the garbage collections their allocation triggered. A block's analysis runs start to finish on one
        ///     thread (analyzer callbacks are synchronous), so each thread keeps one idle scan and reuses it. The pool's
        ///     contract is <see cref="Reset"/>: every reference to the compilation (symbols, operations, the context) is
        ///     dropped before a scan is parked, so a parked scan never keeps a compilation alive in a long-lived host (the
        ///     IDE, the compiler server), and a collection that one huge body grew is replaced instead of kept.
        /// </remarks>
        private sealed class BlockScan
        {
            /// <summary>This thread's idle scan (null while the thread's scan is in use, or before its first block).</summary>
            [System.ThreadStatic] private static BlockScan t_spare;

            /// <summary>A collection holding more than this many entries is replaced on <see cref="Reset"/> instead of cleared, so one huge body does not pin its capacity for the rest of the process.</summary>
            private const int RetainedCapacity = 256;

            /// <summary>The most emptied use lists kept for reuse between blocks.</summary>
            private const int SpareUseLists = 64;

            // The block's compilation-wide inputs, set by Init and dropped by Reset.
            private KnownTypes _k;
            private OwnershipModel _model;
            private NumSharpNameIndex _names;
            private OperationBlockAnalysisContext _ctx;

            // The compilation's copy of the driver's generated-code test, asked before every report.
            private GeneratedCodeFilter _generated;

            // True when the build declared the weaver inactive: a dropped NDArray then gets the text that does
            // not offer [NDScoped] (inert without NumSharp.Build). Fixed per compilation.
            private bool _weaverInactive;

            // local → its non-write value uses; the owning locals; the ones a 'using' already reclaims;
            // and the expressions that FEED a local (skipped in the expression pass — reported via the local).
            private Dictionary<ILocalSymbol, List<IOperation>> _localUses = NewLocalUses();
            private HashSet<ILocalSymbol> _owningLocals = NewLocalSet();
            private HashSet<ILocalSymbol> _aliasedLocals = NewLocalSet();
            private HashSet<ILocalSymbol> _usingLocals = NewLocalSet();
            private HashSet<IOperation> _localSourceExprs = new HashSet<IOperation>();

            // The producers of owned values, in walk order (the order the expression pass reports them in).
            private List<IOperation> _producers = new List<IOperation>();

            // Emptied use lists of earlier blocks, handed out again by Uses.
            private readonly Stack<List<IOperation>> _spareUseLists = new Stack<List<IOperation>>();

            /// <summary>A fresh local → uses map (symbol equality: a local is one symbol for the whole body).</summary>
            /// <returns>An empty map.</returns>
            private static Dictionary<ILocalSymbol, List<IOperation>> NewLocalUses()
                => new Dictionary<ILocalSymbol, List<IOperation>>(SymbolEqualityComparer.Default);

            /// <summary>A fresh set of locals (symbol equality).</summary>
            /// <returns>An empty set.</returns>
            private static HashSet<ILocalSymbol> NewLocalSet() => new HashSet<ILocalSymbol>(SymbolEqualityComparer.Default);

            /// <summary>Binds an idle scan to one operation block that passed <see cref="Analyze"/>'s gates; <see cref="Run"/> does the work.</summary>
            /// <param name="facts">The compilation's shared facts: NumSharp's types, the ownership model (which values
            ///     are owned, contagion included) and the name index (which bodies can involve NumSharp at all).</param>
            /// <param name="ctx">The block being analyzed; diagnostics are reported through it.</param>
            /// <param name="weaverInactive">True when the build declared the weaver inactive, which selects
            ///     <see cref="Nw012NoWeaver"/> for a dropped NDArray.</param>
            private void Init(CompilationFacts facts, OperationBlockAnalysisContext ctx, bool weaverInactive)
            {
                _k = facts.Known;
                _model = facts.Model;
                _names = facts.Names;
                _generated = facts.Generated;
                _ctx = ctx;
                _weaverInactive = weaverInactive;
            }

            /// <summary>
            ///     Returns the scan to its idle state: every collection emptied (or replaced when it grew past
            ///     <see cref="RetainedCapacity"/>), the use lists parked for reuse, and every reference to the compilation
            ///     dropped - the condition for parking it in <see cref="t_spare"/>.
            /// </summary>
            private void Reset()
            {
                foreach (var uses in _localUses.Values)
                {
                    uses.Clear();
                    if (_spareUseLists.Count < SpareUseLists && uses.Capacity <= RetainedCapacity)
                        _spareUseLists.Push(uses);
                }

                if (_localUses.Count > RetainedCapacity) _localUses = NewLocalUses(); else _localUses.Clear();
                if (_owningLocals.Count > RetainedCapacity) _owningLocals = NewLocalSet(); else _owningLocals.Clear();
                if (_aliasedLocals.Count > RetainedCapacity) _aliasedLocals = NewLocalSet(); else _aliasedLocals.Clear();
                if (_usingLocals.Count > RetainedCapacity) _usingLocals = NewLocalSet(); else _usingLocals.Clear();
                if (_localSourceExprs.Count > RetainedCapacity) _localSourceExprs = new HashSet<IOperation>(); else _localSourceExprs.Clear();
                if (_producers.Capacity > RetainedCapacity) _producers = new List<IOperation>(); else _producers.Clear();

                _k = null;
                _model = null;
                _names = null;
                _generated = null;
                _ctx = default;
                _weaverInactive = false;
            }

            /// <summary>
            ///     Analyzes one operation block and reports its leaks. The work is ordered cheapest first, and every step
            ///     that can prove "nothing to report" ends the analysis before the next one runs:
            ///     <list type="number">
            ///         <item>the name index (syntax only): a body that names nothing through which NumSharp enters code,
            ///         in a type that is not relevant, cannot hold an owned value - its operation tree is never created,
            ///         which is most of the cost of analyzing a body (an NDArray-free project pays only this step);</item>
            ///         <item>the method's own scope attributes (cached symbol data);</item>
            ///         <item>ONE walk of the operations that collects the locals, the producers and a hand-written
            ///         <c>NDScope.Open</c> together (the scan used to walk the tree four times);</item>
            ///         <item>the two passes over what the walk collected;</item>
            ///         <item>only when a leak would be reported: the inherited-scope exemption, whose override and
            ///         interface walk is the most expensive check a body can need.</item>
            ///     </list>
            ///     The answers are the same as running every check up front: each exemption suppresses every report of
            ///     the body, so checking it last changes nothing but its cost. The first two steps run here, before this
            ///     thread's pooled <see cref="BlockScan"/> is taken: a body they end is never bound to it.
            /// </summary>
            /// <param name="facts">The compilation's shared facts (NumSharp's types, the ownership model, the name index).</param>
            /// <param name="ctx">The block being analyzed; diagnostics are reported through it.</param>
            /// <param name="weaverInactive">True when the build declared the weaver inactive, which selects
            ///     <see cref="Nw012NoWeaver"/> for a dropped NDArray.</param>
            public static void Analyze(CompilationFacts facts, OperationBlockAnalysisContext ctx, bool weaverInactive)
            {
                if (!(ctx.OwningSymbol is IMethodSymbol method) || method.IsImplicitlyDeclared)
                    return;

                // Pay-for-play (1): syntax. Every report needs a value of an owned type, and such a value enters a
                // body only through a name the index knows (see NumSharpNameIndex).
                if (!facts.Names.BodyMayInvolveNumSharp(method))
                    return;

                // Exempt: a method the author has already marked scope-covered — a
                // [NDScoped]/[NDScopedAsync] boundary (the weaver reclaims every transient it
                // constructs), or an [NDScopedCovered] (an analyzer-only assertion that this helper
                // always runs under a caller's ambient scope, so that scope reclaims its transients).
                if (HasOwnScopeAttribute(method, facts.Known))
                    return;

                var scan = t_spare ?? new BlockScan();
                t_spare = null; // taken: should a block ever be analyzed inside another on this thread, it builds its own
                try
                {
                    scan.Init(facts, ctx, weaverInactive);
                    scan.Run(method);
                }
                finally
                {
                    // Emptied before it is parked, on every exit (a cancellation included): a parked scan that held an
                    // operation or a symbol would keep the whole compilation alive in a long-lived host.
                    scan.Reset();
                    t_spare = scan;
                }
            }

            /// <summary>The steps of <see cref="Analyze"/> that need the per-body state: the walk, the two passes, the inherited-scope exemption and the reports.</summary>
            /// <param name="method">The block's owning method, already past the name and own-attribute gates.</param>
            private void Run(IMethodSymbol method)
            {
                // One walk; false when the body opens an NDScope by hand (exempt: its own scope owns reclamation, so
                // flagging its temps would be noise - the same signal the weaver's idempotence check uses).
                if (!Collect(_ctx.OperationBlocks))
                    return;
                if (_owningLocals.Count == 0 && _producers.Count == 0)
                    return; // no owned value at all: nothing can be reported

                var reports = new List<(Location Location, ITypeSymbol Type)>();

                // 1) owning locals whose value never escapes / is never reclaimed.
                foreach (var local in _owningLocals)
                {
                    if (_usingLocals.Contains(local))
                        continue;
                    if (DispositionOfLocal(local) == Disposition.Leaks)
                        reports.Add((local.Locations.Length > 0 ? local.Locations[0] : Location.None, local.Type));
                }

                // 2) owning expressions whose value is not captured by a local and never escapes.
                foreach (var op in _producers)
                {
                    if (_localSourceExprs.Contains(op))
                        continue;
                    if (DispositionOf(op) != Disposition.Leaks)
                        continue;
                    // Report only the OUTERMOST leaking expression of a chain — a nested owned
                    // sub-expression whose owning ancestor also leaks is subsumed by that ancestor.
                    if (HasLeakingOwnedAncestor(op))
                        continue;
                    reports.Add((op.Syntax.GetLocation(), op.Type));
                }

                if (reports.Count == 0)
                    return;

                // Exempt by INHERITANCE: an override/implementation of a scoped (or covered) virtual, abstract or
                // interface declaration is woven - or covered - under that declaration's attribute exactly as if it
                // carried it (ScopeInheritance, the weaver's rule), so its transients are reclaimed the same way. Asked
                // last because the walk is the costliest check, and the name index skips it when no declaration of
                // the method's name could be inherited.
                if (_names.MayInheritScope(method) && ScopeInheritance.Inherited(method, _k) != null)
                    return;

                foreach (var (location, type) in reports)
                    Report(location, method, type);
            }

            /// <summary>
            ///     The single walk: collects the uses and sources of every local (what <see cref="DispositionOfLocal"/>
            ///     and the owning/aliased sets need) and every producer of an owned value, in walk order.
            /// </summary>
            /// <param name="blocks">The body's operation blocks.</param>
            /// <returns>False when the body calls <c>NDScope.Open</c> / <c>OpenOrResume</c> (exempt); true otherwise.</returns>
            private bool Collect(ImmutableArray<IOperation> blocks)
            {
                foreach (var block in blocks)
                {
                    foreach (var op in block.DescendantsAndSelf())
                    {
                        // Dispatched on the operation's Kind, one virtual read, rather than on its interface: a type
                        // pattern per arm cost every operation up to five interface casts (profiled: an eighth of the
                        // analyzer's time on NumSharp's own source). Each Kind belongs to exactly one interface, so the
                        // cast in an arm cannot fail and the arms see the operations the patterns saw.
                        switch (op.Kind)
                        {
                            case OperationKind.Invocation:
                                if (IsNDScopeMethod(((IInvocationOperation)op).TargetMethod, ScopeOpenNames))
                                    return false;
                                break;

                            case OperationKind.LocalReference:
                                var lref = (ILocalReferenceOperation)op;
                                if (!IsWriteTarget(lref) && !IsDeclarationReference(lref))
                                    Uses(lref.Local).Add(lref);
                                break;

                            case OperationKind.VariableDeclarator:
                                var decl = (IVariableDeclaratorOperation)op;
                                RegisterUsingIfAny(decl);
                                if (decl.GetVariableInitializer()?.Value is IOperation init)
                                    RegisterSource(decl.Symbol, init);
                                break;

                            case OperationKind.SimpleAssignment:
                                var asg = (ISimpleAssignmentOperation)op;
                                if (asg.Target is ILocalReferenceOperation t)
                                    RegisterSource(t.Local, asg.Value);
                                break;

                            case OperationKind.DeconstructionAssignment:
                                var deconstruct = (IDeconstructionAssignmentOperation)op;
                                foreach (var target in DeconstructionLocals(deconstruct.Target))
                                    RegisterSource(target, deconstruct.Value);
                                break;
                        }

                        if (ProducesOwnedNDArray(op))
                            _producers.Add(op);
                    }
                }

                // A local that is EVER assigned a plain input (a parameter / field / another local — an
                // alias, not a value this method owns) is dropped from the owning set: disposing it could
                // free something the caller owns (R2). One owning assignment with no alias assignment ⇒ owned.
                _owningLocals.ExceptWith(_aliasedLocals);
                return true;
            }

            /// <summary>
            ///     Reports one leaked value with the text that fits it: the holder text for a disposable that stores
            ///     NDArrays (only its own <c>Dispose</c> reclaims it, so no scope attribute is suggested); for an
            ///     NDArray, the text that offers the scope attribute fitting the enclosing method, or, when the build
            ///     declared the weaver inactive, the text that does not (the attribute would be inert).
            /// </summary>
            /// <param name="loc">Where the value is created: a local's declarator, or the leaking expression.</param>
            /// <param name="method">The method whose body drops the value; decides <c>[NDScoped]</c> versus <c>[NDScopedAsync]</c>.</param>
            /// <param name="type">The value's static type; decides between the texts and how the value is named.</param>
            private void Report(Location loc, IMethodSymbol method, ITypeSymbol type)
            {
                Diagnostic diagnostic;
                if (IsHolderValue(type))
                    diagnostic = Diagnostic.Create(Nw012Holder, loc, DescribeOwned(type));
                else
                    diagnostic = Diagnostic.Create(_weaverInactive ? Nw012NoWeaver : Nw012, loc,
                        EnclosingSuggestion(method), DescribeOwned(type));
                // The driver's generated-code test is off for this analyzer (ReportDiagnostics); this is its copy.
                if (_generated.IsInGeneratedCode(loc, _ctx.Options, _ctx.CancellationToken))
                    return;
                _ctx.ReportDiagnostic(diagnostic);
            }

            /// <summary>
            ///     True for a value that only its own <c>Dispose</c> reclaims: a disposable that stores NDArrays, or an
            ///     array of them — the two kinds <see cref="DescribeOwned"/> names as instances. An NDArray, an
            ///     NDArray carrier, or a tuple of them is not one: a scope reclaims those.
            /// </summary>
            /// <param name="t">The leaked value's static type.</param>
            /// <returns>True when the holder text applies.</returns>
            private bool IsHolderValue(ITypeSymbol t)
                => (t is IArrayTypeSymbol arr && _model.IsNDOwningDisposable(arr.ElementType)) || _model.IsNDOwningDisposable(t);

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

            /// <summary>The use list of <paramref name="local"/>, created on its first use (from the lists earlier blocks emptied, when any).</summary>
            /// <param name="local">A local of the body.</param>
            /// <returns>The list the walk appends the local's value uses to.</returns>
            private List<IOperation> Uses(ILocalSymbol local)
            {
                if (!_localUses.TryGetValue(local, out var list))
                    _localUses[local] = list = _spareUseLists.Count > 0 ? _spareUseLists.Pop() : new List<IOperation>();
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
                    if (IsNDScopeMethod(inv.TargetMethod, ScopeReclaimNames))
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

            /// <summary>
            ///     Whether the method itself carries a scope-family attribute - on the method, or for an accessor on its
            ///     property: a [NDScoped]/[NDScopedAsync] boundary (the weaver reclaims every transient it constructs) or
            ///     an [NDScopedCovered] helper (the author's assertion that a caller's ambient scope reclaims them). The
            ///     INHERITED form of the same exemption is checked by <see cref="Run"/> last, only when a leak would be
            ///     reported: it needs the override and interface walk.
            /// </summary>
            /// <param name="method">The method whose block is analyzed.</param>
            /// <param name="k">NumSharp's types of the compilation (the three attribute classes).</param>
            /// <returns>True when the method is exempt by an attribute of its own.</returns>
            private static bool HasOwnScopeAttribute(IMethodSymbol method, KnownTypes k)
            {
                if (HasScopeAttribute(method, k))
                    return true;
                var carrier = method.AssociatedSymbol; // property accessor → property
                return carrier != null && HasScopeAttribute(carrier, k);
            }

            /// <summary>Whether <paramref name="symbol"/> carries [NDScoped], [NDScopedAsync] or [NDScopedCovered].</summary>
            /// <param name="symbol">A method or property.</param>
            /// <param name="k">NumSharp's types of the compilation (the three attribute classes).</param>
            /// <returns>True when one of the three attributes is present.</returns>
            private static bool HasScopeAttribute(ISymbol symbol, KnownTypes k)
            {
                foreach (var a in symbol.GetAttributes())
                {
                    var cls = a.AttributeClass;
                    if (SymbolEqualityComparer.Default.Equals(cls, k.SyncAttr) ||
                        SymbolEqualityComparer.Default.Equals(cls, k.AsyncAttr) ||
                        SymbolEqualityComparer.Default.Equals(cls, k.CoveredAttr))
                        return true;
                }

                return false;
            }

            /// <summary>
            ///     An owned NDArray carrier value — the kinds whose buffer this method must reclaim:
            ///     NDArray / NDArray[] / a tuple with one / an INDArrayCarrier struct, and — ownership
            ///     being contagious — an NDArray-owning DISPOSABLE (a consumer's holder class, NumSharp's
            ///     np.nditer iterator) or a rank-1 array of them.
            ///     <para>
            ///     Asked for every producing operation (a call, a construction, an operator) and every local of every
            ///     analyzed body, so it is answered in three tiers: kinds of type that are never owned return at once
            ///     (every special type - the primitives, string, object, Nullable and the special collection interfaces -,
            ///     interfaces, enums, delegates, pointers, dynamic: none can be NDArray-like, a tuple, a carrier struct or
            ///     a disposable class or struct that stores NDArrays); then NDArray itself, by identity (the commonest
            ///     owned type by far, and a definition Roslyn keeps one instance of); then the compilation's memo
            ///     (<see cref="OwnershipModel.OwnedValueTypes"/>, by symbol equality, so the fresh instances Roslyn
            ///     creates of one constructed type share an entry), and only a type seen nowhere before runs the
            ///     classification.
            ///     </para>
            /// </summary>
            /// <param name="t">The type of a value; null answers false.</param>
            /// <returns>True when a value of the type is one this method must reclaim.</returns>
            /// <remarks>
            ///     No per-body cache sits in front of the memo: keyed by reference it missed on every fresh instance of a
            ///     constructed type (most of what reaches here once NDArray and the special types are answered above), and
            ///     each miss paid the runtime's first-hash path for the new object plus an insert. Profiled on NumSharp's
            ///     own source, that cache was the most expensive part of the analyzer.
            /// </remarks>
            private bool IsOwnedNDArrayType(ITypeSymbol t)
            {
                if (t == null || t.SpecialType != SpecialType.None)
                    return false;
                switch (t.TypeKind)
                {
                    case TypeKind.Interface:
                    case TypeKind.Enum:
                    case TypeKind.Delegate:
                    case TypeKind.Pointer:
                    case TypeKind.FunctionPointer:
                    case TypeKind.Dynamic:
                        return false;
                }

                // NDArray is the classification's first rule (IsNDArrayCarrying), so this answer is the memo's.
                if (SymbolEqualityComparer.Default.Equals(t, _k.NDArray))
                    return true;

                var memo = _model.OwnedValueTypes;
                if (!memo.TryGetValue(t, out var owned))
                {
                    owned = ComputeIsOwnedNDArrayType(t);
                    memo.TryAdd(t, owned); // a racing writer stores the same answer
                }

                return owned;
            }

            /// <summary>The uncached <see cref="IsOwnedNDArrayType"/>: the classification itself.</summary>
            /// <param name="t">A non-null type.</param>
            /// <returns>True when a value of the type is one this method must reclaim.</returns>
            private bool ComputeIsOwnedNDArrayType(ITypeSymbol t)
            {
                if (TypeHelpers.IsNDArrayCarrying(t, _k))          // NDArray / NDArray[]
                    return true;
                if (t.IsTupleType && TupleHoldsOwned((INamedTypeSymbol)t))
                    return true;
                if (ImplementsCarrier(t))                          // INDArrayCarrier result struct
                    return true;
                if (_model.IsNDOwningDisposable(t))                // a disposable that stores NDArrays
                    return true;
                if (t is IArrayTypeSymbol arr && arr.Rank == 1 && _model.IsNDOwningDisposable(arr.ElementType))
                    return true;
                return false;
            }

            /// <summary>Whether an element of a tuple type is an owned value (a tuple holds every element at once).</summary>
            /// <param name="tuple">A tuple type.</param>
            /// <returns>True when at least one element's type is owned.</returns>
            /// <remarks>
            ///     Up to seven elements, a tuple's element types ARE its ValueTuple type arguments, which Roslyn already
            ///     holds; <c>TupleElements</c> builds a field symbol per element of every tuple instance it is asked about
            ///     (profiled: 7% of the analyzer's time on NumSharp's own source, which returns tuples everywhere). A longer
            ///     tuple nests its rest in an eighth argument, so it still asks <c>TupleElements</c>, which flattens them.
            /// </remarks>
            private bool TupleHoldsOwned(INamedTypeSymbol tuple)
            {
                var arguments = tuple.TypeArguments;
                if (arguments.Length < 8)
                {
                    foreach (var element in arguments)
                        if (IsOwnedNDArrayType(element))
                            return true;
                    return false;
                }

                foreach (var element in tuple.TupleElements)
                    if (IsOwnedNDArrayType(element.Type))
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

            /// <summary>A value type implementing <c>INDArrayCarrier</c> - a scope-yieldable result struct.</summary>
            /// <param name="t">A non-null type.</param>
            /// <returns>True for a carrier struct.</returns>
            private bool ImplementsCarrier(ITypeSymbol t)
            {
                if (_k.INDArrayCarrier == null || !t.IsValueType)
                    return false;
                // INDArrayCarrier is not generic: the definition's set answers for every constructed instance
                // without Roslyn substituting the whole interface set again for each fresh instance it hands out.
                foreach (var i in TypeHelpers.InterfacesForNonGenericTest(t))
                    if (SymbolEqualityComparer.Default.Equals(i, _k.INDArrayCarrier))
                        return true;
                return false;
            }

            /// <summary>An expression that PRODUCES an owned NDArray value (a construction, call, or operator — not a view-y property/indexer read or a plain reference).</summary>
            /// <param name="op">Any operation of the body; null answers false.</param>
            /// <returns>True when the operation creates a value of an owned type that this method then holds.</returns>
            /// <remarks>
            ///     The operation's KIND is tested before its type: only the kinds below can produce an owned value, and
            ///     every arm requires an owned type, so the answer is the one the type-first test gave - but a reference,
            ///     a literal, an argument or a statement (most of a body's operations; the walk asks about every one) is
            ///     answered by one read of its Kind instead of a classification of its value's type. The arms keep the
            ///     order the type patterns they replace were tested in: an array creation without an initializer, and
            ///     every kind not listed first, still reach the collection-expression test before the conditional
            ///     values.
            /// </remarks>
            private bool ProducesOwnedNDArray(IOperation op)
            {
                if (op == null)
                    return false;
                switch (op.Kind)
                {
                    case OperationKind.ObjectCreation:
                    case OperationKind.Binary:
                    case OperationKind.Unary:
                        return IsOwnedNDArrayType(op.Type);
                    case OperationKind.Invocation:
                        // NDScope.Returns/Attach return an NDArray but are reclamation plumbing, not a
                        // fresh temp to flag.
                        return IsOwnedNDArrayType(op.Type) &&
                               !IsNDScopeMethod(((IInvocationOperation)op).TargetMethod, ScopeReclaimNames);

                    // A tuple literal / array creation OWNS whatever its elements own, and it holds ALL
                    // of them at once — so it is a producer iff AT LEAST ONE element is itself a
                    // producer. A tuple/array of plain inputs ((a, b), new[]{ a, b }) owns nothing, and
                    // flagging it would demand reclaiming the caller's values (R2).
                    case OperationKind.Tuple:
                        if (!IsOwnedNDArrayType(op.Type))
                            return false;
                        foreach (var e in ((ITupleOperation)op).Elements)
                            if (ProducesOwnedNDArray(Unwrap(e)))
                                return true;
                        return false;
                    case OperationKind.ArrayCreation:
                        var arr = (IArrayCreationOperation)op;
                        if (arr.Initializer == null)
                            break; // the type-pattern arm required an initializer: on to the collection-expression test
                        if (!IsOwnedNDArrayType(op.Type))
                            return false;
                        foreach (var e in arr.Initializer.ElementValues)
                            if (ProducesOwnedNDArray(Unwrap(e)))
                                return true;
                        return false;
                }

                // The C# 12 collection-expression spelling of the same thing: `NDArray[] g = [a + b];`.
                // Matched by SYNTAX because this analyzer compiles against Roslyn 4.8 (the .NET 8 SDK
                // floor), which has the language feature but not ICollectionExpressionOperation — a 4.8
                // host surfaces the expression as an unrecognized operation, newer hosts as the typed
                // one; both carry the elements as child operations. (A spread element is not a
                // producer, so a spread-only expression stays silent.)
                if (op.Syntax is CollectionExpressionSyntax)
                {
                    if (!IsOwnedNDArrayType(op.Type))
                        return false;
                    foreach (var e in op.ChildOperations)
                        if (ProducesOwnedNDArray(Unwrap(e)))
                            return true;
                    return false;
                }

                // A conditional value (ternary / switch expression / coalesce) holds only ONE of its
                // branches, so it is a producer only when EVERY branch produces — the same
                // conservatism as the if/else local rule, where a single aliasing branch drops the
                // local from the owning set: if any branch may be the caller's value, demanding the
                // result be reclaimed is not safe.
                switch (op.Kind)
                {
                    case OperationKind.Conditional:
                        var cond = (IConditionalOperation)op;
                        return cond.WhenFalse != null && IsOwnedNDArrayType(op.Type) &&
                               BranchProduces(cond.WhenTrue) && BranchProduces(cond.WhenFalse);
                    case OperationKind.SwitchExpression:
                        var sw = (ISwitchExpressionOperation)op;
                        if (!IsOwnedNDArrayType(op.Type) || sw.Arms.Length == 0)
                            return false;
                        foreach (var arm in sw.Arms)
                            if (!BranchProduces(arm.Value))
                                return false;
                        return true;
                    case OperationKind.Coalesce:
                        var coal = (ICoalesceOperation)op;
                        return IsOwnedNDArrayType(op.Type) && BranchProduces(coal.Value) && BranchProduces(coal.WhenNull);
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

            /// <summary>The NDScope methods that open a scope by hand: a body that calls one owns its reclamation and is exempt.</summary>
            private static readonly string[] ScopeOpenNames = { "Open", "OpenOrResume" };

            /// <summary>The NDScope methods that are reclamation plumbing: an NDArray handed to (or returned by) one is accounted for.</summary>
            private static readonly string[] ScopeReclaimNames = { "Returns", "Attach", "Detach" };

            /// <summary>Whether <paramref name="m"/> is one of <c>NDScope</c>'s methods named in <paramref name="names"/>.</summary>
            /// <param name="m">A call's target; null answers false.</param>
            /// <param name="names">One of the static name lists above (a shared array: a <c>params</c> list allocated one per call, and the walk asks for every invocation).</param>
            /// <returns>True for a named NDScope method.</returns>
            private bool IsNDScopeMethod(IMethodSymbol m, string[] names)
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
