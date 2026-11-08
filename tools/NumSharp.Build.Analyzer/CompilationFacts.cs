using System;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.CodeAnalysis;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     What the three NumSharp analyzers know about one compilation - the resolved NumSharp symbols, the
    ///     <see cref="NumSharpNameIndex"/>, the <see cref="OwnershipModel"/> and the <see cref="GeneratedCodeFilter"/> -
    ///     built ONCE and shared by all three.
    ///     <para>
    ///     Without the sharing each analyzer built its own model (two ownership models bound the same members twice) and
    ///     would build its own name index. The facts are keyed weakly by the compilation, so an IDE that creates a new
    ///     compilation on every edit drops the old facts with the old compilation; a command-line build has one.
    ///     </para>
    ///     <para>
    ///     <b>Waiting.</b> The first analyzer to ask builds the facts; another that asks meanwhile blocks until they are
    ///     published (a <see cref="Lazy{T}"/> with execution-and-publication safety), which costs one build, not one per
    ///     analyzer. The build is syntax plus metadata type names, a few milliseconds per thousand files.
    ///     </para>
    /// </summary>
    internal sealed class CompilationFacts
    {
        /// <summary>The facts of every live compilation that NumSharp's analyzers have seen.</summary>
        private static readonly ConditionalWeakTable<Compilation, Lazy<CompilationFacts>> s_facts =
            new ConditionalWeakTable<Compilation, Lazy<CompilationFacts>>();

        /// <summary>The NumSharp symbols of the compilation.</summary>
        public KnownTypes Known { get; }

        /// <summary>The compilation's name index (the pay-for-play gate).</summary>
        public NumSharpNameIndex Names { get; }

        /// <summary>The compilation's ownership model (shared: its memos serve both the leak and the holder analyzer).</summary>
        public OwnershipModel Model { get; }

        /// <summary>
        ///     The compilation's generated-code test for a diagnostic's location: every analyzer asks it before reporting,
        ///     because they register with <c>GeneratedCodeAnalysisFlags.ReportDiagnostics</c> to skip the driver's own,
        ///     much costlier test (see <see cref="GeneratedCodeFilter"/>).
        /// </summary>
        public GeneratedCodeFilter Generated { get; }

        /// <summary>Builds the facts for one compilation.</summary>
        /// <param name="known">Its resolved NumSharp symbols (non-null, with NDArray resolved).</param>
        /// <param name="compilation">The compilation.</param>
        private CompilationFacts(KnownTypes known, Compilation compilation)
        {
            Known = known;
            Names = new NumSharpNameIndex(compilation, known);
            Model = new OwnershipModel(known, compilation, Names);
            Generated = new GeneratedCodeFilter(compilation);
        }

        /// <summary>
        ///     The facts of <paramref name="compilation"/>, built on the first call; null when the compilation does not
        ///     reference NumSharp (no NDScoped attribute or no NDArray resolves), the case in which every analyzer has
        ///     nothing to do.
        /// </summary>
        /// <param name="compilation">The compilation an analyzer's compilation-start action received.</param>
        /// <returns>The shared facts, or null.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="compilation"/> is null.</exception>
        public static CompilationFacts Get(Compilation compilation)
        {
            if (compilation == null)
                throw new ArgumentNullException(nameof(compilation));

            var lazy = s_facts.GetValue(compilation, c => new Lazy<CompilationFacts>(() =>
            {
                var known = KnownTypes.Resolve(c);
                return known == null || known.NDArray == null ? null : new CompilationFacts(known, c);
            }, LazyThreadSafetyMode.ExecutionAndPublication));
            return lazy.Value;
        }
    }
}
