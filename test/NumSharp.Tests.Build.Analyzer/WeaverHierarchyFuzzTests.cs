using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Property-based gate for declaration-site inheritance: a seeded generator builds random class
    ///     chains (2–5 levels, 1–4 virtual/abstract members, one interface member implemented somewhere
    ///     in the chain) with random <c>[NDScoped]</c>/<c>[NDScopedCovered]</c> placement on the
    ///     declarations AND on random overrides, and an independent REFERENCE MODEL of the rule —
    ///     "the nearest declaration up the chain carrying a scope-family attribute wins; the
    ///     interface's attribute is the fallback for the member that implements it; only a body
    ///     under <c>[NDScoped]</c> is woven" — predicts, per generated method, whether the weaver
    ///     weaves it and whether the leak analyzer exempts it. Both are then asserted against the real
    ///     weave (a scope local per method) and the real analyzer (an NDW012 exactly on the dead-temp
    ///     line of every method the model calls unscoped). Seeded <see cref="Random"/> is stable, so
    ///     every run replays the same hierarchies; a failure prints the seed and the generated source.
    /// </summary>
    [TestClass]
    public class WeaverHierarchyFuzzTests
    {
        private enum Attr { None, Scoped, Covered }

        private sealed class GenMethod
        {
            public string Type;
            public string Name;
            public bool HasBody;
            public Attr Own;
            public Attr Effective;
            public int Line;
            public override string ToString() => $"{Type}.{Name} (own {Own}, effective {Effective}, body {HasBody})";
        }

        private static string Spell(Attr a) => a switch
        {
            Attr.Scoped => "[NDScoped] ",
            Attr.Covered => "[NDScopedCovered] ",
            _ => "",
        };

        private static Attr Pick(Random rng, int noneWeight, int scopedWeight, int coveredWeight)
        {
            int r = rng.Next(noneWeight + scopedWeight + coveredWeight);
            return r < noneWeight ? Attr.None : r < noneWeight + scopedWeight ? Attr.Scoped : Attr.Covered;
        }

        /// <summary>One generated hierarchy: the source (one method per line) and the model's verdict per method.</summary>
        private static (string Source, List<GenMethod> Methods) Generate(int seed)
        {
            var rng = new Random(seed);
            int levels = 2 + rng.Next(4);         // 2..5 classes in the chain
            int members = 1 + rng.Next(4);        // 1..4 chain members M0..
            int implementer = rng.Next(levels);   // the level that lists IApply and declares Apply
            var interfaceAttr = Pick(rng, 2, 2, 1);

            var sb = new StringBuilder("using System;\nusing NumSharp;\n");
            var methods = new List<GenMethod>();
            int line = 3;

            sb.AppendLine($"public interface IApply {{ {Spell(interfaceAttr)}NDArray Apply(NDArray a); }}");
            line++;

            // declared[level][member] = the attribute that level's declaration carries (null = not declared there)
            var declared = new Attr?[levels, members + 1];   // column `members` is Apply
            const string body = "{ var t = a + 1.0; return a.copy(); }";

            for (int level = 0; level < levels; level++)
            {
                string bases = level == 0 ? "" : $" : C{level - 1}";
                if (level == implementer)
                    bases += level == 0 ? " : IApply" : ", IApply";
                sb.AppendLine($"public abstract class C{level}{bases}");
                line++;
                sb.AppendLine("{");
                line++;

                for (int j = 0; j < members; j++)
                {
                    bool declare = level == 0 || rng.Next(10) < 6;
                    if (!declare)
                        continue;
                    var own = level == 0 ? Pick(rng, 4, 4, 2) : Pick(rng, 7, 2, 1);
                    bool abstractRoot = level == 0 && rng.Next(2) == 0;
                    declared[level, j] = own;
                    string decl = level == 0
                        ? abstractRoot ? $"    {Spell(own)}public abstract NDArray M{j}(NDArray a);"
                                       : $"    {Spell(own)}public virtual NDArray M{j}(NDArray a) {body}"
                        : $"    {Spell(own)}public override NDArray M{j}(NDArray a) {body}";
                    sb.AppendLine(decl);
                    methods.Add(new GenMethod { Type = $"C{level}", Name = $"M{j}", HasBody = !abstractRoot, Own = own, Effective = EffectiveOf(declared, level, j, Attr.None), Line = line });
                    line++;
                }

                // The interface member: declared (virtual) at the implementer level, overridable below it.
                if (level == implementer || (level > implementer && rng.Next(10) < 6))
                {
                    var own = Pick(rng, 7, 2, 1);
                    declared[level, members] = own;
                    string decl = level == implementer
                        ? $"    {Spell(own)}public virtual NDArray Apply(NDArray a) {body}"
                        : $"    {Spell(own)}public override NDArray Apply(NDArray a) {body}";
                    sb.AppendLine(decl);
                    methods.Add(new GenMethod { Type = $"C{level}", Name = "Apply", HasBody = true, Own = own, Effective = EffectiveOf(declared, level, members, interfaceAttr), Line = line });
                    line++;
                }

                sb.AppendLine("}");
                line++;
            }

            return (sb.ToString(), methods);
        }

        /// <summary>The reference model: nearest attributed declaration at or above <paramref name="level"/>, else the interface's attribute (for Apply), else none.</summary>
        private static Attr EffectiveOf(Attr?[,] declared, int level, int member, Attr fallback)
        {
            for (int l = level; l >= 0; l--)
                if (declared[l, member] is { } a && a != Attr.None)
                    return a;
            return fallback;
        }

        private const int Seeds = 40;

        [TestMethod]
        [Timeout(300000)]
        public async Task RandomHierarchies_WeaveAndExempt_ExactlyWhatTheModelPredicts()
        {
            int hierarchies = 0, methodsChecked = 0, inheritedWoven = 0;
            for (int seed = 0; seed < Seeds; seed++)
            {
                var (source, methods) = Generate(seed);
                var context = $"seed {seed}:\n{source}";

                // The weaver.
                var run = WeaverTestHarness.CompileAndWeave(source, "Fuzz" + seed);
                Assert.AreEqual(0, run.Result.Errors, $"{context}\n{run.Report}");
                using (var asm = run.ReadCecil())
                {
                    int expectedWoven = 0;
                    foreach (var m in methods)
                    {
                        bool expected = m.HasBody && m.Effective == Attr.Scoped;
                        var def = WeaveRun.Find(asm.MainModule, m.Type, m.Name);
                        Assert.AreEqual(expected, WeaveRun.HasScopeLocal(def), $"{context}\n{m}: woven?");
                        if (expected)
                        {
                            expectedWoven++;
                            if (m.Own == Attr.None)
                                inheritedWoven++;
                        }

                        methodsChecked++;
                    }

                    Assert.AreEqual(expectedWoven, run.Result.Woven, $"{context}\nwoven count\n{run.Report}");
                }

                // The analyzer: exactly one NDW012 on the dead-temp line of every unscoped body, nothing else.
                var analyzed = await AnalyzerTestHarness.RunAsync(source, $"fuzz{seed}.cs");
                Assert.IsTrue(analyzed.CompileErrors.IsEmpty, context);
                var expectedLines = methods.Where(m => m.HasBody && m.Effective == Attr.None).Select(m => m.Line).OrderBy(l => l).ToList();
                var actualLines = analyzed.Ndw.Where(d => d.Id == "NDW012").Select(AnalyzerTestHarness.LineOf).OrderBy(l => l).ToList();
                CollectionAssert.AreEqual(expectedLines, actualLines, $"{context}\nNDW012 lines: expected [{string.Join(",", expectedLines)}] actual [{string.Join(",", actualLines)}]");
                Assert.AreEqual(0, analyzed.Ndw.Count(d => d.Id != "NDW012"), $"{context}\nunexpected non-NDW012 diagnostics: {string.Join(", ", analyzed.Ndw.Select(d => d.Id + "@L" + AnalyzerTestHarness.LineOf(d)))}");

                hierarchies++;
            }

            Assert.AreEqual(Seeds, hierarchies);
            Assert.IsTrue(methodsChecked >= Seeds * 3, $"the generator should produce a real surface (checked {methodsChecked} methods)");
            Assert.IsTrue(inheritedWoven >= Seeds, $"the generator should produce plenty of INHERITED weave targets (saw {inheritedWoven})");
        }

        [TestMethod]
        public void Generator_IsDeterministic_AndCoversEveryAttributePlacement()
        {
            var a = Generate(7);
            var b = Generate(7);
            Assert.AreEqual(a.Source, b.Source, "the same seed must replay the same hierarchy");

            var all = Enumerable.Range(0, Seeds).SelectMany(s => Generate(s).Methods).ToList();
            Assert.IsTrue(all.Any(m => !m.HasBody && m.Own == Attr.Scoped), "an abstract [NDScoped] contract");
            Assert.IsTrue(all.Any(m => m.HasBody && m.Own == Attr.None && m.Effective == Attr.Scoped), "an override inheriting [NDScoped]");
            Assert.IsTrue(all.Any(m => m.HasBody && m.Own == Attr.None && m.Effective == Attr.Covered), "an override inheriting [NDScopedCovered]");
            Assert.IsTrue(all.Any(m => m.HasBody && m.Own == Attr.Covered && m.Type != "C0"), "an override opting out with its own [NDScopedCovered]");
            Assert.IsTrue(all.Any(m => m.HasBody && m.Own == Attr.Scoped && m.Type != "C0"), "an override re-stating [NDScoped]");
            Assert.IsTrue(all.Any(m => m.Name == "Apply" && m.Own == Attr.None && m.Effective == Attr.Scoped), "an interface member's [NDScoped] reaching an implementation");
            Assert.IsTrue(all.Any(m => m.HasBody && m.Effective == Attr.None), "unscoped controls");
        }
    }
}
