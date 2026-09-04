using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The property-based invariant gate for the ownership diagnostics (the RB-5 technique applied
    ///     to types). A deterministic seeded generator composes TYPES from a grammar of member
    ///     templates (holding and non-holding), makes each disposable or not, and disposes a random
    ///     subset of the holders on the Dispose path with a random spelling. It then asserts:
    ///     <list type="bullet">
    ///         <item>NDW016 count == the number of NON-disposable types with at least one holder;</item>
    ///         <item>NDW017 count == the number of holders of DISPOSABLE types that were NOT disposed;</item>
    ///         <item>no NDW012 at all (the generated bodies create nothing).</item>
    ///     </list>
    ///     Seeded <see cref="Random"/> is stable, so every run replays the same 120 types; a mismatch
    ///     re-runs each type alone and reports the seed and the generated source.
    /// </summary>
    [TestClass]
    public class OwnershipPropertyFuzzTests
    {
        // (member template, holds?, dispose-statement template) — {0} is a unique member suffix.
        private static readonly (string Member, bool Holds, string Dispose)[] Grammar =
        {
            ("private NDArray _f{0};",                                        true,  "_f{0}?.Dispose();"),
            ("private NDArray[] _f{0};",                                      true,  "foreach (var x in _f{0}) x?.Dispose();"),
            ("private List<NDArray> _f{0} = new List<NDArray>();",            true,  "_f{0}.ForEach(x => x.Dispose());"),
            ("private Dictionary<string, NDArray> _f{0};",                    true,  "foreach (var kv in _f{0}) kv.Value.Dispose();"),
            ("private (NDArray a, int n) _f{0};",                             true,  "_f{0}.a?.Dispose();"),
            ("public NDArray P{0} {{ get; set; }}",                           true,  "P{0}?.Dispose();"),
            ("private Lazy<NDArray> _f{0};",                                  true,  "if (_f{0} != null && _f{0}.IsValueCreated) _f{0}.Value.Dispose();"),
            ("private Owner _f{0};",                                          true,  "_f{0}?.Dispose();"),
            ("private int _f{0};",                                            false, ""),
            ("private Func<NDArray> _f{0};",                                  false, ""),
            ("[NDBorrowed] private NDArray _f{0};",                           false, ""),
            ("private static NDArray _f{0};",                                 false, ""),
            ("private IComparer<NDArray> _f{0};",                             false, ""),
            ("public NDArray V{0} => null;",                                  false, ""),
        };

        private sealed class Generated
        {
            public int Seed;
            public string Source;
            public int Expected16;
            public int Expected17;
        }

        private static Generated Generate(int seed)
        {
            var rng = new Random(seed);
            bool disposable = rng.Next(2) == 0;
            int count = 1 + rng.Next(6); // 1..6 members
            var members = new StringBuilder();
            var disposes = new StringBuilder();
            int holders = 0, disposed = 0;
            for (int i = 0; i < count; i++)
            {
                var (member, holds, dispose) = Grammar[rng.Next(Grammar.Length)];
                members.Append("    ").AppendFormat(member, seed + "_" + i).AppendLine();
                if (!holds)
                    continue;
                holders++;
                if (disposable && rng.Next(2) == 0)
                {
                    disposed++;
                    disposes.Append("        ").AppendFormat(dispose, seed + "_" + i).AppendLine();
                }
            }

            var src = new StringBuilder();
            src.AppendLine("using System;\nusing System.Collections.Generic;\nusing NumSharp;");
            src.AppendLine("public class Owner : IDisposable { NDArray _o; public void Dispose() => _o?.Dispose(); }");
            src.Append("public class T").Append(seed).Append(disposable ? " : IDisposable" : "").AppendLine(" {");
            src.Append(members);
            if (disposable)
            {
                // Route the dispose calls through a randomly chosen shape: direct, Dispose(bool), or a helper.
                int shape = rng.Next(3);
                switch (shape)
                {
                    case 0:
                        src.AppendLine("    public void Dispose() {").Append(disposes).AppendLine("    }");
                        break;
                    case 1:
                        src.AppendLine("    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }");
                        src.AppendLine("    protected virtual void Dispose(bool disposing) { if (!disposing) return;").Append(disposes).AppendLine("    }");
                        break;
                    default:
                        src.AppendLine("    public void Dispose() => Release();");
                        src.AppendLine("    private void Release() {").Append(disposes).AppendLine("    }");
                        break;
                }
            }
            src.AppendLine("}");

            return new Generated
            {
                Seed = seed,
                Source = src.ToString(),
                Expected16 = !disposable && holders > 0 ? 1 : 0,
                Expected17 = disposable ? holders - disposed : 0,
            };
        }

        [TestMethod]
        public async Task RandomizedTypes_FlagExactlyTheUndisposedHolders()
        {
            const int seeds = 120;
            int total16 = 0, total17 = 0;
            var all = new List<Generated>();
            for (int seed = 0; seed < seeds; seed++)
            {
                var g = Generate(seed);
                all.Add(g);
                total16 += g.Expected16;
                total17 += g.Expected17;
            }
            Assert.IsTrue(total16 > 0 && total17 > 0, "the grammar must have produced both kinds of finding (non-vacuous)");

            // Every type is compiled ALONE (each carries its own Owner helper), so a mismatch names its seed.
            foreach (var g in all)
            {
                var r = await AnalyzerTestHarness.RunAsync(g.Source, $"ownership_fuzz_seed{g.Seed}.cs");
                Assert.IsTrue(r.CompileErrors.IsEmpty, $"seed {g.Seed}: generated source must compile:\n{g.Source}\n{string.Join("\n", r.CompileErrors)}");
                Assert.AreEqual(g.Expected16, r.CountOf("NDW016"), $"seed {g.Seed}: NDW016 for:\n{g.Source}");
                Assert.AreEqual(g.Expected17, r.CountOf("NDW017"), $"seed {g.Seed}: NDW017 for:\n{g.Source}");
                Assert.AreEqual(0, r.CountOf("NDW012"), $"seed {g.Seed}: nothing in a generated type leaks:\n{g.Source}");
            }
        }
    }
}
