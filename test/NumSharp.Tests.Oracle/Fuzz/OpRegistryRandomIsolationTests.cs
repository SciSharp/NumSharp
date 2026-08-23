using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    [TestClass]
    [DoNotParallelize]
    public class OpRegistryRandomIsolationTests
    {
        [TestMethod]
        public void StatefulRandomOperations_DoNotMutateGlobalRandomState()
        {
            NativeRandomState original = np.random.get_state();
            int originalSeed = np.random.Seed;
            try
            {
                np.random.seed(123456789u);
                NativeRandomState expected = np.random.get_state();
                int expectedSeed = np.random.Seed;

                void AssertGlobalStateUnchanged(string operation)
                {
                    AssertStateEqual(expected, np.random.get_state(), operation);
                    Assert.AreEqual(expectedSeed, np.random.Seed, operation);
                }

                using (var doc = JsonDocument.Parse(
                           "{\"seed\":4294967295,\"size\":[3]}"))
                {
                    _ = OpRegistry.Apply("seed", Params(doc), System.Array.Empty<NDArray>());
                }
                AssertGlobalStateUnchanged("seed");

                using (var doc = JsonDocument.Parse(
                           "{\"dist\":\"uniform\",\"seed\":4294967295,\"size\":[3],\"args\":[0.0,1.0]}"))
                {
                    _ = OpRegistry.Apply("rnd", Params(doc), System.Array.Empty<NDArray>());
                }
                AssertGlobalStateUnchanged("rnd");

                var stateSource = np.random.RandomState(42).get_state();
                using (var doc = JsonDocument.Parse(
                           "{\"pos\":624,\"has_gauss\":0,\"cached_gaussian\":0.0,\"size\":[3]}"))
                {
                    _ = OpRegistry.Apply("set_state", Params(doc),
                        new[] { np.array(stateSource.Key) });
                }
                AssertGlobalStateUnchanged("set_state");

                using (var doc = JsonDocument.Parse(
                           "{\"seed\":4294967295,\"draws\":3}"))
                {
                    _ = OpRegistry.ApplyText("get_state", Params(doc), System.Array.Empty<NDArray>());
                }
                AssertGlobalStateUnchanged("get_state");

                foreach (string corpus in new[] { "random_parity.jsonl", "random_parity_host.jsonl" })
                {
                    foreach (var c in FuzzCorpus.Load(corpus).Where(c => c.Op == "rnd"))
                    {
                        var operands = c.Operands.Select(FuzzCorpus.Reconstruct).ToArray();
                        _ = OpRegistry.Apply(c.Op, c.Params, operands);
                        AssertGlobalStateUnchanged(c.Id);
                    }
                }

                string[] additionalDistributions =
                {
                    "{\"dist\":\"binomial\",\"seed\":42,\"size\":[3],\"args\":[10,0.35]}",
                    "{\"dist\":\"f\",\"seed\":42,\"size\":[3],\"args\":[5.0,10.0]}",
                    "{\"dist\":\"multinomial\",\"seed\":42,\"size\":[3],\"args\":[20],\"pvals\":[0.2,0.3,0.5]}",
                    "{\"dist\":\"multivariate_normal\",\"seed\":42,\"size\":[2],\"args\":[],\"mean\":[0.0,1.0],\"cov\":[1.0,0.0,0.0,1.0]}",
                    "{\"dist\":\"negative_binomial\",\"seed\":42,\"size\":[3],\"args\":[5.0,0.4]}",
                    "{\"dist\":\"pareto\",\"seed\":42,\"size\":[3],\"args\":[3.0]}",
                    "{\"dist\":\"standard_cauchy\",\"seed\":42,\"size\":[3],\"args\":[]}",
                };
                foreach (string payload in additionalDistributions)
                {
                    using var doc = JsonDocument.Parse(payload);
                    _ = OpRegistry.Apply("rnd", Params(doc), System.Array.Empty<NDArray>());
                    AssertGlobalStateUnchanged(doc.RootElement.GetProperty("dist").GetString());
                }
            }
            finally
            {
                np.random.set_state(original);
                np.random.Seed = originalSeed;
            }
        }

        private static IReadOnlyDictionary<string, JsonElement> Params(JsonDocument doc)
            => doc.RootElement.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);

        private static void AssertStateEqual(
            NativeRandomState expected, NativeRandomState actual, string operation)
        {
            Assert.AreEqual(expected.Algorithm, actual.Algorithm, operation);
            Assert.AreEqual(expected.Pos, actual.Pos, operation);
            Assert.AreEqual(expected.HasGauss, actual.HasGauss, operation);
            Assert.AreEqual(expected.CachedGaussian, actual.CachedGaussian, operation);
            CollectionAssert.AreEqual(expected.Key, actual.Key, operation);
        }
    }
}
