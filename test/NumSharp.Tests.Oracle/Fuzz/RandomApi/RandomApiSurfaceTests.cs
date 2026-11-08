using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     Gate G1 of the random-API oracle: the committed inventory <c>test/oracle/random_surface.json</c> equals the
    ///     reflected public random surface. Every other random-API gate measures coverage AGAINST this inventory, so a stale
    ///     inventory would let a new overload go uncovered without anything turning red.
    /// </summary>
    [TestClass]
    public class RandomApiSurfaceTests
    {
        /// <summary>
        ///     The committed inventory is exactly what reflection produces now — or, with
        ///     <see cref="RandomApiSurface.WriteVariable"/> set to <c>1</c>, rewrites it from reflection (the one supported
        ///     way to regenerate it, so the file and the gate cannot disagree about the canonical form).
        /// </summary>
        /// <remarks>
        ///     On a mismatch the message lists the signatures added and removed, which is what a reviewer needs to decide
        ///     whether the surface change was intended; then regenerate the inventory and the corpus
        ///     (<c>python test/oracle/gen_random_oracle.py</c>) together.
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void CommittedInventory_EqualsTheReflectedSurface()
        {
            var members = RandomApiSurface.Reflect();
            string json = RandomApiSurface.ToJson(members);
            string path = RandomApiSurface.CommittedPath();

            if (Environment.GetEnvironmentVariable(RandomApiSurface.WriteVariable) == "1")
            {
                File.WriteAllText(path, json);
                Console.WriteLine($"[RandomApiSurface] wrote {members.Count} members to {path}");
                return;
            }

            Assert.IsTrue(File.Exists(path), $"The random surface inventory {path} is missing; regenerate it with " +
                                             $"{RandomApiSurface.WriteVariable}=1.");
            string committed = File.ReadAllText(path).Replace("\r\n", "\n");
            if (committed == json)
                return;

            // Name the difference by signature: the committed file's "sig" values against reflection's.
            var committedSigs = committed.Split('\n')
                .Select(line => line.Trim())
                .Where(line => line.StartsWith("{\"sig\": ", StringComparison.Ordinal))
                .Select(line => System.Text.Json.JsonDocument.Parse(line.TrimEnd(',')).RootElement.GetProperty("sig").GetString())
                .ToHashSet(StringComparer.Ordinal);
            var reflected = members.Select(m => m.Sig).ToHashSet(StringComparer.Ordinal);
            var added = reflected.Except(committedSigs).OrderBy(s => s, StringComparer.Ordinal).ToArray();
            var removed = committedSigs.Except(reflected).OrderBy(s => s, StringComparer.Ordinal).ToArray();
            Assert.Fail($"test/oracle/random_surface.json is stale ({added.Length} added, {removed.Length} removed" +
                        (added.Length + removed.Length == 0 ? ", same signatures: parameter names/defaults/returns changed" : "") +
                        $"). Regenerate it with {RandomApiSurface.WriteVariable}=1, then regenerate the random-API corpus.\n" +
                        "  added:   " + string.Join("\n           ", added.Take(40)) + "\n" +
                        "  removed: " + string.Join("\n           ", removed.Take(40)));
        }
    }
}
