using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     STATIC pool-bypass gate, complementing the runtime one in UndisposedIntermediateTests:
    ///     the runtime sweep can only see a bypass whose buffer reaches the RESULT (fresh array,
    ///     zero pool traffic) — an internal scratch buffer allocated raw and freed raw inside an
    ///     op is invisible to it by construction. This gate closes that hole at the source level:
    ///     every raw native-allocation call site in NumSharp.Core
    ///     (NativeMemory.Alloc/AllocZeroed/AllocAligned/Realloc, Marshal.AllocHGlobal,
    ///     VirtualAlloc*) must be a known chokepoint. The allowlist pins file → site count, so a
    ///     NEW raw allocation anywhere — a new file, or a new site in an allowed file — fails
    ///     until it is either routed through SizeBucketedBufferPool / StackedMemoryPool or
    ///     consciously added here with a reason.
    ///
    ///     Comment-only mentions are ignored (lines whose first token is a comment marker); the
    ///     scan needs the SOURCE tree, so a bin-only/CI-package run reports Inconclusive rather
    ///     than a false green.
    /// </summary>
    [TestClass]
    public class NativeAllocationChokepointTests
    {
        /// <summary>
        ///     file (relative to src/NumSharp.Core, forward slashes) → exact raw-allocation site
        ///     count at gate landing (2026-08-26). The two pools and the guard-page allocator ARE
        ///     the chokepoints; NDIter's buffered-mode scratch and bincount's counting table are
        ///     internal alloc+free pairs (invisible to the runtime sweep — the reason this gate
        ///     exists) carried as audit debt to route through the pool.
        /// </summary>
        private static readonly Dictionary<string, int> Allowlist = new(StringComparer.Ordinal)
        {
            ["Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs"] = 4,  // the bucketed pool itself
            ["Backends/Unmanaged/Pooling/StackedMemoryPool.cs"] = 1,       // the scalar pool itself
            ["Backends/Unmanaged/Pooling/OsVirtualMemory.cs"] = 5,         // guard-page allocator (env-opt-in, pool-routed)
            ["Backends/Iterators/NDIter.cs"] = 2,                          // audit debt: iterator scratch, alloc+free internal
            ["Backends/Iterators/NDIter.State.cs"] = 2,                    // audit debt: iterator state blocks
            ["Backends/Iterators/NDIterBufferManager.cs"] = 3,             // audit debt: buffered-mode chunk buffers
            ["Sorting_Searching_Counting/np.bincount.cs"] = 1,             // audit debt: privatized counting table
            ["Backends/Default/LinearAlgebra/ManagedLu.cs"] = 2,           // audit debt: LU scratch (factor copy + pivot vector), alloc+free per call in try/finally — landed 48b00e00 without this pin
            ["Backends/Default/Sorting/AxisSort.cs"] = 8,                  // audit debt: radix key/temp/histogram + argsort index columns, alloc+free per line in try/finally — UNMANAGED so a line may exceed int.MaxValue (64-bit sort core, ab15b165); the pool is int-capped
            ["Backends/Default/Sorting/AxisPartition.cs"] = 4,             // audit debt: introselect line scratch (+NaN tail) + argpartition index column, alloc+free per line in try/finally — same 64-bit >int.MaxValue reason as AxisSort
        };

        private static readonly Regex RawAlloc = new(
            @"NativeMemory\s*\.\s*(Alloc|AllocZeroed|AllocAligned|Realloc)\b" +
            @"|Marshal\s*\.\s*AllocHGlobal\b" +
            @"|\bVirtualAlloc\w*\s*\(",
            RegexOptions.Compiled);

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [TestCategory("ScopeAudit")]
        public void EveryRawNativeAllocation_IsAKnownChokepoint()
        {
            string core = FindCoreSourceDir();
            if (core == null)
            {
                Assert.Inconclusive("src/NumSharp.Core source tree not found relative to the test " +
                                    "sources — static allocation scan needs a repo checkout.");
                return;
            }

            var found = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var path in Directory.EnumerateFiles(core, "*.cs", SearchOption.AllDirectories))
            {
                int count = 0;
                foreach (var line in File.ReadLines(path))
                {
                    var t = line.TrimStart();
                    if (t.StartsWith("//", StringComparison.Ordinal) ||
                        t.StartsWith("*", StringComparison.Ordinal))
                        continue;   // doc/comment-only mentions are not call sites
                    if (RawAlloc.IsMatch(line))
                        count++;
                }
                if (count > 0)
                    found[Path.GetRelativePath(core, path).Replace('\\', '/')] = count;
            }

            // Non-vacuity: the pools themselves must be visible to the scan.
            Assert.IsTrue(found.ContainsKey("Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs"),
                "scan found no raw allocations even in the pool itself — regex or path regression?");

            var problems = new List<string>();
            foreach (var kv in found)
            {
                if (!Allowlist.TryGetValue(kv.Key, out int allowed))
                    problems.Add($"NEW raw-allocation file: {kv.Key} ({kv.Value} site(s)) — route through " +
                                 "SizeBucketedBufferPool/StackedMemoryPool or allowlist with a reason");
                else if (kv.Value > allowed)
                    problems.Add($"{kv.Key}: {kv.Value} raw-allocation sites, allowlist pins {allowed} — " +
                                 "a new site appeared; route it through a pool or bump the pin with a reason");
            }
            foreach (var kv in Allowlist)
                if (!found.TryGetValue(kv.Key, out int n) || n < kv.Value)
                    Console.WriteLine($"[chokepoint] {kv.Key}: sites dropped below the pinned {kv.Value} — " +
                                      "tighten the allowlist (progress!)");

            if (problems.Count > 0)
                Assert.Fail($"{problems.Count} unexpected raw native-allocation site(s):\n  " +
                            string.Join("\n  ", problems));
        }

        /// <summary>Walk up from this source file to the repo root and return src/NumSharp.Core.</summary>
        private static string FindCoreSourceDir([CallerFilePath] string thisFile = null)
        {
            for (var dir = Path.GetDirectoryName(thisFile); dir != null; dir = Path.GetDirectoryName(dir))
            {
                var candidate = Path.Combine(dir, "src", "NumSharp.Core");
                if (Directory.Exists(candidate))
                    return candidate;
            }
            return null;
        }
    }
}
