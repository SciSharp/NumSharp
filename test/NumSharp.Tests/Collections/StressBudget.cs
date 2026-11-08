using System;

namespace NumSharp.Tests.Collections
{
    /// <summary>
    ///     Chooses between the AUTHORED ("golden") workload of a concurrent-collection storm and its per-push CI
    ///     workload. Every storm spells both values at its call site —
    ///     <c>StressBudget.Pick(full: 20_000, ci: 4_000)</c> — so the golden budget stays in the code verbatim
    ///     and one environment variable restores it exactly.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why two budgets.</b> The storms were sized to drain inside their 2-minute join ceiling, not
    ///     for a per-push gate, and their cost is roughly <c>operations × live collection size</c>: an ordered
    ///     removal compacts into FRESH arrays (the copy-on-write that keeps lock-free readers consistent), so
    ///     every one is O(n), and readers enumerate O(n) snapshots in a loop while the writers run. At the golden
    ///     budget the Collections suites took 121 s of a 257 s local run (net10.0, 2026-09-24) and 99–124 s per
    ///     target framework on CI's ubuntu and windows legs (run 35892546210) — close to half of
    ///     <c>NumSharp.Tests</c>' ~16,300 tests, spent on the ~180 in this folder.
    ///     </para>
    ///     <para><b>What the CI budget keeps.</b> Thread counts are untouched (contention is the point), and so is
    ///     every scenario's operation MIX and every assertion. The CI budget shrinks the collection (key
    ///     universes, seeds, partitions) and trims the operation count: the racy interleavings these storms hunt
    ///     — a reader overlapping a compaction, a swap-back or a growth — happen per OPERATION, and a smaller
    ///     collection makes each O(n) operation cheaper without making it less racy (readers finish more passes
    ///     per compaction, not fewer).</para>
    ///     <para><b>Where the golden budget still runs.</b> The nightly <c>collections-stress</c> job in
    ///     <c>.github/workflows/fuzz-soak.yml</c> sets <see cref="EnvironmentVariable"/> to <c>full</c>; so can a
    ///     developer: <c>NUMSHARP_TEST_STRESS=full dotnet test --filter "FullyQualifiedName~Collections"</c>.</para>
    ///     <para><b>Wall-clock storms.</b> The time-boxed race hunts (the two bounded storms in
    ///     <c>OrderedDictionaryContractTests</c>, the compact type's three swap-back guns) pick their DURATION here
    ///     too, but only after their per-push windows were re-calibrated against the races they pin: each known
    ///     defect was re-introduced and every per-push run still failed, pinned to 4 CPUs and unpinned (the numbers
    ///     are on each storm). A new time-boxed storm needs the same calibration before it gets a short window.</para>
    /// </remarks>
    internal static class StressBudget
    {
        /// <summary>The environment variable that selects the golden budget when set to <c>full</c> (case-insensitive).</summary>
        public const string EnvironmentVariable = "NUMSHARP_TEST_STRESS";

        /// <summary>
        ///     True when the golden (authored) budgets are selected, read once per process. Any value other than
        ///     <c>full</c> — including unset — selects the CI budgets, so a typo can only make a run cheaper, never
        ///     silently heavier.
        /// </summary>
        public static bool Full { get; } =
            string.Equals(Environment.GetEnvironmentVariable(EnvironmentVariable), "full", StringComparison.OrdinalIgnoreCase);

        /// <summary>Returns the storm parameter for the selected budget.</summary>
        /// <param name="full">The authored ("golden") value — what the storm ran with before per-push budgets existed.</param>
        /// <param name="ci">The per-push value; must preserve every structural precondition the storm relies on
        /// (e.g. a key band that must still fit inside a seeded range), which is the call site's responsibility.</param>
        /// <returns><paramref name="full"/> under <c>NUMSHARP_TEST_STRESS=full</c>, otherwise <paramref name="ci"/>.</returns>
        public static int Pick(int full, int ci) => Full ? full : ci;
    }
}
