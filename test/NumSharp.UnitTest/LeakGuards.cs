using AwesomeAssertions;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Deterministic lifetime assertions for the <c>using</c>-on-intermediate refactors, and the
    ///     record of why this repo does <b>not</b> assert on <c>Process.WorkingSet64</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Twelve tests across eleven files used to run a tight loop of some operation and assert that
    ///     <c>Process.WorkingSet64</c> grew by less than 20-30 MiB, each claiming to guard the
    ///     <c>using</c> on an intermediate buffer. <b>Do not reintroduce that shape.</b> It fails on
    ///     correct code and passes on broken code:
    ///     </para>
    ///     <list type="bullet">
    ///       <item><b>It passes when the leak is present.</b> Running the allclose loop with the
    ///             <c>using</c> on the <c>np.isclose</c> intermediate REMOVED — the exact leak the test
    ///             named — measured 3, 0, 0, 0 and -7 MiB across five trials against a 20 MiB limit.
    ///             It never once tripped, and on one trial the broken code scored 7 MiB BETTER than the
    ///             fixed code. The reading is uncorrelated with the invariant.</item>
    ///       <item><b>It fails when the code is correct.</b> The same assertion failed CI on Ubuntu at
    ///             39 MiB with the <c>using</c> intact, having passed on Windows and macOS in the same
    ///             run — and it passes on Ubuntu in isolation. What it actually samples is whatever the
    ///             other ~11,700 tests left in the shared process.</item>
    ///       <item><b>Several could never trip at all.</b> The quantity guarded is often far below the
    ///             threshold: eye's <c>flat</c> is a view wrapper (~100 B x 1000 iterations vs a 20 MiB
    ///             limit), tile's intermediates are ~10 MiB against a 30 MiB limit, and flatten's own
    ///             comment concedes the buffer "is kept alive by the returned flat anyway".</item>
    ///       <item><b>The probe is the wrong instrument.</b> <c>WorkingSet64</c> is OS-level RSS for the
    ///             whole process. Windows trims it — every loop here measures 0 MiB locally, which is
    ///             why the family looked green for so long — while Linux does not return freed pages on
    ///             <c>GC.Collect</c>, so ordinary allocator churn reads as permanent growth. That is the
    ///             entire Windows/Ubuntu split, and it is a property of the allocator, not of NumSharp.</item>
    ///     </list>
    ///     <para>
    ///     The <c>using</c>s those tests named are correct and still in place. Under-disposal is simply
    ///     not observable from a runtime probe at these sizes, so it rests on review. <b>Over</b>-disposal
    ///     is the hazard the refactor can actually introduce — if an "intermediate" turns out to alias a
    ///     caller's array (as <c>np.asanyarray</c> returns its input unchanged) then <c>using</c> frees
    ///     memory still in use — and that IS deterministically observable, through the ARC refcount.
    ///     <see cref="StillUsable"/> is that assertion, and it is what replaced the working-set tests.
    ///     </para>
    /// </remarks>
    internal static class LeakGuards
    {
        /// <summary>
        ///     Asserts <paramref name="array"/> was not disposed and its buffer was not released — i.e. a
        ///     <c>using</c> on some intermediate did not take this array's memory with it.
        /// </summary>
        /// <remarks>
        ///     Both halves matter. <c>IsDisposed</c> catches the wrapper being disposed; <c>IsReleased</c>
        ///     catches the ARC refcount hitting zero underneath a live wrapper, which is what over-disposing
        ///     an aliased intermediate actually does. The read-back then proves the memory is not merely
        ///     un-freed but still holds the right bytes.
        /// </remarks>
        public static void StillUsable(NDArray array, string because = "")
        {
            array.IsDisposed.Should().BeFalse("the array must survive the call" + because);

            var buffer = array.Storage?.InternalArray;
            if (buffer != null)
                buffer.IsReleased.Should().BeFalse("the array's buffer must not be released" + because);
        }
    }
}
