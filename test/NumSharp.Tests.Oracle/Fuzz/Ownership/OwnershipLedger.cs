using System;
using System.Collections.Generic;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Fuzz.Ownership
{
    /// <summary>
    ///     The ways a call can damage an array it does not own (its caller's operand, out=, mask, index or value array, a
    ///     fixture, a process-wide singleton). The first four are OVER-DISPOSALS; <see cref="ReferenceRetained"/> is the
    ///     opposite direction and is reported apart.
    /// </summary>
    internal enum OwnershipViolation
    {
        /// <summary>The call disposed the caller's array object: its <see cref="NDArray.IsDisposed"/> flag is set, so the
        ///     caller's own <c>Dispose</c> is a no-op and the caller's next read may land in a buffer the pool already
        ///     handed to someone else.</summary>
        Disposed,

        /// <summary>The memory behind the caller's array was released (pooled, freed, unpinned or unmapped) while the
        ///     caller still holds the array: every later read or write through it is a use-after-free.</summary>
        Released,

        /// <summary>The memory's counted references fell below the snapshot: the call released a reference it never
        ///     took, so the memory is freed as soon as the caller drops its OWN reference, while views of it may live on.</summary>
        ReferenceDropped,

        /// <summary>The caller's array now refers to different memory (its storage's buffer was swapped): the call
        ///     replaced data the caller owns.</summary>
        StorageReplaced,

        /// <summary>The buffer pool holds memory inside the caller's array (see <see cref="PoolScan"/>): the next allocation
        ///     of that size is handed the caller's bytes, so two owners write them. The caller's own block may look intact
        ///     (another block returned the memory), which is why this is a kind of its own.</summary>
        PooledWhileHeld,

        /// <summary>The memory's counted references are ABOVE the snapshot after the call's result was disposed: a
        ///     reference to the caller's memory outlived the call (an undisposed alias that only the GC reclaims). Not an
        ///     over-disposal: the caller's memory lives longer than it should, it is never freed early.</summary>
        ReferenceRetained,
    }

    /// <summary>One violation found by <see cref="OwnershipLedger.Check"/>.</summary>
    /// <param name="Kind">What happened.</param>
    /// <param name="Role">Which caller array it happened to (<c>operand 0</c>, <c>mask 1</c>, <c>fixture M</c>,
    ///     <c>singleton np.ma.nomask</c>, …).</param>
    /// <param name="Detail">The state before and after, for the report.</param>
    internal sealed record OwnershipFinding(OwnershipViolation Kind, string Role, string Detail)
    {
        /// <summary>True for the over-disposal kinds (everything but <see cref="OwnershipViolation.ReferenceRetained"/>).</summary>
        public bool IsOverDisposal => Kind != OwnershipViolation.ReferenceRetained;

        /// <inheritdoc/>
        public override string ToString() => $"{Kind} ({Role}): {Detail}";
    }

    /// <summary>
    ///     A snapshot of the arrays a call does NOT own, taken before the call, that tells after it whether the call
    ///     disposed, released, swapped or retained any of them.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a snapshot and not the pool balance.</b> <see cref="NDArray.Dispose"/> is idempotent and every leak-sweep
    ///     family runs one UNMEASURED warm call before it measures. An op that disposes its caller's array therefore returns
    ///     the buffer during the warm call; the measured calls then run over a disposed operand and balance their own takes
    ///     and returns, and the caller's later <c>Dispose</c> does nothing. The pool balance never moves, so the leak sweep
    ///     cannot see an over-disposal. The ledger reads the ownership state itself, right after the warm call.
    ///     </para>
    ///     <para>
    ///     <b>What one entry records.</b> The array object, its storage's slice at snapshot time, whether it was already
    ///     disposed, and that slice's ARC state (<see cref="ArcProbe"/>). Holding the original slice lets the check read the ORIGINAL
    ///     memory's state even when the call swapped the array onto new memory.
    ///     </para>
    ///     <para>
    ///     <b>Distinct arrays only.</b> The same object listed twice (an alias case, a masked array whose <c>_data</c> is
    ///     its source) is recorded once, under its first role. Two different arrays over the same memory (a fixture view of
    ///     a fixture) are both recorded; each check compares each entry with its own snapshot, so the shared count is
    ///     judged consistently.
    ///     </para>
    /// </remarks>
    internal sealed class OwnershipLedger
    {
        /// <summary>One array's snapshot.</summary>
        /// <param name="Array">The caller's array object.</param>
        /// <param name="Role">Its role in the call, for the report.</param>
        /// <param name="Slice">Its storage's slice at snapshot time (null for an array without memory).</param>
        /// <param name="Before">The slice's ARC state at snapshot time.</param>
        /// <param name="DisposedBefore">Whether the array was already disposed at snapshot time (then only changes count).</param>
        /// <param name="Lo">The first byte of the slice's memory (0 for an array without memory).</param>
        /// <param name="Hi">One past its last byte.</param>
        /// <param name="ViewBytes">The array's own byte size (its elements times the item size): with the slice's byte
        ///     length, the bucket sizes a buffer returned in the caller's place would have (see <see cref="PoolScan"/>).</param>
        private sealed record Entry(NDArray Array, string Role, IArraySlice Slice, ArcProbe.ArcState Before, bool DisposedBefore,
                                    ulong Lo, ulong Hi, long ViewBytes);

        /// <summary>The recorded arrays, in snapshot order.</summary>
        private readonly List<Entry> _entries = new();

        /// <summary>An empty ledger (a case without caller arrays checks nothing).</summary>
        public static readonly OwnershipLedger Empty = new();

        /// <summary>How many distinct arrays the ledger watches.</summary>
        public int Count => _entries.Count;

        /// <summary>
        ///     Snapshots the given caller arrays. Null entries and repeats (by reference) are skipped; a disposed array is
        ///     recorded as it is, so a check reports only what changed after the snapshot.
        /// </summary>
        /// <param name="arrays">The arrays the coming call does not own, each with its role.</param>
        /// <returns>The ledger.</returns>
        /// <exception cref="InvalidOperationException">The ARC probe cannot read a slice (see <see cref="ArcProbe.Read"/>).</exception>
        public static OwnershipLedger Of(IEnumerable<(NDArray Array, string Role)> arrays)
        {
            var ledger = new OwnershipLedger();
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            foreach (var (a, role) in arrays)
            {
                if (a is null || !seen.Add(a))
                    continue;
                var slice = a.Storage?.InternalArray;
                var (lo, hi) = RangeOf(slice);
                long viewBytes = 0;
                try
                {
                    viewBytes = a.size * a.dtypesize;
                }
                catch (Exception)
                {
                    // An array that cannot report its item size (a disposed or unusual storage) is still watched by the ARC
                    // checks; the pool scan then reads only its slice's bucket.
                }
                ledger._entries.Add(new Entry(a, role, slice, ArcProbe.Read(slice), a.IsDisposed, lo, hi, viewBytes));
            }
            return ledger;
        }

        /// <summary>
        ///     Snapshots operands named by position: <c>operand 0</c>, <c>operand 1</c>, …
        /// </summary>
        /// <param name="operands">The call's operand arrays (nulls allowed).</param>
        /// <param name="role">The role prefix (<c>operand</c>, <c>mask</c>, …).</param>
        /// <returns>The ledger.</returns>
        /// <exception cref="InvalidOperationException">The ARC probe cannot read a slice.</exception>
        public static OwnershipLedger OfPositional(IReadOnlyList<NDArray> operands, string role = "operand")
        {
            var list = new List<(NDArray, string)>(operands.Count);
            for (int i = 0; i < operands.Count; i++)
                list.Add((operands[i], $"{role} {i}"));
            return Of(list);
        }

        /// <summary>The result of a clean check (shared: a clean check allocates nothing).</summary>
        private static readonly IReadOnlyList<OwnershipFinding> NoFindings = Array.Empty<OwnershipFinding>();

        /// <summary>
        ///     Compares every recorded array with its snapshot.
        /// </summary>
        /// <returns>
        ///     Every finding, the over-disposal kinds first (a released array is reported as <c>Released</c>, not also as a
        ///     dropped reference). Empty when every array is intact.
        /// </returns>
        /// <param name="scanPool">Also look for pool buffers parked inside a held array's memory
        ///     (<see cref="OwnershipViolation.PooledWhileHeld"/>). The scan walks every bucket, so the sweep asks for it once
        ///     per case (after the warm run), not after every region.</param>
        /// <exception cref="InvalidOperationException">The ARC probe or the pool scan cannot read an internal field.</exception>
        /// <remarks>A clean check reads two flags and a counter per array through compiled accessors and allocates
        ///     nothing, so the sweep can afford it after every region of every case.</remarks>
        public IReadOnlyList<OwnershipFinding> Check(bool scanPool = false)
        {
            List<OwnershipFinding> overDisposals = null, retained = null;
            foreach (var e in _entries)
            {
                if (e.Array.IsDisposed && !e.DisposedBefore)
                    (overDisposals ??= new()).Add(new OwnershipFinding(OwnershipViolation.Disposed, e.Role,
                        $"the {Describe(e.Array)} was disposed by the call"));

                // The ORIGINAL memory's state, re-read on the owner the snapshot found (valid even when the call swapped the
                // array onto new memory: that owner is still the old block's disposer).
                var now = ArcProbe.ReadOwner(e.Before);
                if (now.Freed && !e.Before.Freed)
                {
                    (overDisposals ??= new()).Add(new OwnershipFinding(OwnershipViolation.Released, e.Role,
                        $"its {e.Before.Kind} memory was released (refcount {e.Before.RefCount} -> {now.RefCount})"));
                }
                else if (e.Before.CountsReferences && now.RefCount < e.Before.RefCount)
                {
                    (overDisposals ??= new()).Add(new OwnershipFinding(OwnershipViolation.ReferenceDropped, e.Role,
                        $"its {e.Before.Kind} memory lost counted references ({e.Before.RefCount} -> {now.RefCount})"));
                }
                else if (e.Before.CountsReferences && now.RefCount > e.Before.RefCount)
                {
                    (retained ??= new()).Add(new OwnershipFinding(OwnershipViolation.ReferenceRetained, e.Role,
                        $"its {e.Before.Kind} memory gained counted references ({e.Before.RefCount} -> {now.RefCount}) " +
                        "that outlived the call"));
                }

                // A swapped buffer: the array's CURRENT memory is not the snapshot's. The slice object is stored boxed in the
                // storage, so an unchanged reference proves nothing moved; a different one is judged by owner identity (two
                // slices share memory exactly when their disposers are the same object — a re-boxed slice over the same
                // block is not a swap).
                if (!e.Array.IsDisposed && e.Slice is not null)
                {
                    var currentSlice = e.Array.Storage?.InternalArray;
                    if (!ReferenceEquals(currentSlice, e.Slice))
                    {
                        var current = ArcProbe.Read(currentSlice);
                        if (!ReferenceEquals(current.Owner, e.Before.Owner))
                            (overDisposals ??= new()).Add(new OwnershipFinding(OwnershipViolation.StorageReplaced, e.Role,
                                $"the {Describe(e.Array)} now refers to other memory ({e.Before.Kind} -> {current.Kind ?? "none"})"));
                    }
                }
            }

            if (scanPool)
            {
                // Only the arrays whose memory is still theirs: a released array's buffer in the pool is the Released
                // finding above, not a second one.
                var held = new List<(ulong lo, ulong hi)>(_entries.Count);
                var heldEntries = new List<Entry>(_entries.Count);
                var sizes = new HashSet<long>();
                foreach (var e in _entries)
                {
                    if (e.Hi <= e.Lo || e.Array.IsDisposed || ArcProbe.ReadOwner(e.Before).Freed)
                        continue;
                    held.Add((e.Lo, e.Hi));
                    heldEntries.Add(e);
                    sizes.Add((long)(e.Hi - e.Lo));
                    if (e.ViewBytes > 0)
                        sizes.Add(e.ViewBytes);
                }

                var pooled = PoolScan.FirstInside(held, sizes);
                if (pooled != IntPtr.Zero)
                {
                    ulong a = (ulong)pooled.ToInt64();
                    var e = heldEntries.Find(x => a >= x.Lo && a < x.Hi);
                    (overDisposals ??= new()).Add(new OwnershipFinding(OwnershipViolation.PooledWhileHeld, e.Role,
                        $"the buffer pool holds 0x{a:X} inside its {e.Before.Kind} memory [0x{e.Lo:X}, 0x{e.Hi:X})"));
                }
            }

            if (overDisposals is null && retained is null)
                return NoFindings;
            overDisposals ??= new();
            if (retained is not null)
                overDisposals.AddRange(retained);
            return overDisposals;
        }

        /// <summary>The byte range of a slice's memory.</summary>
        /// <param name="slice">The slice (null for an array without memory).</param>
        /// <returns><c>[lo, hi)</c>; an empty range for no memory.</returns>
        private static unsafe (ulong lo, ulong hi) RangeOf(IArraySlice slice)
        {
            if (slice is null)
                return (0, 0);
            ulong lo = (ulong)slice.Address;
            return lo == 0 ? (0, 0) : (lo, lo + (ulong)Math.Max(0L, slice.BytesLength));
        }

        /// <summary>A short description of an array for a report line (dtype and shape).</summary>
        /// <param name="a">The array.</param>
        /// <returns>For example <c>float64 (3, 4) array</c>.</returns>
        private static string Describe(NDArray a)
        {
            try
            {
                return $"{a.dtype} {a.Shape} array";
            }
            catch (Exception)
            {
                // A damaged array may not even describe itself; the report must still be written.
                return "array";
            }
        }
    }
}
