using System;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     Detach / re-borrow plumbing that lets a long-lived MANAGED object own an iterator.
    ///
    ///     <see cref="NDIterRef"/> is a <c>ref struct</c>, so it cannot be stored in a class field
    ///     — but <c>np.nditer</c> is a class whose lifetime spans user code. The bridge is the
    ///     already-heap-allocated <see cref="NDIterState"/>: <see cref="NDIterRef.Detach"/> hands
    ///     the pointer (plus the operands and any pending COPY_IF_OVERLAP write-backs) to the
    ///     owner, and <see cref="NDIterRef.Borrow"/> rebuilds a NON-owning <see cref="NDIterRef"/>
    ///     around it for the duration of a single call.
    ///
    ///     This extends the existing <see cref="NDIterRef.ReleaseState"/> /
    ///     <see cref="NDIterRef.FreeState"/> pair, which drops the write-back registrations on the
    ///     floor: <c>ReleaseState</c> nulls the state but leaves <c>_writebackOriginals</c> on the
    ///     dying ref struct, so a COPY_IF_OVERLAP temp would never be copied back.
    /// </summary>
    public unsafe ref partial struct NDIterRef
    {
        /// <summary>
        ///     Hand this iterator's entire heap state to a long-lived owner. After the call this
        ///     instance is inert (its <see cref="Dispose"/> becomes a no-op) and the owner is
        ///     responsible for <see cref="FreeState"/> plus resolving <paramref name="writebackOriginals"/>
        ///     via <see cref="ResolveDetachedWritebacks"/>.
        /// </summary>
        /// <param name="operands">The operand arrays the iterator was built over.</param>
        /// <param name="writebackOriginals">
        ///     COPY_IF_OVERLAP registrations: where entry <c>i</c> is non-null,
        ///     <paramref name="operands"/>[i] is a temporary copy whose contents must be copied
        ///     back into the stored original at teardown. Null when nothing was copied.
        /// </param>
        internal NDIterState* Detach(out NDArray[]? operands, out NDArray?[]? writebackOriginals)
        {
            if (!_ownsState)
                throw new InvalidOperationException("Iterator does not own its state; cannot detach.");

            operands = _operands;
            writebackOriginals = _writebackOriginals;

            var released = _state;
            _state = null;
            _ownsState = false;
            _operands = null;
            _writebackOriginals = null;
            return released;
        }

        /// <summary>
        ///     Rebuild a NON-owning <see cref="NDIterRef"/> over state previously handed out by
        ///     <see cref="Detach"/>. The result must NOT be disposed — the detached owner frees
        ///     the state — so callers use it bare rather than in a <c>using</c>.
        /// </summary>
        /// <param name="cachedIterNext">
        ///     The owner's cached advance delegate, so a borrowed instance does not re-resolve it
        ///     on every step. Pass null to let <see cref="GetIterNext"/> resolve it, then read the
        ///     result back through <see cref="PeekCachedIterNext"/> and store it.
        /// </param>
        internal static NDIterRef Borrow(NDIterState* state, NDArray[]? operands, NDIterNextFunc? cachedIterNext = null)
        {
            return new NDIterRef
            {
                _state = state,
                _ownsState = false,
                _operands = operands,
                _cachedIterNext = cachedIterNext,
            };
        }

        /// <summary>
        ///     The advance delegate resolved during this borrow, so the owner can keep it warm.
        ///     Returns null after a reconfiguration (RemoveMultiIndex / EnableExternalLoop clear
        ///     the cache), which is exactly the signal to drop the owner's copy too.
        /// </summary>
        internal NDIterNextFunc? PeekCachedIterNext() => _cachedIterNext;

        /// <summary>
        ///     The inner-loop stride of operand <paramref name="op"/> in ELEMENTS — the unit
        ///     <see cref="Shape"/> strides use, so a caller can wrap the current inner loop in an
        ///     NDArray view. Mirrors the private <c>GetInnerLoopByteStrides</c>: buffered operands
        ///     take <see cref="NDIterState.BufStrides"/> (which are BYTES, hence the divide),
        ///     unbuffered ones the innermost axis stride (already elements).
        /// </summary>
        internal long GetInnerLoopElementStride(int op)
        {
            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) != 0)
            {
                int elementSize = _state->ElementSizes[op];
                return elementSize == 0 ? 0 : _state->BufStrides[op] / elementSize;
            }

            if (_state->NDim == 0)
                return 0;

            return _state->GetStride(_state->NDim - 1, op);
        }

        /// <summary>
        ///     Resolve write-backs for a detached iterator — the standalone form of the private
        ///     <c>ResolveWritebacks</c> that <see cref="Dispose"/> runs, operating on the arrays
        ///     handed out by <see cref="Detach"/>.
        /// </summary>
        internal static void ResolveDetachedWritebacks(NDArray[]? operands, NDArray?[]? writebackOriginals)
        {
            if (writebackOriginals is null || operands is null)
                return;

            for (int iop = 0; iop < writebackOriginals.Length && iop < operands.Length; iop++)
            {
                var original = writebackOriginals[iop];
                if (original is null)
                    continue;

                np.copyto(original, operands[iop]);
            }
        }
    }
}
