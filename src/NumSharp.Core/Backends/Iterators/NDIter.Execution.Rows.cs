using System;
using System.Runtime.CompilerServices;

// =============================================================================
// NDIter.Execution.Rows.cs — the rows driver: one kernel call per BLOCK of rows
// =============================================================================
//
// WHY THIS EXISTS
//   Under EXTERNAL_LOOP an unbuffered iteration that does not coalesce to one axis reaches its kernel once per inner
//   row: a (1000, 3) slice `m[:, :3] == "abc"` is a thousand calls of three elements. Each call pays the odometer advance
//   AND the kernel's own prologue (pointer and stride snapshots, the broadcast-pattern dispatch, the hoisting of a
//   broadcast operand), about 15 ns, for three elements of work. The strings phase-2 discovery measured that case at
//   19.3 µs through ForEach against 2.4 µs when one call walks every row (docs/plans/strings-phase2.md §2.5, D16).
//
// THE CONTRACT
//   The kernel is an NDInnerLoop2DFunc: (dataptrs, innerStrides, innerCount, outerStrides, outerCount, auxdata). It
//   processes `outerCount` rows of `innerCount` elements; row r of operand k starts at dataptrs[k] + r * outerStrides[k]
//   and its elements are innerStrides[k] bytes apart. Strides are BYTES and may be zero (a broadcast operand) or
//   negative. The kernel owns the row loop, so its prologue runs once per block.
//
// THE BLOCK
//   The trailing outer axes that EVERY operand walks as a continuation of the next inner one
//   (stride[d] == stride[d + 1] * shape[d + 1]) fold into one (outerCount, outerStride) block — the same folding the
//   numeric 2-D block kernels use (NDIter.Execution.Custom.cs, TryExecute2DElementwise). The leading axes that do not
//   fold are walked here with a small odometer: one kernel call per block, never per row.
//
// SCOPE
//   Unbuffered (a buffered iteration hands its kernel whole buffer fills, already one call per fill), EXTERNAL_LOOP,
//   no ARRAYMASK operand (a where= mask the kernel blends itself is passed as an ordinary operand), not ranged, at the
//   start of the iteration, and at least two axes after the iterator's coalescing (a one-axis iteration is already one
//   call). Element strides are unrestricted: the callers' kernels read through any stride.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     The rows form of <see cref="NDInnerLoopFunc"/>: one call processes <paramref name="outerCount"/> rows of
    ///     <paramref name="innerCount"/> elements, so a kernel whose rows are short pays its prologue (pointer snapshots,
    ///     broadcast dispatch, hoisted operands) once per block instead of once per row. Driven by
    ///     <see cref="NDIterRef.ForEachRows"/>.
    /// </summary>
    /// <param name="dataptrs">One pointer per operand: the first element of the block's first row.</param>
    /// <param name="innerStrides">Per operand, the byte distance between two elements of a row (0 for a broadcast
    ///     operand; may be negative).</param>
    /// <param name="innerCount">The number of elements in every row.</param>
    /// <param name="outerStrides">Per operand, the byte distance between the first elements of two consecutive rows (0
    ///     for an operand broadcast across rows; may be negative).</param>
    /// <param name="outerCount">The number of rows in the block.</param>
    /// <param name="auxdata">The caller's cookie, passed through unchanged (may be null).</param>
    internal unsafe delegate void NDInnerLoop2DFunc(
        void** dataptrs, long* innerStrides, long innerCount, long* outerStrides, long outerCount, void* auxdata);

    public unsafe ref partial struct NDIterRef
    {
        /// <summary>
        ///     Whether <see cref="ForEachRows"/> can drive this iteration: unbuffered, <c>EXTERNAL_LOOP</c>, more than one
        ///     inner chunk, no <c>ARRAYMASK</c> operand, not ranged, still at its start, and at least two axes after
        ///     coalescing. When false, <see cref="ForEach"/> is the driver (a one-axis iteration is a single call there
        ///     already; a buffered one is one call per buffer fill).
        /// </summary>
        /// <remarks>
        ///     A caller asks this and then picks its kernel form: the per-chunk <see cref="NDInnerLoopFunc"/> for
        ///     <see cref="ForEach"/>, the rows <see cref="NDInnerLoop2DFunc"/> for <see cref="ForEachRows"/>. Both forms
        ///     must compute the same elements, so the choice can only change speed.
        /// </remarks>
        internal bool CanRunRows
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                uint f = _state->ItFlags;
                if ((f & (uint)(NDIterFlags.BUFFER | NDIterFlags.RANGE | NDIterFlags.ONEITERATION)) != 0)
                    return false;
                if ((f & (uint)NDIterFlags.EXLOOP) == 0)
                    return false;
                if (_state->MaskOp >= 0 || _state->IterIndex != 0)
                    return false;
                return _state->NDim >= 2;
            }
        }

        /// <summary>
        ///     Drives the whole iteration through a rows kernel: the trailing outer axes every operand walks contiguously
        ///     fold into one block, the leading ones are walked here, one kernel call per block (see the file header).
        /// </summary>
        /// <param name="kernel">The rows kernel; it receives every operand of the iterator, in operand order.</param>
        /// <param name="auxdata">Passed to every kernel call unchanged.</param>
        /// <exception cref="ArgumentNullException"><paramref name="kernel"/> is null.</exception>
        /// <exception cref="InvalidOperationException"><see cref="CanRunRows"/> is false: the iteration is buffered,
        ///     masked, ranged, already advanced, or has fewer than two axes (a caller bug: ask first).</exception>
        /// <remarks>
        ///     The iterator's own position is not moved: the blocks run over private copies of its start pointers, as the
        ///     numeric 2-D block route does, so the iterator is finished only in the sense that every element has been
        ///     visited. A caller disposes it afterwards as usual.
        /// </remarks>
        internal void ForEachRows(NDInnerLoop2DFunc kernel, void* auxdata = null)
        {
            if (kernel is null)
                throw new ArgumentNullException(nameof(kernel));
            if (!CanRunRows)
                throw new InvalidOperationException(
                    "ForEachRows needs an unbuffered, unmasked EXTERNAL_LOOP iteration of at least two axes at its start.");

            int nop = _state->NOp;
            int ndim = _state->NDim;
            int inner = ndim - 1;
            long* shape = _state->Shape;
            // The data pointers walk SOURCE memory, so element strides scale by the source element sizes (the string
            // operands' descriptor widths among them), exactly as ExternalLoopNext scales them.
            int* sizes = _state->SrcElementSizes;

            // The block grows outward from the innermost outer axis while every operand walks the next axis out as a
            // continuation of it (a broadcast operand's 0 == 0 * extent continues too).
            int first = inner - 1;
            while (first > 0)
            {
                bool continues = true;
                for (int op = 0; op < nop; op++)
                {
                    if (_state->GetStride(first - 1, op) != _state->GetStride(first, op) * shape[first])
                    {
                        continues = false;
                        break;
                    }
                }
                if (!continues)
                    break;
                first--;
            }

            long innerCount = shape[inner];
            long outerCount = 1;
            for (int d = first; d < inner; d++)
                outerCount *= shape[d];

            long* innerStrides = stackalloc long[nop];
            long* outerStrides = stackalloc long[nop];
            for (int op = 0; op < nop; op++)
            {
                innerStrides[op] = _state->GetStride(inner, op) * sizes[op];
                outerStrides[op] = _state->GetStride(inner - 1, op) * sizes[op];
            }

            void** start = GetDataPtrArray();
            if (first == 0)
            {
                kernel(start, innerStrides, innerCount, outerStrides, outerCount, auxdata);
                return;
            }

            // Leading axes 0 .. first-1: an odometer over private pointer copies. The innermost of them advances by its
            // stride; when an axis wraps, its pointers step back by (extent - 1) strides and the next axis out advances.
            void** ptrs = stackalloc void*[nop];
            for (int op = 0; op < nop; op++)
                ptrs[op] = start[op];
            long* coords = stackalloc long[first];
            for (int d = 0; d < first; d++)
                coords[d] = 0;

            while (true)
            {
                kernel(ptrs, innerStrides, innerCount, outerStrides, outerCount, auxdata);

                int axis = first - 1;
                for (; axis >= 0; axis--)
                {
                    if (++coords[axis] < shape[axis])
                    {
                        for (int op = 0; op < nop; op++)
                            ptrs[op] = (byte*)ptrs[op] + _state->GetStride(axis, op) * sizes[op];
                        break;
                    }
                    coords[axis] = 0;
                    for (int op = 0; op < nop; op++)
                        ptrs[op] = (byte*)ptrs[op] - _state->GetStride(axis, op) * (shape[axis] - 1) * sizes[op];
                }
                if (axis < 0)
                    return;
            }
        }
    }
}
