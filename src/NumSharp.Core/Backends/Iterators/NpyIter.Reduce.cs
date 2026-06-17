using System;
using NumSharp.Backends.Kernels;

// =============================================================================
// NpyIter.Reduce.cs — reusable 2-operand REDUCE iterator builder
// =============================================================================
//
// RATIONALE
// ---------
// Axis reductions in NumPy are driven by an nditer constructed with a writable
// output operand whose stride along every reduced axis is 0 (REDUCE). The
// canonical loop then visits each input element exactly once and the kernel
// accumulates into the pinned (or scattered) output slot:
//
//     do { inner(dataptrs, strides, count, aux); } while (iternext(iter));
//
// np.average already builds this shape by hand for its fused 4-operand
// weighted-sum (Statistics/np.average.cs : TryFusedWeightedSum). This file
// hoists that construction into a single reusable factory so every np.*
// reduction (sum / prod / min / max / mean, single- or multi-axis) shares one
// audited op_axes builder instead of re-deriving it per call site.
//
// CONTRACT
// --------
// Operands: [input, output]
//   input  — READONLY
//   output — READWRITE, pre-seeded with the reduction identity by the caller
//            (0 for Sum, 1 for Prod, ±inf for Min/Max). The kernel reads the
//            current slot, folds the inner stripe in, writes it back, so a slab
//            output that is revisited across outer iterations accumulates.
// Flags  : REDUCE_OK | EXTERNAL_LOOP (+ caller extras, e.g. COPY_IF_OVERLAP for
//          a user-supplied out= that may alias the input).
// op_axes: input  = identity (axis i -> i)
//          output = oc++ for kept axes, -1 for reduced axes (stride 0 ⇒ REDUCE)
//
// This is non-buffered REDUCE+EXTERNAL_LOOP, which is N-D capable via Advance
// (it sidesteps the 2-D-only buffered reduce double-loop). The inner loop the
// kernel sees is one contiguous stripe of the iteration's innermost axis; the
// output stride is 0 (pinned) when the reduced axis is innermost, else nonzero
// (slab) when a kept axis is innermost.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public unsafe ref partial struct NpyIterRef
    {
        /// <summary>
        /// Build a 2-operand REDUCE iterator over <paramref name="input"/> reducing
        /// the single axis <paramref name="axis"/> into <paramref name="output"/>.
        /// <paramref name="output"/> must already have the reduced shape
        /// (input shape with <paramref name="axis"/> removed) and be pre-seeded with
        /// the reduction identity.
        /// </summary>
        /// <param name="input">Read-only source operand.</param>
        /// <param name="output">Read-write destination; reduced shape, identity-seeded.</param>
        /// <param name="axis">Normalized (non-negative) axis to reduce.</param>
        /// <param name="extraFlags">Extra global flags OR-ed onto REDUCE_OK|EXTERNAL_LOOP.</param>
        public static NpyIterRef NewReduce(
            NDArray input, NDArray output, int axis,
            NpyIterGlobalFlags extraFlags = NpyIterGlobalFlags.None)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));

            int ndim = input.ndim;
            if (axis < 0 || axis >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");

            int[] inAxes = new int[ndim];
            int[] outAxes = new int[ndim];
            int oc = 0;
            for (int i = 0; i < ndim; i++)
            {
                inAxes[i] = i;
                outAxes[i] = (i == axis) ? -1 : oc++;
            }

            return BuildReduce(input, output, ndim, inAxes, outAxes, extraFlags);
        }

        /// <summary>
        /// Multi-axis variant of <see cref="NewReduce(NDArray, NDArray, int, NpyIterGlobalFlags)"/>.
        /// Every axis in <paramref name="axes"/> is reduced (stride 0 in the output);
        /// the remaining axes map to the output in ascending order. <paramref name="axes"/>
        /// must be normalized (non-negative) and contain no duplicates.
        /// </summary>
        public static NpyIterRef NewReduce(
            NDArray input, NDArray output, int[] axes,
            NpyIterGlobalFlags extraFlags = NpyIterGlobalFlags.None)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (output is null) throw new ArgumentNullException(nameof(output));
            if (axes is null) throw new ArgumentNullException(nameof(axes));

            int ndim = input.ndim;
            int[] inAxes = new int[ndim];
            int[] outAxes = new int[ndim];
            int oc = 0;
            for (int i = 0; i < ndim; i++)
            {
                inAxes[i] = i;
                bool reduced = Array.IndexOf(axes, i) >= 0;
                outAxes[i] = reduced ? -1 : oc++;
            }

            return BuildReduce(input, output, ndim, inAxes, outAxes, extraFlags);
        }

        private static NpyIterRef BuildReduce(
            NDArray input, NDArray output, int ndim,
            int[] inAxes, int[] outAxes, NpyIterGlobalFlags extraFlags)
        {
            return AdvancedNew(
                2,
                new[] { input, output },
                NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.EXTERNAL_LOOP | extraFlags,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
                null,
                ndim,
                new[] { inAxes, outAxes });
        }
    }
}
