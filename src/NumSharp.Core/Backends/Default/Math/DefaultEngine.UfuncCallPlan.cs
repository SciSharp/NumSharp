using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// The inner-loop calls NumPy 2.4.2 makes for one elementwise ufunc call with one output: the order its loop visits
        /// the iteration space in, where every call starts and ends, and the byte step of each operand inside a call.
        /// Built by <see cref="NumPyUfuncCalls"/> from the shared port of NumPy's loop decision (the trivial-loop shortcut
        /// or NpyIter's axis sort, flips, coalescing and buffering; see DefaultEngine.UfuncInnerSteps.cs).
        /// </summary>
        /// <remarks>
        /// <para>Use it when a NumPy inner loop's answer depends on WHERE an element falls inside a call, not only on its
        /// operands — np.fmax / np.fmin give a ±0 tie and a NaN pair one rule inside whole SIMD vectors and another in the
        /// scalar tail (QA-20), so the element's offset in its call decides its bits. A loop that branches only on its
        /// steps needs <see cref="Steps"/> alone (<see cref="NumPyUfuncInputSteps"/>).</para>
        /// <para>Positions are NumPy's iteration index: position <c>p</c> is the <c>p</c>-th element the loop visits, and
        /// the calls are consecutive position ranges. Every slab of <see cref="SlabSize"/> positions is cut into calls of
        /// <see cref="CallSize"/> positions, the last one of a slab shorter when the slab is not a multiple; no call
        /// crosses a slab. <see cref="LogicalIndex"/> turns a position into the broadcast result's multi-index.</para>
        /// <para>A where= mask splits each call further into its runs of true elements (NumPy's masked strided loop calls
        /// the inner loop once per run with the call's steps); the plan does not model that split, because it depends on
        /// the mask's values — the caller walks the runs.</para>
        /// </remarks>
        internal sealed class NumPyUfuncCallPlan
        {
            /// <summary>The extent of each iterator axis, fastest-varying first (the axes before coalescing, which
            /// merges neighbours without changing the visiting order). Empty for a 0-d iteration.</summary>
            public readonly long[] AxisShape;

            /// <summary>The broadcast result's axis each iterator axis walks (<c>AxisArray[0]</c> varies fastest).</summary>
            public readonly int[] AxisArray;

            /// <summary>Whether NpyIter walks the iterator axis backwards (npyiter_flip_negative_strides: the axis's
            /// strides were all non-positive with one negative, and no output was allocated) — position 0 along it is the
            /// result's LAST index there.</summary>
            public readonly bool[] AxisReversed;

            /// <summary>The number of positions (the broadcast result's element count; 1 for a 0-d iteration).</summary>
            public readonly long Size;

            /// <summary>The positions of one pass over NpyIter's buffering core and outer dimension
            /// (<c>coresize * outer extent</c>); a call never crosses a slab boundary. <see cref="Size"/> for the trivial
            /// loop and for an iteration that coalesces to one dimension.</summary>
            public readonly long SlabSize;

            /// <summary>The positions of one call inside a slab (NpyIter's buffer size: the slab itself unless an operand
            /// is buffered and the slab exceeds <c>np.getbufsize()</c>, then the largest multiple of the core size that fits).</summary>
            public readonly long CallSize;

            /// <summary>The byte step of every operand in every call: the inputs in order, then the output (length
            /// <c>nin + 1</c>). A buffered operand steps over its contiguous buffer (the loop dtype's item size, or 0 for a
            /// broadcast single-stride operand); an unbuffered one keeps its stride along the innermost coalesced axis.</summary>
            public readonly long[] Steps;

            /// <summary>True when NumPy runs try_trivial_single_output_loop: ONE call over every position, each operand at
            /// its own data pointer with its fixed step — the only route whose operands can overlap inside a call
            /// (NpyIter's COPY_IF_OVERLAP leaves either disjoint operands or exact elementwise aliases).</summary>
            public readonly bool Trivial;

            /// <summary>For the trivial loop: the byte address of each operand's first element in the call (the inputs,
            /// then the output); 0 for a fresh operand (a copy check_for_trivial_loop made, a converted scalar, or an
            /// allocated output), which overlaps nothing. Null for an NpyIter plan.</summary>
            public readonly long[] Starts;

            /// <summary>
            /// Stores a computed plan. Only <see cref="NumPyUfuncCalls"/>'s two builders call it, after deriving every part
            /// from NumPy's decision, so nothing is validated here; the arrays are kept by reference.
            /// </summary>
            /// <param name="axisShape">Iterator axis extents, fastest first.</param>
            /// <param name="axisArray">The result axis each iterator axis walks.</param>
            /// <param name="axisReversed">Whether each iterator axis is walked backwards.</param>
            /// <param name="size">The number of positions.</param>
            /// <param name="slabSize">Positions per slab (calls never cross one); must divide <paramref name="size"/>.</param>
            /// <param name="callSize">Positions per call inside a slab; at most <paramref name="slabSize"/>.</param>
            /// <param name="steps">The byte step of each input, then of the output.</param>
            /// <param name="trivial">True for the trivial loop's single call.</param>
            /// <param name="starts">The trivial loop's operand addresses (0 = fresh), or null for NpyIter.</param>
            internal NumPyUfuncCallPlan(long[] axisShape, int[] axisArray, bool[] axisReversed, long size, long slabSize,
                                        long callSize, long[] steps, bool trivial, long[] starts)
            {
                AxisShape = axisShape;
                AxisArray = axisArray;
                AxisReversed = axisReversed;
                Size = size;
                SlabSize = slabSize;
                CallSize = callSize;
                Steps = steps;
                Trivial = trivial;
                Starts = starts;
            }

            /// <summary>
            /// The call that contains position <paramref name="p"/>: its first position and its length.
            /// </summary>
            /// <param name="p">A position, <c>0 &lt;= p &lt; Size</c>.</param>
            /// <param name="start">Receives the call's first position.</param>
            /// <param name="length">Receives the call's number of positions.</param>
            public void CallAt(long p, out long start, out long length)
            {
                long slab = p / SlabSize;
                long inSlab = (p - slab * SlabSize) / CallSize * CallSize;
                start = slab * SlabSize + inSlab;
                length = Math.Min(CallSize, SlabSize - inSlab);
            }

            /// <summary>
            /// Writes the broadcast result's multi-index of position <paramref name="p"/> into <paramref name="index"/>:
            /// <paramref name="p"/> is decomposed over <see cref="AxisShape"/> (axis 0 fastest) and each digit mapped to
            /// its result axis, counted from the end along a reversed axis.
            /// </summary>
            /// <param name="p">A position, <c>0 &lt;= p &lt; Size</c>.</param>
            /// <param name="index">Receives one index per result axis (its length is the result's ndim; untouched for a
            /// 0-d iteration).</param>
            public void LogicalIndex(long p, Span<long> index)
            {
                for (int k = 0; k < AxisShape.Length; k++)
                {
                    long extent = AxisShape[k];
                    long digit = p % extent;
                    p /= extent;
                    index[AxisArray[k]] = AxisReversed[k] ? extent - 1 - digit : digit;
                }
            }
        }
    }
}
