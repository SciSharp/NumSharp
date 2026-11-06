using System;
using System.Text;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     The <c>out=</c> store shared by <see cref="ReduceCumAdd"/> and <see cref="ReduceCumMul"/>: NumPy's
        ///     add.accumulate / multiply.accumulate require <paramref name="out"/> to have the natural output shape (any
        ///     layout) and store the result into it with UNSAFE casting (float to int truncates, complex to real drops the
        ///     imaginary part), returning <paramref name="out"/> itself.
        /// </summary>
        /// <param name="result">The accumulation, a fresh array of the loop dtype.</param>
        /// <param name="out">The caller's <c>out=</c>, or null.</param>
        /// <param name="arr">The array the scan accumulated, as the caller passed it (before any flattening); read only to
        ///     word a shape refusal.</param>
        /// <param name="axis">The accumulated axis as the caller passed it, or null for the flattened array; read only to
        ///     word a shape refusal.</param>
        /// <param name="loop">The loop dtype the scan computed in (<see cref="AccumulateLoop"/>; never null when
        ///     <paramref name="out"/> is given); read only to word a shape refusal.</param>
        /// <returns><paramref name="result"/> when <paramref name="out"/> is null, else <paramref name="out"/> after the
        ///     store.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>).</exception>
        /// <exception cref="NumSharpException">A read-only <paramref name="out"/>: the <see cref="ValueError"/> <c>output array
        ///     is read-only</c> that <see cref="NumSharpException.ThrowIfNotWriteable"/> raises, checked first.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have <paramref name="result"/>'s shape:
        ///     the text of the NumPy statement that refuses it (<see cref="AccumulateOutShapeError"/>; parity QA-31), the
        ///     house shape exception standing in for NumPy's ValueError.</exception>
        /// <remarks>
        ///     Moved from <c>APIs/np.cumsum.cs</c> with the two scans (the np-layer dtype plan's P0 pilot). The store still
        ///     goes through <c>np.copyto</c>, a structural call into the facade that the plan's P6 replaces with an engine
        ///     copy; the up-call ratchet pins it with that reason.
        /// </remarks>
        internal static NDArray WriteScanToOut(NDArray result, NDArray @out, NDArray arr, int? axis, DType loop)
        {
            if (@out is null)
                return result;
            // A read-only out refuses FIRST — probed 2.4.2: a wrong-sized read-only out reports
            // "output array is read-only", not the size error (accumulate validates writeability
            // ahead of the accumulation-shape check). Without the guard the copyto below raised
            // the generic assignment text instead of NumPy's ufunc-out wording.
            NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            if (!@out.Shape.Equals(result.Shape))
                // Which statement of PyUFunc_Accumulate refuses depends on the path it takes; the house
                // IncorrectShapeException stands in for NumPy's ValueError per the shape-error convention.
                throw AccumulateOutShapeError(@out, arr, axis, loop);
            UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_UNSAFE_CASTING);
            return @out;
        }

        /// <summary>
        ///     NumPy's refusal of an accumulation <c>out=</c> whose dimensions are not the accumulation's, as
        ///     <c>PyUFunc_Accumulate</c> words it on the path it takes: its outer ITERATOR (an input of more than one
        ///     dimension, a loop dtype other than the input's or the out's, an unaligned operand, an out whose bounds overlap
        ///     the input's) raises NpyIter's texts (<see cref="AccumulateIteratorShapeError"/>: <c>operands could not be
        ///     broadcast together with remapped shapes [original-&gt;remapped]: (4,3)-&gt;(4,3) (3,4)-&gt;(3,4) </c> and its two
        ///     siblings); the DIRECT 1-D path raises <c>provided out is the wrong size for the accumulation.</c>.
        /// </summary>
        /// <param name="out">The caller's out (its dimensions differ from the accumulation's).</param>
        /// <param name="arr">The array as the caller passed it (before any flattening).</param>
        /// <param name="axis">The accumulated axis, or null for the flattened array.</param>
        /// <param name="loop">The loop dtype (NumPy's resolved <c>descrs</c>: all three are this dtype for add / multiply).</param>
        /// <returns>The exception to throw (never null).</returns>
        /// <remarks>
        ///     <para>
        ///     Divergence (parity QA-54, pinned): NumPy ACCEPTS some outs this engine refuses, and for those the direct path's
        ///     text is returned. On the iterator path an out with more dimensions than the input whose leading dimensions fit
        ///     (its trailing axes stay at index 0), or an input of extent 1 against a longer out (the input broadcasts; along
        ///     the accumulated axis only the input's extent is written); on the direct path an empty input or an empty out
        ///     (both return the out untouched before the dimensions are compared). The StringDType accumulation
        ///     (<see cref="Strings.NDVStringReduction"/>) implements those acceptances.
        ///     </para>
        ///     <para>
        ///     A 0-d out on the direct path crashes NumPy 2.4.2 (it reads the out's stride along the axis); the direct path's
        ///     text is returned instead.
        ///     </para>
        /// </remarks>
        internal static IncorrectShapeException AccumulateOutShapeError(NDArray @out, NDArray arr, int? axis, DType loop)
        {
            // PyArray_CheckAxis: axis=None, and any axis of a 0-d input, accumulate the C-order RAVEL, a 1-D array that is a
            // view of a C-contiguous input (its memory, so its alignment and its overlap with the out) and a fresh aligned
            // copy otherwise (it shares nothing). Every other call accumulates the input itself.
            bool raveled = axis is null || arr.ndim == 0;
            long[] inDims = raveled ? new[] { arr.size } : arr.Shape.dimensions;
            bool readsInputMemory = !raveled || arr.Shape.IsContiguous;
            // need_outer_iterator, NumPy's test order: more than one dimension, an unaligned operand or a loop descriptor
            // that is not equivalent to an operand's, else an out whose bounds may overlap the input's (NPY_MAY_SHARE_BOUNDS:
            // a longer view of the input's own buffer takes the iterator).
            bool iterator = inDims.Length > 1
                            || readsInputMemory && !UfuncLoopOperand.IsAligned(arr)
                            || !UfuncLoopOperand.IsAligned(@out)
                            || !DTypeCasting.EquivTypes(loop, arr.dtype)
                            || !DTypeCasting.EquivTypes(loop, @out.dtype)
                            || readsInputMemory && NDMemOverlap.SolveMayShareMemory(@out, arr, maxWork: 0) != MemOverlap.No;
            if (iterator && AccumulateIteratorShapeError(@out.Shape.dimensions ?? Array.Empty<long>(), inDims) is { } refusal)
                return refusal;
            // The direct path (and the outs NumPy's iterator accepts: see the remarks).
            return new IncorrectShapeException("provided out is the wrong size for the accumulation.");
        }

        /// <summary>
        ///     The shape errors of <c>PyUFunc_Accumulate</c>'s iterator over (out, input) with <c>op_axes</c>
        ///     <c>[0 .. ndim-1]</c> for both, in NpyIter's order: the out's <c>op_axes</c> entries first (an out with fewer
        ///     dimensions than the input), then the broadcast of each operand's extents, then the out's NO_BROADCAST check.
        ///     Shared by the numeric scans (<see cref="AccumulateOutShapeError"/>) and the StringDType accumulation.
        /// </summary>
        /// <param name="outDims">The out's dimensions.</param>
        /// <param name="inDims">The accumulated input's dimensions (already flattened for <c>axis=None</c>).</param>
        /// <returns>The error, or null when the iterator accepts the operands.</returns>
        /// <remarks>
        ///     Every text is NumPy's as it prints it, quirks included: the <c>op_axes</c> error reports the entry index
        ///     <c>ndim - idim - 1</c> beside the axis value <c>idim</c>, and the non-broadcastable error's "remapped to" lists
        ///     the out's extents in REVERSED axis order (NumPy builds it from <c>axes[ndim - idim - 1]</c>).
        /// </remarks>
        internal static IncorrectShapeException AccumulateIteratorShapeError(long[] outDims, long[] inDims)
        {
            int ndim = inDims.Length, ondim = outDims.Length;
            var bshape = new long[ndim];
            Array.Fill(bshape, 1L);
            // Operand 0, the out: its op_axes name axes 0 .. ndim-1, and an axis past its rank is invalid.
            for (int idim = 0; idim < ndim; idim++)
            {
                if (idim >= ondim)
                    return new IncorrectShapeException(
                        $"Iterator input op_axes[0][{ndim - idim - 1}] (=={idim}) is not a valid axis of op[0], which has {ondim} dimensions ");
                if (bshape[idim] == 1)
                    bshape[idim] = outDims[idim];
            }
            // Operand 1, the input: an extent that neither matches nor is 1 does not broadcast.
            for (int idim = 0; idim < ndim; idim++)
            {
                long b = bshape[idim], s = inDims[idim];
                if (b == 1)
                    bshape[idim] = s;
                else if (b != s && s != 1)
                {
                    var sb = new StringBuilder("operands could not be broadcast together with remapped shapes [original->remapped]: ");
                    var outRemapped = new long[ndim];
                    Array.Copy(outDims, outRemapped, ndim);
                    sb.Append(Shape.ConvertShapeToString(outDims)).Append("->").Append(Shape.ConvertShapeToString(outRemapped)).Append(' ');
                    sb.Append(Shape.ConvertShapeToString(inDims)).Append("->").Append(Shape.ConvertShapeToString(inDims)).Append(' ');
                    return new IncorrectShapeException(sb.ToString());
                }
            }
            // The out never broadcasts: an extent of 1 where the broadcast is longer (NumPy fills the axes last first, and
            // the text does not depend on which axis fails).
            for (int idim = 0; idim < ndim; idim++)
            {
                if (bshape[idim] != 1 && outDims[idim] == 1)
                {
                    var reversed = new long[ndim];
                    for (int k = 0; k < ndim; k++)
                        reversed[k] = outDims[ndim - k - 1];
                    return new IncorrectShapeException(
                        $"non-broadcastable output operand with shape {Shape.ConvertShapeToString(outDims)} [remapped to " +
                        $"{Shape.ConvertShapeToString(reversed)}] doesn't match the broadcast shape {Shape.ConvertShapeToString(bshape)}");
                }
            }
            return null;
        }

        /// <summary>
        ///     The loop dtype of add.accumulate / multiply.accumulate (<see cref="ReduceCumAdd"/>,
        ///     <see cref="ReduceCumMul"/>): the <c>dtype=</c> request; else, with an <c>out=</c>,
        ///     <c>promote_types(out.dtype, arr.dtype)</c> WITHOUT NEP 50's widening of small integers
        ///     (<c>reducelike_promote_and_resolve</c>, the reductions' rule of <see cref="ReduceIntoOut"/>); else null, which
        ///     selects the input's NEP 50 accumulator.
        /// </summary>
        /// <param name="arr">The accumulated array (numeric: the scans' text routes answered before this is read).</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> array, or null.</param>
        /// <returns>The loop dtype, or null for the NEP 50 accumulator.</returns>
        /// <remarks>
        ///     <para>NumPy 2.4.2: int8 [100, 100, 56] accumulated into a bool out runs the int8 loop, whose running sum wraps to
        ///     0 at the third element (stored False; the int64 accumulator stored True there); float32 [16777216, 1, 1] into a
        ///     float64 out adds in float64 (16777217, 16777218); float16 [300, 300] multiplied into a float32 out gives 90000
        ///     (the float16 accumulator overflowed to inf); uint64 [2^63 + 1, 0] into an int64 out runs the float64 loop
        ///     (stored INT64_MIN twice, where the uint64 accumulator stored 2^63 + 1's bits).</para>
        ///     <para>The running value stays in the loop dtype across NumPy's buffer fills (an int8 run of 24,576 elements into
        ///     a bool out is False at every third element), so accumulating the whole run in the loop and storing it once
        ///     (<see cref="WriteScanToOut"/>) is NumPy's answer; the reductions have a residual there that accumulations do
        ///     not.</para>
        /// </remarks>
        private static DType AccumulateLoop(NDArray arr, DType dtype, NDArray @out)
            => dtype ?? (@out is null ? null : DTypePromotion.PromoteTypes(@out.dtype, arr.dtype));
    }
}
