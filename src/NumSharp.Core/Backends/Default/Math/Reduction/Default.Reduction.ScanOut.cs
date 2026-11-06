using System;
using System.Globalization;
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
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have <paramref name="result"/>'s shape and
        ///     NumPy's accumulate refuses it: the text of the statement that refuses (<see cref="StoreIntoMisfitOut"/>; parity
        ///     QA-31), the house shape exception standing in for NumPy's ValueError. The outs NumPy accepts are stored
        ///     (parity QA-54).</exception>
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
                // Another shape: PyUFunc_Accumulate's path decides between a refusal (its statement's text) and a store into
                // the cells its iterator walks.
                return StoreIntoMisfitOut(result, @out, arr, axis, loop);
            UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_UNSAFE_CASTING);
            return @out;
        }

        /// <summary>
        ///     NumPy's <c>PyUFunc_Accumulate</c> for an <c>out=</c> whose dimensions are not the accumulation's, on the path it
        ///     takes. Its outer ITERATOR (an input of more than one dimension, a loop dtype other than the input's or the
        ///     out's, an unaligned operand, an out whose bounds overlap the input's) refuses with NpyIter's texts
        ///     (<see cref="AccumulateIteratorShapeError"/>: <c>operands could not be broadcast together with remapped shapes
        ///     [original-&gt;remapped]: (4,3)-&gt;(4,3) (3,4)-&gt;(3,4) </c> and its two siblings; parity QA-31) and otherwise
        ///     WRITES the cells it walks; the DIRECT 1-D path refuses with <c>provided out is the wrong size for the
        ///     accumulation.</c>. On both paths an empty input along the axis or an empty out returns the out untouched,
        ///     before the direct path compares the dimensions (parity QA-54).
        /// </summary>
        /// <param name="result">The accumulation over the input (the input's dimensions; 1-D for the flattened form).</param>
        /// <param name="out">The caller's writeable out (its dimensions differ from <paramref name="result"/>'s).</param>
        /// <param name="arr">The array as the caller passed it (before any flattening).</param>
        /// <param name="axis">The accumulated axis (already validated), or null for the flattened array.</param>
        /// <param name="loop">The loop dtype (NumPy's resolved <c>descrs</c>: all three are this dtype for add / multiply).</param>
        /// <returns><paramref name="out"/> itself, written where NumPy's iterator writes it, or untouched.</returns>
        /// <exception cref="IncorrectShapeException">An out NumPy's accumulate refuses: the iterator's texts, or the direct
        ///     path's wrong-size text.</exception>
        /// <remarks>
        ///     <para>
        ///     The iterator's writes (op_axes naming the input's axes for both operands, then the accumulated axis removed
        ///     and walked by the inner loop over the INPUT's extent): an out with more dimensions than the input keeps its
        ///     extra trailing axes at index 0; along the accumulated axis only the input's extent is written (an input of
        ///     extent 1 writes the first cell, copying the input's first element); along the other axes an input of extent 1
        ///     broadcasts against the out's longer extent. Accumulating first and broadcasting the result after is the same
        ///     value, since the broadcast axis is not the accumulated one. The cells outside keep the out's contents. The
        ///     StringDType accumulation (<see cref="Strings.NDVStringReduction"/>) shares the views.
        ///     </para>
        ///     <para>
        ///     A 0-d out on the direct path crashes NumPy 2.4.2 (it reads the out's stride along the axis before its empty
        ///     exits); the direct path's text is raised instead.
        ///     </para>
        /// </remarks>
        internal static NDArray StoreIntoMisfitOut(NDArray result, NDArray @out, NDArray arr, int? axis, DType loop)
        {
            // PyArray_CheckAxis: axis=None, and any axis of a 0-d input, accumulate the C-order RAVEL, a 1-D array that is a
            // view of a C-contiguous input (its memory, so its alignment and its overlap with the out) and a fresh aligned
            // copy otherwise (it shares nothing). Every other call accumulates the input itself.
            bool raveled = axis is null || arr.ndim == 0;
            long[] inDims = raveled ? new[] { arr.size } : arr.Shape.dimensions;
            int accAxis = raveled ? 0 : NormalizeAxis(axis.Value, arr.ndim);
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
            if (iterator)
            {
                // NpyIter_AdvancedNew refuses before anything runs.
                if (AccumulateIteratorShapeError(@out.Shape.dimensions ?? Array.Empty<long>(), inDims) is { } refusal)
                    throw refusal;
            }
            else if (@out.ndim == 0)
            {
                // The direct path reads the out's stride along the axis first, which a 0-d out lacks: NumPy 2.4.2 crashes
                // there, for an empty input too. Its text for an out of the wrong size is NumSharp's answer.
                throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");
            }

            // `if (PyArray_DIM(op[1], axis) == 0) goto finish; else if (PyArray_SIZE(op[0]) == 0) goto finish;` — both paths,
            // before the direct path compares the dimensions: nothing is written and the out comes back.
            long count = inDims[accAxis];
            if (count == 0 || @out.size == 0)
                return @out;
            if (!iterator)
                throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");

            // The cells the iterator walks: the out's leading input-rank axes (later ones at index 0), the accumulated axis cut
            // to the input's extent; the result broadcasts along the axes where the input had extent 1.
            NDArray view = Strings.NDVStringReduction.LeadingAxesView(@out, inDims.Length);
            NDArray target = view.Shape.dimensions[accAxis] == count ? view : Strings.NDVStringReduction.AxisPrefixView(view, accAxis, count);
            try
            {
                // The loop wrote in its own dtype and NumPy's UPDATEIFCOPY copy-back casts it unsafely, as the full store does.
                UnmanagedStorage.Copyto(target, result, NPY_CASTING.NPY_UNSAFE_CASTING);
            }
            finally
            {
                // The views are this call's own; the caller's out is returned.
                if (!ReferenceEquals(target, view))
                    target.Dispose();
                if (!ReferenceEquals(view, @out))
                    view.Dispose();
            }

            return @out;
        }

        /// <summary>
        ///     The shape errors of <c>PyUFunc_Accumulate</c>'s iterator over (out, input) with <c>op_axes</c>
        ///     <c>[0 .. ndim-1]</c> for both, in NpyIter's order: the out's <c>op_axes</c> entries first (an out with fewer
        ///     dimensions than the input), then the broadcast of each operand's extents, then the out's NO_BROADCAST check.
        ///     Shared by the numeric scans (<see cref="StoreIntoMisfitOut"/>) and the StringDType accumulation.
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
                        string.Create(CultureInfo.InvariantCulture, $"Iterator input op_axes[0][{ndim - idim - 1}] (=={idim}) is not a valid axis of op[0], which has {ondim} dimensions "));
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
