using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    // ============================== np.diff ==============================
    // Calculate the n-th discrete difference along the given axis.
    //
    //   out[i] = a[i+1] - a[i]   (repeated n times, recursively)
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::diff
    //
    // Implementation mirrors NumPy's structure exactly:
    //   1. Optionally concatenate [prepend, a, append] along the axis
    //      (scalar prepend/append broadcast to length 1 along the axis).
    //   2. Repeat n times:  a = op(a[1:], a[:-1])  along the axis,
    //      where op is `!=` (not_equal) for boolean arrays and `-`
    //      (subtract) for everything else.
    //
    // Both `op`s are backed by NumSharp's SIMD IL kernels (TensorEngine
    // .Subtract / .NotEqual), so the element-wise loop runs through the
    // ILKernelGenerator path — diff itself contains no per-element loop.
    // The two operands `a[1:]` and `a[:-1]` are overlapping strided views of
    // the same buffer; the kernel reads them and writes a fresh output, so
    // there is no read/write aliasing hazard.
    public static partial class np
    {
        /// <summary>
        ///     Calculate the n-th discrete difference along the given axis.
        ///     The first difference is <c>out[i] = a[i+1] - a[i]</c>; higher
        ///     differences are computed recursively.
        /// </summary>
        /// <param name="a">Input array (must be at least one dimensional).</param>
        /// <param name="n">
        ///     The number of times values are differenced. If zero, the input
        ///     is returned as-is. Must be non-negative.
        /// </param>
        /// <param name="axis">
        ///     The axis along which the difference is taken; default is the
        ///     last axis. Negative axes count from the end.
        /// </param>
        /// <param name="prepend">
        ///     Value(s) to prepend to <paramref name="a"/> along
        ///     <paramref name="axis"/> prior to differencing. Scalars expand to
        ///     length 1 along the axis. <c>null</c> means "not supplied"
        ///     (NumPy's <c>np._NoValue</c>).
        /// </param>
        /// <param name="append">
        ///     Value(s) to append to <paramref name="a"/> along
        ///     <paramref name="axis"/> prior to differencing. Scalars expand to
        ///     length 1 along the axis. <c>null</c> means "not supplied".
        /// </param>
        /// <returns>
        ///     The n-th differences. The shape matches the (optionally
        ///     prepend/append-extended) input except along <paramref name="axis"/>
        ///     where the size shrinks by <paramref name="n"/>. The dtype is
        ///     preserved (boolean input yields boolean output via not_equal).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.diff.html</remarks>
        public static NDArray diff(NDArray a, int n = 1, int axis = -1,
                                   object prepend = null, object append = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // n == 0 returns the input unchanged (NumPy returns the same object).
            if (n == 0) return a;
            if (n < 0)
                throw new ArgumentException(
                    $"order must be non-negative but got {n}", nameof(n));

            int nd = a.ndim;
            if (nd == 0)
                throw new ArgumentException(
                    "diff requires input that is at least one dimensional");

            // normalize_axis_index(axis, nd)
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {nd}");

            // Build [prepend?, a, append?] and concatenate along the axis when
            // anything was supplied. `a` itself is never disposed here.
            NDArray work;
            bool workOwned;
            if (prepend is null && append is null)
            {
                work = a;
                workOwned = false;
            }
            else
            {
                var parts = new List<NDArray>(3);
                var toDispose = new List<NDArray>(4);
                if (prepend is not null)
                    parts.Add(DiffPrepareEnd(prepend, a, ax, toDispose));
                parts.Add(a);
                if (append is not null)
                    parts.Add(DiffPrepareEnd(append, a, ax, toDispose));

                work = np.concatenate(parts.ToArray(), ax);
                workOwned = true;

                // Dispose intermediates view-before-owner (reverse insertion).
                for (int i = toDispose.Count - 1; i >= 0; i--)
                    toDispose[i].Dispose();
            }

            // op = not_equal if a.dtype == bool else subtract — decided from the
            // POST-concatenation dtype (a bool array with an int prepend promotes
            // to int and therefore subtracts).
            bool useNotEqual = work.GetTypeCode == NPTypeCode.Boolean;

            NDArray current = work;
            bool currentOwned = workOwned;
            for (int it = 0; it < n; it++)
            {
                long len = current.shape[ax];
                long m = len > 0 ? len - 1 : 0;                      // result length along axis
                var hi = SliceAlongAxis(current, ax, len - m, len);  // a[1:]  (last  m)
                var lo = SliceAlongAxis(current, ax, 0, m);          // a[:-1] (first m)
                // Bool diffs via not_equal (the `!=` IL kernel). Numeric diffs go
                // through the lean NDIter subtract (DiffSubtractViaNDIter), which
                // writes into an uninitialised output and skips the type-promotion /
                // broadcast / F-analysis the `-` operator would re-derive — operands
                // here are always equal-shape, equal-dtype, non-broadcast. Falls back
                // to the `-` operator for any dtype the kernel emitter rejects.
                NDArray next = useNotEqual
                    ? (hi != lo)
                    : (DiffSubtractViaNDIter(hi, lo) ?? (hi - lo));
                hi.Dispose();
                lo.Dispose();
                if (currentOwned) current.Dispose();
                current = next;
                currentOwned = true;
            }

            return current;
        }

        /// <summary>
        ///     Normalises a prepend/append operand: converts scalars/array-likes
        ///     to an <see cref="NDArray"/>, and broadcasts 0-D values to
        ///     <paramref name="a"/>'s shape with the diff axis set to length 1
        ///     (NumPy expands scalar prepend/append to length-1 along the axis).
        ///     Any array allocated here is registered in
        ///     <paramref name="toDispose"/> for cleanup after the concatenate.
        /// </summary>
        private static NDArray DiffPrepareEnd(object value, NDArray a, int axis, List<NDArray> toDispose)
        {
            NDArray v;
            if (value is NDArray nd)
                v = nd;                       // caller-owned; do not dispose
            else
            {
                v = np.asanyarray(value);
                toDispose.Add(v);
            }

            if (v.ndim == 0)
            {
                long[] dims = new long[a.ndim];
                for (int i = 0; i < a.ndim; i++) dims[i] = a.shape[i];
                dims[axis] = 1;
                var bcast = np.broadcast_to(v, new Shape(dims));
                toDispose.Add(bcast);
                return bcast;
            }

            return v;
        }

        // [hi(READONLY), lo(READONLY), out(WRITEONLY)] operand flags, hoisted so
        // the per-iteration subtract doesn't re-allocate the flags array.
        private static readonly NDIterPerOpFlags[] _diffRRW =
        {
            NDIterPerOpFlags.READONLY,
            NDIterPerOpFlags.READONLY,
            NDIterPerOpFlags.WRITEONLY,
        };

        /// <summary>
        ///     Lean NDIter subtract used by the diff loop: computes
        ///     <c>hi - lo</c> into a freshly-allocated, <b>uninitialised</b>
        ///     C-contiguous output via the NDIter Tier-3B inner-loop kernel
        ///     (4×-unrolled SIMD + scalar-strided shell). <paramref name="hi"/>
        ///     and <paramref name="lo"/> are always equal-shape, equal-dtype and
        ///     non-broadcast, so this skips the type-promotion, broadcast
        ///     resolution and F-contig analysis that <see cref="DefaultEngine"/>'s
        ///     general binary path performs. Returns <c>null</c> when the kernel
        ///     emitter rejects the dtype, signalling the caller to fall back to
        ///     the <c>-</c> operator.
        /// </summary>
        private static unsafe NDArray DiffSubtractViaNDIter(NDArray hi, NDArray lo)
        {
            NPTypeCode dt = hi.GetTypeCode;

            // Fresh C-contiguous, uninitialised output with hi's dimensions.
            int nd = hi.ndim;
            long[] dims = new long[nd];
            for (int i = 0; i < nd; i++) dims[i] = hi.shape[i];
            var outp = new NDArray(hi.dtype, new Shape(dims), false);

            // Empty result: the NDIter element-wise path must not run over zero
            // elements (it walks broadcast dims as if non-empty). Nothing to do.
            if (outp.size == 0) return outp;

            bool simd = DirectILKernelGenerator.CanUseSimd(dt)
                        && DirectILKernelGenerator.CanUseSimdForOp(BinaryOp.Subtract);
            Action<ILGenerator> scalarBody =
                il => DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Subtract, dt);
            Action<ILGenerator> vectorBody = simd
                ? il => DirectILKernelGenerator.EmitVectorOperation(il, BinaryOp.Subtract, dt)
                : null;

            try
            {
                using var iter = NDIterRef.MultiNew(
                    3, new[] { hi, lo, outp },
                    NDIterGlobalFlags.EXTERNAL_LOOP,
                    NPY_ORDER.NPY_CORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    _diffRRW);

                iter.ExecuteElementWiseBinary(dt, dt, dt, scalarBody, vectorBody, DiffSubKey(dt));
            }
            catch (NotSupportedException)
            {
                outp.Dispose();
                return null; // fall back to the `-` operator
            }

            return outp;
        }

        /// <summary>
        ///     Allocation-free, per-dtype kernel cache key for the diff subtract.
        ///     Returns interned literals so the hot loop never allocates a string.
        /// </summary>
        private static string DiffSubKey(NPTypeCode dt) => dt switch
        {
            NPTypeCode.Byte => "npy_diff_sub_Byte",
            NPTypeCode.SByte => "npy_diff_sub_SByte",
            NPTypeCode.Int16 => "npy_diff_sub_Int16",
            NPTypeCode.UInt16 => "npy_diff_sub_UInt16",
            NPTypeCode.Int32 => "npy_diff_sub_Int32",
            NPTypeCode.UInt32 => "npy_diff_sub_UInt32",
            NPTypeCode.Int64 => "npy_diff_sub_Int64",
            NPTypeCode.UInt64 => "npy_diff_sub_UInt64",
            NPTypeCode.Char => "npy_diff_sub_Char",
            NPTypeCode.Half => "npy_diff_sub_Half",
            NPTypeCode.Single => "npy_diff_sub_Single",
            NPTypeCode.Double => "npy_diff_sub_Double",
            NPTypeCode.Decimal => "npy_diff_sub_Decimal",
            NPTypeCode.Complex => "npy_diff_sub_Complex",
            _ => "npy_diff_sub_" + dt,
        };
    }
}
