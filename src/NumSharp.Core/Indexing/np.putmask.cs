using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="a"/> based on a boolean mask and
        ///     input values. Where <paramref name="mask"/> is true (walked in
        ///     C-order), writes <c>a.flat[i] = values.flat[i % values.size]</c>.
        ///     In-place.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="a"/> — NumPy allows shape mismatch as long as element
        ///     counts match. Non-bool masks are cast to bool (<c>!= 0</c>; NaN/inf → true).
        /// </param>
        /// <param name="values">
        ///     Values to write, cast to <paramref name="a"/>'s dtype. Cycled by
        ///     <em>position</em>: the value written at flat index <c>i</c> is
        ///     <c>values.flat[i % values.size]</c> (the cursor advances on every
        ///     element, unlike <see cref="place"/> which advances only on True
        ///     positions). An empty <paramref name="values"/> is a no-op — NumPy
        ///     returns without touching <paramref name="a"/> (unlike
        ///     <see cref="place"/>, which raises).
        /// </param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.putmask.html
        ///     <para>
        ///     Differs from <see cref="place"/> in two probed ways (NumPy 2.4.2):
        ///     the values cursor advances by position (every element) rather than per
        ///     True, and an empty <paramref name="values"/> is a silent no-op rather
        ///     than a <c>ValueError</c>.
        ///     </para>
        /// </remarks>
        public static void putmask(NDArray a, NDArray mask, NDArray values)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (mask is null) throw new ArgumentNullException(nameof(mask));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // Read-only target reported FIRST — NumPy checks FailUnlessWriteable right after the
            // is-an-array check, before the mask-size test and before the nv==0 no-op (probed 2.4.2:
            // putmask(ro, all-false-mask, v) and putmask(ro, mask, []) both raise read-only). The
            // contiguous kernel writes a.Storage.Address directly; the non-contiguous path is guarded
            // via np.copyto below.
            NumSharpException.ThrowIfNotWriteable(a.Shape, "putmask: output array");

            if (mask.size != a.size)
                throw new ArgumentException(
                    "putmask: mask and data must be the same size",
                    nameof(mask));

            // Empty a ⇒ empty mask ⇒ no-op (the loop runs zero times in NumPy).
            if (a.size == 0)
                return;

            // Empty values ⇒ no-op. NumPy: nv <= 0 returns None without writing (probed 2.4.2).
            // Checked after writeable + mask-size (NumPy's order), before the contig copy so an
            // empty-values call never allocates a scratch.
            if (values.size == 0)
                return;

            // NumPy WRITEBACKIFCOPY / ENSURECOPY semantics (PyArray_PutMask): copy `a` when it is
            // non-contiguous OR may share memory with `values`/`mask`. NumPy's `arrays_overlap` is
            // `solve_may_share_memory(..., NPY_MAY_SHARE_BOUNDS)` == `SolveMayShareMemory(maxWork:0)`
            // (a cheap bounds-only check: `!= No` means "may share"). The fresh scratch cannot alias
            // the read operands, so a write into it never disturbs a later cyclic read of `values`
            // that aliases `a` — e.g. `putmask(a, mask, a[2:8])`, where the contiguous fast path
            // would write a[i] and then read it back as the wrapped `values[i % nv]` and diverge.
            // The kernel writes the scratch in C-order flat (matching NumPy's copy-and-writeback);
            // np.copyto pushes the result back through `a`'s strides to the parent storage. The
            // recursive call sees a fresh scratch that overlaps neither operand, so it takes the
            // direct path (no re-copy, no recursion).
            bool overlaps =
                NDMemOverlap.SolveMayShareMemory(a, values, maxWork: 0) != MemOverlap.No ||
                NDMemOverlap.SolveMayShareMemory(a, mask, maxWork: 0) != MemOverlap.No;
            if (overlaps || !a.Shape.IsContiguous)
            {
                // C-contiguous a (the overlap case) → a.copy() is a fresh C-order buffer;
                // any non-contiguous layout → ascontiguousarray makes the C-order copy.
                var scratch = a.Shape.IsContiguous ? a.copy() : np.ascontiguousarray(a);
                try
                {
                    putmask(scratch, mask, values);
                    np.copyto(a, scratch, casting: "unsafe");
                }
                finally { scratch.Dispose(); }
                return;
            }

            // Cast mask to contig bool; cast values to contig a.dtype. Both are raveled in C-order,
            // aligning flat position i of the mask with flat position i of a (shapes need not match,
            // only sizes — NumPy checks size, not shape).
            NDArray maskCast;
            bool ownMask;
            if (mask.GetTypeCode == NPTypeCode.Boolean && mask.Shape.IsContiguous)
            {
                maskCast = mask;
                ownMask = false;
            }
            else
            {
                maskCast = mask.GetTypeCode == NPTypeCode.Boolean
                    ? np.ascontiguousarray(mask)
                    : mask.astype(NPTypeCode.Boolean);
                ownMask = !ReferenceEquals(maskCast, mask);
            }

            NDArray valsCast;
            bool ownVals;
            if (values.GetTypeCode == a.GetTypeCode && values.Shape.IsContiguous)
            {
                valsCast = values;
                ownVals = false;
            }
            else if (values.GetTypeCode == a.GetTypeCode)
            {
                var c = np.ascontiguousarray(values);
                valsCast = c;
                ownVals = !ReferenceEquals(c, values);
            }
            else
            {
                valsCast = values.astype(a.GetTypeCode);
                ownVals = true;
            }

            try
            {
                ExecutePutMask(a, maskCast, valsCast);
            }
            finally
            {
                if (ownMask) maskCast.Dispose();
                if (ownVals) valsCast.Dispose();
            }
        }

        private static unsafe void ExecutePutMask(NDArray a, NDArray maskBool, NDArray valsCast)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);
            var kernel = DirectILKernelGenerator.GetPutMaskKernel(copyKind);
            if (kernel == null)
                throw new NotSupportedException("np.putmask: IL kernel unavailable");

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, a.size, valsPtr, valsCast.size, a.dtypesize);
        }
    }
}
