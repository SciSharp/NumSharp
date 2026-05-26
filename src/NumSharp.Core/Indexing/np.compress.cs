using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return selected slices of <paramref name="a"/> along the given
        ///     <paramref name="axis"/> at positions where the 1-D
        ///     <paramref name="condition"/> is truthy.
        /// </summary>
        /// <param name="condition">
        ///     <strong>1-D</strong> array of booleans (or any dtype interpreted as
        ///     truthy). Must be 1-D — a 2-D or 0-D condition raises
        ///     <see cref="ArgumentException"/>, mirroring NumPy's
        ///     <c>ValueError("condition must be a 1-d array")</c>. If
        ///     <c>len(condition) &lt; a.shape[axis]</c>, only the first
        ///     <c>len(condition)</c> positions along <paramref name="axis"/> are
        ///     considered; if longer, any True beyond <c>a.shape[axis]</c> raises
        ///     <see cref="IndexOutOfRangeException"/>.
        /// </param>
        /// <param name="a">Source array.</param>
        /// <param name="axis">
        ///     Axis along which to slice. <c>null</c> (default) flattens
        ///     <paramref name="a"/> first.
        /// </param>
        /// <param name="out">
        ///     Optional destination. When supplied, shape must match the natural
        ///     output and <c>out.dtype</c> must be safely castable to
        ///     <c>a.dtype</c>; values are written via <see cref="np.copyto"/> with
        ///     unsafe casting and the method returns <paramref name="out"/> itself
        ///     (matches NumPy's out= dispatch via PyArray_TakeFrom).
        /// </param>
        /// <returns>
        ///     A copy of <paramref name="a"/> without the slices along
        ///     <paramref name="axis"/> for which <paramref name="condition"/> is
        ///     false. Dtype matches <paramref name="a"/> (or
        ///     <paramref name="out"/>'s dtype when supplied).
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.compress.html
        ///     <para>
        ///     Two execution paths:
        ///     <list type="bullet">
        ///       <item>
        ///         <b>Fast path</b>: bool condition, contig source, no out= or
        ///         out= matches src.dtype, <c>condition.size &lt;= a.shape[axis]</c>.
        ///         Runs the fused mask-driven gather kernel (popcount → alloc →
        ///         SIMD bit-scan + cpblk-per-outer-slab), skipping the
        ///         flatnonzero indices NDArray.
        ///       </item>
        ///       <item>
        ///         <b>Generic path</b>: mirrors NumPy's
        ///         <c>PyArray_Compress</c> chain — flatnonzero(cond) → take(a,
        ///         indices, axis, out, "raise"). Handles non-bool conditions,
        ///         out= with safe dtype cast, and OOB-True via take's RAISE mode.
        ///       </item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     When <paramref name="condition"/> is 1-D <see cref="np.extract"/> with
        ///     <paramref name="axis"/> = <c>null</c> is equivalent.
        ///     </para>
        /// </remarks>
        public static NDArray compress(NDArray condition, NDArray a, int? axis = null, NDArray @out = null)
        {
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy hard-requires 1-D condition; 0-D and 2-D+ both fail here.
            if (condition.ndim != 1)
                throw new ArgumentException(
                    "condition must be a 1-d array",
                    nameof(condition));

            // Fast path: bool cond, contig src, no out= override (so result dtype
            // == src dtype). Wires straight into the fused gather kernel.
            if (TryCompressFast(condition, a, axis, @out, out var fast))
                return fast;

            // Generic path: flatnonzero handles non-bool / strided cond; take
            // handles non-contig src, out= dispatch with safe-cast validation,
            // and OOB-True via RAISE mode.
            var indices = np.flatnonzero(condition);
            try
            {
                return np.take(a, indices, axis, @out);
            }
            finally
            {
                indices.Dispose();
            }
        }

        /// <summary>
        ///     Fused fast path for <c>np.compress</c>. Computes outer/inner
        ///     factorisation around the requested axis and runs the
        ///     <see cref="FilterAxisKernel"/> in one pass. Returns <c>false</c>
        ///     when preconditions aren't met (bool cond, contig src, etc.) so
        ///     the caller falls through to the flatnonzero+take chain.
        /// </summary>
        private static unsafe bool TryCompressFast(
            NDArray condition, NDArray a, int? axis, NDArray @out, out NDArray result)
        {
            result = null;
            if (condition.GetTypeCode != NPTypeCode.Boolean) return false;
            if (!condition.Shape.IsContiguous) return false;
            if (!a.Shape.IsContiguous) return false;

            // out= path needs the take-level out dispatch (shape check + safe-cast).
            // Skip fast path when out is supplied with mismatched dtype; for matching
            // dtype we'd still need shape validation, so just defer to generic path.
            if (!(@out is null)) return false;

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(typeof(bool));
            // innerSize bucketing is decided once we know the axis factorisation.
            // For axis=k with innerCount>1 the slab spans multiple elements; we
            // pass innerCount*elemBytes — typically large, lands in the bulk
            // cpblk variant. For axis=None / 1-D src / axis=last the inner is
            // just one element (typed copy variant — significantly faster).
            if (countKernel == null) return false;

            // Compute outerSize × axisSize × innerSize factorisation around axis.
            // axis=null flattens a (treat as 1-D of a.size).
            // 0-d source with axis=null → 1-element 1-D pseudo-axis (handled below).
            long outerSize, axisSize, innerCount;
            long[] outDims;

            if (a.ndim == 0)
            {
                // 0-d treated as 1-element 1-D; behaviour identical to axis=None.
                outerSize = 1;
                axisSize = 1;
                innerCount = 1;
                outDims = new long[1] { 0 }; // placeholder; will overwrite with count below
            }
            else if (axis is null)
            {
                outerSize = 1;
                axisSize = a.size;
                innerCount = 1;
                outDims = new long[1] { 0 };
            }
            else
            {
                int ax = axis.Value;
                if (ax < 0) ax += a.ndim;
                if (ax < 0 || ax >= a.ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis),
                        $"axis {axis.Value} is out of bounds for array of dimension {a.ndim}");

                outerSize = 1;
                for (int d = 0; d < ax; d++) outerSize *= a.Shape.dimensions[d];
                axisSize = a.Shape.dimensions[ax];
                innerCount = 1;
                for (int d = ax + 1; d < a.ndim; d++) innerCount *= a.Shape.dimensions[d];

                // Output shape: a.shape with axis dim replaced by N (count).
                outDims = new long[a.ndim];
                for (int d = 0; d < a.ndim; d++) outDims[d] = a.Shape.dimensions[d];
            }

            long elemBytes = a.dtypesize;
            long innerBytes = innerCount * elemBytes;
            long maskSize = condition.size;
            long effectiveScan = Math.Min(maskSize, axisSize);
            byte* maskPtr = (byte*)condition.Storage.Address + condition.Shape.offset;

            // OOB-True detection: cond longer than axisSize and any True past
            // axisSize would index OOB on `a` → mirror IndexError.
            if (maskSize > axisSize)
            {
                long tailTrues = countKernel(maskPtr + axisSize, maskSize - axisSize);
                if (tailTrues > 0)
                    throw new IndexOutOfRangeException(
                        $"index {axisSize} is out of bounds for axis with size {axisSize}");
            }

            // Pass 1: popcount mask over the effective scan range.
            long count = countKernel(maskPtr, effectiveScan);

            // Build output shape: axis dim → count. For axis=None / 0-d, shape is (count,).
            if (a.ndim == 0 || axis is null)
                outDims = new long[1] { count };
            else
            {
                int ax = axis.Value < 0 ? axis.Value + a.ndim : axis.Value;
                outDims[ax] = count;
            }

            result = new NDArray(a.typecode, new Shape(outDims), false);

            if (count == 0 || outerSize == 0 || innerCount == 0)
                return true;

            // Source pointer with the (always-contig here) offset; 0-d source has
            // no offset stride contribution.
            byte* srcPtr = (byte*)a.Storage.Address + a.Shape.offset * elemBytes;
            byte* dstPtr = (byte*)result.Storage.Address;

            long srcOuterStride = axisSize * innerBytes;
            long dstOuterStride = count * innerBytes;

            // Pick the typed-or-bulk variant by innerBytes — single-element
            // slabs (innerCount==1) lift to the typed Ldind/Stind path.
            var filterKernel = DirectILKernelGenerator.GetFilterAxisKernel(innerBytes);
            if (filterKernel == null) return false;

            long written = filterKernel(
                srcPtr, maskPtr, effectiveScan,
                outerSize, srcOuterStride, dstOuterStride,
                innerBytes,
                dstPtr);

            if (written != count)
                throw new InvalidOperationException(
                    $"compress: filter kernel wrote {written}, expected {count}");

            return true;
        }
    }
}
