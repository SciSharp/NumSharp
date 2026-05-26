using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the elements of <paramref name="arr"/> that satisfy some
        ///     <paramref name="condition"/>. Equivalent to
        ///     <c>np.take(np.ravel(arr), np.flatnonzero(np.ravel(condition)))</c> —
        ///     i.e. <c>arr.ravel()[condition.ravel()]</c> when condition is boolean.
        /// </summary>
        /// <param name="condition">
        ///     Array whose nonzero / True entries indicate the elements of
        ///     <paramref name="arr"/> to extract. May be any dtype (treated as
        ///     truthy via NumPy's "nonzero" semantics). May be any shape — it is
        ///     ravel'd before alignment with <paramref name="arr"/>.
        /// </param>
        /// <param name="arr">Input array. May be any shape; it is ravel'd.</param>
        /// <returns>
        ///     Rank-1 <see cref="NDArray"/> of values from <paramref name="arr"/>
        ///     where the corresponding ravel'd <paramref name="condition"/> entry
        ///     is truthy. Dtype matches <paramref name="arr"/>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.extract.html
        ///     <para>
        ///     Two execution paths:
        ///     <list type="bullet">
        ///       <item>
        ///         <b>Fast path</b>: bool condition, contig arr + condition,
        ///         <c>condition.size &lt;= arr.size</c>. Runs a fused IL kernel
        ///         (popcount → alloc → SIMD bit-scan + cpblk) avoiding the
        ///         indices NDArray allocation that the generic path needs.
        ///       </item>
        ///       <item>
        ///         <b>Generic path</b>: mirrors NumPy's literal chain
        ///         <c>take(ravel(arr), flatnonzero(ravel(condition)))</c>. Handles
        ///         non-bool conditions (any dtype interpreted as nonzero), broadcast
        ///         / strided / negative-stride sources, and the OOB-True case
        ///         (raises via take's RAISE mode).
        ///       </item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     Note that <see cref="place"/> is the inverse operation.
        ///     </para>
        /// </remarks>
        public static NDArray extract(NDArray condition, NDArray arr)
        {
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            // Fast path: bool cond + contig source/cond + cond.size <= arr.size.
            // The kernel walks the mask and gathers from src in one pass with no
            // intermediate indices materialisation.
            if (TryExtractFast(condition, arr, out var fast))
                return fast;

            // Generic path: flatnonzero handles non-bool / 0-d / strided cond
            // correctly (includes ravel internally); take handles non-contig arr
            // and OOB-True via its RAISE mode.
            var indices = np.flatnonzero(condition);
            try
            {
                var flatArr = np.ravel(arr);
                return np.take(flatArr, indices);
            }
            finally
            {
                indices.Dispose();
            }
        }

        /// <summary>
        ///     Fused fast path for <c>np.extract</c>. Skips index materialisation
        ///     and runs a single IL kernel (popcount + SIMD bit-scan + cpblk) on
        ///     the bool mask + contig source. Returns <c>false</c> when the
        ///     preconditions aren't met so the caller falls back to the generic
        ///     path (flatnonzero → take).
        /// </summary>
        private static unsafe bool TryExtractFast(NDArray condition, NDArray arr, out NDArray result)
        {
            result = null;
            if (condition.GetTypeCode != NPTypeCode.Boolean) return false;

            // Materialise to a contig 1-D view of the bool mask and source. The
            // raveled views share storage with their parents when contig; otherwise
            // they allocate a fresh contig copy (which we Dispose at the end).
            // For non-contig parents the generic path is already correct and not
            // markedly slower than copying, so we skip the fast path there.
            if (!condition.Shape.IsContiguous) return false;
            if (!arr.Shape.IsContiguous) return false;

            long maskSize = condition.size;
            long arrSize = arr.size;

            // OOB-True detection: if cond is longer than arr, any True at index
            // >= arr.size triggers an out-of-range read. We mirror NumPy's
            // IndexError. Cheap check: popcount the tail. If it's >0, raise.
            long effectiveScan = Math.Min(maskSize, arrSize);

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(typeof(bool));
            var filterKernel = DirectILKernelGenerator.GetFilterAxisKernel(arr.dtypesize);
            if (countKernel == null || filterKernel == null) return false;

            byte* maskPtr = (byte*)condition.Storage.Address + condition.Shape.offset;

            if (maskSize > arrSize)
            {
                long tailTrues = countKernel(maskPtr + arrSize, maskSize - arrSize);
                if (tailTrues > 0)
                    throw new IndexOutOfRangeException(
                        $"index {arrSize} is out of bounds for axis 0 with size {arrSize}");
            }

            // Pass 1: popcount mask over the effective scan range.
            long count = countKernel(maskPtr, effectiveScan);

            // Allocate result: shape (count,) with arr's dtype, C-contig.
            result = new NDArray(arr.typecode, new Shape(count), false);

            if (count == 0)
                return true;

            // Pass 2: fused gather.
            byte* srcPtr = (byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize;
            byte* dstPtr = (byte*)result.Storage.Address;
            long written = filterKernel(
                srcPtr, maskPtr, effectiveScan,
                outerSize: 1, srcOuterStride: 0, dstOuterStride: 0,
                innerSize: arr.dtypesize,
                dstPtr);

            // Sanity: the popcount and the kernel walk the same mask, so written
            // must equal count. A mismatch indicates kernel-corruption.
            if (written != count)
                throw new InvalidOperationException(
                    $"extract: filter kernel wrote {written}, expected {count}");

            return true;
        }
    }
}
