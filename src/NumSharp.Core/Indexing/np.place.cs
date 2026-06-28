using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="arr"/> based on a boolean mask.
        ///     Where <paramref name="mask"/> is true (walked in C-order), the
        ///     next value from <paramref name="vals"/> (cycling) is written into
        ///     <paramref name="arr"/>. In-place.
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="arr"/> — NumPy allows shape mismatch as long as
        ///     element counts match.
        /// </param>
        /// <param name="vals">
        ///     Values to write. Cast to <paramref name="arr"/>'s dtype and cycled
        ///     to fill the True positions. Must be non-empty when at least one
        ///     mask entry is True (NumPy parity).
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.place.html</remarks>
        public static void place(NDArray arr, NDArray mask, NDArray vals)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (mask is null) throw new ArgumentNullException(nameof(mask));
            if (vals is null) throw new ArgumentNullException(nameof(vals));

            if (mask.size != arr.size)
                throw new ArgumentException(
                    $"place: mask and data must be the same size (mask.size={mask.size}, arr.size={arr.size}).",
                    nameof(mask));

            // Empty arr ⇒ empty mask ⇒ no-op.
            if (arr.size == 0)
                return;

            // NumPy WRITEBACKIFCOPY semantics for non-contig views — see np.put for
            // the same pattern. ascontiguousarray clones the view into a contig
            // scratch; the kernel writes into the scratch; np.copyto pushes the
            // changes back through the view's strides to the parent storage.
            if (!arr.Shape.IsContiguous)
            {
                var scratch = np.ascontiguousarray(arr);
                try
                {
                    place(scratch, mask, vals);
                    np.copyto(arr, scratch, casting: "unsafe");
                }
                finally { scratch.Dispose(); }
                return;
            }

            // Cast mask to contig bool; cast vals to contig arr.dtype.
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
            if (vals.GetTypeCode == arr.GetTypeCode && vals.Shape.IsContiguous)
            {
                valsCast = vals;
                ownVals = false;
            }
            else if (vals.GetTypeCode == arr.GetTypeCode)
            {
                var c = np.ascontiguousarray(vals);
                valsCast = c;
                ownVals = !ReferenceEquals(c, vals);
            }
            else
            {
                valsCast = vals.astype(arr.GetTypeCode);
                ownVals = true;
            }

            try
            {
                if (valsCast.size == 0 && AnyTrueMask(maskCast))
                    throw new ArgumentException("Cannot insert from an empty array!", nameof(vals));

                if (valsCast.size == 0)
                    return;   // mask all-false and empty vals — NumPy treats as no-op

                ExecutePlace(arr, maskCast, valsCast);
            }
            finally
            {
                if (ownMask) maskCast.Dispose();
                if (ownVals) valsCast.Dispose();
            }
        }

        private static unsafe bool AnyTrueMask(NDArray maskBool)
        {
            // Walk the contig bool buffer linearly. mask.size > 0 by the time we get here.
            byte* p = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            long n = maskBool.size;
            for (long i = 0; i < n; i++)
                if (p[i] != 0) return true;
            return false;
        }

        private static unsafe void ExecutePlace(NDArray arr, NDArray maskBool, NDArray valsCast)
        {
            var kernel = DirectILKernelGenerator.GetPlaceKernel();
            if (kernel == null)
                throw new NotSupportedException("np.place: IL kernel unavailable");

            byte* dstPtr = (byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, arr.size, valsPtr, valsCast.size, arr.dtypesize);
        }
    }
}
