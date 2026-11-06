using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The array object's statements of numpy/lib/_nanfunctions_impl.py's helpers, for the nan functions' routes that build
    // their answer above the engine: _replace_nan, whose dtype test (only an inexact array has a NaN mask) is the one
    // decision of np.nanmax / np.nanmin's subclass branch (np.max.Keywords.cs, MatrixNanExtreme), kept here below the
    // facade (the np-layer dtype plan's ratchet R1). The engine's nan functions run their own copies over plain arrays
    // (DefaultEngine's ReplaceNanForScan, the fused fills of the nan reductions).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>_replace_nan(a, val)</c>: for an inexact array (float16, float32, float64, complex128) the mask
        ///     <c>np.isnan(a)</c>, a K-order copy (<c>np.array(a, subok=True, copy=True)</c>) and <c>np.copyto(copy, val,
        ///     where=mask)</c>; for any other dtype no mask and the array itself. Use it where a nan function's route needs
        ///     NumPy's replaced copy AND its mask (np.nanmax / np.nanmin of a numpy.matrix); the engine's nan reductions fuse
        ///     the replacement into their own passes instead.
        /// </summary>
        /// <param name="a">The array; any class (the copy is the plain array the engine members make).</param>
        /// <param name="val">The replacement, a Python float (<c>-inf</c> for nanmax, <c>+inf</c> for nanmin), cast into
        ///     the copy under same_kind (a complex copy gets <c>val+0j</c>).</param>
        /// <param name="mask">The NaN mask (a new plain bool array of <paramref name="a"/>'s shape), or <c>null</c> for a
        ///     dtype that holds no NaN (the integers, bool, char, decimal, text): NumPy's <c>mask is None</c>.</param>
        /// <returns>The new replaced copy, or <paramref name="a"/> itself when <paramref name="mask"/> is <c>null</c>.
        ///     Footgun: the caller owns the copy and the mask (dispose them, or call from an <c>[NDScoped]</c>
        ///     member).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>NumPy also masks an object array (<c>np.not_equal(a, a)</c>); NumSharp has no object dtype.</remarks>
        internal static NDArray ReplaceNaN(NDArray a, double val, out NDArray mask)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    // mask = np.isnan(a); a = np.array(a, subok=True, copy=True); np.copyto(a, val, where=mask) - the
                    // engine's isnan and the array object's K-order copy, so a matrix operand gives plain arrays (the
                    // caller decides the class of what it builds from them).
                    mask = a.TensorEngine.IsNan(a);
                    NDArray copy = a.copy('K');
                    UnmanagedStorage.CopytoPythonScalar(copy, val, PyScalarKind.Float, NPY_CASTING.NPY_SAME_KIND_CASTING, mask);
                    return copy;
                default:
                    // Integer, bool, char, decimal and text arrays cannot hold a NaN: NumPy's mask is None.
                    mask = null;
                    return a;
            }
        }
    }
}
