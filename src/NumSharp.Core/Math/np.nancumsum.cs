using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cumulative sum of array elements over a given axis treating Not a Numbers
        ///     (NaNs) as zero. The cumulative sum does not change when NaNs are encountered and leading
        ///     NaNs are replaced by zeros. Zeros are returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which the cumulative sum is computed. The default (None) is to compute the cumsum over the flattened array.</param>
        /// <param name="typeCode">Type of the returned array and of the accumulator in which the elements are summed. If not specified, it defaults to the dtype of <paramref name="a"/>, unless <paramref name="a"/> has an integer dtype with a precision less than that of the default platform integer, in which case the default platform integer is used.</param>
        /// <param name="out">Alternate output array in which to place the result. It must have the same shape and buffer length as the expected output, but its dtype may differ (the result is cast into it with NumPy's unsafe casting) and a reference to <paramref name="out"/> is returned.</param>
        /// <returns>A new array holding the result unless <paramref name="out"/> is specified, in which case a reference to <paramref name="out"/> is returned. The result has the same size as <paramref name="a"/>, and the same shape if <paramref name="axis"/> is not None or <paramref name="a"/> is 1-D.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html
        /// Port of numpy/lib/_nanfunctions_impl.py::nancumsum: <c>_replace_nan(a, 0)</c> then <c>np.cumsum</c>.
        /// Only float-family dtypes can contain NaN, so integer/bool/decimal inputs are equivalent to <see cref="cumsum"/>.
        /// </remarks>
        public static NDArray nancumsum(NDArray a, int? axis = null, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            // Boundary scope: the full-size NaN-replaced copy is reclaimed at exit; the scan
            // result (or the caller's @out — untracked, a Returns no-op) is yielded.
            using var scope = NDScope.Open();
            // NumPy: a, mask = _replace_nan(a, 0); return np.cumsum(a, axis, dtype, out). The mask is
            // discarded by nancumsum (unlike nanmean/nanvar), so we only need the NaN-replaced copy.
            return scope.Returns(cumsum(_replace_nan_for_scan(a, 0), axis, typeCode, @out));
        }

        /// <summary>
        /// Port of numpy/lib/_nanfunctions_impl.py::_replace_nan for the cumulative scans (the mask is
        /// not returned because nancumsum/nancumprod discard it). Only <b>inexact</b> dtypes
        /// (Half/Single/Double/Complex — NumPy's <c>issubclass(dtype.type, np.inexact)</c>) can hold a
        /// NaN, so those get a fresh array with every NaN replaced by <paramref name="val"/> (0 for the
        /// sum scan, 1 for the product scan); every other dtype cannot be NaN and passes through
        /// unchanged (NumPy returns it as-is with mask=None). This is a semantic applicability gate, not
        /// a per-dtype compute path: the replacement itself is the generic, SIMD <see cref="isnan"/> +
        /// <see cref="where(NDArray,NDArray,NDArray)"/> pipeline for all four inexact dtypes.
        /// </summary>
        /// <remarks>
        /// Complex NaN is handled exactly as NumPy's <c>np.copyto(a, val, where=np.isnan(a))</c>:
        /// <see cref="isnan"/> flags an element when EITHER component is NaN and <see cref="where"/>
        /// overwrites the WHOLE element with <paramref name="val"/> (0+0j / 1+0j), discarding a finite
        /// imaginary part. The fill scalar is built in <paramref name="a"/>'s own dtype so
        /// <see cref="where"/> takes its same-dtype fast path and the result keeps <paramref name="a"/>'s
        /// dtype (a strong int32 0 would otherwise promote float16 → float32 under NEP50).
        /// </remarks>
        private static NDArray _replace_nan_for_scan(NDArray a, int val)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    // Hot path: a fused single-pass SIMD fill for a C-contiguous float32/float64 operand
                    // (isnan + where would be two passes plus a bool-mask temp — see NanReplace). Every
                    // other inexact case (Half, Complex, or any non-C-contiguous view — F-order/strided/
                    // transposed/broadcast/reversed) takes the composed where(isnan(a), val, a) path, which
                    // is bit-identical and fully gated but reads logical order correctly for those layouts.
                    if ((a.typecode == NPTypeCode.Single || a.typecode == NPTypeCode.Double)
                        && a.Shape.IsContiguous && !a.Shape.IsBroadcasted)
                        return NanReplace.ReplaceContiguousFloat(a, val);

                    NDArray fill = NDArray.Scalar(val);
                    if (fill.typecode != a.typecode)
                        fill = fill.astype(a.typecode);   // 0/1 -> (Half)0/1, 0+0j/1+0j — dtype-preserving
                    return where(isnan(a), fill, a);
                default:
                    return a;                             // integer / bool / decimal / char cannot be NaN
            }
        }
    }
}
