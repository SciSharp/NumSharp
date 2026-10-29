// NumPy's numpy/lib/_type_check_impl.py functions that compute, as DefaultEngine's members: np.nan_to_num (the fused
// scalar-fill pass and NumPy's copyto composition). The file sits beside the facades while it calls np.*: ratchet R2
// counts np.* calls against every file under Backends/ (the np-layer dtype plan, C3); it moves under Backends/Default/
// in P6.

using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray NanToNum(NDArray x, bool copy = true, object nan = null, object posinf = null,
            object neginf = null)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            // NumPy: x = array(x, copy=copy). copy=True -> C-contiguous copy; copy=False operates on x.
            var work = copy ? x.copy() : x;
            var tc = work.typecode;

            // Only inexact dtypes (Half/Single/Double/Complex) are touched; anything else has no
            // NaN/Inf domain and is returned unchanged (the copy when copy=True, else x).
            bool inexact = tc == NPTypeCode.Half || tc == NPTypeCode.Single
                        || tc == NPTypeCode.Double || tc == NPTypeCode.Complex;
            if (!inexact)
                return work;

            // Real dtype of the values being replaced (float64 for complex128) — drives the finfo limits.
            NPTypeCode realTc = tc == NPTypeCode.Complex ? NPTypeCode.Double : tc;

            bool anyArrayFill = IsArrayFill(nan) || IsArrayFill(posinf) || IsArrayFill(neginf);

            // Fast fused path: scalar fills into a contiguous target. Complex rides the Double kernel
            // over the raw interleaved buffer (real & imag each get the same scalar fill).
            if (!anyArrayFill && work.Shape.IsContiguous && work.Shape.Offset == 0)
                RunNanToNumFused(work, tc, realTc, nan, posinf, neginf);
            else
                RunNanToNumComposition(work, tc, realTc, nan, posinf, neginf);

            return work;
        }

        // A fill is "array-valued" when it is an NDArray or a C# array; everything else (int/float/bool/
        // null) is a scalar the fused kernel can bake into a register.
        private static bool IsArrayFill(object o) => o is NDArray || (o is not null && o.GetType().IsArray);

        // The finfo(realdtype).max / .min pair NumPy uses for the default +Inf / -Inf fills.
        private static (double max, double min) NanToNumFiniteRange(NPTypeCode realTc) => realTc switch
        {
            NPTypeCode.Half => (65504.0, -65504.0),
            NPTypeCode.Single => (float.MaxValue, float.MinValue),
            _ => (double.MaxValue, double.MinValue) // Double (also complex128's real dtype)
        };

        private static double FillToDouble(object o) => o switch
        {
            null => 0.0,
            Half h => (double)h,
            _ => Convert.ToDouble(o)
        };

        // ---- Fused scalar-fill path (Half/Single/Double kernel; Complex -> Double over 2N) ----
        private static unsafe void RunNanToNumFused(NDArray work, NPTypeCode tc, NPTypeCode realTc,
            object nan, object posinf, object neginf)
        {
            var (maxf, minf) = NanToNumFiniteRange(realTc);
            double nanD = FillToDouble(nan);
            double posD = posinf is null ? maxf : FillToDouble(posinf);
            double negD = neginf is null ? minf : FillToDouble(neginf);

            long size = tc == NPTypeCode.Complex ? work.size * 2 : work.size;
            void* data = work.Address;

            // Fills materialized in the kernel dtype (same_kind cast of the double value, exactly as
            // NumPy's copyto(d, fill) casts to d.dtype).
            byte* fnan = stackalloc byte[8];
            byte* fpos = stackalloc byte[8];
            byte* fneg = stackalloc byte[8];
            WriteNanToNumScalar(realTc, fnan, nanD);
            WriteNanToNumScalar(realTc, fpos, posD);
            WriteNanToNumScalar(realTc, fneg, negD);

            DirectILKernelGenerator.NanToNum(realTc, data, data, size, fnan, fpos, fneg);
        }

        private static unsafe void WriteNanToNumScalar(NPTypeCode realTc, byte* p, double v)
        {
            switch (realTc)
            {
                case NPTypeCode.Half: *(Half*)p = (Half)v; break;
                case NPTypeCode.Single: *(float*)p = (float)v; break;
                default: *(double*)p = v; break; // Double
            }
        }

        // ---- Composition path (array-valued fills, or non-contiguous in-place targets) ----
        private static void RunNanToNumComposition(NDArray work, NPTypeCode tc, NPTypeCode realTc,
            object nan, object posinf, object neginf)
        {
            var (maxf, minf) = NanToNumFiniteRange(realTc);
            NDArray nanArr = NanToNumFillArray(nan, 0.0, realTc);
            NDArray posArr = NanToNumFillArray(posinf, maxf, realTc);
            NDArray negArr = NanToNumFillArray(neginf, minf, realTc);

            if (tc == NPTypeCode.Complex)
            {
                // NumPy: dest = (x.real, x.imag) — writeable strided float64 views, both fed the same fills.
                NanToNumReplaceInto(np.real(work), nanArr, posArr, negArr);
                NanToNumReplaceInto(np.imag(work), nanArr, posArr, negArr);
            }
            else
            {
                NanToNumReplaceInto(work, nanArr, posArr, negArr);
            }
        }

        // NumPy: copyto(d, nan, where=isnan(d)); copyto(d, maxf, where=isposinf(d)); copyto(d, minf,
        // where=isneginf(d)). The three masks are disjoint, so evaluating each fresh is equivalent to
        // NumPy pre-computing all three on the original values.
        private static void NanToNumReplaceInto(NDArray dest, NDArray nanArr, NDArray posArr, NDArray negArr)
        {
            np.copyto(dest, nanArr, where: np.isnan(dest));
            np.copyto(dest, posArr, where: np.isposinf(dest));
            np.copyto(dest, negArr, where: np.isneginf(dest));
        }

        private static NDArray NanToNumFillArray(object fill, double def, NPTypeCode realTc)
        {
            NDArray arr;
            if (fill is null) arr = NDArray.Scalar(def);
            else if (fill is NDArray nd) arr = nd;
            else if (fill is Array a) arr = np.array(a);
            else arr = NDArray.Scalar(FillToDouble(fill));
            // Cast the fill to the destination dtype up front (NumPy's copyto casts fill -> d.dtype).
            return arr.typecode == realTc ? arr : arr.astype(realTc);
        }
    }
}
