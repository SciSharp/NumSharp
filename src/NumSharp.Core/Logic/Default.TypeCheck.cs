// NumPy's numpy/lib/_type_check_impl.py functions that compute, as DefaultEngine's members: np.nan_to_num (the fused
// scalar-fill pass and NumPy's copyto composition), np.isreal, np.iscomplex and np.real_if_close (whose facade is in
// Math/). The file sits beside the facades while it calls np.*: ratchet R2
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

        /// <inheritdoc />
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.equal
        public override NDArray Isreal(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy (_type_check_impl.isreal): `return imag(x) == 0`.
            // For a real/int/bool dtype the imaginary part is identically zero, so the answer is
            // all-True regardless of value — emit a fresh C-contiguous bool array from the DIMENSIONS
            // only (never `a.Shape`, whose strides/offset for a sliced/broadcast/strided view make
            // np.ones read past the logical window and return garbage bytes).
            // Text: imag(x) is the empty string, and `'' == 0` has no comparison loop, so NumPy's == answers False
            // everywhere (_failed_comparison_workaround) — a string array is never "real" (probed 2.4.2).
            if (a.GetTypeCode == NPTypeCode.String)
                return np.zeros(Shape.Adopt(a.Shape.dimensions), NPTypeCode.Boolean);

            if (a.GetTypeCode != NPTypeCode.Complex)
                return np.ones(Shape.Adopt(a.Shape.dimensions), NPTypeCode.Boolean);

            // Complex: the imaginary lane (a strided float64 view via np.imag) compared to zero.
            // IEEE `==` gives NumPy's semantics exactly: -0.0 == 0 (real), NaN/Inf != 0 (not real).
            return np.equal(np.imag(a), NDArray.Scalar(0d));
        }

        /// <inheritdoc />
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.not_equal
        public override NDArray Iscomplex(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy (_type_check_impl.iscomplex): complex dtype -> `ax.imag != 0`, otherwise
            // `zeros(shape, bool)`. What is tested is the VALUE of the imaginary part, not the dtype.
            // Non-complex short-circuits to all-False from the DIMENSIONS only (not `a.Shape`, whose
            // view strides/offset would make np.zeros emit garbage bytes on a strided/broadcast input).
            if (a.GetTypeCode != NPTypeCode.Complex)
            {
                // NumPy returns `res[()]` of `res = zeros(shape, bool)`: for an array that is a VIEW of the fresh buffer
                // (flags.owndata false), for a 0-d input the scalar; the view takes its own reference on the buffer.
                var res = np.zeros(Shape.Adopt((long[])a.Shape.dimensions.Clone()), NPTypeCode.Boolean);
                if (res.ndim == 0)
                    return res;
                var view = new NDArray(res.Storage.Alias(res.Shape));
                res.Dispose();
                return view;
            }

            // Complex: the imaginary lane (a strided float64 view via np.imag) compared to zero.
            // IEEE `!=` gives NumPy's semantics exactly: -0.0 != 0 is False, NaN/Inf != 0 is True.
            return np.not_equal(np.imag(a), NDArray.Scalar(0d));
        }

        /// <summary>
        ///     The float64 machine epsilon (2^-52), i.e. <c>np.finfo(np.complex128).eps</c> ==
        ///     <c>np.finfo(np.float64).eps</c>. NumSharp has only complex128, so this is the ONLY epsilon
        ///     <c>np.real_if_close</c> ever multiplies <c>tol</c> by — matching
        ///     NumPy's <c>getlimits.finfo(a.dtype.type).eps</c> for the sole complex dtype.
        /// </summary>
        private const double Float64Eps = 2.220446049250313e-16;

        /// <inheritdoc />
        public override NDArray RealIfClose(NDArray a, double tol = 100)
        {
            // NumPy's `asanyarray(a)` is already satisfied by the NDArray parameter. A non-complex dtype
            // is returned unchanged (the real part of a real number is itself) — the same instance, as
            // NumPy returns `a` untouched for a non-complexfloating type.
            if (a.typecode != NPTypeCode.Complex)
                return a;

            // tol > 1 is interpreted in machine epsilons of the complex type (float64 eps for complex128);
            // tol <= 1 stays an absolute tolerance. IEEE-identical to NumPy's `f.eps * tol` double multiply.
            if (tol > 1)
                tol = Float64Eps * tol;

            // Collapse to the real lane iff EVERY imaginary part is strictly within tol of zero. The scan
            // returns true for an empty array (vacuous all -> empty complex collapses) and false the moment
            // it sees a magnitude >= tol, a NaN, or an infinity.
            if (ImagCloseScan.AllImagWithinTol(a, tol))
                return np.real(a); // float64 view onto the real lane (shares memory), == NumPy's a.real

            // Some imaginary part is out of band (or NaN/inf): the array stays complex, unchanged.
            return a;
        }
    }
}
