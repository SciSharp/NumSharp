// NumPy's numpy/lib/_polynomial_impl.py (the legacy np.poly* functions) as DefaultEngine's members: each function's
// statements as its np.* facade held them. The file sits beside the facades while its statements call np.*: ratchet R2
// counts np.* calls against every file under Backends/ (the np-layer dtype plan, C3); it moves under Backends/Default/
// in P6.

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the working copies, casts and intermediates of NumPy's statements are this call's own
        public override NDArray Poly(NDArray seq_of_zeros)
        {
            seq_of_zeros = np.atleast_1d(seq_of_zeros);
            bool text = Backends.Strings.NDStringUfuncs.IsText(seq_of_zeros);
            int nd = seq_of_zeros.ndim;

            NDArray seq;
            if (nd == 2 && seq_of_zeros.Shape.dimensions[0] == seq_of_zeros.Shape.dimensions[1] && seq_of_zeros.Shape.dimensions[0] != 0)
            {
                // Characteristic polynomial: build from the matrix's eigenvalues (a text matrix fails their finiteness
                // check, NumPy's isfinite TypeError).
                seq = np.linalg.eigvals(seq_of_zeros);
            }
            else if (nd == 1)
            {
                // A real (non-object) sequence is cast to its mintypecode float — float32 stays float32,
                // everything else (int/bool/float16/float64) becomes float64; complex stays complex. Text has no
                // mintypecode of its own ('U' is in no typeset), so it takes the default 'd': parsed into float64.
                seq = seq_of_zeros.astype(text ? NPTypeCode.Double : PolyMinType(seq_of_zeros.typecode));
            }
            else
            {
                throw new ValueError("input must be 1d or non-empty square 2d array.");
            }

            if (seq.size == 0)
                return NDArray.Scalar(1.0);

            NPTypeCode dt = seq.typecode;
            NDArray onesScalar = np.ones(new Shape(1), dt);
            NDArray negSeq = np.negative(seq);

            // a = ones(1); for each root: a = convolve(a, [1, -root])  (accumulates (x - root) factors).
            NDArray a = onesScalar;
            for (long i = 0; i < seq.size; i++)
            {
                NDArray factor = np.concatenate(new[] { onesScalar, negSeq[$"{i}:{i + 1}"] }, 0);
                a = np.convolve(a, factor, "full");
            }

            // If the roots come as complex conjugate pairs, the coefficients are real — take the real part.
            if (dt == NPTypeCode.Complex)
            {
                NDArray rootsC = seq.astype(NPTypeCode.Complex);
                if (np.all(np.equal(np.sort(rootsC), np.sort(np.conjugate(rootsC)))))
                    a = np.real(a).copy();
            }

            return a;
        }

        /// <inheritdoc />
        [NDScoped] // the working copies, casts and intermediates of NumPy's statements are this call's own
        public override NDArray Polyval(NDArray p, NDArray x)
        {
            p = np.asarray(p);

            // A poly1d `x` composes the two polynomials — handled by the poly1d overload below.
            x = np.asanyarray(x);

            // NumPy's `for pv in p` cannot iterate a 0-d coefficient array (a scalar is not a
            // length-1 polynomial — that would be `[c]`). A 1-D length-1 array is a valid constant.
            if (p.ndim == 0)
                throw new TypeError("iteration over a 0-d array");

            // NumPy's `for pv in p` takes len(p) coefficients: the first axis, each coefficient a row of p.shape[1:] for an
            // N-D p (p's size would count every element).
            long n = p.Shape.dimensions[0];
            // No coefficients — NumPy's loop never runs, so the result is zeros_like(x) (a string x included).
            if (n == 0)
                return np.zeros_like(x);

            // NumPy's Horner loop starts with y = zeros_like(x) and y * x: text in x has no multiply loop. Text in p
            // meets the first add, y * x + pv, whose left side keeps x's dtype; iterating a 1-D p yields NumPy SCALARS,
            // whose descriptor is their own text length (np.str_('a') is U1, '' too), while an N-D p yields rows of p's
            // dtype. (A StringDType x is named by class; a 1-D StringDType p iterates as Python strs, so its add meets
            // the legacy resolver with the str's own U descriptor, as a U p's np.str_ does; an N-D one names the
            // classes.)
            if (Backends.Strings.NDStringUfuncs.IsText(x))
                throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", x.dtype, x.dtype);
            if (Backends.Strings.NDStringUfuncs.IsText(p))
                throw Backends.Strings.NDStringUfuncs.NoLoop("add", x.dtype,
                    p.ndim == 1 ? Backends.Strings.NDStringArrays.ScalarDescr(p.Storage.GetAtIndex(0)) : p.dtype);

            // An N-D p: each coefficient is a row that NumPy's `y = y * x + pv` broadcasts against x, so the result takes
            // the broadcast shape (rows that do not broadcast with x raise the add's error) and the dtype each step
            // promotes to. NumPy's statements, literally: the fixed-dtype loop below assumes y keeps x's shape.
            if (p.ndim > 1)
            {
                NDArray yy = np.zeros_like(x);
                for (long k = 0; k < n; k++)
                    yy = yy * x + p[k.ToString()];
                return yy;
            }

            // Horner accumulates `y = y*x + p[k]`. NumPy lets `y` (starting at zeros_like(x)) grow in
            // dtype as wider coefficients arrive; the fixed common dtype below is bit-identical because
            // every widening cast (int->wider, float32->float64, real->complex) is exact and `y` starts
            // at zero. Computing in `result_type` up front lets the whole loop reuse two buffers.
            NPTypeCode t = np.result_type(p, x);
            NDArray xc = x.astype(t);
            NDArray y = np.zeros(x.Shape, t);

            // Iterate from k = 0 so the very first `0 * x` reproduces NumPy's `0*inf -> nan` edge.
            for (long k = 0; k < n; k++)
            {
                np.multiply(y, xc, @out: y);
                np.add(y, p[k.ToString()], @out: y);
            }

            return y;
        }

        /// <inheritdoc />
        [NDScoped] // the working copies, casts and intermediates of NumPy's statements are this call's own
        public override NDArray Polyint(NDArray p, int m = 1, NDArray k = null)
        {
            if (m < 0)
                throw new ValueError("Order of integral must be positive (see polyder)");

            // Integration constants: default zeros(m); a length-1 k with m>1 is broadcast to m.
            NDArray kk;
            if (k is null)
            {
                kk = np.zeros(new Shape(m), NPTypeCode.Double);
            }
            else
            {
                kk = np.atleast_1d(k);
                // NumPy measures k with len(): its first axis (an N-D k's rows are its constants).
                if (kk.Shape.dimensions[0] == 1 && m > 1)
                    kk = kk["0"] * np.ones(new Shape(m), NPTypeCode.Double);
                if (kk.Shape.dimensions[0] < m)
                    throw new ValueError("k must be a scalar or a rank-1 array of length 1 or >m.");
            }

            p = np.asarray(p);

            if (m == 0)
                return p;

            // NumPy's first integration is p.__truediv__(arange(len(p), 0, -1)): len() of a 0-d array refuses first,
            // then the division has no loop for text — a StringDType p is named by class beside the int64 exponents, a
            // fixed-width one meets the legacy resolver.
            if (p.ndim == 0)
                throw new TypeError("len() of unsized object");
            if (Backends.Strings.NDStringUfuncs.IsVString(p))
                throw Backends.Strings.NDStringUfuncs.VStringNoLoop("divide", new[] { p.dtype.Meta.ToString(), np.int64.Meta.ToString() });
            if (Backends.Strings.NDStringUfuncs.IsString(p))
                throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("divide");

            // Integrate once per order: prepend the term-wise integral p / [len, len-1, ..., 1] and
            // append the constant k[i]. Division by the int64 arange floats the whole result. len is NumPy's len(): the
            // first axis, so an N-D p meets the division's broadcast (or the concatenation's rank) error as in NumPy.
            NDArray result = p;
            for (int i = 0; i < m; i++)
            {
                long len = result.Shape.dimensions[0];
                NDArray divided = np.true_divide(result, np.arange(len, 0L, -1L));
                NDArray kElem = kk[$"{i}:{i + 1}"];
                result = np.concatenate(new[] { divided, kElem }, 0);
            }

            return result;
        }

        /// <inheritdoc />
        public override NDArray Polyadd(NDArray a1, NDArray a2) => PolyAddSub(a1, a2, subtract: false);

        /// <inheritdoc />
        public override NDArray Polysub(NDArray a1, NDArray a2) => PolyAddSub(a1, a2, subtract: true);

        /// <summary>
        ///     NumPy's <c>polyadd</c> / <c>polysub</c> statements: <c>atleast_1d</c> both, zero-extend the shorter on
        ///     the high-degree side, then add or subtract.
        /// </summary>
        /// <param name="a1">The first polynomial's coefficients.</param>
        /// <param name="a2">The second polynomial's coefficients.</param>
        /// <param name="subtract">Subtract (<c>a1 - a2</c>) instead of add.</param>
        /// <returns>The sum or difference, in the operator's promoted dtype.</returns>
        /// <exception cref="TypeError">Coefficients the operator has no loop for.</exception>
        [NDScoped]
        private static NDArray PolyAddSub(NDArray a1, NDArray a2, bool subtract)
        {
            a1 = np.atleast_1d(a1);
            a2 = np.atleast_1d(a2);
            // NumPy's diff = len(a2) - len(a1): the first axes. The 1-D zeros it pads with meet an N-D operand in
            // np.concatenate's rank error; equal first axes go straight to the operator's broadcast.
            long diff = a2.Shape.dimensions[0] - a1.Shape.dimensions[0];

            if (diff > 0)
                a1 = np.concatenate(new[] { np.zeros(new Shape(diff), a1.dtype), a1 }, 0);
            else if (diff < 0)
                a2 = np.concatenate(new[] { np.zeros(new Shape(-diff), a2.dtype), a2 }, 0);

            return subtract ? a1 - a2 : a1 + a2;
        }

        /// <inheritdoc />
        [NDScoped] // the trimmed poly1d coefficient copies are disposed by their using; the product escapes
        public override NDArray Polymul(NDArray a1, NDArray a2)
        {
            // Text: NumPy's poly1d trims leading "zeros" of text (empty strings; the trim cannot fail and an all-empty
            // input becomes the text '0') and convolve then refuses any text operand for want of an item size — so
            // after each input's rank check the answer is that refusal, whatever the text holds. StringDType: poly1d
            // trims each input (leading empty strings; an all-empty input becomes the text '0') and np.convolve answers
            // the pair — its legacy descriptor lookup refuses a StringDType first operand, or falls back to an OBJECT
            // product whose first element product decides (np.convolve's StringDType route).
            if (Backends.Strings.NDStringUfuncs.IsVString(a1) || Backends.Strings.NDStringUfuncs.IsVString(a2))
            {
                using var t1 = new poly1d(a1);
                using var t2 = new poly1d(a2);
                return np.convolve(t1.coeffs, t2.coeffs, "full");
            }
            if (Backends.Strings.NDStringUfuncs.IsString(a1) || Backends.Strings.NDStringUfuncs.IsString(a2))
            {
                if (np.atleast_1d(a1).ndim > 1 || np.atleast_1d(a2).ndim > 1)
                    throw new ValueError("Polynomial must be 1d only.");
                throw Backends.Strings.NDStringUfuncs.NoItemsize();
            }
            // poly1d normalisation trims leading zeros (and maps an all-zero input to [0]). The polynomials
            // own their coefficient arrays (field egress — no ambient scope reclaims them), so they are
            // disposed here; the convolution is a fresh array, yielded by the [NDScoped] weaver.
            using var p1 = new poly1d(a1);
            using var p2 = new poly1d(a2);
            return np.convolve(p1.coeffs, p2.coeffs, "full");
        }

        /// <inheritdoc />
        // NDScoped: the floated inputs, per-iteration scaled quotient/remainder products and the
        // leading-zero-strip view chain are reclaimed; both tuple components (q, r) are yielded.
        [NDScoped]
        public override (NDArray q, NDArray r) Polydiv(NDArray u, NDArray v)
        {
            u = np.atleast_1d(u) + 0.0;   // floated, exactly like NumPy's `+ 0.0`
            v = np.atleast_1d(v) + 0.0;

            // w = u[0] + v[0]: NumPy forms the sum and keeps its dtype, the common type of two NumPy values, both strong
            // (np.result_type reads them so; NumSharp's operator would take a 0-d operand as a weak scalar). Forming the
            // sum is what raises for an N-D u and v whose rows do not broadcast (the add's error, u's row first), before
            // anything else is read, so it is formed too, in that dtype.
            NPTypeCode w = np.result_type(u["0"], v["0"]);
            using (np.add(u["0"], v["0"], dtype: w)) { }
            // NumPy's m = len(u) - 1, n = len(v) - 1: the first axes (an N-D operand's rows).
            long m = u.Shape.dimensions[0] - 1;
            long n = v.Shape.dimensions[0] - 1;
            NDArray scale = 1.0 / v["0"];

            long qlen = Math.Max(m - n + 1, 1);
            NDArray q = np.zeros(new Shape(qlen), w);
            NDArray r = u.astype(w);

            for (long k = 0; k < m - n + 1; k++)
            {
                // d = scale * r[k]. NumPy evaluates this SCALAR*SCALAR product through scalarmath —
                // the NAIVE a_re*b_re - a_im*b_im — NOT the FMA simd_cmul its ARRAY multiply (and hence
                // NumSharp's complex `*`) uses. For a complex loop we therefore multiply the scalar d
                // naively to stay bit-identical to NumPy's polydiv; `d * v` below is a scalar*ARRAY
                // product, which NumPy DOES take through the FMA array loop, so it keeps NumSharp's
                // (now FMA) complex multiply. (Real dtypes have no scalar/array multiply distinction.)
                NDArray rk = r[k.ToString()];
                // q[k] = d is NumPy's scalar setitem. d = scale * r[k] has the dimensions of v[0] and r[k], a row when u
                // or v is N-D, and a d with dimensions is a sequence: a complex slot refuses it through complex()
                // (TypeError), a real one as a sequence (ValueError). Checked before d is formed: forming it cannot fail
                // once w = u[0] + v[0] did not (the same shapes), and the complex product below reads one element.
                if (scale.ndim != 0 || rk.ndim != 0)
                    throw w == NPTypeCode.Complex
                        ? new TypeError("only 0-dimensional arrays can be converted to Python scalars")
                        : new ValueError("setting an array element with a sequence.");
                NDArray d = w == NPTypeCode.Complex ? NaiveComplexScalarMultiply(scale, rk) : scale * rk;
                q[$"{k}:{k + 1}"] = np.atleast_1d(d);
                // r[k : k+n+1] -= d * v   (an in-place, length-(n+1) vector update)
                NDArray seg = r[$"{k}:{k + n + 1}"];
                np.subtract(seg, d * v, @out: seg);
            }

            // Strip leading near-zero remainder coefficients (NumPy: rtol 1e-14, keep at least one).
            while (r.Shape.dimensions[r.ndim - 1] > 1 && np.allclose(r["0"], NDArray.Scalar(0.0), rtol: 1e-14))
                r = r["1:"];

            return (q, r);
        }

        /// <summary>
        ///     Naive (un-fused) complex scalar product <c>a·b</c> — <c>(a_re·b_re − a_im·b_im,
        ///     a_re·b_im + a_im·b_re)</c> — matching NumPy's <c>scalarmath</c> path, which a
        ///     <c>np.complex128</c> scalar multiply takes (its ARRAY multiply is the FMA <c>simd_cmul</c>
        ///     instead). Used by <c>np.polydiv</c>'s statements for the scalar quotient coefficient so it stays
        ///     bit-identical to NumPy; do NOT route array multiplies here.
        /// </summary>
        /// <param name="a">Left 0-D complex scalar.</param>
        /// <param name="b">Right 0-D complex scalar.</param>
        /// <returns>A 0-D complex NDArray holding the naive product.</returns>
        private static NDArray NaiveComplexScalarMultiply(NDArray a, NDArray b)
        {
            System.Numerics.Complex x = a.GetData<System.Numerics.Complex>()[0];
            System.Numerics.Complex y = b.GetData<System.Numerics.Complex>()[0];
            return NDArray.Scalar(new System.Numerics.Complex(
                x.Real * y.Real - x.Imaginary * y.Imaginary,
                x.Real * y.Imaginary + x.Imaginary * y.Real));
        }

        /// <summary>
        ///     NumPy's <c>mintypecode(dtype.char)</c> as applied by <c>poly</c>: the smallest floating type
        ///     that holds the roots. float32 preserves; complex preserves; every other type widens to float64.
        /// </summary>
        /// <param name="t">The roots' dtype.</param>
        /// <returns>Single for float32, Complex for complex, Double for everything else.</returns>
        private static NPTypeCode PolyMinType(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Single: return NPTypeCode.Single;
                case NPTypeCode.Complex: return NPTypeCode.Complex;
                default: return NPTypeCode.Double;
            }
        }
    }
}
