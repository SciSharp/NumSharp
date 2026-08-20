using System;

namespace NumSharp
{
    /// <summary>
    ///     A one-dimensional polynomial class — the NumSharp port of <c>numpy.poly1d</c>. Encapsulates a
    ///     coefficient vector (highest power first) and the natural polynomial operations.
    ///     <para>
    ///     C# has no call syntax, so NumPy's <c>p(x)</c> evaluation is spelled <c>np.polyval(p, x)</c>; the
    ///     indexer <c>p[k]</c> retrieves the coefficient of <c>x**k</c> exactly like NumPy. NumPy's
    ///     <c>p ** n</c> (polynomial power) has no <c>**</c> operator in C# — spell it with repeated
    ///     multiplication (<c>p * p * p</c>), which NumPy defines it to equal. <see cref="ToString"/> renders
    ///     NumPy's <c>repr</c> form (<c>poly1d([...])</c>); NumPy's separate pretty <c>str()</c> form is not
    ///     reproduced (C# has a single <c>ToString</c>).
    ///     </para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html</remarks>
    public sealed class poly1d
    {
        private NDArray _coeffs;
        private readonly string _variable;

        /// <summary>
        ///     Construct a polynomial from its coefficients (highest power first), or — when
        ///     <paramref name="r"/> is true — from its roots.
        /// </summary>
        /// <param name="c_or_r">Coefficients, or (if <paramref name="r"/>) the polynomial's roots.</param>
        /// <param name="r">If true, <paramref name="c_or_r"/> gives the roots.</param>
        /// <param name="variable">Variable name used when printing (default <c>"x"</c>).</param>
        public poly1d(NDArray c_or_r, bool r = false, string variable = null)
        {
            if (r)
                c_or_r = np.poly(c_or_r);

            c_or_r = np.atleast_1d(c_or_r);
            if (c_or_r.ndim > 1)
                throw new ValueError("Polynomial must be 1d only.");

            c_or_r = np.trim_zeros(c_or_r, "f");
            if (c_or_r.size == 0)
                c_or_r = np.zeros(new Shape(1), c_or_r.typecode);

            _coeffs = c_or_r;
            _variable = variable ?? "x";
        }

        /// <summary>Copy constructor — shares the coefficients and (unless overridden) the variable name.</summary>
        public poly1d(poly1d source, string variable = null)
        {
            _coeffs = source._coeffs;
            _variable = variable ?? source._variable;
        }

        /// <summary>The polynomial coefficients, highest power first.</summary>
        public NDArray coeffs => _coeffs;

        /// <summary>Alias of <see cref="coeffs"/> (NumPy's <c>c</c>).</summary>
        public NDArray c => _coeffs;

        /// <summary>Alias of <see cref="coeffs"/> (NumPy's <c>coef</c>).</summary>
        public NDArray coef => _coeffs;

        /// <summary>Alias of <see cref="coeffs"/> (NumPy's <c>coefficients</c>).</summary>
        public NDArray coefficients => _coeffs;

        /// <summary>The name of the polynomial variable.</summary>
        public string variable => _variable;

        /// <summary>
        ///     NumPy's <c>__array__</c>: a poly1d IS its coefficient vector, so it flows into any np.* function.
        ///     A consequence is that <c>np.polyder(poly1d)</c> etc. return the bare coefficient array — use the
        ///     instance methods (<see cref="deriv"/>, <see cref="integ"/>) or the operators for a poly1d result.
        /// </summary>
        public static implicit operator NDArray(poly1d p) => p?._coeffs;

        /// <summary>The order (degree) of the polynomial.</summary>
        public int order => (int)(_coeffs.size - 1);

        /// <summary>Alias of <see cref="order"/> (NumPy's <c>o</c>).</summary>
        public int o => order;

        /// <summary>The roots of the polynomial, where <c>self(x) == 0</c>.</summary>
        public NDArray roots => np.roots(_coeffs);

        /// <summary>Alias of <see cref="roots"/> (NumPy's <c>r</c>).</summary>
        public NDArray r => np.roots(_coeffs);

        /// <summary>Return an antiderivative (indefinite integral) of this polynomial.</summary>
        public poly1d integ(int m = 1, NDArray k = null) => new poly1d(np.polyint(_coeffs, m, k ?? NDArray.Scalar(0.0)));

        /// <summary>Return a derivative of this polynomial.</summary>
        public poly1d deriv(int m = 1) => new poly1d(np.polyder(_coeffs, m));

        /// <summary>
        ///     The coefficient of <c>x**k</c> (NumPy's <c>p[k]</c>): a 0-d zero of the coefficient dtype when
        ///     <paramref name="k"/> is out of <c>[0, order]</c>.
        /// </summary>
        public NDArray this[int k]
        {
            get
            {
                if (k < 0 || k > order)
                    return NDArray.Scalar(0).astype(_coeffs.typecode);
                return _coeffs[(order - k).ToString()];
            }
            set
            {
                if (k < 0)
                    throw new ValueError("Does not support negative powers.");
                int ind;
                if (k > order)
                {
                    NDArray zr = np.zeros(new Shape(k - order), _coeffs.typecode);
                    _coeffs = np.concatenate(new[] { zr, _coeffs }, 0);
                    ind = 0;
                }
                else
                {
                    ind = order - k;
                }
                _coeffs[$"{ind}:{ind + 1}"] = np.atleast_1d(value);
            }
        }

        /// <summary>Sum of two polynomials.</summary>
        public static poly1d operator +(poly1d a, poly1d b) => new poly1d(np.polyadd(a._coeffs, b._coeffs));

        /// <summary>Sum of a polynomial and coefficients (NumPy's <c>poly1d.__add__</c> -> polyadd).</summary>
        public static poly1d operator +(poly1d a, NDArray b) => new poly1d(np.polyadd(a._coeffs, new poly1d(b)._coeffs));

        // No `operator +(NDArray, poly1d)`: NumPy's `array + poly1d` is ELEMENT-WISE (the ndarray wins
        // via poly1d.__array__), NOT polyadd — the implicit poly1d->NDArray conversion already yields
        // that. A reflected polyadd overload would both diverge from NumPy AND hijack `"string" + p`
        // (the string implicitly becomes a char NDArray) into a garbage polyadd. Same for subtraction.

        /// <summary>Difference of two polynomials.</summary>
        public static poly1d operator -(poly1d a, poly1d b) => new poly1d(np.polysub(a._coeffs, b._coeffs));

        /// <summary>Difference of a polynomial and coefficients (NumPy's <c>poly1d.__sub__</c> -> polysub).</summary>
        public static poly1d operator -(poly1d a, NDArray b) => new poly1d(np.polysub(a._coeffs, new poly1d(b)._coeffs));

        /// <summary>Negation.</summary>
        public static poly1d operator -(poly1d a) => new poly1d(-a._coeffs);

        /// <summary>Unary plus (returns the same polynomial).</summary>
        public static poly1d operator +(poly1d a) => a;

        /// <summary>Product of two polynomials.</summary>
        public static poly1d operator *(poly1d a, poly1d b) => new poly1d(np.polymul(a._coeffs, b._coeffs));

        /// <summary>Product of a polynomial and coefficients.</summary>
        public static poly1d operator *(poly1d a, NDArray b) => new poly1d(np.polymul(a._coeffs, new poly1d(b)._coeffs));

        /// <summary>Scale every coefficient by a scalar (NumPy's <c>isscalar</c> branch).</summary>
        public static poly1d operator *(poly1d a, double s) => new poly1d(a._coeffs * s);

        /// <summary>Scale every coefficient by a scalar.</summary>
        public static poly1d operator *(double s, poly1d a) => new poly1d(s * a._coeffs);

        /// <summary>Divide every coefficient by a scalar.</summary>
        public static poly1d operator /(poly1d a, double s) => new poly1d(a._coeffs / s);

        /// <summary>Polynomial division — returns <c>(quotient, remainder)</c> as poly1d objects.</summary>
        public static (poly1d q, poly1d r) operator /(poly1d a, poly1d b)
        {
            var (q, rem) = np.polydiv(a._coeffs, b._coeffs);
            return (new poly1d(q), new poly1d(rem));
        }

        /// <summary>
        ///     Polynomial division by raw coefficients — <c>(quotient, remainder)</c>. NumPy's
        ///     <c>poly1d.__truediv__</c> wraps a non-scalar in <c>poly1d</c> then divides, so <c>p / array</c>
        ///     is polynomial division (a TUPLE), NOT the element-wise coefficient division the implicit
        ///     <c>poly1d -&gt; NDArray</c> conversion would otherwise pick.
        /// </summary>
        public static (poly1d q, poly1d r) operator /(poly1d a, NDArray b)
        {
            var (q, rem) = np.polydiv(a._coeffs, new poly1d(b)._coeffs);
            return (new poly1d(q), new poly1d(rem));
        }

        /// <summary>Value equality of two polynomials (same coefficient shape and values).</summary>
        public static bool operator ==(poly1d a, poly1d b)
        {
            if (a is null) return b is null;
            if (b is null) return false;
            if (a._coeffs.size != b._coeffs.size) return false;
            return np.all(np.equal(a._coeffs, b._coeffs));
        }

        /// <summary>Value inequality of two polynomials.</summary>
        public static bool operator !=(poly1d a, poly1d b) => !(a == b);

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is poly1d p && this == p;

        /// <inheritdoc/>
        public override int GetHashCode() => (int)(_coeffs.size ^ (_coeffs.size << 16));

        /// <summary>Coefficient list, formatted as <c>poly1d([...])</c> (NumPy's <c>repr</c>).</summary>
        public override string ToString()
        {
            // NumPy's poly1d.__repr__: `repr(self.coeffs)[6:-1]` wrapped in poly1d(...) — i.e. the
            // array repr with the "array(" prefix and trailing ")" stripped. NumSharp's array_repr
            // (ToString(true)) is byte-exact to NumPy's, so this reproduces the alignment/precision.
            string vals = _coeffs.ToString(true);
            if (vals.StartsWith("array(", StringComparison.Ordinal) && vals.EndsWith(")", StringComparison.Ordinal))
                vals = vals.Substring(6, vals.Length - 7);
            return $"poly1d({vals})";
        }
    }

    public static partial class np
    {
        /// <summary>Evaluate a polynomial <paramref name="p"/> at another polynomial (composition).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyval.html</remarks>
        public static poly1d polyval(poly1d p, poly1d x)
        {
            // NumPy: y = 0; for pv in p.coeffs: y = y*x + pv  — polynomial composition.
            poly1d y = new poly1d(NDArray.Scalar(0));
            for (long k = 0; k < p.coeffs.size; k++)
                y = y * x + p.coeffs[k.ToString()];
            return y;
        }
    }
}
