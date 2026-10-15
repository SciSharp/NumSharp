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
    public sealed class poly1d : IDisposable
    {
        // OWNED: the constructor yields the normalized coefficient array into this field (or copies
        // the source's), so the polynomial is the array's owner and Dispose releases it — the
        // ownership analyzer (NDW016) is what made that explicit.
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
            // Reclaim the transient coefficient arrays this chain builds. The from-roots branch's
            // np.poly returns a FRESH vector that np.trim_zeros then aliases as a VIEW — dropping
            // the base, whose buffer would dangle behind the view to a future GC. A hand-written
            // NDScope tracks every array constructed here and disposes all but the one kept for the
            // field; the ctor's egress is a FIELD (not a return), which the [NDScoped] weaver can't
            // express, so the scope is spelled out. Reclamation is ARC release, so keeping the
            // trim_zeros view keeps its base alive (the view holds the surviving ref). On the
            // from-coeffs path trim_zeros aliases the untracked operand, so nothing is reclaimed —
            // matching that path's already-clean balance.
            using var scope = NDScope.Open();

            if (r)
                c_or_r = np.poly(c_or_r);

            c_or_r = np.atleast_1d(c_or_r);
            if (c_or_r.ndim > 1)
                throw new ValueError("Polynomial must be 1d only.");

            c_or_r = np.trim_zeros(c_or_r, "f");
            if (c_or_r.size == 0)
                c_or_r = np.zeros(new Shape(1), c_or_r.typecode);

            // A FIELD egress: this polynomial owns the array and Dispose releases it. Detach (not Returns):
            // Returns re-tracks the array into the CALLER's ambient scope, which then released the
            // coefficients of any polynomial built inside an [NDScoped] method or a user NDScope that
            // outlived it (returned or stored) — a use-after-free of the coefficient view. The copy
            // constructor below already egresses this way. c_or_r is always an array constructed in this
            // scope (trim_zeros of a 1-D input always returns a NEW view; the all-zero case a new zeros),
            // never the caller's own array, so detaching it can never capture a borrowed input.
            NDScope.Detach(c_or_r);
            _coeffs = c_or_r;
            _variable = variable ?? "x";
        }

        /// <summary>
        ///     Copy constructor — copies the coefficients and (unless overridden) the variable name.
        ///     NumPy's copy shares the coefficient array by reference count; here each polynomial owns
        ///     its own array (see <see cref="Dispose"/>), so a copy is taken rather than an alias that two
        ///     owners would dispose.
        /// </summary>
        public poly1d(poly1d source, string variable = null)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            var c = source._coeffs.copy();
            NDScope.Detach(c); // a field egress: owned by this polynomial, not by any ambient scope
            _coeffs = c;
            _variable = variable ?? source._variable;
        }

        /// <summary>
        ///     Releases the coefficient array this polynomial owns. Idempotent. The polynomial is the
        ///     array's ONLY owner (both constructors detach it from every ambient scope), so a polynomial
        ///     that is never disposed keeps its coefficient buffer referenced until a GC finalizes it —
        ///     dispose every polynomial you create, including temporaries.
        /// </summary>
        public void Dispose()
        {
            var c = _coeffs;
            _coeffs = null;
            c?.Dispose();
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

        /// <summary>Return an antiderivative (indefinite integral) of this polynomial — a NEW polynomial the caller owns.</summary>
        /// <param name="m">Order of the antiderivative.</param>
        /// <param name="k">Integration constants (NumPy's <c>k</c>); null means zeros.</param>
        /// <returns>The antiderivative; dispose it.</returns>
        /// <exception cref="ValueError"><paramref name="m"/> is negative, or <paramref name="k"/> has the wrong length (np.polyint).</exception>
        public poly1d integ(int m = 1, NDArray k = null)
        {
            // The default integration constant is this method's own 0-d temp — released here, not left to a GC.
            NDArray zero = k is null ? NDArray.Scalar(0.0) : null;
            try
            {
                return FromFresh(np.polyint(_coeffs, m, k ?? zero));
            }
            finally
            {
                zero?.Dispose();
            }
        }

        /// <summary>Return a derivative of this polynomial — a NEW polynomial the caller owns.</summary>
        /// <param name="m">Order of the derivative.</param>
        /// <returns>The derivative; dispose it.</returns>
        /// <exception cref="ValueError"><paramref name="m"/> is negative (np.polyder).</exception>
        public poly1d deriv(int m = 1) => FromFresh(np.polyder(_coeffs, m));

        /// <summary>
        ///     Wraps a FRESH coefficient array — a np.poly* / arithmetic result this class computed — in a new
        ///     polynomial, then releases the intermediate's own reference.
        /// </summary>
        /// <remarks>
        ///     The constructor never keeps <paramref name="fresh"/> itself: it keeps a trim_zeros VIEW of it (or
        ///     a fresh zeros for an all-zero input), which holds its own counted reference. Without this release
        ///     every derived polynomial (operators, <see cref="deriv"/>, <see cref="integ"/>) stranded the
        ///     intermediate's reference — its buffer outlived the polynomial until a GC (measured by the scope
        ///     audit: one pooled buffer per call).
        /// </remarks>
        /// <param name="fresh">An array this class just computed and nobody else references.</param>
        /// <returns>The new polynomial (the caller owns and disposes it).</returns>
        /// <exception cref="ValueError"><paramref name="fresh"/> is not 1-D (the constructor's check).</exception>
        private static poly1d FromFresh(NDArray fresh)
        {
            try
            {
                return new poly1d(fresh);
            }
            finally
            {
                fresh.Dispose();
            }
        }

        /// <summary>
        ///     Evaluates the polynomial at <paramref name="x"/> — NumPy's <c>p(x)</c> (<c>poly1d.__call__</c>), which C#
        ///     cannot spell as an invocation of the object itself; equivalent to <c>np.polyval(p, x)</c>.
        /// </summary>
        public NDArray Call(NDArray x) => np.polyval(_coeffs, x);

        /// <summary>Evaluates the polynomial at a scalar — NumPy's <c>p(x)</c> for a Python float.</summary>
        /// <param name="x">The point to evaluate at.</param>
        /// <returns>The polynomial's value at <paramref name="x"/> (a double; a complex polynomial's real part is not
        /// representable here — use <see cref="Call(NDArray)"/>).</returns>
        public double Call(double x)
        {
            // Both the 1-element operand and polyval's result are this method's temps: released now, not by a GC.
            using var xs = np.array(new[] { x });
            using var y = np.polyval(_coeffs, xs);
            return y.GetDouble(0);
        }

        /// <summary>
        ///     The coefficient of <c>x**k</c> (NumPy's <c>p[k]</c>): a 0-d zero of the coefficient dtype when
        ///     <paramref name="k"/> is out of <c>[0, order]</c>.
        /// </summary>
        public NDArray this[int k]
        {
            get
            {
                if (k < 0 || k > order)
                {
                    // The int 0-d seed is a temp; the astype COPY is the caller's result.
                    using var zero = NDArray.Scalar(0);
                    return zero.astype(_coeffs.typecode);
                }
                return _coeffs[(order - k).ToString()];
            }
            set
            {
                if (k < 0)
                    throw new ValueError("Does not support negative powers.");
                int ind;
                if (k > order)
                {
                    // Growing REPLACES the owned field: the zero pad is a temp, the old coefficient array is
                    // released (this polynomial was its only owner), and the concatenation is detached from any
                    // ambient scope — a field egress, exactly like the constructors.
                    using (NDArray zr = np.zeros(new Shape(k - order), _coeffs.typecode))
                    {
                        var grown = np.concatenate(new[] { zr, _coeffs }, 0);
                        NDScope.Detach(grown);
                        var old = _coeffs;
                        _coeffs = grown;
                        old.Dispose();
                    }
                    ind = 0;
                }
                else
                {
                    ind = order - k;
                }
                // atleast_1d returns `value` itself when it is already >= 1-D (never dispose the caller's array);
                // otherwise a new 1-D view whose reference is released after the write.
                var v = np.atleast_1d(value);
                try
                {
                    _coeffs[$"{ind}:{ind + 1}"] = v;
                }
                finally
                {
                    if (!ReferenceEquals(v, value))
                        v.Dispose();
                }
            }
        }

        /// <summary>Sum of two polynomials.</summary>
        public static poly1d operator +(poly1d a, poly1d b) => FromFresh(np.polyadd(a._coeffs, b._coeffs));

        /// <summary>Sum of a polynomial and coefficients (NumPy's <c>poly1d.__add__</c> -> polyadd).</summary>
        public static poly1d operator +(poly1d a, NDArray b)
        {
            using var pb = new poly1d(b); // normalizes b (trims leading zeros); its coefficient view is released on exit
            return FromFresh(np.polyadd(a._coeffs, pb._coeffs));
        }

        // No `operator +(NDArray, poly1d)`: NumPy's `array + poly1d` is ELEMENT-WISE (the ndarray wins
        // via poly1d.__array__), NOT polyadd — the implicit poly1d->NDArray conversion already yields
        // that. A reflected polyadd overload would both diverge from NumPy AND hijack `"string" + p`
        // (the string implicitly becomes a char NDArray) into a garbage polyadd. Same for subtraction.

        /// <summary>Difference of two polynomials.</summary>
        public static poly1d operator -(poly1d a, poly1d b) => FromFresh(np.polysub(a._coeffs, b._coeffs));

        /// <summary>Difference of a polynomial and coefficients (NumPy's <c>poly1d.__sub__</c> -> polysub).</summary>
        public static poly1d operator -(poly1d a, NDArray b)
        {
            using var pb = new poly1d(b); // normalizes b; its coefficient view is released on exit
            return FromFresh(np.polysub(a._coeffs, pb._coeffs));
        }

        /// <summary>Negation.</summary>
        public static poly1d operator -(poly1d a) => FromFresh(-a._coeffs);

        /// <summary>Unary plus (returns the same polynomial).</summary>
        public static poly1d operator +(poly1d a) => a;

        /// <summary>Product of two polynomials.</summary>
        public static poly1d operator *(poly1d a, poly1d b) => FromFresh(np.polymul(a._coeffs, b._coeffs));

        /// <summary>Product of a polynomial and coefficients.</summary>
        public static poly1d operator *(poly1d a, NDArray b)
        {
            using var pb = new poly1d(b); // normalizes b; its coefficient view is released on exit
            return FromFresh(np.polymul(a._coeffs, pb._coeffs));
        }

        /// <summary>Scale every coefficient by a scalar (NumPy's <c>isscalar</c> branch).</summary>
        public static poly1d operator *(poly1d a, double s) => FromFresh(a._coeffs * s);

        /// <summary>Scale every coefficient by a scalar.</summary>
        public static poly1d operator *(double s, poly1d a) => FromFresh(s * a._coeffs);

        /// <summary>Divide every coefficient by a scalar.</summary>
        public static poly1d operator /(poly1d a, double s) => FromFresh(a._coeffs / s);

        /// <summary>Polynomial division — returns <c>(quotient, remainder)</c> as poly1d objects.</summary>
        public static (poly1d q, poly1d r) operator /(poly1d a, poly1d b)
        {
            var (q, rem) = np.polydiv(a._coeffs, b._coeffs);
            return WrapQuotient(q, rem);
        }

        /// <summary>
        ///     Polynomial division by raw coefficients — <c>(quotient, remainder)</c>. NumPy's
        ///     <c>poly1d.__truediv__</c> wraps a non-scalar in <c>poly1d</c> then divides, so <c>p / array</c>
        ///     is polynomial division (a TUPLE), NOT the element-wise coefficient division the implicit
        ///     <c>poly1d -&gt; NDArray</c> conversion would otherwise pick.
        /// </summary>
        public static (poly1d q, poly1d r) operator /(poly1d a, NDArray b)
        {
            using var pb = new poly1d(b); // normalizes b; its coefficient view is released on exit
            var (q, rem) = np.polydiv(a._coeffs, pb._coeffs);
            return WrapQuotient(q, rem);
        }

        /// <summary>
        ///     Wraps a np.polydiv <c>(q, r)</c> pair as two NEW polynomials (see <see cref="FromFresh"/>),
        ///     releasing the remainder's array too if wrapping the quotient throws.
        /// </summary>
        /// <param name="q">The fresh quotient coefficients.</param>
        /// <param name="rem">The fresh remainder coefficients.</param>
        /// <returns>The quotient and remainder polynomials (the caller owns and disposes both).</returns>
        private static (poly1d q, poly1d r) WrapQuotient(NDArray q, NDArray rem)
        {
            poly1d pq;
            try
            {
                pq = FromFresh(q);
            }
            catch
            {
                rem.Dispose();
                throw;
            }
            return (pq, FromFresh(rem));
        }

        /// <summary>Value equality of two polynomials (same coefficient shape and values).</summary>
        public static bool operator ==(poly1d a, poly1d b)
        {
            if (a is null) return b is null;
            if (b is null) return false;
            if (a._coeffs.size != b._coeffs.size) return false;
            // The element-wise comparison is a temp bool array — released here rather than left to a GC.
            using var eq = np.equal(a._coeffs, b._coeffs);
            return np.all(eq);
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
            // NumPy: y = 0; for pv in p.coeffs: y = y*x + pv  — polynomial composition. Each Horner
            // step builds two polynomials (the product, then the sum that becomes the accumulator);
            // the product and the superseded accumulator are transients this method owns and releases.
            // The 0-d seed is a temp too: the accumulator keeps its own view of it (ARC keeps the scalar alive).
            using var zero = NDArray.Scalar(0);
            poly1d y = new poly1d(zero);
            for (long k = 0; k < p.coeffs.size; k++)
            {
                using var yx = y * x;
                using var pv = p.coeffs[k.ToString()]; // the k-th coefficient as a 0-d view
                var next = yx + pv;
                y.Dispose();
                y = next;
            }
            return y;
        }
    }
}
