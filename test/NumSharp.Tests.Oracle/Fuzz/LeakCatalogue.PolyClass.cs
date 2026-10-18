using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for the members of numpy.polynomial's six series classes that the polyclass tier
    ///     (<c>polyclass.jsonl</c>) does not reach through a key of the same Python name: <c>Dispose</c> (C#'s release,
    ///     with no NumPy counterpart), <c>GetHashCode</c> (which raises, NumPy's <c>__hash__ = None</c>), the operators
    ///     of the base class <see cref="NDPolyBase"/>, which bind only where a series is typed <see cref="NDPolyBase"/>
    ///     (the tier's registry binds dynamically, so each concrete class's own operators run there), and the
    ///     numpy.polynomial package's re-exports of the classes (<c>np.polynomial.Chebyshev(...)</c>; the tier's
    ///     constructor cases call the submodules' factories).
    /// </summary>
    /// <remarks>
    ///     Every entry builds its series inside the entry and releases them before it returns, except the results it
    ///     returns, which the harness disposes (a series is <see cref="IDisposable"/>). A series copies its coefficients,
    ///     so the fixture arrays it is built from are never shared with it.
    /// </remarks>
    internal static partial class LeakCatalogue
    {
        /// <summary>The series-class members outside the polyclass tier's keys.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddPolyClass(List<LeakCase> l)
        {
            // The six classes with their (coef, domain) constructors, keyed by the leak owner id. A local, not a static field
            // of this partial file: LeakCatalogue.All runs Build() from LeakCatalogue.cs's static initializer, and the
            // initializers of a partial class's files run in an order C# does not define, so such a field could still be null.
            var seriesClasses = new (string Owner, Func<object, object, NDPolyBase> New)[]
            {
                ("NDPolynomial", (c, d) => new NDPolynomial(c, d)),
                ("NDChebyshev", (c, d) => new NDChebyshev(c, d)),
                ("NDLegendre", (c, d) => new NDLegendre(c, d)),
                ("NDLaguerre", (c, d) => new NDLaguerre(c, d)),
                ("NDHermite", (c, d) => new NDHermite(c, d)),
                ("NDHermiteE", (c, d) => new NDHermiteE(c, d)),
            };
            foreach (var (owner, make) in seriesClasses)
            {
                E(l, owner + ".Dispose", "releases the copied coefficients and the series' own domain", f =>
                {
                    var p = make(f.V3, new object[] { 0.0, 2.0 });
                    p.Dispose();
                    return null;
                });
                T(l, owner + ".GetHashCode", "raises by design (a series is unhashable: NumPy's __hash__ = None)", f =>
                {
                    using var p = make(f.V3, null);
                    return Box(p.GetHashCode());
                });
            }

            // The numpy.polynomial package's re-exports of the classes (np.polynomial.Chebyshev(...)): the tier's constructor
            // cases call the submodules' factories (np.polynomial.chebyshev.Chebyshev), so the package's are measured here.
            E(l, "np.polynomial.Polynomial", "coefficients and a domain", f => np.polynomial.Polynomial(f.V3, new object[] { 0.0, 2.0 }));
            E(l, "np.polynomial.Chebyshev", "coefficients and a domain", f => np.polynomial.Chebyshev(f.V3, new object[] { 0.0, 2.0 }));
            E(l, "np.polynomial.Legendre", "coefficients and a domain", f => np.polynomial.Legendre(f.V3, new object[] { 0.0, 2.0 }));
            E(l, "np.polynomial.Laguerre", "coefficients and a domain", f => np.polynomial.Laguerre(f.V3, new object[] { 0.0, 2.0 }));
            E(l, "np.polynomial.Hermite", "coefficients and a domain", f => np.polynomial.Hermite(f.V3, new object[] { 0.0, 2.0 }));
            E(l, "np.polynomial.HermiteE", "coefficients and a domain", f => np.polynomial.HermiteE(f.V3, new object[] { 0.0, 2.0 }));

            // NDPolyBase's own operators: the static type NDPolyBase selects them over the concrete classes' sets.
            E(l, "NDPolyBase.op_Addition", "series + series, series + number, array + series, list + series", f =>
            {
                using NDPolyBase p = new NDChebyshev(f.V3);
                using NDPolyBase q = new NDChebyshev(f.V3);
                return new object[] { p + q, p + 2.5, f.Zero + p, new object[] { 1.0 } + p };
            });
            // A C# array binds the (series, Array) overload: no call-site NDArray conversion, so nothing for the GC.
            E(l, "NDPolyBase.op_Addition", "series + C# array (NumPy's p + [1, 2])", f =>
            {
                using NDPolyBase p = new NDChebyshev(f.V3);
                return p + new[] { 1.0, 2.0, 3.0, 4.0 };
            });
            T(l, "NDPolyBase.op_Addition", "raises by design (series + str: NumPy's TypeError)", f =>
            {
                using NDPolyBase p = new NDChebyshev(f.V3);
                return p + "abcd";
            });
            E(l, "NDPolyBase.op_Subtraction", "series - series, series - number, array - series", f =>
            {
                using NDPolyBase p = new NDLegendre(f.V3);
                using NDPolyBase q = new NDLegendre(f.V3);
                return new object[] { p - q, p - 2.5, f.Zero - p };
            });
            E(l, "NDPolyBase.op_Multiply", "series * series, series * number, array * series", f =>
            {
                using NDPolyBase p = new NDHermite(f.V3);
                using NDPolyBase q = new NDHermite(f.V3);
                return new object[] { p * q, p * 2.5, f.Zero * p };
            });
            E(l, "NDPolyBase.op_Division", "series / number (true division by a number)", f =>
            {
                using NDPolyBase p = new NDLaguerre(f.V3);
                return p / 2.5;
            });
            T(l, "NDPolyBase.op_Division", "raises by design (number / series: NumPy's __rtruediv__ is NotImplemented)", f =>
            {
                using NDPolyBase p = new NDLaguerre(f.V3);
                return f.Zero / p;
            });
            E(l, "NDPolyBase.op_Modulus", "series % series, series % number", f =>
            {
                using NDPolyBase p = new NDHermiteE(f.V3);
                using NDPolyBase q = new NDHermiteE(new object[] { 1.0, 1.0 });
                return new object[] { p % q, p % 2.5 };
            });
            E(l, "NDPolyBase.op_UnaryNegation", "-series", f =>
            {
                using NDPolyBase p = new NDPolynomial(f.V3);
                return -p;
            });
            // +p is p itself (NumPy's __pos__): the entry's own using releases it, the harness's dispose is a no-op.
            E(l, "NDPolyBase.op_UnaryPlus", "+series (the series itself)", f =>
            {
                using NDPolyBase p = new NDPolynomial(f.V3);
                return Box(ReferenceEquals(+p, p));
            });
        }
    }
}
