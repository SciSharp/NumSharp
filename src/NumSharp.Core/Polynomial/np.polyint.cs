using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an antiderivative (indefinite integral) of a polynomial.
        /// </summary>
        /// <param name="p">Polynomial to integrate (coefficients, highest degree first).</param>
        /// <param name="m">Order of the antiderivative (default 1).</param>
        /// <param name="k">
        ///     Integration constants, highest-order term first. <c>null</c> (default) means all zero.
        ///     For <c>m == 1</c> a single scalar may be given.
        /// </param>
        /// <returns>The antiderivative — always floating (or complex) because of the term-wise division.</returns>
        /// <exception cref="ValueError">A negative <paramref name="m"/>, or fewer integration constants than
        ///     <paramref name="m"/>.</exception>
        /// <exception cref="TypeError">With <paramref name="m"/> above 0: a 0-d <paramref name="p"/> (<c>len() of unsized
        ///     object</c>), or text, whose division by the exponents has no loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyint.html</remarks>
        [NDScoped]
        public static NDArray polyint(NDArray p, int m = 1, NDArray k = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(p, k);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
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
                if (kk.size == 1 && m > 1)
                    kk = kk["0"] * np.ones(new Shape(m), NPTypeCode.Double);
                if (kk.size < m)
                    throw new ValueError("k must be a scalar or a rank-1 array of length 1 or >m.");
            }

            p = np.asarray(p);

            if (m == 0)
                return p;

            // NumPy's first integration is p.__truediv__(arange(len(p), 0, -1)): len() of a 0-d array refuses first, then
            // the division has no loop for text — a StringDType p is named by class beside the int64 exponents, a
            // fixed-width one meets the legacy resolver.
            if (p.ndim == 0)
                throw new TypeError("len() of unsized object");
            if (Backends.Strings.NDStringUfuncs.IsVString(p))
                throw Backends.Strings.NDStringUfuncs.VStringNoLoop("divide", new[] { p.dtype.Meta.ToString(), np.int64.Meta.ToString() });
            if (Backends.Strings.NDStringUfuncs.IsString(p))
                throw Backends.Strings.NDStringUfuncs.NotSupportedForInputTypes("divide");

            // Integrate once per order: prepend the term-wise integral p / [len, len-1, ..., 1] and
            // append the constant k[i]. Division by the int64 arange floats the whole result.
            NDArray result = p;
            for (int i = 0; i < m; i++)
            {
                long len = result.size;
                NDArray divided = np.true_divide(result, np.arange(len, 0L, -1L));
                NDArray kElem = kk[$"{i}:{i + 1}"];
                result = np.concatenate(new[] { divided, kElem }, 0);
            }

            return result;
        }

        /// <summary>Convenience overload — <c>polyint</c> with a single scalar integration constant.</summary>
        [NDScoped] // the 0-d k wrapper is built in THIS frame, before the main overload's scope opens
        public static NDArray polyint(NDArray p, int m, double k)
        {
            using var plain = NDMatrixBoundary.PlainFor(p);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return polyint(p, m, NDArray.Scalar(k));
        }
    }
}
