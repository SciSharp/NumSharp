using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Extract a diagonal or construct a diagonal array.
        /// </summary>
        /// <param name="v">
        ///     If <paramref name="v"/> is 2-D, return a copy of its <paramref name="k"/>-th
        ///     diagonal. If <paramref name="v"/> is 1-D, return a 2-D array with
        ///     <paramref name="v"/> on the <paramref name="k"/>-th diagonal.
        /// </param>
        /// <param name="k">
        ///     Diagonal in question. Use <c>k &gt; 0</c> for diagonals above the main diagonal,
        ///     and <c>k &lt; 0</c> for diagonals below the main diagonal.
        /// </param>
        /// <returns>The extracted diagonal or constructed diagonal array.</returns>
        /// <exception cref="ArgumentException">
        ///     <paramref name="v"/> is not 1- or 2-dimensional —
        ///     <c>Input must be 1- or 2-d.</c> (NumPy's <c>ValueError</c>).
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        ///     <para>
        ///     <b>The two branches differ in view-ness</b>, exactly as in NumPy:
        ///     the 1-D branch <em>constructs</em> a fresh, writeable, C-contiguous
        ///     <c>(n, n)</c> array (<c>n = v.size + |k|</c>); the 2-D branch delegates to
        ///     <see cref="diagonal"/> and therefore returns a <b>read-only view</b> that shares
        ///     storage with <paramref name="v"/> — despite the NumPy docstring's talk of "a
        ///     copy" (probed against 2.4.2: <c>np.shares_memory</c> is True and
        ///     <c>flags.writeable</c> is False).
        ///     </para>
        /// </remarks>
        public static NDArray diag(NDArray v, int k = 0)
        {
            if (v is null) throw new ArgumentNullException(nameof(v));

            switch (v.ndim)
            {
                case 1:
                    return DiagonalEmbed(v, k);
                case 2:
                    return diagonal(v, k);
                default:
                    // NumPy: raise ValueError("Input must be 1- or 2-d.")
                    throw new ArgumentException("Input must be 1- or 2-d.");
            }
        }

        /// <summary>
        ///     Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">Input data, which is flattened (in C order) and set as the <paramref name="k"/>-th diagonal of the output.</param>
        /// <param name="k">
        ///     Diagonal to set; 0, the default, corresponds to the "main" diagonal, a positive
        ///     (negative) <paramref name="k"/> giving the number of the diagonal above (below)
        ///     the main.
        /// </param>
        /// <returns>
        ///     The 2-D output array of shape <c>(n, n)</c> where <c>n = v.size + |k|</c>.
        ///     Always a freshly allocated, C-contiguous, writeable array.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html
        ///     <para>
        ///     Unlike <see cref="diag"/>, <c>diagflat</c> accepts <b>any</b> dimensionality —
        ///     the input is raveled first, so a 0-d scalar produces a <c>(1, 1)</c> array and a
        ///     3-D input is flattened in C order. Ravel order is <b>logical</b> C order, so an
        ///     F-contiguous or strided input is read in row-major index order, not memory order
        ///     (probed against NumPy 2.4.2).
        ///     </para>
        /// </remarks>
        public static NDArray diagflat(NDArray v, int k = 0)
        {
            if (v is null) throw new ArgumentNullException(nameof(v));
            return DiagonalEmbed(v.ravel(), k);
        }

        /// <summary>
        ///     Shared back-end of <see cref="diag"/>'s 1-D branch and <see cref="diagflat"/>:
        ///     allocate an <c>(n, n)</c> zero matrix and drop <paramref name="v1d"/> onto its
        ///     <paramref name="k"/>-th diagonal.
        /// </summary>
        /// <remarks>
        ///     NumPy writes the values through a strided flat slice
        ///     (<c>res.flat[i::n+1] = v</c>). NumSharp does the structurally identical thing but
        ///     without an element loop: it <see cref="Backends.Unmanaged.UnmanagedStorage.Alias(Shape)"/>es
        ///     a <b>writeable</b> length-<c>v.size</c>, stride-<c>(n+1)</c> view onto the fresh
        ///     zero buffer and lets the ordinary copy machinery fill it. The view is internal and
        ///     never escapes — it is a write target, not a result — which is why it does not carry
        ///     <see cref="diagonal"/>'s read-only contract.
        /// </remarks>
        private static NDArray DiagonalEmbed(NDArray v1d, int k)
        {
            long s = v1d.size;
            long absK = System.Math.Abs((long)k);
            long n = s + absK;

            if (n > int.MaxValue)
                throw new ArgumentException(
                    $"array is too big; `arr.size * arr.itemsize` is larger than the maximum possible size (n={n}).");

            var res = np.zeros(Shape.Matrix((int)n, (int)n), v1d.typecode);
            if (s == 0)
                return res;

            // Flat index of the first element of the k-th diagonal, then a constant
            // stride of n+1 walks it — the same arithmetic NumPy's flat-slice uses.
            long start = k >= 0 ? k : absK * n;
            var diagShape = new Shape(new[] {s}, new[] {n + 1}, start, res.Shape.BufferSize);

            var target = new NDArray(res.Storage.Alias(diagShape)) {TensorEngine = res.TensorEngine};
            target.SetData(v1d);

            return res;
        }
    }
}
