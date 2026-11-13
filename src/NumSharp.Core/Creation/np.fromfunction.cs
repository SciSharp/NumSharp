using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Constructs an array by executing a function over each coordinate — NumPy's
        ///     <c>np.fromfunction(function, shape, *, dtype=float)</c>: <paramref name="function"/> is called ONCE with one
        ///     index grid per dimension (NumPy's <c>np.indices(shape, dtype)</c>), and its result is returned as is.
        /// </summary>
        /// <param name="function">
        ///     The function, called once with the coordinate grids — NumPy's <c>function(*args)</c> with the grids as one
        ///     array: <c>ix =&gt; ix[0] + ix[1]</c> is NumPy's <c>lambda i, j: i + j</c>, for any rank. It must be
        ///     vectorized (it receives whole grids, not single coordinates). NumPy's <c>**kwargs</c> (passed on to the
        ///     function) are values the C# lambda captures instead.
        /// </param>
        /// <param name="shape">The output shape, which also sets each grid's shape: a tuple (<c>(2, 3)</c>), an
        ///     <c>int[]</c> / <c>long[]</c> or a <see cref="Shape"/>.</param>
        /// <param name="dtype">The grids' dtype (NumPy's <c>dtype=float</c> default): null is float64; pass
        ///     <c>np.int64</c> for integer coordinates.</param>
        /// <returns>Whatever <paramref name="function"/> returns — usually an array of <paramref name="shape"/>; a C#
        ///     number the lambda returns arrives as a 0-d array (C# converts it to the <see cref="NDArray"/> the delegate
        ///     returns, where NumPy hands back the Python number itself).</returns>
        /// <exception cref="ValueError">A negative dimension (<c>negative dimensions are not allowed</c>), raised before
        ///     <paramref name="function"/> runs, as NumPy's <c>np.indices</c> raises it.</exception>
        /// <exception cref="TypeError"><paramref name="function"/> is null (<c>'NoneType' object is not callable</c>,
        ///     raised after the grids are built, as in NumPy).</exception>
        /// <exception cref="Exception">Whatever <paramref name="function"/> throws, unchanged.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html — NumPy's body is
        ///     <c>args = indices(shape, dtype=dtype); return function(*args, **kwargs)</c>.
        ///     <para><b>Ownership</b> (the <c>np.ma.fromfunction</c> contract): whatever <paramref name="function"/>
        ///     returns is the CALLER's, so the call runs outside any scope of this method's. The index grids handed to the
        ///     function are this method's own and are released when it returns (on a throw too) — so a function must not
        ///     keep its ARGUMENTS past the call. A grid it returns as is IS the result and is kept; a view of a grid holds
        ///     its own reference to the grid buffer.</para>
        ///     <para>Two C# differences (pinned as [Misaligned] tests): an <c>int</c> <paramref name="shape"/> converts to
        ///     the 1-D shape <c>(n,)</c>, where NumPy needs a tuple (<c>TypeError: 'int' object is not iterable</c>); and
        ///     the number-result conversion above. <c>like=</c> is not modelled (the parity plan's exclusion). Added to
        ///     <c>np</c> in API-parity batch 7 (it existed as <c>np.ma.fromfunction</c>).</para>
        /// </remarks>
        public static NDArray fromfunction(Func<NDArray[], NDArray> function, Shape shape, DType dtype = null)
        {
            // NumPy's np.indices allocates the (ndim, *shape) grid first: a negative dimension is its ValueError, before the
            // function is looked at.
            long[] dims = shape.dimensions ?? Array.Empty<long>();
            foreach (long d in dims)
                if (d < 0)
                    throw new ValueError("negative dimensions are not allowed");

            // Shape (ndim, *dims); grid[i] is the i-th coordinate grid, float64 unless the caller asked otherwise.
            NDArray grid = indices(new Shape(dims, copy: false), dtype ?? float64);
            var args = new NDArray[dims.Length];
            NDArray result = null;
            try
            {
                for (int i = 0; i < dims.Length; i++)
                    args[i] = grid[i];
                // Python's call of None: the grids exist by now (NumPy computes them first), then the call fails.
                if (function is null)
                    throw new TypeError("'NoneType' object is not callable");
                result = function(args);
                return result;
            }
            finally
            {
                // Release the grid views the function was handed — except one it returned verbatim, which IS the result —
                // then the grid wrapper itself. A returned view of the grid keeps the buffer alive through its own counted
                // reference, so this never frees memory the result still reads.
                foreach (var arg in args)
                    if (arg is not null && !ReferenceEquals(arg, result))
                        arg.Dispose();
                grid.Dispose();
            }
        }
    }
}
