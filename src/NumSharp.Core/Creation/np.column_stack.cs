using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack 1-D arrays as columns into a 2-D array.
        ///     Take a sequence of 1-D arrays and stack them as columns to make a
        ///     single 2-D array. 2-D arrays are stacked as-is, just like with
        ///     <see cref="hstack"/>. 1-D arrays are turned into 2-D columns first.
        /// </summary>
        /// <param name="tup">
        ///     Arrays to stack. All of them must have the same first dimension.
        /// </param>
        /// <returns>The 2-D array formed by stacking the given arrays.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html</remarks>
        public static NDArray column_stack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // Port of numpy/lib/_shape_base_impl.py::column_stack —
            //   arr = array(arr, copy=None, subok=True, ndmin=2).T  when ndim < 2
            // ndmin=2 prepends a leading 1 as a VIEW: (N,) → (1,N), () → (1,1);
            // the transpose then turns the row into a column: (N,1) / (1,1).
            // Inputs of ndim >= 2 pass through untouched. Errors (empty tup,
            // ndim/shape mismatches, promotion) all flow from concatenate,
            // exactly as in NumPy.
            var arrays = new NDArray[tup.Length];
            NDArray[] columnViews = null; // views created here; released post-concat
            for (int i = 0; i < tup.Length; i++)
            {
                var arr = tup[i];
                if (arr is null)
                    throw new ArgumentNullException($"{nameof(tup)}[{i}]");

                if (arr.ndim < 2)
                {
                    arr = AsColumnView(arr);
                    (columnViews ??= new NDArray[tup.Length])[i] = arr;
                }

                arrays[i] = arr;
            }

            try
            {
                return np.concatenate(arrays, 1);
            }
            finally
            {
                // The (N,1) wrappers alias caller storage and are dead after the
                // copy — release them eagerly (same pattern as concatenate's
                // axis=null ravel intermediates).
                if (columnViews != null)
                    for (int i = 0; i < columnViews.Length; i++)
                        columnViews[i]?.Dispose();
            }
        }

        /// <summary>
        ///     <c>array(arr, copy=None, ndmin=2).T</c> as ONE view: (N,) → (N,1)
        ///     and () → (1,1), preserving stride/offset (a strided 1-D input stays
        ///     a view, like NumPy's ndmin path). The dim-1 column stride mirrors
        ///     what the (1,N)-transpose produces (N*stride), keeping the view
        ///     both C- and F-contiguous for contiguous inputs — which is what the
        ///     concatenate layout vote (ambiguous → C order) keys off, per NumPy.
        /// </summary>
        private static NDArray AsColumnView(NDArray arr)
        {
            var sh = arr.Shape;
            long n, stride0;
            if (arr.ndim == 0)
            {
                n = 1;
                stride0 = 1;
            }
            else
            {
                n = sh.dimensions[0];
                stride0 = sh.strides[0];
            }

            var column = new Shape(new[] { n, 1L }, new[] { stride0, n * stride0 }, sh.offset, sh.bufferSize);
            return new NDArray(arr.Storage.Alias(column), arr.TensorEngine, skipEngineResolve: true);
        }
    }
}
