using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The Python functions of NumPy's lib/_twodim_base_impl.py over the array object (Appendix D's NDArray.TwoDimBase.cs):
    // np.eye, `m = zeros((N, M), dtype=dtype, order=order); m[:M-k].flat[i::M+1] = 1`. It is the array object's own member
    // (the np-layer dtype plan's D1: functions that create arrays), and np.eye and np.identity are facades over it.
    //
    // The file sits beside its facades (Creation/) rather than under Backends/ because its statements still call np.*
    // functions (np.zeros), which ratchet R2 forbids under Backends/ (plan C3).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.eye</c> (<c>_twodim_base_impl.py</c>), the body of
        ///     <see cref="np.eye(int, int?, int, DType, char, string)"/> once the device was checked: an (N, M) array of zeros
        ///     with the <paramref name="k"/>-th diagonal set to one, in C or F order.
        /// </summary>
        /// <param name="N">The number of rows.</param>
        /// <param name="M">The number of columns, or null for <paramref name="N"/>.</param>
        /// <param name="k">The diagonal: 0 the main one, a positive value an upper one, a negative value a lower one. A
        ///     diagonal outside the matrix leaves it all zeros.</param>
        /// <param name="dtype">The dtype, or null for float64.</param>
        /// <param name="order"><c>'C'</c> or <c>'F'</c>, in either case.</param>
        /// <returns>A new (N, M) array: one on the diagonal, zero elsewhere.</returns>
        /// <exception cref="ValueError">An order other than C or F (<c>'A'</c> and <c>'K'</c>: <c>only 'C' or 'F' order is
        ///     permitted</c>, as NumPy's zeros refuses them), checked first; <paramref name="N"/> or the column count is
        ///     negative (<c>negative dimensions are not allowed</c>, the allocation's text); or a dtype whose <c>setitem</c>
        ///     refuses the Python int 1 (a <c>StringDType(coerce=False)</c>), raised after the zeros are allocated and before
        ///     any diagonal element is written.</exception>
        /// <remarks>An F-order result is the C result copied (<c>m.copy('F')</c>), so it costs a second allocation.</remarks>
        internal static unsafe NDArray Eye(int N, int? M, int k, DType dtype, char order)
        {
            int cols = M ?? N;
            // np.zeros' own order of checks once np.eye checked the order's letter and the device: A and K are refused
            // (only C and F lay out a new array), then a negative dimension by the allocation (AllocationGuard's NumPy text,
            // "negative dimensions are not allowed", for N or M alike).
            char physical = OrderResolver.Resolve(order);

            var resolvedType = dtype ?? DType.Double;
            var m = np.zeros(Shape.Matrix(N, cols), resolvedType);
            if (N == 0 || cols == 0)
                return physical == 'F' ? m.copy('F') : m;

            // Diagonal element count: rows where 0 <= i < N and 0 <= i+k < cols
            int rowStart = Math.Max(0, -k);
            int rowEnd = Math.Min(N, cols - k);
            if (rowEnd <= rowStart)
                return physical == 'F' ? m.copy('F') : m;

            // NumPy's `m[:M-k].flat[i::M+1] = 1`: the flat iterator's assignment converts the Python int 1 ONCE through the
            // class's setitem (PyArray_Pack: True, 1, 1.0, 1+0j, '1' for text; a StringDType(coerce=False) refuses it, as
            // NumPy's does) and copies that element into the other diagonal slots. Element (i, i+k) of the fresh C-ordered
            // (N, cols) array is flat element i*cols + (i+k).
            var descr = m.dtype;
            var meta = descr.Meta;
            long itemsize = descr.itemsize;
            byte* data = m.Storage.Address;
            IArraySlice buffer = m.Storage.InternalArray;
            byte* first = data + ((long)rowStart * cols + (rowStart + k)) * itemsize;
            try
            {
                meta.setitem(descr, 1L, first, buffer);
                // A class whose items hold references (PyDataType_REFCHK: StringDType, whose cell points into the array's
                // allocator) makes its copies through the class too, as NumPy's copy loop for it allocates a string per
                // element: each slot is stored by setitem. Every other class copies the converted element's bytes.
                bool references = meta.HasReferences;
                for (int i = rowStart + 1; i < rowEnd; i++)
                {
                    byte* slot = data + ((long)i * cols + (i + k)) * itemsize;
                    if (references)
                        meta.setitem(descr, 1L, slot, buffer);
                    else
                        Buffer.MemoryCopy(first, slot, itemsize, itemsize);
                }
            }
            catch
            {
                // The class refused the one (a StringDType(coerce=False)): the zeros never reach the caller, as NumPy drops
                // its m when the assignment raises.
                m.Dispose();
                throw;
            }

            return physical == 'F' ? m.copy('F') : m;
        }
    }
}
