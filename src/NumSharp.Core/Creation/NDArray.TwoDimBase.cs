using System;
using NumSharp.Backends;
using NumSharp.Utilities;

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
        /// <param name="order"><c>'C'</c> or <c>'F'</c> (<c>'A'</c> and <c>'K'</c> mean C).</param>
        /// <returns>A new (N, M) array: one on the diagonal, zero elsewhere.</returns>
        /// <exception cref="ArgumentException"><paramref name="N"/> or the column count is negative (<c>negative dimensions
        ///     are not allowed</c>).</exception>
        /// <exception cref="ValueError">An order other than C, F, A or K.</exception>
        /// <remarks>An F-order result is the C result copied (<c>m.copy('F')</c>), so it costs a second allocation.</remarks>
        internal static NDArray Eye(int N, int? M, int k, DType dtype, char order)
        {
            int cols = M ?? N;
            if (N < 0)
                throw new ArgumentException($"negative dimensions are not allowed (N={N})", nameof(N));
            if (cols < 0)
                throw new ArgumentException($"negative dimensions are not allowed (M={cols})", nameof(M));

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

            var typeCode = resolvedType.GetTypeCode();
            object one;
            switch (typeCode)
            {
                case NPTypeCode.Complex: one = new System.Numerics.Complex(1d, 0d); break;
                case NPTypeCode.Half:    one = (Half)1; break;
                case NPTypeCode.SByte:   one = (sbyte)1; break;
                case NPTypeCode.String:  one = "1"; break;
                case NPTypeCode.Char:    one = (char)1; break;
                default:                 one = Converts.ChangeType((byte)1, typeCode); break;
            }

            // Flat index of element (i, i+k) in row-major (N, cols) layout = i*cols + (i+k).
            // `flat` is an owning view wrapper around `m`'s storage — used only to drive
            // SetAtIndex over the diagonal, never returned.
            using var flat = m.flat;
            for (int i = rowStart; i < rowEnd; i++)
                flat.SetAtIndex(one, (long)i * cols + (i + k));

            return physical == 'F' ? m.copy('F') : m;
        }
    }
}
