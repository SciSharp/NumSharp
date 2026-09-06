using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Check if the array is Fortran contiguous but <b>not</b> C contiguous — exactly NumPy's
        ///     <c>a.flags.fnc</c> (<c>numpy/_core/numeric.py::isfortran</c>, a pure flags read).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>
        ///     True iff <paramref name="a"/> is F-contiguous and not C-contiguous. Note the asymmetry
        ///     NumPy documents: a 1-D (or 0-d / empty) array is BOTH C- and F-contiguous, so
        ///     <c>isfortran</c> is False for it — this reports column-major MEMORY ORDER, not mere
        ///     F-contiguity (use <c>a.flags.f_contiguous</c> for that).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isfortran.html</remarks>
        public static bool isfortran(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return a.Shape.IsFContiguous && !a.Shape.IsContiguous;
        }
    }
}
