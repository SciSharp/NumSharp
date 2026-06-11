using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Numerical positive, element-wise.
        /// This is an identity operation - returns +x (a copy of the input).
        /// Equivalent to np.array(a, copy=True).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.positive.html</remarks>
        public NDArray positive()
        {
            // NumPy: positive has identity loops for every numeric dtype EXCEPT
            // bool ('b->b'..'G->G' in ufunc.types, no '?->?') — probed 2.4.2,
            // text verbatim.
            if (GetTypeCode == NPTypeCode.Boolean)
                throw new TypeError(
                    "ufunc 'positive' did not contain a loop with signature matching types " +
                    "<class 'numpy.dtypes.BoolDType'> -> None");

            // np.positive is the identity function: +x == x
            // It returns a copy of the array, preserving all values as-is
            return this.Clone();
        }
    }
}
