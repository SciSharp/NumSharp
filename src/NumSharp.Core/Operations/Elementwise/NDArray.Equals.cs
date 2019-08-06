using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Determines if NDArray data is same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            unsafe
            {
                if (obj is null)
                    return false;

                if (ReferenceEquals(this, obj))
                    return true;

                // Using this comparison allows less restrictive semantics,
                // like comparing a scalar to an array
                // we can use unmanaged access because the result of == op is never a slice.
                var results = (this == obj);
                var len = results.size;
                var addr = results.Address;

                for (int i = 0; i < len; i++)
                    if (!*(addr + i))
                        return false;

                return true;
            }
        }

        public static NDArray<bool> operator ==(NDArray left, object right)
        {
            if (right is null)
                return Scalar<bool>(ReferenceEquals(left, null)).MakeGeneric<bool>();

            if (left is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            if (left.Shape.IsEmpty || left.size == 0)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return left.TensorEngine.Compare(left, np.asanyarray(right));
        }

        /// NumPy signature: numpy.equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'equal'>
        /// <summary>
        /// Compare two NDArrays element wise
        /// </summary>
        /// <param name="np2">NDArray to compare with</param>
        /// <returns>NDArray with result of each element compare</returns>
        private NDArray<bool> equal(NDArray np2)
        {
            return this == np2;
        }

        /// <summary>
        ///     True if two arrays have the same shape and elements, False otherwise.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="rhs">Input array.</param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.array_equal.html</remarks>
        public bool array_equal(NDArray rhs)
        {
            unsafe
            {
                if (ReferenceEquals(this, rhs))
                    return true;

                if (ReferenceEquals(Storage, rhs.Storage))
                    return true;

                //this is the same memory block
                if ((IntPtr)this.Address == (IntPtr)rhs.Address && this.size == rhs.size && this.typecode == rhs.typecode)
                    return true;

                //if shape is different
                if (Shape != rhs.Shape)
                    return false;

                //compare all values
                var cmp = (this == rhs);
                var len = cmp.size;
                var ptr = cmp.Address; //this never a slice so we can use unmanaged memory.
                for (int i = 0; i < len; i++)
                    if (!*(ptr + i))
                        return false;

                return true;
            }
        }

    }
}
