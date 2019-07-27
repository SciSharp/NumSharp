using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using NumSharp.Generic;
using System.Linq;
using NumSharp.Backends;

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
            return this == obj;
        }

        public static NDArray<bool> operator ==(NDArray left, object right)
        {
            if (right is null)
                return Scalar<bool>(ReferenceEquals(left, null)).MakeGeneric<bool>();

            if (left is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            //rhs is a number
            var rhs_type = right.GetType();
            if (rhs_type.IsPrimitive || rhs_type == typeof(decimal)) //numerical
            {
                if (left.Shape.IsEmpty || left.size == 0)
                    return Scalar<bool>(false).MakeGeneric<bool>();

                return left.TensorEngine.Compare(left, Scalar((ValueType)right));
            }

            if (right is NDArray rarr)
                return left.TensorEngine.Compare(left, rarr);

            throw new NotSupportedException();
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
                //this is the same memory block
                if ((IntPtr)this.Address == (IntPtr)rhs.Address && this.size == rhs.size && GetTypeCode == rhs.GetTypeCode)
                    return true;

                //if shape is different
                if (Shape != rhs.Shape)
                    return false;

                //compare all values
                var cmp = (this == rhs);
                var len = cmp.size;
                for (int i = 0; i < len; i++)
                    if (!cmp.GetAtIndex<bool>(i))
                        return false;

                return true;
            }
        }
    }
}
