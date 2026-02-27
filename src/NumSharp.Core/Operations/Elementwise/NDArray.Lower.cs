using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise less-than comparison (&lt;).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator <(NDArray lhs, NDArray rhs)
        {
            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs.TensorEngine.Less(lhs, rhs);
        }

        /// <summary>
        /// Element-wise less-than comparison with scalar (&lt;).
        /// </summary>
        public static NDArray<bool> operator <(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs < np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise less-than comparison with scalar on left (&lt;).
        /// </summary>
        public static NDArray<bool> operator <(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) < rhs;
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison (&lt;=).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator <=(NDArray lhs, NDArray rhs)
        {
            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs.TensorEngine.LessEqual(lhs, rhs);
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison with scalar (&lt;=).
        /// </summary>
        public static NDArray<bool> operator <=(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return lhs <= np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise less-than-or-equal comparison with scalar on left (&lt;=).
        /// </summary>
        public static NDArray<bool> operator <=(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) <= rhs;
        }
    }
}
