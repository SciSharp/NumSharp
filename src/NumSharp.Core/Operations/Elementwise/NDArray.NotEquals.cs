using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise not-equal comparison (!=).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator !=(NDArray lhs, NDArray rhs)
        {
            if (lhs is null && rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            if (lhs is null || rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return Scalar<bool>(true).MakeGeneric<bool>();

            return lhs.TensorEngine.NotEqual(lhs, rhs);
        }

        /// <summary>
        /// Element-wise not-equal comparison with scalar (!=).
        /// </summary>
        public static NDArray<bool> operator !=(NDArray lhs, object rhs)
        {
            if (lhs is null)
                return Scalar<bool>(rhs != null).MakeGeneric<bool>();

            if (rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            return lhs != np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise not-equal comparison with scalar on left (!=).
        /// </summary>
        public static NDArray<bool> operator !=(object lhs, NDArray rhs)
        {
            if (rhs is null)
                return Scalar<bool>(lhs != null).MakeGeneric<bool>();

            if (lhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            return np.asanyarray(lhs) != rhs;
        }
    }
}
