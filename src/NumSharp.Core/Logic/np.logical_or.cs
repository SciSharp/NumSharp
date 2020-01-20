using System;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the truth value of x1 OR x2 element-wise.
        /// </summary>
        /// <param name="lhs">Input boolean array.</param>
        /// <param name="rhs">Input boolean array.</param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_or.html</remarks>
        public static unsafe NDArray<bool> logical_or(NDArray lhs, NDArray rhs)
        {
            if (lhs.typecode == NPTypeCode.Boolean && rhs.typecode == NPTypeCode.Boolean)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(bool*)lhs.Address || *(bool*)rhs.Address).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (bool*)lhs.Address;
                var rhs_address = (bool*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) || *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }
            else
            {
                throw new NotImplementedException($"{lhs.typecode} > {rhs.typecode}");
            }
        }
    }
}
