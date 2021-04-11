using System;
using System.Diagnostics;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class ByteEngine
    {
        public unsafe override NDArray<bool> Compare(in NDArray lhs, in NDArray rhs)
        {
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                return NDArray.Scalar(*(byte*)lhs.Address == *(byte*)rhs.Address).MakeGeneric<bool>();
            (Shape leftshape, Shape rightshape) = Broadcast(lhs.Shape, rhs.Shape);
            var lhs_address = (byte*)lhs.Address;
            var rhs_address = (byte*)rhs.Address;
            var ret = new NDArray<bool>(new Shape(leftshape.dimensions), false);
            Shape retShape = ret.Shape;
            var leftLinear = !leftshape.IsBroadcasted && !leftshape.IsSliced;
            var rightLinear = !rightshape.IsBroadcasted && !rightshape.IsSliced;

            var ret_address = ret.Address;
            if (leftLinear && rightLinear)
            {
                var len = ret.size;
                Debug.Assert(leftshape.size == len && rightshape.size == len);
                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var rval = *rhs_address;
                    for (int i = 0; i < ret.size; i++)
                        *(ret_address + i) = *(lhs_address + i) == rval;
                }
                else if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var lval = *lhs_address;
                    for (int i = 0; i < ret.size; i++)
                        *(ret_address + i) = lval == *(rhs_address + i);
                }
                else
                {
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = *(lhs_address + i) == *(rhs_address + i);
                }
            }
            else if (leftLinear)
            { // && !rightLinear
                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var rval = *rhs_address;
                    for (int i = 0; i < ret.size; i++)
                        *(ret_address + i) = *(lhs_address + i) == rval;
                }
                else
                {
                    int leftOffset = 0;
                    int retOffset = 0;
                    var incr = new NDCoordinatesIncrementor(ref retShape);
                    int[] current = incr.Index;
                    Func<int[], int> rightOffset = rightshape.GetOffset;
                    do
                    {
                        *(ret_address + retOffset++) = *(lhs_address + leftOffset++) == *(rhs_address + rightOffset(current));
                    } while (incr.Next() != null);
                }
            }
            else if (rightLinear)
            { // !leftLinear && 
                if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var lval = *lhs_address;
                    for (int i = 0; i < ret.size; i++)
                        *(ret_address + i) = lval == *(rhs_address + i);
                }
                else
                {
                    int rightOffset = 0;
                    int retOffset = 0;
                    var incr = new NDCoordinatesIncrementor(ref retShape);
                    int[] current = incr.Index;
                    Func<int[], int> leftOffset = leftshape.GetOffset;
                    do
                    {
                        *(ret_address + retOffset++) = *(lhs_address + leftOffset(current)) == *(rhs_address + rightOffset++);
                    } while (incr.Next() != null);
                }
            }
            else
            {
                int retOffset = 0;
                var incr = new NDCoordinatesIncrementor(ref retShape);
                int[] current = incr.Index;
                Func<int[], int> rightOffset = rightshape.GetOffset;
                Func<int[], int> leftOffset = leftshape.GetOffset;
                do
                {
                    *(ret_address + retOffset++) = *(lhs_address + leftOffset(current)) == *(rhs_address + rightOffset(current));
                } while (incr.Next() != null);
            }

            return ret;
        }
    }
}
