using System;
using System.Diagnostics;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class SingleEngine
    {
        public unsafe override NDArray Mod(in NDArray lhs, in NDArray rhs)
        {
            //if return type is scalar
            var ret_type = np._FindCommonType(lhs, rhs);
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                return NDArray.Scalar(*(float*)lhs.Address % *(float*)rhs.Address);

            (Shape leftshape, Shape rightshape) = Broadcast(lhs.Shape, rhs.Shape);
            var lhs_address = (float*)lhs.Address;
            var rhs_address = (float*)rhs.Address;
            var retShape = leftshape.Clean();
            var ret = new NDArray(ret_type, retShape, false);
            var leftLinear = !leftshape.IsBroadcasted && !leftshape.IsSliced;
            var rightLinear = !rightshape.IsBroadcasted && !rightshape.IsSliced;
            var len = ret.size;
            var ret_address = (float*)ret.Address;
            if (leftLinear && rightLinear)
            {
                Debug.Assert(leftshape.size == len && rightshape.size == len);
                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var rval = *rhs_address;
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = *(lhs_address + i) % rval;
                }
                else if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var lval = *lhs_address;
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = lval % *(rhs_address + i);
                }
                else
                {
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = *(lhs_address + i) % *(rhs_address + i);
                }
            }
            else if (leftLinear)
            { // && !rightLinear
                if (rightshape.IsBroadcasted && rightshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var rval = *rhs_address;
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = *(lhs_address + i) % rval;
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
                        *(ret_address + retOffset++) = *(lhs_address + leftOffset++) % *(rhs_address + rightOffset(current));
                    } while (incr.Next() != null);
                }
            }
            else if (rightLinear)
            { // !leftLinear && 
                if (leftshape.IsBroadcasted && leftshape.BroadcastInfo.OriginalShape.IsScalar)
                {
                    var lval = *lhs_address;
                    for (int i = 0; i < len; i++)
                        *(ret_address + i) = lval % *(rhs_address + i);
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
                        *(ret_address + retOffset++) = *(lhs_address + leftOffset(current)) % *(rhs_address + rightOffset++);
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
                    *(ret_address + retOffset++) = *(lhs_address + leftOffset(current)) % *(rhs_address + rightOffset(current));
                } while (incr.Next() != null);
            }

            return ret;
        }
    }
}
