#if _REGEN_TEMPLATE
%template "../Equals/Default.Equals.#1.cs" for every except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")
#endif

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        [MethodImpl((MethodImplOptions)768)]
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [SuppressMessage("ReSharper", "CompareOfFloatsByEqualityOperator")]
        public unsafe NDArray<bool> Equals__1__(in NDArray lhs, in NDArray rhs)
        {
            //lhs is NDArray of __2__
            switch (rhs.GetTypeCode)
            {
#if _REGEN1
                %op = "=="
                %op_bool = "=="
                case NPTypeCode.Boolean:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) %(op) (*((bool*)rhs.Address) ? (__2__) 1 : (__2__) 0))).MakeGeneric<bool>();;

                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (bool*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
                    //iterate
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op_bool) 0 != *(rhs_address + BroadcastedRightShape.GetOffset(current));
                    } while (incr.Next() != null);

                    return ret;
                }

	            %foreach except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")%
                case NPTypeCode.#1:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) #(op) *((#2*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (#2*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                %
                default:
		            throw new NotSupportedException();
#else

                case NPTypeCode.Boolean:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == (*((bool*)rhs.Address) ? (__2__) 1 : (__2__) 0))).MakeGeneric<bool>();;

                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (bool*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
                    //iterate
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) == 0 != *(rhs_address + BroadcastedRightShape.GetOffset(current));
                    } while (incr.Next() != null);

                    return ret;
                }

                case NPTypeCode.Byte:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == *((byte*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (byte*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) == (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                case NPTypeCode.Int32:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == *((int*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (int*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) == (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                case NPTypeCode.Int64:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == *((long*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (long*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) == (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                case NPTypeCode.Single:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == *((float*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (float*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) == (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                case NPTypeCode.Double:
                {
                    //if return type is scalar
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                        return NDArray.Scalar<bool>((*((__2__*)lhs.Address) == *((double*)rhs.Address))).MakeGeneric<bool>();
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (double*)rhs.Address;
                    var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                    Shape retShape = ret.Shape;
                    
		            var ret_address = (bool*)ret.Address;
                    var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                    int[] current = incr.Index;
                    do
                    {
                        *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) == (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                    } while (incr.Next() != null);

                    return ret;
                }

                default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
