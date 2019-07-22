#if _REGEN_TEMPLATE
%template "../Add/Default.Add.#1.cs" for every except(supported_currently_supported, "Boolean"), except(supported_currently_supported_lowercase, "bool")
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
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        [MethodImpl((MethodImplOptions)768)]
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [SuppressMessage("ReSharper", "CompareOfFloatsByEqualityOperator")]
        public unsafe NDArray Add__1__(in NDArray lhs, in NDArray rhs)
        {
            //lhs is NDArray of __2__
            switch (rhs.GetTypeCode)
            {
#if _REGEN
                %op = "+"
                %op_bool = "|"
                case NPTypeCode.Boolean:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Convert.ChangeType(*((__2__*)lhs.Address) %(op) (*((bool*)rhs.Address) ? (__2__) 1 : (__2__) 0), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (bool*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = (*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) != 0) %(op_bool) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToByte(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToChar(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToDouble(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToSingle(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToDecimal(((__2__)*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetIndexInShape(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

	            %foreach except(supported_currently_supported, "Boolean"), except(supported_currently_supported_lowercase, "bool")%
                case NPTypeCode.#1:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Convert.ChangeType(*((__2__*)lhs.Address) #(op) *((#2*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (#2*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToByte(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt16(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt32(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToInt64(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToChar(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToDouble(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToSingle(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDIndexArrayIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetIndexInShape(current)) = Convert.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetIndexInShape(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetIndexInShape(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                %
                default:
		            throw new NotSupportedException();
#else

#endif
            }
        }
    }
}
