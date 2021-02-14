#if _REGEN_TEMPLATE
%template "../Dot/Default.Dot.#1.cs" for every except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")
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
        public unsafe NDArray Dot__1__(in NDArray lhs, in NDArray rhs)
        {
            //lhs is NDArray of __2__
            switch (rhs.GetTypeCode)
            {
#if _REGEN1
                %op = "*"
                %op_bool = "*"
                case NPTypeCode.Boolean:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) %(op) (*((bool*)rhs.Address) ? (__2__) 1 : (__2__) 0), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (bool*)rhs.Address;
                    var ret = new NDArray(ret_type, BroadcastedLeftShape.Clean(), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToBoolean((*(lhs_address + BroadcastedLeftShape.GetOffset(current)) != 0 ? 1 : 0) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? 1 : 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) %(op) (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

	            %foreach except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")%
                case NPTypeCode.#1:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) #(op) *((#2*)rhs.Address), (TypeCode)ret_type));
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
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) #(op) (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
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

                case NPTypeCode.Boolean:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * (*((bool*)rhs.Address) ? (__2__) 1 : (__2__) 0), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (bool*)rhs.Address;
                    var ret = new NDArray(ret_type, BroadcastedLeftShape.Clean(), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToBoolean((*(lhs_address + BroadcastedLeftShape.GetOffset(current)) != 0 ? 1 : 0) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? 1 : 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(((__2__)*(lhs_address + BroadcastedLeftShape.GetOffset(current))) * (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? (__2__) 1 : (__2__) 0));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * *((byte*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (byte*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * *((int*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (int*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * *((long*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (long*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * *((float*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (float*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    //if return type is scalar
                    var ret_type = np._FindCommonType(lhs, rhs);
                    if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    {
                        return NDArray.Scalar(Converts.ChangeType(*((__2__*)lhs.Address) * *((double*)rhs.Address), (TypeCode)ret_type));
                    }
                    (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                    var lhs_address = (__2__*)lhs.Address;
                    var rhs_address = (double*)rhs.Address;
                    var ret = new NDArray(ret_type, new Shape(BroadcastedLeftShape.dimensions), false);
                    Shape retShape = ret.Shape;
                    
                    switch (ret_type)
                    {
	                    case NPTypeCode.Boolean:
	                    {
		                    var ret_address = (bool*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = 0 != (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Byte:
	                    {
		                    var ret_address = (byte*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToByte(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int16:
	                    {
		                    var ret_address = (short*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt16:
	                    {
		                    var ret_address = (ushort*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt16(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int32:
	                    {
		                    var ret_address = (int*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt32:
	                    {
		                    var ret_address = (uint*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt32(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Int64:
	                    {
		                    var ret_address = (long*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.UInt64:
	                    {
		                    var ret_address = (ulong*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToUInt64(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Char:
	                    {
		                    var ret_address = (char*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToChar(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Double:
	                    {
		                    var ret_address = (double*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDouble(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Single:
	                    {
		                    var ret_address = (float*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToSingle(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    case NPTypeCode.Decimal:
	                    {
		                    var ret_address = (decimal*)ret.Address;
                            var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                            int[] current = incr.Index;
                            do
                            {
                                *(ret_address + retShape.GetOffset(current)) = Converts.ToDecimal(*(lhs_address + BroadcastedLeftShape.GetOffset(current)) * (__2__) *(rhs_address + BroadcastedRightShape.GetOffset(current)));
                            } while (incr.Next() != null);

                            return ret;
	                    }
	                    default:
		                    throw new NotSupportedException();

                    }

                    break;
                }

                default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
