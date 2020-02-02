using System;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public static unsafe NumSharp.Generic.NDArray<bool> operator >(NDArray lhs, NDArray rhs)
        {
#if _REGEN1
            if(lhs.typecode != rhs.typecode)
            {
                throw new NotImplementedException("please make sure operands have the same data type");
            }
            else if (lhs.typecode == NPTypeCode.Boolean)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>((*(bool*)lhs.Address ? 1 : 0) > (*(bool*)rhs.Address ? 1 : 0)).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) ? 1 : 0) > (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? 1 : 0);
                } while (incr.Next() != null);

                return ret;
            }
            %op = ">"
	        %foreach except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")%
            else if (lhs.typecode == NPTypeCode.#1)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(#2*)lhs.Address #(op) *(#2*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (#2*)lhs.Address;
                var rhs_address = (#2*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) #(op) *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            %
            else
            {
                throw new NotImplementedException($"{lhs.typecode} > {rhs.typecode}");
            }
#else
            if(lhs.typecode != rhs.typecode)
            {
                throw new NotImplementedException("please make sure operands have the same data type");
            }
            else if (lhs.typecode == NPTypeCode.Boolean)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>((*(bool*)lhs.Address ? 1 : 0) > (*(bool*)rhs.Address ? 1 : 0)).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) ? 1 : 0) > (*(rhs_address + BroadcastedRightShape.GetOffset(current)) ? 1 : 0);
                } while (incr.Next() != null);

                return ret;
            }
            else if (lhs.typecode == NPTypeCode.Byte)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(byte*)lhs.Address > *(byte*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (byte*)lhs.Address;
                var rhs_address = (byte*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) > *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Int32)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(int*)lhs.Address > *(int*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (int*)lhs.Address;
                var rhs_address = (int*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) > *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Int64)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(long*)lhs.Address > *(long*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (long*)lhs.Address;
                var rhs_address = (long*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) > *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Single)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(float*)lhs.Address > *(float*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (float*)lhs.Address;
                var rhs_address = (float*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) > *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Double)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar<bool>(*(double*)lhs.Address > *(double*)rhs.Address).MakeGeneric<bool>();

                (Shape BroadcastedLeftShape, Shape BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (double*)lhs.Address;
                var rhs_address = (double*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) > *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }

            else
            {
                throw new NotImplementedException($"{lhs.typecode} > {rhs.typecode}");
            }
#endif
        }

        public static NumSharp.Generic.NDArray<bool> operator >(NDArray np, int obj)
        {
            return (np > (System.Object)obj);
        }

        public static NumSharp.Generic.NDArray<bool> operator >(NDArray np, object obj)
        {
            return null;

            // var boolTensor = new NDArray(typeof(bool),np.shape);
            //bool[] bools = boolTensor.Storage.GetData() as bool[];

            //switch (np.Storage.GetData())
            //{
            //    case int[] values :
            //    {
            //        int value = Converts.ToInt32(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] > value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case Int64[] values :
            //    {
            //        Int64 value = Converts.ToInt64(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] > value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case float[] values :
            //    {
            //        float value = Converts.ToSingle(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] > value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case double[] values :
            //    {
            //        double value = Converts.ToDouble(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] > value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    default :
            //    {
            //        throw new IncorrectTypeException();
            //    } 
            //}

            //return boolTensor.MakeGeneric<bool>();
        }
    }
}
