using System;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the truth value of x1 AND x2 element-wise.
        /// </summary>
        /// <param name="lhs">Input boolean array.</param>
        /// <param name="rhs">Input boolean array.</param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_and.html</remarks>
        public static unsafe NDArray<bool> logical_and(NDArray lhs, NDArray rhs)
        {
#if _REGEN
            if(lhs.typecode != rhs.typecode)
            {
                throw new NotImplementedException("please make sure operands have the same data type");
            }
            else if (lhs.typecode == NPTypeCode.Boolean)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(bool*)lhs.Address && *(bool*)rhs.Address).MakeGeneric<bool>();

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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) && *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }
            %op = "&&"
	        %foreach except(supported_dtypes, "Boolean"), except(supported_dtypes_lowercase, "bool")%
            else if (lhs.typecode == NPTypeCode.#1)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(#2*)lhs.Address > 0 && *(#2*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            %
            else
            {
                throw new NotImplementedException($"{lhs.typecode} && {rhs.typecode}");
            }
#else

            if(lhs.typecode != rhs.typecode)
            {
                throw new NotImplementedException("please make sure operands have the same data type");
            }
            else if (lhs.typecode == NPTypeCode.Boolean)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(bool*)lhs.Address && *(bool*)rhs.Address).MakeGeneric<bool>();

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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current))) && *(rhs_address + BroadcastedRightShape.GetOffset(current));
                } while (incr.Next() != null);

                return ret;
            }
            else if (lhs.typecode == NPTypeCode.Byte)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(byte*)lhs.Address > 0 && *(byte*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Int16)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(short*)lhs.Address > 0 && *(short*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (short*)lhs.Address;
                var rhs_address = (short*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.UInt16)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(ushort*)lhs.Address > 0 && *(ushort*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (ushort*)lhs.Address;
                var rhs_address = (ushort*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Int32)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(int*)lhs.Address > 0 && *(int*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.UInt32)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(uint*)lhs.Address > 0 && *(uint*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (uint*)lhs.Address;
                var rhs_address = (uint*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Int64)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(long*)lhs.Address > 0 && *(long*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.UInt64)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(ulong*)lhs.Address > 0 && *(ulong*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (ulong*)lhs.Address;
                var rhs_address = (ulong*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Char)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(char*)lhs.Address > 0 && *(char*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (char*)lhs.Address;
                var rhs_address = (char*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Double)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(double*)lhs.Address > 0 && *(double*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Single)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(float*)lhs.Address > 0 && *(float*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
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
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else if (lhs.typecode == NPTypeCode.Decimal)
            {
                if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
                    return NDArray.Scalar(*(decimal*)lhs.Address > 0 && *(decimal*)rhs.Address > 0).MakeGeneric<bool>();

                var (BroadcastedLeftShape, BroadcastedRightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);
                var lhs_address = (decimal*)lhs.Address;
                var rhs_address = (decimal*)rhs.Address;
                var ret = new NDArray<bool>(new Shape(BroadcastedLeftShape.dimensions), true);
                Shape retShape = ret.Shape;

                //iterate
                var ret_address = (bool*)ret.Address;
                var incr = new NDCoordinatesIncrementor(BroadcastedLeftShape.dimensions); //doesn't matter which side it is.
                int[] current = incr.Index;
                do
                {
                    *(ret_address + retShape.GetOffset(current)) = (*(lhs_address + BroadcastedLeftShape.GetOffset(current)) > 0) && (*(rhs_address + BroadcastedRightShape.GetOffset(current)) > 0);
                } while (incr.Next() != null);

                return ret;
            }

            else
            {
                throw new NotImplementedException($"{lhs.typecode} && {rhs.typecode}");
            }
#endif
        }
    }
}
