using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override unsafe NDArray ReduceCumAdd(in NDArray arr, int? axis_, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty || shape.size == 0)
                return arr;

            if (shape.IsScalar || shape.size == 1 && shape.dimensions.Length == 1)
                return typeCode.HasValue ? Cast(arr, typeCode.Value, copy: true) : arr.Clone();

            if (axis_ == null)
            {
                var r = cumsum_elementwise(arr, typeCode);
                if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 and can be squeezed out.
                return np.squeeze_fast(arr, axis);
            }

            //prepare ret
            var ret = new NDArray(typeCode ?? (arr.GetTypeCode.GetAccumulatingType()), shape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var slices = iterAxis.Slices;

#if _REGEN1
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (ret.GetTypeCode)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_accumulatingType_defaultvals,supported_numericals_accumulatingType%
			            case NPTypeCode.#101: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<#2>();
                                var iterAxedRet = ret[slices].AsIterator<#102>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                |#102 sum = default;
                                while (hasNext())
                                {
                                    sum += (#102) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (arr.GetTypeCode)
		    {
			    case NPTypeCode.Byte: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int32: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int64: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Single: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Double: 
                {
                    switch (ret.GetTypeCode)
		            {
			            case NPTypeCode.Byte: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                byte sum = default;
                                while (hasNext())
                                {
                                    sum += (byte) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                int sum = default;
                                while (hasNext())
                                {
                                    sum += (int) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                long sum = default;
                                while (hasNext())
                                {
                                    sum += (long) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                float sum = default;
                                while (hasNext())
                                {
                                    sum += (float) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var iterAxedRet = ret[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                var setNext = iterAxedRet.MoveNextReference;
                                double sum = default;
                                while (hasNext())
                                {
                                    sum += (double) moveNext();
                                    setNext() = sum;
                                }
                            } while (iterAxis.Next() != null);
                            break;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif

            return ret;
        }

        public NDArray CumSumElementwise<T>(in NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var ret = cumsum_elementwise(arr, typeCode);
            return typeCode.HasValue && typeCode.Value != ret.typecode ? ret.astype(typeCode.Value, true) : ret;
        }

        protected unsafe NDArray cumsum_elementwise(in NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retType, Shape.Vector(arr.size));

#if _REGEN1
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase%
			            case NPTypeCode.#101: 
                        {
                            var iter = arr.AsIterator<#2>();
                            var addr = (#102*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            |#102 sum = default;
                            while (hasNext())
                            {
                                sum += (#102) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            %
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (arr.GetTypeCode)
		    {
			    case NPTypeCode.Byte: 
                {
                    switch (retType)
		            {
			            case NPTypeCode.Byte: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int32: 
                {
                    switch (retType)
		            {
			            case NPTypeCode.Byte: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<int>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Int64: 
                {
                    switch (retType)
		            {
			            case NPTypeCode.Byte: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<long>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Single: 
                {
                    switch (retType)
		            {
			            case NPTypeCode.Byte: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<float>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    case NPTypeCode.Double: 
                {
                    switch (retType)
		            {
			            case NPTypeCode.Byte: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (byte*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            byte sum = default;
                            while (hasNext())
                            {
                                sum += (byte) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (int*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            int sum = default;
                            while (hasNext())
                            {
                                sum += (int) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (long*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            long sum = default;
                            while (hasNext())
                            {
                                sum += (long) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (float*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            float sum = default;
                            while (hasNext())
                            {
                                sum += (float) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<double>();
                            var addr = (double*)ret.Address;
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int i = 0;
                            double sum = default;
                            while (hasNext())
                            {
                                sum += (double) moveNext();
                                *(addr + i++) = sum;
                            }

                            return ret;
                        }
			            default:
				            throw new NotSupportedException();
		            }
                    break;
                }
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }
    }
}
