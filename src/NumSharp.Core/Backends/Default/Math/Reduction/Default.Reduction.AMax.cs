using System;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null)
        {
            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || shape.size == 1 && shape.dimensions.Length == 1)
            {
                var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);

                return r;
            }

            if (axis_ == null)
            {
                var r = NDArray.Scalar(amax_elementwise(arr, typeCode));
                if (keepdims)
                    r.Storage.ExpandDimension(0);
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
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
                if (keepdims)
                    return new NDArray(arr.Storage.Alias());
                return np.squeeze_fast(arr, axis);
            }

            //handle keepdims
            Shape axisedShape = Shape.GetAxis(shape, axis);

            //prepare ret
            var ret = new NDArray(typeCode ?? arr.GetTypeCode, axisedShape, false);
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var iterRet = new NDCoordinatesIncrementor(ref axisedShape);
            var iterIndex = iterRet.Index;
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
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals%
			            case NPTypeCode.#101: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<#2>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                |#102 max = (#102)moveNext();
                                while (hasNext())
                                    max = (#102) Math.Max((#102)moveNext(), max);

                                ret.Set#101(Converts.To#101(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte max = (byte)moveNext();
                                while (hasNext())
                                    max = (byte) Math.Max((byte)moveNext(), max);

                                ret.SetByte(Converts.ToByte(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int max = (int)moveNext();
                                while (hasNext())
                                    max = (int) Math.Max((int)moveNext(), max);

                                ret.SetInt32(Converts.ToInt32(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long max = (long)moveNext();
                                while (hasNext())
                                    max = (long) Math.Max((long)moveNext(), max);

                                ret.SetInt64(Converts.ToInt64(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float max = (float)moveNext();
                                while (hasNext())
                                    max = (float) Math.Max((float)moveNext(), max);

                                ret.SetSingle(Converts.ToSingle(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<byte>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double max = (double)moveNext();
                                while (hasNext())
                                    max = (double) Math.Max((double)moveNext(), max);

                                ret.SetDouble(Converts.ToDouble(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte max = (byte)moveNext();
                                while (hasNext())
                                    max = (byte) Math.Max((byte)moveNext(), max);

                                ret.SetByte(Converts.ToByte(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int max = (int)moveNext();
                                while (hasNext())
                                    max = (int) Math.Max((int)moveNext(), max);

                                ret.SetInt32(Converts.ToInt32(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long max = (long)moveNext();
                                while (hasNext())
                                    max = (long) Math.Max((long)moveNext(), max);

                                ret.SetInt64(Converts.ToInt64(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float max = (float)moveNext();
                                while (hasNext())
                                    max = (float) Math.Max((float)moveNext(), max);

                                ret.SetSingle(Converts.ToSingle(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<int>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double max = (double)moveNext();
                                while (hasNext())
                                    max = (double) Math.Max((double)moveNext(), max);

                                ret.SetDouble(Converts.ToDouble(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte max = (byte)moveNext();
                                while (hasNext())
                                    max = (byte) Math.Max((byte)moveNext(), max);

                                ret.SetByte(Converts.ToByte(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int max = (int)moveNext();
                                while (hasNext())
                                    max = (int) Math.Max((int)moveNext(), max);

                                ret.SetInt32(Converts.ToInt32(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long max = (long)moveNext();
                                while (hasNext())
                                    max = (long) Math.Max((long)moveNext(), max);

                                ret.SetInt64(Converts.ToInt64(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float max = (float)moveNext();
                                while (hasNext())
                                    max = (float) Math.Max((float)moveNext(), max);

                                ret.SetSingle(Converts.ToSingle(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<long>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double max = (double)moveNext();
                                while (hasNext())
                                    max = (double) Math.Max((double)moveNext(), max);

                                ret.SetDouble(Converts.ToDouble(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte max = (byte)moveNext();
                                while (hasNext())
                                    max = (byte) Math.Max((byte)moveNext(), max);

                                ret.SetByte(Converts.ToByte(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int max = (int)moveNext();
                                while (hasNext())
                                    max = (int) Math.Max((int)moveNext(), max);

                                ret.SetInt32(Converts.ToInt32(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long max = (long)moveNext();
                                while (hasNext())
                                    max = (long) Math.Max((long)moveNext(), max);

                                ret.SetInt64(Converts.ToInt64(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float max = (float)moveNext();
                                while (hasNext())
                                    max = (float) Math.Max((float)moveNext(), max);

                                ret.SetSingle(Converts.ToSingle(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<float>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double max = (double)moveNext();
                                while (hasNext())
                                    max = (double) Math.Max((double)moveNext(), max);

                                ret.SetDouble(Converts.ToDouble(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                byte max = (byte)moveNext();
                                while (hasNext())
                                    max = (byte) Math.Max((byte)moveNext(), max);

                                ret.SetByte(Converts.ToByte(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int32: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                int max = (int)moveNext();
                                while (hasNext())
                                    max = (int) Math.Max((int)moveNext(), max);

                                ret.SetInt32(Converts.ToInt32(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Int64: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                long max = (long)moveNext();
                                while (hasNext())
                                    max = (long) Math.Max((long)moveNext(), max);

                                ret.SetInt64(Converts.ToInt64(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Single: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                float max = (float)moveNext();
                                while (hasNext())
                                    max = (float) Math.Max((float)moveNext(), max);

                                ret.SetSingle(Converts.ToSingle(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
                            break;
                        }
			            case NPTypeCode.Double: 
                        {
                            do
                            {
                                var iter = arr[slices].AsIterator<double>();
                                var moveNext = iter.MoveNext;
                                var hasNext = iter.HasNext;
                                double max = (double)moveNext();
                                while (hasNext())
                                    max = (double) Math.Max((double)moveNext(), max);

                                ret.SetDouble(Converts.ToDouble(max), iterIndex);
                            } while (iterAxis.Next() != null && iterRet.Next() != null);
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

            if (keepdims)
                ret.Storage.ExpandDimension(axis);

            return ret;
        }

        public T AMaxElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            return (T)Converts.ChangeType(amax_elementwise(arr, typeCode), InfoOf<T>.NPTypeCode);
        }

        protected object amax_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.size == 1 && arr.Shape.NDim == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? arr.GetTypeCode;
#if _REGEN1
            #region Compute
            switch (arr.GetTypeCode)
		    {
			    %foreach supported_numericals,supported_numericals_lowercase%
			    case NPTypeCode.#1: 
                {
                    switch (retType)
		            {
			            %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals%
			            case NPTypeCode.#101: 
                        {
                            var iter = arr.AsIterator<#2>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            |#102 max = (#102)moveNext();
                            while (hasNext())
                                max = (#102) Math.Max((#102)moveNext(), max);

                            return Converts.To#101(max);
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
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            byte max = (byte)moveNext();
                            while (hasNext())
                                max = (byte) Math.Max((byte)moveNext(), max);

                            return Converts.ToByte(max);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int max = (int)moveNext();
                            while (hasNext())
                                max = (int) Math.Max((int)moveNext(), max);

                            return Converts.ToInt32(max);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long max = (long)moveNext();
                            while (hasNext())
                                max = (long) Math.Max((long)moveNext(), max);

                            return Converts.ToInt64(max);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float max = (float)moveNext();
                            while (hasNext())
                                max = (float) Math.Max((float)moveNext(), max);

                            return Converts.ToSingle(max);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<byte>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double max = (double)moveNext();
                            while (hasNext())
                                max = (double) Math.Max((double)moveNext(), max);

                            return Converts.ToDouble(max);
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
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            byte max = (byte)moveNext();
                            while (hasNext())
                                max = (byte) Math.Max((byte)moveNext(), max);

                            return Converts.ToByte(max);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int max = (int)moveNext();
                            while (hasNext())
                                max = (int) Math.Max((int)moveNext(), max);

                            return Converts.ToInt32(max);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long max = (long)moveNext();
                            while (hasNext())
                                max = (long) Math.Max((long)moveNext(), max);

                            return Converts.ToInt64(max);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float max = (float)moveNext();
                            while (hasNext())
                                max = (float) Math.Max((float)moveNext(), max);

                            return Converts.ToSingle(max);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<int>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double max = (double)moveNext();
                            while (hasNext())
                                max = (double) Math.Max((double)moveNext(), max);

                            return Converts.ToDouble(max);
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
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            byte max = (byte)moveNext();
                            while (hasNext())
                                max = (byte) Math.Max((byte)moveNext(), max);

                            return Converts.ToByte(max);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int max = (int)moveNext();
                            while (hasNext())
                                max = (int) Math.Max((int)moveNext(), max);

                            return Converts.ToInt32(max);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long max = (long)moveNext();
                            while (hasNext())
                                max = (long) Math.Max((long)moveNext(), max);

                            return Converts.ToInt64(max);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float max = (float)moveNext();
                            while (hasNext())
                                max = (float) Math.Max((float)moveNext(), max);

                            return Converts.ToSingle(max);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<long>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double max = (double)moveNext();
                            while (hasNext())
                                max = (double) Math.Max((double)moveNext(), max);

                            return Converts.ToDouble(max);
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
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            byte max = (byte)moveNext();
                            while (hasNext())
                                max = (byte) Math.Max((byte)moveNext(), max);

                            return Converts.ToByte(max);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int max = (int)moveNext();
                            while (hasNext())
                                max = (int) Math.Max((int)moveNext(), max);

                            return Converts.ToInt32(max);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long max = (long)moveNext();
                            while (hasNext())
                                max = (long) Math.Max((long)moveNext(), max);

                            return Converts.ToInt64(max);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float max = (float)moveNext();
                            while (hasNext())
                                max = (float) Math.Max((float)moveNext(), max);

                            return Converts.ToSingle(max);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<float>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double max = (double)moveNext();
                            while (hasNext())
                                max = (double) Math.Max((double)moveNext(), max);

                            return Converts.ToDouble(max);
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
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            byte max = (byte)moveNext();
                            while (hasNext())
                                max = (byte) Math.Max((byte)moveNext(), max);

                            return Converts.ToByte(max);
                        }
			            case NPTypeCode.Int32: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            int max = (int)moveNext();
                            while (hasNext())
                                max = (int) Math.Max((int)moveNext(), max);

                            return Converts.ToInt32(max);
                        }
			            case NPTypeCode.Int64: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            long max = (long)moveNext();
                            while (hasNext())
                                max = (long) Math.Max((long)moveNext(), max);

                            return Converts.ToInt64(max);
                        }
			            case NPTypeCode.Single: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            float max = (float)moveNext();
                            while (hasNext())
                                max = (float) Math.Max((float)moveNext(), max);

                            return Converts.ToSingle(max);
                        }
			            case NPTypeCode.Double: 
                        {
                            var iter = arr.AsIterator<double>();
                            var moveNext = iter.MoveNext;
                            var hasNext = iter.HasNext;
                            double max = (double)moveNext();
                            while (hasNext())
                                max = (double) Math.Max((double)moveNext(), max);

                            return Converts.ToDouble(max);
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
