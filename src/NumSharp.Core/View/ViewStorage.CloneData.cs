//using System;
//using System.Numerics;
//using NumSharp.Backends;
//using NumSharp.Utilities;

//namespace NumSharp
//{
//    public partial class ViewStorage
//    {
//        /// <summary>
//        ///     Copies all elements this <see cref="ViewStorage"/> represent and if necessary casts to <typeparamref name="T"/>
//        /// </summary>
//        /// <returns>reference to cloned storage as T[]</returns>
//        public Array CloneData()
//        {
//            // since the view is a subset of the data we have to copy here
//            int size = _nonReducedShape.Size;
//            var returnTypeCode = _data.TypeCode;
//            var allocated = Arrays.Create(returnTypeCode, size);
//            // the algorithm is split into 1-D and N-D because for 1-D we need not go through coordinate transformation

//            switch (returnTypeCode)
//            {
//#if _REGEN
//                %foreach supported_dtypes%
//                case NPTypeCode.#1:
//                {
//                    var arr = (#1[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = Get#1(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = Get#1(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                } 
//                %
//                default:
//                    throw new NotImplementedException();
//#else
//                case NPTypeCode.NDArray:
//                {
//                    var arr = (NDArray[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetNDArray(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetNDArray(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Complex:
//                {
//                    var arr = (Complex[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetComplex(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetComplex(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Boolean:
//                {
//                    var arr = (Boolean[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetBoolean(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetBoolean(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Byte:
//                {
//                    var arr = (Byte[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetByte(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetByte(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Int16:
//                {
//                    var arr = (Int16[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt16(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt16(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.UInt16:
//                {
//                    var arr = (UInt16[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt16(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt16(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Int32:
//                {
//                    var arr = (Int32[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt32(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt32(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.UInt32:
//                {
//                    var arr = (UInt32[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt32(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt32(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Int64:
//                {
//                    var arr = (Int64[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt64(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetInt64(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.UInt64:
//                {
//                    var arr = (UInt64[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt64(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetUInt64(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Char:
//                {
//                    var arr = (Char[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetChar(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetChar(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Double:
//                {
//                    var arr = (Double[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetDouble(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetDouble(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Single:
//                {
//                    var arr = (Single[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetSingle(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetSingle(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.Decimal:
//                {
//                    var arr = (Decimal[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetDecimal(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetDecimal(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                case NPTypeCode.String:
//                {
//                    var arr = (String[])allocated;
//                    if (_slices.Length == 1)
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetString(i);
//                    }
//                    else
//                    {
//                        for (var i = 0; i < size; i++)
//                            arr[i] = GetString(_nonReducedShape.GetDimIndexOutShape(i));
//                    }

//                    return arr;
//                }

//                default:
//                    throw new NotImplementedException();
//#endif
//            }
//        }
//    }
//}


