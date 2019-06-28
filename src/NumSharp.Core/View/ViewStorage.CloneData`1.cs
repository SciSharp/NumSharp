using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;
using NumSharp.Utilities;

#if _REGEN_GLOBAL
    %supportedTypes = ["NDArray","Complex","Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Char","Double","Single","Decimal","String"]
    %supportTypesLower = ["NDArray","Complex","bool","byte","short","ushort","int","uint","long","ulong","char","double","float","decimal","string"]

    %supportedTypes_Primitives = ["Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Char","Double","Single","Decimal","String"]
    %supportTypesLower_Primitives = ["bool","byte","short","ushort","int","uint","long","ulong","char","double","float","decimal","string"]
#endif

namespace NumSharp
{
    public partial class ViewStorage
    {
        /// <summary>
        ///     Copies all elements this <see cref="ViewStorage"/> represent and if necessary casts to <typeparamref name="T"/>
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as T[]</returns>
        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        public T[] CloneData<T>()
        {
            //todo! unit test this
            // since the view is a subset of the data we have to copy here
            var returnType = typeof(T);
            var returnTypeCode = returnType.GetTypeCode();
            int size = _nonReducedShape.Size;
            var allocated = Arrays.Create(returnTypeCode, size);
            //dtypes are equal
            if (returnType == DType)
                return (T[])GetData(); 

            var sourceType = _data.DType;
            var sourceTypeCode = sourceType.GetTypeCode();

            //Inner regen generation
#if __REGEN
                %foreach supportedTypes_Primitives%
                case NPTypeCode.#1:
                {
                    var output = (OutputType[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToOutputType(Get#1(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToOutputType(Get#1(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
#endif
            switch (returnTypeCode)
            {
#if _REGEN
            %foreach supportedTypes_Primitives%
            case NPTypeCode.#1: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (#1[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.To#1(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
#else

            case NPTypeCode.Boolean: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Boolean[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToBoolean(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Byte: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Byte[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToByte(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Int16: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Int16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt16(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.UInt16: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (UInt16[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt16(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Int32: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Int32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt32(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.UInt32: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (UInt32[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt32(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Int64: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Int64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToInt64(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.UInt64: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (UInt64[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToUInt64(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Char: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Char[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToChar(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Double: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Double[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDouble(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Single: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Single[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToSingle(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.Decimal: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (Decimal[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToDecimal(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
            case NPTypeCode.String: {
            switch (sourceTypeCode)
            {

                case NPTypeCode.NDArray:
                {
                    throw new NotSupportedException("NDArray is only assignable to NDArray");
                }

                case NPTypeCode.Complex:
                {
                    Console.WriteLine("ComplexWarning: Casting complex values to real discards the imaginary part");

                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetComplex(i).Real);
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetComplex(_nonReducedShape.GetDimIndexOutShape(i)).Real);
                    }

                    return (T[])(object)output;
                        throw new NotSupportedException();
                }

                case NPTypeCode.Boolean:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetBoolean(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetBoolean(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Byte:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetByte(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetByte(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int16:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt16:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt16(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt16(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int32:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt32:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt32(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt32(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Int64:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.UInt64:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt64(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetUInt64(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Char:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetChar(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetChar(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Double:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetDouble(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetDouble(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Single:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetSingle(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetSingle(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.Decimal:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetDecimal(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetDecimal(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                case NPTypeCode.String:
                {
                    var output = (String[])allocated;
                    if (_slices.Length == 1)
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetString(i));
                    }
                    else
                    {
                        for (var i = 0; i < size; i++)
                            output[i] = Convert.ToString(GetString(_nonReducedShape.GetDimIndexOutShape(i)));
                    }

                    return (T[])(object)output;
                }
                default:
                    throw new ArgumentOutOfRangeException();
                }}
#endif

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
