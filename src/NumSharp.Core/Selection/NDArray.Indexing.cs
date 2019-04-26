using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Get and set element wise data
        /// Low performance
        /// Use generic Data<T> and SetData<T>(value, shape) method for better performance
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public NDArray this[params int[] select]
        {
            get
            {
                return GetData(select);
            }

            set
            {
                Storage.SetData(value, select);
            }
        }

        public NDArray this[NDArray indices]
        {
            get
            {
                NDArray nd = null;

                switch (Type.GetTypeCode(dtype))
                {
                    case TypeCode.Byte:
                        nd = setValue<byte>(indices);
                        break;
                    case TypeCode.Int32:
                        nd = setValue<int>(indices);
                        break;
                    case TypeCode.Int64:
                        nd = setValue<long>(indices);
                        break;
                    case TypeCode.Single:
                        nd = setValue<float>(indices);
                        break;
                    case TypeCode.Double:
                        nd = setValue<double>(indices);
                        break;
                    case TypeCode.Decimal:
                        nd = setValue<decimal>(indices);
                        break;
                }

                return nd;
            }

            set
            {

            }
        }

        public NDArray this[string slice]
        {
            get
            {
                var s = new Slice(slice);
                s.Start = s.Start.HasValue ? s.Start.Value : 0;
                s.Stop = s.Stop.HasValue ? s.Stop.Value : shape[0];
                var nd = new NDArray(Array, new int[] { s.Length.Value }.Concat(Shape.GetShape(shape, 0)).ToArray());
                nd.Storage.Slice = s;
                return nd;
            }

            set
            {
                throw new NotImplementedException("slice data set is not implemented.");
            }
        }

        private NDArray setValue<T>(NDArray indexes)
        {
            Shape newShape = new int[] { indexes.size }.Concat(shape.Skip(1)).ToArray();
            var buf = Data<T>();
            var idx = indexes.Data<int>();
            var array = new T[newShape.Size];

            var indice = Shape.GetShape(newShape.Dimensions, axis: 0);
            var length = Shape.GetSize(indice);

            for (var row = 0; row < newShape[0]; row++)
            {
                var d = buf.AsSpan(idx[row] * length, length);
                d.CopyTo(array.AsSpan(row * length));
            }

            var nd = new NDArray(array, newShape);
            return nd;
        }

        public NDArray this[NDArray<bool> booleanArray]
        {
            get
            {
                if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
                {
                    throw new IncorrectShapeException();
                }

                var boolDotNetArray = booleanArray.Data<bool>();

                switch (dtype.Name)
                {
                    case "Int32":
                        {
                            var nd = new List<int>();

                            for (int idx = 0; idx < boolDotNetArray.Length; idx++)
                            {
                                if (boolDotNetArray[idx])
                                {
                                    nd.Add(Data<int>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
                                }
                            }

                            return new NDArray(nd.ToArray(), nd.Count);
                        }
                    case "Double":
                        {
                            var nd = new List<double>();

                            for (int idx = 0; idx < boolDotNetArray.Length; idx++)
                            {
                                if (boolDotNetArray[idx])
                                {
                                    nd.Add(Data<double>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
                                }
                            }

                            return new NDArray(nd.ToArray(), nd.Count);
                        }
                }

                throw new NotImplementedException("");

            }
            set
            {
                if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
                {
                    throw new IncorrectShapeException();
                }

                object scalarObj = value.Storage.GetData().GetValue(0);

                bool[] boolDotNetArray = booleanArray.Storage.GetData() as bool[];

                int elementsAmount = booleanArray.size;

                for (int idx = 0; idx < elementsAmount; idx++)
                {
                    if (boolDotNetArray[idx])
                    {
                        int[] indexes = booleanArray.Storage.Shape.GetDimIndexOutShape(idx);
                        Array.SetValue(scalarObj, Storage.Shape.GetIndexInShape(indexes));
                    }
                }

            }
        }

        /// <summary>
        /// Get single value from internal storage and do not cast dtype
        /// </summary>
        /// <param name="indice">indexes</param>
        /// <returns>element from internal storage</returns>
        private NDArray GetData(params int[] indice)
        {
            int offset = 0;

            Shape s1 = shape.Skip(indice.Length).ToArray();
            int idx = s1.GetIndexInShape(indice);

            int stride = Shape.GetSize(s1.Dimensions);

            if (ndim == 1)
                offset = idx + (slice is null ? 0 : slice.Start.Value);
            else
                offset = idx + (slice is null ? 0 : slice.Start.Value) * stride;

            switch (Type.GetTypeCode(dtype))
            {
                case TypeCode.Boolean:
                    return GetBoolean(indice);
                case TypeCode.Int16:
                    return GetInt16(indice);
                case TypeCode.Int32:
                    return GetInt32(indice);
                case TypeCode.Int64:
                    return GetInt64(indice);
                case TypeCode.Single:
                    return GetSingle(indice);
                case TypeCode.Double:
                    return GetDouble(indice);
                case TypeCode.Decimal:
                    return GetDecimal(indice);
                case TypeCode.String:
                    return GetString(indice);
                default:
                    return GetNDArray(indice);
            }

            /*if (indexes.Length == ndim ||
                shape.Last() == 1)
            {
                switch (Type.GetTypeCode(dtype))
                {
                    case TypeCode.Boolean:
                        return GetBoolean(indexes);
                    case TypeCode.Int16:
                        return GetInt16(indexes);
                    case TypeCode.Int32:
                        return GetInt32(indexes);
                    case TypeCode.Int64:
                        return GetInt64(indexes);
                    case TypeCode.Single:
                        return GetSingle(indexes);
                    case TypeCode.Double:
                        return GetDouble(indexes);
                    case TypeCode.Decimal:
                        return GetDecimal(indexes);
                    case TypeCode.String:
                        return GetString(indexes);
                    default:
                        return GetNDArray(indexes);
                }
            }
            else if (indexes.Length == ndim - 1)
            {
                var offset = new int[ndim];
                for (int i = 0; i < ndim - 1; i++)
                    offset[i] = indexes[i];

                var nd = new NDArray(dtype, shape[ndim - 1]);
                var data = GetData();
                for (int i = 0; i < shape[ndim - 1]; i++)
                {
                    offset[offset.Length - 1] = i;
                    //nd.SetData(data.GetValue(Shape.GetIndexInShape(offset)), i);
                }

                return nd;
            }
            // 3 Dim
            else if (indexes.Length == ndim - 2)
            {
                var offset = new int[ndim];
                var nd = new NDArray(dtype, new int[] { shape[ndim - 2], shape[ndim - 1] });
                var data = GetData();
                for (int i = 0; i < shape[ndim - 2]; i++)
                {
                    for (int j = 0; j < shape[ndim - 1]; j++)
                    {
                        offset[0] = 0;
                        offset[1] = i;
                        offset[2] = j;
                        //nd.SetData(data.GetValue(Shape.GetIndexInShape(offset)), i, j);
                    }
                }

                return nd;
            }*/

            throw new Exception("NDStorage.GetData");
        }
    }
}
