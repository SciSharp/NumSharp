using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NumSharp.Generic;
using NumSharp.Utilities;

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

        /// <summary>
        ///     Used to perform filtering by whats true and whats false.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public NDArray this[NDArray indices]
        {
            get
            {
                return null;

//                NDArray nd = null;
//                switch (this.Storage.TypeCode)
//                {
//#if _REGEN
//	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
//	                case NPTypeCode.#1:
//	                {

//	                }
//	                %
//	                default:
//		                throw new NotSupportedException();
//#else
//#endif
//                }

//                switch (this.Storage.TypeCode)
//                {
//                    case TypeCode.Byte:
//                        nd = setValue<byte>(indices);
//                        break;
//                    case TypeCode.Int32:
//                        nd = setValue<int>(indices);
//                        break;
//                    case TypeCode.Int64:
//                        nd = setValue<long>(indices);
//                        break;
//                    case TypeCode.Single:
//                        nd = setValue<float>(indices);
//                        break;
//                    case TypeCode.Double:
//                        nd = setValue<double>(indices);
//                        break;
//                    case TypeCode.Decimal:
//                        nd = setValue<decimal>(indices);
//                        break;
//                    case TypeCode.String:
//                        nd = setValue<string>(indices);
//                        break;
//                }

//                return nd;
//                NDArray setValue<T>(NDArray indexes)
//                {
//                    Shape newShape = new int[] { indexes.size }.Concat(shape.Skip(1)).ToArray();
//                    var buf = Data<T>();
//                    var idx = indexes.Data<int>();
//                    var array = new T[newShape.Size];

//                    var indice = Shape.GetShape(newShape.Dimensions, axis: 0);
//                    var length = Shape.GetSize(indice);

//                    for (var row = 0; row < newShape[0]; row++)
//                    {
//                        var d = buf.AsSpan(idx[row] * length, length);
//                        d.CopyTo(array.AsSpan(row * length));
//                    }

//                    var nd = new NDArray(array, newShape);
//                    return nd;
//                }
            }
        }

        public NDArray this[string slice]
        {
            get
            {
                return this[Slice.ParseSlices(slice)];
            }

            set
            {
                throw new NotImplementedException("slice data set is not implemented for types less NDArray's.");
            }
        }

        public NDArray this[params Slice[] slices]
        {
            get
            {
                if (slices.Length == 0)
                    throw new ArgumentException("At least one slice definition expected");
                //if (slices.Length == 1)
                //{
                //    var s = slices[0];
                //    if (s.Step == 1)
                //    {
                //        if (s.IsIndex)
                //            return GetData(s.Start ?? 0);
                //        s.Start = (slice is null ? 0 : slice.Start) + Math.Max(0, s.Start.HasValue ? s.Start.Value : 0);
                //        s.Stop = (slice is null ? 0 : slice.Start) +
                //                 Math.Min(s.Stop.HasValue ? s.Stop.Value : shape[0], shape[0]);
                //        var new_shape= new int[] {s.Length.Value}.Concat(Shape.GetShape(shape, 0)).ToArray();
                //        var nd = new NDArray(Array, new_shape);
                //        nd.Storage.Slice = s;
                //        return nd;
                //    }
                //}
                return null;
                //TODO! return new NDArray(new ViewStorage(Storage, slices));
            }
            set
            {
                throw new NotImplementedException("slice data set is not implemented.");
            }
        }
        //TODO! What is the use of this? Makes no sense
        //public NDArray this[NDArray<bool> booleanArray]
        //{
        //    get
        //    {
        //        if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
        //        {
        //            throw new IncorrectShapeException();
        //        }

        //        var boolDotNetArray = booleanArray.Data<bool>();

        //        switch (dtype.Name)
        //        {
        //            case "Int32":
        //                {
        //                    var nd = new List<int>();

        //                    for (int idx = 0; idx < boolDotNetArray.Length; idx++)
        //                    {
        //                        if (boolDotNetArray[idx])
        //                        {
        //                            nd.Add(Data<int>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
        //                        }
        //                    }

        //                    return new NDArray(nd.ToArray(), nd.Count);
        //                }
        //            case "Double":
        //                {
        //                    var nd = new List<double>();

        //                    for (int idx = 0; idx < boolDotNetArray.Length; idx++)
        //                    {
        //                        if (boolDotNetArray[idx])
        //                        {
        //                            nd.Add(Data<double>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
        //                        }
        //                    }

        //                    return new NDArray(nd.ToArray(), nd.Count);
        //                }
        //        }

        //        throw new NotImplementedException("");

        //    }
        //    set
        //    {
        //        if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
        //        {
        //            throw new IncorrectShapeException();
        //        }

        //        object scalarObj = value.Storage.GetData().GetValue(0);

        //        bool[] boolDotNetArray = booleanArray.Storage.GetData() as bool[];

        //        int elementsAmount = booleanArray.size;

        //        for (int idx = 0; idx < elementsAmount; idx++)
        //        {
        //            if (boolDotNetArray[idx])
        //            {
        //                int[] indexes = booleanArray.Storage.Shape.GetDimIndexOutShape(idx);
        //                Array.SetValue(scalarObj, Storage.Shape.GetIndexInShape(slice, indexes));
        //            }
        //        }

        //    }
        //}

        /// <summary>
        /// Get n-th dimension data
        /// </summary>
        /// <param name="indices">indexes</param>
        /// <returns>NDArray</returns>
        private NDArray GetData(params int[] indices)
        {
            return null;
            //if (indices.Length == 0)
            //    return this;
            //if (Storage.SupportsSpan)
            //{
            //    Shape s1 = shape.Skip(indices.Length).ToArray();
            //    var nd = new NDArray(dtype, s1);
            //    //nd.Storage.Slice = new Slice($"{}");
            //    switch (Type.GetTypeCode(dtype))
            //    {
            //        case TypeCode.Boolean:
            //            nd.Array = Storage.GetSpanData<bool>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Byte:
            //            nd.Array = Storage.GetSpanData<byte>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Int16:
            //            nd.Array = Storage.GetSpanData<short>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Int32:
            //            nd.Array = Storage.GetSpanData<int>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Int64:
            //            nd.Array = Storage.GetSpanData<long>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Single:
            //            nd.Array = Storage.GetSpanData<float>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Double:
            //            nd.Array = Storage.GetSpanData<double>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.Decimal:
            //            nd.Array = Storage.GetSpanData<decimal>(slice, indices).ToArray();
            //            break;
            //        case TypeCode.String:
            //            nd.Array = Storage.GetSpanData<string>(slice, indices).ToArray();
            //            break;
            //        default:
            //            return Storage.GetSpanData<NDArray>(slice, indices).ToArray()[0];
            //    }
            //    return nd;
            //}

            //if (indices.Length < Storage.Shape.NDim)
            //{
            //    // a slice was requested
            //    return this[indices.Select(i => Slice.Index(i)).ToArray()];
            //}
            //else if (indices.Length == Storage.Shape.NDim)
            //{
            //    // a scalar was indexed
            //    var nd = new NDArray(this.dtype, new Shape());
            //    switch (Type.GetTypeCode(dtype))
            //    {
            //        case TypeCode.Boolean:
            //            nd.Array = new []{ Storage.GetData<bool>( indices)};
            //            break;
            //        case TypeCode.Byte:
            //            nd.Array = new[] { Storage.GetData<byte>(indices) };
            //            break;
            //        case TypeCode.Int16:
            //            nd.Array = new[] { Storage.GetData<short>(indices) };
            //            break;
            //        case TypeCode.Int32:
            //            nd.Array = new[] { Storage.GetData<int>(indices) };
            //            break;
            //        case TypeCode.Int64:
            //            nd.Array = new[] { Storage.GetData<long>(indices) };
            //            break;
            //        case TypeCode.Single:
            //            nd.Array = new[] { Storage.GetData<float>(indices) };
            //            break;
            //        case TypeCode.Double:
            //            nd.Array = new[] { Storage.GetData<double>(indices) };
            //            break;
            //        case TypeCode.Decimal:
            //            nd.Array = new[] { Storage.GetData<decimal>(indices) };
            //            break;
            //        case TypeCode.String:
            //            nd.Array = new[] { Storage.GetData<string>(indices) };
            //            break;
            //        default:
            //            return Storage.GetData<NDArray>(indices);
            //    }
            //    return nd;
            //}
            //else
            //{
            //    throw new ArgumentException("Too many index dimensions for shape " + Storage.Shape);
            //}
        }
    }
}
