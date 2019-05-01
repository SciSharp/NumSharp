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
                    case TypeCode.String:
                        nd = setValue<string>(indices);
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
            get { return this[Slice.ParseSlices(slice)]; }

            set
            {
                throw new NotImplementedException("slice data set is not implemented.");
            }
        }

        public NDArray this[params Slice[] slices]
        {
            get
            {
                if (slices.Length == 0)
                    throw new ArgumentException("At least one slice definition expected");
                if (slices.Length == 1)
                {
                    var s = slices[0];
                    s.Start = Math.Max(0, s.Start.HasValue ? s.Start.Value : 0);
                    s.Stop = Math.Min(s.Stop.HasValue ? s.Stop.Value : shape[0], shape[0]);
                    var nd = new NDArray(Array, new int[] { s.Length.Value }.Concat(Shape.GetShape(shape, 0)).ToArray());
                    nd.Storage.Slice = s;
                    return nd;
                }
                throw new NotImplementedException("Multidimensional slicing not implemented yet");
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
        /// Get n-th dimension data
        /// </summary>
        /// <param name="indices">indexes</param>
        /// <returns>NDArray</returns>
        private NDArray GetData(params int[] indices)
        {
            Shape s1 = shape.Skip(indices.Length).ToArray();
            var nd = new NDArray(dtype, s1);
            //nd.Storage.Slice = new Slice($"{}");
            switch (Type.GetTypeCode(dtype))
            {
                case TypeCode.Boolean:
                    nd.Array = Storage.GetSpanData<bool>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Int16:
                    nd.Array = Storage.GetSpanData<short>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Int32:
                    nd.Array = Storage.GetSpanData<int>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Int64:
                    nd.Array = Storage.GetSpanData<long>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Single:
                    nd.Array = Storage.GetSpanData<float>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Double:
                    nd.Array = Storage.GetSpanData<double>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.Decimal:
                    nd.Array = Storage.GetSpanData<decimal>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                case TypeCode.String:
                    nd.Array = Storage.GetSpanData<string>(indices).ToArray().Step(slice is null ? 1 : slice.Step);
                    break;
                default:
                    return Storage.GetSpanData<NDArray>(indices).ToArray()[0]; // todo: how to step this??
            }

            return nd;
        }


    }
}
