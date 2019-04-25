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
                return Storage.GetData(select);
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
                var indices = np.arange(s.Start.Value, s.Stop.Value);
                var nd = this[indices];
                nd.slice = s;
                return nd;
            }

            set
            {
                
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
    }
}
