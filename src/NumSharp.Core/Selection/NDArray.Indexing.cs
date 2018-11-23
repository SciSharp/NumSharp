using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public object this[params int[] select]
        {
            get
            {
                if (select.Length == NDim)
                {
                    return Storage[Shape.GetIndexInShape(select)];
                }
                else
                {
                    int start = Shape.GetIndexInShape(select);
                    int length = Shape.DimOffset[select.Length - 1];

                    var n = new NDArray(dtype);

                    switch (Storage.Data())
                    {
                        case double[] values:
                            Span<double> double8 = Storage.Data<double>();
                            n.Storage.Set(double8.Slice(start, length).ToArray());
                            break;
                        case int[] values:
                            Span<int> int32 = Storage.Data<int>();
                            n.Storage.Set(int32.Slice(start, length).ToArray());
                            break;
                    }

                    int[] shape = new int[Shape.Length - select.Length];
                    for (int i = select.Length; i < Shape.Length; i++)
                    {
                        shape[i - select.Length] = Shape[i];
                    }
                    n.Shape = new Shape(shape);

                    return n;
                }
            }

            set
            {
                if (select.Length == NDim)
                {
                    Storage[Shape.GetIndexInShape(select)] = value;
                }
                else
                {

                }
            }
        }
    }

    public partial class NDArrayGeneric<T>
    {
        /// <summary>
        /// Index accessor
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public T this[params int[] select]
        {
            get
            {
                return Data[GetIndexInShape(select)];
            }

            set
            {
                Data[GetIndexInShape(select)] = value;
            }
        }

        public NDArrayGeneric<T> this[Shape select]
        {
            get
            {
                if (select.Length == NDim)
                {
                    throw new Exception("Please use NDArray[m, n] to access element.");
                }
                else
                {
                    int start = GetIndexInShape(select.Shapes.ToArray());
                    int length = Shape.DimOffset[select.Length - 1];

                    var n = new NDArrayGeneric<T>();
                    Span<T> data = Data;
                    n.Data = data.Slice(start, length).ToArray();
                    int[] shape = new int[Shape.Length - select.Length];
                    for (int i = select.Length; i < Shape.Length; i++)
                    {
                        shape[i - select.Length] = Shape[i];
                    }
                    n.Shape = new Shape(shape);
                    // n.Shape = new Shape(Shape.Shapes.ToArray().AsSpan().Slice(select.Length).ToArray());
                    return n;
                }
            }
        }

        /// <summary>
        /// Filter specific elements through select.
        /// </summary>
        /// <param name="select"></param>
        /// <returns>Return a new NDArray with filterd elements.</returns>
        public NDArrayGeneric<T> this[IList<int> select]
        {
            get
            {
                var n = new NDArrayGeneric<T>();
                if (NDim == 1)
                {
                    n.Data = new T[select.Count];
                    n.Shape = new Shape(select.Count);
                    for (int i = 0; i < select.Count; i++)
                    {
                        n[i] = this[select[i]];
                    }
                }
                else if (NDim == 2)
                {
                    n.Data = new T[select.Count * Shape[1]];
                    n.Shape = new Shape(select.Count, Shape[1]);
                    for (int i = 0; i < select.Count; i++)
                    {
                        for (int j = 0; j < Shape[1]; j++)
                        {
                            n[i, j] = this[select[i], j];
                        }
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }

                return n;
            }
        }

        /// <summary>
        /// Overload
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public NDArrayGeneric<T> this[NDArrayGeneric<int> select] => this[select.Data.ToList()];

        private int GetIndexInShape(params int[] select)
        {
            int idx = 0;
            for (int i = 0; i < select.Length; i++)
            {
                idx += Shape.DimOffset[i] * select[i];
            }

            return idx;
        }
    }
}
