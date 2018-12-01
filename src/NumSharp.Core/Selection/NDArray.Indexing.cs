using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        /// <summary>
        /// Retrieve element
        /// low performance, use generic Data<T> method for performance sensitive invoke
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public object this[params int[] select]
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

        /// <summary>
        /// Filter specific elements through select.
        /// </summary>
        /// <param name="select"></param>
        /// <returns>Return a new NDArray with filterd elements.</returns>
        public NDArray this[IList<int> select]
        {
            get
            {
                var n = new NDArray(dtype);
                if (NDim == 1)
                {
                    n.Storage.Shape = new Shape(select.Count);
                    for (int i = 0; i < select.Count; i++)
                    {
                        n[i] = this[select[i]];
                    }
                }
                else if (NDim == 2)
                {
                    n.Storage.Shape = new Shape(select.Count, Shape[1]);
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

        public NDArray this[NDArray select] => this[select.Data<int>().ToList()];

        public T[] Data<T>() => Storage.Data<T>();

        public T Data<T>(params int[] shape) => Storage.Data<T>()[Shape.GetIndexInShape(shape)];

        /// <summary>
        /// shortcut for Double data type, 8 bytes
        /// </summary>
        public double[] Double => Storage.Data<double>();

        /// <summary>
        /// shortcut for Int32 data type
        /// </summary>
        public int[] Int32 => Storage.Data<int>();

        // <summary>
        /// shortcut for string data type
        /// </summary>
        public string[] Chars => Storage.Data<string>();

        public void Set<T>(T[] data) => Storage.Set(data);

        public void Set<T>(Shape shape, T value) => Storage.Set(shape, value);
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
                if (select.NDim == NDim)
                {
                    throw new Exception("Please use NDArray[m, n] to access element.");
                }
                else
                {
                    int start = GetIndexInShape(select.Shapes.ToArray());
                    int length = Shape.DimOffset[select.NDim - 1];

                    var n = new NDArrayGeneric<T>();
                    Span<T> data = Data;
                    n.Data = data.Slice(start, length).ToArray();
                    int[] shape = new int[Shape.NDim - select.NDim];
                    for (int i = select.NDim; i < Shape.NDim; i++)
                    {
                        shape[i - select.NDim] = Shape[i];
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
