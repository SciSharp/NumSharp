using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<T>
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

        public NDArray<T> this[Shape select]
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

                    var n = new NDArray<T>();
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
        public NDArray<T> this[IList<int> select]
        {
            get
            {
                var n = new NDArray<T>();
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
        public NDArray<T> this[NDArray<int> select] => this[select.Data.ToList()];

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
