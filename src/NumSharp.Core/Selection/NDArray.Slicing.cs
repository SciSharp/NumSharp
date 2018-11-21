using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray<T>
    {
        public NDArray<NDArray<T>> this[Slice select]
        {
            get
            {
                var result = new NDArray<NDArray<T>>();
                result.Data = new NDArray<T>[select.Length];
                result.shape = new Shape(select.Length);

                int[] shape = new int[Shape.Length];
                for (int i = 0; i < Shape.Length; i++)
                {
                    if (i == 0)
                    {
                        shape[i] = select.Step;
                    }
                    else
                    {
                        shape[i] = Shape[i];
                    }
                }

                int index = 0;
                var list = new NDArray<T>();
                for (int s = select.Start; s< select.Stop; s+= select.Step)
                {
                    var n = new NDArray<T>();
                    Span<T> data = Data;
                    n.Data = data.Slice(s, select.Step).ToArray();
                    n.Shape = new Shape(shape);

                    result.Data[index] = n;
                    index++;
                }

                return result;
            }
        }
    }
}
