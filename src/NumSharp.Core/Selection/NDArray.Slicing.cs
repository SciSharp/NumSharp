using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArrayGeneric<T>
    {
        public NDArrayGeneric<NDArrayGeneric<T>> this[Slice select]
        {
            get
            {
                var result = new NDArrayGeneric<NDArrayGeneric<T>>();
                result.Data = new NDArrayGeneric<T>[select.Length];
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
                var list = new NDArrayGeneric<T>();
                for (int s = select.Start; s< select.Stop; s+= select.Step)
                {
                    var n = new NDArrayGeneric<T>();
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
