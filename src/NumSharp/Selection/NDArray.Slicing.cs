using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<T>
    {
        public NDArray<T>[] this[Slice select]
        {
            get
            {
                var list = new List<NDArray<T>>();

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

                for (int s = select.Start; s< select.Stop; s+= select.Step)
                {
                    var n = new NDArray<T>();
                    Span<T> data = Data;
                    n.Data = data.Slice(s, select.Step).ToArray();
                    n.Shape = new Shape(shape);

                    list.Add(n);
                }

                return list.ToArray();
            }
        }
    }
}
