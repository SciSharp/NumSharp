using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public class NDArrayOptimized<T>
    {
        public T[] Data { get; set; }

        public IList<int> Shape { get; set; }

        public int NDim { get { return Shape.Count; } }

        public int Size { get { return Data.Length; } }

        public int Length
        {
            get
            {
                int size = 0;
                for (int d = 0; d < NDim; d++)
                {
                    if (size == 0) size = 1;
                    // size *= DimensionSize(d + 1);
                }

                return size;
            }
        }

        public NDArrayOptimized()
        {
            Shape = new List<int>();
        }

        public T this[params int[] select]
        {
            get
            {
                // 2 dim
                /*if(select.Length == 2)
                {
                    int i = 0;
                    int idx = Shape[Shape.Count - 1 - i] * select[i] + select[select.Length - 1];

                    return Data[idx];
                }*/

                int idx = 0;
                // n dim
                for (int i = 0; i < select.Length - 1; i++)
                {
                    idx += Shape[Shape.Count - 1 - i] * select[i];
                }
                idx += select[select.Length - 1];

                return Data[idx];
            }
        }

        public NDArrayOptimized<T> arange(int stop, int start = 0, int step = 1)
        {
            Shape = new List<int>(stop);

            int index = 0;

            Data = Enumerable.Range(start, stop - start)
                                .Where(x => index++ % step == 0)
                                .Select(x => (T)TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(x.ToString()))
                                .ToArray();
            return this;
        }

        public NDArrayOptimized<T> reshape(params int[] n)
        {
            Shape.Clear();
            Shape = n;

            return this;
        }

        public override string ToString()
        {
            string output = "array([";

            // loop
            for (int r = 0; r < Data.Length; r++)
            {
                output += (r == 0) ? Data[r] + "" : ", " + Data[r];
            }

            output += "])";

            return output;
        }
    }
}
