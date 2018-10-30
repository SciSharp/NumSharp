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

        public NDArrayOptimized<T> arange(int stop, int start = 0, int step = 1)
        {
            Shape.Clear();
            Shape.Add(stop);

            int index = 0;

            Data = Enumerable.Range(start, stop - start)
                                .Where(x => index++ % step == 0)
                                .Select(x => (T)TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(x.ToString()))
                                .ToArray();
            return this;
        }
    }
}
