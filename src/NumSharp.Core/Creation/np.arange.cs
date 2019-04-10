using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray arange(float stop)
        {
            return arange(0, stop, 1);
        }

        public static NDArray arange(double stop)
        {
            return arange(0, stop, 1);
        }

        public static NDArray arange(float start, float stop, float step = 1)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            int length = (int)Math.Ceiling((stop - start + 0.0) / step);

            var nd = new NDArray(typeof(float), new Shape(length));

            float[] puffer = nd.Array as float[];

            for (int index = 0; index < length; index++)
            {
                float value = start + index * step;
                puffer[index] = value;
            }

            return nd;
        }

        public static NDArray arange(double start, double stop, double step = 1)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            int length = (int)Math.Ceiling((stop - start + 0.0) / step);

            var nd = new NDArray(typeof(double), new Shape(length));

            double[] puffer = nd.Array as double[];

            for (int index = 0; index < length; index++)
            {
                double value = start + index * step;
                puffer[index] = value;
            }

            return nd;
        }

        public static NDArray arange(int stop)
        {
            return arange(0, stop, 1);
        }

        public static NDArray arange(int start, int stop, int step = 1)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            int length = (int)Math.Ceiling((stop - start + 0.0) / step);
            int index = 0;

            var nd = new NDArray(np.int32, new Shape(length));
            
            var a = new int[length];
            for (int i = start; i < stop; i += step)
                a[index++] = i;
            nd.SetData(a);

            return nd;
        }
    }
}
