using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray arange(double stop)
        {
            return arange(0, stop, 1);
        }

        public NDArray arange(double start, double stop, double step = 1)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            int length = (int)Math.Ceiling((stop - start + 0.0) / step);
            int index = 0;

            var nd = new NDArray(double8)
            {
                Shape = new Shape(length)
            };

            for (double i = start; i < stop; i += step)
                nd.Data<double>()[index++] = i;

            return nd;
        }

        public NDArray arange(int stop)
        {
            return arange(0, stop, 1);
        }

        public NDArray arange(int start, int stop, int step = 1)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            int length = (int)Math.Ceiling((stop - start + 0.0) / step);
            int index = 0;

            var nd = new NDArray(int32)
            {
                Shape = new Shape(length)
            };

            if(nd.Data<int>() is int[] a)
            {
                for (int i = start; i < stop; i += step)
                    a[index++] = i;
            }
            
            return nd;
        }
    }
}
