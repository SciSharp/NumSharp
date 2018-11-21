using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public class NumSharp
    {
        public static Type int16 = typeof(short);
        public static Type double8 = typeof(double);
        public static Type decimal16 = typeof(decimal);

        public NDArrayWithDType arange(int stop, Type dtype = null)
        {
            if(dtype == null)
            {
                dtype = NumSharp.int16;
            }

            return arange(0, stop, 1, dtype);
        }

        public NDArrayWithDType arange(int start, int stop, int step = 1, Type dtype = null)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid");
            }

            switch (dtype.Name)
            {
                case "Int32":
                    {
                        var n = new NDArrayWithDType(NumSharp.int16);
                        n.arange(stop, start, step);
                        return n;
                    }

                case "Double":
                    {
                        var n = new NDArrayWithDType(NumSharp.double8);
                        n.arange(stop, start, step);
                        return n;
                    }
                default:
                    throw new NotImplementedException();
            }
        }

        public NDArrayWithDType reshape(NDArrayWithDType nd, params int[] shape)
        {
            nd.Shape = new Shape(shape);

            return nd;
        }
    }
}
