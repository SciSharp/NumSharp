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
        public NDArray arange(int stop, Type dtype = null)
        {
            return arange(0, stop, 1, dtype);
        }

        public NDArray arange(int start, int stop, int step = 1, Type dtype = null)
        {
            if (start > stop)
            {
                throw new Exception("parameters invalid, start is greater than stop.");
            }

            var list = new int[(int)Math.Ceiling((stop - start + 0.0) / step)];
            int index = 0;

            if (dtype == null)
            {
                dtype = typeof(int);
            }

            var nd = new NDArray(dtype)
            {
                Shape = new Shape(list.Length)
            };

            nd.Storage.Allocate(list.Length);

            switch (dtype.Name)
            {
                case "Int32":
                    for (int i = start; i < stop; i += step)
                        nd.Storage.Int32[index++] = i;
                    break;

                case "Double":
                    for (int i = start; i < stop; i += step)
                        nd.Storage.Double8[index++] = i;
                    break;

                /*    case double[] dataArray : 
                    {
                        for(int idx = 0; idx < dataArray.Length;idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                    case float[] dataArray : 
                    {
                        for(int idx = 0; idx < dataArray.Length;idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                    case Complex[] dataArray : 
                    {
                        // no performance critial operation
                        dataArray = list.Select(x => (Complex) x ).ToArray();
                        break;
                    }
                    case Quaternion[] dataArray : 
                    {
                        // no performance critial operation
                        dataArray = list.Select(x => new Quaternion(new Vector3(0,0,0),x) ).ToArray();
                        break;
                    }
                    default : 
                    {
                        throw new Exception("This method was not yet implemented for this type" + typeof(T).Name);
                    }*/
            }

            return nd;
        }
    }
}
