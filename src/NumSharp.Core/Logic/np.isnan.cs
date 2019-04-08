using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        /// Test element-wise for Not a Number.
        /// </summary>
        /// <param name="a"></param>
        /// <returns>The result is returned as a boolean array.</returns>
        public static NDArray<bool> isnan(NDArray a)
        {
            NDArray<bool> result = new NDArray<bool>(a.shape);
            Array data = a.Array;
            var res = result.Array as bool[];

            switch (data)
            {
                case double[] arr:
                    {
                        for (int i = 0; i < arr.Length; i++)
                            res[i] = double.IsNaN(arr[i]);
                        break;
                    }
                case float[] arr:
                    {
                        for (int i = 0; i < arr.Length; i++)
                            res[i] = float.IsNaN(arr[i]);
                        break;
                    }
                case int[] arr:
                    {
                        //for (int i = 0; i < data.Length; i++)
                        //    res[i] = false;
                        break;
                    }
                case Int64[] arr:
                    {
                        //for (int i = 0; i < data.Length; i++)
                        //    res[i] = false;
                        break;
                    }
                case Complex[] arr:
                    {
                        throw new NotImplementedException("Checking Complex array for NaN is not implemented yet.");
                        break;
                    }
                default:
                    {
                        throw new IncorrectTypeException();
                    }
            }
            return result;
        }
    }
}
