using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray power(NDArray nd, ValueType exponent)
        {
            return nd.power(exponent);
        }
        public static NDArray power<T>(NDArray nd, T exponent)
        {
            var sinArray = new NDArray(nd.dtype);
            sinArray.Storage.Reshape(nd.shape);

            switch (sinArray[0])
            {
                case double[] sinData:
                    {
                        for (int idx = 0; idx < sinData.Length; idx++)
                        {
                            sinArray[idx] = Math.Pow(sinData[idx], (double)(object)exponent);
                        }
                        break;
                    }
                default:
                    {
                        throw new Exception("The operation is not implemented for the");
                    }

            }
            return sinArray;
        }
    }
}
