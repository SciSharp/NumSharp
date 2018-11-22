using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray power<T>(NDArray nd, T exponent)
        {
            var sinArray = new NDArray(nd.dtype)
            {
                Shape = new Shape(nd.Shape.Shapes)
            };

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
