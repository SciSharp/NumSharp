using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArrayGeneric<T> 
    {
        public NDArrayGeneric<T> power(T exponent)
        {
            NDArrayGeneric<T> sinArray = new NDArrayGeneric<T>();
            sinArray.Data = new T[this.Size];
            sinArray.Shape = new Shape(this.Shape.Shapes);

            switch (Data)
            {
                case double[] sinData : 
                {
                    for (int idx = 0; idx < sinData.Length;idx++)
                    {
                            sinArray[idx] = (T)(object)Math.Pow(sinData[idx], (double)(object)exponent);
                    }
                    break;
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the");
                }
                
            }
            return sinArray;
        }
    }
}
