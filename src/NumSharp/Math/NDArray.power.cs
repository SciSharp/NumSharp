using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray<T> 
    {
        public NDArray<T> power(T exponent)
        {
            NDArray<T> sinArray = new NDArray<T>();
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
