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
        public NDArray<T> sin()
        {
            NDArray<T> sinArray = new NDArray<T>();
            sinArray.Data = new T[this.Data.Length];
            sinArray.Shape = new Shape(this.Shape.Shapes);

            switch (sinArray.Data)
            {
                case double[] sinData : 
                {
                    double[] npData = this.Data as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        sinData[idx] = Math.Sin(npData[idx]);
                    }
                    break;
                }
                case float[] sinData : 
                {
                    double[] npData = this.Data as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        // boxing necessary because Math.Sin just for double
                        sinData[idx] = (float) Math.Sin(npData[idx]);
                    }
                    break;
                }
                case Complex[] sinData : 
                {
                    Complex[] npData = this.Data as Complex[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        // boxing necessary because Math.Sin just for double
                        sinData[idx] = Complex.Sin(  npData[idx]);
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
