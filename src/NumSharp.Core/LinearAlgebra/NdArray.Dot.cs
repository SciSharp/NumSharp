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
        /// <summary>
        /// 
        /// </summary>
        /// <param name="np1"></param>
        /// <param name="np2"></param>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public NDArrayGeneric<T> dot(NDArrayGeneric<T> np2)
        {
            if ((this.Shape.Length == 1 ) & (np2.Shape.Length == 1))
                if (this.Shape.Shapes[0] != np2.Shape.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented."); 
                else 
                {
                    np2.Shape = new Shape(np2.Data.Length,1);
                    this.Shape = new Shape(1,this.Data.Length);
                }
            else
                if (this.Shape.Shapes[1] != np2.Shape.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented.");
            
            int iterator = this.Shape.Shapes[1];
            int dim0 = this.Shape.Shapes[0];
            int dim1 = np2.Shape.Shapes[1];
            
            NDArrayGeneric<T> prod = new NDArrayGeneric<T>();
            prod.Shape = new Shape(dim0,dim1);
            prod.Data = new T[prod.Shape.Size];

            switch (this.Data)
            {
                case double[] np1Array : 
                {
                    double[] result = prod.Data as double[];
                    double[] np2Array = np2.Data as double[];
                    
                    for (int idx = 0; idx < prod.Data.Length;idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator;kdx++)
                            result[idx] += np1Array[puffer3 + kdx] * np2Array[ dim1 * kdx + puffer2];    
                    }
                    break;
                }
                case float[] np1Array : 
                {
                    float[] result = prod.Data as float[];
                    float[] np2Array = np2.Data as float[];

                    for (int idx = 0; idx < prod.Data.Length;idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator;kdx++)
                            result[idx] += np1Array[puffer3 + kdx] * np2Array[ dim1 * kdx + puffer2];    
                    }

                    break;
                }
                case Complex[] np1Array : 
                {
                    Complex[] result = prod.Data as Complex[];
                    Complex[] np2Array = np2.Data as Complex[];

                    for (int idx = 0; idx < prod.Data.Length;idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator;kdx++)
                            result[idx] += np1Array[puffer3 + kdx] * np2Array[ dim1 * kdx + puffer2];    
                    }
                    break;
                }
                case Quaternion[] np1Array : 
                {
                    Quaternion[] result = prod.Data as Quaternion[];
                    Quaternion[] np2Array = np2.Data as Quaternion[];

                    for (int idx = 0; idx < prod.Data.Length;idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator;kdx++)
                            result[idx] += np1Array[puffer3 + kdx] * np2Array[ dim1 * kdx + puffer2];    
                    }
                    break;
                }
                default : 
                {
                    throw new Exception("The Dot method is not implemented for the "  + typeof(T).Name);
                }
            }
            if ((this.Shape.Length == 1 ) & (np2.Shape.Length == 1))
            {
                this.Shape = new Shape(this.Data.Length);
                np2.Shape = new Shape(np2.Data.Length);
                prod.Shape = new Shape(1);
            }

            return prod;
        }
    }
}
