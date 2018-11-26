using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="np1"></param>
        /// <param name="nd2"></param>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public NDArray dot<T>(NDArray nd2)
        {
            if(NDim == 2 & nd2.NDim == 1)
            {
                var nd = new NDArray(typeof(T), Shape[0]);

                for(int i = 0; i < nd.Size; i++)
                {
                    for(int j = 0; j < nd2.Size; j++)
                    {
                        switch (dtype.Name)
                        {
                            case "Int32":
                                nd[i] = nd.Data<int>(i) + Data<int>(i, j) * nd2.Int32[j];
                                break;
                        }
                        
                    }
                }

                return nd;
            } 

            if ((this.Shape.NDim == 1) & (nd2.Shape.NDim == 1))
                if (this.Shape.Shapes[0] != nd2.Shape.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented.");
                else
                {
                    nd2.Shape = new Shape(nd2.Size, 1);
                    this.Shape = new Shape(1, Size);
                }
            else
                if (this.Shape.Shapes[1] != nd2.Shape.Shapes[0])
                throw new Exception("The Dot method does not work with this shape or was not already implemented.");

            int iterator = this.Shape.Shapes[1];
            int dim0 = this.Shape.Shapes[0];
            int dim1 = nd2.Shape.Shapes[1];

            var prod = new NDArray(typeof(T), dim0, dim1);

            switch (Data<T>())
            {
                case double[] np1Array:
                    {
                        double[] result = prod.Data<double>();
                        double[] np2Array = nd2.Data<double>();

                        for (int idx = 0; idx < prod.Size; idx++)
                        {
                            int puffer1 = idx / dim1;
                            int puffer2 = idx % dim1;
                            int puffer3 = puffer1 * iterator;
                            for (int kdx = 0; kdx < iterator; kdx++)
                                result[idx] += np1Array[puffer3 + kdx] * np2Array[dim1 * kdx + puffer2];
                        }
                        break;
                    }
                case float[] np1Array:
                    {
                        float[] result = prod.Data<float>();
                        float[] np2Array = nd2.Data<float>();

                        for (int idx = 0; idx < prod.Size; idx++)
                        {
                            int puffer1 = idx / dim1;
                            int puffer2 = idx % dim1;
                            int puffer3 = puffer1 * iterator;
                            for (int kdx = 0; kdx < iterator; kdx++)
                                result[idx] += np1Array[puffer3 + kdx] * np2Array[dim1 * kdx + puffer2];
                        }

                        break;
                    }
                case Complex[] np1Array:
                    {
                        Complex[] result = prod.Data<Complex>();
                        Complex[] np2Array = nd2.Data<Complex>();

                        for (int idx = 0; idx < prod.Size; idx++)
                        {
                            int puffer1 = idx / dim1;
                            int puffer2 = idx % dim1;
                            int puffer3 = puffer1 * iterator;
                            for (int kdx = 0; kdx < iterator; kdx++)
                                result[idx] += np1Array[puffer3 + kdx] * np2Array[dim1 * kdx + puffer2];
                        }
                        break;
                    }
                case Quaternion[] np1Array:
                    {
                        Quaternion[] result = prod.Data<Quaternion>();
                        Quaternion[] np2Array = nd2.Data<Quaternion>();

                        for (int idx = 0; idx < prod.Size; idx++)
                        {
                            int puffer1 = idx / dim1;
                            int puffer2 = idx % dim1;
                            int puffer3 = puffer1 * iterator;
                            for (int kdx = 0; kdx < iterator; kdx++)
                                result[idx] += np1Array[puffer3 + kdx] * np2Array[dim1 * kdx + puffer2];
                        }
                        break;
                    }
                default:
                    {
                        throw new Exception("The Dot method is not implemented for the " + typeof(T).Name);
                    }
            }
            if ((this.Shape.NDim == 1) & (nd2.Shape.NDim == 1))
            {
                this.Shape = new Shape(Size);
                nd2.Shape = new Shape(nd2.Size);
                prod.Shape = new Shape(1);
            }

            return prod;
        }
    }

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
            if ((this.Shape.NDim == 1 ) & (np2.Shape.NDim == 1))
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
            if ((this.Shape.NDim == 1 ) & (np2.Shape.NDim == 1))
            {
                this.Shape = new Shape(this.Data.Length);
                np2.Shape = new Shape(np2.Data.Length);
                prod.Shape = new Shape(1);
            }

            return prod;
        }
    }
}
