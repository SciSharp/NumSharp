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
        public NDArray dot(NDArray nd2)
        {
            var pufferShape = nd2.Storage.Shape;

            if ((this.Shape.NDim == 1 ) & (nd2.Shape.NDim == 1))
                if (this.Shape.Shapes[0] != nd2.Shape.Shapes[0])
                    throw new IncorrectShapeException(); 
                else 
                {
                    nd2.Storage.Shape = new Shape(nd2.Storage.GetData().Length,1);
                    this.Storage.Shape = new Shape(1,this.Storage.GetData().Length);
                }
            else
                if (this.Shape.Shapes[1] != nd2.Shape.Shapes[0])
                    throw new IncorrectShapeException();
            
            if ((Shape.NDim == 2) & (nd2.Shape.NDim == 1))
            {
                var pufferList = pufferShape.Shapes.ToList();
                pufferList.Add(1);
                nd2.Storage.Shape = new Shape(pufferList.ToArray());
            }
            
            int iterator = this.Shape.Shapes[1];
            int dim0 = this.Shape.Shapes[0];
            int dim1 = nd2.Shape.Shapes[1];
            
            var prod = new NDArray(this.Storage.dtype, dim0, dim1);

            Array nd1SystemArray = this.Storage.GetData();

            switch (nd1SystemArray) 
            {
                case int[] nd1Array : 
                {
                    int[] result = prod.Storage.GetData<int>();
                    int[] nd2Array = nd2.Storage.GetData<int>();

                    for (int idx = 0; idx < prod.Size; idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd1Array[puffer3 + kdx] * nd2Array[dim1 * kdx + puffer2];
                    }
                    break;
                }
                case double[] nd1Array : 
                {
                    double[] result = prod.Storage.GetData<double>();
                    double[] nd2Array = nd2.Storage.GetData<double>();

                    for (int idx = 0; idx < prod.Size; idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd1Array[puffer3 + kdx] * nd2Array[dim1 * kdx + puffer2];
                    }
                    break;
                }
                case float[] nd1Array : 
                {
                    float[] result = prod.Storage.GetData<float>();
                    float[] nd2Array = nd2.Storage.GetData<float>();

                    for (int idx = 0; idx < prod.Size; idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd1Array[puffer3 + kdx] * nd2Array[dim1 * kdx + puffer2];
                    }
                    break;
                }
                case Complex[] nd1Array : 
                {
                    Complex[] result = prod.Storage.GetData<Complex>();
                    Complex[] nd2Array = nd2.Storage.GetData<Complex>();

                    for (int idx = 0; idx < prod.Size; idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd1Array[puffer3 + kdx] * nd2Array[dim1 * kdx + puffer2];
                    }
                    break;
                }
                case Quaternion[] nd1Array : 
                {
                    Quaternion[] result = prod.Storage.GetData<Quaternion>();
                    Quaternion[] nd2Array = nd2.Storage.GetData<Quaternion>();

                    for (int idx = 0; idx < prod.Size; idx++)
                    {
                        int puffer1 = idx / dim1;
                        int puffer2 = idx % dim1;
                        int puffer3 = puffer1 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd1Array[puffer3 + kdx] * nd2Array[dim1 * kdx + puffer2];
                    }
                    break;
                }
                default : 
                {
                    throw new NotImplementedException();
                }
            }

            if ((this.Shape.NDim == 1 ) & (nd2.Shape.NDim == 1))
            {
                this.Storage.Shape = new Shape(this.Storage.GetData().Length);
                nd2.Storage.Shape = new Shape(nd2.Storage.GetData().Length);
                prod.Storage.Shape = new Shape(1);
            }

            nd2.Storage.Shape = pufferShape;

            return prod;
        }
    }    
}
