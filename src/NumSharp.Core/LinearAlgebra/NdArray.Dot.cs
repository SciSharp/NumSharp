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

            // in case must do a reshape
            var oldStorage1 = this.Storage;
            var oldStorage2 = nd2.Storage;

            if ((this.shape.NDim == 1 ) & (nd2.shape.NDim == 1))
                if (this.shape.Dimensions[0] != nd2.shape.Dimensions[0])
                    throw new IncorrectShapeException(); 
                else 
                {
                    this.Storage = new NDStorage();
                    this.Storage.Allocate(oldStorage1.DType,new Shape(1,oldStorage1.GetData().Length),1);
                    this.Storage.SetData(oldStorage1.GetData());

                    nd2.Storage = new NDStorage();
                    nd2.Storage.Allocate(oldStorage2.DType, new Shape(oldStorage2.GetData().Length,1),1);
                    nd2.Storage.SetData(oldStorage2.GetData());
                }
            else
                if (this.shape.Dimensions[1] != nd2.shape.Dimensions[0])
                    throw new IncorrectShapeException();
            
            if ((shape.NDim == 2) & (nd2.shape.NDim == 1))
            {
                var pufferList = pufferShape.Dimensions.ToList();
                pufferList.Add(1);
                nd2.Storage.Reshape(pufferList.ToArray());
            }
            
            int iterator = this.shape.Dimensions[1];
            int dim0 = this.shape.Dimensions[0];
            int dim1 = nd2.shape.Dimensions[1];
            
            var prod = new NDArray(this.Storage.DType, dim0, dim1);

            Array nd1SystemArray = this.Storage.GetData();

            switch (nd1SystemArray) 
            {
                case int[] nd1Array : 
                {
                    int[] result = prod.Storage.GetData<int>();
                    int[] nd2Array = nd2.Storage.GetData<int>();

                    for (int idx = 0; idx < prod.size; idx++)
                    {
                        int puffer1 = idx % dim0;
                        int puffer2 = idx / dim0;
                        int puffer3 = puffer2 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
                    }
                    break;
                }
                case double[] nd1Array : 
                {
                    double[] result = prod.Storage.GetData<double>();
                    double[] nd2Array = nd2.Storage.GetData<double>();

                    for (int idx = 0; idx < prod.size; idx++)
                    {
                        int puffer1 = idx % dim0;
                        int puffer2 = idx / dim0;
                        int puffer3 = puffer2 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
                    }
                    break;
                }
                case float[] nd1Array : 
                {
                    float[] result = prod.Storage.GetData<float>();
                    float[] nd2Array = nd2.Storage.GetData<float>();

                    for (int idx = 0; idx < prod.size; idx++)
                    {
                        int puffer1 = idx % dim0;
                        int puffer2 = idx / dim0;
                        int puffer3 = puffer2 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
                    }
                    break;
                }
                case Complex[] nd1Array : 
                {
                    Complex[] result = prod.Storage.GetData<Complex>();
                    Complex[] nd2Array = nd2.Storage.GetData<Complex>();

                    for (int idx = 0; idx < prod.size; idx++)
                    {
                        int puffer1 = idx % dim0;
                        int puffer2 = idx / dim0;
                        int puffer3 = puffer2 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
                    }
                    break;
                }
                case Quaternion[] nd1Array : 
                {
                    Quaternion[] result = prod.Storage.GetData<Quaternion>();
                    Quaternion[] nd2Array = nd2.Storage.GetData<Quaternion>();

                    for (int idx = 0; idx < prod.size; idx++)
                    {
                        int puffer1 = idx % dim0;
                        int puffer2 = idx / dim0;
                        int puffer3 = puffer2 * iterator;
                        for (int kdx = 0; kdx < iterator; kdx++)
                            result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
                    }
                    break;
                }
                default : 
                {
                    throw new NotImplementedException();
                }
            }

            if ((this.shape.NDim == 1 ) & (nd2.shape.NDim == 1))
            {
                this.Storage.Reshape(this.Storage.GetData().Length);
                nd2.Storage.Reshape(nd2.Storage.GetData().Length);
                prod.Storage.Reshape(1);
            }

            this.Storage = oldStorage1;
            nd2.Storage = oldStorage2;

            return prod;
        }
    }    
}
