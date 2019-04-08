using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {

        /// <summary>
        /// Return the maximum of an array or minimum along an axis
        /// </summary>
        /// <returns></returns>
        public T amax<T>()
        {
            switch (dtype.Name)
            {
                case "Int32":
                    {
                        var npArr = Data<int>();
                        int max = npArr[0];
                        for (int i = 0; i < npArr.Length; i++)
                            max = Math.Max(max, npArr[i]);
                        return (T)(object)max;
                    }
                case "Double":
                    {
                        var npArr = Data<double>();
                        double max = npArr[0];
                        for (int i = 0; i < npArr.Length; i++)
                            max = Math.Max(max, npArr[i]);
                        return (T)(object)max;
                    }
                default:
                    throw new NotImplementedException($"Data type not supported yet {dtype.Name}");
            }
        }

        /// <summary>
        /// Return the maximum of an array or minimum along an axis
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        public NDArray amax(int axis)
        {
            var res = new NDArray(dtype);

            if (axis < 0 || axis >= this.ndim)
                throw new Exception("Invalid input: axis");

            var npArr = this.Storage.GetData<int>();

            int[] resShapes = new int[this.shape.Length - 1];
            int index = 0; //index for result shape set
                           //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
            int prev = 1;
            int cur = 1;
            int post = 1;
            int size = 1; //total number of the elements for result
                          //Calculate new Shape
            for (int i = 0; i < this.shape.Length; i++)
            {
                if (i == axis)
                    cur = this.shape[i];
                else
                {
                    resShapes[index++] = this.shape[i];
                    size *= this.shape[i];
                    if (i < axis)
                        prev *= this.shape[i];
                    else
                        post *= this.shape[i];
                }
            }

            //Fill in data
            index = 0; //index for result data set
            int sameSetOffset = this.Storage.Shape.DimOffset[axis];
            int increments = cur * post;
            int[] resData = new int[size];  //res.Data = new double[size];
            int start = 0;
            int min = 0;
            for (int i = 0; i < this.size; i += increments)
            {
                for (int j = i; j < i + post; j++)
                {
                    start = j;
                    min = npArr[start];
                    for (int k = 0; k < cur; k++)
                    {
                        min = Math.Max(min, npArr[start]);
                        start += sameSetOffset;
                    }
                    resData[index++] = min;
                }
            }
            res.Storage = new NDStorage(dtype);
            res.Storage.Allocate(new Shape(resShapes)); // (resData);
            res.Storage.SetData(resData);
            return res;
        }
    }
}
