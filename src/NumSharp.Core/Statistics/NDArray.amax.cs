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
            if (axis < 0 || axis >= ndim)
                throw new Exception("Invalid input: axis");

            int[] resShapes = new int[shape.Length - 1];
            int index = 0; //index for result shape set
                           //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
            int prev = 1;
            int cur = 1;
            int post = 1;
            int size = 1; //total number of the elements for result
                          //Calculate new Shape

            var res = new NDArray(dtype, resShapes);
            for (int i = 0; i < shape.Length; i++)
            {
                if (i == axis)
                    cur = shape[i];
                else
                {
                    resShapes[index++] = shape[i];
                    size *= shape[i];
                    if (i < axis)
                        prev *= shape[i];
                    else
                        post *= shape[i];
                }
            }

            //Fill in data
            index = 0; //index for result data set
            int sameSetOffset = Storage.Shape.Strides[axis];
            int increments = cur * post;

            switch (dtype.Name)
            {
                case "Int32":
                    {
                        int[] resData = new int[size];
                        int start = 0;
                        int min = 0;
                        var npArr = Data<int>();
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
                        res.Array = resData;
                    }
                    break;

                case "Single":
                    {
                        var resData = new float[size];
                        int start = 0;
                        float min = 0;
                        var npArr = Data<float>();
                        for (int i = 0; i < size; i += increments)
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
                        res.Array = resData;
                    }
                    break;
            }

            return res;
        }
    }
}
