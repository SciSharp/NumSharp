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
        /// <param name="np"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public NDArray amax(int? axis = null)
        {
            switch (dtype.Name)
            {
                case "Int32":
                    return amax<int>(axis);
                case "Single":
                    return amax<float>(axis);
                case "Double":
                    return amax<double>(axis);
            }

            throw new NotImplementedException($"amax {dtype.Name}");
        }

        private NDArray amax<T>(int? axis = null) where T : struct
        {
            var res = new NDArray(dtype);

            if (axis == null)
            {
                switch (dtype.Name)
                {
                    case "Int32":
                        {
                            var npArr = Data<int>();
                            var max = npArr[0];
                            for (int i = 0; i < npArr.Length; i++)
                                max = Math.Max(max, npArr[i]);
                            return max;
                        }

                    case "Double":
                        {
                            var npArr = Data<double>();
                            var max = npArr[0];
                            for (int i = 0; i < npArr.Length; i++)
                                max = Math.Max(max, npArr[i]);
                            return max;
                        }
                }
            }
            else
            {
                if (axis < 0 || axis >= this.ndim)
                    throw new Exception("Invalid input: axis");

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
                int sameSetOffset = Storage.Shape.Strides[axis.Value];
                int increments = cur * post;
                switch (dtype.Name)
                {
                    case "Int32":
                        {
                            var npArr = Data<int>();
                            var resData = new int[size];  //res.Data = new double[size];
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
                            res.Storage = new ArrayStorage(dtype);
                            res.Storage.Allocate(new Shape(resShapes)); // (resData);
                            res.Storage.SetData(resData);
                        }
                        break;
                    case "Single":
                        {
                            var npArr = Data<float>();
                            var resData = new float[size];  //res.Data = new double[size];
                            int start = 0;
                            float min = 0;
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
                            res.Storage = new ArrayStorage(dtype);
                            res.Storage.Allocate(new Shape(resShapes)); // (resData);
                            res.Storage.SetData(resData);
                        }
                        break;
                }

            }
            return res;
        }
    }
}
