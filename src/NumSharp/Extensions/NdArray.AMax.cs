using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Return the maximum of an array or minimum along an axis
        /// </summary>
        /// <param name="np"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static NDArray<double> AMax(this NDArray<double> np, int? axis = null)
        {
            NDArray<double> res = new NDArray<double>();
            if (axis == null)
            {
                double min = np.Data[0];
                for (int i = 0; i < np.Size; i++)
                    min = Math.Max(min, np.Data[i]);
                res.Data = new double[1];
                res.Data[0] = min;
                res.Shape = new Shape(new int[] { 1 });
            }
            else
            {
                if (axis < 0 || axis >= np.Shape.Length)
                    throw new Exception("Invalid input: axis");
                int[] resShapes = new int[np.Shape.Shapes.Count - 1];
                int index = 0; //index for result shape set
                int prev = 1;
                int cur = 1; //numbers in a comparing set
                int post = 1;
                int size = 1; //total number of the elements for result
                //Calculate new Shape
                for (int i = 0; i < np.Shape.Shapes.Count; i++)
                {
                    if (i == axis)
                        cur = np.Shape.Shapes[i];
                    else
                    {
                        resShapes[index++] = np.Shape.Shapes[i];
                        size *= np.Shape.Shapes[i];
                        if (i < axis)
                            prev *= np.Shape.Shapes[i];
                        else
                            post *= np.Shape.Shapes[i];
                    }
                }
                res.Shape = new Shape(resShapes);
                //Fill in data
                index = 0; //index for result data set
                int sameSetOffset = np.Shape.DimOffset[axis.Value];
                int increments = cur * post;
                res.Data = new double[size];
                int start = 0;
                double min = 0;
                for (int i = 0; i < np.Size; i += increments)
                {
                    for (int j = i; j < i + post; j++)
                    {
                        start = j;
                        min = np.Data[start];
                        for (int k = 0; k < cur; k++)
                        {
                            min = Math.Max(min, np.Data[start]);
                            start += sameSetOffset;
                        }
                        res.Data[index++] = min;
                    }
                }
            }
            return res;
        }
    }
}
