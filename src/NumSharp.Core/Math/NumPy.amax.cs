using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray amax(NDArray nd, int? axis = null)
        {
            var res = new NDArray(nd.dtype, nd.shape);

            if (axis == null)
            {
                res.Set(new double[1] { nd.Data<double>().Max() });
                res.Storage.Shape = new Shape(new int[] { 1 });
            }
            else
            {
                if (axis < 0 || axis >= nd.ndim)
                    throw new Exception("Invalid input: axis");
                int[] resShapes = new int[nd.ndim - 1];
                int index = 0; //index for result shape set
                //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
                int prev = 1;
                int cur = 1;
                int post = 1;
                int size = 1; //total number of the elements for result
                //Calculate new Shape
                for (int i = 0; i < nd.ndim; i++)
                {
                    if (i == axis)
                        cur = nd.shape.Shapes[i];
                    else
                    {
                        resShapes[index++] = nd.shape.Shapes[i];
                        size *= nd.shape.Shapes[i];
                        if (i < axis)
                            prev *= nd.shape.Shapes[i];
                        else
                            post *= nd.shape.Shapes[i];
                    }
                }
                res.Storage.Shape = new Shape(resShapes);
                //Fill in data
                index = 0; //index for result data set
                int sameSetOffset = nd.shape.DimOffset[axis.Value];
                int increments = cur * post;
                int start = 0;
                double min = 0;
                for (int i = 0; i < nd.size; i += increments)
                {
                    for (int j = i; j < i + post; j++)
                    {
                        start = j;
                        min = nd.Data<double>()[start];
                        for (int k = 0; k < cur; k++)
                        {
                            min = Math.Max(min, nd.Data<double>()[start]);
                            start += sameSetOffset;
                        }
                        res[index++] = min;
                    }
                }
            }

            return res;
        }
    }
}
