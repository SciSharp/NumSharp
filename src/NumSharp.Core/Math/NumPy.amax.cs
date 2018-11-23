using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray amax(NDArray nd, int? axis = null)
        {
            var res = new NDArray(nd.dtype, nd.Shape);

            if (axis == null)
            {
                double min = nd.Data<double>(0);
                for (int i = 0; i < nd.Size; i++)
                    min = Math.Max(min, nd.Data<double>(i));
                res.Set(new double[1]);
                res[0] = min;
                res.Shape = new Shape(new int[] { 1 });
            }
            else
            {
                if (axis < 0 || axis >= nd.Shape.Length)
                    throw new Exception("Invalid input: axis");
                int[] resShapes = new int[nd.Shape.Shapes.Count - 1];
                int index = 0; //index for result shape set
                //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
                int prev = 1;
                int cur = 1;
                int post = 1;
                int size = 1; //total number of the elements for result
                //Calculate new Shape
                for (int i = 0; i < nd.Shape.Shapes.Count; i++)
                {
                    if (i == axis)
                        cur = nd.Shape.Shapes[i];
                    else
                    {
                        resShapes[index++] = nd.Shape.Shapes[i];
                        size *= nd.Shape.Shapes[i];
                        if (i < axis)
                            prev *= nd.Shape.Shapes[i];
                        else
                            post *= nd.Shape.Shapes[i];
                    }
                }
                res.Shape = new Shape(resShapes);
                //Fill in data
                index = 0; //index for result data set
                int sameSetOffset = nd.Shape.DimOffset[axis.Value];
                int increments = cur * post;
                int start = 0;
                double min = 0;
                for (int i = 0; i < nd.Size; i += increments)
                {
                    for (int j = i; j < i + post; j++)
                    {
                        start = j;
                        min = nd.Data<double>(start);
                        for (int k = 0; k < cur; k++)
                        {
                            min = Math.Max(min, nd.Data<double>(start));
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
