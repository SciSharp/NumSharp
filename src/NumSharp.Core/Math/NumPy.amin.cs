using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray amin(NDArray nd, int? axis = null)
        {
            NDArray res = null;

            if (axis == null)
            {
                res = new NDArray(nd.dtype,new Shape(new int[] { 1 }));
                
                Array resArray = Array.CreateInstance(nd.dtype,1);
                resArray.SetValue(nd.Storage.GetData<double>().Min(),0);

                res.Storage.SetData(resArray);
                
            }
            else
            {
                if (axis < 0 || axis >= nd.NDim)
                    throw new Exception("Invalid input: axis");
                int[] resShapes = new int[nd.NDim - 1];
                int index = 0; //index for result shape set
                //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
                int prev = 1;
                int cur = 1;
                int post = 1;
                int size = 1; //total number of the elements for result
                //Calculate new Shape
                for (int i = 0; i < nd.NDim; i++)
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
                
                //Fill in data
                index = 0; //index for result data set
                int sameSetOffset = nd.Shape.DimOffset[axis.Value];
                int increments = cur * post;
                int start = 0;
                double min = 0;
                
                res = new NDArray(nd.dtype,new Shape(resShapes));
                
                for (int i = 0; i < nd.Size; i += increments)
                {
                    for (int j = i; j < i + post; j++)
                    {
                        start = j;
                        min = nd.Data<double>()[start];
                        for (int k = 0; k < cur; k++)
                        {
                            min = Math.Min(min, nd.Data<double>()[start]);
                            start += sameSetOffset;
                        }
                        res.Data<double>()[index++] = min;
                    }
                }
            }

            return res;
        }
    }
}
