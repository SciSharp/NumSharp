using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArrayWithDType
    {
        public NDArrayWithDType AMin(int? axis = null)
        {
            NDArrayWithDType res = new NDArrayWithDType(dtype);
            if (axis == null)
            {
                switch (dtype.Name)
                {
                    case "Double":
                        res.Storage.Set(new double[1] { Storage.Double8.Min() });
                        break;
                }
                
                res.Shape = new Shape(new int[] { 1 });
            }
            else
            {
                if (axis < 0 || axis >= Shape.Length)
                    throw new Exception("Invalid input: axis");
                int[] resShapes = new int[Shape.Shapes.Count - 1];
                int index = 0; //index for result shape set
                //axis departs the shape into three parts: prev, cur and post. They are all product of shapes
                int prev = 1; 
                int cur = 1;
                int post = 1;
                int size = 1; //total number of the elements for result
                //Calculate new Shape
                for (int i = 0; i < Shape.Shapes.Count; i++)
                {
                    if (i == axis)
                        cur = Shape.Shapes[i];
                    else
                    {
                        resShapes[index++] = Shape.Shapes[i];
                        size *= Shape.Shapes[i];
                        if (i < axis)
                            prev *= Shape.Shapes[i];
                        else
                            post *= Shape.Shapes[i];
                    }
                }
                res.Shape = new Shape(resShapes);
                //Fill in data
                index = 0; //index for result data set
                int sameSetOffset = Shape.DimOffset[axis.Value];
                int increments = cur * post;
                res.Storage.Allocate(size);
                int start = 0;
                double min;
                for (int i = 0; i < Size; i += increments)
                {
                    for (int j = i; j < i + post; j++)
                    {
                        start = j;
                        min = Storage.Double8[start];
                        for (int k = 0; k < cur; k++)
                        {
                            min = Math.Min(min, Storage.Double8[start]);
                            start += sameSetOffset;
                        }
                        res.Storage.Double8[index++] = min;
                    }
                }
            }
            return res;
        }
    }
}
