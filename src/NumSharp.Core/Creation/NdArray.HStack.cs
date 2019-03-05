using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        /// <summary>
        /// Stack arrays in sequence horizontally
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public NDArray hstack(params NDArray[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");

            var npAll = new NDArray[nps.Length + 1];
            npAll[0] = this;

            for (int idx = 0; idx < nps.Length; idx++)
                if (nps[0].Storage.Shape != nps[idx].Storage.Shape)
                    throw new Exception("Arrays mush have same shapes");
                else
                    npAll[idx + 1] = nps[idx];

            NDArray nd = new NDArray();

            // easy 1D case
            if (this.ndim == 1)
            {
                var list1D = new List<object>();
                for (int idx = 0; idx < npAll.Length; idx++)
                    list1D.AddRange(npAll[idx].Storage.GetData<object>().ToList());

                nd = np.array(list1D.ToArray(), this.dtype);
            }
            else
            {
                for (int idx = 0; idx < npAll.Length; idx++)
                    npAll[idx].Storage.ChangeTensorLayout(2);
                int total = npAll[0].ndim == 1 ? 1 : npAll[0].shape[0];
                var list = new List<object>(); 

                for (int i = 0; i < total; i++)
                {
                    for (int k = 0; k < npAll.Length; k++)
                    {
                        var pufferShape = new Shape(npAll[k].shape);
                        pufferShape.ChangeTensorLayout(this.Storage.Shape.TensorLayout);
                        int pageSize = npAll[k].ndim == 1 ? npAll[k].shape[0] : pufferShape.DimOffset[0];
                        for (int j = i * pageSize; j < (i + 1) * pageSize; j++)
                        {
                            var ele = npAll[k].Storage.GetData<object>()[j];
                            list.Add(ele);
                        }
                    }
                }

                int[] shapes = new int[npAll[0].shape.Length];
                npAll[0].shape.CopyTo(shapes, 0);
                if (shapes.Length == 1)
                    shapes[0] *= npAll.Length;
                else
                    shapes[1] = npAll.Sum(x => x.shape[1]);

                nd.Storage.Allocate(nd.Storage.DType, new Shape(shapes), 2);
                nd.Storage.SetData(list.ToArray());
                nd.Storage.ChangeTensorLayout(1);

            }

            return nd;
        }
    }
}
