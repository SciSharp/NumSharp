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

            var npAll = new NDArray[nps.Length+1];
            npAll[0] = this;

            for (int idx = 0; idx < nps.Length; idx++)
                if (nps[0].shape != nps[idx].shape)
                    throw new Exception("Arrays mush have same shapes");
                else
                    npAll[idx+1] = nps[idx];

            NDArray nd = new NDArray();

            // easy 1D case
            if (this.ndim == 1)
            {
                var list1D = new List<object>();
                for (int idx = 0; idx < npAll.Length;idx++)
                    list1D.AddRange(npAll[idx].Storage.GetData<object>().ToList());
                
                nd = np.array(list1D.ToArray(),this.dtype);
            }
            else
            {
                var list = new List<object>();

                int total = npAll[0].ndim == 1 ? 1 : npAll[0].shape.Shapes[0];

                int pageSize = npAll[0].ndim == 1 ? npAll[0].shape.Shapes[0] : npAll[0].shape.DimOffset[0];

                for (int i = 0; i < total; i++)
                {
                    for (int k = 0; k < npAll.Length; k++)
                    {
                        for (int j = i * pageSize; j < (i + 1) * pageSize; j++)
                        list.Add(npAll[k].Storage.GetData<object>()[j]);
                    }
                }
                
                nd.Storage.SetData( list.ToArray());

                int[] shapes = npAll[0].shape.Shapes.ToArray();

                if (shapes.Length == 1)
                    shapes[0] *= npAll.Length;
                else
                    shapes[1] *= npAll.Length;

                nd.Storage.Shape = new Shape(shapes);

            
            }

            return nd;
        }
    }
}
