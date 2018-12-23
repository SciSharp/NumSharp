using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray vstack(params NDArray[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            
            List<object> list = this.Storage.GetData<object>().ToList();

            NDArray np = new NDArray(dtype);

            foreach (NDArray ele in nps)
            {
                if (nps[0].shape != ele.shape)
                    throw new Exception("Arrays mush have same shapes");
                
                list.AddRange(ele.Storage.GetData<object>());
            }

            np.Storage.SetData(list.ToArray());
            
            if (nps[0].ndim == 1)
            {
                np.Storage.Reshape(new int[] { nps.Length +1, nps[0].shape[0] });
            }
            else
            {
                int[] shapes = nps[0].shape;
                shapes[0] *= nps.Length + 1;
                np.Storage.Reshape(shapes) ;
            }
            return np;
        }
    }
  
}
