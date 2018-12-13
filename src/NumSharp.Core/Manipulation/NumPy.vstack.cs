using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPy
    {
        /// <summary>
        /// Stack arrays in sequence vertically (row wise).
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArray vstack<T>(params NDArray[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            List<T> list = new List<T>();
            var np = new NDArray(typeof(T));
            foreach (NDArray ele in nps)
            {
                if (nps[0].shape != ele.shape)
                    throw new Exception("Arrays mush have same shapes");
                list.AddRange(ele.Data<T>());
            }
            np.Set(list.ToArray());
            if (nps[0].ndim == 1)
            {
                np.Storage.Shape = new Shape(new int[] { nps.Length, nps[0].shape.Shapes[0] });
            }
            else
            {
                int[] shapes = nps[0].shape.Shapes.ToArray();
                shapes[0] *= nps.Length;
                np.Storage.Shape = new Shape(shapes);
            }
            return np;
        }
    }
}
