using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Stack arrays in sequence vertically (row wise).
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArray<T> VStack<T>(this NDArray<T> np1, params NDArray<T>[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            List<T> list = new List<T>();
            NDArray<T> np = new NDArray<T>();
            foreach (NDArray<T> ele in nps)
            {
                if (nps[0].Shape != ele.Shape)
                    throw new Exception("Arrays mush have same shapes");
                list.AddRange(ele.Data);
            }
            np.Data = list.ToArray();
            if (nps[0].Shape.Length == 1)
            {
                np.Shape = new Shape(new int[] { nps.Length, nps[0].Shape.Shapes[0] });
            }
            else
            {
                int[] shapes = nps[0].Shape.Shapes.ToArray();
                shapes[0] *= nps.Length;
                np.Shape = new Shape(shapes);
            }
            return np;
        }
    }
}
