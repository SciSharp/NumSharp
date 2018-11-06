using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Stack arrays in sequence vertically (row wise).
        /// </summary>
        /// <param name="np"></param>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArray<double> VStack(params NDArray<double>[] nps)
        {
            if (nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            List<double> list = new List<double>();
            NDArray<double> np = nps[0];
            for (int i = 0; i < np.Shape.Length; i++)
            {
                foreach (NDArray<double> ele in nps)
                {
                    if (np.Shape.Shapes[i] != ele.Shape.Shapes[i])
                        throw new Exception("Arrays mush have same shapes");
                    list.AddRange(ele.Data);
                }
            }
            np.Data = list.ToArray();
            if (np.Shape.Length == 1)
            {
                np.Shape = new Shape(new int[] { nps.Length, np.Shape.Shapes[0] });
            }
            else
            {
                int[] shapes = np.Shape.Shapes;
                shapes[0] *= nps.Length;
                np.Shape = new Shape(shapes);
            }
            return np;
        }
    }
}
