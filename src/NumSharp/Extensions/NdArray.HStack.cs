using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Stack arrays in sequence horizontally
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArray<T> HStack<T>(params NDArray<T>[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            List<T> list = new List<T>();
            NDArray<T> np = new NDArray<T>();
            foreach (NDArray<T> ele in nps)
            {
                if (nps[0].Shape != ele.Shape)
                    throw new Exception("Arrays mush have same shapes");
            }
            int total = nps[0].Shape.Length == 1 ? 1 : nps[0].Shape.Shapes[0];
            int pageSize = nps[0].Shape.Length == 1 ? nps[0].Shape.Shapes[0] : nps[0].Shape.DimOffset[0];

            //int i = 0;
            //Enumerable.Range(0, total)
            //                    .Select(x =>
            //                    {
            //                        foreach (NDArray<T> ele in nps)
            //                        {
            //                            for (int j = i * pageSize; j < (i + 1) * pageSize; j++)
            //                                list.Add(ele.Data[j]);
            //                        }
            //                        return i++;
            //                    })
            //                    .ToArray();

            for (int i = 0; i < total; i++)
            {
                for (int k = 0; k < nps.Length; k++)
                {
                    for (int j = i * pageSize; j < (i + 1) * pageSize; j++)
                        list.Add(nps[k].Data[j]);
                }
            }
            np.Data = list.ToArray();
            int[] shapes = nps[0].Shape.Shapes.ToArray();
            if (shapes.Length == 1)
                shapes[0] *= nps.Length;
            else
                shapes[1] *= nps.Length;
            np.Shape = new Shape(shapes);
            return np;
        }
    }
}
