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
            
            if (nps[0].shape.Shapes.Count == 1)
            {
                np.Storage.Shape = new Shape(new int[] { nps.Length +1, nps[0].shape.Shapes[0] });
            }
            else
            {
                int[] shapes = nps[0].shape.Shapes.ToArray();
                shapes[0] *= nps.Length + 1;
                np.Storage.Shape = new Shape(shapes) ;
            }
            return np;
        }
    }
    public static partial class NDArrayExtensions
    {
        /// <summary>
        /// Stack arrays in sequence vertically (row wise).
        /// </summary>
        /// <param name="nps"></param>
        /// <returns></returns>
        public static NDArrayGeneric<T> VStack<T>(this NDArrayGeneric<T> np1, params NDArrayGeneric<T>[] nps)
        {
            if (nps == null || nps.Length == 0)
                throw new Exception("Input arrays can not be empty");
            List<T> list = new List<T>();
            NDArrayGeneric<T> np = new NDArrayGeneric<T>();
            foreach (NDArrayGeneric<T> ele in nps)
            {
                if (nps[0].Shape != ele.Shape)
                    throw new Exception("Arrays mush have same shapes");
                list.AddRange(ele.Data);
            }
            np.Data = list.ToArray();
            if (nps[0].NDim == 1)
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
