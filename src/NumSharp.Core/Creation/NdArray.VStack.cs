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

            int dataLength = this.Storage.Length;

            for (int idx = 0;idx < nps.Length;idx++)     
                dataLength += nps[idx].Storage.Length;       

            Array dataSysArr = Array.CreateInstance(this.dtype,dataLength);
            NDArray np = new NDArray(this.dtype);

            this.Storage.GetData(this.dtype).CopyTo(dataSysArr,0);

            int idxOfLastCopyiedElement = this.Storage.Length;

            for (int idx = 0; idx < nps.Length;idx++)
            {
                nps[idx].Storage.GetData(this.dtype).CopyTo(dataSysArr,idxOfLastCopyiedElement);
                idxOfLastCopyiedElement += nps[idx].Storage.Length;
            }
            
            var check = dataSysArr.GetValue(idxOfLastCopyiedElement-1);

            if (ndim == 1)
            {
                np.Storage.Shape = new Shape(new int[] { nps.Length+1, shape.Shapes[0] });
            }
            else
            {
                int[] shapes = nps[0].shape.Shapes.ToArray();
                shapes[0] *= nps.Length;
                np.Storage.Shape = new Shape(shapes);
            }

            np.Storage.SetData(dataSysArr);

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
