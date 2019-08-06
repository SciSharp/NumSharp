namespace NumSharp
{
    public static partial class np
    {
        /*
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
                list.AddRange(ele.Storage.GetData<T>());
            }
            np.Storage.ReplaceData(list.ToArray());
            if (nps[0].ndim == 1)
            {
                np.Storage.Reshape(new int[] { nps.Length, nps[0].shape[0] });
            }
            else
            {
                int[] shapes = nps[0].shape;
                shapes[0] *= nps.Length;
                np.Storage.Reshape(shapes);
            }
            return np;
        }
        */
    }
}
