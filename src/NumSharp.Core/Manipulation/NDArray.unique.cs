namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Find the unique elements of an array.
        /// 
        /// Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:
        /// * the indices of the input array that give the unique values
        /// * the indices of the unique array that reconstruct the input array
        /// * the number of times each unique value comes up in the input array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public NDArray unique<T>()
        {
            return null;
            //var nd = new NDArray(dtype);
            //var data = Storage.GetData<T>().Distinct().ToArray();
            //nd.Storage.ReplaceData(data);

            //nd.Storage.Reshape(data.Length);

            //return nd;
        }
    }
}
