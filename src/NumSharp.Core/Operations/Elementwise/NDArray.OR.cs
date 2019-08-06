namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray operator |(NDArray np_, NDArray obj_)
        {
            return null;
            //var boolTensor = new NDArray(typeof(bool),np_.shape);
            //bool[] bools = boolTensor.Storage.GetData() as bool[];

            //bool[] np = np_.MakeGeneric<bool>().Storage.GetData() as bool[];
            //bool[] obj = obj_.MakeGeneric<bool>().Storage.GetData() as bool[];

            // for(int idx = 0;idx < bools.Length;idx++)
            //    bools[idx] = np[idx] || obj[idx];

            //return boolTensor.MakeGeneric<bool>();
        }
    }
}
