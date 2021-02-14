using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray<bool> operator &(NDArray lhs, NDArray rhs)
        {
            return null;
            //var boolTensor = new NDArray(typeof(bool),lhs.shape);
            //bool[] bools = boolTensor.Storage.GetData<bool>();

            //bool[] np = lhs.Storage.GetData<bool>();
            //bool[] obj = rhs.Storage.GetData<bool>();

            // for(int i = 0;i < bools.Length;i++)
            //    bools[i] = np[i] && obj[i];

            //return boolTensor.MakeGeneric<bool>();
        }

        public static NDArray<byte> operator &(NDArray lhs, int rhs)
        {
            return null;
        }

        public static NDArray<byte> operator &(NDArray lhs, byte rhs)
        {
            return null;
            //var result = new NDArray(typeof(byte), lhs.shape);
            //byte[] resultBytes = result.Storage.GetData<byte>();

            //byte[] lhsValues = lhs.Storage.GetData<byte>();

            //for (int i = 0; i < resultBytes.Length; i++)
            //    resultBytes[i] = (byte)(lhsValues[i] & rhs);

            //return result.MakeGeneric<byte>();
        }
    }
}
