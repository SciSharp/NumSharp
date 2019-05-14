using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        
        public static NDArray<bool> operator &(NDArray np_, NDArray obj_)
        {
            var boolTensor = new NDArray(typeof(bool),np_.shape);
            bool[] bools = boolTensor.Storage.GetData<bool>();

            bool[] np = np_.Storage.GetData<bool>();
            bool[] obj = obj_.Storage.GetData<bool>();

             for(int i = 0;i < bools.Length;i++)
                bools[i] = np[i] && obj[i];
            
            return boolTensor.MakeGeneric<bool>();
        }

        public static NDArray<byte> operator &(NDArray np_, byte value)
        {
            var byteTensor = new NDArray(typeof(byte), np_.shape);
            byte[] bytes = byteTensor.Storage.GetData<byte>();

            byte[] np = np_.Storage.GetData<byte>();

            for (int i = 0; i < bytes.Length; i++)
                bytes[i] = (byte)(np[i] & value);

            return byteTensor.MakeGeneric<byte>();
        }

    }
}
