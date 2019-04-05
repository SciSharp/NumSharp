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
    }
}
