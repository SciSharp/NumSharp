using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray<T> MakeGeneric<T>() where T : struct
        {
            var genericArray = new NDArray<T>(Storage.Shape);

            genericArray.Array = Storage.GetData<T>();

            return genericArray;
        }
    }
}
