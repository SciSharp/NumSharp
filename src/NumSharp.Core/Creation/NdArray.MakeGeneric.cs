using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NumSharp.Generic.NDArray<T> MakeGeneric<T>() where T : struct
        {
            var genericArray = new NumSharp.Generic.NDArray<T>(this.Storage.Shape);

            genericArray.Storage.SetData(this.Storage.GetData<T>());

            return genericArray;
        }
    }
}