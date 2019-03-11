using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray frombuffer(byte[] bytes, Type dtype)
        {
            if(dtype.Name == "Int32")
            {
                var size = bytes.Length / sizeof(int);
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt32(bytes, index * sizeof(int));
                }

                return new NDArray(ints);
            }

            throw new NotImplementedException("");
        }
    }
}
