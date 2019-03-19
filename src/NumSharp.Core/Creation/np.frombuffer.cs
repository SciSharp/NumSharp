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
            if (dtype.Name == "Int32")
            {
                var size = bytes.Length / sizeof(int);
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt32(bytes, index * sizeof(int));
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == "Byte")
            {
                var size = bytes.Length / sizeof(byte);
                var ints = bytes;
                return new NDArray(bytes);
            }

            throw new NotImplementedException("");
        }

        public static NDArray frombuffer(byte[] bytes, string dtype)
        {
            if (dtype == ">u4")
            {
                var size = bytes.Length / sizeof(uint);
                var ints = new uint[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = (uint)(bytes[0] * 256 + bytes[1] + bytes[2] * 256 + bytes[3]);
                }

                return new NDArray(ints);
            }

            throw new NotImplementedException("");
        }
    }
}
