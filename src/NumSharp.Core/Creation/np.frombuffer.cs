using System;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray frombuffer(byte[] bytes, Type dtype)
        {

            //TODO! all types
            if (dtype.Name == "Int32")
            {
                var size = bytes.Length / InfoOf<int>.Size;
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt32(bytes, index * InfoOf<int>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == "Byte")
            {
                var size = bytes.Length / InfoOf<byte>.Size;
                var ints = bytes;
                return new NDArray(bytes);
            }

            throw new NotImplementedException("");
        }

        public static NDArray frombuffer(byte[] bytes, string dtype)
        {
            if (dtype == ">u4")
            {
                var size = bytes.Length / InfoOf<int>.Size;
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                    ints[index] = bytes[0] * 256 + bytes[1] + bytes[2] * 256 + bytes[3];

                return new NDArray(ints);
            }

            throw new NotImplementedException("");
        }
    }
}
