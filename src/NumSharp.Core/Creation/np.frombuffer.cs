using System;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray frombuffer(byte[] bytes, Type dtype)
        {
            //TODO! all types
            if (dtype.Name == nameof(Int16))
            {
                var size = bytes.Length / InfoOf<short>.Size;
                var ints = new short[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt16(bytes, index * InfoOf<short>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(Int32))
            {
                var size = bytes.Length / InfoOf<int>.Size;
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt32(bytes, index * InfoOf<int>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(Int64))
            {
                var size = bytes.Length / InfoOf<long>.Size;
                var ints = new long[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToInt64(bytes, index * InfoOf<long>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(UInt16))
            {
                var size = bytes.Length / InfoOf<ushort>.Size;
                var ints = new ushort[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToUInt16(bytes, index * InfoOf<ushort>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(UInt32))
            {
                var size = bytes.Length / InfoOf<uint>.Size;
                var ints = new uint[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToUInt32(bytes, index * InfoOf<uint>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(UInt64))
            {
                var size = bytes.Length / InfoOf<ulong>.Size;
                var ints = new ulong[size];
                for (var index = 0; index < size; index++)
                {
                    ints[index] = BitConverter.ToUInt64(bytes, index * InfoOf<ulong>.Size);
                }

                return new NDArray(ints);
            }
            else if (dtype.Name == nameof(Single))
            {
                var size = bytes.Length / InfoOf<float>.Size;
                var floats = new float[size];
                for (var index = 0; index < size; index++)
                {
                    floats[index] = BitConverter.ToSingle(bytes, index * InfoOf<float>.Size);
                }

                return new NDArray(floats);
            }
            else if (dtype.Name == nameof(Double))
            {
                var size = bytes.Length / InfoOf<double>.Size;
                var floats = new double[size];
                for (var index = 0; index < size; index++)
                {
                    floats[index] = BitConverter.ToDouble(bytes, index * InfoOf<double>.Size);
                }

                return new NDArray(floats);
            }
            else if (dtype.Name == nameof(Byte))
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
                var size = bytes.Length / InfoOf<uint>.Size;
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
