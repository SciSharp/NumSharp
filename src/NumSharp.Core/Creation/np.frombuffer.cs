using System;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray fromspan(Span<byte> span, Type dtype)
        {
            if (dtype == typeof(int))
            {
                return new NDArray(MemoryMarshal.Cast<byte, int>(span).ToArray());
            }
            else if (dtype == typeof(uint))
            {
                return new NDArray(MemoryMarshal.Cast<byte, uint>(span).ToArray());
            }
            else if (dtype == typeof(short))
            {
                return new NDArray(MemoryMarshal.Cast<byte, short>(span).ToArray());
            }
            else if (dtype == typeof(ushort))
            {
                return new NDArray(MemoryMarshal.Cast<byte, ushort>(span).ToArray());
            }
            else if (dtype == typeof(long))
            {
                return new NDArray(MemoryMarshal.Cast<byte, long>(span).ToArray());
            }
            else if (dtype == typeof(ulong))
            {
                return new NDArray(MemoryMarshal.Cast<byte, ulong>(span).ToArray());
            }
            else if (dtype == typeof(bool))
            {
                return new NDArray(MemoryMarshal.Cast<byte, bool>(span).ToArray());
            }
            else if (dtype == typeof(char))
            {
                return new NDArray(MemoryMarshal.Cast<byte, char>(span).ToArray());
            }
            else if (dtype == typeof(byte))
            {
                return new NDArray(span.ToArray());
            }

            throw new NotImplementedException("");
        }


        public static NDArray frombuffer(byte[] bytes, Type dtype)
        {
            if (dtype == typeof(int))
            {
                var span = MemoryMarshal.Cast<byte, int>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(uint))
            {
                var span = MemoryMarshal.Cast<byte, uint>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(short))
            {
                var span = MemoryMarshal.Cast<byte, short>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(ushort))
            {
                var span = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(long))
            {
                var span = MemoryMarshal.Cast<byte, long>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(ulong))
            {
                var span = MemoryMarshal.Cast<byte, ulong>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(bool))
            {
                var span = MemoryMarshal.Cast<byte, bool>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(char))
            {
                var span = MemoryMarshal.Cast<byte, char>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(float))
            {
                var span = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(double))
            {
                var span = MemoryMarshal.Cast<byte, double>(bytes.AsSpan());
                return new NDArray(span.ToArray());
            }
            else if (dtype == typeof(byte))
            {
                return new NDArray(bytes);
            }

            throw new NotImplementedException($"frombuffer: dtype {dtype} not supported.");
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
