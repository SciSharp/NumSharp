using System;

namespace NumSharp
{
    public partial class NDArray
    {
        public byte[] ToByteArray()
        {
            if (size == 0)
                return System.Array.Empty<byte>();

            unsafe
            {
                var addr = Storage.Address;
                var len = Storage.InternalArray.BytesLength;

                byte[] bytes = new byte[len];
                fixed (byte* @out = bytes) 
                    Buffer.MemoryCopy(addr, @out, len, len);
                return bytes;
            }
        }
    }
}
