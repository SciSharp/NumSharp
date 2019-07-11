using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy Signature: numpy.fromfile(file, dtype=float, count=-1, sep='')
        public static NDArray fromfile(string fileName, Type dtype)
        {
            unsafe
            {
                var bytes = File.ReadAllBytes(fileName);
                switch (dtype.GetTypeCode())
                {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase%
	                case NPTypeCode.#1:
	                {
                        return new NDArray(new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromBuffer(bytes, false)));
	                }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        return new NDArray(new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Int16:
	                {
                        return new NDArray(new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.UInt16:
	                {
                        return new NDArray(new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Int32:
	                {
                        return new NDArray(new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.UInt32:
	                {
                        return new NDArray(new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Int64:
	                {
                        return new NDArray(new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.UInt64:
	                {
                        return new NDArray(new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Char:
	                {
                        return new NDArray(new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Double:
	                {
                        return new NDArray(new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Single:
	                {
                        return new NDArray(new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromBuffer(bytes, false)));
	                }
	                case NPTypeCode.Decimal:
	                {
                        return new NDArray(new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromBuffer(bytes, false)));
	                }
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
        }
    }
}
