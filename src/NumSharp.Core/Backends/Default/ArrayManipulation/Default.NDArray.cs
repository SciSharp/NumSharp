using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray CreateNDArray(Shape shape, Type dtype = null, Array buffer = null, char order = 'C')
        {
            if (dtype == null)
                dtype = np.float32;

            IArraySlice slice;

            switch (dtype.GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase,supported_currently_supported_defaultvals%
	            case NPTypeCode.#1:
	            {
                    slice = new ArraySlice<#2>(buffer == null ? new UnmanagedMemoryBlock<#2>(shape.size, #3) : UnmanagedMemoryBlock<#2>.FromArray((#2[])buffer));
                    break;
	            }

	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
	            {
                    slice = new ArraySlice<bool>(buffer == null ? new UnmanagedMemoryBlock<bool>(shape.size, false) : UnmanagedMemoryBlock<bool>.FromArray((bool[])buffer));
                    break;
	            }

	            case NPTypeCode.Byte:
	            {
                    slice = new ArraySlice<byte>(buffer == null ? new UnmanagedMemoryBlock<byte>(shape.size, 0) : UnmanagedMemoryBlock<byte>.FromArray((byte[])buffer));
                    break;
	            }

	            case NPTypeCode.Int16:
	            {
                    slice = new ArraySlice<short>(buffer == null ? new UnmanagedMemoryBlock<short>(shape.size, 0) : UnmanagedMemoryBlock<short>.FromArray((short[])buffer));
                    break;
	            }

	            case NPTypeCode.UInt16:
	            {
                    slice = new ArraySlice<ushort>(buffer == null ? new UnmanagedMemoryBlock<ushort>(shape.size, 0) : UnmanagedMemoryBlock<ushort>.FromArray((ushort[])buffer));
                    break;
	            }

	            case NPTypeCode.Int32:
	            {
                    slice = new ArraySlice<int>(buffer == null ? new UnmanagedMemoryBlock<int>(shape.size, 0) : UnmanagedMemoryBlock<int>.FromArray((int[])buffer));
                    break;
	            }

	            case NPTypeCode.UInt32:
	            {
                    slice = new ArraySlice<uint>(buffer == null ? new UnmanagedMemoryBlock<uint>(shape.size, 0u) : UnmanagedMemoryBlock<uint>.FromArray((uint[])buffer));
                    break;
	            }

	            case NPTypeCode.Int64:
	            {
                    slice = new ArraySlice<long>(buffer == null ? new UnmanagedMemoryBlock<long>(shape.size, 0L) : UnmanagedMemoryBlock<long>.FromArray((long[])buffer));
                    break;
	            }

	            case NPTypeCode.UInt64:
	            {
                    slice = new ArraySlice<ulong>(buffer == null ? new UnmanagedMemoryBlock<ulong>(shape.size, 0UL) : UnmanagedMemoryBlock<ulong>.FromArray((ulong[])buffer));
                    break;
	            }

	            case NPTypeCode.Char:
	            {
                    slice = new ArraySlice<char>(buffer == null ? new UnmanagedMemoryBlock<char>(shape.size, '\0') : UnmanagedMemoryBlock<char>.FromArray((char[])buffer));
                    break;
	            }

	            case NPTypeCode.Double:
	            {
                    slice = new ArraySlice<double>(buffer == null ? new UnmanagedMemoryBlock<double>(shape.size, 0d) : UnmanagedMemoryBlock<double>.FromArray((double[])buffer));
                    break;
	            }

	            case NPTypeCode.Single:
	            {
                    slice = new ArraySlice<float>(buffer == null ? new UnmanagedMemoryBlock<float>(shape.size, 0f) : UnmanagedMemoryBlock<float>.FromArray((float[])buffer));
                    break;
	            }

	            case NPTypeCode.Decimal:
	            {
                    slice = new ArraySlice<decimal>(buffer == null ? new UnmanagedMemoryBlock<decimal>(shape.size, 0m) : UnmanagedMemoryBlock<decimal>.FromArray((decimal[])buffer));
                    break;
	            }

	            default:
		            throw new NotSupportedException();
#endif
            }

            return new NDArray(slice, shape: shape, order: order);
        }

        public override NDArray CreateNDArray(Shape shape, Type dtype = null, IArraySlice buffer = null, char order = 'C')
        {
            if (dtype == null)
                dtype = np.float32;

            if (buffer == null)
                switch (dtype.GetTypeCode())
                {
#if _REGEN
	                %foreach supported_currently_supported,supported_currently_supported_lowercase,supported_currently_supported_defaultvals%
	                case NPTypeCode.#1:
	                {
                        buffer = new ArraySlice<#2>(new UnmanagedMemoryBlock<#2>(shape.size, #3));
                        break;
	                }

	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean:
	                {
                        buffer = new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>(shape.size, false));
                        break;
	                }

	                case NPTypeCode.Byte:
	                {
                        buffer = new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(shape.size, 0));
                        break;
	                }

	                case NPTypeCode.Int16:
	                {
                        buffer = new ArraySlice<short>(new UnmanagedMemoryBlock<short>(shape.size, 0));
                        break;
	                }

	                case NPTypeCode.UInt16:
	                {
                        buffer = new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>(shape.size, 0));
                        break;
	                }

	                case NPTypeCode.Int32:
	                {
                        buffer = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(shape.size, 0));
                        break;
	                }

	                case NPTypeCode.UInt32:
	                {
                        buffer = new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>(shape.size, 0u));
                        break;
	                }

	                case NPTypeCode.Int64:
	                {
                        buffer = new ArraySlice<long>(new UnmanagedMemoryBlock<long>(shape.size, 0L));
                        break;
	                }

	                case NPTypeCode.UInt64:
	                {
                        buffer = new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(shape.size, 0UL));
                        break;
	                }

	                case NPTypeCode.Char:
	                {
                        buffer = new ArraySlice<char>(new UnmanagedMemoryBlock<char>(shape.size, '\0'));
                        break;
	                }

	                case NPTypeCode.Double:
	                {
                        buffer = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(shape.size, 0d));
                        break;
	                }

	                case NPTypeCode.Single:
	                {
                        buffer = new ArraySlice<float>(new UnmanagedMemoryBlock<float>(shape.size, 0f));
                        break;
	                }

	                case NPTypeCode.Decimal:
	                {
                        buffer = new ArraySlice<decimal>(new UnmanagedMemoryBlock<decimal>(shape.size, 0m));
                        break;
	                }

	                default:
		                throw new NotSupportedException();
#endif
                }

            return new NDArray(buffer, shape: shape, order: order);
        }
    }
}
