using System;
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
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase,supported_dtypes_defaultvals%
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

	            case NPTypeCode.Int32:
	            {
                    slice = new ArraySlice<int>(buffer == null ? new UnmanagedMemoryBlock<int>(shape.size, 0) : UnmanagedMemoryBlock<int>.FromArray((int[])buffer));
                    break;
	            }

	            case NPTypeCode.Int64:
	            {
                    slice = new ArraySlice<long>(buffer == null ? new UnmanagedMemoryBlock<long>(shape.size, 0L) : UnmanagedMemoryBlock<long>.FromArray((long[])buffer));
                    break;
	            }

	            case NPTypeCode.Single:
	            {
                    slice = new ArraySlice<float>(buffer == null ? new UnmanagedMemoryBlock<float>(shape.size, 0f) : UnmanagedMemoryBlock<float>.FromArray((float[])buffer));
                    break;
	            }

	            case NPTypeCode.Double:
	            {
                    slice = new ArraySlice<double>(buffer == null ? new UnmanagedMemoryBlock<double>(shape.size, 0d) : UnmanagedMemoryBlock<double>.FromArray((double[])buffer));
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
#if _REGEN1
	                %foreach supported_dtypes,supported_dtypes_lowercase,supported_dtypes_defaultvals%
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

	                case NPTypeCode.Int32:
	                {
                        buffer = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(shape.size, 0));
                        break;
	                }

	                case NPTypeCode.Int64:
	                {
                        buffer = new ArraySlice<long>(new UnmanagedMemoryBlock<long>(shape.size, 0L));
                        break;
	                }

	                case NPTypeCode.Single:
	                {
                        buffer = new ArraySlice<float>(new UnmanagedMemoryBlock<float>(shape.size, 0f));
                        break;
	                }

	                case NPTypeCode.Double:
	                {
                        buffer = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(shape.size, 0d));
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
