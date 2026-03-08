using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        public unsafe NDArray this[NDArray<bool> mask]
        {
            get
            {
                // NumPy requires mask size to match array size
                if (mask.size != this.size)
                    throw new IndexOutOfRangeException(
                        $"boolean index did not match indexed array along axis 0; " +
                        $"size of axis is {this.size} but size of boolean index is {mask.size}");

                // SIMD fast path: contiguous arrays of same size
                var kp = DefaultEngine.DefaultKernelProvider;
                if (kp.Enabled && kp.VectorBits > 0 &&
                    mask.Shape.IsContiguous && this.Shape.IsContiguous &&
                    mask.size == this.size)
                {
                    return BooleanMaskFastPath(mask);
                }

                // Fallback: use nonzero + fancy indexing
                return FetchIndices(this, np.nonzero(mask), null, true);
            }
            set
            {
                throw new NotImplementedException("Setter is not implemented yet");
            }
        }

        /// <summary>
        /// SIMD-optimized boolean masking for contiguous arrays.
        /// </summary>
        private unsafe NDArray BooleanMaskFastPath(NDArray<bool> mask)
        {
            int size = this.size;
            var kp = DefaultEngine.DefaultKernelProvider;

            // Count true values using SIMD
            int trueCount = kp.CountTrue((bool*)mask.Address, size);

            if (trueCount == 0)
                return new NDArray(this.dtype, Shape.Empty(1)); // Empty 1D result

            // Create result array
            var result = new NDArray(this.dtype, new Shape(trueCount));

            // Copy elements where mask is true
            switch (this.typecode)
            {
                case NPTypeCode.Boolean:
                    kp.CopyMasked((bool*)this.Address, (bool*)mask.Address, (bool*)result.Address, size);
                    break;
                case NPTypeCode.Byte:
                    kp.CopyMasked((byte*)this.Address, (bool*)mask.Address, (byte*)result.Address, size);
                    break;
                case NPTypeCode.Int16:
                    kp.CopyMasked((short*)this.Address, (bool*)mask.Address, (short*)result.Address, size);
                    break;
                case NPTypeCode.UInt16:
                    kp.CopyMasked((ushort*)this.Address, (bool*)mask.Address, (ushort*)result.Address, size);
                    break;
                case NPTypeCode.Int32:
                    kp.CopyMasked((int*)this.Address, (bool*)mask.Address, (int*)result.Address, size);
                    break;
                case NPTypeCode.UInt32:
                    kp.CopyMasked((uint*)this.Address, (bool*)mask.Address, (uint*)result.Address, size);
                    break;
                case NPTypeCode.Int64:
                    kp.CopyMasked((long*)this.Address, (bool*)mask.Address, (long*)result.Address, size);
                    break;
                case NPTypeCode.UInt64:
                    kp.CopyMasked((ulong*)this.Address, (bool*)mask.Address, (ulong*)result.Address, size);
                    break;
                case NPTypeCode.Char:
                    kp.CopyMasked((char*)this.Address, (bool*)mask.Address, (char*)result.Address, size);
                    break;
                case NPTypeCode.Single:
                    kp.CopyMasked((float*)this.Address, (bool*)mask.Address, (float*)result.Address, size);
                    break;
                case NPTypeCode.Double:
                    kp.CopyMasked((double*)this.Address, (bool*)mask.Address, (double*)result.Address, size);
                    break;
                case NPTypeCode.Decimal:
                    kp.CopyMasked((decimal*)this.Address, (bool*)mask.Address, (decimal*)result.Address, size);
                    break;
                default:
                    throw new NotSupportedException($"Type {this.typecode} not supported for boolean masking");
            }

            return result;
        }
    }
}
