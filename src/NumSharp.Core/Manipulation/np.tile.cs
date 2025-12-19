using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Utilities;


namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Construct an array by repeating A the number of times given by reps.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <param name="reps">The number of repetitions of A along each axis</param>
        /// <returns>The tiled output array</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html</remarks>
        public static NDArray tile(NDArray a, int[] repeats)
        {
            // Validate input
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            
            if (repeats == null)
                throw new ArgumentNullException(nameof(repeats));
            
            foreach (var rep in repeats)
                if (rep < 0)
                    throw new ArgumentException($"Negative repeat value: {rep}");

            int rlen = repeats.Length;
            int ndim = a.ndim;
            
            // Normalize repeats array to match the dimensionality of input array
            int[] normalizedRepeats;
            int[] aShape = a.shape;
            
            if (rlen < ndim)
            {
                // If reps has fewer dimensions than A, prepend ones
                normalizedRepeats = new int[ndim];
                for (int i = 0; i < ndim - rlen; i++)
                    normalizedRepeats[i] = 1;
                Array.Copy(repeats, 0, normalizedRepeats, ndim - rlen, rlen);
            }
            else if (rlen > ndim)
            {
                // If reps has more dimensions than A, prepend ones to A's shape
                normalizedRepeats = (int[])repeats.Clone();
                aShape = new int[rlen];
                for (int i = 0; i < rlen - ndim; i++)
                    aShape[i] = 1;
                Array.Copy(a.shape, 0, aShape, rlen - ndim, ndim);
            }
            else
            {
                normalizedRepeats = (int[])repeats.Clone();
            }

            // Calculate the shape of the output array
            int[] outShape = new int[normalizedRepeats.Length];
            for (int i = 0; i < normalizedRepeats.Length; i++)
            {
                outShape[i] = aShape[i] * normalizedRepeats[i];
            }

            // Create output array
            NDArray result = new NDArray(a.typecode, new Shape(outShape));
            
            // Handle scalar case
            if (a.size == 1)
            {
                // For scalar input, fill the entire result array with the scalar value
                unsafe
                {
                    switch (result.typecode)
                    {
                        case NPTypeCode.Boolean:
                            {
                                var value = a.MakeGeneric<bool>()[0];
                                var data = result.MakeGeneric<bool>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Byte:
                            {
                                var value = a.MakeGeneric<byte>()[0];
                                var data = result.MakeGeneric<byte>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Int16:
                            {
                                var value = a.MakeGeneric<short>()[0];
                                var data = result.MakeGeneric<short>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.UInt16:
                            {
                                var value = a.MakeGeneric<ushort>()[0];
                                var data = result.MakeGeneric<ushort>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Int32:
                            {
                                var value = a.MakeGeneric<int>()[0];
                                var data = result.MakeGeneric<int>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.UInt32:
                            {
                                var value = a.MakeGeneric<uint>()[0];
                                var data = result.MakeGeneric<uint>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Int64:
                            {
                                var value = a.MakeGeneric<long>()[0];
                                var data = result.MakeGeneric<long>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.UInt64:
                            {
                                var value = a.MakeGeneric<ulong>()[0];
                                var data = result.MakeGeneric<ulong>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Char:
                            {
                                var value = a.MakeGeneric<char>()[0];
                                var data = result.MakeGeneric<char>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Double:
                            {
                                var value = a.MakeGeneric<double>()[0];
                                var data = result.MakeGeneric<double>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Single:
                            {
                                var value = a.MakeGeneric<float>()[0];
                                var data = result.MakeGeneric<float>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        case NPTypeCode.Decimal:
                            {
                                var value = a.MakeGeneric<decimal>()[0];
                                var data = result.MakeGeneric<decimal>();
                                for (int i = 0; i < result.size; i++)
                                    data[i] = value;
                                break;
                            }
                        default:
                            throw new NotSupportedException($"Type {result.typecode} is not supported.");
                    }
                }
                return result;
            }

            // Use repeat function for each dimension
            NDArray current = a.Clone();
            
            // Start from the last dimension and work backwards
            for (int dim = normalizedRepeats.Length - 1; dim >= 0; dim--)
            {
                if (normalizedRepeats[dim] <= 1)
                    continue;
                    
                // For the current dimension, we need to repeat the array
                // We'll use a block-based approach for better performance
                
                // Calculate the size of each block to repeat
                int blockSize = 1;
                for (int i = dim + 1; i < normalizedRepeats.Length; i++)
                {
                    blockSize *= current.Shape.dimensions[i];
                }
                
                // Calculate the number of blocks
                int numBlocks = current.size / blockSize;
                
                // Create a temporary array for the repeated result
                int[] tempShape = (int[])current.Shape.dimensions.Clone();
                tempShape[dim] = tempShape[dim] * normalizedRepeats[dim];
                NDArray temp = new NDArray(current.typecode, new Shape(tempShape));
                
                // Copy and repeat blocks
                unsafe
                {
                    switch (current.typecode)
                    {
#if _REGEN
                        %foreach supported_dtypes,supported_dtypes_lowercase%
                        case NPTypeCode.#1:
                        {
                            var srcData = current.MakeGeneric<#2>();
                            var dstData = temp.MakeGeneric<#2>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                // Calculate the source position
                                int srcPos = block * blockSize;
                                
                                // Repeat the block normalizedRepeats[dim] times
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    
                                    // Copy the block
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        %
#else
                        case NPTypeCode.Boolean:
                        {
                            var srcData = current.MakeGeneric<bool>();
                            var dstData = temp.MakeGeneric<bool>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Byte:
                        {
                            var srcData = current.MakeGeneric<byte>();
                            var dstData = temp.MakeGeneric<byte>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Int16:
                        {
                            var srcData = current.MakeGeneric<short>();
                            var dstData = temp.MakeGeneric<short>();
                            int srcStride = current.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.UInt16:
                        {
                            var srcData = current.MakeGeneric<ushort>();
                            var dstData = temp.MakeGeneric<ushort>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Int32:
                        {
                            var srcData = current.MakeGeneric<int>();
                            var dstData = temp.MakeGeneric<int>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.UInt32:
                        {
                            var srcData = current.MakeGeneric<uint>();
                            var dstData = temp.MakeGeneric<uint>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Int64:
                        {
                            var srcData = current.MakeGeneric<long>();
                            var dstData = temp.MakeGeneric<long>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.UInt64:
                        {
                            var srcData = current.MakeGeneric<ulong>();
                            var dstData = temp.MakeGeneric<ulong>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Char:
                        {
                            var srcData = current.MakeGeneric<char>();
                            var dstData = temp.MakeGeneric<char>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Double:
                        {
                            var srcData = current.MakeGeneric<double>();
                            var dstData = temp.MakeGeneric<double>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Single:
                        {
                            var srcData = current.MakeGeneric<float>();
                            var dstData = temp.MakeGeneric<float>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        case NPTypeCode.Decimal:
                        {
                            var srcData = current.MakeGeneric<decimal>();
                            var dstData = temp.MakeGeneric<decimal>();
                            int srcStride = current.Shape.strides[dim];
                            int dstStride = temp.Shape.strides[dim];
                            
                            for (int block = 0; block < numBlocks; block++)
                            {
                                int srcPos = block * blockSize;
                                for (int repeat = 0; repeat < normalizedRepeats[dim]; repeat++)
                                {
                                    int dstPos = block * blockSize * normalizedRepeats[dim] + repeat * blockSize;
                                    for (int i = 0; i < blockSize; i++)
                                    {
                                        dstData[dstPos + i] = srcData[srcPos + i];
                                    }
                                }
                            }
                            break;
                        }
                        default:
                            throw new NotSupportedException($"Type {current.typecode} is not supported.");
#endif
                    }
                }
                
                current = temp;
            }
            
            return current;
        }
    }
}
