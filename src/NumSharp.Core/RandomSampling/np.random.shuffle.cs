using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Modify a sequence in-place by shuffling its contents.
        /// </summary>
        /// <param name="x">The array or list to be shuffled.</param>
        /// <param name="passes">How many times to pass all items in a complexity of O(n*passes)</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.random.shuffle.html <br></br>Does not copy <paramref name="x"/></remarks>
        [SuppressMessage("ReSharper", "TooWideLocalVariableScope")]
        public void shuffle(NDArray x, int passes = 2)
        {
            var size = x.size;
            var count = size * 2;
            Func<int, int> transformOffset = x.Shape.TransformOffset;
            unsafe
            {
#if _REGEN
                #region Compute
		        switch (x.typecode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1:
			        {
				        var addr = (#2*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        |#2 tmp; //index 0
                        |#2* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }
                        break;
			        }
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute

                switch (x.typecode)
                {
                    case NPTypeCode.Boolean:
                    {
                        var addr = (bool*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        bool tmp; //index 0
                        bool* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Byte:
                    {
                        var addr = (byte*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        byte tmp; //index 0
                        byte* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var addr = (short*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        short tmp; //index 0
                        short* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var addr = (ushort*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        ushort tmp; //index 0
                        ushort* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var addr = (int*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        int tmp; //index 0
                        int* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var addr = (uint*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        uint tmp; //index 0
                        uint* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var addr = (long*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        long tmp; //index 0
                        long* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var addr = (ulong*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        ulong tmp; //index 0
                        ulong* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var addr = (char*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        char tmp; //index 0
                        char* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var addr = (double*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        double tmp; //index 0
                        double* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var addr = (float*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        float tmp; //index 0
                        float* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var addr = (decimal*)x.Address;
                        var addr_index0 = addr + transformOffset(0);
                        decimal tmp; //index 0
                        decimal* addr_swap;
                        while (count-- > 1)
                        {
                            tmp = *addr_index0;
                            addr_swap = addr + transformOffset(randomizer.Next(size));
                            *addr_index0 = *addr_swap;
                            *addr_swap = tmp;
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
                }

                #endregion

#endif
            }
        }
    }
}
