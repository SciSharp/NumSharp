using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NumSharp.Backends.Unmanaged
{
    public partial class UnmanagedByteStorage<T>
    {
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedByteStorage<T> operator -(UnmanagedByteStorage<T> rhs)
        {
            return Scalar(default) - rhs;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static unsafe UnmanagedByteStorage<T> operator -(UnmanagedByteStorage<T> lhs, UnmanagedByteStorage<T> rhs)
        {
            int lhsCount = lhs.Count > rhs.Count ? lhs.Count : rhs.Count;
            UnmanagedByteStorage<T> results;
            var isLeftScalar = lhs.Shape.IsScalar;
            var isRightScalar = rhs.Shape.IsScalar;
            if (isLeftScalar && isRightScalar)
            {
                //Both are a scalar
                results = new UnmanagedByteStorage<T>(rhs._shape);

                #region Math

                switch (TypeCode)
                {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                    case NPTypeCode.#1: 
                    {
                        return Scalar((T) (ValueType) (*(#2*) lhs._arrayAddress - *(#2*) rhs._arrayAddress));
                    }
                %
                    default:
                        throw new NotSupportedException();
#else
                    case NPTypeCode.Byte:
                    {
                        return Scalar((T)(ValueType)(*(byte*)lhs.Address - *(byte*)rhs.Address));
                    }

                    case NPTypeCode.Int16:
                    {
                        return Scalar((T)(ValueType)(*(short*)lhs.Address - *(short*)rhs.Address));
                    }

                    case NPTypeCode.UInt16:
                    {
                        return Scalar((T)(ValueType)(*(ushort*)lhs.Address - *(ushort*)rhs.Address));
                    }

                    case NPTypeCode.Int32:
                    {
                        return Scalar((T)(ValueType)(*(int*)lhs.Address - *(int*)rhs.Address));
                    }

                    case NPTypeCode.UInt32:
                    {
                        return Scalar((T)(ValueType)(*(uint*)lhs.Address - *(uint*)rhs.Address));
                    }

                    case NPTypeCode.Int64:
                    {
                        return Scalar((T)(ValueType)(*(long*)lhs.Address - *(long*)rhs.Address));
                    }

                    case NPTypeCode.UInt64:
                    {
                        return Scalar((T)(ValueType)(*(ulong*)lhs.Address - *(ulong*)rhs.Address));
                    }

                    case NPTypeCode.Char:
                    {
                        return Scalar((T)(ValueType)(*(char*)lhs.Address - *(char*)rhs.Address));
                    }

                    case NPTypeCode.Double:
                    {
                        return Scalar((T)(ValueType)(*(double*)lhs.Address - *(double*)rhs.Address));
                    }

                    case NPTypeCode.Single:
                    {
                        return Scalar((T)(ValueType)(*(float*)lhs.Address - *(float*)rhs.Address));
                    }

                    case NPTypeCode.Decimal:
                    {
                        return Scalar((T)(ValueType)(*(decimal*)lhs.Address - *(decimal*)rhs.Address));
                    }

                    default:
                        throw new NotSupportedException();
#endif

                    #endregion
                }
            }

            if (!isLeftScalar && !isRightScalar)
            {
                //None is a scalar is a scalar
                Debug.Assert(lhs.Shape == rhs.Shape);
                results = new UnmanagedByteStorage<T>(lhs._shape);

                #region Math

                switch (TypeCode)
                {
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                    case NPTypeCode.#1: {
                        var resultStart = (#2*) results._arrayAddress;
                        var leftStart = (#2*) lhs._arrayAddress;
                        var rightStart = (#2*) rhs._arrayAddress;
                        if (lhsCount > ParallelLimit) {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2)(*(leftStart + i) - *(rightStart + i)); });
                        } else {
                            for (int i = 0; i < lhsCount; i++) {
                                *(resultStart + i) = (#2)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }
                %
                  
                    default:
                        throw new NotSupportedException();
#else


                    case NPTypeCode.Byte:
                    {
                        var resultStart = (byte*)results.Address;
                        var leftStart = (byte*)lhs.Address;
                        var rightStart = (byte*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var resultStart = (short*)results.Address;
                        var leftStart = (short*)lhs.Address;
                        var rightStart = (short*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results.Address;
                        var leftStart = (ushort*)lhs.Address;
                        var rightStart = (ushort*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var resultStart = (int*)results.Address;
                        var leftStart = (int*)lhs.Address;
                        var rightStart = (int*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var resultStart = (uint*)results.Address;
                        var leftStart = (uint*)lhs.Address;
                        var rightStart = (uint*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var resultStart = (long*)results.Address;
                        var leftStart = (long*)lhs.Address;
                        var rightStart = (long*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results.Address;
                        var leftStart = (ulong*)lhs.Address;
                        var rightStart = (ulong*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var resultStart = (char*)results.Address;
                        var leftStart = (char*)lhs.Address;
                        var rightStart = (char*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var resultStart = (double*)results.Address;
                        var leftStart = (double*)lhs.Address;
                        var rightStart = (double*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var resultStart = (float*)results.Address;
                        var leftStart = (float*)lhs.Address;
                        var rightStart = (float*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results.Address;
                        var leftStart = (decimal*)lhs.Address;
                        var rightStart = (decimal*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(*(leftStart + i) - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(*(leftStart + i) - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
#endif
                }

                #endregion

                return results;
            }

            if (isLeftScalar)
            {
                //Left is a scalar
                results = new UnmanagedByteStorage<T>(rhs._shape);

                #region Math

                switch (TypeCode)
                {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                    case NPTypeCode.#1: {
                        var resultStart = (#2*) results._arrayAddress;
                        var rightStart = (#2*) rhs._arrayAddress;
                        var leftScalar = *(#2*) lhs._arrayAddress;
                        if (lhsCount > ParallelLimit) {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2)(leftScalar - *(rightStart + i)); });
                        } else {
                            for (int i = 0; i < lhsCount; i++) {
                                *(resultStart + i) = (#2)(leftScalar - *(rightStart + i));
                            }
                        }
                        break;
                    }
                %

                    default:
                        throw new NotSupportedException();
#else
                    case NPTypeCode.Byte:
                    {
                        var resultStart = (byte*)results.Address;
                        var rightStart = (byte*)rhs.Address;
                        var leftScalar = *(byte*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var resultStart = (short*)results.Address;
                        var rightStart = (short*)rhs.Address;
                        var leftScalar = *(short*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results.Address;
                        var rightStart = (ushort*)rhs.Address;
                        var leftScalar = *(ushort*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var resultStart = (int*)results.Address;
                        var rightStart = (int*)rhs.Address;
                        var leftScalar = *(int*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var resultStart = (uint*)results.Address;
                        var rightStart = (uint*)rhs.Address;
                        var leftScalar = *(uint*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var resultStart = (long*)results.Address;
                        var rightStart = (long*)rhs.Address;
                        var leftScalar = *(long*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results.Address;
                        var rightStart = (ulong*)rhs.Address;
                        var leftScalar = *(ulong*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var resultStart = (char*)results.Address;
                        var rightStart = (char*)rhs.Address;
                        var leftScalar = *(char*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var resultStart = (double*)results.Address;
                        var rightStart = (double*)rhs.Address;
                        var leftScalar = *(double*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var resultStart = (float*)results.Address;
                        var rightStart = (float*)rhs.Address;
                        var leftScalar = *(float*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results.Address;
                        var rightStart = (decimal*)rhs.Address;
                        var leftScalar = *(decimal*)lhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(leftScalar - *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(leftScalar - *(rightStart + i));
                            }
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
#endif

                    #endregion
                }

                return results;
            }

            if (isRightScalar)
            {
//Right is a scalar
                results = new UnmanagedByteStorage<T>(lhs._shape);
                switch (TypeCode)
                {
#if _REGEN
	                %foreach supported_numericals,supported_numericals_lowercase%
                        case NPTypeCode.#1: {
                            var resultStart = (#2*) results._arrayAddress;
                            var leftStart = (#2*) lhs._arrayAddress;
                            var rightScalar = *(#2*)rhs._arrayAddress;
                            if (lhsCount > ParallelLimit) {
                                Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2) (*(leftStart + i) - rightScalar); });
                            } else {
                                for (int i = 0; i < lhsCount; i++) {
                                    *(resultStart + i) = (#2) (*(leftStart + i) - rightScalar);
                                }
                            }

                            break;
                        }
                    %
                                      
                        default:
                            throw new NotSupportedException();
#else
                    case NPTypeCode.Byte:
                    {
                        var resultStart = (byte*)results.Address;
                        var leftStart = (byte*)lhs.Address;
                        var rightScalar = *(byte*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var resultStart = (short*)results.Address;
                        var leftStart = (short*)lhs.Address;
                        var rightScalar = *(short*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results.Address;
                        var leftStart = (ushort*)lhs.Address;
                        var rightScalar = *(ushort*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var resultStart = (int*)results.Address;
                        var leftStart = (int*)lhs.Address;
                        var rightScalar = *(int*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var resultStart = (uint*)results.Address;
                        var leftStart = (uint*)lhs.Address;
                        var rightScalar = *(uint*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var resultStart = (long*)results.Address;
                        var leftStart = (long*)lhs.Address;
                        var rightScalar = *(long*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results.Address;
                        var leftStart = (ulong*)lhs.Address;
                        var rightScalar = *(ulong*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var resultStart = (char*)results.Address;
                        var leftStart = (char*)lhs.Address;
                        var rightScalar = *(char*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var resultStart = (double*)results.Address;
                        var leftStart = (double*)lhs.Address;
                        var rightScalar = *(double*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var resultStart = (float*)results.Address;
                        var leftStart = (float*)lhs.Address;
                        var rightScalar = *(float*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results.Address;
                        var leftStart = (decimal*)lhs.Address;
                        var rightScalar = *(decimal*)rhs.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(*(leftStart + i) - rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(*(leftStart + i) - rightScalar);
                            }
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
#endif
                }

                return results;
            }

            throw new NotSupportedException();
        }
    }
}
