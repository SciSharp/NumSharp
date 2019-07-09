using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace OOMath
{
    public partial class UnmanagedByteStorage<T>
    {
        internal const int ParallelAbove = 84999;

        [MethodImpl((MethodImplOptions)768)]
        public static unsafe UnmanagedByteStorage<T> operator +(UnmanagedByteStorage<T> lhs, UnmanagedByteStorage<T> rhs)
        {
            int lhsCount = lhs.Count > rhs.Count ? lhs.Count : rhs.Count;
            UnmanagedByteStorage<T> results;

            var isLeftScalar = lhs.Shape.IsScalar;
            var isRightScalar = rhs.Shape.IsScalar;
            if (isLeftScalar && isRightScalar)
            {
                //Both are a scalar

                #region Math

                switch (TypeCode)
                {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                    case TypeCode.#1: {
                        return Scalar((T) (ValueType) (*(#2*) lhs._arrayAddress + *(#2*) rhs._arrayAddress));
                    }
                %
                    default:
                        throw new NotSupportedException();
#else
                    case TypeCode.Byte:
                    {
                        return Scalar((T)(ValueType)(*(byte*)lhs._arrayAddress + *(byte*)rhs._arrayAddress));
                    }

                    case TypeCode.Int16:
                    {
                        return Scalar((T)(ValueType)(*(short*)lhs._arrayAddress + *(short*)rhs._arrayAddress));
                    }

                    case TypeCode.UInt16:
                    {
                        return Scalar((T)(ValueType)(*(ushort*)lhs._arrayAddress + *(ushort*)rhs._arrayAddress));
                    }

                    case TypeCode.Int32:
                    {
                        return Scalar((T)(ValueType)(*(int*)lhs._arrayAddress + *(int*)rhs._arrayAddress));
                    }

                    case TypeCode.UInt32:
                    {
                        return Scalar((T)(ValueType)(*(uint*)lhs._arrayAddress + *(uint*)rhs._arrayAddress));
                    }

                    case TypeCode.Int64:
                    {
                        return Scalar((T)(ValueType)(*(long*)lhs._arrayAddress + *(long*)rhs._arrayAddress));
                    }

                    case TypeCode.UInt64:
                    {
                        return Scalar((T)(ValueType)(*(ulong*)lhs._arrayAddress + *(ulong*)rhs._arrayAddress));
                    }

                    case TypeCode.Char:
                    {
                        return Scalar((T)(ValueType)(*(char*)lhs._arrayAddress + *(char*)rhs._arrayAddress));
                    }

                    case TypeCode.Double:
                    {
                        return Scalar((T)(ValueType)(*(double*)lhs._arrayAddress + *(double*)rhs._arrayAddress));
                    }

                    case TypeCode.Single:
                    {
                        return Scalar((T)(ValueType)(*(float*)lhs._arrayAddress + *(float*)rhs._arrayAddress));
                    }

                    case TypeCode.Decimal:
                    {
                        return Scalar((T)(ValueType)(*(decimal*)lhs._arrayAddress + *(decimal*)rhs._arrayAddress));
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
                    case TypeCode.#1: {
                        var resultStart = (#2*) results._arrayAddress;
                        var leftStart = (#2*) lhs._arrayAddress;
                        var rightStart = (#2*) rhs._arrayAddress;
                        if (lhsCount > ParallelLimit) {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2)(*(leftStart + i) + *(rightStart + i)); });
                        } else {
                            for (int i = 0; i < lhsCount; i++) {
                                *(resultStart + i) = (#2)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }
                %
                  
                    default:
                        throw new NotSupportedException();
#else


                    case TypeCode.Byte:
                    {
                        var resultStart = (byte*)results._arrayAddress;
                        var leftStart = (byte*)lhs._arrayAddress;
                        var rightStart = (byte*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int16:
                    {
                        var resultStart = (short*)results._arrayAddress;
                        var leftStart = (short*)lhs._arrayAddress;
                        var rightStart = (short*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results._arrayAddress;
                        var leftStart = (ushort*)lhs._arrayAddress;
                        var rightStart = (ushort*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int32:
                    {
                        var resultStart = (int*)results._arrayAddress;
                        var leftStart = (int*)lhs._arrayAddress;
                        var rightStart = (int*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt32:
                    {
                        var resultStart = (uint*)results._arrayAddress;
                        var leftStart = (uint*)lhs._arrayAddress;
                        var rightStart = (uint*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int64:
                    {
                        var resultStart = (long*)results._arrayAddress;
                        var leftStart = (long*)lhs._arrayAddress;
                        var rightStart = (long*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results._arrayAddress;
                        var leftStart = (ulong*)lhs._arrayAddress;
                        var rightStart = (ulong*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Char:
                    {
                        var resultStart = (char*)results._arrayAddress;
                        var leftStart = (char*)lhs._arrayAddress;
                        var rightStart = (char*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Double:
                    {
                        var resultStart = (double*)results._arrayAddress;
                        var leftStart = (double*)lhs._arrayAddress;
                        var rightStart = (double*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Single:
                    {
                        var resultStart = (float*)results._arrayAddress;
                        var leftStart = (float*)lhs._arrayAddress;
                        var rightStart = (float*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(*(leftStart + i) + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results._arrayAddress;
                        var leftStart = (decimal*)lhs._arrayAddress;
                        var rightStart = (decimal*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(*(leftStart + i) + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(*(leftStart + i) + *(rightStart + i));
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

            if (isRightScalar)
            {
                //Right is a scalar
                results = new UnmanagedByteStorage<T>(lhs._shape);
                switch (TypeCode)
                {
#if _REGEN
	                %foreach supported_numericals,supported_numericals_lowercase%
                        case TypeCode.#1: {
                            var resultStart = (#2*) results._arrayAddress;
                            var leftStart = (#2*) lhs._arrayAddress;
                            var rightScalar = *(#2*)rhs._arrayAddress;
                            if (lhsCount > ParallelLimit) {
                                Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2) (*(leftStart + i) + rightScalar); });
                            } else {
                                for (int i = 0; i < lhsCount; i++) {
                                    *(resultStart + i) = (#2) (*(leftStart + i) + rightScalar);
                                }
                            }

                            break;
                        }
                    %
                                      
                        default:
                            throw new NotSupportedException();
#else
                    case TypeCode.Byte:
                    {
                        var resultStart = (byte*)results._arrayAddress;
                        var leftStart = (byte*)lhs._arrayAddress;
                        var rightScalar = *(byte*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Int16:
                    {
                        var resultStart = (short*)results._arrayAddress;
                        var leftStart = (short*)lhs._arrayAddress;
                        var rightScalar = *(short*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results._arrayAddress;
                        var leftStart = (ushort*)lhs._arrayAddress;
                        var rightScalar = *(ushort*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Int32:
                    {
                        var resultStart = (int*)results._arrayAddress;
                        var leftStart = (int*)lhs._arrayAddress;
                        var rightScalar = *(int*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt32:
                    {
                        var resultStart = (uint*)results._arrayAddress;
                        var leftStart = (uint*)lhs._arrayAddress;
                        var rightScalar = *(uint*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Int64:
                    {
                        var resultStart = (long*)results._arrayAddress;
                        var leftStart = (long*)lhs._arrayAddress;
                        var rightScalar = *(long*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results._arrayAddress;
                        var leftStart = (ulong*)lhs._arrayAddress;
                        var rightScalar = *(ulong*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Char:
                    {
                        var resultStart = (char*)results._arrayAddress;
                        var leftStart = (char*)lhs._arrayAddress;
                        var rightScalar = *(char*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Double:
                    {
                        var resultStart = (double*)results._arrayAddress;
                        var leftStart = (double*)lhs._arrayAddress;
                        var rightScalar = *(double*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Single:
                    {
                        var resultStart = (float*)results._arrayAddress;
                        var leftStart = (float*)lhs._arrayAddress;
                        var rightScalar = *(float*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(*(leftStart + i) + rightScalar);
                            }
                        }

                        break;
                    }

                    case TypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results._arrayAddress;
                        var leftStart = (decimal*)lhs._arrayAddress;
                        var rightScalar = *(decimal*)rhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(*(leftStart + i) + rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(*(leftStart + i) + rightScalar);
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

            if (isLeftScalar)
            {
                //Left is a scalar
                results = new UnmanagedByteStorage<T>(rhs._shape);

                #region Math

                switch (TypeCode)
                {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                    case TypeCode.#1: {
                        var resultStart = (#2*) results._arrayAddress;
                        var rightStart = (#2*) rhs._arrayAddress;
                        var leftScalar = *(#2*) lhs._arrayAddress;
                        if (lhsCount > ParallelLimit) {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2)(leftScalar + *(rightStart + i)); });
                        } else {
                            for (int i = 0; i < lhsCount; i++) {
                                *(resultStart + i) = (#2)(leftScalar + *(rightStart + i));
                            }
                        }
                        break;
                    }
                %

                    default:
                        throw new NotSupportedException();
#else
                    case TypeCode.Byte:
                    {
                        var resultStart = (byte*)results._arrayAddress;
                        var rightStart = (byte*)rhs._arrayAddress;
                        var leftScalar = *(byte*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int16:
                    {
                        var resultStart = (short*)results._arrayAddress;
                        var rightStart = (short*)rhs._arrayAddress;
                        var leftScalar = *(short*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results._arrayAddress;
                        var rightStart = (ushort*)rhs._arrayAddress;
                        var leftScalar = *(ushort*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int32:
                    {
                        var resultStart = (int*)results._arrayAddress;
                        var rightStart = (int*)rhs._arrayAddress;
                        var leftScalar = *(int*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt32:
                    {
                        var resultStart = (uint*)results._arrayAddress;
                        var rightStart = (uint*)rhs._arrayAddress;
                        var leftScalar = *(uint*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Int64:
                    {
                        var resultStart = (long*)results._arrayAddress;
                        var rightStart = (long*)rhs._arrayAddress;
                        var leftScalar = *(long*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results._arrayAddress;
                        var rightStart = (ulong*)rhs._arrayAddress;
                        var leftScalar = *(ulong*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Char:
                    {
                        var resultStart = (char*)results._arrayAddress;
                        var rightStart = (char*)rhs._arrayAddress;
                        var leftScalar = *(char*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Double:
                    {
                        var resultStart = (double*)results._arrayAddress;
                        var rightStart = (double*)rhs._arrayAddress;
                        var leftScalar = *(double*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Single:
                    {
                        var resultStart = (float*)results._arrayAddress;
                        var rightStart = (float*)rhs._arrayAddress;
                        var leftScalar = *(float*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(leftScalar + *(rightStart + i));
                            }
                        }

                        break;
                    }

                    case TypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results._arrayAddress;
                        var rightStart = (decimal*)rhs._arrayAddress;
                        var leftScalar = *(decimal*)lhs._arrayAddress;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(leftScalar + *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(leftScalar + *(rightStart + i));
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

            throw new NotSupportedException();
        }

        //[MethodImpl((MethodImplOptions) 768)]
        //public static unsafe DArray<T> operator +(DArray<T> lhs, DArray<T> rhs) {
        //    int lhsCount = Math.Max(lhs.Count, rhs.Count);
        //    //this is a large array, use vectors.
        //    int numVectors = lhsCount / vectorSlots;
        //    int ceiling = numVectors * vectorSlots;

        //    DArray<T> results;
        //    var isLeftScalar = lhs.Shape.Size == 1;
        //    var isRightScalar = rhs.Shape.Size == 1;
        //    if (!isLeftScalar && !isRightScalar)
        //    {
        //        //None is a scalar is a scalar
        //        Debug.Assert(lhs.Shape == rhs.Shape);
        //        results = new DArray<T>(lhs._shape, false);
        //        Span<Vector<T>> resultsVecArray = CastVector(results);
        //        ReadOnlySpan<Vector<T>> rightVecArray = CastVector(rhs);
        //        ReadOnlySpan<Vector<T>> leftVecArray = CastVector(lhs);
        //        for (int i = 0; i < numVectors; i++)
        //        {
        //            resultsVecArray[i] = leftVecArray[i] + rightVecArray[i];
        //        }

        //        var resultAddr = results._arrayAddress;
        //        // Finish operation with any numbers leftover
        //        for (int i = ceiling; i < lhsCount; i++)
        //        {
        //            *(resultAddr + i) = (new Vector<T>(*(lhs._arrayAddress + i)) + new Vector<T>(*(rhs._arrayAddress + i)))[0];
        //        }
        //    }
        //    else if (isLeftScalar)
        //    {
        //        //Left is a scalar
        //        results = new DArray<T>(rhs._shape, false);
        //        Span<Vector<T>> resultsVecArray = CastVector(results);
        //        ReadOnlySpan<Vector<T>> rightVecArray = CastVector(rhs);
        //        var leftScalar = new Vector<T>(lhs.GetIndex(0));
        //        for (int i = 0; i < numVectors; i++)
        //        {
        //            resultsVecArray[i] = leftScalar + rightVecArray[i];
        //        }

        //        var resultAddr = results._arrayAddress;

        //        // Finish operation with any numbers leftover
        //        for (int i = ceiling; i < lhsCount; i++)
        //        {
        //            *(resultAddr + i) = (leftScalar + new Vector<T>(*(rhs._arrayAddress + i)))[0];
        //        }
        //    }
        //    else
        //    {
        //        //Right is a scalar
        //        results = new DArray<T>(lhs._shape, false);
        //        Span<Vector<T>> resultsVecArray = CastVector(results);
        //        ReadOnlySpan<Vector<T>> leftVecArray = CastVector(lhs);
        //        var rightScalar = new Vector<T>(rhs.GetIndex(0));
        //        for (int i = 0; i < numVectors; i++)
        //        {
        //            resultsVecArray[i] = leftVecArray[i] + rightScalar;
        //        }

        //        var resultAddr = results._arrayAddress;
        //        // Finish operation with any numbers leftover
        //        for (int i = ceiling; i < lhsCount; i++)
        //        {
        //            *(resultAddr + i) = (new Vector<T>(*(lhs._arrayAddress + i)) + rightScalar)[0];
        //        }
        //    }

        //    return results;
        //}
        [
            MethodImpl((MethodImplOptions)768)]
        public static UnmanagedByteStorage<T> operator +(UnmanagedByteStorage<T> lhs)
        {
            return lhs.Clone();
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedByteStorage<T> operator +(UnmanagedByteStorage<T> lhs, T rhs)
        {
            return lhs + Scalar(rhs);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedByteStorage<T> operator +(T lhs, UnmanagedByteStorage<T> rhs)
        {
            return Scalar(lhs) + rhs;
        }
    }
}
