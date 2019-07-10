using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NumSharp.Backends.Unmanaged
{
    public partial class UnmanagedByteStorage<T>
    {
        public static (UnmanagedByteStorage<T> arrOne, UnmanagedByteStorage<T> arrTwo) Broadcast(UnmanagedByteStorage<T> arrOne, UnmanagedByteStorage<T> arrTwo)
        {
            //PyArrayMultiIterObject *mit
            int i, nd, k, j;
            int tmp;
            Shape it;
            var iters = new[] {arrOne, arrTwo};
            var itersLength = iters.Length;

            /* Discover the broadcast number of dimensions */
            //Gets the largest ndim of all iterators
            for (i = 0, nd = 0; i < itersLength; i++)
            {
                nd = Math.Max(nd, iters[i]._shape.NDim);
            }

            //this is the shared shape aka the target broadcast
            var mit = Shape.Empty(nd);

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit.Dimensions[i] = 1;
                for (j = 0; j < itersLength; j++)
                {
                    it = iters[j]._shape;
                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + it.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = it.Dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit.Dimensions[i] == 1)
                        {
                            mit.Dimensions[i] = tmp;
                        }
                        else if (mit.Dimensions[i] != tmp)
                        {
                            throw new Exception("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                        }
                    }
                }
            }


            /*
             * Reset the iterator dimensions and strides of each iterator
             * object -- using 0 valued strides for broadcasting
             * Need to check for overflow
             */

            tmp = 1;
            for (i = 0; i < mit.NDim; i++)
            {
                tmp += tmp * mit.Dimensions[i];
            }

            mit.size = tmp;
            var retiters = new[] {arrOne.CreateAlias(mit), arrTwo.CreateAlias(mit)};


            for (i = 0; i < itersLength; i++)
            {
                var ogiter = iters[i];
                var iter = retiters[i];
                it = iter._shape;
                nd = ogiter._shape.NDim;
                it.size = tmp;
                //todo if (nd != 0)
                //todo {
                //todo     it->factors[mit.nd - 1] = 1;
                //todo }
                for (j = 0; j < mit.NDim; j++)
                {
                    //todo it->dims_m1[j] = mit.dimensions[j] - 1;
                    k = j + nd - mit.NDim;
                    /*
                     * If this dimension was added or shape of
                     * underlying array was 1
                     */
                    if ((k < 0) ||
                        ogiter._shape.dimensions[k] != mit.dimensions[j])
                    {
                        it.layout = 'C';
                        it.strides[j] = 0;
                    }
                    else
                    {
                        it.strides[j] = ogiter._shape.strides[k];
                    }

                    //todo it.backstrides[j] = it.strides[j] * (it.dimensions[j] - 1);
                    //todo if (j > 0)
                    //todo     it.factors[mit.NDim - j - 1] = it.factors[mit.NDim - j] * mit.dimensions[mit.NDim - j];
                }
            }

            return (retiters[0], retiters[1]);
        }


        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedByteStorage<T> operator *(UnmanagedByteStorage<T> lhs, UnmanagedByteStorage<T> rhs)
        {
            return Multiply(lhs, rhs);
        }

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html</remarks>
        public static unsafe UnmanagedByteStorage<T> Multiply(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            UnmanagedByteStorage<T> results;
            var leftshape = left._shape;
            var rightshape = right._shape;
            var isLeftScalar = leftshape.IsScalar;
            var isRightScalar = rightshape.IsScalar;

            if (isLeftScalar && isRightScalar)
            {
                return MultiplyScalar(left, right);
            }

            if (isRightScalar)
            {
                //Right is a scalar
                int lhsCount = left.Count > right.Count ? left.Count : right.Count;
                results = new UnmanagedByteStorage<T>(leftshape);
                switch (TypeCode)
                {
#if _REGEN
	                %foreach supported_numericals,supported_numericals_lowercase%
                        case NPTypeCode.#1: 
                        {
                            var resultStart = (#2*) results._arrayAddress;
                            var leftStart = (#2*) left._arrayAddress;
                            var rightScalar = *(#2*)right._arrayAddress;
                            if (lhsCount > ParallelLimit) {
                                Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2) (*(leftStart + i) * rightScalar); });
                            } else {
                                for (int i = 0; i < lhsCount; i++) {
                                    *(resultStart + i) = (#2) (*(leftStart + i) * rightScalar);
                                }
                            }

                            return results;
                        }
                    %
                                      
                        default:
                            throw new NotSupportedException();
#else
                    case NPTypeCode.Byte:
                    {
                        var resultStart = (byte*)results.Address;
                        var leftStart = (byte*)left.Address;
                        var rightScalar = *(byte*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (byte)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Int16:
                    {
                        var resultStart = (short*)results.Address;
                        var leftStart = (short*)left.Address;
                        var rightScalar = *(short*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (short)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results.Address;
                        var leftStart = (ushort*)left.Address;
                        var rightScalar = *(ushort*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ushort)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Int32:
                    {
                        var resultStart = (int*)results.Address;
                        var leftStart = (int*)left.Address;
                        var rightScalar = *(int*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (int)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var resultStart = (uint*)results.Address;
                        var leftStart = (uint*)left.Address;
                        var rightScalar = *(uint*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (uint)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Int64:
                    {
                        var resultStart = (long*)results.Address;
                        var leftStart = (long*)left.Address;
                        var rightScalar = *(long*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (long)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results.Address;
                        var leftStart = (ulong*)left.Address;
                        var rightScalar = *(ulong*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (ulong)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Char:
                    {
                        var resultStart = (char*)results.Address;
                        var leftStart = (char*)left.Address;
                        var rightScalar = *(char*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (char)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Double:
                    {
                        var resultStart = (double*)results.Address;
                        var leftStart = (double*)left.Address;
                        var rightScalar = *(double*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (double)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Single:
                    {
                        var resultStart = (float*)results.Address;
                        var leftStart = (float*)left.Address;
                        var rightScalar = *(float*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (float)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results.Address;
                        var leftStart = (decimal*)left.Address;
                        var rightScalar = *(decimal*)right.Address;
                        if (lhsCount > ParallelLimit)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(*(leftStart + i) * rightScalar); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++)
                            {
                                *(resultStart + i) = (decimal)(*(leftStart + i) * rightScalar);
                            }
                        }

                        return results;
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
                results = new UnmanagedByteStorage<T>(rightshape);
                int lhsCount = left.Count > right.Count ? left.Count : right.Count;

                #region Math

                switch (TypeCode)
                {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                    case NPTypeCode.#1: {
                        var resultStart = (#2*) results._arrayAddress;
                        var rightStart = (#2*) right._arrayAddress;
                        var leftScalar = *(#2*) left._arrayAddress;
                        if (lhsCount > ParallelAbove) {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (#2)(leftScalar * *(rightStart + i)); });
                        } else {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++) {
                                *resultStart = (#2)(leftScalar * *rightStart);
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
                        var rightStart = (byte*)right.Address;
                        var leftScalar = *(byte*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (byte)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (byte)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var resultStart = (short*)results.Address;
                        var rightStart = (short*)right.Address;
                        var leftScalar = *(short*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (short)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (short)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var resultStart = (ushort*)results.Address;
                        var rightStart = (ushort*)right.Address;
                        var leftScalar = *(ushort*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ushort)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (ushort)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var resultStart = (int*)results.Address;
                        var rightStart = (int*)right.Address;
                        var leftScalar = *(int*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (int)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (int)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var resultStart = (uint*)results.Address;
                        var rightStart = (uint*)right.Address;
                        var leftScalar = *(uint*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (uint)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (uint)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var resultStart = (long*)results.Address;
                        var rightStart = (long*)right.Address;
                        var leftScalar = *(long*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (long)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (long)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var resultStart = (ulong*)results.Address;
                        var rightStart = (ulong*)right.Address;
                        var leftScalar = *(ulong*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (ulong)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (ulong)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var resultStart = (char*)results.Address;
                        var rightStart = (char*)right.Address;
                        var leftScalar = *(char*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (char)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (char)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var resultStart = (double*)results.Address;
                        var rightStart = (double*)right.Address;
                        var leftScalar = *(double*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (double)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (double)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var resultStart = (float*)results.Address;
                        var rightStart = (float*)right.Address;
                        var leftScalar = *(float*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (float)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (float)(leftScalar * *rightStart);
                            }
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var resultStart = (decimal*)results.Address;
                        var rightStart = (decimal*)right.Address;
                        var leftScalar = *(decimal*)left.Address;
                        if (lhsCount > ParallelAbove)
                        {
                            Parallel.For(0, lhsCount, i => { *(resultStart + i) = (decimal)(leftScalar * *(rightStart + i)); });
                        }
                        else
                        {
                            for (int i = 0; i < lhsCount; i++, resultStart++, rightStart++)
                            {
                                *resultStart = (decimal)(leftScalar * *rightStart);
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

            var ndimleft = leftshape.NDim;
            var ndimright = rightshape.NDim;

            if (!leftshape.IsScalar && !rightshape.IsScalar)
            {
                Debug.Assert(leftshape.Dimensions[ndimleft - 1] == rightshape.Dimensions[ndimright - 1]);

                if (ndimleft == ndimright && leftshape == rightshape)
                {
                    return MultiplyMatrixesLinearly(left, right);
                }

                if (ndimleft == 2 && ndimright == 1)
                {
                    right._shape = ExpandEndDim(rightshape);
                    UnmanagedByteStorage<T> ret = null;
                    try
                    {
                        ret = MultiplyMatrix(left, right);
                        return ret;
                    }
                    finally
                    {
                        right._shape = rightshape;
                        // ReSharper disable once PossibleNullReferenceException
                        ret._shape = rightshape;
                    }
                }

                if (ndimleft == 1 && ndimright == 2)
                {
                    left._shape = ExpandEndDim(leftshape);
                    UnmanagedByteStorage<T> ret = null;
                    try
                    {
                        ret = MultiplyMatrix(left, right);
                        return ret;
                    }
                    finally
                    {
                        left._shape = leftshape;
                        // ReSharper disable once PossibleNullReferenceException
                        ret._shape = leftshape;
                    }
                }

                if (ndimleft == 1 && ndimright == 1)
                {
                    return MultiplyVector(left, right);
                }
            }

            if (ndimleft > 2 && ndimright > 2)
            {
                //todo here we take only last two and multiply them via MultiplyMatrix? - test in numpy.
            }

            return null;
        }

        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        internal static UnmanagedByteStorage<T> MultiplyMatrix(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            Debug.Assert(left.Shape.NDim == 2);
            Debug.Assert(right.Shape.NDim == 2);
            var shape = left._shape;
            var rows = shape[0];
            var columns = shape[1];
            var othercolumns = right._shape[1];
            UnmanagedByteStorage<T> result = new UnmanagedByteStorage<T>(new int[] {rows, othercolumns});
            switch (TypeCode)
            {
                //todo! we can speed this up by somehow having a mapped int[,] of [int row, int column] that'll give us the offset so we dont have to go through all the complex math internally at GetScalar.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    for (int row = 0; row < rows; ++row) {
                        for (int column = 0; column < othercolumns; ++column) {
                            ;#2 sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (#2)((#2) (ValueType) left.GetScalar(row, index) * (#2) (ValueType) right.GetScalar(index, column));
                            result.Set((T) (ValueType) sumProduct, row, column);
                        }
                    }

                    break;
                }
                %
#else

                case NPTypeCode.Byte:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            byte sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (byte)((byte)(ValueType)left.GetScalar(row, index) * (byte)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            short sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (short)((short)(ValueType)left.GetScalar(row, index) * (short)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            ushort sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (ushort)((ushort)(ValueType)left.GetScalar(row, index) * (ushort)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            int sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (int)((int)(ValueType)left.GetScalar(row, index) * (int)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            uint sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (uint)((uint)(ValueType)left.GetScalar(row, index) * (uint)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            long sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (long)((long)(ValueType)left.GetScalar(row, index) * (long)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            ulong sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (ulong)((ulong)(ValueType)left.GetScalar(row, index) * (ulong)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            char sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (char)((char)(ValueType)left.GetScalar(row, index) * (char)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            double sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (double)((double)(ValueType)left.GetScalar(row, index) * (double)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            float sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (float)((float)(ValueType)left.GetScalar(row, index) * (float)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < othercolumns; ++column)
                        {
                            ;
                            decimal sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (decimal)((decimal)(ValueType)left.GetScalar(row, index) * (decimal)(ValueType)right.GetScalar(index, column));
                            result.Set((T)(ValueType)sumProduct, row, column);
                        }
                    }

                    break;
                }
#endif
            }

            return result;
        }

        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        internal static UnmanagedByteStorage<T> MultiplyMatrixAgainstVector(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            if (left.Shape.NDim == 1 && right.Shape.NDim == 2)
            {
                var tmp = left;
                left = right;
                right = tmp;
            }

            Debug.Assert(left.Shape.NDim == 2);
            Debug.Assert(right.Shape.NDim == 1);
            Debug.Assert(left.Shape[0] == right.Shape[0]); //left rows == right columns
            Debug.Assert(right.Shape.NDim == 1);

            var rightShape = right._shape;
            try
            {
                right.Shape = ExpandEndDim(right.Shape);
                return MultiplyMatrix(left, right);
            }
            finally
            {
                right.Shape = rightShape;
            }
        }

        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        internal static unsafe UnmanagedByteStorage<T> MultiplyMatrixesLinearly(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            Debug.Assert(left.Shape.Size == right.Shape.Size);
            Shape shape = left._shape;
            UnmanagedByteStorage<T> results = new UnmanagedByteStorage<T>(shape);
            int size = shape.Size;
            switch (TypeCode)
            {
                //TODO! it is possible that if we use Partitioner.Create(0, 100, 100/4) to get 4 different ranges of istart to iend - parallel for will work much faster.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    var resAddr = (#2*) results._arrayAddress;
                    var leftAddr = (#2*) left._arrayAddress;
                    var rightAddr = (#2*) right._arrayAddress;
                    if (size > ParallelAbove) {
                        Parallel.For(0, size, i => { *(resAddr + i) = (#2) (*(leftAddr + i) * *(rightAddr + i)); });
                    } else {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++) {
                            *(resAddr) = (#2) (*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }
                %
                default:
                    throw new NotSupportedException();
#else

                case NPTypeCode.Byte:
                {
                    var resAddr = (byte*)results.Address;
                    var leftAddr = (byte*)left.Address;
                    var rightAddr = (byte*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (byte)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (byte)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int16:
                {
                    var resAddr = (short*)results.Address;
                    var leftAddr = (short*)left.Address;
                    var rightAddr = (short*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (short)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (short)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt16:
                {
                    var resAddr = (ushort*)results.Address;
                    var leftAddr = (ushort*)left.Address;
                    var rightAddr = (ushort*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (ushort)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (ushort)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int32:
                {
                    var resAddr = (int*)results.Address;
                    var leftAddr = (int*)left.Address;
                    var rightAddr = (int*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (int)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (int)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt32:
                {
                    var resAddr = (uint*)results.Address;
                    var leftAddr = (uint*)left.Address;
                    var rightAddr = (uint*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (uint)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (uint)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int64:
                {
                    var resAddr = (long*)results.Address;
                    var leftAddr = (long*)left.Address;
                    var rightAddr = (long*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (long)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (long)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt64:
                {
                    var resAddr = (ulong*)results.Address;
                    var leftAddr = (ulong*)left.Address;
                    var rightAddr = (ulong*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (ulong)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (ulong)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Char:
                {
                    var resAddr = (char*)results.Address;
                    var leftAddr = (char*)left.Address;
                    var rightAddr = (char*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (char)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (char)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Double:
                {
                    var resAddr = (double*)results.Address;
                    var leftAddr = (double*)left.Address;
                    var rightAddr = (double*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (double)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (double)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Single:
                {
                    var resAddr = (float*)results.Address;
                    var leftAddr = (float*)left.Address;
                    var rightAddr = (float*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (float)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (float)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Decimal:
                {
                    var resAddr = (decimal*)results.Address;
                    var leftAddr = (decimal*)left.Address;
                    var rightAddr = (decimal*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (decimal)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (decimal)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        internal static unsafe UnmanagedByteStorage<T> MultiplyVector(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            Debug.Assert(left.Shape.NDim == 1 && right.Shape.NDim == 1);
            Debug.Assert(left.Shape.Size == right.Shape.Size);
            Shape shape = left._shape;
            UnmanagedByteStorage<T> results = new UnmanagedByteStorage<T>(shape);
            var size = shape.Size;
            switch (TypeCode)
            {
                //TODO! it is possible that if we use Partitioner.Create(0, 100, 100/4) to get 4 different ranges of istart to iend - parallel for will work much faster.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    var resAddr = (#2*) results._arrayAddress;
                    var leftAddr = (#2*) left._arrayAddress;
                    var rightAddr = (#2*) right._arrayAddress;
                    if (size > ParallelAbove) {
                        Parallel.For(0, size, i => { *(resAddr + i) = (#2) (*(leftAddr + i) * *(rightAddr + i)); });
                    } else {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++) {
                            *(resAddr) = (#2) (*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }
                %
                default:
                    throw new NotSupportedException();
#else

                case NPTypeCode.Byte:
                {
                    var resAddr = (byte*)results.Address;
                    var leftAddr = (byte*)left.Address;
                    var rightAddr = (byte*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (byte)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (byte)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int16:
                {
                    var resAddr = (short*)results.Address;
                    var leftAddr = (short*)left.Address;
                    var rightAddr = (short*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (short)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (short)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt16:
                {
                    var resAddr = (ushort*)results.Address;
                    var leftAddr = (ushort*)left.Address;
                    var rightAddr = (ushort*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (ushort)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (ushort)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int32:
                {
                    var resAddr = (int*)results.Address;
                    var leftAddr = (int*)left.Address;
                    var rightAddr = (int*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (int)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (int)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt32:
                {
                    var resAddr = (uint*)results.Address;
                    var leftAddr = (uint*)left.Address;
                    var rightAddr = (uint*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (uint)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (uint)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Int64:
                {
                    var resAddr = (long*)results.Address;
                    var leftAddr = (long*)left.Address;
                    var rightAddr = (long*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (long)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (long)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.UInt64:
                {
                    var resAddr = (ulong*)results.Address;
                    var leftAddr = (ulong*)left.Address;
                    var rightAddr = (ulong*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (ulong)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (ulong)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Char:
                {
                    var resAddr = (char*)results.Address;
                    var leftAddr = (char*)left.Address;
                    var rightAddr = (char*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (char)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (char)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Double:
                {
                    var resAddr = (double*)results.Address;
                    var leftAddr = (double*)left.Address;
                    var rightAddr = (double*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (double)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (double)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Single:
                {
                    var resAddr = (float*)results.Address;
                    var leftAddr = (float*)left.Address;
                    var rightAddr = (float*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (float)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (float)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                case NPTypeCode.Decimal:
                {
                    var resAddr = (decimal*)results.Address;
                    var leftAddr = (decimal*)left.Address;
                    var rightAddr = (decimal*)right.Address;
                    if (size > ParallelAbove)
                    {
                        Parallel.For(0, size, i => { *(resAddr + i) = (decimal)(*(leftAddr + i) * *(rightAddr + i)); });
                    }
                    else
                    {
                        for (int i = 0; i < size; i++, resAddr++, leftAddr++, rightAddr++)
                        {
                            *(resAddr) = (decimal)(*(leftAddr) * *(rightAddr));
                        }
                    }

                    return results;
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        internal static unsafe UnmanagedByteStorage<T> MultiplyScalar(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            switch (TypeCode)
            {
#if _REGEN
	                %foreach supported_numericals,supported_numericals_lowercase%
                        case NPTypeCode.#1: {
                            return Scalar((T) (ValueType) (#2) (*(#2*) left._arrayAddress * *(#2*) right._arrayAddress));
                        }
                    %
                                      
                        default:
                            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return Scalar((T)(ValueType)(byte)(*(byte*)left.Address * *(byte*)right.Address));
                }

                case NPTypeCode.Int16:
                {
                    return Scalar((T)(ValueType)(short)(*(short*)left.Address * *(short*)right.Address));
                }

                case NPTypeCode.UInt16:
                {
                    return Scalar((T)(ValueType)(ushort)(*(ushort*)left.Address * *(ushort*)right.Address));
                }

                case NPTypeCode.Int32:
                {
                    return Scalar((T)(ValueType)(int)(*(int*)left.Address * *(int*)right.Address));
                }

                case NPTypeCode.UInt32:
                {
                    return Scalar((T)(ValueType)(uint)(*(uint*)left.Address * *(uint*)right.Address));
                }

                case NPTypeCode.Int64:
                {
                    return Scalar((T)(ValueType)(long)(*(long*)left.Address * *(long*)right.Address));
                }

                case NPTypeCode.UInt64:
                {
                    return Scalar((T)(ValueType)(ulong)(*(ulong*)left.Address * *(ulong*)right.Address));
                }

                case NPTypeCode.Char:
                {
                    return Scalar((T)(ValueType)(char)(*(char*)left.Address * *(char*)right.Address));
                }

                case NPTypeCode.Double:
                {
                    return Scalar((T)(ValueType)(double)(*(double*)left.Address * *(double*)right.Address));
                }

                case NPTypeCode.Single:
                {
                    return Scalar((T)(ValueType)(float)(*(float*)left.Address * *(float*)right.Address));
                }

                case NPTypeCode.Decimal:
                {
                    return Scalar((T)(ValueType)(decimal)(*(decimal*)left.Address * *(decimal*)right.Address));
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html</remarks>
        public static UnmanagedByteStorage<T> Dot(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            //Dot product of two arrays.Specifically,
            //If both a and b are 1 - D arrays, it is inner product of vectors(without complex conjugation).
            //If both a and b are 2 - D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            //V If either a or b is 0 - D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a *b is preferred.
            //If a is an N - D array and b is a 1 - D array, it is a sum product over the last axis of a and b.
            //If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the last axis of a and the second-to - last axis of b:
            //  dot(a, b)[i, j, k, m] = sum(a[i, j,:] * b[k,:, m])
            var leftshape = left._shape;
            var rightshape = right._shape;
            var isLeftScalar = leftshape.IsScalar;
            var isRightScalar = rightshape.IsScalar;

            if (isLeftScalar && isRightScalar)
            {
                return MultiplyScalar(left, right);
            }

            //If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
            if (isLeftScalar || isRightScalar)
            {
                return Multiply(left, right);
            }

            //If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            if (leftshape.NDim == 2 && rightshape.NDim == 2)
            {
                return MultiplyMatrix(left, right);
            }

            //If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            if (leftshape.NDim == 1 && rightshape.NDim == 1)
            {
                Debug.Assert(leftshape[0] == rightshape[0]);
                return MultiplyVector(left, right);
            }

            //If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            if (leftshape.NDim >= 2 && rightshape.NDim == 1)
            {
                right._shape = ExpandEndDim(rightshape);
                UnmanagedByteStorage<T> ret = null;
                try
                {
                    ret = MultiplyMatrix(left, right);
                    return ret;
                }
                finally
                {
                    right._shape = rightshape;
                    // ReSharper disable once PossibleNullReferenceException
                    ret._shape = rightshape;
                }
            }

            if (leftshape.NDim == 1)
            {
                throw new NotSupportedException("lhs cannot be 1-D, use `np.multiply` or `*` for this case."); //todo!  ValueError: shapes (4,) and (2,4) not aligned: 4 (dim 0) != 2 (dim 0)
            }

            //left cant be 0 or 1 by this point
            //If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
            //dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
            if (rightshape.NDim >= 2)
            {
                throw new NotImplementedException();
            }

            throw new NotSupportedException();
        }

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html</remarks>
        public static UnmanagedByteStorage<T> Matmul(UnmanagedByteStorage<T> left, UnmanagedByteStorage<T> right)
        {
            var leftshape = left._shape;
            var rightshape = right._shape;
            if (leftshape.IsScalar || rightshape.IsScalar)
                throw new InvalidOperationException("Matmul can't handle scalar multiplication, use `*` or `np.dot(..)` instead");
            var ndimLeft = leftshape.NDim;
            var ndimright = rightshape.NDim;

            //If both arguments are 2-D they are multiplied like conventional matrices.
            if (ndimLeft == 2 && ndimright == 2)
            {
                return MultiplyMatrix(left, right);
            }

            //todo If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

            //If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
            if ((ndimLeft == 2 && ndimright == 1))
            {
                right._shape = ExpandEndDim(rightshape);
                UnmanagedByteStorage<T> ret = null;
                try
                {
                    ret = MultiplyMatrix(left, right);
                    return ret;
                }
                finally
                {
                    right._shape = rightshape;
                    // ReSharper disable once PossibleNullReferenceException
                    ret._shape = rightshape;
                }
            }

            //If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
            if (ndimLeft == 1 && ndimright == 2)
            {
                throw new NotSupportedException("Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)");
            }


            return null;
        }

        private static int[] ExpandStartDim(Shape shape)
        {
            var ret = new int[shape.NDim + 1];
            ret[0] = 1;
            Array.Copy(shape.Dimensions, 0, ret, 1, shape.NDim);
            return ret;
        }

        private static Shape ExpandEndDim(Shape shape)
        {
            var ret = new int[shape.NDim + 1];
            ret[ret.Length - 1] = 1;
            Array.Copy(shape.Dimensions, 0, ret, 0, shape.NDim);
            return new Shape(ret);
        }
    }
}
