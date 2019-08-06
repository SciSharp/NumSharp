using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html</remarks>
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        internal static NDArray MultiplyMatrix(NDArray left, NDArray right, NDArray @out = null)
        {
            Debug.Assert(left.Shape.NDim == 2);
            Debug.Assert(right.Shape.NDim == 2);
            Debug.Assert(@out is null || @out.shape[0] == left.Shape[0] && right.Shape[1] == 2);
            var shape = left.Shape;
            var rows = shape[0];
            var columns = shape[1];
            var othercolumns = right.Shape[1];

            NDArray result = @out ?? new NDArray(np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode), Shape.Matrix(rows, othercolumns));
            switch (result.GetTypeCode)
            {
                //todo! we can speed this up by somehow having a mapped int[,] of [int row, int column] that'll give us the offset so we dont have to go through all the complex math internally at GetScalar.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    for (int row = 0; row < rows; ++row) {
                        for (int column = 0; column < othercolumns; ++column) {
                            |#2 sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (#2)((#2) (ValueType) left.GetValue(row, index) * (#2) (ValueType) right.GetValue(index, column));
                            result.SetValue(sumProduct, row, column);
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
                            byte sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (byte)((byte)(ValueType)left.GetValue(row, index) * (byte)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            short sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (short)((short)(ValueType)left.GetValue(row, index) * (short)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            ushort sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (ushort)((ushort)(ValueType)left.GetValue(row, index) * (ushort)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            int sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (int)((int)(ValueType)left.GetValue(row, index) * (int)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            uint sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (uint)((uint)(ValueType)left.GetValue(row, index) * (uint)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            long sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (long)((long)(ValueType)left.GetValue(row, index) * (long)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            ulong sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (ulong)((ulong)(ValueType)left.GetValue(row, index) * (ulong)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            char sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (char)((char)(ValueType)left.GetValue(row, index) * (char)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            double sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (double)((double)(ValueType)left.GetValue(row, index) * (double)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            float sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (float)((float)(ValueType)left.GetValue(row, index) * (float)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
                            decimal sumProduct = default;
                            for (int index = 0; index < columns; ++index)
                                sumProduct += (decimal)((decimal)(ValueType)left.GetValue(row, index) * (decimal)(ValueType)right.GetValue(index, column));
                            result.SetValue((object)sumProduct, row, column);
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
        internal static NDArray MultiplyMatrixAgainstVector(NDArray left, NDArray right)
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

            var rightShape = right.Shape;
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
        internal static unsafe NDArray MultiplyMatrixesLinearly(NDArray left, NDArray right)
        {
            Debug.Assert(left.Shape.Size == right.Shape.Size);
            Shape shape = left.Shape;
            NDArray results = new NDArray(np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode), shape);
            int size = shape.Size;
            switch (results.GetTypeCode)
            {
                //TODO! it is possible that if we use Partitioner.Create(0, 100, 100/4) to get 4 different ranges of istart to iend - parallel for will work much faster.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    var resAddr = (#2*) results.Address;
                    var leftAddr = (#2*) left.Address;
                    var rightAddr = (#2*) right.Address;
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
        internal static unsafe NDArray MultiplyVector(NDArray left, NDArray right)
        {
            Debug.Assert(left.Shape.NDim == 1 && right.Shape.NDim == 1);
            Debug.Assert(left.Shape.Size == right.Shape.Size);
            Shape shape = left.Shape;
            NDArray results = new NDArray(np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode), shape);
            var size = shape.Size;
            switch (results.GetTypeCode)
            {
                //TODO! it is possible that if we use Partitioner.Create(0, 100, 100/4) to get 4 different ranges of istart to iend - parallel for will work much faster.
#if _REGEN
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: {
                    var resAddr = (#2*) results.Address;
                    var leftAddr = (#2*) left.Address;
                    var rightAddr = (#2*) right.Address;
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

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html</remarks>
        public override NDArray Dot(in NDArray left, in NDArray right)
        {
            //Dot product of two arrays.Specifically,
            //If both a and b are 1 - D arrays, it is inner product of vectors(without complex conjugation).
            //If both a and b are 2 - D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            //V If either a or b is 0 - D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a *b is preferred.
            //If a is an N - D array and b is a 1 - D array, it is a sum product over the last axis of a and b.
            //If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the last axis of a and the second-to - last axis of b:
            //  dot(a, b)[i, j, k, m] = sum(a[i, j,:] * b[k,:, m])
            var leftshape = left.Shape;
            var rightshape = right.Shape;
            var isLeftScalar = leftshape.IsScalar;
            var isRightScalar = rightshape.IsScalar;

            if (isLeftScalar && isRightScalar)
            {
                return Multiply(left, right);
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
                return Multiply(left, right); //MultiplyVector
            }

            //If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            if (leftshape.NDim >= 2 && rightshape.NDim == 1)
            {
                right.Shape = ExpandEndDim(rightshape);
                NDArray ret = null;
                try
                {
                    ret = MultiplyMatrix(left, right);
                    return ret;
                }
                finally
                {
                    right.Shape = rightshape;
                    // ReSharper disable once PossibleNullReferenceException
                    ret.Shape = rightshape;
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
