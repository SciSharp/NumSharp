using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        #region 22matmul
#if MINIMAL
        protected static NDArray MultiplyMatrix(NDArray left, NDArray right, NDArray @out = null)
        {
            return null;
        }
#else
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html</remarks>
        [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
        [MethodImpl((MethodImplOptions)768)]
        protected static NDArray MultiplyMatrix(NDArray left, NDArray right, NDArray @out = null)
        {
            Debug.Assert(left.Shape.NDim == 2);
            Debug.Assert(right.Shape.NDim == 2);
            Debug.Assert(@out is null || @out.shape[0] == left.Shape[0] && right.Shape[1] == 2);

            if (left.Shape[-1] != right.Shape[-2])
                throw new IncorrectShapeException($"shapes {left.Shape} and {right.Shape} not aligned: {left.Shape[-1]} (dim 2) != {right.Shape[-2]} (dim 1)");

            var shape = left.Shape;
            var rows = shape[0];
            var columns = shape[1];

            var othercolumns = right.Shape[1];

            if (!(@out is null) && (@out.ndim != 2 || @out.Shape[0] != rows || @out.Shape[1] != othercolumns))
                throw new IncorrectShapeException($"shapes {left.Shape} and {right.Shape} are not compatible with given @out array's shape {@out.Shape} for matrix multiplication.");

            NDArray result = @out ?? new NDArray(np._FindCommonArrayType(left.GetTypeCode, right.GetTypeCode), Shape.Matrix(rows, othercolumns));

#if _REGEN1
#region Compute
            switch (result.typecode)
            {
	            %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1: { 
                    switch (left.typecode)
                    {
	                    %foreach supported_numericals,supported_numericals_lowercase%
                        case NPTypeCode.#101: { 
                            switch (right.typecode)
                            {
	                            %foreach supported_numericals,supported_numericals_lowercase%
                                case NPTypeCode.#201: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            |#2 sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (#2)(Operator.Multiply(left.Get#101(row, i), right.Get#201(i, column)));
                                            result.Set#1(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                %
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        %
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
                %
            }
#endregion
#else

#region Compute
            switch (result.typecode)
            {
                case NPTypeCode.Byte: { 
                    switch (left.typecode)
                    {
                        case NPTypeCode.Byte: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetByte(row, i), right.GetByte(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetByte(row, i), right.GetInt32(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetByte(row, i), right.GetInt64(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetByte(row, i), right.GetSingle(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetByte(row, i), right.GetDouble(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int32: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt32(row, i), right.GetByte(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt32(row, i), right.GetInt32(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt32(row, i), right.GetInt64(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt32(row, i), right.GetSingle(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt32(row, i), right.GetDouble(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int64: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt64(row, i), right.GetByte(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt64(row, i), right.GetInt32(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt64(row, i), right.GetInt64(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt64(row, i), right.GetSingle(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetInt64(row, i), right.GetDouble(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Single: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetSingle(row, i), right.GetByte(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetSingle(row, i), right.GetInt32(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetSingle(row, i), right.GetInt64(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetSingle(row, i), right.GetSingle(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetSingle(row, i), right.GetDouble(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Double: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetDouble(row, i), right.GetByte(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetDouble(row, i), right.GetInt32(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetDouble(row, i), right.GetInt64(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetDouble(row, i), right.GetSingle(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            byte sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (byte)(Operator.Multiply(left.GetDouble(row, i), right.GetDouble(i, column)));
                                            result.SetByte(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
                case NPTypeCode.Int32: { 
                    switch (left.typecode)
                    {
                        case NPTypeCode.Byte: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetByte(row, i), right.GetByte(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetByte(row, i), right.GetInt32(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetByte(row, i), right.GetInt64(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetByte(row, i), right.GetSingle(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetByte(row, i), right.GetDouble(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int32: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt32(row, i), right.GetByte(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt32(row, i), right.GetInt32(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt32(row, i), right.GetInt64(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt32(row, i), right.GetSingle(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt32(row, i), right.GetDouble(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int64: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt64(row, i), right.GetByte(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt64(row, i), right.GetInt32(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt64(row, i), right.GetInt64(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt64(row, i), right.GetSingle(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetInt64(row, i), right.GetDouble(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Single: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetSingle(row, i), right.GetByte(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetSingle(row, i), right.GetInt32(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetSingle(row, i), right.GetInt64(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetSingle(row, i), right.GetSingle(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetSingle(row, i), right.GetDouble(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Double: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetDouble(row, i), right.GetByte(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetDouble(row, i), right.GetInt32(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetDouble(row, i), right.GetInt64(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetDouble(row, i), right.GetSingle(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            int sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (int)(Operator.Multiply(left.GetDouble(row, i), right.GetDouble(i, column)));
                                            result.SetInt32(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
                case NPTypeCode.Int64: { 
                    switch (left.typecode)
                    {
                        case NPTypeCode.Byte: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetByte(row, i), right.GetByte(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetByte(row, i), right.GetInt32(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetByte(row, i), right.GetInt64(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetByte(row, i), right.GetSingle(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetByte(row, i), right.GetDouble(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int32: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt32(row, i), right.GetByte(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt32(row, i), right.GetInt32(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt32(row, i), right.GetInt64(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt32(row, i), right.GetSingle(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt32(row, i), right.GetDouble(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int64: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt64(row, i), right.GetByte(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt64(row, i), right.GetInt32(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt64(row, i), right.GetInt64(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt64(row, i), right.GetSingle(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetInt64(row, i), right.GetDouble(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Single: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetSingle(row, i), right.GetByte(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetSingle(row, i), right.GetInt32(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetSingle(row, i), right.GetInt64(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetSingle(row, i), right.GetSingle(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetSingle(row, i), right.GetDouble(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Double: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetDouble(row, i), right.GetByte(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetDouble(row, i), right.GetInt32(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetDouble(row, i), right.GetInt64(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetDouble(row, i), right.GetSingle(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            long sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (long)(Operator.Multiply(left.GetDouble(row, i), right.GetDouble(i, column)));
                                            result.SetInt64(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
                case NPTypeCode.Single: { 
                    switch (left.typecode)
                    {
                        case NPTypeCode.Byte: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetByte(row, i), right.GetByte(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetByte(row, i), right.GetInt32(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetByte(row, i), right.GetInt64(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetByte(row, i), right.GetSingle(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetByte(row, i), right.GetDouble(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int32: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt32(row, i), right.GetByte(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt32(row, i), right.GetInt32(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt32(row, i), right.GetInt64(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt32(row, i), right.GetSingle(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt32(row, i), right.GetDouble(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int64: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt64(row, i), right.GetByte(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt64(row, i), right.GetInt32(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt64(row, i), right.GetInt64(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt64(row, i), right.GetSingle(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetInt64(row, i), right.GetDouble(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Single: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetSingle(row, i), right.GetByte(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetSingle(row, i), right.GetInt32(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetSingle(row, i), right.GetInt64(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetSingle(row, i), right.GetSingle(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetSingle(row, i), right.GetDouble(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Double: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetDouble(row, i), right.GetByte(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetDouble(row, i), right.GetInt32(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetDouble(row, i), right.GetInt64(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetDouble(row, i), right.GetSingle(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            float sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (float)(Operator.Multiply(left.GetDouble(row, i), right.GetDouble(i, column)));
                                            result.SetSingle(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
                case NPTypeCode.Double: { 
                    switch (left.typecode)
                    {
                        case NPTypeCode.Byte: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetByte(row, i), right.GetByte(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetByte(row, i), right.GetInt32(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetByte(row, i), right.GetInt64(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetByte(row, i), right.GetSingle(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetByte(row, i), right.GetDouble(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int32: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt32(row, i), right.GetByte(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt32(row, i), right.GetInt32(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt32(row, i), right.GetInt64(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt32(row, i), right.GetSingle(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt32(row, i), right.GetDouble(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Int64: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt64(row, i), right.GetByte(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt64(row, i), right.GetInt32(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt64(row, i), right.GetInt64(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt64(row, i), right.GetSingle(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetInt64(row, i), right.GetDouble(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Single: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetSingle(row, i), right.GetByte(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetSingle(row, i), right.GetInt32(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetSingle(row, i), right.GetInt64(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetSingle(row, i), right.GetSingle(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetSingle(row, i), right.GetDouble(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        case NPTypeCode.Double: { 
                            switch (right.typecode)
                            {
                                case NPTypeCode.Byte: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetDouble(row, i), right.GetByte(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int32: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetDouble(row, i), right.GetInt32(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Int64: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetDouble(row, i), right.GetInt64(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Single: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetDouble(row, i), right.GetSingle(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                case NPTypeCode.Double: { 
                                    for (int row = 0; row < rows; row++)
                                    {
                                        for (int column = 0; column < othercolumns; column++)
                                        {
                                            double sum = default;
                                            for (int i = 0; i < columns; i++)
                                                sum += (double)(Operator.Multiply(left.GetDouble(row, i), right.GetDouble(i, column)));
                                            result.SetDouble(sum, row, column);
                                        }
                                    }

                                    break;
                                }
                                default:
                                    throw new NotSupportedException();
                            }

                            break;
                        }
                        default:
                            throw new NotSupportedException();
                    }
                    
                    break;
                }
            }
#endregion
#endif

            return result;
        }

#endif
#endregion
    }
}
