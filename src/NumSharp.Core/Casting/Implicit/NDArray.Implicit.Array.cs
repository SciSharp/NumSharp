/*
 * NumSharp
 * Copyright (C) 2018 Haiping Chen
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Apache License 2.0 as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the Apache License 2.0
 * along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.0/>.
 */

using System;
using System.Linq;
using System.Text.RegularExpressions;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Converts a managed array (any rank, or jagged) into an <see cref="NDArray"/> copy of its data; a null array
        ///     converts to a null <see cref="NDArray"/>. Text — a <see cref="string"/> array, or an <c>object</c> array holding
        ///     strings / <c>byte[]</c> values — becomes NumPy's fixed-width string array (<c>np.array(['a', 'bb'])</c> is
        ///     <c>&lt;U2</c>), so <c>NDArray x = new[] { "a", "bb" }</c> and every <c>NDArray</c> parameter fed a C# string
        ///     array (C# prefers the <c>NDArray</c> overload to an <c>object</c> one) read the text as NumPy does.
        /// </summary>
        /// <param name="array">The array to convert, or null.</param>
        /// <returns>A new array holding a copy of the elements, or null when <paramref name="array"/> is null.</returns>
        /// <exception cref="NotSupportedException">The element type is not one of NumSharp's dtypes.</exception>
        /// <exception cref="ValueError">A ragged jagged array of text (NumPy's inhomogeneous-shape text).</exception>
        /// <remarks>
        ///     C# runs a user-defined conversion even for a null source, so without the null pass-through an optional
        ///     <see cref="NDArray"/> parameter fed from a null <c>double[]</c> variable (a common way to spell NumPy's
        ///     <c>p=None</c>) crashed with a <see cref="NullReferenceException"/> instead of meaning "not given".
        /// </remarks>
        public static implicit operator NDArray(Array array)
        {
            if (array is null)
                return null;

            // Text values build a fixed-width string array (the same discovery as np.array); everything else is numeric.
            if (Backends.Strings.NDStringArrays.IsTextInput(array, null))
                return Backends.Strings.NDStringArrays.FromManaged(array, null, 0, 'C');

            bool isJaggedArray = array.GetType().GetElementType().IsArray;
            var underlying = array.ResolveElementType();
            if (isJaggedArray)
            {
                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
	                // %foreach supported_dtypes,supported_dtypes_lowercase%
	                // case NPTypeCode.#1: return NDArray.FromJaggedArray<#2>(array);
	                // %
	                // default:
		                // throw new NotSupportedException();
                    case NPTypeCode.Boolean: return NDArray.FromJaggedArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromJaggedArray<byte>(array);
	                case NPTypeCode.SByte: return NDArray.FromJaggedArray<sbyte>(array);
	                case NPTypeCode.Int16: return NDArray.FromJaggedArray<short>(array);
	                case NPTypeCode.UInt16: return NDArray.FromJaggedArray<ushort>(array);
	                case NPTypeCode.Int32: return NDArray.FromJaggedArray<int>(array);
	                case NPTypeCode.UInt32: return NDArray.FromJaggedArray<uint>(array);
	                case NPTypeCode.Int64: return NDArray.FromJaggedArray<long>(array);
	                case NPTypeCode.UInt64: return NDArray.FromJaggedArray<ulong>(array);
	                case NPTypeCode.Char: return NDArray.FromJaggedArray<char>(array);
	                case NPTypeCode.Half: return NDArray.FromJaggedArray<Half>(array);
	                case NPTypeCode.Double: return NDArray.FromJaggedArray<double>(array);
	                case NPTypeCode.Single: return NDArray.FromJaggedArray<float>(array);
	                case NPTypeCode.Decimal: return NDArray.FromJaggedArray<decimal>(array);
	                case NPTypeCode.Complex: return NDArray.FromJaggedArray<System.Numerics.Complex>(array);
	                default:
		                throw new NotSupportedException();
                }
            }
            else
            {

                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
	                // %foreach supported_dtypes,supported_dtypes_lowercase%
	                // case NPTypeCode.#1: return NDArray.FromMultiDimArray<#2>(array);
	                // %
	                // default:
		                // throw new NotSupportedException();
                    case NPTypeCode.Boolean: return NDArray.FromMultiDimArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromMultiDimArray<byte>(array);
	                case NPTypeCode.SByte: return NDArray.FromMultiDimArray<sbyte>(array);
	                case NPTypeCode.Int16: return NDArray.FromMultiDimArray<short>(array);
	                case NPTypeCode.UInt16: return NDArray.FromMultiDimArray<ushort>(array);
	                case NPTypeCode.Int32: return NDArray.FromMultiDimArray<int>(array);
	                case NPTypeCode.UInt32: return NDArray.FromMultiDimArray<uint>(array);
	                case NPTypeCode.Int64: return NDArray.FromMultiDimArray<long>(array);
	                case NPTypeCode.UInt64: return NDArray.FromMultiDimArray<ulong>(array);
	                case NPTypeCode.Char: return NDArray.FromMultiDimArray<char>(array);
	                case NPTypeCode.Half: return NDArray.FromMultiDimArray<Half>(array);
	                case NPTypeCode.Double: return NDArray.FromMultiDimArray<double>(array);
	                case NPTypeCode.Single: return NDArray.FromMultiDimArray<float>(array);
	                case NPTypeCode.Decimal: return NDArray.FromMultiDimArray<decimal>(array);
	                case NPTypeCode.Complex: return NDArray.FromMultiDimArray<System.Numerics.Complex>(array);
	                default:
		                throw new NotSupportedException();
                }
            }

            throw new NotImplementedException("implicit operator NDArray(Array array)");
        }

        public static explicit operator Array(NDArray nd)
        {
            switch (nd.GetTypeCode)
            {
	            // %foreach supported_dtypes,supported_dtypes_lowercase%
	            // case NPTypeCode.#1: return nd.ToMuliDimArray<#2>();
	            // %
	            // default:
		            // throw new NotSupportedException();
	            case NPTypeCode.Boolean: return nd.ToMuliDimArray<bool>();
	            case NPTypeCode.Byte: return nd.ToMuliDimArray<byte>();
	            case NPTypeCode.SByte: return nd.ToMuliDimArray<sbyte>();
	            case NPTypeCode.Int16: return nd.ToMuliDimArray<short>();
	            case NPTypeCode.UInt16: return nd.ToMuliDimArray<ushort>();
	            case NPTypeCode.Int32: return nd.ToMuliDimArray<int>();
	            case NPTypeCode.UInt32: return nd.ToMuliDimArray<uint>();
	            case NPTypeCode.Int64: return nd.ToMuliDimArray<long>();
	            case NPTypeCode.UInt64: return nd.ToMuliDimArray<ulong>();
	            case NPTypeCode.Char: return nd.ToMuliDimArray<char>();
	            case NPTypeCode.Half: return nd.ToMuliDimArray<Half>();
	            case NPTypeCode.Double: return nd.ToMuliDimArray<double>();
	            case NPTypeCode.Single: return nd.ToMuliDimArray<float>();
	            case NPTypeCode.Decimal: return nd.ToMuliDimArray<decimal>();
	            case NPTypeCode.Complex: return nd.ToMuliDimArray<System.Numerics.Complex>();
	            // Text: a C# array of the decoded elements (string for U, byte[] for S) in the array's own shape.
	            case NPTypeCode.String: return Backends.Strings.NDStringArrays.ToManagedArray(nd);
	            default:
		            throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     A C# string is a Python <c>str</c>: the 0-d text array <c>np.array(str)</c> builds (<c>&lt;U{len}</c>), marked as
        ///     the Python str it stands for (<see cref="NDArray.WasPythonStr"/>), so <c>a[i] = "xy"</c> stores text into an
        ///     S / U array and <c>a[i] = "12"</c> parses it into a numeric one with NumPy's setitem texts.
        /// </summary>
        /// <param name="str">The text.</param>
        /// <returns>The 0-d text array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="str"/> is null.</exception>
        /// <remarks>
        ///     Until 2026-10-12 a text shaped like <c>"[1, 2, 3]"</c> or <c>"[1 2; 3 4]"</c> was parsed into a float64 vector or
        ///     matrix here (a legacy NumSharp shorthand). NumPy never reads a str as numbers in <c>np.array</c>
        ///     (<c>np.array('[1, 2, 3]')</c> is the 0-d <c>&lt;U9</c> text), and the owner decided (strings plan D6) to drop
        ///     the shorthand; <see cref="np.matrix(string, DType, bool)"/> keeps its own literal grammar (<c>"1 2; 3 4"</c>).
        /// </remarks>
        public static implicit operator NDArray(string str)
        {
            if (str is null)
                throw new ArgumentNullException(nameof(str));
            return NDArray.PythonStrLiteral(str);
        }
    }
}
