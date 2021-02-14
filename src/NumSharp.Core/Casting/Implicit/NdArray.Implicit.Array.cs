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
using System.Text;
using System.Text.RegularExpressions;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public static implicit operator NDArray(Array array)
        {
            // ReSharper disable once PossibleNullReferenceException
            bool isJaggedArray = array.GetType().GetElementType().IsArray;
            var underlying = array.ResolveElementType();
            if (isJaggedArray)
            {
                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
#if _REGEN1
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1: return NDArray.FromJaggedArray<#2>(array);
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean: return NDArray.FromJaggedArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromJaggedArray<byte>(array);
	                case NPTypeCode.Int32: return NDArray.FromJaggedArray<int>(array);
	                case NPTypeCode.Int64: return NDArray.FromJaggedArray<long>(array);
	                case NPTypeCode.Single: return NDArray.FromJaggedArray<float>(array);
	                case NPTypeCode.Double: return NDArray.FromJaggedArray<double>(array);
	                default:
		                throw new NotSupportedException();
#endif
                }
            }
            else
            {

                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
#if _REGEN1
	                %foreach supported_dtypes,supported_dtypes_lowercase%
	                case NPTypeCode.#1: return NDArray.FromMultiDimArray<#2>(array);
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean: return NDArray.FromMultiDimArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromMultiDimArray<byte>(array);
	                case NPTypeCode.Int32: return NDArray.FromMultiDimArray<int>(array);
	                case NPTypeCode.Int64: return NDArray.FromMultiDimArray<long>(array);
	                case NPTypeCode.Single: return NDArray.FromMultiDimArray<float>(array);
	                case NPTypeCode.Double: return NDArray.FromMultiDimArray<double>(array);
	                default:
		                throw new NotSupportedException();
#endif
                }
            }

            throw new NotImplementedException("implicit operator NDArray(Array array)");
        }

        public static explicit operator Array(NDArray nd)
        {
            switch (nd.GetTypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return nd.ToMuliDimArray<#2>();
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return nd.ToMuliDimArray<bool>();
	            case NPTypeCode.Byte: return nd.ToMuliDimArray<byte>();
	            case NPTypeCode.Int32: return nd.ToMuliDimArray<int>();
	            case NPTypeCode.Int64: return nd.ToMuliDimArray<long>();
	            case NPTypeCode.Single: return nd.ToMuliDimArray<float>();
	            case NPTypeCode.Double: return nd.ToMuliDimArray<double>();
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public static explicit operator string(NDArray nd)
        {
            return UTF8Encoding.UTF8.GetString(nd.GetData<byte>().ToArray());
        }

        public static explicit operator NDArray(string str)
        {
            // process "[1, 2, 3]" 
            if (new Regex(@"^\[[\d,\s\.]+\]$").IsMatch(str))
            {
                var data = str.Substring(1, str.Length - 2)
                    .Split(',')
                    .Select(x => double.Parse(x)).ToArray();
                var nd = new NDArray(data, new Shape(data.Length));
                return nd;
            }

            Regex reg = new Regex(@"\[((\d,?)+|;)+\]");

            if (reg.IsMatch(str))
            {
                NDArray nd = null;

                string[][] splitted = null;
                str = str.Substring(1, str.Length - 2);

                if (str.Contains(","))
                {
                    splitted = str.Split(';')
                        .Select(x => x.Split(','))
                        .ToArray();
                }
                else
                {
                    splitted = str.Split(';')
                        .Select(x => x.Split(' '))
                        .ToArray();
                }


                int dim0 = splitted.Length;
                int dim1 = splitted[0].Length;

                var shape = new Shape(new int[] {dim0, dim1});

                nd = new NDArray(typeof(double), shape);

                for (int idx = 0; idx < splitted.Length; idx++)
                {
                    for (int jdx = 0; jdx < splitted[0].Length; jdx++)
                    {
                        nd[idx, jdx] = (NDArray)Double.Parse(splitted[idx][jdx]);
                    }
                }

                return nd;
            }
            else
            {
                var nd = new NDArray(UTF8Encoding.UTF8.GetBytes(str));
                return nd;
            }
        }
    }
}
