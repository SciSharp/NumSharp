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
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Collections;
using NumSharp;
using System.Text.RegularExpressions;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// User-defined conversion from double to Digit
        /// </summary>
        /// <param name="array"></param>
        public static implicit operator NDArray(Array array)
        {
            // ReSharper disable once PossibleNullReferenceException
            bool isJaggedArray = array.GetType().GetElementType().IsArray;

            if (isJaggedArray)
            {
                var type = array.GetType().GetElementType();

                switch (array.GetType().GetArrayRank())
                {
                    case 1:
                        type = type.GetElementType();
                        break;
                    case 2:
                        type = type.GetElementType().GetElementType();
                        break;
                    case 3:
                        type = type.GetElementType().GetElementType().GetElementType();
                        break;
                    default:
                        throw new NotImplementedException("implicit operator NDArray(Array array)");
                }

                var nd = new NDArray(type);
                switch (Type.GetTypeCode(nd.dtype))
                {
                    case TypeCode.Boolean:
                        return nd.FromJaggedArray<bool>(array);
                    case TypeCode.Int16:
                        return nd.FromJaggedArray<short>(array);
                    case TypeCode.Int32:
                        return nd.FromJaggedArray<int>(array);
                    case TypeCode.Int64:
                        return nd.FromJaggedArray<long>(array);
                    case TypeCode.Single:
                        return nd.FromJaggedArray<float>(array);
                    case TypeCode.Double:
                        return nd.FromJaggedArray<double>(array);
                    case TypeCode.Decimal:
                        return nd.FromJaggedArray<decimal>(array);
                }
            }
            else
            {
                var nd = new NDArray(array.GetType().GetElementType());
                switch (Type.GetTypeCode(nd.dtype))
                {
                    case TypeCode.Boolean:
                        return nd.FromMultiDimArray<bool>(array);
                    case TypeCode.Int16:
                        return nd.FromMultiDimArray<short>(array);
                    case TypeCode.Int32:
                        return nd.FromMultiDimArray<int>(array);
                    case TypeCode.Int64:
                        return nd.FromMultiDimArray<long>(array);
                    case TypeCode.Single:
                        return nd.FromMultiDimArray<float>(array);
                    case TypeCode.Double:
                        return nd.FromMultiDimArray<double>(array);
                    case TypeCode.Decimal:
                        return nd.FromMultiDimArray<decimal>(array);
                }
            }

            throw new NotImplementedException("implicit operator NDArray(Array array)");
        }

        public static explicit operator Array(NDArray nd)
        {
            //todo! cache generic invocation.
            var methods = nd.GetType().GetMethods().Where(x => x.Name.Equals("ToMuliDimArray") && x.IsGenericMethod && x.ReturnType.Name.Equals("Array"));
            var genMethods = methods.First().MakeGenericMethod(nd.dtype);

            return (Array)genMethods.Invoke(nd, null);
        }

        public static implicit operator NDArray(string str)
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
                        nd[idx, jdx] = Double.Parse(splitted[idx][jdx]);
                    }
                }

                return nd;
            }
            else
            {
                var nd = new NDArray(typeof(string), new int[0]);
                nd.Storage.ReplaceData((Array)new string[] {str});

                return nd;
            }
        }
    }
}
