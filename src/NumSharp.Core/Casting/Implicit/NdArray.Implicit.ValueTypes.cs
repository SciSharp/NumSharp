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

using System.Numerics;

namespace NumSharp
{
    public partial class NDArray
    {
        //return null;
        //public static implicit operator string(NDArray nd)
        //{
        //    if (nd.ndim > 0)
        //        throw new IncorrectShapeException();

        //    return nd.Data<string>(0);
        //}


#if __REGEN //DO NOT REGENERATE
        %foreach supported_dtypes,supported_dtypes_lowercase%

            //#2 operators
            public static implicit operator NDArray(#2 d) => NDArray.Scalar<#2>(d);

            public static implicit operator #2(NDArray nd)
            {
                if (nd.ndim != 0)
                    throw new IncorrectShapeException();

                return nd.GetAtIndex<#2>(0);
            }
        %
#else

        //bool operators
        public static implicit operator NDArray(bool d) => NDArray.Scalar<bool>(d);

        public static implicit operator bool(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<bool>(0);
        }

        //short operators
        public static implicit operator NDArray(short d) => NDArray.Scalar<short>(d);

        public static implicit operator short(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<short>(0);
        }

        //ushort operators
        public static explicit operator NDArray(ushort d) => NDArray.Scalar<ushort>(d);

        public static explicit operator ushort(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<ushort>(0);
        }

        //int operators
        public static implicit operator NDArray(int d) => NDArray.Scalar<int>(d);

        public static implicit operator int(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<int>(0);
        }

        //uint operators
        public static implicit operator NDArray(uint d) => NDArray.Scalar<uint>(d);

        public static implicit operator uint(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<uint>(0);
        }

        //long operators
        public static implicit operator NDArray(long d) => NDArray.Scalar<long>(d);

        public static implicit operator long(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<long>(0);
        }

        //ulong operators
        public static implicit operator NDArray(ulong d) => NDArray.Scalar<ulong>(d);

        public static implicit operator ulong(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<ulong>(0);
        }

        // byte operators
        public static implicit operator NDArray(byte d) => NDArray.Scalar<byte>(d);

        public static implicit operator byte(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<byte>(0);
        }

        public void Deconstruct(out byte b, out byte g, out byte r)
        {
            if (ndim == 0)
                throw new IncorrectShapeException();

            b = GetAtIndex<byte>(0);
            g = GetAtIndex<byte>(1);
            r = GetAtIndex<byte>(2);
        }

        //char operators
        public static implicit operator NDArray(char d) => NDArray.Scalar<char>(d);

        public static implicit operator char(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<char>(0);
        }

        //double operators
        public static implicit operator NDArray(double d) => NDArray.Scalar<double>(d);

        public static implicit operator double(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<double>(0);
        }

        //float operators
        public static implicit operator NDArray(float d) => NDArray.Scalar<float>(d);

        public static implicit operator float(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<float>(0);
        }

        //decimal operators
        public static implicit operator NDArray(decimal d) => NDArray.Scalar<decimal>(d);

        public static implicit operator decimal(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException();

            return nd.GetAtIndex<decimal>(0);
        }
#endif

        public static implicit operator NDArray(Complex d) => NDArray.Scalar<Complex>(d);
    }
}
