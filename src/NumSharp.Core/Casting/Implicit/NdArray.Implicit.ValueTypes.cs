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

        public static implicit operator NDArray(bool d)
        {
            var ndArray = new NDArray(typeof(bool), new int[0]);
            ndArray.Storage.ReplaceData((Array)new bool[] {d});

            return ndArray;
        }

        public static implicit operator bool(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<bool>(0);
        }

        public static implicit operator NDArray(float d)
        {
            var ndArray = new NDArray(typeof(float), new int[0]);
            ndArray.Storage.ReplaceData((Array)new float[] {d});

            return ndArray;
        }

        public static implicit operator float(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            if (nd.dtype.Name == "NDArray")
                return nd.Array.GetIndex<float>(0);
            else
                return nd.GetSingle(0);
        }

        public static implicit operator NDArray(double d)
        {
            var ndArray = new NDArray(typeof(double), new int[0]);
            ndArray.Storage.ReplaceData((Array)new double[] {d});

            return ndArray;
        }

        public static implicit operator double(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<double>(0);
        }

        public static implicit operator NDArray(short d)
        {
            var ndArray = new NDArray(typeof(short), new int[0]);
            ndArray.Storage.ReplaceData((Array)new short[] {d});

            return ndArray;
        }

        public static implicit operator short(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<short>(0);
        }

        public static implicit operator NDArray(int d)
        {
            var ndArray = new NDArray(typeof(int), new int[0]);
            ndArray.Storage.ReplaceData((Array)new int[] {d});

            return ndArray;
        }

        public static implicit operator int(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<int>(0);
        }

        public static implicit operator NDArray(long d)
        {
            var ndArray = new NDArray(typeof(Int64), new int[0]);
            ndArray.Storage.ReplaceData((Array)new Int64[] {d});

            return ndArray;
        }

        public static implicit operator long(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<long>(0);
        }

        public static implicit operator NDArray(ulong d)
        {
            var ndArray = new NDArray(typeof(UInt64), new int[0]);
            ndArray.Storage.ReplaceData((Array)new UInt64[] {d});

            return ndArray;
        }

        public static implicit operator ulong(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<ulong>(0);
        }

        public static implicit operator NDArray(decimal d)
        {
            var ndArray = new NDArray(typeof(decimal), new int[0]);
            ndArray.Storage.ReplaceData((Array)new decimal[] {d});

            return ndArray;
        }

        public static implicit operator decimal(NDArray nd)
        {
            if (nd.ndim > 0)
                throw new IncorrectShapeException();

            return nd.Data<decimal>(0);
        }

        public static implicit operator NDArray(Complex d)
        {
            var ndArray = new NDArray(typeof(Complex), new int[0]);
            ndArray.Storage.ReplaceData((Array)new Complex[] {d});

            return ndArray;
        }

        /*public static implicit operator NDArray(Quaternion d)
        {
            var ndArray = new NDArray(typeof(Quaternion),new int[0]);
            ndArray.Storage.ReplaceData(new Quaternion[]{d});

            return ndArray;
        }*/
    }
}
