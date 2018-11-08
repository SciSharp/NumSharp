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

using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArray<T>
    {
        public NDArray<T> HStack(NDArray<T> np2 )
        {
            dynamic npDyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.HStack(npDyn,np2Dyn);
        }

        public NDArray<double> AMin(int? axis = null)
        {
            dynamic npDyn = this;
            return NumSharp.Extensions.NDArrayExtensions.AMin(npDyn, axis);
        }

        public int ArgMax()
        {
            dynamic npDyn = this;

            return NumSharp.Extensions.NDArrayExtensions.ArgMax(npDyn);
        }
        public Matrix<double> AsMatrix()
        {
            dynamic np = this;
            return NumSharp.Extensions.NDArrayExtensions.AsMatrix(np);
        }
        public NDArray<T> Convolve(NDArray<T> np2, string mode = "full")
        {
            dynamic npDyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.Convolve(npDyn, np2Dyn,mode);
        }
        public NDArray<T> Dot(NDArray<T> np2)
        {
            dynamic np1Dyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.Dot(np1Dyn,np2Dyn);
        }
        public NDArray<T> Dot(T scalar)
        {
            dynamic np1Dyn = this;
            dynamic scalarDyn = scalar;
            
            return NumSharp.Extensions.NDArrayExtensions.Dot(np1Dyn,scalarDyn);
        }
        public NDArray<double> HStack(params NDArray<double>[] nps)
        {
            return NumSharp.Extensions.NDArrayExtensions.HStack(nps);
        }
        public NDArray<T> Onces(params int[] shape)
        {
            return NumSharp.Extensions.NDArrayExtensions.Onces(this, shape);
        }
        public NDArray<T> ReShape(params int[] shape)
        {
            return NumSharp.Extensions.NDArrayExtensions.ReShape(this, shape);
        }
        public NDArray<T> VStack(params NDArray<T>[] nps)
        {
            return NumSharp.Extensions.NDArrayExtensions.VStack(nps);
        }
        public NDArray<T> Zeros(params int[] select)
        {
            dynamic np1Dyn = this;

            return NumSharp.Extensions.NDArrayExtensions.Zeros(np1Dyn, select);
        }
    }
}
