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
    public partial class NDArray_Legacy<TData>
    {
        public NDArray_Legacy<TData> HStack(NDArray_Legacy<TData> np2 )
        {
            dynamic npDyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.HStack(npDyn,np2Dyn);
        }
        public NDArray_Legacy<TData> ARange(int stop, int start = 0, int step = 1)
        {
            dynamic npDyn = this;

            dynamic npResult = NumSharp.Extensions.NDArrayExtensions.ARange(npDyn, stop,start,step);

            return npResult;
        }
        public int ArgMax()
        {
            dynamic npDyn = this;

            return NumSharp.Extensions.NDArrayExtensions.ArgMax(npDyn);
        }
        public NDArray_Legacy<TData> Convolve(NDArray_Legacy<TData> np2, string mode = "full")
        {
            dynamic npDyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.Convolve(npDyn, np2Dyn,mode);
        }
        public NDArray_Legacy<TData> Dot(NDArray_Legacy<TData> np2)
        {
            dynamic np1Dyn = this;
            dynamic np2Dyn = np2;

            return NumSharp.Extensions.NDArrayExtensions.Dot(np1Dyn,np2Dyn);
        }
        public NDArray_Legacy<TData> Dot(TData scalar)
        {
            dynamic np1Dyn = this;
            dynamic scalarDyn = scalar;
            
            return NumSharp.Extensions.NDArrayExtensions.Dot(np1Dyn,scalarDyn);
        }
        public NDArray_Legacy<TData> Delete(IEnumerable<TData> delete)
        {
            return NumSharp.Extensions.NDArrayExtensions.Delete(this, delete);
        }
        public Matrix<double> AsMatrix()
        {
            dynamic np = this;
            return NumSharp.Extensions.NDArrayExtensions.AsMatrix(np);
        }
    }
}
