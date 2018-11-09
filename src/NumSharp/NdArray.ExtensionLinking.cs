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
        public NDArray<double> HStack(params NDArray<double>[] nps)
        {
            return NumSharp.Extensions.NDArrayExtensions.HStack(nps);
        }
        public NDArray<T> reshape(params int[] shape)
        {
            return NumSharp.Extensions.NDArrayExtensions.reshape(this, shape);
        }
        public NDArray<T> VStack(params NDArray<T>[] nps)
        {
            return NumSharp.Extensions.NDArrayExtensions.VStack(nps);
        }
    }
}
