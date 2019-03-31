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
using Array = ArrayFire.Array;

namespace NumSharp
{
    public partial class NDArray
    {
        public static implicit operator Array(NDArray nd)
        {
            switch (nd.ndim)
            {
                case 1:
                    return ArrayFire.Data.CreateArray(nd.Data<int>());
                case 2:
                    return ArrayFire.Data.CreateArray(nd.ToMuliDimArray<int>() as int[,]);
            }

            throw new NotImplementedException("");
        }

        public static implicit operator NDArray(Array array)
        {
            throw new NotImplementedException("");
        }
    }
}
