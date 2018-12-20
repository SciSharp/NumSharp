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
using NumSharp.Core;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public Array ToMuliDimArray<T>()
        {
            Array dotNetArray = null;

            switch (ndim)
            {
                case 1 : 
                {
                    dotNetArray = Storage.GetData<T>() ;    
                    break;
                }
                case 2 : 
                {
                    T[] data = Storage.GetData<T>();
                    T[,] dotNetArrayPuffer = new T[shape.Dimensions[0],shape.Dimensions[1]];
                    for (int idx = 0;idx < data.Length;idx++ )
                        dotNetArrayPuffer[idx/shape.Dimensions[1],idx%shape.Dimensions[1]] = data[idx];
                    
                    dotNetArray = dotNetArrayPuffer;
                    
                    break;
                }

            }

            return dotNetArray;
        }
        
    }
}
