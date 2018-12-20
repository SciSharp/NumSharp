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
        public Array ToJaggedArray<T>()
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
                    T[][] dotNetArrayPuffer = new T[shape.Dimensions[0]][]; 
                    for (int idx = 0; idx < dotNetArrayPuffer.Length;idx++)
                        dotNetArrayPuffer[idx] = new T[shape.Dimensions[1]];
                        
                    for (int idx = 0;idx < dotNetArrayPuffer.Length;idx++ )
                        for (int jdx = 0; jdx < dotNetArrayPuffer[0].Length;jdx++)
                            dotNetArrayPuffer[idx][jdx] = (T) this[idx,jdx];
                    
                    dotNetArray = dotNetArrayPuffer;
                    
                    break;
                }
                case 3 : 
                {
                    T[] data = Storage.GetData<T>();
                    T[][][] dotNetArrayPuffer = new T[shape.Dimensions[0]][][]; 
                    for (int idx = 0; idx < dotNetArrayPuffer.Length;idx++)
                    {
                        dotNetArrayPuffer[idx] = new T[shape.Dimensions[1]][];
                        for (int jdx = 0; jdx < dotNetArrayPuffer[idx].Length;jdx++)
                            dotNetArrayPuffer[idx][jdx] = new T[shape.Dimensions[2]];
                    }
                    
                    for (int idx = 0; idx < shape.Dimensions[0];idx++)
                        for (int jdx = 0;jdx < shape.Dimensions[1];jdx++)
                            for(int kdx = 0; kdx < shape.Dimensions[2];kdx++)
                                dotNetArrayPuffer[idx][jdx][kdx] = (T) this[idx,jdx,kdx];
                     
                    dotNetArray = dotNetArrayPuffer;
                    
                    break;
                }
                default : 
                {
                    throw new IncorrectShapeException();
                }

            }

            return dotNetArray;
        }
        
    }
}
