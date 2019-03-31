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
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        public void FromMultiDimArray(Array dotNetArray)
        {
            if(dotNetArray.GetType().GetElementType().IsArray)
                throw new Exception("Jagged arrays are not allowed here!");

            int[] dims = new int[dotNetArray.Rank];

            for(int idx = 0; idx < dims.Length;idx++)
                dims[idx] = dotNetArray.GetLength(idx);

            Storage = new NDStorage();
            Storage.Allocate(dotNetArray.GetType().GetElementType(),new Shape(dims));

            Array internalStrg = Storage.GetData();

            var pufferShape = new Shape(dims);
            pufferShape.ChangeTensorLayout(); 

            int[] idxDims = null;
            object valueIdx = null;

            for(int idx = 0; idx < Storage.Shape.Size;idx++)
            {
                idxDims = pufferShape.GetDimIndexOutShape(idx);
                valueIdx = dotNetArray.GetValue(pufferShape.GetDimIndexOutShape(idx));
                internalStrg.SetValue(valueIdx,Storage.Shape.GetIndexInShape(idxDims));
            }
        }
        
    }
}
