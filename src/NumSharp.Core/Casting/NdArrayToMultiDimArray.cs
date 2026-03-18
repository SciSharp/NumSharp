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
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public T[] ToArray<T>() where T : unmanaged
        {
            return Storage.ToArray<T>();
        }

        public Array ToMuliDimArray<T>() where T : unmanaged
        {
            // Arrays.Create requires int[] - .NET limitation
            foreach (var d in shape)
            {
                if (d > int.MaxValue)
                    throw new InvalidOperationException($"Dimension {d} exceeds int.MaxValue. Cannot convert to .NET multi-dimensional array.");
            }
            var intShape = System.Array.ConvertAll(shape, d => (int)d);
            var ret = Arrays.Create(typeof(T), intShape);

            var iter = this.AsIterator<T>();
            var hasNext = iter.HasNext;
            var next = iter.MoveNext;
            var coorditer = new ValueCoordinatesIncrementor(shape);
            var indices = coorditer.Index;
            // .NET's Array.SetValue only accepts int[] indices, convert from long[]
            var intIndices = new int[indices.Length];

            while (hasNext())
            {
                for (int i = 0; i < indices.Length; i++)
                    intIndices[i] = (int)indices[i];
                ret.SetValue(next(), intIndices);
                if (coorditer.Next() == null)
                    break;
            }

            return ret;
        }


    }
    
}
