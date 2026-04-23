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
            // Arrays.Create requires int[] — .NET limitation on array rank indexing.
            foreach (var d in shape)
            {
                if (d > int.MaxValue)
                    throw new InvalidOperationException($"Dimension {d} exceeds int.MaxValue. Cannot convert to .NET multi-dimensional array.");
            }
            var intShape = System.Array.ConvertAll(shape, d => (int)d);
            var ret = Arrays.Create(typeof(T), intShape);

            // Storage.ToArray<T>() already walks the NDArray in C-order and
            // produces a flat T[] that matches the row-major layout of the
            // .NET multi-dimensional array. For primitive types we then bulk
            // memcpy via Buffer.BlockCopy (several times faster than
            // Array.SetValue which does per-element runtime type checking).
            T[] flat = ToArray<T>();

            if (typeof(T) != typeof(decimal))
            {
                int byteCount = checked(flat.Length * dtypesize);
                Buffer.BlockCopy(flat, 0, ret, 0, byteCount);
                return ret;
            }

            // decimal is not a primitive — BlockCopy rejects it. Fall back to
            // the coordinate-walk + SetValue path for that one dtype.
            var coorditer = new ValueCoordinatesIncrementor(shape);
            var indices = coorditer.Index;
            var intIndices = new int[indices.Length];
            long flatIdx = 0;
            do
            {
                for (int i = 0; i < indices.Length; i++)
                    intIndices[i] = (int)indices[i];
                ret.SetValue(flat[flatIdx++], intIndices);
            } while (coorditer.Next() != null);

            return ret;
        }


    }

}
