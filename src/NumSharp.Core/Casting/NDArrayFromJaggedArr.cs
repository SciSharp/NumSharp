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

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     A jagged C# array of <typeparamref name="T"/> (<c>T[][]</c> up to <c>T[][][][]</c>) into a new C-contiguous
        ///     array: the shape is read from the FIRST child at every level (a ragged array is not detected: a shorter later
        ///     child throws while it is copied, a longer one is truncated) and the elements are copied one by one, which is
        ///     slow for large arrays. The implicit conversion from a C# array
        ///     (<see cref="Backends.UnmanagedStorage.FromManagedArray(Array)"/>) dispatches here for a jagged array.
        /// </summary>
        /// <typeparam name="T">The innermost element type, one of NumSharp's dtypes.</typeparam>
        /// <param name="dotNetArray">The jagged array (its element type is an array type).</param>
        /// <returns>The new array.</returns>
        /// <exception cref="Exception"><paramref name="dotNetArray"/> is not jagged (a rectangular array).</exception>
        /// <exception cref="NotImplementedException">More than four levels of nesting.</exception>
        /// <exception cref="InvalidCastException">An innermost element type other than <typeparamref name="T"/>.</exception>
        internal static NDArray FromJaggedArray<T>(Array dotNetArray)
        {
            if (!dotNetArray.GetType().GetElementType().IsArray)
                throw new Exception("Multi dim arrays are not allowed here!");

            var dimList = new List<int>();

            dimList.Add(dotNetArray.Length);

            object currentArr = dotNetArray;

            while (currentArr.GetType().GetElementType().IsArray)
            {
                Array child = (Array)((Array)currentArr).GetValue(0);
                dimList.Add(child.Length);
                currentArr = child;
            }

            switch (dimList.Count)
            {
                case 1:
                    return np.array((T[])dotNetArray);
                case 2:
                    return np.array((T[][])dotNetArray);
                case 3:
                    return np.array((T[][][])dotNetArray);
                case 4:
                    return np.array((T[][][][])dotNetArray);
            }

            throw new NotImplementedException("FromJaggedArray<T>(Array dotNetArray)");
        }
    }
}
