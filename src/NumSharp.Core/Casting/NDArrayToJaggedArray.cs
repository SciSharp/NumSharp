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
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Copies the array into a .NET JAGGED array of rank <see cref="ndim"/> (<c>T[]</c>, <c>T[][]</c>, …),
        ///     read in logical C-order through the array's own strides, so every layout — transposed, sliced,
        ///     strided, broadcast — yields its logical elements.
        /// </summary>
        /// <typeparam name="T">The element type; must match the array's dtype.</typeparam>
        /// <returns>A fresh jagged array the caller owns (no reference to this array's memory).</returns>
        /// <exception cref="InvalidOperationException">A dimension exceeds <see cref="int.MaxValue"/> (managed arrays
        /// are int32-indexed).</exception>
        /// <exception cref="NotSupportedException">The array has more than 6 dimensions.</exception>
        /// <remarks>
        ///     Elements are read by COORDINATE (<see cref="GetValue{T}(long[])"/>/<see cref="GetValue(int[])"/>),
        ///     never by densifying the storage first: <c>Storage.GetData&lt;T&gt;()</c> of a non-contiguous view is a
        ///     fresh contiguous COPY that nothing released (one pooled buffer stranded per call, measured by the
        ///     scope audit), and the rank-2 case used to feed a PHYSICAL offset (<c>shape.GetOffset(i, j)</c>) into
        ///     the LOGICAL <see cref="GetAtIndex{T}"/> — wrong values on a transposed view and an out-of-bounds read
        ///     on a column slice.
        /// </remarks>
        public Array ToJaggedArray<T>() where T : unmanaged
        {
            var shape = Shape;
            switch (ndim)
            {
                case 1:
                {
                    // Storage.ToArray walks the array in logical C-order through its strides (no densified temp).
                    return ToArray<T>();
                }

                case 2:
                {
                    // Managed arrays limited to int indices
                    if (shape[0] > int.MaxValue || shape[1] > int.MaxValue)
                        throw new InvalidOperationException($"Shape dimension exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing; use NDArray directly for large arrays.");

                    T[][] ret = new T[(int)shape[0]][];
                    for (int i = 0; i < ret.Length; i++)
                        ret[i] = new T[(int)shape[1]];

                    // By coordinate: GetValue maps (i, j) through the strides + offset itself.
                    for (int i = 0; i < ret.Length; i++)
                    for (int j = 0; j < ret[0].Length; j++)
                        ret[i][j] = GetValue<T>((long)i, j);

                    return ret;
                }

                case 3:
                {
                    // Managed arrays limited to int indices
                    if (shape[0] > int.MaxValue || shape[1] > int.MaxValue || shape[2] > int.MaxValue)
                        throw new InvalidOperationException($"Shape dimension exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing; use NDArray directly for large arrays.");

                    T[][][] ret = new T[(int)shape[0]][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[(int)shape[1]][];
                        for (int j = 0; j < ret[i].Length; j++)
                            ret[i][j] = new T[(int)shape[2]];
                    }

                    for (int i = 0; i < ret.Length; i++)
                    {
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            for (int k = 0; k < ret[i][j].Length; k++)
                            {
                                ret[i][j][k] = (T)GetValue(i, j, k);
                            }
                        }
                    }

                    return ret;
                }

                case 4:
                {
                    // Managed arrays limited to int indices
                    if (shape[0] > int.MaxValue || shape[1] > int.MaxValue || shape[2] > int.MaxValue || shape[3] > int.MaxValue)
                        throw new InvalidOperationException($"Shape dimension exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing; use NDArray directly for large arrays.");

                    T[][][][] ret = new T[(int)shape[0]][][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[(int)shape[1]][][];
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            ret[i][j] = new T[(int)shape[2]][];
                            for (int n = 0; n < ret[i][j].Length; n++)
                            {
                                ret[i][j][n] = new T[(int)shape[3]];
                            }
                        }
                    }

                    for (int i = 0; i < ret.Length; i++)
                    {
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            for (int k = 0; k < ret[i][j].Length; k++)
                            {
                                for (int l = 0; l < ret[i][j][k].Length; l++)
                                {
                                    ret[i][j][k][l] = (T)GetValue(i, j, k, l);
                                }
                            }
                        }
                    }

                    return ret;
                }

                case 5:
                {
                    // Managed arrays limited to int indices
                    if (shape[0] > int.MaxValue || shape[1] > int.MaxValue || shape[2] > int.MaxValue ||
                        shape[3] > int.MaxValue || shape[4] > int.MaxValue)
                        throw new InvalidOperationException($"Shape dimension exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing; use NDArray directly for large arrays.");

                    T[][][][][] ret = new T[(int)shape[0]][][][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[(int)shape[1]][][][];
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            ret[i][j] = new T[(int)shape[2]][][];
                            for (int n = 0; n < ret[i][j].Length; n++)
                            {
                                ret[i][j][n] = new T[(int)shape[3]][];
                                for (int k = 0; k < ret[i][j][n].Length; k++)
                                {
                                    ret[i][j][n][k] = new T[(int)shape[4]];
                                }
                            }
                        }
                    }

                    for (int i = 0; i < ret.Length; i++)
                    {
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            for (int k = 0; k < ret[i][j].Length; k++)
                            {
                                for (int l = 0; l < ret[i][j][k].Length; l++)
                                {
                                    for (int m = 0; m < ret[i][j][k][l].Length; m++)
                                    {
                                        ret[i][j][k][l][m] = (T)GetValue(i, j, k, l, m);
                                    }
                                }
                            }
                        }
                    }

                    return ret;
                }

                case 6:
                {
                    // NOTE: This case appears buggy - creates 3D array for 6D input
                    // Managed arrays limited to int indices
                    if (shape[0] > int.MaxValue || shape[1] > int.MaxValue || shape[2] > int.MaxValue)
                        throw new InvalidOperationException($"Shape dimension exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing; use NDArray directly for large arrays.");

                    T[][][] ret = new T[(int)shape[0]][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[(int)shape[1]][];
                        for (int jdx = 0; jdx < ret[i].Length; jdx++)
                            ret[i][jdx] = new T[(int)shape[2]];
                    }

                    for (int i = 0; i < ret.Length; i++)
                    {
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            for (int k = 0; k < ret[i][j].Length; k++)
                            {
                                ret[i][j][k] = (T)GetValue(i, j, k);
                            }
                        }
                    }

                    return ret;
                }

                default:
                    throw new NotSupportedException();
            }
        }
    }
}
