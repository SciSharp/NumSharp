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
        public Array ToJaggedArray<T>() where T : unmanaged
        {
            ArraySlice<T> data = Storage.GetData<T>();
            var shape = Shape;
            switch (ndim)
            {
                case 1:
                {
                    return data.ToArray();
                }

                case 2:
                {
                    T[][] ret = new T[shape[0]][];
                    for (int i = 0; i < ret.Length; i++)
                        ret[i] = new T[shape[1]];

                    for (int i = 0; i < ret.Length; i++)
                    for (int j = 0; j < ret[0].Length; j++)
                        ret[i][j] = GetAtIndex<T>(shape.GetOffset(i, j));

                    return ret;

                    break;
                }

                case 3:
                {
                    T[][][] ret = new T[shape[0]][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[shape[1]][];
                        for (int j = 0; j < ret[i].Length; j++)
                            ret[i][j] = new T[shape[2]];
                    }

                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                ret[i][j][k] = (T)GetValue(i, j, k);
                            }
                        }
                    }

                    return ret;
                }

                case 4:
                {
                    T[][][][] ret = new T[shape[0]][][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[shape[1]][][];
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            ret[i][j] = new T[shape[2]][];
                            for (int n = 0; n < ret[i].Length; n++)
                            {
                                ret[i][j][n] = new T[shape[3]];
                            }
                        }
                    }

                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
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
                    T[][][][][] ret = new T[shape[0]][][][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[shape[1]][][][];
                        for (int j = 0; j < ret[i].Length; j++)
                        {
                            ret[i][j] = new T[shape[2]][][];
                            for (int n = 0; n < ret[i].Length; n++)
                            {
                                ret[i][j][n] = new T[shape[3]][];
                                for (int k = 0; k < ret[i].Length; k++)
                                {
                                    ret[i][j][n][k] = new T[shape[4]];
                                }
                            }
                        }
                    }

                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    for (int m = 0; m < shape[4]; m++)
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
                    T[][][] ret = new T[shape[0]][][];
                    for (int i = 0; i < ret.Length; i++)
                    {
                        ret[i] = new T[shape[1]][];
                        for (int jdx = 0; jdx < ret[i].Length; jdx++)
                            ret[i][jdx] = new T[shape[2]];
                    }

                    for (int i = 0; i < shape[0]; i++)
                    {
                        for (int j = 0; j < shape[1]; j++)
                        {
                            for (int k = 0; k < shape[2]; k++)
                            {
                                for (int l = 0; l < shape[3]; l++)
                                {
                                    for (int m = 0; m < shape[4]; m++)
                                    {
                                        for (int n = 0; n < shape[5]; n++)
                                        {
                                            ret[i][j][k] = (T)GetValue(i, j, k);
                                        }
                                    }
                                }
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
