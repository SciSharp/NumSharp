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

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public class NDArrayOptimized<T>
    {
        public T[] Data { get; set; }

        public IList<int> Shape { get; set; }

        public int NDim { get { return Shape.Count; } }

        public int Size { get { return Data.Length; } }

        public int Length { get { return Shape[0]; } }

        public NDArrayOptimized()
        {
            Shape = new List<int>();
        }

        public T this[params int[] select]
        {
            get
            {
                int idx = 0;
                for(int i = 0; i < select.Length - 1; i++)
                {
                    int cnt = 1;
                    for (int s = i + 1; s < Shape.Count; s++)
                    {
                        cnt *= Shape[s];
                    }

                    idx += cnt * select[i];
                }
                idx += select[select.Length - 1];

                return Data[idx];
            }
        }

        public NDArrayOptimized<T> arange(int stop, int start = 0, int step = 1)
        {
            Shape = new List<int>() { stop };

            int index = 0;

            Data = Enumerable.Range(start, stop - start)
                                .Where(x => index++ % step == 0)
                                .Select(x => (T)TypeDescriptor.GetConverter(typeof(T)).ConvertFrom(x.ToString()))
                                .ToArray();
            return this;
        }

        public NDArrayOptimized<T> reshape(params int[] n)
        {
            Shape.Clear();
            Shape = n;

            return this;
        }

        public override string ToString()
        {
            string output = "array([";

            // loop
            for (int r = 0; r < Data.Length; r++)
            {
                output += (r == 0) ? Data[r] + "" : ", " + Data[r];
            }

            output += "])";

            return output;
        }
    }
}
