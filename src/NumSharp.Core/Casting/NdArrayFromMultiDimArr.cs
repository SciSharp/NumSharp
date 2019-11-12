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

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Creates an NDArray out of given array of type <typeparamref name="T"/>
        /// </summary>
        /// <param name="ndarray"></param>
        /// <param name="copy">true for making </param>
        public static NDArray FromMultiDimArray<T>(Array ndarray, bool copy = true) where T : unmanaged
        {
            if (ndarray == null)
                throw new ArgumentNullException(nameof(ndarray));

            var elem = ndarray.GetType().GetElementType();
            if (elem == null || elem.IsArray)
                throw new ArgumentException("Given array is not a multi-dimensional array (e.g. T[,,]).");

            if (elem != typeof(T))
                throw new ArgumentException("T constraint must match the element type of the given array.");

            switch (ndarray.Rank)
            {
                case 1:
                    return np.array((T[])ndarray, copy);
                case 2:
                    return np.array((T[,])ndarray, copy);
                case 3:
                    return np.array((T[,,])ndarray, copy);
                case 4:
                    return np.array((T[,,,])ndarray, copy);
                case 5:
                    return np.array((T[,,,,])ndarray, copy);
                case 6:
                    return np.array((T[,,,,,])ndarray, copy);
                case 7:
                    return np.array((T[,,,,,,])ndarray, copy);
                case 8:
                    return np.array((T[,,,,,,,])ndarray, copy);
                case 9:
                    return np.array((T[,,,,,,,,])ndarray, copy);
                case 10:
                    return np.array((T[,,,,,,,,,])ndarray, copy);
                case 11:
                    return np.array((T[,,,,,,,,,,])ndarray, copy);
                case 12:
                    return np.array((T[,,,,,,,,,,,])ndarray, copy);
                case 13:
                    return np.array((T[,,,,,,,,,,,,])ndarray, copy);
                case 14:
                    return np.array((T[,,,,,,,,,,,,,])ndarray, copy); 
                case 15:
                    return np.array((T[,,,,,,,,,,,,,,])ndarray, copy);
                case 16:
                    return np.array((T[,,,,,,,,,,,,,,,])ndarray, copy);
            }

            throw new NotSupportedException();
        }
    }
}
