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
        /// <param name="dotNetArray"></param>
        public static NDArray FromMultiDimArray<T>(Array dotNetArray)
        {
            if (dotNetArray.GetType().GetElementType().IsArray)
                throw new Exception("Jagged arrays are not allowed here!");

            switch (dotNetArray.Rank)
            {
                case 1:
                    return np.array((T[])dotNetArray);
                case 2:
                    return np.array((T[,])dotNetArray);
                case 3:
                    return np.array((T[,,])dotNetArray);
                case 4:
                    return np.array((T[,,,])dotNetArray);
                case 5:
                    return np.array((T[,,,,])dotNetArray);
                case 6:
                    return np.array((T[,,,,,])dotNetArray);
                case 7:
                    return np.array((T[,,,,,,])dotNetArray);
                case 8:
                    return np.array((T[,,,,,,,])dotNetArray);
                case 9:
                    return np.array((T[,,,,,,,,])dotNetArray);
                case 10:
                    return np.array((T[,,,,,,,,,])dotNetArray);
                case 11:
                    return np.array((T[,,,,,,,,,,])dotNetArray);
                case 12:
                    return np.array((T[,,,,,,,,,,,])dotNetArray);
            }

            throw new NotImplementedException("FromMultiDimArray<T>(Array dotNetArray)");
        }
    }
}
