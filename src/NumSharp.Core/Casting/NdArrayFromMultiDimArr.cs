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
        /// <summary>
        /// low performance due to loop element-wise
        /// </summary>
        /// <param name="dotNetArray"></param>
        public NDArray FromMultiDimArray<T>(Array dotNetArray)
        {
            if(dotNetArray.GetType().GetElementType().IsArray)
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
            }

            throw new NotImplementedException("FromMultiDimArray<T>(Array dotNetArray)");
        }
        
    }
}
