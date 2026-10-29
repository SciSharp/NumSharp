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
using System.Linq;
using System.Text.RegularExpressions;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Converts a managed array (any rank, or jagged) into an <see cref="NDArray"/> copy of its data; a null array
        ///     converts to a null <see cref="NDArray"/>. Text — a <see cref="string"/> array, or an <c>object</c> array holding
        ///     strings / <c>byte[]</c> values — becomes NumPy's fixed-width string array (<c>np.array(['a', 'bb'])</c> is
        ///     <c>&lt;U2</c>), so <c>NDArray x = new[] { "a", "bb" }</c> and every <c>NDArray</c> parameter fed a C# string
        ///     array (C# prefers the <c>NDArray</c> overload to an <c>object</c> one) read the text as NumPy does.
        /// </summary>
        /// <param name="array">The array to convert, or null.</param>
        /// <returns>A new array holding a copy of the elements, or null when <paramref name="array"/> is null.</returns>
        /// <exception cref="NotSupportedException">The element type is not one of NumSharp's dtypes.</exception>
        /// <exception cref="ValueError">A ragged jagged array of text (NumPy's inhomogeneous-shape text).</exception>
        /// <remarks>
        ///     C# runs a user-defined conversion even for a null source, so without the null pass-through an optional
        ///     <see cref="NDArray"/> parameter fed from a null <c>double[]</c> variable (a common way to spell NumPy's
        ///     <c>p=None</c>) crashed with a <see cref="NullReferenceException"/> instead of meaning "not given".
        /// </remarks>
        public static implicit operator NDArray(Array array)
        {
            if (array is null)
                return null;

            return UnmanagedStorage.FromManagedArray(array);
        }

        public static explicit operator Array(NDArray nd)
        {
            return UnmanagedStorage.ToManagedArray(nd);
        }

        /// <summary>
        ///     A C# string is a Python <c>str</c>: the 0-d text array <c>np.array(str)</c> builds (<c>&lt;U{len}</c>), marked as
        ///     the Python str it stands for (<see cref="NDArray.WasPythonStr"/>), so <c>a[i] = "xy"</c> stores text into an
        ///     S / U array and <c>a[i] = "12"</c> parses it into a numeric one with NumPy's setitem texts.
        /// </summary>
        /// <param name="str">The text.</param>
        /// <returns>The 0-d text array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="str"/> is null.</exception>
        /// <remarks>
        ///     Until 2026-10-12 a text shaped like <c>"[1, 2, 3]"</c> or <c>"[1 2; 3 4]"</c> was parsed into a float64 vector or
        ///     matrix here (a legacy NumSharp shorthand). NumPy never reads a str as numbers in <c>np.array</c>
        ///     (<c>np.array('[1, 2, 3]')</c> is the 0-d <c>&lt;U9</c> text), and the owner decided (strings plan D6) to drop
        ///     the shorthand; <see cref="np.matrix(string, DType, bool)"/> keeps its own literal grammar (<c>"1 2; 3 4"</c>).
        /// </remarks>
        public static implicit operator NDArray(string str)
        {
            if (str is null)
                throw new ArgumentNullException(nameof(str));
            return NDArray.PythonStrLiteral(str);
        }
    }
}
