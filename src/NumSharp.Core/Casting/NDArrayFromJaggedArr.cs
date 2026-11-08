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
using System.Globalization;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's shape discovery for a jagged C# array (<c>PyArray_DiscoverDTypeAndShape</c> over nested sequences, the
        ///     reading of <c>np.array([[1], [2, 3]])</c>): every child at every level is checked against its level's length,
        ///     so a ragged array raises NumPy's inhomogeneous-shape ValueError where the readers used to take the FIRST
        ///     child's length (a longer later row was cut silently, a shorter one threw IndexOutOfRange: parity QA-45).
        /// </summary>
        /// <param name="jagged">A jagged array (its element type an array type), or a plain 1-D array.</param>
        /// <returns>A fresh dimension array: one entry per nesting level, FEWER when a level holds no children — NumPy's
        ///     reading of nested lists ends at the first empty level (<c>[]</c> is <c>(0,)</c>, <c>[[], []]</c> is
        ///     <c>(2, 0)</c>), so the array those dims describe is empty.</returns>
        /// <exception cref="ValueError">A child whose length differs from the rest of its level, or a null child (NumPy reads
        ///     None among rows as a scalar where a sequence is expected): <c>setting an array element with a sequence. The
        ///     requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous
        ///     part.</c></exception>
        /// <remarks>
        ///     The walk is breadth first, so the shallowest inconsistency is the one reported — the same depth NumPy's
        ///     depth-first walk ends at, since it keeps lowering the agreed depth to the minimum over the whole input. It holds
        ///     one level's child references at a time (the elements themselves are copied by the caller afterwards).
        /// </remarks>
        internal static long[] DiscoverJaggedShape(Array jagged)
        {
            var dims = new List<long> { jagged.Length };
            var level = new List<Array> { jagged };
            var elementType = jagged.GetType().GetElementType();
            while (elementType is { IsArray: true })
            {
                var next = new List<Array>();
                long expected = -1;
                foreach (var parent in level)
                {
                    foreach (var item in parent)
                    {
                        // A null row is no sequence: NumPy's discovery sees a scalar beside sequences, the same refusal.
                        if (item is not Array child)
                            throw InhomogeneousJagged(dims);
                        if (expected < 0)
                            expected = child.Length;
                        else if (child.Length != expected)
                            throw InhomogeneousJagged(dims);
                        next.Add(child);
                    }
                }
                // No children at this level (every parent empty): there is nothing left to read the deeper lengths from.
                if (expected < 0)
                    break;
                dims.Add(expected);
                level = next;
                elementType = elementType.GetElementType();
            }
            return dims.ToArray();
        }

        /// <summary>
        ///     NumPy's ragged-input error for a jagged array: the dims every child agreed on, as a Python tuple repr
        ///     (<c>(2,)</c>, <c>(2, 2)</c>), culture-invariant.
        /// </summary>
        /// <param name="agreed">The dims agreed so far (the level whose children disagree comes next).</param>
        /// <returns>The exception to throw.</returns>
        private static ValueError InhomogeneousJagged(List<long> agreed)
        {
            int n = agreed.Count;
            var parts = new string[n];
            for (int k = 0; k < n; k++)
                parts[k] = agreed[k].ToString(CultureInfo.InvariantCulture);
            string shape = n == 1 ? $"({parts[0]},)" : "(" + string.Join(", ", parts) + ")";
            return new ValueError("setting an array element with a sequence. The requested array has an inhomogeneous shape " +
                                  string.Create(CultureInfo.InvariantCulture, $"after {n} dimensions. The detected shape was {shape} + inhomogeneous part."));
        }

        /// <summary>
        ///     A jagged C# array of <typeparamref name="T"/> (<c>T[][]</c> up to <c>T[][][][]</c>) into a new C-contiguous
        ///     array, through the typed <c>np.array</c> overloads: their shape is NumPy's discovery
        ///     (<see cref="DiscoverJaggedShape"/>: every child checked, a ragged array is NumPy's ValueError) and the elements
        ///     are copied one by one, which is slow for large arrays. The implicit conversion from a C# array
        ///     (<see cref="Backends.UnmanagedStorage.FromManagedArray(Array)"/>) dispatches here for a jagged array.
        /// </summary>
        /// <typeparam name="T">The innermost element type, one of NumSharp's dtypes.</typeparam>
        /// <param name="dotNetArray">The jagged array (its element type is an array type).</param>
        /// <returns>The new array.</returns>
        /// <exception cref="Exception"><paramref name="dotNetArray"/> is not jagged (a rectangular array).</exception>
        /// <exception cref="NotImplementedException">More than four levels of nesting.</exception>
        /// <exception cref="InvalidCastException">An innermost element type other than <typeparamref name="T"/>.</exception>
        /// <exception cref="ValueError">A ragged array (NumPy's inhomogeneous-shape text).</exception>
        internal static NDArray FromJaggedArray<T>(Array dotNetArray)
        {
            if (!dotNetArray.GetType().GetElementType().IsArray)
                throw new Exception("Multi dim arrays are not allowed here!");

            // The nesting depth comes from the TYPE (T[][] is 2 deep): reading it from the first child at each level threw for
            // an empty array (new int[0][] has no element 0) and a null first row; the typed overloads below discover and
            // check the lengths themselves.
            int depth = 1;
            for (var t = dotNetArray.GetType().GetElementType(); t is { IsArray: true }; t = t.GetElementType())
                depth++;

            switch (depth)
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
