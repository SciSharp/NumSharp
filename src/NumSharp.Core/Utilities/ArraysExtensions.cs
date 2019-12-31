using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    public static class ArraysExtensions
    {
        /// <summary>
        ///     Slice an array.
        /// </summary>
        /// <remarks>Supports negative <paramref name="end"/> index</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static int[] CloneArray(this int[] source)
            => (int[])source.Clone();
    }
}
