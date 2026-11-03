using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Stores one boxed element at the coordinates <paramref name="shape"/>'s dimensions name — NumPy 1.x's
        ///     <c>ndarray.itemset</c> (removed in NumPy 2.0, whose spelling is <c>a[i, j] = value</c>).
        /// </summary>
        /// <param name="shape">A shape whose DIMENSIONS are read as the element's coordinates (a legacy spelling).</param>
        /// <param name="val">The element, boxed as the dtype's exact CLR type (an unboxing, never a conversion), or
        ///     <c>null</c> for Python's <c>None</c>.</param>
        /// <exception cref="ValueError">The array is read-only (<c>assignment destination is read-only</c>).</exception>
        /// <exception cref="InvalidCastException">A non-null value that is not the dtype's exact CLR type.</exception>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing is written).</exception>
        public void itemset(ref Shape shape, object val)
        {
            SetValue(val, shape.dimensions);
        }

        /// <summary>
        ///     Stores one boxed element at the coordinates <paramref name="shape"/>'s dimensions name — NumPy 1.x's
        ///     <c>ndarray.itemset</c> (removed in NumPy 2.0, whose spelling is <c>a[i, j] = value</c>).
        /// </summary>
        /// <param name="shape">A shape whose DIMENSIONS are read as the element's coordinates (a legacy spelling).</param>
        /// <param name="val">The element, boxed as the dtype's exact CLR type (an unboxing, never a conversion), or
        ///     <c>null</c> for Python's <c>None</c>.</param>
        /// <exception cref="ValueError">The array is read-only (<c>assignment destination is read-only</c>).</exception>
        /// <exception cref="InvalidCastException">A non-null value that is not the dtype's exact CLR type.</exception>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing is written).</exception>
        public void itemset(Shape shape, object val)
        {
            SetValue(val, shape.dimensions);
        }

        /// <summary>
        ///     Stores one boxed element at the coordinates <paramref name="shape"/> — NumPy 1.x's <c>ndarray.itemset</c>
        ///     (removed in NumPy 2.0, whose spelling is <c>a[i, j] = value</c>).
        /// </summary>
        /// <param name="shape">The element's coordinates; a negative coordinate counts from the end of its axis.</param>
        /// <param name="val">The element, boxed as the dtype's exact CLR type (an unboxing, never a conversion), or
        ///     <c>null</c> for Python's <c>None</c>.</param>
        /// <exception cref="ValueError">The array is read-only (<c>assignment destination is read-only</c>).</exception>
        /// <exception cref="InvalidCastException">A non-null value that is not the dtype's exact CLR type.</exception>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing is written).</exception>
        public void itemset(int[] shape, object val)
        {
            SetValue(val, shape);
        }

        /// <summary>
        ///     Stores <paramref name="val"/> at the coordinates <paramref name="shape"/>, cast to the array's dtype with
        ///     NumPy's cast when <typeparamref name="T"/> is another type — NumPy 1.x's <c>ndarray.itemset</c> (removed in
        ///     NumPy 2.0, whose spelling is <c>a[i, j] = value</c>).
        /// </summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="shape">The element's coordinates; a negative coordinate counts from the end of its axis.</param>
        /// <param name="val">The value to store.</param>
        /// <exception cref="ValueError">The array is read-only (<c>assignment destination is read-only</c>).</exception>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing is written).</exception>
        /// <remarks>Before parity QA-12 a <typeparamref name="T"/> other than the dtype wrote <c>sizeof(T)</c> raw bytes
        ///     (the generic setter now converts).</remarks>
        public void itemset<T>(int[] shape, T val) where T : unmanaged
        {
            SetValue<T>(val, shape);
        }
    }
}
