using System;
using System.Collections;

namespace NumSharp
{
    /// <summary>
    /// Container protocol implementation for NDArray.
    /// Provides Python-compatible container protocol methods:
    /// __contains__, __hash__, __len__, __iter__, __getitem__, __setitem__
    /// </summary>
    public partial class NDArray
    {
        #region Contains (__contains__)

        /// <summary>
        /// Returns true if value is found in the array (linear search).
        /// Equivalent to NumPy's <c>value in arr</c>.
        /// </summary>
        /// <param name="value">Value to search for.</param>
        /// <returns>True if value exists in the array.</returns>
        /// <remarks>
        /// This is a linear O(n) search. For sorted arrays, consider using np.searchsorted.
        /// NaN handling: NaN == NaN is false in IEEE 754, so Contains(float.NaN) returns false
        /// for arrays containing NaN. Use np.any(np.isnan(arr)) to check for NaN.
        /// </remarks>
        /// <example>
        /// <code>
        /// var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        /// arr.Contains(3);  // true
        /// arr.Contains(10); // false
        /// </code>
        /// </example>
        public bool Contains(object value)
        {
            if (value is null)
                return false;

            // NumPy: (self == el).any()
            // NumPy compares the raw object directly, not an array-converted version.
            // When comparing incompatible types (e.g., int array with string),
            // NumPy returns False for each element.

            // Check for fundamentally incompatible types that would cause shape mismatch
            // NumSharp's asanyarray("hello") creates char[5], not a scalar like NumPy
            // This causes broadcasting errors when comparing with e.g. int[3]
            if (value is string s)
            {
                // String in non-char array: incompatible types → return False
                // NumPy: 'hello' in np.array([1,2,3]) returns False
                if (typecode != NPTypeCode.Char)
                    return false;

                // For char arrays, compare with the string characters
                // but only if shapes are compatible (string length matches or is scalar)
                if (s.Length != 1 && ndim == 1 && shape[0] != s.Length)
                {
                    // Shape mismatch for non-scalar string comparison
                    // e.g., char[3] vs "hello" (length 5)
                    return false;
                }
            }

            // Convert value to NDArray for comparison
            var scalar = np.asanyarray(value);

            // Check for shape compatibility before comparing
            // NumPy throws ValueError for incompatible shapes, but for Contains
            // we want to return False for any shape that can't broadcast to a scalar-like comparison
            if (!scalar.Shape.IsScalar && scalar.ndim > 0)
            {
                // Non-scalar search value - check if shapes are broadcast-compatible
                // For Contains, we typically expect a scalar search value
                // If shapes don't match and can't broadcast, return false
                if (ndim == 1 && scalar.ndim == 1 && shape[0] != scalar.shape[0])
                    return false;
            }

            // Use element-wise comparison and check if any match
            // This handles type promotion and broadcasting correctly
            var comparison = this == scalar;
            return np.any(comparison);
        }

        /// <summary>
        /// Python-compatible __contains__ method.
        /// Equivalent to <see cref="Contains(object)"/>.
        /// </summary>
        /// <param name="value">Value to search for.</param>
        /// <returns>True if value exists in the array.</returns>
        /// <remarks>
        /// This method exists for Python interoperability and naming consistency.
        /// In Python: <c>value in arr</c> calls <c>arr.__contains__(value)</c>
        /// </remarks>
        public bool __contains__(object value) => Contains(value);

        #endregion

        #region Hash (__hash__)

        /// <summary>
        /// NDArray is unhashable because it is mutable.
        /// </summary>
        /// <returns>Never returns - always throws.</returns>
        /// <exception cref="NotSupportedException">Always thrown.</exception>
        /// <remarks>
        /// <para>
        /// NumPy arrays are unhashable because they are mutable. If an array were used
        /// as a dictionary key and then modified, the hash would change, breaking the
        /// dictionary's invariants.
        /// </para>
        /// <para>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> hash(np.array([1, 2, 3]))
        /// TypeError: unhashable type: 'numpy.ndarray'
        /// </code>
        /// </para>
        /// <para>
        /// Workarounds:
        /// <list type="bullet">
        /// <item>Use <c>arr.tobytes()</c> as a hashable key (immutable snapshot)</item>
        /// <item>Use <c>ReferenceEqualityComparer.Instance</c> for identity-based dictionaries</item>
        /// <item>Convert to tuple: <c>tuple(arr.ToArray())</c></item>
        /// </list>
        /// </para>
        /// </remarks>
        public override int GetHashCode()
        {
            throw new NotSupportedException(
                "NDArray is unhashable because it is mutable. " +
                "Workarounds: use arr.tobytes() as key, use ReferenceEqualityComparer.Instance, " +
                "or convert to an immutable type.");
        }

        /// <summary>
        /// Python-compatible __hash__ method.
        /// NDArray is unhashable because it is mutable.
        /// </summary>
        /// <returns>Never returns - always throws.</returns>
        /// <exception cref="NotSupportedException">Always thrown.</exception>
        /// <remarks>
        /// This method exists for Python interoperability and naming consistency.
        /// In Python: <c>hash(arr)</c> calls <c>arr.__hash__()</c>
        ///
        /// NumPy behavior:
        /// <code>
        /// >>> arr = np.array([1, 2, 3])
        /// >>> hash(arr)
        /// TypeError: unhashable type: 'numpy.ndarray'
        /// </code>
        /// </remarks>
        public int __hash__()
        {
            throw new NotSupportedException(
                "NDArray is unhashable because it is mutable. " +
                "Workarounds: use arr.tobytes() as key, use ReferenceEqualityComparer.Instance, " +
                "or convert to an immutable type.");
        }

        #endregion

        #region Length (__len__)

        /// <summary>
        /// Python-compatible __len__ method.
        /// Returns the length of the first dimension (like Python's len()).
        /// </summary>
        /// <returns>Length of the first dimension, or 1 for scalars.</returns>
        /// <remarks>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> len(np.array([1, 2, 3]))
        /// 3
        /// >>> len(np.array([[1, 2], [3, 4]]))
        /// 2  # First dimension
        /// >>> len(np.array(5))
        /// TypeError: len() of unsized object
        /// </code>
        ///
        /// Note: For scalars (0-d arrays), NumPy raises TypeError.
        /// NumSharp returns 1 for consistency with C# conventions.
        /// Use <see cref="size"/> for total element count.
        /// </remarks>
        public long __len__()
        {
            if (ndim == 0)
                throw new TypeError("len() of unsized object (0-d array)");
            return shape[0];
        }

        #endregion

        #region Iteration (__iter__)

        /// <summary>
        /// Python-compatible __iter__ method.
        /// Returns an enumerator over the first axis.
        /// </summary>
        /// <returns>Enumerator yielding NDArray slices along the first axis.</returns>
        /// <remarks>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> for row in np.array([[1, 2], [3, 4]]):
        /// ...     print(row)
        /// [1 2]
        /// [3 4]
        /// </code>
        ///
        /// For 1-D arrays, iterates over scalar elements.
        /// For N-D arrays, iterates over (N-1)-D slices.
        /// </remarks>
        public IEnumerator __iter__() => GetEnumerator();

        #endregion

        #region Indexing (__getitem__, __setitem__)

        /// <summary>
        /// Python-compatible __getitem__ method with integer index.
        /// </summary>
        /// <param name="index">Index along the first axis.</param>
        /// <returns>Element or slice at the given index.</returns>
        /// <remarks>
        /// Equivalent to <c>arr[index]</c> in Python.
        /// Supports negative indexing (-1 = last element).
        /// </remarks>
        public NDArray __getitem__(int index) => this[index];

        /// <summary>
        /// Python-compatible __getitem__ method with long index.
        /// </summary>
        /// <param name="index">Index along the first axis.</param>
        /// <returns>Element or slice at the given index.</returns>
        public NDArray __getitem__(long index) => this[index];

        /// <summary>
        /// Python-compatible __getitem__ method with slice string.
        /// </summary>
        /// <param name="slice">Slice specification (e.g., "1:3", "::-1", "..., 0").</param>
        /// <returns>Sliced view of the array.</returns>
        /// <remarks>
        /// Equivalent to <c>arr[slice]</c> in Python.
        /// Examples:
        /// <code>
        /// arr.__getitem__(":3")      // First 3 elements
        /// arr.__getitem__("1:-1")    // All but first and last
        /// arr.__getitem__("::-1")    // Reversed
        /// arr.__getitem__("..., 0")  // All rows, first column
        /// </code>
        /// </remarks>
        public NDArray __getitem__(string slice) => this[slice];

        /// <summary>
        /// Python-compatible __getitem__ method with params indices.
        /// </summary>
        /// <param name="indices">Indices for each dimension.</param>
        /// <returns>Element or slice at the given indices.</returns>
        public NDArray __getitem__(params int[] indices) => this[indices];

        /// <summary>
        /// Python-compatible __setitem__ method with integer index.
        /// </summary>
        /// <param name="index">Index along the first axis.</param>
        /// <param name="value">Value to set (scalar or NDArray).</param>
        public void __setitem__(int index, object value)
        {
            var arr = value as NDArray ?? np.asanyarray(value);
            this[index] = arr;
        }

        /// <summary>
        /// Python-compatible __setitem__ method with long index.
        /// </summary>
        /// <param name="index">Index along the first axis.</param>
        /// <param name="value">Value to set (scalar or NDArray).</param>
        public void __setitem__(long index, object value)
        {
            var arr = value as NDArray ?? np.asanyarray(value);
            this[index] = arr;
        }

        /// <summary>
        /// Python-compatible __setitem__ method with slice string.
        /// </summary>
        /// <param name="slice">Slice specification.</param>
        /// <param name="value">Value to set (scalar or NDArray).</param>
        public void __setitem__(string slice, object value)
        {
            var arr = value as NDArray ?? np.asanyarray(value);
            this[slice] = arr;
        }

        #endregion
    }
}
