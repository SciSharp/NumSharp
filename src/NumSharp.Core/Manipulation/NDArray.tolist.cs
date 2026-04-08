using System.Collections.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the array as an (possibly nested) list.
        /// </summary>
        /// <returns>
        ///     The possibly nested list of array elements.
        ///     - For 0-d arrays (scalars): returns the scalar value itself
        ///     - For 1-d arrays: returns List&lt;object&gt; of elements
        ///     - For n-d arrays: returns nested List&lt;object&gt; structures
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
        ///
        ///     Copy of the array data as a (nested) Python list. Data items are converted
        ///     to the nearest compatible builtin Python type, via the item function.
        ///
        ///     If a.ndim is 0, then since the depth of the nested list is 0, it will not
        ///     be a list at all, but a simple Python scalar.
        /// </remarks>
        public object tolist()
        {
            // 0-d array (scalar): return the scalar value directly
            if (ndim == 0 || Shape.IsScalar)
                return Storage.GetAtIndex(0);

            // 1-d array: flat list of elements
            if (ndim == 1)
            {
                var result = new List<object>((int)System.Math.Min(size, int.MaxValue));
                for (long i = 0; i < size; i++)
                    result.Add(item(i));
                return result;
            }

            // n-d array: recursively build nested lists
            return ToListRecursive(this);
        }

        /// <summary>
        ///     Recursively converts an NDArray to nested lists.
        /// </summary>
        private static object ToListRecursive(NDArray arr)
        {
            // Base case: 1-d array returns flat list
            if (arr.ndim == 1)
            {
                var result = new List<object>((int)System.Math.Min(arr.size, int.MaxValue));
                for (long i = 0; i < arr.size; i++)
                    result.Add(arr.item(i));
                return result;
            }

            // Recursive case: iterate over first dimension
            var list = new List<object>((int)System.Math.Min(arr.shape[0], int.MaxValue));
            for (long i = 0; i < arr.shape[0]; i++)
            {
                // Get sub-array along first dimension
                var subArray = arr[i];
                list.Add(ToListRecursive(subArray));
            }
            return list;
        }
    }
}
