using System.Collections;
using System.Collections.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of this array with elements at <paramref name="indices"/>
        ///     removed. Equivalent to <c>np.delete(this, indices, axis: null)</c> — the
        ///     array is flattened first, matching NumPy's <c>axis=None</c> behaviour.
        /// </summary>
        /// <param name="indices">Indices (any <see cref="IEnumerable"/> of integers).
        ///     Negative indices are normalised; duplicates are silently collapsed.</param>
        /// <returns>A new 1-D array with the selected elements removed.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.delete.html</remarks>
        public NDArray delete(IEnumerable indices)
        {
            var list = new List<long>();
            if (indices != null)
            {
                foreach (var item in indices)
                {
                    switch (item)
                    {
                        case long l:   list.Add(l); break;
                        case int  i:   list.Add(i); break;
                        case short s:  list.Add(s); break;
                        case byte b:   list.Add(b); break;
                        case sbyte sb: list.Add(sb); break;
                        case uint ui:  list.Add(ui); break;
                        case ushort us:list.Add(us); break;
                        case ulong ul: list.Add((long)ul); break;
                        default:
                            // Fall back to System.Convert for IConvertible numeric types.
                            list.Add(System.Convert.ToInt64(item));
                            break;
                    }
                }
            }

            return np.delete(this, list.ToArray(), axis: null);
        }
    }
}
