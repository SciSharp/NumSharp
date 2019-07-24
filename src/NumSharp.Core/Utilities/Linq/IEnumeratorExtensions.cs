using System.Collections.Generic;

namespace NumSharp.Utilities.Linq
{
    public static class IEnumeratorExtensions
    {
        /// <summary>
        ///     Turns <see cref="IEnumerator{T}"/> to an <see cref="IEnumerable{T}"/>.
        /// </summary>
        public static IEnumerable<T> ToIEnumerable<T>(this IEnumerator<T> enumerator)
        {
            while (enumerator.MoveNext())
            {
                yield return enumerator.Current;
            }
        }
    }
}
