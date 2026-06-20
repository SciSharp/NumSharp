using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp
{
    /// <summary>
    ///     Test-only flat (1-D, C-order) element enumeration that replaces the removed
    ///     <c>NDArray.AsIterator&lt;T&gt;()</c> / <c>NDIterator</c> in the test suite.
    ///     Yields each logical element for ANY layout (contiguous / sliced / strided /
    ///     broadcast) via <see cref="NDArray.GetAtIndex(long)"/>, and also exposes the
    ///     stateful <c>MoveNext</c>/<c>HasNext</c> delegate surface a few call sites used.
    /// </summary>
    internal sealed class ElementSeq<T> : IEnumerable<T> where T : unmanaged
    {
        private readonly NDArray _nd;
        private long _cursor;

        public ElementSeq(NDArray nd) => _nd = nd;

        private static T Cast(object o) => o is T t ? t : (T)Convert.ChangeType(o, typeof(T));

        /// <summary>Advances a shared cursor (mirrors the old NDIterator.MoveNext delegate field).</summary>
        public Func<T> MoveNext => () => Cast(_nd.GetAtIndex(_cursor++));

        /// <summary>True while the shared cursor has not reached the end (mirrors NDIterator.HasNext).</summary>
        public Func<bool> HasNext => () => _cursor < _nd.size;

        public IEnumerator<T> GetEnumerator()
        {
            long n = _nd.size;
            for (long i = 0; i < n; i++)
                yield return Cast(_nd.GetAtIndex(i));
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    internal static class NdElementEnumerationExtensions
    {
        /// <summary>Flat, typed, C-order element enumeration (replaces AsIterator&lt;T&gt;()).</summary>
        public static ElementSeq<T> AsElements<T>(this NDArray nd) where T : unmanaged => new ElementSeq<T>(nd);

        /// <summary>Flat, boxed, C-order element enumeration (replaces the non-generic AsIterator()).</summary>
        public static IEnumerable<object> AsElements(this NDArray nd)
        {
            long n = nd.size;
            for (long i = 0; i < n; i++)
                yield return nd.GetAtIndex(i);
        }
    }
}
