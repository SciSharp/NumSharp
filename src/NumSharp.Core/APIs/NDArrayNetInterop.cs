using System;
using System.Collections.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Fluent bridges from NumSharp iteration to built-in .NET types — the "other types
    ///     (foreach or non-foreach)" companions to the by-<c>ref T</c>/<c>Span&lt;T&gt;</c> walks and
    ///     to <see cref="NDArray._Unsafe"/>.
    ///
    ///     <para>
    ///     These are the SAFE side of the integration: <see cref="AsEnumerable{T}(NDArray)"/> gives
    ///     LINQ/collection interop with UNBOXED elements (only the enumerator is heap-allocated, once,
    ///     not each element), and the iterator <c>ToArray</c>/<c>CopyTo</c> extensions MATERIALIZE or
    ///     COPY into caller-owned .NET storage — so, unlike the aliasing <c>nd.Unsafe.*</c> views,
    ///     the result does not depend on the NDArray staying alive. For a zero-copy aliasing
    ///     <c>Span&lt;T&gt;</c>/<c>Memory&lt;T&gt;</c> over the buffer, use <c>nd.Unsafe</c>
    ///     (see <see cref="NDArray.Unsafe"/>).
    ///     </para>
    /// </summary>
    public static class NDArrayNetInterop
    {
        private static void RequireDtype<T>(NDArray a) where T : unmanaged
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (a.typecode != InfoOf<T>.NPTypeCode)
                throw new ArgumentException(
                    $"AsEnumerable<{typeof(T).Name}> called on a {a.dtype.type.Name} array. The element type must " +
                    $"match the dtype exactly (no conversion is performed) — cast first with a.astype(typeof({typeof(T).Name})).",
                    nameof(a));
        }

        /// <summary>
        ///     Walk the array's elements in logical C-order as an <see cref="IEnumerable{T}"/> — the
        ///     LINQ / <c>foreach</c> / collection bridge, with elements UNBOXED (<typeparamref name="T"/>,
        ///     not <c>object</c>). Works for EVERY memory layout (transposed, sliced, strided,
        ///     negative-stride, broadcast) — each element is read through the array's strides — so it
        ///     needs no contiguity and never copies the whole array. Only the enumerator is a heap
        ///     object (one allocation per enumeration); no per-element boxing.
        ///
        ///     <code>
        ///     double mean = a.AsEnumerable&lt;double&gt;().Average();
        ///     var evens   = a.AsEnumerable&lt;int&gt;().Where(x =&gt; x % 2 == 0).ToList();
        ///     </code>
        ///
        ///     For the fastest, allocation-free element walk prefer <c>foreach (ref T x in np.flat&lt;T&gt;(a))</c>;
        ///     reach for this when you need LINQ or an <see cref="IEnumerable{T}"/>-shaped API.
        /// </summary>
        /// <typeparam name="T">Must be the array's exact element type.</typeparam>
        public static IEnumerable<T> AsEnumerable<T>(this NDArray a) where T : unmanaged
        {
            RequireDtype<T>(a);         // fail fast, before deferred enumeration begins
            return Iterate(a);

            static IEnumerable<T> Iterate(NDArray a)
            {
                long n = a.size;
                for (long i = 0; i < n; i++)
                    yield return a.GetAtIndex<T>(i);   // maps flat C-order index through the strides
            }
        }

        // ------------------------------------------------------------------ iterator -> .NET buffer

        /// <summary>
        ///     Materialize the elements of an <see cref="np.NDRefIter{T}"/> (memory-order, from
        ///     <c>np.nditer&lt;T&gt;</c>) into a fresh <c>T[]</c>.
        /// </summary>
        public static T[] ToArray<T>(this np.NDRefIter<T> it) where T : unmanaged
        {
            var list = new List<T>();
            foreach (ref readonly T x in it)
                list.Add(x);
            return list.ToArray();
        }

        /// <summary>
        ///     Copy the elements of an <see cref="np.NDRefIter{T}"/> into <paramref name="destination"/>
        ///     (memory order). Returns the number of elements copied; throws if the destination is too
        ///     short.
        /// </summary>
        public static int CopyTo<T>(this np.NDRefIter<T> it, Span<T> destination) where T : unmanaged
        {
            int i = 0;
            foreach (ref readonly T x in it)
            {
                if (i >= destination.Length)
                    throw new ArgumentException("Destination is too short for the iterated elements.", nameof(destination));
                destination[i++] = x;
            }
            return i;
        }

        /// <summary>
        ///     Materialize the elements of a <see cref="np.FlatRefIter{T}"/> (logical C-order, from
        ///     <c>np.flat&lt;T&gt;</c>) into a fresh <c>T[]</c>.
        /// </summary>
        public static T[] ToArray<T>(this np.FlatRefIter<T> it) where T : unmanaged
        {
            var list = new List<T>();
            foreach (ref readonly T x in it)
                list.Add(x);
            return list.ToArray();
        }

        /// <summary>
        ///     Copy the elements of a <see cref="np.FlatRefIter{T}"/> into <paramref name="destination"/>
        ///     (logical C-order). Returns the number of elements copied; throws if too short.
        /// </summary>
        public static int CopyTo<T>(this np.FlatRefIter<T> it, Span<T> destination) where T : unmanaged
        {
            int i = 0;
            foreach (ref readonly T x in it)
            {
                if (i >= destination.Length)
                    throw new ArgumentException("Destination is too short for the iterated elements.", nameof(destination));
                destination[i++] = x;
            }
            return i;
        }

        /// <summary>
        ///     Materialize the chunks of an <see cref="np.NDChunkIter{T}"/> (from
        ///     <c>np.nditer_chunks&lt;T&gt;</c>) into a single fresh <c>T[]</c>, concatenated in
        ///     iteration (memory) order.
        /// </summary>
        public static T[] ToArray<T>(this np.NDChunkIter<T> it) where T : unmanaged
        {
            var list = new List<T>();
            foreach (Span<T> chunk in it)
                foreach (T v in chunk)
                    list.Add(v);
            return list.ToArray();
        }

        /// <summary>
        ///     Copy the chunks of an <see cref="np.NDChunkIter{T}"/> into <paramref name="destination"/>
        ///     (iteration/memory order). Returns the number of elements copied; throws if too short.
        /// </summary>
        public static int CopyTo<T>(this np.NDChunkIter<T> it, Span<T> destination) where T : unmanaged
        {
            int i = 0;
            foreach (Span<T> chunk in it)
            {
                if (i + chunk.Length > destination.Length)
                    throw new ArgumentException("Destination is too short for the iterated chunks.", nameof(destination));
                chunk.CopyTo(destination.Slice(i));
                i += chunk.Length;
            }
            return i;
        }
    }
}
