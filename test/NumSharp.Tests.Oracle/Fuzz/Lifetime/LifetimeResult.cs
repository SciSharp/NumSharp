using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using NumSharp;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     One recorded result item: NumPy's descriptor string, the shape, and the C-order bytes (or a byte string).
    /// </summary>
    /// <param name="Kind"><c>array</c> or <c>bytes</c>.</param>
    /// <param name="Dtype">The <c>dtype.str</c> of an array item; null for bytes.</param>
    /// <param name="Shape">The shape of an array item; null for bytes.</param>
    /// <param name="Bytes">The item's bytes.</param>
    internal sealed record LifetimeItem(string Kind, string Dtype, long[] Shape, byte[] Bytes)
    {
        /// <summary>The lower-case hex SHA-256 of <see cref="Bytes"/> (NumPy's recorded digest).</summary>
        public string Sha256 => Convert.ToHexString(SHA256.HashData(Bytes)).ToLowerInvariant();
    }

    /// <summary>
    ///     Records a call's result the way the generator's <c>record</c> does, and compares records.
    /// </summary>
    internal static class LifetimeResult
    {
        /// <summary>
        ///     Records a result: an array, a tuple of arrays (an <see cref="NDArray"/> array or a value tuple), a scalar
        ///     (as NumPy's 0-d array of its type), or a byte string.
        /// </summary>
        /// <param name="result">What the call returned.</param>
        /// <returns>The items (one for a single result).</returns>
        /// <exception cref="NotSupportedException">The result is of a type the tier never produces.</exception>
        public static List<LifetimeItem> Record(object result)
        {
            var items = new List<LifetimeItem>();
            switch (result)
            {
                case NDArray[] arr:
                    foreach (var a in arr) items.Add(Item(a));
                    break;
                case ITuple tuple when result is not NDArray:
                    for (int i = 0; i < tuple.Length; i++) items.Add(Item(tuple[i]));
                    break;
                default:
                    items.Add(Item(result));
                    break;
            }
            return items;
        }

        /// <summary>Records one item.</summary>
        /// <param name="x">An array, a byte string or a scalar.</param>
        /// <returns>The item.</returns>
        /// <exception cref="NotSupportedException">The value is of a type the tier never produces.</exception>
        private static LifetimeItem Item(object x) => x switch
        {
            byte[] b => new LifetimeItem("bytes", null, null, b),
            NDArray a => new LifetimeItem("array", a.dtype.str, a.shape.Select(d => (long)d).ToArray(), a.tobytes()),
            double d => new LifetimeItem("array", "<f8", Array.Empty<long>(), BitConverter.GetBytes(d)),
            long l => new LifetimeItem("array", "<i8", Array.Empty<long>(), BitConverter.GetBytes(l)),
            int i => new LifetimeItem("array", "<i4", Array.Empty<long>(), BitConverter.GetBytes(i)),
            bool b2 => new LifetimeItem("array", "|b1", Array.Empty<long>(), new[] { (byte)(b2 ? 1 : 0) }),
            null => throw new NotSupportedException("a lifetime call returned null"),
            _ => throw new NotSupportedException($"a lifetime call returned a {x.GetType().Name}")
        };

        /// <summary>
        ///     Compares a recorded result with NumPy's expected record.
        /// </summary>
        /// <param name="items">The recorded items.</param>
        /// <param name="expected">NumPy's record.</param>
        /// <param name="compareBytes">False to compare only kinds, dtypes and shapes (a <c>parity: false</c> case, or a
        ///     host-pinned case on another host).</param>
        /// <returns>Null when they agree, else a one-line description of the first difference.</returns>
        public static string CompareWithNumPy(List<LifetimeItem> items, LifetimeExpected expected, bool compareBytes)
        {
            var want = expected.Kind == "tuple" ? expected.Items : new[] { expected };
            if (items.Count != want.Length)
                return $"{items.Count} result item(s), NumPy has {want.Length}";
            for (int k = 0; k < want.Length; k++)
            {
                var got = items[k];
                var w = want[k];
                if (got.Kind != w.Kind)
                    return $"item {k}: kind {got.Kind}, NumPy {w.Kind}";
                if (w.Kind == "array")
                {
                    if (got.Dtype != w.Dtype)
                        return $"item {k}: dtype {got.Dtype}, NumPy {w.Dtype}";
                    if (!got.Shape.SequenceEqual(w.Shape ?? Array.Empty<long>()))
                        return $"item {k}: shape ({string.Join(",", got.Shape)}), NumPy ({string.Join(",", w.Shape ?? Array.Empty<long>())})";
                }
                else if (got.Bytes.LongLength != w.Len && compareBytes)
                    return $"item {k}: {got.Bytes.LongLength} bytes, NumPy {w.Len}";
                if (compareBytes && got.Sha256 != w.Sha256)
                    return $"item {k}: bytes differ from NumPy's (sha256 {got.Sha256[..12]}…, NumPy {w.Sha256[..12]}…)";
            }
            return null;
        }

        /// <summary>
        ///     Compares a stressed repetition's record with the reference record, byte for byte.
        /// </summary>
        /// <param name="items">The repetition's items.</param>
        /// <param name="reference">The reference run's items (all operands held).</param>
        /// <returns>Null when identical, else the index of the first differing item and the first differing byte.</returns>
        public static string CompareWithReference(List<LifetimeItem> items, List<LifetimeItem> reference)
        {
            if (items.Count != reference.Count)
                return $"{items.Count} item(s), the reference has {reference.Count}";
            for (int k = 0; k < items.Count; k++)
            {
                var a = items[k];
                var b = reference[k];
                if (a.Dtype != b.Dtype || (a.Shape != null && !a.Shape.SequenceEqual(b.Shape)))
                    return $"item {k}: dtype/shape {a.Dtype}({string.Join(",", a.Shape ?? Array.Empty<long>())}) vs reference {b.Dtype}({string.Join(",", b.Shape ?? Array.Empty<long>())})";
                if (!a.Bytes.AsSpan().SequenceEqual(b.Bytes))
                {
                    int first = a.Bytes.AsSpan().CommonPrefixLength(b.Bytes);
                    return $"item {k}: bytes differ from the reference from byte {first} of {b.Bytes.Length}";
                }
            }
            return null;
        }

        /// <summary>Disposes every array a result holds (a tuple's items included); scalars and byte strings own nothing.</summary>
        /// <param name="result">What the call returned.</param>
        /// <param name="keep">Arrays NOT to dispose (held operands a call may return as itself, e.g. <c>copyto</c>'s target).</param>
        public static void Dispose(object result, IReadOnlyCollection<NDArray> keep)
        {
            switch (result)
            {
                case NDArray a when !keep.Any(k => ReferenceEquals(k, a)):
                    a.Dispose();
                    break;
                case NDArray[] arr:
                    foreach (var a in arr)
                        if (a is not null && !keep.Any(k => ReferenceEquals(k, a)))
                            a.Dispose();
                    break;
                case ITuple tuple when result is not NDArray:
                    for (int i = 0; i < tuple.Length; i++)
                        if (tuple[i] is NDArray t && !keep.Any(k => ReferenceEquals(k, t)))
                            t.Dispose();
                    break;
            }
        }
    }
}
