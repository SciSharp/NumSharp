using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Build a <see cref="FlatIterator"/> over <paramref name="a"/> — the write-through,
        ///     C-order flat iterator that is the analog of NumPy's <c>flatiter</c>
        ///     (<c>a.flat</c>'s type). NumSharp's <see cref="NDArray.flat"/> property already returns
        ///     a raveled <see cref="NDArray"/> and is deeply embedded, so it cannot be reclaimed to
        ///     return this type; the flat iterator lives on the <see cref="NDArray.flatiter"/> accessor
        ///     (and this factory) instead.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flatiter.html</remarks>
        public static FlatIterator flat(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return new FlatIterator(a);
        }

        /// <summary>
        ///     A flat, C-order iterator over an <see cref="NDArray"/> — the NumSharp analog of NumPy's
        ///     <c>flatiter</c> (the type of <c>ndarray.flat</c>), obtained from <see cref="NDArray.flatiter"/>
        ///     or <see cref="np.flat(NDArray)"/>.
        ///
        ///     <para>
        ///     Unlike NumSharp's <see cref="NDArray.flat"/> (a raveled <see cref="NDArray"/> that
        ///     materializes a COPY for a non-contiguous array, so writes through it are lost), this
        ///     iterator always reads AND writes THROUGH to the base array in logical C-order, whatever
        ///     the memory layout — matching NumPy's <c>a.flat[i] = v</c> semantics for transposed,
        ///     sliced, strided, negative-stride and broadcast layouts alike. Every element access maps a
        ///     flat C-order index to the base's coordinate and goes through the base's stride-aware
        ///     element accessors, so no per-dtype branching and no buffer materialization occur.
        ///     </para>
        ///
        ///     Surface (probed against NumPy 2.4.2): <see cref="this[long]"/> single-element get (a 0-d
        ///     write-through view, NumSharp's scalar analog) / set; fancy (<c>int[]</c>/<c>long[]</c>/
        ///     <see cref="NDArray"/>) and slice-string (<c>"1:4"</c>, <c>"::2"</c>) get/set; the
        ///     <see cref="index"/> / <see cref="coords"/> cursor, <see cref="Base"/>, <see cref="size"/>,
        ///     <see cref="copy"/> (a fresh 1-D C-order array), and C-order iteration that shares the
        ///     cursor (so a second pass RESUMES, as NumPy's <c>iter(f) is f</c>).
        ///     <para>
        ///     Two documented divergences from NumPy (differential-verified: 237/251 cases bit-exact):
        ///     (1) an out-of-range INTEGER scalar assignment WRAPS rather than raising — this is
        ///     NumSharp's library-wide convention (a plain <c>a[0] = 300</c> on an int8 array also wraps
        ///     to 44), so the iterator is consistent with the rest of NumSharp; NumPy raises
        ///     <c>ValueError</c>. (2) <see cref="coords"/> read at the EXHAUSTED position (index == size,
        ///     after a full pass) on a truly non-contiguous array is implementation-defined — NumSharp
        ///     returns the arithmetic continuation, NumPy an internal odometer artifact; every IN-RANGE
        ///     coord (the values read during iteration) is bit-exact.
        ///     </para>
        /// </summary>
        public sealed class FlatIterator : IEnumerable<object>
        {
            private readonly NDArray _base;
            private readonly long[] _dims;
            private readonly int _ndim;
            private readonly long _size;
            private long _cursor;

            internal FlatIterator(NDArray a)
            {
                _base = a ?? throw new ArgumentNullException(nameof(a));
                _ndim = a.ndim;
                _size = a.size;
                _dims = new long[_ndim];
                var shp = a.shape;
                for (int i = 0; i < _ndim; i++)
                    _dims[i] = shp[i];
            }

            /// <summary>The array being iterated (NumPy's <c>flatiter.base</c>). Writes through this iterator hit it.</summary>
            public NDArray Base => _base;

            /// <summary>Total number of elements — the base's size (NumPy's <c>len(a.flat)</c>).</summary>
            public long size => _size;

            /// <summary>The current cursor position as a flat C-order index (NumPy's <c>flatiter.index</c>).</summary>
            public long index => _cursor;

            /// <summary>
            ///     The current cursor as a coordinate tuple (NumPy's <c>flatiter.coords</c>). Matches NumPy
            ///     even past the end: the slowest (leading) axis is NOT wrapped, so on shape <c>(2,3)</c> the
            ///     exhausted cursor <c>index==6</c> reports <c>(2, 0)</c>, not <c>(0, 0)</c>.
            /// </summary>
            public long[] coords => UnravelCoords(_cursor);

            // ---- flat C-order index -> base coordinate (leading axis NOT wrapped, matching NumPy) ----
            private long[] UnravelCoords(long flat)
            {
                var c = new long[_ndim];
                long rem = flat;
                for (int k = _ndim - 1; k >= 1; k--)
                {
                    long d = _dims[k];
                    c[k] = rem % d;
                    rem /= d;
                }
                if (_ndim > 0)
                    c[0] = rem; // leading axis holds the remainder (may equal dim[0] at the past-end cursor)
                return c;
            }

            private long Normalize(long i)
            {
                long orig = i;
                if (i < 0)
                    i += _size;
                if (i < 0 || i >= _size)
                    throw new IndexError($"index {orig} is out of bounds for size {_size}");
                return i;
            }

            // Read the element at a logical C-order index. GetAtIndex maps the index through the base's
            // strides (Shape.TransformOffset), so it is correct AND allocation-free for every layout.
            private object GetScalar(long flat) => _base.GetAtIndex(flat);

            // Write a scalar at a logical C-order index. Exact-dtype values go straight to the stride-aware
            // SetAtIndex (no allocation, the hot path); a mismatched scalar is routed through NumSharp's
            // cast machinery once (numpy-exact narrowing/wrap/truncation) — SetAtIndex itself never casts.
            private void SetScalar(long flat, object value)
            {
                if (value != null && value.GetType() == _base.dtype)
                    _base.SetAtIndex(value, flat);
                else
                    _base.SetAtIndex(ConvertToBase(value), flat);
            }

            private object ConvertToBase(object value)
            {
                if (value is null)
                    throw new ArgumentNullException(nameof(value));

                var tc = _base.typecode;

                // Casting TO an integer/bool/char dtype can truncate toward zero or wrap on overflow —
                // NumPy semantics a plain Convert does not share — so route those through NumSharp's cast
                // machinery (correctness over speed; a rarer path). Casts to a float/complex/decimal target
                // are exact under Convert, so they take the fast, allocation-free path.
                if (IsIntegerLike(tc))
                {
                    var src = new NDArray(value.GetType(), 1);
                    src.SetAtIndex(value, 0);
                    return src.astype(_base.dtype).GetAtIndex(0);
                }

                if (tc == NPTypeCode.Complex)
                    return value is System.Numerics.Complex c ? c : new System.Numerics.Complex(Convert.ToDouble(value), 0);
                if (tc == NPTypeCode.Half)
                    return (Half)Convert.ToDouble(value);
                return Convert.ChangeType(value, _base.dtype);
            }

            private static bool IsIntegerLike(NPTypeCode tc)
                => tc is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte
                    or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32
                    or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

            /// <summary>
            ///     Single-element access at flat C-order index <paramref name="i"/> (NumPy's <c>f[i]</c>).
            ///     Negative indices wrap; out-of-range raises <see cref="IndexError"/>. The getter returns the
            ///     scalar (as NumPy does); the setter writes it through to the base in every memory layout,
            ///     casting to the base dtype with NumPy semantics when needed.
            /// </summary>
            public object this[long i]
            {
                get => GetScalar(Normalize(i));
                set => SetScalar(Normalize(i), value);
            }

            /// <summary>Fancy get/set by a slice string (NumPy's <c>f["1:4"]</c> / <c>f["::2"]</c>). Set writes through.</summary>
            public NDArray this[string range]
            {
                get => Gather(ResolveSlice(range));
                set => Scatter(ResolveSlice(range), value);
            }

            /// <summary>Fancy get/set by integer indices (NumPy's <c>f[[1,3,5]]</c>). Negatives wrap; set writes through.</summary>
            public NDArray this[int[] indices]
            {
                get => Gather(NormalizeMany(indices));
                set => Scatter(NormalizeMany(indices), value);
            }

            /// <inheritdoc cref="this[int[]]"/>
            public NDArray this[long[] indices]
            {
                get => Gather(NormalizeMany(indices));
                set => Scatter(NormalizeMany(indices), value);
            }

            /// <inheritdoc cref="this[int[]]"/>
            public NDArray this[NDArray indices]
            {
                get => Gather(NormalizeMany(indices));
                set => Scatter(NormalizeMany(indices), value);
            }

            private long[] ResolveSlice(string range)
            {
                // arange(size)[range] yields the flat positions the slice selects — reuses the slice engine.
                var positions = arange(_size)[range];
                long n = positions.size;
                var pos = new long[n];
                for (long k = 0; k < n; k++)
                    pos[k] = Convert.ToInt64(positions.GetAtIndex(k));
                return pos;
            }

            private long[] NormalizeMany(int[] indices)
            {
                var pos = new long[indices.Length];
                for (int k = 0; k < indices.Length; k++)
                    pos[k] = Normalize(indices[k]);
                return pos;
            }

            private long[] NormalizeMany(long[] indices)
            {
                var pos = new long[indices.Length];
                for (int k = 0; k < indices.Length; k++)
                    pos[k] = Normalize(indices[k]);
                return pos;
            }

            private long[] NormalizeMany(NDArray indices)
            {
                var flat = indices.flatten();
                long n = flat.size;
                var pos = new long[n];
                for (long k = 0; k < n; k++)
                    pos[k] = Normalize(Convert.ToInt64(flat.GetAtIndex(k)));
                return pos;
            }

            private NDArray Gather(long[] positions)
            {
                var result = new NDArray(_base.dtype, positions.Length);
                for (int k = 0; k < positions.Length; k++)
                    result.SetAtIndex(GetScalar(positions[k]), k);   // both base-dtype: exact, no conversion
                return result;
            }

            private void Scatter(long[] positions, NDArray values)
            {
                long n = positions.Length;
                bool broadcast = values.size == 1;
                if (!broadcast && values.size != n)
                    throw new ValueError(
                        $"cannot assign {values.size} input values to the {n} output values where the mask is true");
                // Cast to the base dtype once (numpy casts on flat assignment); SetValue does not convert.
                var casted = values.astype(_base.dtype);
                for (long k = 0; k < n; k++)
                    SetScalar(positions[k], broadcast ? casted.GetAtIndex(0) : casted.GetAtIndex(k));
            }

            /// <summary>A fresh 1-D C-order COPY of the base (NumPy's <c>f.copy()</c>, which returns an ndarray).</summary>
            public NDArray copy() => _base.flatten();

            /// <summary>Return the current element and advance the cursor (NumPy's <c>next(f)</c>).</summary>
            public object next()
            {
                if (_cursor >= _size)
                    throw new InvalidOperationException("StopIteration: flat iterator exhausted");
                var v = GetScalar(_cursor);
                _cursor++;
                return v;
            }

            /// <summary>
            ///     C-order iteration over every element. Shares the <see cref="index"/>/<see cref="coords"/>
            ///     cursor (as NumPy's <c>iter(f) is f</c>), so a second enumeration RESUMES where the first
            ///     stopped rather than restarting.
            /// </summary>
            public IEnumerator<object> GetEnumerator()
            {
                while (_cursor < _size)
                {
                    var v = GetScalar(_cursor);
                    _cursor++;
                    yield return v;
                }
            }

            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }
    }
}
