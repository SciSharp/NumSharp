using System;
using System.Collections.Generic;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's NEP 55 variable-width string DType class, <c>numpy.dtypes.StringDType</c> (kind and char <c>'T'</c>, type
    ///     number 2056): every element is a 16-byte cell that holds a short UTF-8 string inline or points into an arena /
    ///     heap the array owns, so one array can hold strings of any length without padding them all to the longest.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Not a legacy class.</b> Unlike the fixed-width <see cref="StringDTypeMeta"/> pair, <c>StringDType</c> is a
    ///     NEP 42 class: its descriptors are never builtin singletons (<c>np.dtype('T') is np.dtype('T')</c> is False),
    ///     report <c>isbuiltin == 0</c>, print their parameters (<c>StringDType(na_object=None)</c>) as both <c>str()</c>
    ///     and <c>repr()</c>, have no byte order (<c>'|'</c>; <c>newbyteorder</c> raises), and compare equal only to
    ///     another <c>StringDType</c> descriptor with the same <c>na_object</c> and <c>coerce</c> — a dtype STRING is not
    ///     coerced (<c>np.dtype('T') == 'T'</c> is False).
    ///     </para>
    ///     <para>
    ///     <b>Parameters</b> (<see cref="VStringParameters"/>): <c>na_object</c>, the optional missing-data sentinel (NumPy
    ///     keeps a Python object; NumSharp keeps the C# value under the boundary map of <see cref="NDPyObject"/>), and
    ///     <c>coerce</c>. Create descriptors with <see cref="np.dtypes.StringDType(bool)"/> /
    ///     <see cref="np.dtypes.StringDType(object, bool)"/> or <c>np.dtype("T")</c>.
    ///     </para>
    ///     <para>
    ///     <b>One storage lane with <c>S</c> / <c>U</c> (decision D4).</b> Arrays of this class use the
    ///     <see cref="NPTypeCode.String"/> lane, which therefore no longer says how an element is laid out: the descriptor's
    ///     class does. Every route that moves elements as opaque fixed-width blobs is gated on <see cref="StringDTypeMeta"/>,
    ///     never on the type code, and a <c>StringDType</c> array takes its own, reference-aware paths (P1.3 / P1.4 of
    ///     <c>docs/plans/strings-phase1-completion.md</c>). The class owns neither the lane nor the C# <see cref="string"/>
    ///     type in the registry: <c>(DType)NPTypeCode.String</c> and <c>np.dtype(typeof(string))</c> stay <c>&lt;U</c>.
    ///     </para>
    ///     <para>
    ///     <b>The slots</b> are ports of <c>numpy/_core/src/multiarray/stringdtype/dtype.c</c>: <c>common_dtype</c> (only
    ///     <c>StrDType</c> promotes to it — <c>T</c> + <c>U</c> is <c>T</c>, <c>T</c> + <c>S</c> or a number has no common
    ///     DType), <c>common_instance</c> (compatible missing-data objects, the FIRST descriptor's <c>coerce</c>),
    ///     <c>string_discover_descriptor_from_pyobject</c>, <c>stringdtype_ensure_canonical</c> (a descriptor is always
    ///     canonical), <c>stringdtype_is_known_scalar_type</c> and the default descriptor <c>StringDType()</c>.
    ///     </para>
    /// </remarks>
    public sealed class VStringDTypeMeta : DTypeMeta
    {
        /// <summary>
        ///     NumPy's <c>SIZEOF_NPY_PACKED_STATIC_STRING</c> (<c>2 * sizeof(size_t)</c>, <c>stringdtype/dtype.h</c>): the bytes
        ///     of one element cell, the <c>itemsize</c> of every <c>StringDType</c> descriptor.
        /// </summary>
        public const int SIZEOF_NPY_PACKED_STATIC_STRING = 16;

        /// <summary>
        ///     NumPy's <c>ALIGNOF_NPY_PACKED_STATIC_STRING</c> (<c>_Alignof(size_t)</c>): the cell holds a pointer-sized field,
        ///     so a descriptor's <c>alignment</c> is 8.
        /// </summary>
        public const int ALIGNOF_NPY_PACKED_STATIC_STRING = 8;

        /// <summary>
        ///     NumPy's descriptor flags for <c>StringDType</c>: <c>NPY_ITEM_REFCOUNT | NPY_LIST_PICKLE | NPY_NEEDS_INIT |
        ///     NPY_USE_GETITEM | NPY_USE_SETITEM</c> = 1 + 2 + 8 + 32 + 64 = 107.
        /// </summary>
        public const int StringDTypeFlags = 107;

        /// <summary>Builds the class. Called once by <see cref="DTypeRegistry"/>.</summary>
        internal VStringDTypeMeta()
            : base("StringDType", "numpy.dtypes", 2056, typeof(string), "str", NPTypeCode.String,
                kind: 'T', typeChar: 'T', itemSize: SIZEOF_NPY_PACKED_STATIC_STRING,
                alignment: ALIGNOF_NPY_PACKED_STATIC_STRING, DTypeFlags.NPY_DT_PARAMETRIC)
        {
        }

        /// <summary><c>dtype.hasobject</c> is True: a cell can own heap memory that must be freed (NumPy's clear loop).</summary>
        public override bool HasReferences => true;

        /// <summary>The descriptor flags NumPy reports for every <c>StringDType</c> (<see cref="StringDTypeFlags"/>).</summary>
        public override int ItemFlags => StringDTypeFlags;

        /// <summary>A cell is not byte-swappable data: every descriptor reports <c>'|'</c>.</summary>
        internal override bool ByteOrderNotApplicable => true;

        /// <summary>The <see cref="NPTypeCode.String"/> lane belongs to <c>StrDType</c>; this class shares it (decision D4).</summary>
        internal override bool OwnsTypeCode => false;

        /// <summary>The C# <see cref="string"/> keeps resolving to <c>StrDType</c> (<c>np.dtype(typeof(string))</c> is <c>&lt;U</c>).</summary>
        internal override bool OwnsScalarType => false;

        /// <summary>
        ///     A FRESH <c>StringDType()</c> (no missing-data object, coercion on) — NumPy has no builtin singleton for this
        ///     class, because each descriptor instance owns its own string allocator.
        /// </summary>
        /// <returns>A new default descriptor.</returns>
        public override DType default_descr() => new DType(this, VStringParameters.Default);

        /// <summary>
        ///     A new descriptor with the given parameters — NumPy's <c>new_stringdtype_instance(na_object, coerce)</c>.
        /// </summary>
        /// <param name="hasNAObject">True when a missing-data object is given (even <c>None</c> = <see langword="null"/>).</param>
        /// <param name="naObject">The missing-data object (ignored unless <paramref name="hasNAObject"/>).</param>
        /// <param name="coerce">NumPy's <c>coerce</c>.</param>
        /// <returns>A fresh descriptor.</returns>
        /// <exception cref="UnicodeEncodeError">A string missing-data object (or the <c>str()</c> of another one) holds a
        ///     lone surrogate (NumPy keeps both as UTF-8).</exception>
        internal DType Descr(bool hasNAObject, object naObject, bool coerce)
            => hasNAObject || !coerce ? new DType(this, new VStringParameters(hasNAObject, naObject, coerce)) : default_descr();

        /// <summary><c>stringdtype_ensure_canonical</c>: every <c>StringDType</c> descriptor is canonical (it is returned as is).</summary>
        /// <param name="descr">A descriptor of this class.</param>
        /// <returns><paramref name="descr"/> itself.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
        public override DType ensure_canonical(DType descr) => descr ?? throw new ArgumentNullException(nameof(descr));

        /// <summary>
        ///     NumPy's <c>common_dtype</c> slot: <c>StrDType</c> promotes to <c>StringDType</c> (there is a cast from it);
        ///     every other class defers (<see langword="null"/>, NumPy's <c>NotImplemented</c>), so <c>T</c> with <c>S</c>, a
        ///     number, bool or a datetime has no common DType.
        /// </summary>
        /// <param name="other">The class to promote with.</param>
        /// <returns>This class for <c>StrDType</c> (and for itself), else null.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
        public override DTypeMeta common_dtype(DTypeMeta other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            if (ReferenceEquals(other, this))
                return this;
            /*
             *  We have a cast from unicode, so allow unicode to promote
             *  to PyArray_StringDType
             */
            if (other is StringDTypeMeta str && str.IsUnicode)
                return this;
            return null;
        }

        /// <summary>
        ///     NumPy's <c>common_instance</c>: the two descriptors' missing-data objects must be compatible
        ///     (<see cref="VStringParameters.CompatibleNA"/>); the result takes the present one and the FIRST descriptor's
        ///     <c>coerce</c> (NumPy's <c>dtype1-&gt;coerce &amp;&amp; dtype1-&gt;coerce</c>, which reads the first operand twice).
        /// </summary>
        /// <param name="descr1">A descriptor of this class.</param>
        /// <param name="descr2">A descriptor of this class.</param>
        /// <returns>A fresh descriptor.</returns>
        /// <exception cref="ArgumentNullException">Either descriptor is null.</exception>
        /// <exception cref="ArgumentException">Either descriptor is not a <c>StringDType</c> descriptor (a caller error: the
        ///     promotion engine casts both operands to this class first).</exception>
        /// <exception cref="TypeError">Both have missing-data objects and they differ: NumPy's
        ///     <c>Cannot find common instance for incompatible dtypes 'StringDType(na_object=None)' and 'StringDType(na_object=nan)'</c>.</exception>
        public override DType common_instance(DType descr1, DType descr2)
        {
            if (descr1 == null) throw new ArgumentNullException(nameof(descr1));
            if (descr2 == null) throw new ArgumentNullException(nameof(descr2));
            var p1 = descr1.VString ?? throw new ArgumentException($"{descr1.ToString(true)} is not a StringDType descriptor.", nameof(descr1));
            var p2 = descr2.VString ?? throw new ArgumentException($"{descr2.ToString(true)} is not a StringDType descriptor.", nameof(descr2));
            if (!VStringParameters.CompatibleNA(p1, p2, out bool hasNA, out object na))
                throw new TypeError($"Cannot find common instance for incompatible dtypes '{descr1}' and '{descr2}'");
            return Descr(hasNA, na, p1.coerce && p1.coerce);
        }

        /// <summary>
        ///     NumPy's <c>string_discover_descriptor_from_pyobject</c>: the value must convert to text (<c>as_pystring</c> with
        ///     coercion on — a <c>bytes</c> value decodes as strict UTF-8, anything else takes its <c>str()</c>), and the
        ///     discovered descriptor is a fresh <c>StringDType()</c> whatever the value.
        /// </summary>
        /// <param name="obj">An element value.</param>
        /// <returns>A fresh <c>StringDType()</c>.</returns>
        /// <exception cref="UnicodeDecodeError">A <c>byte[]</c> that is not valid UTF-8 (CPython's decoder error).</exception>
        public override DType discover_descr_from_pyobject(object obj)
        {
            if (obj is byte[] b)
                NDUtf8.ThrowIfInvalid(b); // assume UTF-8 encoding: an invalid byte raises the decoder's error
            return default_descr();
        }

        /// <summary>
        ///     NumPy's <c>stringdtype_is_known_scalar_type</c>: the Python builtins (<c>str</c>, <c>bytes</c>, <c>int</c>,
        ///     <c>float</c>, <c>complex</c>, <c>bool</c>) and every builtin NumPy scalar type — in the C# boundary map
        ///     <see cref="string"/>, <see cref="char"/>, <c>byte[]</c>, the integer primitives, <see cref="System.Numerics.BigInteger"/>,
        ///     <see cref="float"/>, <see cref="double"/>, <see cref="Half"/>, <see cref="decimal"/>,
        ///     <see cref="System.Numerics.Complex"/> and <see cref="bool"/>.
        /// </summary>
        /// <param name="type">The C# type of a candidate element value.</param>
        /// <returns>True for those types.</returns>
        public override bool is_known_scalar_type(Type type)
            => type == typeof(string) || type == typeof(char) || type == typeof(byte[]) || type == typeof(bool)
               || type == typeof(sbyte) || type == typeof(byte) || type == typeof(short) || type == typeof(ushort)
               || type == typeof(int) || type == typeof(uint) || type == typeof(long) || type == typeof(ulong)
               || type == typeof(System.Numerics.BigInteger) || type == typeof(float) || type == typeof(double)
               || type == typeof(Half) || type == typeof(decimal) || type == typeof(System.Numerics.Complex);

        /// <summary>The class call with no arguments, <c>StringDType()</c>: a fresh default descriptor.</summary>
        /// <returns>A new <c>StringDType()</c>.</returns>
        public override DType Instantiate() => default_descr();

        // ---------------------------------------------------------------------------------------------
        // Element slots (np-layer dtype plan, P3's batch 1): NumPy's stringdtype_setitem / stringdtype_getitem and the
        // dtype's nonzero, through the buffer's string allocator (NumPy keeps the allocator on the descriptor, NumSharp
        // on the buffer, so the slots take the buffer the element lives in).
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>stringdtype_setitem</c>, by <see cref="VStringArraySlice.SetIndex"/>: the descriptor's NA object packs
        ///     an NA cell, any other value its <c>str()</c> under coercion (a non-text value is refused without it), the old
        ///     string freed and the new one packed through the buffer's allocator. The buffer's descriptor rules the store: a
        ///     NumSharp <c>StringDType</c> buffer carries the descriptor of its array, as NumPy's array does.
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="buffer"/> is not the <c>StringDType</c> buffer
        ///     <paramref name="item"/> is a cell of.</exception>
        /// <exception cref="ValueError">A non-text value into a descriptor with coercion off (<c>StringDType only allows
        ///     string data when string coercion is disabled.</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="OutOfMemoryException">The allocator cannot hold the text (NumPy's <c>MemoryError</c>).</exception>
        internal override unsafe void setitem(DType descr, object value, byte* item, IArraySlice buffer)
            => Cells(buffer, item, out long index).SetIndex(index, value);

        /// <inheritdoc/>
        /// <remarks>
        ///     <see cref="VStringArraySlice.SetRange"/>: the cells and arena bytes NumPy's <c>stringdtype_setitem</c> leaves
        ///     element by element, with the allocator locked once for the run and the arena reserved once for its medium and
        ///     long texts, each encoded straight into its arena slot (SetRange's remarks measure it at 1.3x to 4x the
        ///     per-element stores, the gain growing with the texts' length). Only the arena's capacity differs from the
        ///     per-element stores, and no NumPy API reads it. SetRange fills a buffer from its first cell, so a run that
        ///     starts at a later cell takes the per-element default.
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="buffer"/> is not the <c>StringDType</c> buffer
        ///     <paramref name="first"/> is a cell of.</exception>
        /// <exception cref="ValueError">A non-text value into a descriptor with coercion off (<c>StringDType only allows
        ///     string data when string coercion is disabled.</c>).</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="OutOfMemoryException">The allocator cannot hold the texts (NumPy's <c>MemoryError</c>).</exception>
        internal override unsafe void SetFreshItems(DType descr, IReadOnlyList<object> values, byte* first, IArraySlice buffer)
        {
            // An empty array has no cell to name (Cells refuses an index at or past the count) and nothing to store.
            if (values.Count == 0)
                return;
            var cells = Cells(buffer, first, out long index);
            if (index != 0)
            {
                base.SetFreshItems(descr, values, first, buffer);
                return;
            }
            cells.SetRange(values);
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>stringdtype_getitem</c>, by <see cref="VStringArraySlice.GetIndex(long)"/>: the text as a
        ///     <see cref="string"/>, the descriptor's NA object for an NA cell (<c>''</c> when it has none), read under the
        ///     allocator's lock.</remarks>
        /// <exception cref="ArgumentException"><paramref name="buffer"/> is not the <c>StringDType</c> buffer
        ///     <paramref name="item"/> is a cell of.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell (<c>Failed to load string in StringDType getitem</c>).</exception>
        internal override unsafe object getitem(DType descr, byte* item, IArraySlice buffer)
            => Cells(buffer, item, out long index).GetIndex(index);

        /// <inheritdoc/>
        /// <remarks>A cell points into its buffer's allocator, so it has no byte order to swap and no meaning outside its
        ///     buffer: NumPy refuses a buffer for a dtype that holds references, and NumSharp has no byte-order copy for
        ///     this class.</remarks>
        /// <exception cref="NotSupportedException">Always.</exception>
        internal override unsafe void copyswapn(DType descr, byte* dst, long dstride, byte* src, long sstride, long n, bool swap)
            => throw new NotSupportedException(
                "StringDType cells hold references into their buffer's allocator: they cannot be copied or byte-swapped as raw bytes.");

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>nonzero</c> of <c>stringdtype/dtype.c</c>: an NA cell of a descriptor with an NA object is
        ///     nonzero when that object is NaN-like, or when it is a non-empty string (<see cref="NDVStringCells.NaTruth"/>'s
        ///     nonzero route); any other cell when its text is not empty. The cell's flags and size are in the cell itself,
        ///     so no allocator is needed.</remarks>
        internal override unsafe bool nonzero(DType descr, byte* item)
        {
            if (descr.VString.HasNAObject && VStringAllocator.isnull(item))
                return NDVStringCells.NaTruth(descr, cast: false) != 0;
            return VStringAllocator.size(item) != 0;
        }

        /// <summary>The <c>StringDType</c> buffer <paramref name="item"/> is a cell of, and the cell's index in it.</summary>
        /// <param name="buffer">The buffer the caller names.</param>
        /// <param name="item">The cell's first byte.</param>
        /// <param name="index">Receives the cell's index in the buffer.</param>
        /// <returns>The buffer as a <see cref="VStringArraySlice"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="buffer"/> is not a <c>StringDType</c> buffer, or
        ///     <paramref name="item"/> is not one of its cells.</exception>
        private static unsafe VStringArraySlice Cells(IArraySlice buffer, byte* item, out long index)
        {
            if (buffer is not VStringArraySlice cells)
                throw new ArgumentException(
                    "A StringDType element is stored and read through its buffer's string allocator: pass the VStringArraySlice it lives in.",
                    nameof(buffer));
            // The cell must be one of the buffer's: an allocator only owns the strings of its own cells.
            long offset = item - cells.Address;
            if (offset < 0 || offset % SIZEOF_NPY_PACKED_STATIC_STRING != 0 || offset / SIZEOF_NPY_PACKED_STATIC_STRING >= cells.Count)
                throw new ArgumentException("The element is not a cell of the given StringDType buffer.", nameof(buffer));
            index = offset / SIZEOF_NPY_PACKED_STATIC_STRING;
            return cells;
        }
    }
}
