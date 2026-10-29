using System;
using System.Collections;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

// =============================================================================
// NDCharArray.cs — numpy.char.chararray (numpy/_core/defchararray.py), the class and its object protocol
// =============================================================================
//
// NumPy's chararray is an ndarray SUBCLASS over string data. What it changes, in its own source:
//
//   __array_finalize__   a result whose dtype.char is not one of 'VSUbc' raises
//                        ValueError("Can only create a chararray from string data.")
//   __getitem__          a SCALAR result (a full integer index) is the str / bytes value's .rstrip(): Python's 29
//                        whitespace code points for str, the 6 ASCII ones for bytes. An array result is a chararray.
//   __array_wrap__       a ufunc result of a string dtype becomes a chararray view (phase 3 of the np.char plan)
//   operators, methods   the comparisons strip, +, *, % and 39 string methods (phase 3 and later)
//
// THE STATIC TYPE DECIDES (the NDMatrix rule, Matrixlib/NDMatrix.cs). Python finds the subclass at run time. NumSharp
// binds the members and indexers declared on NDCharArray at COMPILE time: `c[0]` strips when `c` is typed NDCharArray,
// and the same object through an NDArray reference is a plain string array to the indexers and the NDArray members.
// Every kernel and helper of the library is written against NDArray, where `a[i]` is a 0-d VIEW and `a.copy()` a
// plain array; a run-time dispatch would change the meaning of that code for a caller who passed a chararray in.
//
// THE RESULT FORMS. Every member that returns a chararray builds it through one of these:
//   ViewOf(a)          NumPy's a.view(chararray): a new object over a's memory, never an owner; the finalize rule
//                      reads a's dtype.
//   From(r, self)      an ndarray method's result, allocated by NumPy as the subclass (copy, astype, take ...) or a
//                      view (T, reshape ...): the chararray takes over the result's storage object, so it owns its
//                      data exactly when the result did; the finalize rule reads the RESULT's dtype.
//   FromIndexing       ndarray.__getitem__'s array result: a basic index is a view; an index array or a mask gives a
//                      chararray VIEW of the gathered array (mapping.c wraps the subclass result), never an owner.
//   ScalarOf           the scalar a full integer index returns (see THE SCALAR STAND-IN).
// None of them disposes the result it is given: their callers are [NDScoped], so the temporary NDArray wrapper is
// reclaimed by the scope while the chararray keeps its own counted reference on the buffer. The hottest members build
// their result as the subclass directly instead (T over a storage alias, copy over a fresh buffer): no intermediate
// array, no scope.
//
// THE SCALAR STAND-IN. NumPy's c[i] is a Python str / bytes (or, for a chararray over int8 data, a NumPy scalar).
// NumSharp has no scalar type: the indexers return a 0-d NDCharArray marked IsScalarStandIn that holds a copy of the
// stripped text (width = the stripped length, at least 1, as np.array('') is <U1), read-only as a Python str is
// immutable. It prints like a plain 0-d array. A text whose stripped value ends in NUL (Python's 'ab\0' from 'ab\0 ')
// cannot keep that NUL: a fixed-width element reads its trailing NULs as padding. The copy is cut from a per-thread
// arena block (NDCharArray.StandIn.cs) and the element's address comes straight from the strides: an access costs
// about what NumPy's does.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy.char.chararray</c>: a string array (<c>S</c> or <c>U</c>) whose single elements come back with
    ///     their trailing whitespace stripped (<c>c[0]</c> of <c>['ab  ']</c> is <c>'ab'</c>) and whose views and copies
    ///     stay chararrays. Create one with <see cref="CharModule.chararray"/>, <see cref="CharModule.array"/> or
    ///     <see cref="CharModule.asarray"/>, or view an existing string array with
    ///     <see cref="NDArray.view(DType, Type)"/>. NumPy keeps the class for backward compatibility with Numarray and
    ///     recommends plain <c>S</c> / <c>U</c> arrays and the <c>np.char</c> / <c>np.strings</c> functions for new code.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Members and indexers follow the static type.</b> They are chosen by the compiler when the expression is typed
    ///     <see cref="NDCharArray"/>. The same object through an <see cref="NDArray"/> reference is a plain string array:
    ///     its indexer returns an unstripped 0-d view and its members return plain arrays.
    ///     </para>
    ///     <para>
    ///     <b>Element access strips.</b> An index that names one element (an integer per axis) returns a READ-ONLY 0-d
    ///     <see cref="NDCharArray"/> holding the element's text with Python's <c>rstrip()</c> applied: the 29 Unicode
    ///     whitespace code points for <c>U</c>, the 6 ASCII ones for <c>S</c> (trailing NULs are padding and go too). It is
    ///     NumSharp's stand-in for the Python <c>str</c> / <c>bytes</c> NumPy returns. Iterating a 1-d chararray yields the
    ///     stripped values as <see cref="string"/> / <c>byte[]</c>. <see cref="NDArray.item(long)"/>,
    ///     <see cref="NDArray.tolist"/> and <see cref="NDArray.flatiter"/>'s single elements do NOT strip, as in NumPy.
    ///     </para>
    ///     <para>
    ///     <b>The finalize rule.</b> A chararray can be created only over <c>S</c>, <c>U</c> or <c>int8</c> data
    ///     (NumPy's <c>dtype.char in 'VSUbc'</c>; NumSharp's <c>char</c> dtype reports <c>'c'</c> and passes too). Any
    ///     member whose result would have another dtype raises <see cref="ValueError"/>
    ///     (<c>Can only create a chararray from string data.</c>): <c>astype(int)</c>, <c>view(type: typeof(NDCharArray))</c>
    ///     of a numeric array. <see cref="view(DType)"/> is the exception NumPy has: it checks the dtype the array HAS, then
    ///     changes it, so <c>c.view(np.uint32)</c> is a uint32 chararray — whose own views and copies then raise.
    ///     </para>
    ///     <para>
    ///     <b>Lifetime.</b> A chararray is an <see cref="NDArray"/>: it holds one counted reference on its buffer and is
    ///     released with <see cref="NDArray.Dispose"/> or an <see cref="NDScope"/>. A chararray that views another array
    ///     keeps that array's buffer alive.
    ///     </para>
    /// </remarks>
    /// <seealso href="https://numpy.org/doc/stable/reference/generated/numpy.char.chararray.html"/>
    public partial class NDCharArray : NDArray, IEnumerable
    {
        /// <summary>NumPy's <c>__array_finalize__</c> error text, verbatim.</summary>
        internal const string FinalizeText = "Can only create a chararray from string data.";

        /// <summary>
        ///     Wraps <paramref name="storage"/> as a chararray WITHOUT the finalize rule and without copying. Callers check
        ///     the rule first (<see cref="Finalize(DType)"/>) and pass a storage object no other array wraps: an in-place
        ///     shape change of either array would otherwise show through the other.
        /// </summary>
        /// <param name="storage">The storage to wrap; its shape becomes the chararray's shape.</param>
        /// <exception cref="NullReferenceException"><paramref name="storage"/> is <c>null</c>.</exception>
        internal NDCharArray(UnmanagedStorage storage) : base(storage)
        {
        }

        /// <summary>
        ///     Whether this 0-d object is the stand-in for the Python <c>str</c> / <c>bytes</c> (or NumPy scalar) an element
        ///     access returns in NumPy, not a real 0-d chararray. It prints as a plain 0-d array, and the comparison operators
        ///     read it as the scalar it stands for (a Python <c>str</c> compares without stripping).
        /// </summary>
        internal bool IsScalarStandIn { get; private set; }

        /// <inheritdoc/>
        /// <remarks>NumPy prints a chararray as <c>chararray(['a', 'bb'], dtype='&lt;U2')</c>: the class name, with the
        ///     continuation lines indented by its length. The scalar stand-in prints as the plain 0-d array it is in
        ///     NumSharp (NumPy prints the Python str itself).</remarks>
        internal override string ReprClassName => IsScalarStandIn ? "array" : "chararray";

        #region The finalize rule

        /// <summary>
        ///     NumPy's <c>__array_finalize__</c> test: whether a chararray may hold data of <paramref name="dtype"/>, i.e.
        ///     <c>dtype.char in 'VSUbc'</c> — <c>S</c>, <c>U</c>, <c>int8</c> (<c>'b'</c>, kept by NumPy for pickle
        ///     reconstruction), the void and <c>'c'</c> codes (NumSharp's <c>char</c> dtype reports <c>'c'</c>).
        /// </summary>
        /// <param name="dtype">The descriptor of the array being created.</param>
        /// <returns>True when the rule accepts it.</returns>
        internal static bool AcceptsDType(DType dtype)
        {
            char code = dtype.@char;
            return code == 'S' || code == 'U' || code == 'b' || code == 'c' || code == 'V';
        }

        /// <summary>Applies NumPy's finalize rule to the dtype of an array about to become a chararray.</summary>
        /// <param name="dtype">The descriptor of that array.</param>
        /// <exception cref="ValueError">The dtype is not string data (<c>Can only create a chararray from string data.</c>).</exception>
        internal static void Finalize(DType dtype)
        {
            if (!AcceptsDType(dtype))
                throw new ValueError(FinalizeText);
        }

        #endregion

        #region Result construction (view / take-over / indexing / scalar)

        /// <summary>
        ///     NumPy's <c>a.view(chararray)</c>: a NEW chararray object over the memory of <paramref name="a"/>, after the
        ///     finalize rule. Use it to see a string array as a chararray without copying; the result never owns its data
        ///     and keeps the writeability of <paramref name="a"/>.
        /// </summary>
        /// <param name="a">The array to view. Not disposed, not modified.</param>
        /// <returns>A chararray sharing memory with <paramref name="a"/>; never <paramref name="a"/> itself, also when it is
        ///     a chararray already (NumPy's <c>view</c> always makes a new object).</returns>
        /// <exception cref="ValueError"><paramref name="a"/>'s dtype is not string data.</exception>
        internal static NDCharArray ViewOf(NDArray a)
        {
            Finalize(a.dtype);
            return new NDCharArray(a.Storage.Alias());
        }

        /// <summary>
        ///     The chararray for an ndarray method's result (<c>copy</c>, <c>astype</c>, <c>T</c>, <c>reshape</c>,
        ///     <c>take</c> ...): NumPy creates it as the subclass, so the finalize rule reads the RESULT's dtype; the chararray
        ///     takes over the result's storage object, so it owns its data exactly when the result did.
        /// </summary>
        /// <param name="result">What the plain method returned. The caller's scope reclaims its wrapper; this method does
        ///     not dispose it.</param>
        /// <param name="self">The chararray the method was called on (identity rule, see the returns).</param>
        /// <returns><paramref name="self"/> when the method returned that very object (NumPy's <c>self</c> from
        ///     <c>astype(copy=False)</c>, <c>real</c>, <c>squeeze</c>, <c>to_device</c>); the result itself when it already
        ///     is a chararray; otherwise a new chararray over the result's storage.</returns>
        /// <exception cref="ValueError">The result's dtype is not string data.</exception>
        internal static NDCharArray From(NDArray result, NDCharArray self)
        {
            if (result is null)
                return null;
            if (ReferenceEquals(result, self))
                return self;
            if (result is NDCharArray already)
                return already;
            Finalize(result.dtype);
            return TakeOver(result, self);
        }

        /// <summary>
        ///     The chararray over <paramref name="result"/>'s storage WITHOUT the finalize rule: for
        ///     <see cref="view(DType)"/>, where NumPy checks the dtype the array had BEFORE the view changed it.
        /// </summary>
        /// <param name="result">The plain result (not disposed here).</param>
        /// <param name="self">The chararray the method was called on, or <c>null</c>.</param>
        /// <returns>A new chararray over the result's storage object, or over an alias of it when that object is
        ///     <paramref name="self"/>'s own (two arrays must never share a storage object).</returns>
        internal static NDCharArray TakeOver(NDArray result, NDArray self)
        {
            UnmanagedStorage storage = result.Storage;
            // A storage object another live array wraps is never taken over: an engine shortcut may hand the operand's
            // own storage back in a second wrapper, and a later in-place shape change would show through both arrays.
            if (self is not null && ReferenceEquals(storage, self.Storage))
                storage = storage.Alias();
            return new NDCharArray(storage);
        }

        /// <summary>
        ///     The chararray for an ARRAY result of indexing (<c>ndarray.__getitem__</c> on the subclass): the finalize rule
        ///     reads the indexed array's dtype; a basic index (slices, an ellipsis, new axes) gives a view; an index array or
        ///     a mask gives a chararray VIEW of the gathered array, never an owner (mapping.c builds the subclass result as a
        ///     second array over the gathered one); NumPy's <c>HAS_SCALAR_ARRAY</c> case is a copy the chararray owns.
        /// </summary>
        /// <param name="result">What the plain indexing returned (not disposed here).</param>
        /// <param name="self">The indexed chararray.</param>
        /// <param name="scalarArrayCopy">The index holds a 0-d integer ARRAY next to integers and slices only (NumPy's
        ///     <c>HAS_SCALAR_ARRAY</c>): NumPy selects the view and returns a COPY of it.</param>
        /// <returns>The chararray.</returns>
        /// <exception cref="ValueError">The indexed chararray holds data that is not string data (a <c>view(np.uint32)</c>).</exception>
        [NDScopedCovered]
        internal static NDCharArray FromIndexing(NDArray result, NDCharArray self, bool scalarArrayCopy)
        {
            Finalize(result.dtype);
            UnmanagedStorage storage = result.Storage;
            if (scalarArrayCopy)
            {
                // mapping.c: `if (index_type & HAS_SCALAR_ARRAY) result = PyArray_NewCopy(view, NPY_KEEPORDER)`, made as
                // the subclass, so the chararray owns its data. A result that already is a fresh owner is that copy.
                if (!storage.OwnsData)
                    storage = result.copy('K').Storage;
            }
            else if (ReferenceEquals(storage, self.Storage) || storage.OwnsData)
            {
                // A new storage object over the same memory: never share the indexed array's storage object, and a result
                // that owns its data is what an index array or a mask gathered, which NumPy hands back as a view of the
                // hidden gathered array (OWNDATA False).
                storage = storage.Alias();
            }

            return new NDCharArray(storage);
        }

        /// <summary>
        ///     The scalar stand-in for one element (NumPy's Python <c>str</c> / <c>bytes</c>, or a NumPy scalar): a NEW
        ///     read-only 0-d chararray that owns a copy of the element, its text stripped when <paramref name="strip"/>.
        /// </summary>
        /// <param name="element">A 0-d storage naming the element (a view into the indexed array, or a 0-d result).</param>
        /// <param name="strip">True for <c>chararray.__getitem__</c>'s <c>rstrip()</c>; false for the unstripped NumPy
        ///     scalars <c>take</c> and the flat iterator return.</param>
        /// <returns>The stand-in (<see cref="IsScalarStandIn"/>); its width is the kept length, at least one character.</returns>
        /// <exception cref="SystemError">A <c>U</c> code unit above <c>U+10FFFF</c> in the kept text: NumPy cannot create
        ///     the <c>str</c> (<c>invalid maximum character passed to PyUnicode_New</c>).</exception>
        internal static unsafe NDCharArray ScalarOf(UnmanagedStorage element, bool strip)
        {
            if (element.IsFixedString)
            {
                var slice = element.StringSlice;
                return StandIn(slice.Address + element.Shape.offset * (long)slice.ItemSize, slice, strip);
            }

            // A chararray over int8 / char data, or one a view(dtype) left with another dtype: NumPy returns the NumPy
            // scalar of the element. A 0-d copy of the element stands for it.
            var scalar = new NDCharArray(element.Alias().Clone()) { IsScalarStandIn = true };
            scalar.MarkReductionScalar();   // read-only, as the immutable Python value it stands for
            return scalar;
        }

        /// <summary>
        ///     Throws NumPy's <c>SystemError</c> when one of the first <paramref name="count"/> code units is not a code
        ///     point (above <c>U+10FFFF</c>): CPython refuses to build the <c>str</c> NumPy's element access returns.
        /// </summary>
        /// <param name="units">The element's first code unit.</param>
        /// <param name="count">The number of units the scalar keeps.</param>
        /// <exception cref="SystemError">A unit is above <c>U+10FFFF</c>.</exception>
        private static unsafe void RequireValidCodePoints(uint* units, int count)
        {
            for (int i = 0; i < count; i++)
                if (units[i] > 0x10FFFF)
                    throw new SystemError(Backends.Strings.NDStringCodec.InvalidMaxCharMessage);
        }

        #endregion

        #region Python values (the implicit conversions)

        /// <summary>
        ///     A C# string as the Python <c>str</c> it stands for: a read-only 0-d <c>U</c> stand-in holding the text as it is
        ///     (no stripping; NumPy's <c>np.array(str)</c> width, at least one character). It lets
        ///     <c>c[0] = "text"</c> compile against the chararray indexers, whose value type is <see cref="NDCharArray"/>.
        /// </summary>
        /// <param name="value">The text.</param>
        /// <returns>The 0-d stand-in.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        public static implicit operator NDCharArray(string value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            return StandInOf(value);
        }

        /// <summary>
        ///     A C# <c>byte[]</c> as the Python <c>bytes</c> it stands for: a read-only 0-d <c>S</c> stand-in holding the
        ///     bytes as they are. It lets <c>c[0] = bytes</c> compile against the chararray indexers.
        /// </summary>
        /// <param name="value">The bytes.</param>
        /// <returns>The 0-d stand-in.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        public static implicit operator NDCharArray(byte[] value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            return StandInOf(value);
        }

        /// <summary>The read-only 0-d stand-in for one text value (<see cref="string"/> or <c>byte[]</c>).</summary>
        /// <param name="value">The value (not null).</param>
        /// <returns>The stand-in.</returns>
        private static NDCharArray StandInOf(object value)
        {
            // np.array(value): the 0-d text array of the value's own class and length. Its wrapper is released once the
            // stand-in holds its own counted reference on the storage.
            using var built = Backends.UnmanagedStorage.FromAnyScalar(value, null);
            var scalar = new NDCharArray(built.Storage) { IsScalarStandIn = true };
            scalar.MarkReductionScalar();
            return scalar;
        }

        #endregion

        #region Object protocol (clone, iteration)

        /// <summary>
        ///     A deep copy that is a chararray again (NumPy's <c>copy.copy(c)</c>, a KEEPORDER copy made as the subclass).
        ///     It HIDES <see cref="NDArray.Clone"/> and does not override it, on purpose: library code that clones an array it
        ///     holds as <see cref="NDArray"/> must get a plain array back.
        /// </summary>
        /// <returns>A new chararray that owns a copy of the data.</returns>
        /// <exception cref="ValueError">This chararray holds data that is not string data (a <c>view(np.uint32)</c>).</exception>
        public new NDCharArray Clone()
        {
            Finalize(dtype);
            return new NDCharArray(Storage.Clone()) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Iterates the first axis as NumPy's <c>chararray</c> does (each step is <c>c[i]</c>): a 1-d chararray yields its
        ///     elements STRIPPED, as <see cref="string"/> for <c>U</c> and <c>byte[]</c> for <c>S</c>; a chararray with more
        ///     axes yields its rows as chararray VIEWS.
        /// </summary>
        /// <returns>An enumerator over the first axis.</returns>
        /// <exception cref="TypeError">The chararray is 0-d (<c>iteration over a 0-d array</c>).</exception>
        /// <remarks>
        ///     <para>A 1-d chararray over int8 / char data yields its elements as the base array does (boxed numbers). Rows of a
        ///     chararray over data that is not string data raise the finalize <see cref="ValueError"/> at the first row, as
        ///     NumPy's first <c>next()</c> does.</para>
        ///     <para>An array with rows of length 0 still yields its rows (<c>(2, 0)</c> yields two empty chararrays), where
        ///     iterating a plain <see cref="NDArray"/> of that shape yields nothing.</para>
        ///     <para>Each row holds a counted reference on the buffer: dispose the rows in a hot loop, or iterate inside an
        ///     <see cref="NDScope"/>.</para>
        /// </remarks>
        public new IEnumerator GetEnumerator()
        {
            if (ndim == 0)
                throw new TypeError("iteration over a 0-d array");
            if (ndim > 1)
                return Rows().GetEnumerator();
            if (!Storage.IsFixedString)
                return base.GetEnumerator();
            return Elements().GetEnumerator();

            IEnumerable Rows()
            {
                long count = Shape.dimensions[0];
                for (long i = 0; i < count; i++)
                    yield return GetBasic(new[] { Slice.Index(i) });
            }

            IEnumerable Elements()
            {
                // The storage decodes element i through the strides (trailing NULs dropped, a U unit above U+10FFFF
                // refused); __getitem__'s rstrip() then applies.
                long count = size;
                for (long i = 0; i < count; i++)
                    yield return StripValue(Storage.GetAtIndex(i));
            }
        }

        /// <summary>The non-generic enumeration; makes <c>foreach</c> over an <see cref="IEnumerable"/> reference iterate as
        ///     the chararray does.</summary>
        /// <returns>The enumerator of <see cref="GetEnumerator"/>.</returns>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Python's <c>iter(c)</c>: the same as <see cref="GetEnumerator"/>.</summary>
        /// <returns>An enumerator over the first axis (stripped elements of a 1-d chararray, chararray rows otherwise).</returns>
        /// <exception cref="TypeError">The chararray is 0-d.</exception>
        public new IEnumerator __iter__() => GetEnumerator();

        #endregion
    }
}
