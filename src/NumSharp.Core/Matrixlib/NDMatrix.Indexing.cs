using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;

// =============================================================================
// NDMatrix.Indexing.cs — matrix.__getitem__ and the assignment rule a matrix inherits from ndarray
// =============================================================================
//
// GETTING. NumPy's matrix.__getitem__ runs ndarray.__getitem__ with the finalize rule switched off and then
// shapes the result itself:
//
//     if out.ndim == 0:  return out[()]                          # a scalar
//     if out.ndim == 1:
//         n = len(index) if index has a len() else 0
//         out.shape = (sh, 1) if n > 1 and isscalar(index[1]) else (1, sh)
//     return out                                                  # every other rank, as it is
//
// So m[0] and m[0, :] are rows, m[:, 0] is a column (the SECOND index entry is a scalar), m[[0, 1, 0], 1] is a
// column and m[1, [0, 1, 0]] a row, a boolean mask selects a row, and m[newaxis] is a 3-d matrix object. The
// test looks at entry #1 by POSITION: m[..., 0] is a column and m[0, ...] a row.
//
// SETTING. matrix does not override __setitem__, but ndarray's assignment treats a subclass specially: for an
// index that selects a VIEW (integers, slices, ellipsis, new axes) it assigns into self[index] as the subclass's
// own __getitem__ returns it. For a matrix that view is a row or a COLUMN, so
//
//     m[:, 0] = [7, 8, 9]        raises: could not broadcast input array from shape (3,) into shape (3,1)
//     m[:, 0] = [[7], [8], [9]]  works
//
// An index with an index array or a mask takes the plain ndarray path. A full integer index stores one element
// and refuses an array value ("setting an array element with a sequence.").
//
// THE C# SETTER TYPE. An indexer has one type for get and set. These indexers are typed NDMatrix so that
// m[0] * m.T is a matrix product, therefore the value assigned through them is an NDMatrix or a C# number (which
// converts implicitly to a 0-d matrix). Assign any other value (an NDArray, a list) with __setitem__.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDMatrix
    {
        #region Getters

        /// <summary>
        ///     Row <paramref name="index"/> as a <c>(1, N)</c> matrix (NumPy's <c>m[i]</c>). Assigning writes the
        ///     row; the value broadcasts against <c>(1, N)</c>.
        /// </summary>
        /// <param name="index">The row. A negative value counts from the last row.</param>
        /// <value>A view: writes to it reach this matrix.</value>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast to the
        ///     row.</exception>
        public NDMatrix this[int index]
        {
            get => GetBasic(new Slice[] { Slice.Index(index) });
            set => SetBasic(new Slice[] { Slice.Index(index) }, value);
        }

        /// <inheritdoc cref="this[int]"/>
        public NDMatrix this[long index]
        {
            get => GetBasic(new Slice[] { Slice.Index(index) });
            set => SetBasic(new Slice[] { Slice.Index(index) }, value);
        }

        /// <summary>
        ///     The element at (<paramref name="row"/>, <paramref name="column"/>) as a 0-d <see cref="NDArray"/>,
        ///     NumSharp's scalar (NumPy's <c>m[i, j]</c> returns a scalar, not a matrix). Assigning stores one
        ///     element.
        /// </summary>
        /// <param name="row">The row; negative counts from the end.</param>
        /// <param name="column">The column; negative counts from the end.</param>
        /// <value>A 0-d view of the element. Convert it with a cast (<c>(double)m[0, 1]</c>).</value>
        /// <exception cref="IndexError">An index is out of bounds, or the matrix object does not have two axes.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value is an array with one or more
        ///     axes (NumPy's <c>setting an array element with a sequence.</c>).</exception>
        public NDArray this[int row, int column]
        {
            get => new NDArray(ElementView(row, column));
            set => SetBasic(new Slice[] { Slice.Index(row), Slice.Index(column) }, value);
        }

        /// <inheritdoc cref="this[int, int]"/>
        public NDArray this[long row, long column]
        {
            get => new NDArray(ElementView(row, column));
            set => SetBasic(new Slice[] { Slice.Index(row), Slice.Index(column) }, value);
        }

        /// <summary>
        ///     Indexes with Python slice notation (<c>m["1:3, 0"]</c>, <c>m[":, ::-1"]</c>, <c>m["..., 0"]</c>) under
        ///     the matrix rule: a 1-d selection comes back as a row, or as a column when the second entry is an
        ///     integer.
        /// </summary>
        /// <param name="slice">The comma-separated entries: integers, <c>start:stop:step</c>, <c>...</c>,
        ///     <c>np.newaxis</c>.</param>
        /// <value>A view. When every axis is indexed with an integer the result is the 0-d scalar stand-in
        ///     (prefer <c>m[i, j]</c>, which is typed as a scalar).</value>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast to the
        ///     selected view.</exception>
        public new NDMatrix this[string slice]
        {
            get => GetBasic(Slice.ParseSlices(slice));
            set => SetBasic(Slice.ParseSlices(slice), value);
        }

        /// <summary>
        ///     Indexes with slice objects (<c>m[Slice.All, 0]</c>, <c>m[new Slice(1, 3)]</c>) under the matrix rule.
        ///     An <see cref="int"/> and a slice string convert to <see cref="Slice"/>, so <c>m[":", 0]</c> binds here.
        /// </summary>
        /// <param name="slices">One entry per axis, with <see cref="Slice.Ellipsis"/> and <see cref="Slice.NewAxis"/>
        ///     allowed. An index with a new axis returns a matrix object with three axes, as NumPy does.</param>
        /// <value>A view; the 0-d scalar stand-in when every axis is indexed with an integer.</value>
        /// <exception cref="ArgumentNullException"><paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast to the
        ///     selected view.</exception>
        public new NDMatrix this[params Slice[] slices]
        {
            get => GetBasic(slices);
            set => SetBasic(slices, value);
        }

        /// <summary>
        ///     The general index: integers, slices, slice strings, index arrays, boolean masks and sequences in any
        ///     mix, under the matrix rule (file header). A boolean matrix from a comparison binds here:
        ///     <c>m[m &gt; 5]</c> is a <c>(1, K)</c> row.
        /// </summary>
        /// <param name="indices">The index entries (NumPy's index tuple).</param>
        /// <value>A view for an index of integers and slices, a new matrix when an index array or a mask is
        ///     involved; the 0-d scalar stand-in when the selection has no axis left.</value>
        /// <exception cref="IndexError">An index is out of bounds, too many entries, or index arrays that do not
        ///     broadcast together.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not fit the selection.</exception>
        public new NDMatrix this[params object[] indices]
        {
            // An index of integers and slices takes the same view route as the typed indexers: one rule for the
            // row / column shape, NumPy's index errors, and the source's read-only flag kept on the view.
            [NDScoped]
            get => indices is not null && TryAsSlices(indices, out Slice[] slices)
                ? GetBasic(slices)
                : FromIndexing(base[indices], this, SelectsColumn(indices), IsScalarArrayIndex(indices));
            [NDScoped] set => SetItem(indices, value);
        }

        /// <summary>
        ///     Selects with a boolean mask (NumPy's <c>m[mask]</c>): a mask of the matrix's shape gives the selected
        ///     elements as ONE ROW <c>(1, K)</c>, a 1-d mask over the rows gives those rows.
        /// </summary>
        /// <param name="mask">The mask.</param>
        /// <value>A new matrix (a copy). Assigning stores into the selected elements through the plain array rule:
        ///     the value is a scalar or 1-d there, which a 2-d matrix value is not.</value>
        /// <exception cref="IndexError">The mask's shape does not match.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value count differs from the number
        ///     of selected elements.</exception>
        public new NDMatrix this[NDArray<bool> mask]
        {
            [NDScoped] get => FromIndexing(base[mask], this, column: false);
            set => base[mask] = value;
        }

        /// <summary>
        ///     Selects with integer index arrays, one per axis (NumPy's <c>m[rows, cols]</c>). A 1-d selection is a
        ///     row: the second entry is an array, never a scalar.
        /// </summary>
        /// <param name="selection">The index arrays.</param>
        /// <value>A new matrix (a copy).</value>
        /// <exception cref="IndexError">An index is out of bounds, or the arrays do not broadcast together.</exception>
        /// <exception cref="ValueError">(set) The matrix is read-only, or the value does not broadcast to the
        ///     selection.</exception>
        public new NDMatrix this[params NDArray<int>[] selection]
        {
            [NDScoped] get => FromIndexing(base[selection], this, column: false);
            set => base[selection] = value;
        }

        /// <summary>Python's <c>m.__getitem__(i)</c>: the same as <c>m[i]</c>.</summary>
        /// <param name="index">The row.</param>
        /// <returns>The row as a <c>(1, N)</c> view.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds.</exception>
        public new NDMatrix __getitem__(int index) => this[index];

        /// <inheritdoc cref="__getitem__(int)"/>
        public new NDMatrix __getitem__(long index) => this[index];

        /// <summary>Python's <c>m.__getitem__</c> with slice notation: the same as <c>m[slice]</c>.</summary>
        /// <param name="slice">The slice notation.</param>
        /// <returns>The selection under the matrix rule.</returns>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        public new NDMatrix __getitem__(string slice) => this[slice];

        /// <summary>Python's <c>m.__getitem__((i, j))</c> with integer entries.</summary>
        /// <param name="indices">One integer per indexed axis.</param>
        /// <returns>A row for one integer; the 0-d scalar stand-in for two.</returns>
        /// <exception cref="IndexError">An index is out of bounds, or more entries than axes.</exception>
        public new NDMatrix __getitem__(params int[] indices)
        {
            var slices = new Slice[indices.Length];
            for (int i = 0; i < slices.Length; i++)
                slices[i] = Slice.Index(indices[i]);
            return GetBasic(slices);
        }

        #endregion

        #region Setters

        /// <summary>
        ///     Python's <c>m[index] = value</c> for ANY value: an <see cref="NDArray"/>, a C# array, a list or tuple,
        ///     or a number. Use it where the typed indexers do not accept the value (they take an
        ///     <see cref="NDMatrix"/> or a number).
        /// </summary>
        /// <param name="index">One index entry, or the whole index tuple as an <c>object[]</c> or a C# tuple.</param>
        /// <param name="value">The value. A 1-d value assigned to a COLUMN raises, as in NumPy: a column is
        ///     <c>(N, 1)</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">The index is invalid for this matrix.</exception>
        /// <exception cref="ValueError">The matrix is read-only, the value does not broadcast to the selection,
        ///     or an array is assigned to a single element.</exception>
        [NDScoped]
        public void __setitem__(object index, object value)
        {
            object[] entries = index switch
            {
                object[] many => many,
                ITuple tuple => PolySequence.Items(tuple),
                _ => new[] { index },
            };
            SetItem(entries, AsOperand(value));
        }

        /// <summary>Python's <c>m[i] = value</c> for any value; see <see cref="__setitem__(object, object)"/>.</summary>
        /// <param name="index">The row.</param>
        /// <param name="value">The value; it broadcasts against the row <c>(1, N)</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds.</exception>
        /// <exception cref="ValueError">The matrix is read-only, or the value does not broadcast to the row.</exception>
        [NDScoped]
        public new void __setitem__(int index, object value) => SetBasic(new Slice[] { Slice.Index(index) }, AsOperand(value));

        /// <inheritdoc cref="__setitem__(int, object)"/>
        [NDScoped]
        public new void __setitem__(long index, object value) => SetBasic(new Slice[] { Slice.Index(index) }, AsOperand(value));

        /// <summary>Python's <c>m[slice] = value</c> for any value; see <see cref="__setitem__(object, object)"/>.</summary>
        /// <param name="slice">The slice notation.</param>
        /// <param name="value">The value; it broadcasts against the selected view as the matrix shapes it.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">The matrix is read-only, or the value does not broadcast to the view.</exception>
        [NDScoped]
        public new void __setitem__(string slice, object value) => SetBasic(Slice.ParseSlices(slice), AsOperand(value));

        #endregion

        #region Cores

        /// <summary>
        ///     The view an index of integers and slices selects, shaped by the matrix rule. Built on the storage
        ///     directly: no intermediate array is created.
        /// </summary>
        /// <param name="slices">The index entries.</param>
        /// <returns>The matrix view (0-d when no axis is left).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        private NDMatrix GetBasic(Slice[] slices) => new NDMatrix(BasicView(slices));

        /// <summary>The storage of <see cref="GetBasic"/>: the selected view with the matrix rule applied.</summary>
        /// <param name="slices">The index entries.</param>
        /// <returns>A storage object no other array wraps.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        private UnmanagedStorage BasicView(Slice[] slices)
        {
            if (slices is null)
                throw new ArgumentNullException(nameof(slices));
            ValidateBasic(slices);
            UnmanagedStorage view = Storage.GetView(slices);
            // A trivial index may hand the storage itself back; the result must be its own object.
            if (ReferenceEquals(view, Storage))
                view = Storage.Alias();
            // NumPy: len(index) > 1 and isscalar(index[1]) — entry #1 by position, whatever entry #0 is.
            ApplyIndexingRule(view, column: slices.Length > 1 && slices[1].IsIndex);
            return view;
        }

        /// <summary>
        ///     The 0-d view of one element for <c>m[row, column]</c>, with NumPy's bounds errors. The two-axis case
        ///     checks both integers inline, so the hot element read allocates no index array.
        /// </summary>
        /// <param name="row">The row; negative counts from the end.</param>
        /// <param name="column">The column; negative counts from the end.</param>
        /// <returns>A storage object no other array wraps.</returns>
        /// <exception cref="IndexError">An index is out of bounds
        ///     (<c>index 9 is out of bounds for axis 1 with size 4</c>), or the matrix object has fewer than two
        ///     axes (<c>too many indices for array: ...</c>).</exception>
        private UnmanagedStorage ElementView(long row, long column)
        {
            long[] dims = Shape.dimensions;
            if (dims.Length != 2)
            {
                // The rank quirks named on the class (a 3-d matrix after m[newaxis], the 0-d scalar stand-in):
                // the general walk raises NumPy's errors for them.
                return BasicView(new[] { Slice.Index(row), Slice.Index(column) });
            }

            if (row < -dims[0] || row >= dims[0])
                throw new IndexError($"index {row} is out of bounds for axis 0 with size {dims[0]}");
            if (column < -dims[1] || column >= dims[1])
                throw new IndexError($"index {column} is out of bounds for axis 1 with size {dims[1]}");
            return Storage.GetView(Slice.Index(row), Slice.Index(column));
        }

        /// <summary>
        ///     NumPy's checks of an index of integers and slices, in NumPy's order and with its texts
        ///     (<c>prepare_index</c>, then <c>get_view_from_index</c>): a second ellipsis, more entries than axes,
        ///     then each integer against its axis. The storage layer below reports the same conditions with other
        ///     texts and types, so the matrix checks first.
        /// </summary>
        /// <param name="slices">The index entries (not <c>null</c>).</param>
        /// <exception cref="IndexError">
        ///     <c>an index can only have a single ellipsis ('...')</c>;
        ///     <c>too many indices for array: array is 2-dimensional, but 3 were indexed</c>;
        ///     <c>index 9 is out of bounds for axis 1 with size 4</c>.
        /// </exception>
        private void ValidateBasic(Slice[] slices)
        {
            long[] dims = Shape.dimensions;
            int consuming = 0;
            bool ellipsis = false;
            foreach (Slice slice in slices)
            {
                if (slice.IsEllipsis)
                {
                    // NumPy raises at the second ellipsis while it still walks the index, before it counts.
                    if (ellipsis)
                        throw new IndexError("an index can only have a single ellipsis ('...')");
                    ellipsis = true;
                }
                else if (!slice.IsNewAxis)
                    consuming++;
            }

            if (consuming > dims.Length)
                throw new IndexError($"too many indices for array: array is {dims.Length}-dimensional, but {consuming} were indexed");

            int axis = 0;
            foreach (Slice slice in slices)
            {
                if (slice.IsNewAxis)
                    continue;
                if (slice.IsEllipsis)
                {
                    // The ellipsis stands for every axis the other entries do not consume.
                    axis += dims.Length - consuming;
                    continue;
                }

                if (slice.IsIndex)
                {
                    long index = slice.Start ?? 0;
                    long extent = dims[axis];
                    if (index < -extent || index >= extent)
                        throw new IndexError($"index {index} is out of bounds for axis {axis} with size {extent}");
                }

                axis++;
            }
        }

        /// <summary>
        ///     NumPy's assignment through an index of integers and slices on a matrix: the value is broadcast into
        ///     the view the matrix's own indexing returns, with an unsafe cast.
        /// </summary>
        /// <param name="slices">The index entries.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> or <paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">The matrix is read-only (checked first, as NumPy does), the value does not
        ///     broadcast to the view, or an array is assigned to one element.</exception>
        private void SetBasic(Slice[] slices, NDArray value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            // NumPy tests writeability before it looks at the index or the value.
            if (!Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");

            UnmanagedStorage view = BasicView(slices);
            using var target = new NDArray(view);
            if (view.Shape.NDim == 0 && value.ndim != 0)
            {
                // A full integer index stores ONE element through the dtype's setitem (PyArray_Pack), which sees
                // the value as a Python object. An array that still has an axis is refused, in the words of the
                // conversion each dtype tries: complex() of an array, a sequence for the real dtypes. bool is
                // the exception: its setitem takes the truth value, which a one-element array has.
                if (typecode == NPTypeCode.Complex)
                    throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
                if (typecode != NPTypeCode.Boolean || value.size != 1)
                    throw new ValueError("setting an array element with a sequence.");

                using NDArray element = value.reshape(Shape.NewScalar());
                np.copyto(target, element, casting: "unsafe");
                return;
            }

            np.copyto(target, value, casting: "unsafe");
        }

        /// <summary>
        ///     NumPy's <c>m[index] = value</c> for a general index: the matrix view rule when the index selects a
        ///     view, the plain array rule when it holds an index array or a mask.
        /// </summary>
        /// <param name="index">The index entries.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> or <paramref name="index"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">The index is invalid.</exception>
        /// <exception cref="ValueError">The matrix is read-only, or the value does not fit the selection.</exception>
        private void SetItem(object[] index, NDArray value)
        {
            if (index is null)
                throw new ArgumentNullException(nameof(index));
            if (value is null)
                throw new ArgumentNullException(nameof(value));

            if (TryAsSlices(index, out Slice[] slices))
            {
                SetBasic(slices, value);
                return;
            }

            // NumPy tests writeability before it looks at the index or the value.
            if (!Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");
            base[index] = value;
        }

        /// <summary>
        ///     Whether a general index consists of integers, slices and slice strings only, and those entries as
        ///     slices. Such an index selects a view; any other entry (an index array, a mask, a sequence, a bool)
        ///     makes it an advanced index.
        /// </summary>
        /// <param name="index">The index entries. A single C# tuple entry is NumPy's index tuple and is spread.</param>
        /// <param name="slices">The entries as slices when the result is <c>true</c>.</param>
        /// <returns><c>true</c> for an index of integers and slices.</returns>
        private static bool TryAsSlices(object[] index, out Slice[] slices)
        {
            slices = null;
            if (index.Length == 1 && index[0] is ITuple tuple)
                index = PolySequence.Items(tuple);

            var list = new List<Slice>(index.Length);
            foreach (object entry in index)
            {
                switch (entry)
                {
                    case Slice slice:
                        list.Add(slice);
                        break;
                    case string notation:
                        list.AddRange(Slice.ParseSlices(notation));
                        break;
                    case int i:
                        list.Add(Slice.Index(i));
                        break;
                    case long l:
                        list.Add(Slice.Index(l));
                        break;
                    case sbyte or byte or short or ushort or uint:
                        list.Add(Slice.Index(System.Convert.ToInt64(entry)));
                        break;
                    default:
                        return false;
                }
            }

            slices = list.ToArray();
            return true;
        }

        /// <summary>
        ///     Whether a general index is NumPy's <c>HAS_SCALAR_ARRAY</c> case: at least one 0-d INTEGER array
        ///     among entries that are otherwise integers and slices. NumPy treats the 0-d array as an integer,
        ///     selects the view, and returns a copy of it ("to simulate fancy indexing").
        /// </summary>
        /// <param name="index">The index entries. A single C# tuple entry is NumPy's index tuple.</param>
        /// <returns><c>true</c> when the result is a copy the matrix owns; <c>false</c> when any entry is an
        ///     index array with an axis, a mask, a 0-d bool, or a sequence (those gather).</returns>
        private static bool IsScalarArrayIndex(object[] index)
        {
            if (index is null)
                return false;
            if (index.Length == 1 && index[0] is ITuple tuple)
                index = PolySequence.Items(tuple);

            bool scalarArray = false;
            foreach (object entry in index)
            {
                switch (entry)
                {
                    case NDArray { ndim: 0 } zeroD when zeroD.typecode != NPTypeCode.Boolean:
                        scalarArray = true;
                        break;
                    case Slice or string or sbyte or byte or short or ushort or int or uint or long or ulong:
                        break;
                    default:
                        return false;
                }
            }

            return scalarArray;
        }

        /// <summary>
        ///     NumPy's column test for a general index: <c>len(index) &gt; 1 and isscalar(index[1])</c>.
        /// </summary>
        /// <param name="index">The index entries. A single C# tuple entry is the index tuple; a slice string
        ///     counts as the entries it holds.</param>
        /// <returns><c>true</c> when a 1-d result is a column.</returns>
        /// <remarks>A 0-d array is NOT a scalar for NumPy's <c>isscalar</c>, so <c>m[:2, np.array(-1)]</c> is a
        ///     row; a C# integer and an integer slice entry are scalars.</remarks>
        private static bool SelectsColumn(object[] index)
        {
            if (index is null || index.Length == 0)
                return false;
            if (index.Length == 1 && index[0] is ITuple tuple)
                index = PolySequence.Items(tuple);

            // Entry #1 by position, where a slice string holds as many entries as it has commas + 1.
            int position = 0;
            foreach (object entry in index)
            {
                if (entry is string notation)
                {
                    Slice[] parsed = Slice.ParseSlices(notation);
                    if (position + parsed.Length > 1)
                        return parsed[1 - position].IsIndex;
                    position += parsed.Length;
                    continue;
                }

                if (position == 1)
                    return entry is Slice { IsIndex: true } or sbyte or byte or short or ushort or int or uint or long or ulong;
                position++;
            }

            return false;
        }

        /// <summary>
        ///     An operand or assigned value of any C# kind as an array, by the constructor's house mapping: an
        ///     <see cref="NDArray"/> as it is, a typed C# array through <c>np.array</c>, a tuple / list / jagged
        ///     array through NumPy's sequence coercion, a number as a 0-d array of its C# type.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The array. A converted value is a new array the caller's scope reclaims; an
        ///     <see cref="NDArray"/> argument is returned itself.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">A ragged sequence.</exception>
        /// <exception cref="NotSupportedException">A sequence of strings or other objects.</exception>
        internal static NDArray AsOperand(object value)
        {
            switch (value)
            {
                case null:
                    throw new ArgumentNullException(nameof(value));
                case NDArray array:
                    return array;
            }

            if (value is not string && PolySequence.IsSequence(value))
                return PolySequence.ToArray(value);
            return value is Array typed ? np.array(typed) : np.asanyarray(value);
        }

        #endregion
    }
}
