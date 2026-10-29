using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        // ───────────────────────────────────────────────────────────────────────────────────────
        //  prepare_index — a faithful port of NumPy 2.4.2's classifier/validator
        //  (numpy/_core/src/multiarray/mapping.c: prepare_index_noarray, mapping.c:262).
        //
        //  It walks a normalized index tuple ONCE and produces a typed op list + an IndexType
        //  bitmask, doing exactly the validation NumPy does as it goes: at-most-one-ellipsis,
        //  too-many-indices, bool-array dimensionality, and the invalid-index-type error. This
        //  replaces the scattered, per-shape validation in the Try* dispatch — every divergence
        //  where NumSharp "accepts what NumPy rejects" (or vice-versa) for a STRUCTURAL reason is
        //  resolved by routing through this single pass. (The unified gather/scatter that consumes
        //  the op list is Phase D; Phase C wires PrepareIndex purely as the validation gate.)
        // ───────────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     NumPy's IndexError text for an index entry that is neither an integer, a slice, an ellipsis, a new axis nor an
        ///     index array (<c>prepare_index</c>, mapping.c, for an input that was not an array: a float, a float16, a decimal,
        ///     a complex number, text that is not slice notation). An ARRAY of a non-index dtype has its own text
        ///     (<see cref="ThrowIfNotIndexArrayDtype"/>).
        /// </summary>
        /// <remarks>NumPy names the forms with backticks and adds nothing: the type of the refused entry is not in the
        ///     text (NumSharp's used to quote with apostrophes and end with <c>(got 'Double')</c>).</remarks>
        internal const string InvalidIndexMessage =
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices";

        /// <summary>NumPy <c>HAS_*</c> index classification bitmask (mapping.h). Values match NumPy.</summary>
        [Flags]
        internal enum IndexType
        {
            None = 0,
            Integer = 1,
            NewAxis = 2,
            Slice = 4,
            Ellipsis = 8,
            Fancy = 16,
            Bool = 32,
            ScalarArray = 64,
            ZeroDBool = Fancy | 128,   // HAS_0D_BOOL — a fancy op that consumes no source axis
        }

        /// <summary>Kind of one classified index item (newaxis/0-d-bool consume no source axis).</summary>
        internal enum IndexKind : byte { Ellipsis, NewAxis, Slice, Integer, Fancy, Bool, ZeroDBool }

        /// <summary>One classified index item produced by <see cref="PrepareIndex"/>.</summary>
        [NDBorrowed] // an index array the caller passed in (or an ambient-scoped temp), never owned here
        internal readonly struct IndexOp
        {
            public readonly IndexKind Kind;
            public readonly NDArray Array;   // Fancy: the integer index array; Bool: the full mask; ZeroDBool: its zeros(n,) index
            public readonly Slice Slice;     // Slice
            public readonly long IntVal;     // Integer (already normalized, not yet wrapped/bounds-checked)
            public readonly long Value;       // Ellipsis: fill count; Fancy(from bool): the bool axis size (checked); else -1

            public IndexOp(IndexKind kind, NDArray array, Slice slice, long intVal, long value)
            {
                Kind = kind; Array = array; Slice = slice; IntVal = intVal; Value = value;
            }

            public static IndexOp Ell(long fill) => new IndexOp(IndexKind.Ellipsis, null, null, 0, fill);
            public static IndexOp New() => new IndexOp(IndexKind.NewAxis, null, null, 0, 0);
            public static IndexOp Sl(Slice s) => new IndexOp(IndexKind.Slice, null, s, 0, 0);
            public static IndexOp Int(long v) => new IndexOp(IndexKind.Integer, null, null, v, 0);
            public static IndexOp ScalarArr(long v) => new IndexOp(IndexKind.Integer, null, null, v, 0);
            public static IndexOp FancyArr(NDArray a) => new IndexOp(IndexKind.Fancy, a, null, 0, -1);
            public static IndexOp FancyFromBool(NDArray a, long boolAxisSize) => new IndexOp(IndexKind.Fancy, a, null, 0, boolAxisSize);
            public static IndexOp FullBool(NDArray a) => new IndexOp(IndexKind.Bool, a, null, 0, 0);
            public static IndexOp ZeroBool(NDArray zerosIndex, long n) => new IndexOp(IndexKind.ZeroDBool, zerosIndex, null, 0, n);
        }

        /// <summary>Result of <see cref="PrepareIndex"/>: the classified op list + NumPy <c>index_type</c> + dims.</summary>
        [NDBorrowed] // holds IndexOps, which reference the caller's index arrays
        internal readonly struct PreparedIndex
        {
            public readonly IndexOp[] Ops;
            public readonly IndexType Flags;
            public readonly int NewNdim;     // output basic dims (slices + newaxis + ellipsis fill)
            public readonly int FancyNdim;   // broadcast advanced-block rank

            public PreparedIndex(IndexOp[] ops, IndexType flags, int newNdim, int fancyNdim)
            {
                Ops = ops; Flags = flags; NewNdim = newNdim; FancyNdim = fancyNdim;
            }
        }

        /// <summary>
        ///     Classify and VALIDATE a normalized index tuple against <paramref name="shape"/>, exactly
        ///     as NumPy's <c>prepare_index</c>. Throws the NumPy error (IndexError) for every structurally
        ///     invalid combination — too-many-indices, a boolean array whose axis size does not match the
        ///     array, more than one ellipsis, or a non-integer/boolean array index — and returns the typed
        ///     op list + <see cref="IndexType"/> for valid ones. No gather is performed here.
        /// </summary>
        /// <param name="shape">The indexed array's shape.</param>
        /// <param name="raw">The normalized index tuple (<see cref="NormalizeIndexInputs"/>).</param>
        /// <param name="scanFancyBounds">Whether the VALUES of the integer index arrays are bounds-checked here (true, the
        ///     getter and setter dispatch: a malformed index must not reach a kernel). A store that converts its value with
        ///     the array's descriptor passes false: NumPy checks those values only in the MapIter, after the value's
        ///     conversion (<c>int32[[0, 99]] = 2**70</c> is the OverflowError, <c>float64[[0, 99]] = 2**70</c> the
        ///     IndexError), and the store itself checks them afterwards.</param>
        /// <returns>The classified op list, NumPy's <c>index_type</c> and the dimensions.</returns>
        /// <exception cref="IndexError">NumPy's <c>prepare_index</c> texts: an invalid entry, too many indices, a boolean
        ///     index of the wrong shape, an integer (or, with <paramref name="scanFancyBounds"/>, an index-array value) out of
        ///     bounds, index arrays that do not broadcast together.</exception>
        [NDScopedCovered] // only callers: the [NDScoped] FetchIndices(object[]) / SetIndices(object[], values) dispatches and NDArray.ValidateStoreIndex (mask nonzero components + aliases are scope-reclaimed)
        internal static PreparedIndex PrepareIndex(Shape shape, object[] raw, bool scanFancyBounds = true)
        {
            int arrayNdim = shape.NDim;
            var dims = shape.dimensions;

            int indexNdim = raw.Length;
            var ops = new List<IndexOp>(indexNdim + Math.Max(0, arrayNdim));

            IndexType indexType = IndexType.None;
            int usedNdim = 0, newNdim = 0, fancyNdim = 0;
            int ellipsisPos = -1;
            bool ellipsisSeen = false;

            for (int gi = 0; gi < indexNdim; gi++)
            {
                object obj = raw[gi];

                // Normalize a string slice-notation item to a Slice (NumSharp extension; one axis). Text that is not slice
                // notation is what NumPy makes of any str: a non-index entry, refused with its general text.
                if (obj is string s)
                {
                    try { obj = new Slice(s); }
                    catch { throw new IndexError(InvalidIndexMessage); }
                }

                switch (obj)
                {
                    case Slice sl when sl.IsEllipsis:
                        if (ellipsisSeen)
                            throw new IndexError("an index can only have a single ellipsis ('...')");
                        ellipsisSeen = true;
                        indexType |= IndexType.Ellipsis;
                        ellipsisPos = ops.Count;
                        ops.Add(IndexOp.Ell(0));     // fill count resolved after the walk
                        continue;                    // used += 0, new += 0

                    case Slice sl when sl.IsNewAxis:
                        indexType |= IndexType.NewAxis;
                        ops.Add(IndexOp.New());
                        newNdim += 1;                // used += 0
                        continue;

                    case Slice sl:
                        indexType |= IndexType.Slice;
                        ops.Add(IndexOp.Sl(sl));
                        usedNdim += 1; newNdim += 1;
                        continue;

                    case bool b:
                        // A raw C# bool is the 0-d bool array NumPy's PyArray_FROM_O makes of a Python bool. Alone on a 0-d
                        // array that array has the array's own (empty) shape: the single full mask, HAS_BOOL — the boolean
                        // assignment's route, as for an np.array(True) entry below. (It was classified HAS_0D_BOOL there,
                        // which sent a StringDType store's list through MapIter's temporary instead.)
                        if (indexNdim == 1 && arrayNdim == 0)
                        {
                            ops.Add(IndexOp.FullBool(NDArray.Scalar(b)));
                            indexType = IndexType.Bool;
                            return FinishPrepare(ops, indexType, ref usedNdim, ref newNdim, ref fancyNdim,
                                                 arrayNdim, dims, ellipsisPos, ellipsisSeen, indexNdim, brokeOnFullBool: true,
                                                 scanFancyBounds);
                        }
                        // Anywhere else HAS_0D_BOOL: adds a size-1 (True) or size-0 (False) axis, consumes no source axis.
                        indexType |= IndexType.ZeroDBool;
                        ops.Add(IndexOp.ZeroBool(np.array(new long[b ? 1 : 0], copy: false), b ? 1 : 0));
                        if (fancyNdim < 1) fancyNdim = 1;
                        continue;
                }

                // Integer scalar (only when the array is not 0-d, matching NumPy's array_ndims!=0 guard). On a 0-d array an
                // integer goes on to AsIndexArray, which makes it NumPy's PyArray_FROM_O 0-d integer array: an array scalar
                // that consumes an axis, so the post-walk check reports "too many indices" (not an invalid entry).
                if (arrayNdim != 0 && TryAsIntegerScalar(obj, out long ival))
                {
                    indexType |= IndexType.Integer;
                    ops.Add(IndexOp.Int(ival));
                    usedNdim += 1;                   // new += 0
                    continue;
                }

                // From here the item must be an (integer or boolean) array, a 0-d array scalar, or invalid.
                NDArray arr = AsIndexArray(obj);     // throws the NumPy invalid-index IndexError otherwise

                // A non-index dtype is refused whatever the array's size (NumPy probed: a[np.array([], float), 0] is
                // this IndexError). NumPy's force-cast of an EMPTY index to intp (mapping.c) applies only to an index
                // that is not yet an array — a Python list such as [] — and every index AsIndexArray accepts is an
                // array: an empty bool array stays a mask (its nonzero gives one empty index per mask axis, a zero-length
                // mask axis matching any length), an empty integer array is the empty fancy index below. The force-cast
                // used to apply to every empty array, so an empty float array passed and an empty 2-D mask consumed one
                // axis instead of two.
                ThrowIfNotIndexArrayDtype(arr);

                if (arr.typecode == NPTypeCode.Boolean)
                {
                    // Single full-array boolean mask (only item, exact shape) -> HAS_BOOL fast case.
                    if (indexNdim == 1 && arr.ndim == arrayNdim && DimsEqual(arr.Shape.dimensions, dims))
                    {
                        ops.Clear();
                        ops.Add(IndexOp.FullBool(arr));
                        indexType = IndexType.Bool;
                        usedNdim = arrayNdim; fancyNdim = arrayNdim;
                        // mirrors NumPy's `break` out of the parse loop
                        return FinishPrepare(ops, indexType, ref usedNdim, ref newNdim, ref fancyNdim,
                                             arrayNdim, dims, ellipsisPos, ellipsisSeen, indexNdim, brokeOnFullBool: true,
                                             scanFancyBounds);
                    }

                    if (arr.ndim == 0)
                    {
                        // 0-d boolean -> HAS_0D_BOOL: size-1 (True) / size-0 (False) axis, no source axis.
                        indexType |= IndexType.ZeroDBool;
                        bool istrue = (bool)arr;
                        long n = istrue ? 1 : 0;
                        ops.Add(IndexOp.ZeroBool(np.array(new long[n == 1 ? 1 : 0], copy: false), n));
                        if (fancyNdim < 1) fancyNdim = 1;
                        continue;
                    }

                    // k-d boolean -> its nonzero() integer arrays, one per mask axis, each consuming an
                    // axis and recording the mask axis size for the post-walk dimensionality check.
                    var mask = arr.MakeGeneric<bool>();
                    var components = np.nonzero(mask);
                    indexType |= IndexType.Fancy;
                    for (int i = 0; i < components.Length; i++)
                    {
                        ops.Add(IndexOp.FancyFromBool(components[i], arr.Shape.dimensions[i]));
                        usedNdim += 1;
                    }
                    if (fancyNdim < 1) fancyNdim = 1;
                    continue;
                }

                if (IsIntegerArrayType(arr.typecode))
                {
                    if (arr.ndim == 0)
                    {
                        // 0-d integer array == array scalar: HAS_INTEGER | HAS_SCALAR_ARRAY.
                        indexType |= IndexType.Integer | IndexType.ScalarArray;
                        ops.Add(IndexOp.ScalarArr(Convert.ToInt64(arr.GetValue(0), CultureInfo.InvariantCulture)));
                        usedNdim += 1;               // new += 0
                        continue;
                    }

                    if (fancyNdim < arr.ndim) fancyNdim = arr.ndim;
                    indexType |= IndexType.Fancy;
                    ops.Add(IndexOp.FancyArr(arr));
                    usedNdim += 1;
                    continue;
                }

                // An array of a non-index (e.g. float) dtype — unreachable since ThrowIfNotIndexArrayDtype above;
                // kept as the classifier's last word.
                throw new IndexError("arrays used as indices must be of integer (or boolean) type");
            }

            return FinishPrepare(ops, indexType, ref usedNdim, ref newNdim, ref fancyNdim,
                                 arrayNdim, dims, ellipsisPos, ellipsisSeen, indexNdim, brokeOnFullBool: false,
                                 scanFancyBounds);
        }

        /// <summary>
        ///     The post-walk phase of <see cref="PrepareIndex"/> (mapping.c:643–755): resolve / append the
        ///     ellipsis, raise the too-many-indices error, the 0-d <c>a[()]</c> special-case, the
        ///     HAS_SCALAR_ARRAY cleanup, and the boolean-array dimensionality check.
        /// </summary>
        /// <param name="ops">The walked ops (the ellipsis' fill is resolved in place).</param>
        /// <param name="indexType">The walk's index type.</param>
        /// <param name="usedNdim">The source axes the index consumed.</param>
        /// <param name="newNdim">The result's basic axes.</param>
        /// <param name="fancyNdim">The advanced block's rank.</param>
        /// <param name="arrayNdim">The source array's rank.</param>
        /// <param name="dims">The source array's dimensions.</param>
        /// <param name="ellipsisPos">The ellipsis' op position, or -1.</param>
        /// <param name="ellipsisSeen">Whether the index held an ellipsis.</param>
        /// <param name="indexNdim">The number of entries in the index.</param>
        /// <param name="brokeOnFullBool">Whether the walk stopped at a single full boolean mask.</param>
        /// <param name="scanFancyBounds">Whether the integer index arrays' values are bounds-checked (see
        ///     <see cref="PrepareIndex"/>).</param>
        /// <returns>The prepared index.</returns>
        /// <exception cref="IndexError">Too many indices, a boolean index of the wrong shape, an out-of-bounds integer or
        ///     index-array value, index arrays that do not broadcast together (NumPy's texts).</exception>
        private static PreparedIndex FinishPrepare(List<IndexOp> ops, IndexType indexType,
            ref int usedNdim, ref int newNdim, ref int fancyNdim,
            int arrayNdim, long[] dims, int ellipsisPos, bool ellipsisSeen, int indexNdim, bool brokeOnFullBool,
            bool scanFancyBounds)
        {
            if (!brokeOnFullBool)
            {
                if (usedNdim < arrayNdim)
                {
                    if (ellipsisSeen)
                    {
                        long fill = arrayNdim - usedNdim;
                        ops[ellipsisPos] = IndexOp.Ell(fill);
                        usedNdim = arrayNdim;
                        newNdim += (int)fill;
                    }
                    else
                    {
                        // No ellipsis yet and not a full index -> append one filling the rest.
                        long fill = arrayNdim - usedNdim;
                        indexType |= IndexType.Ellipsis;
                        ellipsisPos = ops.Count;
                        ops.Add(IndexOp.Ell(fill));
                        usedNdim = arrayNdim;
                        newNdim += (int)fill;
                    }
                }
                else if (usedNdim > arrayNdim)
                {
                    throw new IndexError(
                        $"too many indices for array: array is {arrayNdim}-dimensional, but {usedNdim} were indexed");
                }
                else if (indexNdim == 0)
                {
                    // a[()] on a 0-d array -> integer index (returns the scalar).
                    usedNdim = 0;
                    indexType = IndexType.Integer;
                }

                // HAS_SCALAR_ARRAY cleanup (mapping.c:687).
                if ((indexType & IndexType.ScalarArray) != 0)
                {
                    if ((indexType & IndexType.Fancy) != 0)
                        indexType &= ~IndexType.ScalarArray;
                    else if (indexType == (IndexType.Integer | IndexType.ScalarArray))
                        indexType &= ~IndexType.ScalarArray;
                }
            }

            // Now that axis placement is known, validate per source axis (mapping.c:709–752 for the
            // boolean dimensionality check; the integer/array VALUE bounds NumPy raises in the gather):
            //   • a bool-derived fancy component's axis size must equal the source axis it acts on;
            //   • a scalar-int / 0-d-array index must be within [-dim, dim) for its axis;
            //   • every element of an integer fancy array must be within [-dim, dim) for its axis.
            // These are exactly the IndexErrors NumPy raises; doing them here rejects the malformed
            // combinations before any gather/scatter (memory safety + accepts-invalid parity).
            {
                // NumPy bounds-checks integer ARRAY values at GATHER time, so when the advanced block
                // broadcasts to an EMPTY result (some advanced index has size 0) nothing is gathered and
                // the (possibly out-of-range) fancy values are never validated — e.g.
                // a[arr([99]), np.array(False)] -> (0,4), a[arr([-3,1,2,3],(4,1)), np.array([],int)]
                // -> (4,0). A SCALAR int index is still validated eagerly (NumPy checks it regardless of
                // the empty block), as is a bool array's axis size. The block is empty iff a Fancy / 0-d
                // bool op is itself size 0; broadcastability is still checked below, so an un-broadcastable
                // empty combo (e.g. (2,2) with (0,)) raises shape-mismatch rather than silently passing.
                bool advancedBlockEmpty = false;
                foreach (var op in ops)
                    if ((op.Kind == IndexKind.Fancy || op.Kind == IndexKind.ZeroDBool) && op.Array is not null && op.Array.size == 0)
                    { advancedBlockEmpty = true; break; }

                int axis = 0;
                foreach (var op in ops)
                {
                    switch (op.Kind)
                    {
                        case IndexKind.Fancy when op.Value > 0:                 // bool-derived: axis-size check
                            if (op.Value != dims[axis])
                                throw new IndexError(
                                    $"boolean index did not match indexed array along axis {axis}; size of axis is " +
                                    $"{dims[axis]} but size of corresponding boolean axis is {op.Value}");
                            break;
                        case IndexKind.Fancy:                                   // integer array (Value == -1): value bounds
                            if (!advancedBlockEmpty && scanFancyBounds)         // skipped for an empty broadcast result
                                ScanFancyBounds(op.Array, dims[axis], axis);
                            break;
                        case IndexKind.Integer:                                 // scalar int / 0-d array: value bounds
                            if (op.IntVal < -dims[axis] || op.IntVal >= dims[axis])
                                throw new IndexError(
                                    $"index {op.IntVal} is out of bounds for axis {axis} with size {dims[axis]}");
                            break;
                    }

                    switch (op.Kind)
                    {
                        case IndexKind.Ellipsis: axis += (int)op.Value; break;
                        case IndexKind.NewAxis:
                        case IndexKind.ZeroDBool: break;     // consume no source axis
                        default: axis += 1; break;
                    }
                }
            }

            // Advanced indices broadcast TOGETHER into one block; if their shapes are incompatible
            // NumPy raises (mapping.c:2617 "shape mismatch: indexing arrays could not be broadcast
            // together"). 0-d bools contribute a length-(1|0) axis; bool-derived components share one
            // length. Validate here so an un-broadcastable advanced combo is rejected up front.
            if ((indexType & IndexType.Fancy) != 0)
            {
                var advanced = new List<NDArray>();
                foreach (var op in ops)
                    if (op.Kind == IndexKind.Fancy || op.Kind == IndexKind.ZeroDBool)
                        advanced.Add(op.Array);
                if (advanced.Count > 1 && !np.are_broadcastable(advanced.ToArray()))
                {
                    string Shp(NDArray a) => "(" + string.Join(",", a.Shape.dimensions) + (a.ndim == 1 ? ",)" : ")");
                    throw new IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes " +
                                         string.Join(" ", advanced.ConvertAll(Shp)));
                }
            }

            return new PreparedIndex(ops.ToArray(), indexType, newNdim, fancyNdim);
        }

        /// <summary>
        ///     NumPy's <c>prepare_index</c> refusal of an index ARRAY whose dtype is neither integer nor bool (float,
        ///     complex, decimal, char, half, text), raised while the index is walked — before the too-many-indices and
        ///     boolean-shape checks — and for an empty or 0-d array too.
        /// </summary>
        /// <param name="arr">The index array.</param>
        /// <exception cref="IndexError"><c>arrays used as indices must be of integer (or boolean) type</c>, NumPy's text.</exception>
        internal static void ThrowIfNotIndexArrayDtype(NDArray arr)
        {
            if (arr.typecode != NPTypeCode.Boolean && !IsIntegerArrayType(arr.typecode))
                throw new IndexError("arrays used as indices must be of integer (or boolean) type");
        }

        /// <summary>True for the integer NPTypeCodes accepted as fancy indices (NumPy <c>PyArray_ISINTEGER</c>).</summary>
        private static bool IsIntegerArrayType(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Coerce a scalar index item to a long (NumPy <c>PyArray_PyIntAsIntp</c> on a non-array): the C# integer kinds
        ///     only, the Python int they stand for. A <see cref="char"/> is an integer here, as C#'s <c>char</c> →
        ///     <c>int</c> conversion makes it for the basic indexer (<see cref="TryBasicIndex"/>).
        /// </summary>
        /// <param name="obj">The index entry.</param>
        /// <param name="value">The integer, when the result is true; a <see cref="ulong"/> past <see cref="long.MaxValue"/>
        ///     wraps (it is out of bounds for every axis, which the bounds check then reports).</param>
        /// <returns>True for an integer scalar; false for everything else, which the caller hands to
        ///     <see cref="AsIndexArray"/>.</returns>
        /// <remarks>
        ///     A floating-point, float16, decimal or complex scalar is NOT an integer, whatever its value: NumPy's
        ///     <c>PyArray_PyIntAsIntp</c> refuses it, and the entry is then the invalid index of
        ///     <see cref="InvalidIndexMessage"/>. Until 2026-10-29 this accepted a <see cref="Half"/> and any
        ///     <see cref="IConvertible"/>, so a double was rounded to an integer: <c>a[0.6, ":"]</c> silently returned row 1
        ///     and <c>a[1.5, ":"]</c> reported index 2 out of bounds.
        /// </remarks>
        private static bool TryAsIntegerScalar(object obj, out long value)
        {
            switch (obj)
            {
                case int i: value = i; return true;
                case long l: value = l; return true;
                case byte b: value = b; return true;
                case sbyte sb: value = sb; return true;
                case short s: value = s; return true;
                case ushort us: value = us; return true;
                case uint ui: value = ui; return true;
                case ulong ul: value = unchecked((long)ul); return true;
                case char c: value = c; return true;
                case nint n: value = n; return true;
                case nuint nu: value = unchecked((long)nu); return true;
                default: value = 0; return false;
            }
        }

        /// <summary>
        ///     Resolve an index item that is not an integer, slice, ellipsis, new axis or bool to its index
        ///     <see cref="NDArray"/> (NumPy <c>PyArray_FROM_O</c>), or raise NumPy's invalid-index IndexError.
        /// </summary>
        /// <param name="obj">The index entry.</param>
        /// <returns>The entry as an array: itself for an <see cref="NDArray"/>, an array over an <c>int[]</c> /
        ///     <c>long[]</c>, and for an integer scalar the 0-d int64 array <c>PyArray_FROM_O</c> makes of a Python int
        ///     (reached only on a 0-d array, where the integer branch is skipped: the array scalar then counts as an
        ///     indexed axis, so the caller reports "too many indices").</returns>
        /// <exception cref="IndexError">Any other entry (a float, a float16, a decimal, a complex number, an object):
        ///     <see cref="InvalidIndexMessage"/>, NumPy's text for an input that was not an array.</exception>
        private static NDArray AsIndexArray(object obj)
        {
            switch (obj)
            {
                case NDArray nd: return nd;
                case int[] ia: return np.array(ia, copy: false);
                case long[] la: return np.array(la, copy: false);
                default:
                    if (TryAsIntegerScalar(obj, out long integer))
                        return NDArray.Scalar(integer);
                    throw new IndexError(InvalidIndexMessage);
            }
        }

        /// <summary>
        ///     Validate every element of an integer fancy index against the source axis it acts on:
        ///     NumPy raises <c>IndexError "index N is out of bounds for axis A with size S"</c> for any
        ///     value outside <c>[-dim, dim)</c>. An out-of-range unsigned value (overflowing Int64) is
        ///     necessarily out of bounds, so it maps to the same IndexError rather than an OverflowException.
        /// </summary>
        private static void ScanFancyBounds(NDArray arr, long dim, int axis)
        {
            if (arr is null || arr.size == 0)
                return;

            var flat = arr.flat;
            for (long i = 0; i < arr.size; i++)
            {
                long v;
                try { v = Convert.ToInt64(flat.GetValue(i), CultureInfo.InvariantCulture); }
                catch (OverflowException ex) { throw new IndexError($"index out of bounds for axis {axis} with size {dim}", ex); }
                if (v < -dim || v >= dim)
                    throw new IndexError($"index {v} is out of bounds for axis {axis} with size {dim}");
            }
        }

        /// <summary>
        ///     The bounds check of the integer index ARRAYS' values that <see cref="PrepareIndex"/> skips under
        ///     <c>scanFancyBounds: false</c>: NumPy runs it in its MapIter, which a store reaches only AFTER converting its
        ///     value — so the setter calls this between the conversion and the scatter (<c>a[[0, 99], 0] = 1000</c> on an int8
        ///     array is the conversion's OverflowError, not this IndexError).
        /// </summary>
        /// <param name="shape">The shape the index was prepared against.</param>
        /// <param name="prepared">What <see cref="PrepareIndex"/> returned for it.</param>
        /// <exception cref="IndexError">An index-array value outside <c>[-dim, dim)</c> for the axis it acts on (NumPy's
        ///     text, <see cref="ScanFancyBounds"/>).</exception>
        /// <remarks>Walks the ops in order with <see cref="PrepareIndex"/>'s own axis bookkeeping (an ellipsis spans its fill,
        ///     a new axis and a 0-d bool consume none) and, like its inline check, reads nothing when the advanced block
        ///     broadcasts to an EMPTY result: nothing is gathered or scattered then, so NumPy never reads the values.</remarks>
        internal static void ScanIndexArrayBounds(Shape shape, PreparedIndex prepared)
        {
            var ops = prepared.Ops;
            foreach (var op in ops)
            {
                switch (op.Kind)
                {
                    case IndexKind.Fancy:
                    case IndexKind.ZeroDBool:
                        // An advanced entry of size 0 makes the whole broadcast block empty: no value is read.
                        if (op.Array is not null && op.Array.size == 0)
                            return;
                        break;
                }
            }

            var dims = shape.dimensions;
            int axis = 0;
            foreach (var op in ops)
            {
                switch (op.Kind)
                {
                    // An integer index array (Value -1); a bool-derived one (Value > 0) had its axis size checked already.
                    case IndexKind.Fancy when op.Value <= 0:
                        ScanFancyBounds(op.Array, dims[axis], axis);
                        axis += 1;
                        break;
                    case IndexKind.Ellipsis: axis += (int)op.Value; break;
                    case IndexKind.NewAxis:
                    case IndexKind.ZeroDBool: break;     // consume no source axis
                    default: axis += 1; break;
                }
            }
        }

        /// <summary>
        ///     The shape of an ADVANCED index's result as NumPy's MapIter builds it (<c>mapiter_fill_info</c> and
        ///     <c>PyArray_MapIterSwapAxes</c>): the advanced block — the broadcast of every index array, a 0-d bool's length 1
        ///     or 0 included — placed where the advanced entries stand when they are consecutive (integers count as advanced
        ///     entries there), else in front, among the dims the slices, the new axes and the ellipsis keep.
        /// </summary>
        /// <param name="shape">The shape the index was prepared against.</param>
        /// <param name="prepared">What <see cref="PrepareIndex"/> returned for an index with an advanced entry.</param>
        /// <returns>The result's dimensions.</returns>
        /// <remarks>
        ///     Computed from the prepared index alone: no index VALUE is read, so no bounds check runs. NumPy fills a
        ///     <c>StringDType</c> store's temporary of this shape (<c>PyArray_CopyObject</c>, its broadcast error) BEFORE
        ///     MapIter checks the index values, so <c>t[[0, 99]] = ["a", "b", "c"]</c> is the broadcast error, not the
        ///     IndexError; a gather on a proxy array would raise the IndexError first. The index arrays broadcast together
        ///     (<see cref="PrepareIndex"/> checked it). A slice written as an integer (<c>"2"</c>, NumSharp's notation) is the
        ///     integer it names.
        /// </remarks>
        internal static long[] MapIterResultDims(Shape shape, PreparedIndex prepared)
        {
            var dims = shape.dimensions;
            var kept = new List<long>();          // the dims the non-advanced entries keep, in order
            var block = new List<long[]>();       // the advanced entries' index-array shapes
            int axis = 0, consec = 0;
            // NumPy's consec_status: -1 no advanced entry yet, 0 inside the block, 1 after it, 2 not consecutive.
            int status = -1;
            foreach (var op in prepared.Ops)
            {
                bool advanced;
                switch (op.Kind)
                {
                    case IndexKind.Fancy:
                    case IndexKind.ZeroDBool:
                    case IndexKind.Integer:
                        advanced = true;
                        break;
                    case IndexKind.Slice when op.Slice.IsIndex:
                        advanced = true;              // an integer written as a slice
                        break;
                    default:
                        advanced = false;
                        break;
                }
                if (advanced)
                {
                    if (status == -1)
                    {
                        consec = kept.Count;
                        status = 0;
                    }
                    else if (status == 1)
                        status = 2;
                }
                else if (status == 0)
                    status = 1;

                switch (op.Kind)
                {
                    case IndexKind.Fancy:
                        block.Add(op.Array.Shape.dimensions);
                        axis += 1;
                        break;
                    case IndexKind.ZeroDBool:
                        block.Add(op.Array.Shape.dimensions);   // its (1,) / (0,) index: the block's last dim times 1 or 0
                        break;
                    case IndexKind.Integer:
                        axis += 1;
                        break;
                    case IndexKind.Slice when op.Slice.IsIndex:
                        axis += 1;
                        break;
                    case IndexKind.Slice:
                        kept.Add(op.Slice.ToSliceDef(dims[axis]).Count);
                        axis += 1;
                        break;
                    case IndexKind.NewAxis:
                        kept.Add(1);
                        break;
                    case IndexKind.Ellipsis:
                        for (long k = 0; k < op.Value; k++)
                            kept.Add(dims[axis + k]);
                        axis += (int)op.Value;
                        break;
                }
            }

            // The block: the index arrays broadcast right-aligned (an axis takes the length that is not 1).
            int blockNdim = 0;
            foreach (var d in block)
                blockNdim = Math.Max(blockNdim, d.Length);
            var blockDims = new long[blockNdim];
            for (int k = 0; k < blockNdim; k++)
                blockDims[k] = 1;
            foreach (var d in block)
                for (int k = 0; k < d.Length; k++)
                    if (d[k] != 1)
                        blockDims[blockNdim - d.Length + k] = d[k];

            int at = status == 2 ? 0 : consec;
            var result = new long[kept.Count + blockNdim];
            for (int k = 0; k < at; k++)
                result[k] = kept[k];
            System.Array.Copy(blockDims, 0, result, at, blockNdim);
            for (int k = at; k < kept.Count; k++)
                result[blockNdim + k] = kept[k];
            return result;
        }

        private static bool DimsEqual(long[] a, long[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) return false;
            return true;
        }
    }
}
