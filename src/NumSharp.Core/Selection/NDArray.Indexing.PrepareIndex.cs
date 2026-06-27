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
        ///     op list + <see cref="IndexType"/> for valid ones. No bounds checking of integer index VALUES
        ///     (NumPy defers that to the gather), and no gather is performed here.
        /// </summary>
        internal static PreparedIndex PrepareIndex(Shape shape, object[] raw)
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

                // Normalize a string slice-notation item to a Slice (NumSharp extension; one axis).
                if (obj is string s)
                {
                    try { obj = new Slice(s); }
                    catch { throw new IndexError($"only integers, slices (':'), ellipsis ('...'), numpy.newaxis ('None') and integer or boolean arrays are valid indices (got '{s}')"); }
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
                        // A raw C# bool behaves like a 0-d boolean (HAS_0D_BOOL): adds a size-1 (True)
                        // or size-0 (False) axis, consumes no source axis.
                        indexType |= IndexType.ZeroDBool;
                        ops.Add(IndexOp.ZeroBool(np.array(new long[b ? 1 : 0], copy: false), b ? 1 : 0));
                        if (fancyNdim < 1) fancyNdim = 1;
                        continue;
                }

                // Integer scalar (only when the array is not 0-d, matching NumPy's array_ndims!=0 guard).
                if (arrayNdim != 0 && TryAsIntegerScalar(obj, out long ival))
                {
                    indexType |= IndexType.Integer;
                    ops.Add(IndexOp.Int(ival));
                    usedNdim += 1;                   // new += 0
                    continue;
                }

                // From here the item must be an (integer or boolean) array, a 0-d array scalar, or invalid.
                NDArray arr = AsIndexArray(obj);     // throws the NumPy invalid-index IndexError otherwise

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
                                             arrayNdim, dims, ellipsisPos, ellipsisSeen, indexNdim, brokeOnFullBool: true);
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

                // An array of a non-index (e.g. float) dtype.
                throw new IndexError("arrays used as indices must be of integer (or boolean) type");
            }

            return FinishPrepare(ops, indexType, ref usedNdim, ref newNdim, ref fancyNdim,
                                 arrayNdim, dims, ellipsisPos, ellipsisSeen, indexNdim, brokeOnFullBool: false);
        }

        /// <summary>
        ///     The post-walk phase of <see cref="PrepareIndex"/> (mapping.c:643–755): resolve / append the
        ///     ellipsis, raise the too-many-indices error, the 0-d <c>a[()]</c> special-case, the
        ///     HAS_SCALAR_ARRAY cleanup, and the boolean-array dimensionality check.
        /// </summary>
        private static PreparedIndex FinishPrepare(List<IndexOp> ops, IndexType indexType,
            ref int usedNdim, ref int newNdim, ref int fancyNdim,
            int arrayNdim, long[] dims, int ellipsisPos, bool ellipsisSeen, int indexNdim, bool brokeOnFullBool)
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
        ///     Coerce a scalar index item to a long (NumPy <c>PyArray_PyIntAsIntp</c> on a non-array).
        ///     Returns false for items that are arrays or otherwise not plain integer scalars.
        /// </summary>
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
                case Half h: value = Converts.ToInt64(h); return true;
                case Complex _: value = 0; return false;          // complex is not a plain integer index
                case NDArray _: value = 0; return false;          // arrays handled separately
                case Array _: value = 0; return false;            // int[]/long[] are fancy, not scalar
                case IConvertible c:
                    try { value = c.ToInt64(CultureInfo.InvariantCulture); return true; }
                    catch { value = 0; return false; }
                default: value = 0; return false;
            }
        }

        /// <summary>
        ///     Resolve a non-scalar index item to its index <see cref="NDArray"/> (NumPy
        ///     <c>PyArray_FROM_O</c>), or raise NumPy's invalid-index IndexError.
        /// </summary>
        private static NDArray AsIndexArray(object obj)
        {
            switch (obj)
            {
                case NDArray nd: return nd;
                case int[] ia: return np.array(ia, copy: false);
                case long[] la: return np.array(la, copy: false);
                default:
                    throw new IndexError(
                        "only integers, slices (':'), ellipsis ('...'), numpy.newaxis ('None') and integer or " +
                        $"boolean arrays are valid indices (got '{(obj?.GetType()?.Name ?? "null")}')");
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
                catch (OverflowException) { throw new IndexError($"index out of bounds for axis {axis} with size {dim}"); }
                if (v < -dim || v >= dim)
                    throw new IndexError($"index {v} is out of bounds for axis {axis} with size {dim}");
            }
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
