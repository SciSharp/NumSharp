using System;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // An array into a C# value: the side of NumPy's convert.c (PyArray_ToList, ndarray.tolist) that NumSharp's explicit
    // conversion to a C# array runs, and PyArray_ToString (ndarray.tobytes). The array object's own members (the np-layer
    // dtype plan's D1); the conversion operator and NDArray.tobytes are the facades over them.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     The explicit conversion of an array to a C# array (<c>(Array)nd</c>): a rectangular C# array of the array's
        ///     shape and element type (NumPy's <c>ndarray.tolist()</c> with C# arrays instead of nested lists); a
        ///     fixed-width string or <c>StringDType</c> array gives its decoded elements (<see cref="string"/> for text,
        ///     <c>byte[]</c> for <c>S</c>).
        /// </summary>
        /// <param name="nd">The array (any layout, read in logical C order).</param>
        /// <returns>The C# array (a 0-d array gives a one-element vector).</returns>
        /// <exception cref="NotSupportedException">A dtype without a C# element type.</exception>
        /// <exception cref="SystemError">A <c>U</c> unit above <c>U+10FFFF</c>.</exception>
        internal static Array ToManagedArray(NDArray nd)
        {
            switch (nd.GetTypeCode)
            {
	            // %foreach supported_dtypes,supported_dtypes_lowercase%
	            // case NPTypeCode.#1: return nd.ToMuliDimArray<#2>();
	            // %
	            // default:
		            // throw new NotSupportedException();
	            case NPTypeCode.Boolean: return nd.ToMuliDimArray<bool>();
	            case NPTypeCode.Byte: return nd.ToMuliDimArray<byte>();
	            case NPTypeCode.SByte: return nd.ToMuliDimArray<sbyte>();
	            case NPTypeCode.Int16: return nd.ToMuliDimArray<short>();
	            case NPTypeCode.UInt16: return nd.ToMuliDimArray<ushort>();
	            case NPTypeCode.Int32: return nd.ToMuliDimArray<int>();
	            case NPTypeCode.UInt32: return nd.ToMuliDimArray<uint>();
	            case NPTypeCode.Int64: return nd.ToMuliDimArray<long>();
	            case NPTypeCode.UInt64: return nd.ToMuliDimArray<ulong>();
	            case NPTypeCode.Char: return nd.ToMuliDimArray<char>();
	            case NPTypeCode.Half: return nd.ToMuliDimArray<Half>();
	            case NPTypeCode.Double: return nd.ToMuliDimArray<double>();
	            case NPTypeCode.Single: return nd.ToMuliDimArray<float>();
	            case NPTypeCode.Decimal: return nd.ToMuliDimArray<decimal>();
	            case NPTypeCode.Complex: return nd.ToMuliDimArray<System.Numerics.Complex>();
	            // Text: a C# array of the decoded elements (string for U, byte[] for S) in the array's own shape.
	            case NPTypeCode.String: return TextToManagedArray(nd);
	            default:
		            throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     The explicit <c>(Array)</c> conversion of a fixed-width string array: a C# array of the array's shape whose
        ///     elements are the decoded values — <see cref="string"/> for <c>U</c>, <c>byte[]</c> for <c>S</c>, trailing NULs
        ///     stripped as NumPy's element access strips them (<c>arr.tolist()</c> with C# arrays instead of nested lists).
        ///     A 0-d array becomes a one-element vector, as <c>ToMuliDimArray</c> does for the numeric dtypes.
        /// </summary>
        /// <param name="a">A string array of any layout (read in logical C order).</param>
        /// <returns><c>string[]</c>, <c>string[,]</c>, ... or <c>byte[][]</c>, <c>byte[][,]</c>, ...</returns>
        /// <exception cref="SystemError">A <c>U</c> unit above <c>U+10FFFF</c> (the decoder's refusal, as NumPy's).</exception>
        /// <exception cref="InvalidOperationException">A dimension longer than a C# array can hold.</exception>
        private static Array TextToManagedArray(NDArray a)
        {
            // StringDType: string elements, or object ones when the NA object is not a string / None (its NA elements carry
            // that object).
            var vparams = a.Storage.IsVString ? a.dtype.VString : null;
            var elementType = vparams != null
                ? (!vparams.HasNAObject || vparams.na_object is null || vparams.na_object is string ? typeof(string) : typeof(object))
                : a.Storage.StringSlice.IsUnicode ? typeof(string) : typeof(byte[]);
            var dims = a.Shape.dimensions;
            if (dims.Length == 0)
            {
                var one = Array.CreateInstance(elementType, 1);
                one.SetValue(a.Storage.GetAtIndex(0), 0);
                return one;
            }

            var lengths = new int[dims.Length];
            for (int d = 0; d < dims.Length; d++)
                lengths[d] = dims[d] <= int.MaxValue
                    ? (int)dims[d]
                    : throw new InvalidOperationException($"Dimension {d} has {dims[d]} elements, more than a C# array holds.");
            var result = Array.CreateInstance(elementType, lengths);
            long n = a.size;
            if (n == 0)
                return result;

            // A vector fills by flat index; higher ranks walk a C-order odometer in lockstep with the flat index, so each
            // element is decoded once and stored at its coordinates (GetAtIndex resolves strides, offset and broadcast).
            if (dims.Length == 1)
            {
                for (int i = 0; i < lengths[0]; i++)
                    result.SetValue(a.Storage.GetAtIndex(i), i);
                return result;
            }
            var coords = new int[dims.Length];
            for (long flat = 0; flat < n; flat++)
            {
                result.SetValue(a.Storage.GetAtIndex(flat), coords);
                for (int d = coords.Length - 1; d >= 0; d--)
                {
                    if (++coords[d] < lengths[d])
                        break;
                    coords[d] = 0;
                }
            }
            return result;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ToString</c> (<c>convert.c</c>), the body of <c>ndarray.tobytes</c>: the array's logical
        ///     elements as raw bytes in the requested order - its own memory when it already lays them out that way, else a
        ///     copy in that order (for a dtype whose items hold references, NumPy's ravel route: the copy's cells).
        /// </summary>
        /// <param name="a">The array (any layout).</param>
        /// <param name="order">NumPy's order letter, either case: C, F, A (F for an F-contiguous-and-not-C array) or K
        ///     (C).</param>
        /// <returns>A fresh, detached byte array of length <c>size * itemsize</c>.</returns>
        /// <exception cref="ValueError">An order other than C / F / A / K (<c>order must be one of 'C', 'F', 'A', or 'K' (got
        ///     'Z')</c>), checked first.</exception>
        /// <exception cref="OverflowException">The bytes would exceed a byte array's length.</exception>
        internal static byte[] ToBytes(NDArray a, char order)
        {
            // NumPy's array_tobytes converts its order (PyArray_OrderConverter) while it parses its arguments, before
            // PyArray_ToString looks at the array: an invalid order raises whatever the size.
            // Resolve the NumPy order char to a physical layout ('C' or 'F').
            //   C/F/A -> OrderResolver (A picks F only when this view is F-contiguous-and-not-C-contiguous,
            //            exactly NumPy's PyArray_ISFORTRAN test). Invalid chars raise here.
            //   K     -> 'C'. NumPy's tobytes routes the (non-reference) numeric dtypes through CopyInto
            //            into a C-contiguous destination, so tobytes('K') == tobytes('C') for every dtype
            //            NumSharp supports. This deliberately differs from OrderResolver's 'K' (which keeps
            //            an F-contiguous source as 'F') because tobytes never preserves F for 'K'.
            char physical = (order == 'K' || order == 'k')
                ? 'C'
                : OrderResolver.Resolve(order, a.Shape);

            if (a.size == 0)
                return System.Array.Empty<byte>();

            // A dtype whose items hold references (PyDataType_REFCHK, the route NumPy's PyArray_ToString takes for it): a
            // StringDType array's bytes are 16-byte cells. A buffer that already lays its
            // elements out in the requested order gives its OWN cells — arena offsets and heap pointers of this buffer —,
            // any other layout is first RAVELLED (the "has references / needs init" path: PyArray_Ravel copies, duplicating
            // every string into a new buffer and arena in the requested order) and that copy's cells are returned.
            if (a.dtype.Meta.HasReferences)
            {
                bool own = physical == 'C' ? a.Shape.IsContiguous : a.Shape.IsFContiguous;
                using var flat = own ? null : Strings.NDVStringMove.Flatten(a, physical);
                var cellSource = flat ?? a;
                using var cells = new NDArray(cellSource.Storage.VStringCellView()) { TensorEngine = a.TensorEngine };
                return cells.tobytes(physical);
            }

            // Fast path: read the logical bytes straight from this view's buffer when its memory already
            // lays elements out in the requested order AND its data starts at Storage.Address. The
            // offset==0 guard matters: simple contiguous slices are re-based (offset folded into Address),
            // but strided/negative-stride/F-sliced views keep their start in Shape.offset, so Address would
            // point at the wrong first element (e.g. a reversed [::-1] view, or an F-contiguous T[...,1:]).
            bool directable = physical == 'C'
                ? (a.Shape.IsContiguous && a.Shape.offset == 0)
                : (a.Shape.IsFContiguous && a.Shape.offset == 0);

            // Otherwise materialize a fresh contiguous copy in the target order. copy() drives the layout
            // resolution + fill through the NDIter copy primitive (handles scalar/(1,)/strided/broadcast/
            // transposed); copy('F') leaves a buffer whose linear bytes are exactly the column-major readout.
            NDArray src = directable ? a : a.copy(physical);

            try
            {
                unsafe
                {
                    var addr = src.Storage.Address;
                    long len = checked((long)src.size * src.dtypesize);

                    // Allocate uninitialized: every byte is overwritten by the copy below, so the CLR's
                    // default zero-fill would be pure waste (a redundant 2nd write over the whole buffer).
                    // This mirrors NumPy's PyBytes_FromStringAndSize(NULL, n) — uninitialized then memcpy'd.
                    // A byte[] is capped at int.MaxValue length anyway, so the checked (int) cast is the
                    // real allocatable bound (a >2GB result throws OverflowException, as new byte[len] would).
                    byte[] bytes = GC.AllocateUninitializedArray<byte>(checked((int)len));
                    fixed (byte* @out = bytes)
                        Buffer.MemoryCopy(addr, @out, len, len);
                    return bytes;
                }
            }
            finally
            {
                // The materialized reorder copy is this method's own intermediate — the caller only
                // ever sees the detached byte[] — so it must return its pooled buffer here rather
                // than strand it for the finalizer (caught by UndisposedIntermediateTests once the
                // instance oracle tier began replaying tobytes over non-directable layouts).
                if (!ReferenceEquals(src, a))
                    src.Dispose();
            }
        }
    }
}
