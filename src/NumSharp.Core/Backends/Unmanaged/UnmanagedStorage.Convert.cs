using System;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // An array into a C# value: the side of NumPy's convert.c (PyArray_ToList, ndarray.tolist) that NumSharp's explicit
    // conversion to a C# array runs. The array object's own member (the np-layer dtype plan's D1); the conversion operator
    // is the facade over it.
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
    }
}
