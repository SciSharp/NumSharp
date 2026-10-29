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
	            case NPTypeCode.String: return Backends.Strings.NDStringArrays.ToManagedArray(nd);
	            default:
		            throw new NotSupportedException();
            }
        }
    }
}
