using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    // ================================================================================
    // TYPE PROMOTION SYSTEM
    // ================================================================================
    //
    // This file implements NumPy-compatible type promotion for arithmetic operations.
    // When two arrays (or an array and a scalar) are combined, this system determines
    // the result dtype.
    //
    // ARCHITECTURE
    // ============
    //
    // Four lookup tables are used (two pairs for Type and NPTypeCode access):
    //
    //   _typemap_arr_arr      / _nptypemap_arr_arr      - Array + Array promotion
    //   _typemap_arr_scalar   / _nptypemap_arr_scalar   - Array + Scalar promotion
    //
    // The tables are FrozenDictionary<(T1, T2), TResult> for O(1) lookup.
    //
    // WHEN EACH TABLE IS USED
    // =======================
    //
    // The _FindCommonType(NDArray, NDArray) method decides which table to use:
    //
    //   if (both are non-scalar arrays)  → _typemap_arr_arr
    //   if (both are scalar arrays)      → _FindCommonScalarType (uses arr_arr rules)
    //   if (one is array, one is scalar) → _typemap_arr_scalar
    //
    // This matters because scalar promotion follows different rules than array promotion.
    //
    // KIND HIERARCHY
    // ==============
    //
    // Types are grouped into "kinds" with a promotion hierarchy:
    //
    //   boolean < integer < floating-point < complex
    //
    // When operands are of different kinds, the result promotes to the higher kind:
    //
    //   int32 + float32  → float64  (int promotes to float)
    //   float32 + complex → complex  (float promotes to complex)
    //
    // WITHIN-KIND PROMOTION
    // =====================
    //
    // When operands are the same kind, promotion depends on the operation type:
    //
    // Array + Array (both non-scalar):
    //   - Result is the "larger" type that can hold both ranges
    //   - uint8 + int16 → int16 (int16 can hold uint8 range + negatives)
    //   - uint32 + int32 → int64 (need 64-bit to hold both ranges)
    //   - uint64 + int64 → float64 (no integer type can hold both!)
    //
    // Array + Scalar (NEP 50 behavior):
    //   - Array dtype wins when scalar is same-kind (e.g., both integers)
    //   - uint8_array + int32_scalar → uint8 (array wins)
    //   - float32_array + int32_scalar → float32 (array wins, same effective kind)
    //
    // EXAMPLES
    // ========
    //
    //   var a = np.array(new byte[] {1, 2, 3});      // uint8
    //   var b = np.array(new int[] {4, 5, 6});       // int32
    //
    //   (a + b).dtype == np.int32                     // arr+arr: promotes to int32
    //   (a + 5).dtype == np.uint8                     // arr+scalar: array wins (NEP 50)
    //   (a + 5.0).dtype == np.float64                 // cross-kind: float wins
    //
    // REFERENCES
    // ==========
    //
    // - NumPy type promotion: https://numpy.org/doc/stable/reference/ufuncs.html#type-casting-rules
    // - NEP 50 (scalar promotion): https://numpy.org/neps/nep-0050-scalar-promotion.html
    // - Array API type promotion: https://data-apis.org/array-api/latest/API_specification/type_promotion.html
    //
    // ================================================================================

    [SuppressMessage("ReSharper", "StaticMemberInitializerReferesToMemberBelow")]
    public static partial class np
    {
        #region Privates

        /// <summary>The length of <see cref="__test_types"/> (NumPy's <c>__len_test_types</c>).</summary>
        const int __len_test_types = 19;

        /// <summary>
        ///     NumPy 1.26.4's <c>__test_types = '?' + typecodes['AllInteger'][:-2] + typecodes['AllFloat'] + 'O'</c>, with 1.26's
        ///     <c>AllInteger</c> = <c>'bBhHiIlLqQpP'</c>. <see cref="_can_coerce_all(NPTypeCode[], int)"/> walks it in this order
        ///     and returns the FIRST type every input casts to safely, so the order is the algorithm.
        /// </summary>
        /// <remarks>
        ///     Pinned to 1.26.4, the last NumPy with <c>find_common_type</c>: NumPy 2's <c>AllInteger</c> inserts <c>'nN'</c>,
        ///     but 2.0 removed the function, so no NumPy 2 list exists. The characters are NumPy's, so their storage codes
        ///     depend on the host (<c>'l'</c> is int32 on Windows): <see cref="FindCommonTypeCandidates"/> resolves them.
        /// </remarks>
        const string __test_types = "?bBhHiIlLqQefdgFDGO";

        /// <summary>
        ///  NumPy 1.26.4's <c>_kind_list</c>, the kind order <c>find_common_type</c> compares an array type's kind with a
        ///  scalar type's kind in:<br></br>
        ///  b -> boolean<br></br>
        ///  u -> unsigned integer<br></br>
        ///  i -> signed integer<br></br>
        ///  f -> floating point<br></br>
        ///  c -> complex<br></br>
        ///  M -> datetime<br></br>
        ///  m -> timedelta<br></br>
        ///  S -> string<br></br>
        ///  U -> Unicode string<br></br>
        ///  V -> record<br></br>
        ///  O -> Python object
        /// </summary>
        private static readonly char[] _kind_list = {'b', 'u', 'i', 'f', 'c', 'S', 'U', 'V', 'O', 'M', 'm'};

        /// <summary>
        ///     The storage code each <see cref="__test_types"/> character names on this host, index for index, or
        ///     <see cref="NPTypeCode.Empty"/> where NumSharp has no such dtype (<c>'F'</c> complex64, <c>'O'</c> object).
        /// </summary>
        /// <remarks>
        ///     Resolved once through <see cref="NPTypeCodeExtensions.ToTypeCode(NPY_TYPECHAR)"/>, the parser, so <c>'l'</c>/<c>'L'</c>
        ///     follow C <c>long</c> and <c>'g'</c>/<c>'G'</c> land on the float64/complex128 lanes. A nested class with an explicit
        ///     static constructor, so the table is built on the first search and never during <see cref="np"/>'s own static
        ///     initialization: a field initializer of <see cref="np"/> could run before the parser's tables (another partial
        ///     file) exist. Skipping the two absent dtypes never changes an answer for NumSharp's inputs: none of them casts
        ///     safely to complex64 without also casting to float64, and every numeric type reaches complex128 before object.
        /// </remarks>
        private static class FindCommonTypeCandidates
        {
            /// <summary>The candidate storage codes, aligned with <see cref="__test_types"/>.</summary>
            internal static readonly NPTypeCode[] Codes;

            /// <summary>Builds <see cref="Codes"/> on first use.</summary>
            static FindCommonTypeCandidates()
            {
                var codes = new NPTypeCode[__len_test_types];
                for (int i = 0; i < codes.Length; i++)
                {
                    try
                    {
                        codes[i] = ((NPY_TYPECHAR)__test_types[i]).ToTypeCode();
                    }
                    catch (NotSupportedException)
                    {
                        // complex64 and object: NumPy's candidates that NumSharp cannot store. The search skips them.
                        codes[i] = NPTypeCode.Empty;
                    }
                }

                Codes = codes;
            }
        }

	        // %foreach all_dtypes%
                // {NPTypeCode.#1, 10000 },
	        // %

        #endregion

        internal static readonly FrozenDictionary<(Type, Type), Type> _typemap_arr_arr;
        internal static readonly FrozenDictionary<(NPTypeCode, NPTypeCode), NPTypeCode> _nptypemap_arr_arr;
        internal static readonly FrozenDictionary<(Type, Type), Type> _typemap_arr_scalar;
        internal static readonly FrozenDictionary<(NPTypeCode, NPTypeCode), NPTypeCode> _nptypemap_arr_scalar;

        [SuppressMessage("ReSharper", "UseObjectOrCollectionInitializer")]
        static np()
        {
            #region arr_arr

            // ============================================================================
            // ARRAY-ARRAY TYPE PROMOTION TABLE
            // ============================================================================
            //
            // This table defines type promotion when TWO ARRAYS are combined.
            // The key is (LeftArrayType, RightArrayType), the value is the result type.
            //
            // PROMOTION RULES:
            //
            // 1. Same type: result is that type
            //    int32 + int32 → int32
            //
            // 2. Same kind, different size: result is larger type
            //    int16 + int32 → int32
            //    float32 + float64 → float64
            //
            // 3. Signed + Unsigned (same size): result is next-larger signed type
            //    int16 + uint16 → int32 (need more bits for both ranges)
            //    int32 + uint32 → int64
            //    int64 + uint64 → float64 (no larger integer exists!)
            //
            // 4. Cross-kind: result is the higher kind
            //    int32 + float32 → float64 (int32 needs float64 precision)
            //    uint8 + float32 → float32 (uint8 fits in float32)
            //
            // 5. Complex: absorbs everything
            //    float32 + complex64 → complex64
            //    int32 + complex64 → complex128 (int32 needs float64 precision)
            //
            // This table matches NumPy 2.x arr+arr behavior exactly.
            //
            // ============================================================================

            // Keyed by the DType descriptors the np.* spellings ARE (np.float64 is np.dtype("float64")); the frozen
            // Type-keyed and NPTypeCode-keyed tables below are derived from it. Structural DType equality collapses to
            // typecode equality for the builtins, so the entries — and the duplicate-key guard — are unchanged.
            var typemap_arr_arr = new Dictionary<(DType, DType), DType>(180);
            typemap_arr_arr.Add((np.@bool, np.@bool), np.@bool);
            typemap_arr_arr.Add((np.@bool, np.uint8), np.uint8);
            typemap_arr_arr.Add((np.@bool, np.int16), np.int16);
            typemap_arr_arr.Add((np.@bool, np.uint16), np.uint16);
            typemap_arr_arr.Add((np.@bool, np.int32), np.int32);
            typemap_arr_arr.Add((np.@bool, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.@bool, np.int64), np.int64);
            typemap_arr_arr.Add((np.@bool, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.@bool, np.float32), np.float32);
            typemap_arr_arr.Add((np.@bool, np.float64), np.float64);
            typemap_arr_arr.Add((np.@bool, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.@bool, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.@bool, np.@char), np.@char);
            typemap_arr_arr.Add((np.@bool, np.int8), np.int8);
            typemap_arr_arr.Add((np.@bool, np.float16), np.float16);

            typemap_arr_arr.Add((np.uint8, np.@bool), np.uint8);
            typemap_arr_arr.Add((np.uint8, np.uint8), np.uint8);
            typemap_arr_arr.Add((np.uint8, np.int16), np.int16);
            typemap_arr_arr.Add((np.uint8, np.uint16), np.uint16);
            typemap_arr_arr.Add((np.uint8, np.int32), np.int32);
            typemap_arr_arr.Add((np.uint8, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.uint8, np.int64), np.int64);
            typemap_arr_arr.Add((np.uint8, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.uint8, np.float32), np.float32);
            typemap_arr_arr.Add((np.uint8, np.float64), np.float64);
            typemap_arr_arr.Add((np.uint8, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.uint8, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint8, np.@char), np.@char);  // Char ranks as uint16: uint8+uint16 -> uint16 (label char)
            typemap_arr_arr.Add((np.uint8, np.int8), np.int16);
            typemap_arr_arr.Add((np.uint8, np.float16), np.float16);

            // int8 (sbyte) - signed 8-bit integer
            typemap_arr_arr.Add((np.int8, np.@bool), np.int8);
            typemap_arr_arr.Add((np.int8, np.uint8), np.int16);
            typemap_arr_arr.Add((np.int8, np.int8), np.int8);
            typemap_arr_arr.Add((np.int8, np.int16), np.int16);
            typemap_arr_arr.Add((np.int8, np.uint16), np.int32);
            typemap_arr_arr.Add((np.int8, np.int32), np.int32);
            typemap_arr_arr.Add((np.int8, np.uint32), np.int64);
            typemap_arr_arr.Add((np.int8, np.int64), np.int64);
            typemap_arr_arr.Add((np.int8, np.uint64), np.float64);
            typemap_arr_arr.Add((np.int8, np.float16), np.float16);
            typemap_arr_arr.Add((np.int8, np.float32), np.float32);
            typemap_arr_arr.Add((np.int8, np.float64), np.float64);
            typemap_arr_arr.Add((np.int8, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.int8, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int8, np.@char), np.int32);  // Char ranks as uint16: int8+uint16 -> int32

            typemap_arr_arr.Add((np.@char, np.@char), np.@char);
            typemap_arr_arr.Add((np.@char, np.@bool), np.@char);
            typemap_arr_arr.Add((np.@char, np.uint8), np.@char);   // Char ranks as uint16: uint16+uint8 -> uint16 (label char)
            typemap_arr_arr.Add((np.@char, np.int16), np.int32);   // Char ranks as uint16: uint16+int16 -> int32
            typemap_arr_arr.Add((np.@char, np.uint16), np.uint16);
            typemap_arr_arr.Add((np.@char, np.int32), np.int32);
            typemap_arr_arr.Add((np.@char, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.@char, np.int64), np.int64);
            typemap_arr_arr.Add((np.@char, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.@char, np.float32), np.float32);
            typemap_arr_arr.Add((np.@char, np.float64), np.float64);
            typemap_arr_arr.Add((np.@char, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.@char, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.@char, np.int8), np.int32);     // Char ranks as uint16: uint16+int8 -> int32
            typemap_arr_arr.Add((np.@char, np.float16), np.float32); // Char ranks as uint16: uint16+float16 -> float32

            typemap_arr_arr.Add((np.int16, np.@bool), np.int16);
            typemap_arr_arr.Add((np.int16, np.uint8), np.int16);
            typemap_arr_arr.Add((np.int16, np.int16), np.int16);
            typemap_arr_arr.Add((np.int16, np.uint16), np.int32);
            typemap_arr_arr.Add((np.int16, np.int32), np.int32);
            typemap_arr_arr.Add((np.int16, np.uint32), np.int64);
            typemap_arr_arr.Add((np.int16, np.int64), np.int64);
            typemap_arr_arr.Add((np.int16, np.uint64), np.float64);
            typemap_arr_arr.Add((np.int16, np.float32), np.float32);
            typemap_arr_arr.Add((np.int16, np.float64), np.float64);
            typemap_arr_arr.Add((np.int16, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.int16, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int16, np.@char), np.int32);  // Char ranks as uint16: int16+uint16 -> int32
            typemap_arr_arr.Add((np.int16, np.int8), np.int16);
            typemap_arr_arr.Add((np.int16, np.float16), np.float32);

            typemap_arr_arr.Add((np.uint16, np.@bool), np.uint16);
            typemap_arr_arr.Add((np.uint16, np.uint8), np.uint16);
            typemap_arr_arr.Add((np.uint16, np.int16), np.int32);
            typemap_arr_arr.Add((np.uint16, np.uint16), np.uint16);
            typemap_arr_arr.Add((np.uint16, np.int32), np.int32);
            typemap_arr_arr.Add((np.uint16, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.uint16, np.int64), np.int64);
            typemap_arr_arr.Add((np.uint16, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.uint16, np.float32), np.float32);
            typemap_arr_arr.Add((np.uint16, np.float64), np.float64);
            typemap_arr_arr.Add((np.uint16, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.uint16, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint16, np.@char), np.uint16);
            typemap_arr_arr.Add((np.uint16, np.int8), np.int32);
            typemap_arr_arr.Add((np.uint16, np.float16), np.float32);

            typemap_arr_arr.Add((np.int32, np.@bool), np.int32);
            typemap_arr_arr.Add((np.int32, np.uint8), np.int32);
            typemap_arr_arr.Add((np.int32, np.int16), np.int32);
            typemap_arr_arr.Add((np.int32, np.uint16), np.int32);
            typemap_arr_arr.Add((np.int32, np.int32), np.int32);
            typemap_arr_arr.Add((np.int32, np.uint32), np.int64);
            typemap_arr_arr.Add((np.int32, np.int64), np.int64);
            typemap_arr_arr.Add((np.int32, np.uint64), np.float64);
            typemap_arr_arr.Add((np.int32, np.float32), np.float64);
            typemap_arr_arr.Add((np.int32, np.float64), np.float64);
            typemap_arr_arr.Add((np.int32, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.int32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int32, np.@char), np.int32);
            typemap_arr_arr.Add((np.int32, np.int8), np.int32);
            typemap_arr_arr.Add((np.int32, np.float16), np.float64);

            typemap_arr_arr.Add((np.uint32, np.@bool), np.uint32);
            typemap_arr_arr.Add((np.uint32, np.uint8), np.uint32);
            typemap_arr_arr.Add((np.uint32, np.int16), np.int64);
            typemap_arr_arr.Add((np.uint32, np.uint16), np.uint32);
            typemap_arr_arr.Add((np.uint32, np.int32), np.int64);
            typemap_arr_arr.Add((np.uint32, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.uint32, np.int64), np.int64);
            typemap_arr_arr.Add((np.uint32, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.uint32, np.float32), np.float64);
            typemap_arr_arr.Add((np.uint32, np.float64), np.float64);
            typemap_arr_arr.Add((np.uint32, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.uint32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint32, np.@char), np.uint32);
            typemap_arr_arr.Add((np.uint32, np.int8), np.int64);
            typemap_arr_arr.Add((np.uint32, np.float16), np.float64);

            typemap_arr_arr.Add((np.int64, np.@bool), np.int64);
            typemap_arr_arr.Add((np.int64, np.uint8), np.int64);
            typemap_arr_arr.Add((np.int64, np.int16), np.int64);
            typemap_arr_arr.Add((np.int64, np.uint16), np.int64);
            typemap_arr_arr.Add((np.int64, np.int32), np.int64);
            typemap_arr_arr.Add((np.int64, np.uint32), np.int64);
            typemap_arr_arr.Add((np.int64, np.int64), np.int64);
            typemap_arr_arr.Add((np.int64, np.uint64), np.float64);
            typemap_arr_arr.Add((np.int64, np.float32), np.float64);
            typemap_arr_arr.Add((np.int64, np.float64), np.float64);
            typemap_arr_arr.Add((np.int64, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.int64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int64, np.@char), np.int64);
            typemap_arr_arr.Add((np.int64, np.int8), np.int64);
            typemap_arr_arr.Add((np.int64, np.float16), np.float64);

            typemap_arr_arr.Add((np.uint64, np.@bool), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.uint8), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.int16), np.float64);
            typemap_arr_arr.Add((np.uint64, np.uint16), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.int32), np.float64);
            typemap_arr_arr.Add((np.uint64, np.uint32), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.int64), np.float64);
            typemap_arr_arr.Add((np.uint64, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.float32), np.float64);
            typemap_arr_arr.Add((np.uint64, np.float64), np.float64);
            typemap_arr_arr.Add((np.uint64, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.uint64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint64, np.@char), np.uint64);
            typemap_arr_arr.Add((np.uint64, np.int8), np.float64);
            typemap_arr_arr.Add((np.uint64, np.float16), np.float64);

            typemap_arr_arr.Add((np.float32, np.@bool), np.float32);
            typemap_arr_arr.Add((np.float32, np.uint8), np.float32);
            typemap_arr_arr.Add((np.float32, np.int16), np.float32);
            typemap_arr_arr.Add((np.float32, np.uint16), np.float32);
            typemap_arr_arr.Add((np.float32, np.int32), np.float64);
            typemap_arr_arr.Add((np.float32, np.uint32), np.float64);
            typemap_arr_arr.Add((np.float32, np.int64), np.float64);
            typemap_arr_arr.Add((np.float32, np.uint64), np.float64);
            typemap_arr_arr.Add((np.float32, np.float32), np.float32);
            typemap_arr_arr.Add((np.float32, np.float64), np.float64);
            typemap_arr_arr.Add((np.float32, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.float32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.float32, np.@char), np.float32);
            typemap_arr_arr.Add((np.float32, np.int8), np.float32);
            typemap_arr_arr.Add((np.float32, np.float16), np.float32);

            // float16 (Half) - 16-bit floating point
            typemap_arr_arr.Add((np.float16, np.@bool), np.float16);
            typemap_arr_arr.Add((np.float16, np.uint8), np.float16);
            typemap_arr_arr.Add((np.float16, np.int8), np.float16);
            typemap_arr_arr.Add((np.float16, np.int16), np.float32);
            typemap_arr_arr.Add((np.float16, np.uint16), np.float32);
            typemap_arr_arr.Add((np.float16, np.int32), np.float64);
            typemap_arr_arr.Add((np.float16, np.uint32), np.float64);
            typemap_arr_arr.Add((np.float16, np.int64), np.float64);
            typemap_arr_arr.Add((np.float16, np.uint64), np.float64);
            typemap_arr_arr.Add((np.float16, np.float16), np.float16);
            typemap_arr_arr.Add((np.float16, np.float32), np.float32);
            typemap_arr_arr.Add((np.float16, np.float64), np.float64);
            typemap_arr_arr.Add((np.float16, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.float16, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.float16, np.@char), np.float32);  // Char ranks as uint16: float16+uint16 -> float32

            typemap_arr_arr.Add((np.float64, np.@bool), np.float64);
            typemap_arr_arr.Add((np.float64, np.uint8), np.float64);
            typemap_arr_arr.Add((np.float64, np.int16), np.float64);
            typemap_arr_arr.Add((np.float64, np.uint16), np.float64);
            typemap_arr_arr.Add((np.float64, np.int32), np.float64);
            typemap_arr_arr.Add((np.float64, np.uint32), np.float64);
            typemap_arr_arr.Add((np.float64, np.int64), np.float64);
            typemap_arr_arr.Add((np.float64, np.uint64), np.float64);
            typemap_arr_arr.Add((np.float64, np.float32), np.float64);
            typemap_arr_arr.Add((np.float64, np.float64), np.float64);
            typemap_arr_arr.Add((np.float64, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.float64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.float64, np.@char), np.float64);
            typemap_arr_arr.Add((np.float64, np.int8), np.float64);
            typemap_arr_arr.Add((np.float64, np.float16), np.float64);

            typemap_arr_arr.Add((np.complex128, np.@bool), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.uint8), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.int16), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.uint16), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.int32), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.uint32), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.int64), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.uint64), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.float32), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.float64), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.@decimal), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.@char), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.int8), np.complex128);
            typemap_arr_arr.Add((np.complex128, np.float16), np.complex128);

            typemap_arr_arr.Add((np.@decimal, np.@bool), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.uint8), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.int16), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.uint16), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.int32), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.uint32), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.int64), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.uint64), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.float32), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.float64), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.complex128), np.complex128);
            typemap_arr_arr.Add((np.@decimal, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.@char), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.int8), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.float16), np.@decimal);

            _typemap_arr_arr = typemap_arr_arr.ToFrozenDictionary(kv => (kv.Key.Item1.type, kv.Key.Item2.type), kv => kv.Value.type);

            var nptypemap_arr_arr = new Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode>(typemap_arr_arr.Count);
            foreach (var tc in typemap_arr_arr) nptypemap_arr_arr[(tc.Key.Item1.GetTypeCode(), tc.Key.Item2.GetTypeCode())] = tc.Value.GetTypeCode();
            _nptypemap_arr_arr = nptypemap_arr_arr.ToFrozenDictionary();

            #endregion

            #region arr_scalar

            // ============================================================================
            // ARRAY-SCALAR TYPE PROMOTION TABLE
            // ============================================================================
            //
            // This table defines type promotion when an array operates with a scalar value.
            // The key is (ArrayType, ScalarType), the value is the result type.
            //
            // NUMSHARP DESIGN DECISION:
            // C# primitive scalars (int, short, long, etc.) are treated as "weakly typed"
            // like Python scalars in NumPy 2.x, NOT like NumPy scalars (np.int32, etc.).
            //
            // This means: np.array(new byte[]{1,2,3}) + 5  →  uint8 result (not int32)
            //
            // WHY: This matches the natural Python/NumPy user experience where `arr + 5`
            // preserves the array's dtype when both are integers. This is consistent with
            // NumPy 2.x behavior under NEP 50 for Python scalar operands.
            //
            // NEP 50 (NumPy Enhancement Proposal 50):
            // https://numpy.org/neps/nep-0050-scalar-promotion.html
            //
            // Key rule: When an array operates with a scalar of the same "kind" (e.g., both
            // are integers), the array dtype wins. Cross-kind operations (int + float) still
            // promote to the higher kind (float).
            //
            // AFFECTED ENTRIES (12 total - all unsigned array + signed scalar):
            //
            // | Array Type | Scalar Types      | NumPy 1.x Result | NumPy 2.x Result |
            // |------------|-------------------|------------------|------------------|
            // | uint8      | int16/int32/int64 | int16/int32/int64| uint8            |
            // | uint16     | int16/int32/int64 | int32/int32/int64| uint16           |
            // | uint32     | int16/int32/int64 | int64/int64/int64| uint32           |
            // | uint64     | int16/int32/int64 | float64 (!)      | uint64           |
            //
            // Verified against NumPy 2.4.2:
            //   >>> (np.array([1,2,3], np.uint8) + 5).dtype
            //   dtype('uint8')
            //
            // ============================================================================

            var typemap_arr_scalar = new Dictionary<(DType, DType), DType>();
            typemap_arr_scalar.Add((np.@bool, np.@bool), np.@bool);
            typemap_arr_scalar.Add((np.@bool, np.uint8), np.uint8);
            typemap_arr_scalar.Add((np.@bool, np.@char), np.@char);  // bug 5: bool array + char scalar -> char (was missing -> KeyNotFoundException)
            typemap_arr_scalar.Add((np.@bool, np.int16), np.int16);
            typemap_arr_scalar.Add((np.@bool, np.uint16), np.uint16);
            // bool array + C# int scalar -> int64, NOT int32. The C# `int` literal is the weak
            // Python int literal (house NEP50 mapping), and NumPy 2.4.2 promotes bool + weak int
            // to the DEFAULT integer because int is a different KIND than bool (probed:
            // (np.array([True]) + 2).dtype == int64; same for *, &, |, ^, <<, >>). The narrower
            // C#-only spellings ((short)2, (byte)2, ...) keep their strong np.int16/np.uint8
            // reading above, matching the oracle's weak-int-boxes-as-long convention.
            typemap_arr_scalar.Add((np.@bool, np.int32), np.int64);
            typemap_arr_scalar.Add((np.@bool, np.uint32), np.uint32);
            typemap_arr_scalar.Add((np.@bool, np.int64), np.int64);
            typemap_arr_scalar.Add((np.@bool, np.uint64), np.uint64);
            typemap_arr_scalar.Add((np.@bool, np.float32), np.float32);
            typemap_arr_scalar.Add((np.@bool, np.float64), np.float64);
            typemap_arr_scalar.Add((np.@bool, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.@bool, np.int8), np.int8);
            typemap_arr_scalar.Add((np.@bool, np.float16), np.float16);

            typemap_arr_scalar.Add((np.uint8, np.@bool), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.uint8), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.@char), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.int16), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.uint16), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.int32), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.uint32), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.int64), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.uint64), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.float32), np.float32);
            typemap_arr_scalar.Add((np.uint8, np.float64), np.float64);
            typemap_arr_scalar.Add((np.uint8, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.uint8, np.int8), np.uint8);
            typemap_arr_scalar.Add((np.uint8, np.float16), np.float16);

            // int8 (sbyte) arr_scalar entries
            typemap_arr_scalar.Add((np.int8, np.@bool), np.int8);
            typemap_arr_scalar.Add((np.int8, np.uint8), np.int8);
            typemap_arr_scalar.Add((np.int8, np.@char), np.int8);
            typemap_arr_scalar.Add((np.int8, np.int8), np.int8);
            typemap_arr_scalar.Add((np.int8, np.int16), np.int8);
            typemap_arr_scalar.Add((np.int8, np.uint16), np.int8);
            typemap_arr_scalar.Add((np.int8, np.int32), np.int8);
            typemap_arr_scalar.Add((np.int8, np.uint32), np.int8);
            typemap_arr_scalar.Add((np.int8, np.int64), np.int8);
            typemap_arr_scalar.Add((np.int8, np.uint64), np.int8);
            typemap_arr_scalar.Add((np.int8, np.float16), np.float16);
            typemap_arr_scalar.Add((np.int8, np.float32), np.float32);
            typemap_arr_scalar.Add((np.int8, np.float64), np.float64);
            typemap_arr_scalar.Add((np.int8, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.int8, np.@decimal), np.int8);

            // Char array behaves as a uint16 array: NEP50 array-wins over every
            // integer/bool scalar (-> char), float/complex scalar promotes by kind.
            typemap_arr_scalar.Add((np.@char, np.@char), np.@char);
            typemap_arr_scalar.Add((np.@char, np.@bool), np.@char);
            typemap_arr_scalar.Add((np.@char, np.uint8), np.@char);
            typemap_arr_scalar.Add((np.@char, np.int16), np.@char);
            typemap_arr_scalar.Add((np.@char, np.uint16), np.@char);
            typemap_arr_scalar.Add((np.@char, np.int32), np.@char);
            typemap_arr_scalar.Add((np.@char, np.uint32), np.@char);
            typemap_arr_scalar.Add((np.@char, np.int64), np.@char);
            typemap_arr_scalar.Add((np.@char, np.uint64), np.@char);
            typemap_arr_scalar.Add((np.@char, np.float32), np.float32);
            typemap_arr_scalar.Add((np.@char, np.float64), np.float64);
            typemap_arr_scalar.Add((np.@char, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.@char, np.int8), np.@char);
            typemap_arr_scalar.Add((np.@char, np.float16), np.float32);  // uint16 array + float16 scalar -> float32

            typemap_arr_scalar.Add((np.int16, np.@bool), np.int16);
            typemap_arr_scalar.Add((np.int16, np.uint8), np.int16);
            typemap_arr_scalar.Add((np.int16, np.@char), np.int16);
            typemap_arr_scalar.Add((np.int16, np.int16), np.int16);
            typemap_arr_scalar.Add((np.int16, np.uint16), np.int16);
            typemap_arr_scalar.Add((np.int16, np.int32), np.int16);
            typemap_arr_scalar.Add((np.int16, np.uint32), np.int16);
            typemap_arr_scalar.Add((np.int16, np.int64), np.int16);
            typemap_arr_scalar.Add((np.int16, np.uint64), np.int16);
            typemap_arr_scalar.Add((np.int16, np.float32), np.float32);
            typemap_arr_scalar.Add((np.int16, np.float64), np.float64);
            typemap_arr_scalar.Add((np.int16, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.int16, np.int8), np.int16);
            typemap_arr_scalar.Add((np.int16, np.float16), np.float32);

            typemap_arr_scalar.Add((np.uint16, np.@bool), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.uint8), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.@char), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.int16), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.uint16), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.int32), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.uint32), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.int64), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.uint64), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.float32), np.float32);
            typemap_arr_scalar.Add((np.uint16, np.float64), np.float64);
            typemap_arr_scalar.Add((np.uint16, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.uint16, np.int8), np.uint16);
            typemap_arr_scalar.Add((np.uint16, np.float16), np.float32);

            typemap_arr_scalar.Add((np.int32, np.@bool), np.int32);
            typemap_arr_scalar.Add((np.int32, np.uint8), np.int32);
            typemap_arr_scalar.Add((np.int32, np.@char), np.int32);
            typemap_arr_scalar.Add((np.int32, np.int16), np.int32);
            typemap_arr_scalar.Add((np.int32, np.uint16), np.int32);
            typemap_arr_scalar.Add((np.int32, np.int32), np.int32);
            typemap_arr_scalar.Add((np.int32, np.uint32), np.int32);
            typemap_arr_scalar.Add((np.int32, np.int64), np.int32);
            typemap_arr_scalar.Add((np.int32, np.uint64), np.int32);
            typemap_arr_scalar.Add((np.int32, np.float32), np.float64);
            typemap_arr_scalar.Add((np.int32, np.float64), np.float64);
            typemap_arr_scalar.Add((np.int32, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.int32, np.int8), np.int32);
            typemap_arr_scalar.Add((np.int32, np.float16), np.float64);

            typemap_arr_scalar.Add((np.uint32, np.@bool), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.uint8), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.@char), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.int16), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.uint16), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.int32), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.uint32), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.int64), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.uint64), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.float32), np.float64);
            typemap_arr_scalar.Add((np.uint32, np.float64), np.float64);
            typemap_arr_scalar.Add((np.uint32, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.uint32, np.int8), np.uint32);
            typemap_arr_scalar.Add((np.uint32, np.float16), np.float64);

            typemap_arr_scalar.Add((np.int64, np.@bool), np.int64);
            typemap_arr_scalar.Add((np.int64, np.uint8), np.int64);
            typemap_arr_scalar.Add((np.int64, np.@char), np.int64);
            typemap_arr_scalar.Add((np.int64, np.int16), np.int64);
            typemap_arr_scalar.Add((np.int64, np.uint16), np.int64);
            typemap_arr_scalar.Add((np.int64, np.int32), np.int64);
            typemap_arr_scalar.Add((np.int64, np.uint32), np.int64);
            typemap_arr_scalar.Add((np.int64, np.int64), np.int64);
            typemap_arr_scalar.Add((np.int64, np.uint64), np.int64);
            typemap_arr_scalar.Add((np.int64, np.float32), np.float64);
            typemap_arr_scalar.Add((np.int64, np.float64), np.float64);
            typemap_arr_scalar.Add((np.int64, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.int64, np.int8), np.int64);
            typemap_arr_scalar.Add((np.int64, np.float16), np.float64);

            typemap_arr_scalar.Add((np.uint64, np.@bool), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.uint8), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.@char), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.int16), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.uint16), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.int32), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.uint32), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.int64), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.uint64), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.float32), np.float64);
            typemap_arr_scalar.Add((np.uint64, np.float64), np.float64);
            typemap_arr_scalar.Add((np.uint64, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.uint64, np.int8), np.uint64);
            typemap_arr_scalar.Add((np.uint64, np.float16), np.float64);

            typemap_arr_scalar.Add((np.float32, np.@bool), np.float32);
            typemap_arr_scalar.Add((np.float32, np.uint8), np.float32);
            typemap_arr_scalar.Add((np.float32, np.@char), np.float32);
            typemap_arr_scalar.Add((np.float32, np.int16), np.float32);
            typemap_arr_scalar.Add((np.float32, np.uint16), np.float32);
            typemap_arr_scalar.Add((np.float32, np.int32), np.float32);
            typemap_arr_scalar.Add((np.float32, np.uint32), np.float32);
            typemap_arr_scalar.Add((np.float32, np.int64), np.float32);
            typemap_arr_scalar.Add((np.float32, np.uint64), np.float32);
            typemap_arr_scalar.Add((np.float32, np.float32), np.float32);
            typemap_arr_scalar.Add((np.float32, np.float64), np.float32);
            typemap_arr_scalar.Add((np.float32, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.float32, np.int8), np.float32);
            typemap_arr_scalar.Add((np.float32, np.float16), np.float32);

            // float16 (Half) arr_scalar entries
            typemap_arr_scalar.Add((np.float16, np.@bool), np.float16);
            typemap_arr_scalar.Add((np.float16, np.uint8), np.float16);
            typemap_arr_scalar.Add((np.float16, np.@char), np.float16);
            typemap_arr_scalar.Add((np.float16, np.int8), np.float16);
            typemap_arr_scalar.Add((np.float16, np.int16), np.float16);
            typemap_arr_scalar.Add((np.float16, np.uint16), np.float16);
            typemap_arr_scalar.Add((np.float16, np.int32), np.float16);
            typemap_arr_scalar.Add((np.float16, np.uint32), np.float16);
            typemap_arr_scalar.Add((np.float16, np.int64), np.float16);
            typemap_arr_scalar.Add((np.float16, np.uint64), np.float16);
            typemap_arr_scalar.Add((np.float16, np.float16), np.float16);
            typemap_arr_scalar.Add((np.float16, np.float32), np.float16);
            typemap_arr_scalar.Add((np.float16, np.float64), np.float16);
            typemap_arr_scalar.Add((np.float16, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.float16, np.@decimal), np.float16);

            typemap_arr_scalar.Add((np.float64, np.@bool), np.float64);
            typemap_arr_scalar.Add((np.float64, np.uint8), np.float64);
            typemap_arr_scalar.Add((np.float64, np.@char), np.float64);
            typemap_arr_scalar.Add((np.float64, np.int16), np.float64);
            typemap_arr_scalar.Add((np.float64, np.uint16), np.float64);
            typemap_arr_scalar.Add((np.float64, np.int32), np.float64);
            typemap_arr_scalar.Add((np.float64, np.uint32), np.float64);
            typemap_arr_scalar.Add((np.float64, np.int64), np.float64);
            typemap_arr_scalar.Add((np.float64, np.uint64), np.float64);
            typemap_arr_scalar.Add((np.float64, np.float32), np.float64);
            typemap_arr_scalar.Add((np.float64, np.float64), np.float64);
            typemap_arr_scalar.Add((np.float64, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.float64, np.int8), np.float64);
            typemap_arr_scalar.Add((np.float64, np.float16), np.float64);

            typemap_arr_scalar.Add((np.complex128, np.@bool), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.uint8), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.@char), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.int16), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.uint16), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.int32), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.uint32), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.int64), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.uint64), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.float32), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.float64), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.int8), np.complex128);
            typemap_arr_scalar.Add((np.complex128, np.float16), np.complex128);

            typemap_arr_scalar.Add((np.@decimal, np.@bool), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.uint8), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.@char), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.int16), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.uint16), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.int32), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.uint32), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.int64), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.uint64), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.float32), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.float64), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.complex128), np.complex128);
            typemap_arr_scalar.Add((np.@decimal, np.@decimal), np.@decimal);
            typemap_arr_scalar.Add((np.@bool, np.@decimal), np.@bool);
            typemap_arr_scalar.Add((np.uint8, np.@decimal), np.uint8);
            typemap_arr_scalar.Add((np.@char, np.@decimal), np.@char);
            typemap_arr_scalar.Add((np.int16, np.@decimal), np.int16);
            typemap_arr_scalar.Add((np.uint16, np.@decimal), np.uint16);
            typemap_arr_scalar.Add((np.int32, np.@decimal), np.int32);
            typemap_arr_scalar.Add((np.uint32, np.@decimal), np.uint32);
            typemap_arr_scalar.Add((np.int64, np.@decimal), np.int64);
            typemap_arr_scalar.Add((np.uint64, np.@decimal), np.uint64);
            typemap_arr_scalar.Add((np.float32, np.@decimal), np.float32);
            typemap_arr_scalar.Add((np.float64, np.@decimal), np.float64);
            typemap_arr_scalar.Add((np.complex128, np.@decimal), np.complex128);
            typemap_arr_scalar.Add((np.@decimal, np.int8), np.@decimal);
            typemap_arr_scalar.Add((np.@decimal, np.float16), np.@decimal);

            _typemap_arr_scalar = typemap_arr_scalar.ToFrozenDictionary(kv => (kv.Key.Item1.type, kv.Key.Item2.type), kv => kv.Value.type);

            var nptypemap_arr_scalar = new Dictionary<(NPTypeCode, NPTypeCode), NPTypeCode>(typemap_arr_scalar.Count);
            foreach (var tc in typemap_arr_scalar) nptypemap_arr_scalar[(tc.Key.Item1.GetTypeCode(), tc.Key.Item2.GetTypeCode())] = tc.Value.GetTypeCode();
            _nptypemap_arr_scalar = nptypemap_arr_scalar.ToFrozenDictionary();

            #endregion
        }

        // @formatter:off — disable formatter after this line
        internal static NPTypeCode[] powerOrder =
        {
            NPTypeCode.Boolean, 
            NPTypeCode.Byte, //Int8
            NPTypeCode.Byte, //unit8
            NPTypeCode.Int16, 
            NPTypeCode.UInt16, 
            NPTypeCode.Int32, 
            NPTypeCode.UInt32, 
            NPTypeCode.Int32, 
            NPTypeCode.UInt32, 
            NPTypeCode.Int64, 
            NPTypeCode.UInt64, 
            //NPTypeCode.Single, //Float16
            NPTypeCode.Single, //Float32
            NPTypeCode.Double, //Float64
            NPTypeCode.Double,
            NPTypeCode.Double,
            NPTypeCode.Decimal,
            NPTypeCode.Decimal,
            NPTypeCode.Complex, //Complex64
            NPTypeCode.Complex, //Complex128
            //NPTypeCode.Complex, //Complex128
            NPTypeCode.Single,
        };
        // @formatter:off — disable formatter after this line

        private static readonly (NPTypeCode Type, int Priority)[] powerPriorities =
        {
            (NPTypeCode.Boolean, NPTypeCode.Boolean.GetPriority()),
            (NPTypeCode.Byte, NPTypeCode.Byte.GetPriority()), //Int8
            (NPTypeCode.Byte, NPTypeCode.Byte.GetPriority()), //unit8
            (NPTypeCode.Int16, NPTypeCode.Int16.GetPriority()),
            (NPTypeCode.UInt16, NPTypeCode.UInt16.GetPriority()),
            (NPTypeCode.Int32, NPTypeCode.Int32.GetPriority()),
            (NPTypeCode.UInt32, NPTypeCode.UInt32.GetPriority()),
            (NPTypeCode.Int32, NPTypeCode.Int32.GetPriority()),
            (NPTypeCode.UInt32, NPTypeCode.UInt32.GetPriority()),
            (NPTypeCode.Int64, NPTypeCode.Int64.GetPriority()),
            (NPTypeCode.UInt64, NPTypeCode.UInt64.GetPriority()),
            //NPTypeCode.Single, NPTypeCode.Single.GetPriority()), //Float16
            (NPTypeCode.Single, NPTypeCode.Single.GetPriority()), //Float32
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority()), //Float64
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority()),
            (NPTypeCode.Double, NPTypeCode.Double.GetPriority() + 1),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Decimal, NPTypeCode.Decimal.GetPriority()),
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex64
            (NPTypeCode.Complex, NPTypeCode.Complex.GetPriority()), //Complex128
            //NPTypeCode.Complex, //Complex128
            (NPTypeCode.Single, NPTypeCode.Single.GetPriority()),
        };

        // @formatter:on — enable formatter after this line

        /// <summary>
        ///     Determine common type following standard coercion rules: NumPy 1.26.4's <c>np.find_common_type</c>, the last
        ///     NumPy that has it (deprecated in 1.25, removed in 2.0). Kept on purpose, so code written against NumPy 1.x
        ///     compiles and runs unchanged; new code writes <see cref="promote_types(DType, DType)"/> or
        ///     <see cref="result_type(object[])"/>, passing the Python values <c>0</c>, <c>0.0</c> or <c>0j</c> for the scalar
        ///     types.
        /// </summary>
        /// <param name="array_types">The dtypes of the arrays. Null counts as none.</param>
        /// <param name="scalar_types">The dtypes of the scalars. Null counts as none.</param>
        /// <returns>
        ///     The common data type, which is the maximum of <paramref name="array_types"/> ignoring
        ///     <paramref name="scalar_types"/>, unless the maximum of <paramref name="scalar_types"/> is of a different, higher
        ///     kind (<c>dtype.kind</c>): then the first type both maxima cast to safely. Null (NumPy's None) when nothing is
        ///     given, or when no NumPy dtype NumSharp has holds the inputs (str with a number: NumPy's answer, object, is not a
        ///     NumSharp dtype).
        /// </returns>
        /// <remarks>
        /// <para>
        ///     NumPy 2 raises <c>AttributeError: `np.find_common_type` was removed in the NumPy 2.0 release. Use
        ///     `numpy.promote_types` or `numpy.result_type` instead. To achieve semantics for the `scalar_types` argument, use
        ///     `numpy.result_type` and pass the Python values `0`, `0.0`, or `0j`.</c>
        /// </para>
        /// <para>
        ///     <b>The maximum is a search, not a promotion.</b> Each maximum is the first type of NumPy's candidate list
        ///     <c>?bBhHiIlLqQefdgFDGO</c> that every input casts to safely, so <c>[int8, uint8, float16]</c> gives float16 where
        ///     a pairwise <c>promote_types</c> chain gives float32. A higher-kind scalar can move the result:
        ///     <c>find_common_type([uint8], [int16])</c> is int16 (NEP 50's weak-scalar rule, which NumSharp's arithmetic uses,
        ///     keeps uint8).
        /// </para>
        /// <para>
        ///     The rule is NumPy 1.x's, not NEP 50's: a scalar type counts only when its KIND is higher than every array
        ///     type's, and then by its type, never by a value. <c>np.result_type</c> is the NumPy 2 rule, where a C#
        ///     literal is a weak Python scalar (<c>np.result_type(int8_array, 300)</c> is int8) and a dtype is always
        ///     strong (<c>np.result_type(np.int8, np.int64)</c> is int64, where <c>find_common_type([int8], [int64])</c> is int8).
        /// </para>
        /// <para>
        ///     NumSharp's own dtypes (<see cref="NPTypeCode.Decimal"/>, <see cref="NPTypeCode.Char"/>) have no place in NumPy's
        ///     candidate list: any list holding one is promoted with NumSharp's promotion table instead, the rule its arithmetic
        ///     uses (decimal wins over the floats, char promotes as uint16).
        /// </para>
        /// <para>
        ///     NumPy 1.26's page: https://numpy.org/doc/1.26/reference/generated/numpy.find_common_type.html; the
        ///     replacement: https://numpy.org/doc/stable/reference/generated/numpy.result_type.html
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentException">
        ///     An entry is <see cref="NPTypeCode.Empty"/> (no dtype), which is also what a <see cref="Type"/> that is not a
        ///     NumSharp element type converts to.
        /// </exception>
        /// <exception cref="NotSupportedException">A dtype string names a dtype NumSharp cannot store (the <see cref="string"/> overload).</exception>
        public static DType find_common_type(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        #region Overloads

        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>
        public static DType find_common_type(Type[] array_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), Array.Empty<NPTypeCode>());
        }

        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>
        public static DType find_common_type(Type[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>
        public static DType find_common_type(string[] array_types, string[] scalar_types)
        {
            // GetTypeCode, not typecode: a datetime string parses to a descriptor without storage, which should say so
            // instead of turning into Empty and failing later as "no dtype".
            return _FindCommonType(array_types?.Select(v => np.dtype(v).GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => np.dtype(v).GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>
        public static DType find_common_type(Type[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>
        public static DType find_common_type(NPTypeCode[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/> with no scalar types, in the <see cref="DType"/> form
        ///     the <c>np.float32</c>/<c>np.int64</c> spellings are (<c>new[] { np.float32, np.int64 }</c> is a <c>DType[]</c>).
        /// </summary>
        /// <param name="array_types">The dtypes of the arrays. Null counts as none.</param>
        /// <returns>The first type of NumPy's candidate list every entry casts to safely, or null (NumPy's None) for none.</returns>
        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])" path="/remarks"/>
        /// <exception cref="NotSupportedException">An entry has no NumSharp storage yet (a datetime64 descriptor).</exception>
        public static DType find_common_type(DType[] array_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/> in the <see cref="DType"/> form the
        ///     <c>np.float32</c>/<c>np.int64</c> spellings are (<c>new[] { np.float32, np.int64 }</c> is a <c>DType[]</c>).
        /// </summary>
        /// <param name="array_types">The dtypes of the arrays. Null counts as none.</param>
        /// <param name="scalar_types">The dtypes of the scalars. Null counts as none.</param>
        /// <returns>See <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>.</returns>
        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])" path="/remarks"/>
        /// <exception cref="NotSupportedException">An entry has no NumSharp storage yet (a datetime64 descriptor).</exception>
        public static DType find_common_type(DType[] array_types, DType[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        #endregion

        internal static NPTypeCode _FindCommonArrayType(Type dtype_left, Type dtype_right)
        {
            return _nptypemap_arr_arr[(dtype_left.GetTypeCode(), dtype_right.GetTypeCode())];
        }

        internal static NPTypeCode _FindCommonArrayType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _nptypemap_arr_arr[(dtype_left, dtype_right)];
        }

        /// <inheritdoc cref="_FindCommonScalarType(NPTypeCode, NPTypeCode)"/>
        internal static NPTypeCode _FindCommonScalarType(Type dtype_left, Type dtype_right)
        {
            return _FindCommonScalarType(dtype_left.GetTypeCode(), dtype_right.GetTypeCode());
        }

        /// <summary>
        ///     The result type of an operation between two 0-d arrays (<see cref="_FindCommonType(NDArray, NDArray)"/>): the
        ///     array-array promotion table, the rule NumPy 2 applies to two strong operands.
        /// </summary>
        /// <param name="dtype_left">The left operand's storage code.</param>
        /// <param name="dtype_right">The right operand's storage code.</param>
        /// <returns>The promoted storage code.</returns>
        /// <remarks>
        ///     A direct table lookup, NOT <c>find_common_type</c>'s search: this is binary-operator promotion, and it reads the
        ///     table with the same key order <c>(right, left)</c> the old two-entry fold did, so every 0-d result is unchanged.
        /// </remarks>
        /// <exception cref="System.Collections.Generic.KeyNotFoundException">A code the promotion table has no row for (<see cref="NPTypeCode.Empty"/>).</exception>
        internal static NPTypeCode _FindCommonScalarType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _nptypemap_arr_arr[(dtype_right, dtype_left)];
        }

        internal static NPTypeCode _FindCommonArrayScalarType(NPTypeCode dtypeArray, NPTypeCode dtypeScalar)
        {
            return _nptypemap_arr_scalar[(dtypeArray, dtypeScalar)];
        }

        internal static NPTypeCode _FindCommonArrayScalarType(Type dtypeArray, Type dtypeScalar)
        {
            return _nptypemap_arr_scalar[(dtypeArray.GetTypeCode(), dtypeScalar.GetTypeCode())];
        }

        /// <summary>
        ///     NumPy 1.26.4's <c>find_common_type(array_types, scalar_types)</c> body over storage codes, the engine of
        ///     <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/>: each list's maximum by <see cref="_can_coerce_all(NPTypeCode[], int)"/>,
        ///     then the array maximum unless the scalar maximum is of a higher kind, when <see cref="_find_common_coerce"/>
        ///     decides.
        /// </summary>
        /// <param name="array_types">The dtypes of the arrays.</param>
        /// <param name="scalar_types">The dtypes of the scalars.</param>
        /// <returns>The common storage code, or <see cref="NPTypeCode.Empty"/> where NumPy returns None.</returns>
        /// <remarks>
        ///     Kinds are ranked through <see cref="DType._kind_list_map"/>, NumSharp's frozen legacy kind map, in NumPy's
        ///     <see cref="_kind_list"/>. It spells bool <c>'?'</c>, which <see cref="_kind_list"/> lacks, so bool ranks -1
        ///     where NumPy's <c>'b'</c> ranks 0: both are the lowest rank, so every comparison agrees. Not a hot path: binary
        ///     operators use <see cref="_FindCommonType(NDArray, NDArray)"/>, which never comes here.
        /// </remarks>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            NPTypeCode maxa = _can_coerce_all(array_types);
            NPTypeCode maxsc = _can_coerce_all(scalar_types);

            if (maxa == NPTypeCode.Empty)
                return maxsc;

            if (maxsc == NPTypeCode.Empty)
                return maxa;

            // NumPy: `_kind_list.index(kind)`, whose ValueError means "return None".
            if (!DType._kind_list_map.TryGetValue(maxa, out char kind_a) || !DType._kind_list_map.TryGetValue(maxsc, out char kind_sc))
                return NPTypeCode.Empty;
            int index_a = Array.IndexOf(_kind_list, kind_a);
            int index_sc = Array.IndexOf(_kind_list, kind_sc);

            // A scalar of a higher kind is NumPy's _find_common_coerce. (This branch used to read the NEP 50 array-scalar
            // table, NumPy 2's weak-scalar arithmetic rule, which is not what find_common_type computes: it kept uint8 for
            // ([uint8], [int16]) where NumPy 1.26.4 answers int16.)
            if (index_sc > index_a)
                return _find_common_coerce(maxsc, maxa);
            return maxa;
        }

        /// <summary>The <see cref="List{T}"/> form of <see cref="_FindCommonType(NPTypeCode[], NPTypeCode[])"/>.</summary>
        /// <param name="array_types">The dtypes of the arrays.</param>
        /// <param name="scalar_types">The dtypes of the scalars.</param>
        /// <returns>The common storage code, or <see cref="NPTypeCode.Empty"/> where NumPy returns None.</returns>
        /// <remarks>One implementation for both forms: the two used to differ (this one alone reached <see cref="_find_common_coerce"/>).</remarks>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(List<NPTypeCode> array_types, List<NPTypeCode> scalar_types)
        {
            return _FindCommonType(array_types.ToArray(), scalar_types.ToArray());
        }

        /// <summary>
        ///     The dtype of an array built from a sequence of scalars of the given dtypes, in their order (the caller is
        ///     <c>np.asanyarray</c> over an <c>object[]</c>): NumPy's array-coercion promotion, <c>handle_promotion</c> in
        ///     <c>array_coercion.c</c>, which promotes the running dtype with each new one.
        /// </summary>
        /// <param name="scalar_types">The dtypes, in element order (repeats may be dropped: a repeat cannot change the result).</param>
        /// <returns>The promoted storage code, or <see cref="NPTypeCode.Empty"/> for no types.</returns>
        /// <remarks>
        ///     A left fold, ORDER-SENSITIVE as NumPy's is: <c>[int8, uint8, float16]</c> is float32 but
        ///     <c>[float16, int8, uint8]</c> is float16 (NumPy 2.4.2, 18 such dtype sets). Not <c>find_common_type</c>'s
        ///     search; the name is historical. The fold it replaces dropped every other pair from four types on.
        /// </remarks>
        /// <exception cref="System.Collections.Generic.KeyNotFoundException">A code the promotion table has no row for (<see cref="NPTypeCode.Empty"/>).</exception>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType_Scalar(params NPTypeCode[] scalar_types)
        {
            return _promote_sequence(scalar_types);
        }

        /// <summary>
        ///     <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/> of array types only: NumPy 1.26.4's
        ///     <c>_can_coerce_all(array_types)</c>.
        /// </summary>
        /// <param name="array_types">The dtypes of the arrays.</param>
        /// <returns>The first candidate every entry casts to safely, the entry itself for one, <see cref="NPTypeCode.Empty"/> for none.</returns>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType_Array(params NPTypeCode[] array_types)
        {
            return _can_coerce_all(array_types);
        }

        /// <summary>
        ///     NumPy 1.26.4's <c>find_common_type</c> over arrays, splitting them as <c>np.find_common_type</c>'s two lists:
        ///     0-d arrays are the scalar types, the rest the array types.
        /// </summary>
        /// <param name="involvedArrays">The arrays.</param>
        /// <returns>The common storage code, or <see cref="NPTypeCode.Empty"/> where NumPy returns None.</returns>
        /// <remarks>Two arrays bind <see cref="_FindCommonType(NDArray, NDArray)"/> (binary-operator promotion) instead.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(params NDArray[] involvedArrays)
        {
            List<NPTypeCode> scalar = new List<NPTypeCode>(involvedArrays.Length);
            List<NPTypeCode> list = new List<NPTypeCode>(involvedArrays.Length);
            foreach (var nd in involvedArrays)
            {
                if (nd.Shape.IsScalar)
                    scalar.Add(nd.GetTypeCode);
                else
                    list.Add(nd.GetTypeCode);
            }

            return _FindCommonType(list, scalar);
        }

        /// <summary>
        ///     <see cref="find_common_type(NPTypeCode[], NPTypeCode[])"/> of dtype strings that all describe arrays (no scalar
        ///     types) — a NumSharp shorthand: NumPy's function always takes the two lists (<c>np.find_common_type("i4", "f8")</c>
        ///     is float64).
        /// </summary>
        /// <param name="involvedTypes">The dtype strings (<c>"int32"</c>, <c>"f8"</c>, <c>"&lt;u2"</c>).</param>
        /// <returns>The first type of NumPy's candidate list every entry casts to safely, the entry itself for one, null for none.</returns>
        /// <inheritdoc cref="find_common_type(NPTypeCode[], NPTypeCode[])" path="/remarks"/>
        /// <exception cref="NotSupportedException">A string names a dtype NumSharp cannot store (NumPy's texts).</exception>
        [MethodImpl(Optimize)]
        public static DType find_common_type(params string[] involvedTypes)
        {
            return _can_coerce_all(involvedTypes.Select(s => dtype(s).GetTypeCode()).ToArray());
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(NDArray firstNDArray, NDArray secondNDArray)
        {
            var lscalar = firstNDArray.Shape.IsScalar;
            var rscalar = secondNDArray.Shape.IsScalar;
            if (!lscalar && !rscalar)
                return _FindCommonArrayType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);

            if (lscalar && rscalar)
                return _FindCommonScalarType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);

            if (lscalar)
                return _FindCommonArrayScalarType(secondNDArray.GetTypeCode, firstNDArray.GetTypeCode);

            //rscalar is true
            return _FindCommonArrayScalarType(firstNDArray.GetTypeCode, secondNDArray.GetTypeCode);
        }

        #region Private of find_common_type

        /// <summary>NumPy 1.26.4's <c>_can_coerce_all(dtypelist)</c>: the search of <see cref="_can_coerce_all(NPTypeCode[], int)"/> from the first candidate.</summary>
        /// <param name="dtypelist">The dtypes.</param>
        /// <returns>The first candidate every entry casts to safely, the entry itself for one, <see cref="NPTypeCode.Empty"/> for none.</returns>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist)
        {
            return _can_coerce_all(dtypelist, 0);
        }

        /// <summary>The <see cref="List{T}"/> form of <see cref="_can_coerce_all(NPTypeCode[])"/>.</summary>
        /// <param name="dtypelist">The dtypes.</param>
        /// <returns>The first candidate every entry casts to safely, the entry itself for one, <see cref="NPTypeCode.Empty"/> for none.</returns>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist)
        {
            return _can_coerce_all(dtypelist.ToArray(), 0);
        }

        /// <summary>
        ///     NumPy 1.26.4's <c>_can_coerce_all(dtypelist, start)</c>: walk <see cref="__test_types"/> from index
        ///     <paramref name="start"/> and return the first type every entry of <paramref name="dtypelist"/> casts to safely
        ///     (NumPy's <c>newdtype &gt;= x</c>, a safe-cast test, never the enum's numeric order).
        /// </summary>
        /// <param name="dtypelist">The dtypes.</param>
        /// <param name="start">The candidate to start from: an index into <see cref="__test_types"/>, NOT into <paramref name="dtypelist"/>.</param>
        /// <returns>
        ///     <see cref="NPTypeCode.Empty"/> for no entries (NumPy's None); the entry itself for one; else the first candidate
        ///     holding them all, or <see cref="NPTypeCode.Empty"/> when none does (NumPy's object, which NumSharp lacks).
        /// </returns>
        /// <remarks>
        ///     A list holding one of NumSharp's own dtypes (decimal, char) leaves NumPy's candidates: NumPy has no such dtype,
        ///     so it would answer None, and NumSharp promotes the list with its table (<see cref="_promote_sequence"/>), the rule
        ///     its arithmetic uses. The fold this replaces read the table for every list and dropped every other pair from four
        ///     dtypes on (<c>[int8, int8, float64, int8]</c> came back int8).
        /// </remarks>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="start"/> is negative.</exception>
        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist, int start)
        {
            int N = dtypelist.Length;
            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            if (start < 0)
                throw new ArgumentOutOfRangeException(nameof(start), start, "A candidate index into __test_types.");

            // The descriptors are compared once per candidate; resolve them once.
            var descrs = new DType[N];
            for (int i = 0; i < N; i++)
            {
                descrs[i] = DType.From(dtypelist[i]);
                if (IsNumSharpOnly(descrs[i]))
                    return _promote_sequence(dtypelist);
            }

            var candidates = FindCommonTypeCandidates.Codes;
            for (int thisind = start; thisind < __len_test_types; thisind++)
            {
                var candidate = candidates[thisind];
                if (candidate == NPTypeCode.Empty)
                    continue; // complex64 or object: NumSharp has neither (see FindCommonTypeCandidates)
                var newdtype = DType.From(candidate);
                int numcoerce = 0;
                for (int i = 0; i < N; i++)
                    if (newdtype >= descrs[i])
                        numcoerce++;
                if (numcoerce == N)
                    return candidate;
            }

            return NPTypeCode.Empty;
        }

        /// <summary>The <see cref="List{T}"/> form of <see cref="_can_coerce_all(NPTypeCode[], int)"/>.</summary>
        /// <param name="dtypelist">The dtypes.</param>
        /// <param name="start">The candidate index into <see cref="__test_types"/> to start from.</param>
        /// <returns>The first candidate every entry casts to safely, the entry itself for one, <see cref="NPTypeCode.Empty"/> for none.</returns>
        /// <exception cref="ArgumentException">An entry is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="start"/> is negative.</exception>
        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist, int start)
        {
            return _can_coerce_all(dtypelist.ToArray(), start);
        }

        /// <summary>
        ///     NumPy 1.26.4's <c>_find_common_coerce(a, b)</c>: <paramref name="a"/> itself when <paramref name="b"/> casts to
        ///     it safely, else the first candidate both cast to, searched from <paramref name="a"/>'s place in
        ///     <see cref="__test_types"/>. <see cref="_FindCommonType(NPTypeCode[], NPTypeCode[])"/> calls it with the scalar
        ///     maximum as <paramref name="a"/> when that is of a higher kind than the array maximum <paramref name="b"/>.
        /// </summary>
        /// <param name="a">The higher-kind maximum (the scalar types').</param>
        /// <param name="b">The other maximum (the array types').</param>
        /// <returns>The common storage code, or <see cref="NPTypeCode.Empty"/> where NumPy returns None.</returns>
        /// <remarks>
        /// <para>
        ///     NumPy's <c>a &gt; b</c> is the dtype comparison: <paramref name="b"/> casts to <paramref name="a"/> safely and is
        ///     not equivalent to it. It used to compare the ENUM values, so (int8 scalar, uint8 array) answered uint8 where NumPy
        ///     answers int16, and (int16, uint16) crashed: the search then sliced the dtype list at the candidate index.
        /// </para>
        /// <para>
        ///     <paramref name="a"/>'s place comes from its NumPy character (<see cref="NPTypeCodeExtensions.ToTYPECHAR"/>,
        ///     platform-resolved: int32 is <c>'l'</c> on Windows, <c>'i'</c> elsewhere; both start at an equivalent type). NumSharp's
        ///     own dtypes have no character (the old mapping read decimal as <c>'q'</c>, so it searched from int64), and are
        ///     promoted with NumSharp's table instead, as in <see cref="_can_coerce_all(NPTypeCode[], int)"/>.
        /// </para>
        /// </remarks>
        private static NPTypeCode _find_common_coerce(NPTypeCode a, NPTypeCode b)
        {
            // find_common_type never passes None (it returns the other maximum first); a direct caller gets the present side.
            if (a == NPTypeCode.Empty)
                return b;
            if (b == NPTypeCode.Empty)
                return a;

            DType da = DType.From(a), db = DType.From(b);
            if (da > db)
                return a;

            if (IsNumSharpOnly(da) || IsNumSharpOnly(db))
                return _promote_sequence(new[] { b, a });

            // Python's str.index raises ValueError for a character outside the list (str, for one) and NumPy returns None;
            // C#'s IndexOf answers -1 instead, which must not become a start index.
            int thisind = __test_types.IndexOf((char)a.ToTYPECHAR());
            if (thisind < 0)
                return NPTypeCode.Empty;

            return _can_coerce_all(new[] { a, b }, thisind);
        }

        /// <summary>
        ///     Whether <paramref name="descr"/> is one of NumSharp's own dtypes (decimal, char): NumSharp numbers them in NumPy's
        ///     user-defined range (<see cref="NPY_TYPES.NPY_USERDEF"/> and up), so no NumPy character or candidate names them.
        /// </summary>
        /// <param name="descr">The descriptor.</param>
        /// <returns>True for a NumSharp-only dtype.</returns>
        private static bool IsNumSharpOnly(DType descr) => descr.num >= (int)NPY_TYPES.NPY_USERDEF;

        /// <summary>
        ///     NumPy's sequential promotion (<c>handle_promotion</c>): the running dtype promoted with each next one through
        ///     the array-array table, left to right. Order-sensitive exactly where NumPy's is.
        /// </summary>
        /// <param name="types">The dtypes, in order.</param>
        /// <returns>The promoted storage code, or <see cref="NPTypeCode.Empty"/> for no types.</returns>
        /// <exception cref="System.Collections.Generic.KeyNotFoundException">A code the promotion table has no row for (<see cref="NPTypeCode.Empty"/>).</exception>
        private static NPTypeCode _promote_sequence(NPTypeCode[] types)
        {
            if (types.Length == 0)
                return NPTypeCode.Empty;
            var ret = types[0];
            // NumPy calls PyArray_PromoteTypes(descr, *out_descr): the new dtype first, the running one second.
            for (int i = 1; i < types.Length; i++)
                ret = _nptypemap_arr_arr[(types[i], ret)];
            return ret;
        }

        #endregion
    }
}
