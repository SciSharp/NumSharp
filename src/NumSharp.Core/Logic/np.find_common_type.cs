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

        const int __len_test_types = 19;
        const string __test_types = "?bBhHiIlLqQefdgFDGO";

        /// <summary>
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

#if __REGEN
	        %foreach all_dtypes%
                {NPTypeCode.#1, 10000 },
	        %
#else
#endif

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

            var typemap_arr_arr = new Dictionary<(Type, Type), Type>(180);
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
            typemap_arr_arr.Add((np.@bool, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.@bool, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.@bool, np.@char), np.@char);

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
            typemap_arr_arr.Add((np.uint8, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.uint8, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint8, np.@char), np.uint8);

            typemap_arr_arr.Add((np.@char, np.@char), np.@char);
            typemap_arr_arr.Add((np.@char, np.@bool), np.@char);
            typemap_arr_arr.Add((np.@char, np.uint8), np.uint8);
            typemap_arr_arr.Add((np.@char, np.int16), np.int16);
            typemap_arr_arr.Add((np.@char, np.uint16), np.uint16);
            typemap_arr_arr.Add((np.@char, np.int32), np.int32);
            typemap_arr_arr.Add((np.@char, np.uint32), np.uint32);
            typemap_arr_arr.Add((np.@char, np.int64), np.int64);
            typemap_arr_arr.Add((np.@char, np.uint64), np.uint64);
            typemap_arr_arr.Add((np.@char, np.float32), np.float32);
            typemap_arr_arr.Add((np.@char, np.float64), np.float64);
            typemap_arr_arr.Add((np.@char, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.@char, np.@decimal), np.@decimal);

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
            typemap_arr_arr.Add((np.int16, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.int16, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int16, np.@char), np.int16);

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
            typemap_arr_arr.Add((np.uint16, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.uint16, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint16, np.@char), np.uint16);

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
            typemap_arr_arr.Add((np.int32, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.int32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int32, np.@char), np.int32);

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
            typemap_arr_arr.Add((np.uint32, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.uint32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint32, np.@char), np.uint32);

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
            typemap_arr_arr.Add((np.int64, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.int64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.int64, np.@char), np.int64);

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
            typemap_arr_arr.Add((np.uint64, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.uint64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.uint64, np.@char), np.uint64);

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
            typemap_arr_arr.Add((np.float32, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.float32, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.float32, np.@char), np.float32);

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
            typemap_arr_arr.Add((np.float64, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.float64, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.float64, np.@char), np.float64);

            typemap_arr_arr.Add((np.complex64, np.@bool), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.uint8), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.int16), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.uint16), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.int32), np.complex128);
            typemap_arr_arr.Add((np.complex64, np.uint32), np.complex128);
            typemap_arr_arr.Add((np.complex64, np.int64), np.complex128);
            typemap_arr_arr.Add((np.complex64, np.uint64), np.complex128);
            typemap_arr_arr.Add((np.complex64, np.float32), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.float64), np.complex128);
            typemap_arr_arr.Add((np.complex64, np.complex64), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.@decimal), np.complex64);
            typemap_arr_arr.Add((np.complex64, np.@char), np.complex64);

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
            typemap_arr_arr.Add((np.@decimal, np.complex64), np.complex128);
            typemap_arr_arr.Add((np.@decimal, np.@decimal), np.@decimal);
            typemap_arr_arr.Add((np.@decimal, np.@char), np.@decimal);

            _typemap_arr_arr = typemap_arr_arr.ToFrozenDictionary();

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

            var typemap_arr_scalar = new Dictionary<(Type, Type), Type>();
            typemap_arr_scalar.Add((np.@bool, np.@bool), np.@bool);
            typemap_arr_scalar.Add((np.@bool, np.uint8), np.uint8);
            typemap_arr_scalar.Add((np.@bool, np.int16), np.int16);
            typemap_arr_scalar.Add((np.@bool, np.uint16), np.uint16);
            typemap_arr_scalar.Add((np.@bool, np.int32), np.int32);
            typemap_arr_scalar.Add((np.@bool, np.uint32), np.uint32);
            typemap_arr_scalar.Add((np.@bool, np.int64), np.int64);
            typemap_arr_scalar.Add((np.@bool, np.uint64), np.uint64);
            typemap_arr_scalar.Add((np.@bool, np.float32), np.float32);
            typemap_arr_scalar.Add((np.@bool, np.float64), np.float64);
            typemap_arr_scalar.Add((np.@bool, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.uint8, np.complex64), np.complex64);

            typemap_arr_scalar.Add((np.@char, np.@char), np.@char);
            typemap_arr_scalar.Add((np.@char, np.@bool), np.@char);
            typemap_arr_scalar.Add((np.@char, np.uint8), np.@char);
            typemap_arr_scalar.Add((np.@char, np.int16), np.int16);
            typemap_arr_scalar.Add((np.@char, np.uint16), np.uint16);
            typemap_arr_scalar.Add((np.@char, np.int32), np.int32);
            typemap_arr_scalar.Add((np.@char, np.uint32), np.uint32);
            typemap_arr_scalar.Add((np.@char, np.int64), np.int64);
            typemap_arr_scalar.Add((np.@char, np.uint64), np.uint64);
            typemap_arr_scalar.Add((np.@char, np.float32), np.float32);
            typemap_arr_scalar.Add((np.@char, np.float64), np.float64);
            typemap_arr_scalar.Add((np.@char, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.int16, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.uint16, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.int32, np.complex64), np.complex128);

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
            typemap_arr_scalar.Add((np.uint32, np.complex64), np.complex128);

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
            typemap_arr_scalar.Add((np.int64, np.complex64), np.complex128);

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
            typemap_arr_scalar.Add((np.uint64, np.complex64), np.complex128);

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
            typemap_arr_scalar.Add((np.float32, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.float64, np.complex64), np.complex128);

            typemap_arr_scalar.Add((np.complex64, np.@bool), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.uint8), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.@char), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.int16), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.uint16), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.int32), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.uint32), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.int64), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.uint64), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.float32), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.float64), np.complex64);
            typemap_arr_scalar.Add((np.complex64, np.complex64), np.complex64);

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
            typemap_arr_scalar.Add((np.@decimal, np.complex64), np.complex128);
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
            typemap_arr_scalar.Add((np.complex64, np.@decimal), np.complex128);

            _typemap_arr_scalar = typemap_arr_scalar.ToFrozenDictionary();

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
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        #region Overloads

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(string[] array_types, string[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => np.dtype(v).typecode).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => np.dtype(v).typecode).ToArray() ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(Type[] array_types, NPTypeCode[] scalar_types)
        {
            return _FindCommonType(array_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>(), scalar_types ?? Array.Empty<NPTypeCode>());
        }

        /// <summary>
        ///     Determine common type following standard coercion rules.
        /// </summary>
        /// <param name="array_types">A list of dtypes or dtype convertible objects representing arrays. Can be null.</param>
        /// <param name="scalar_types">A list of dtypes or dtype convertible objects representing scalars.Can be null.</param>
        /// <returns>The common data type, which is the maximum of array_types ignoring scalar_types, unless the maximum of scalar_types is of a different kind (dtype.kind). If the kind is not understood, then None is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html</remarks>
        public static NPTypeCode find_common_type(NPTypeCode[] array_types, Type[] scalar_types)
        {
            return _FindCommonType(array_types ?? Array.Empty<NPTypeCode>(), scalar_types?.Select(v => v.GetTypeCode()).ToArray() ?? Array.Empty<NPTypeCode>());
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

        internal static NPTypeCode _FindCommonScalarType(Type dtype_left, Type dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left.GetTypeCode(), dtype_right.GetTypeCode()});
        }

        internal static NPTypeCode _FindCommonScalarType(NPTypeCode dtype_left, NPTypeCode dtype_right)
        {
            return _FindCommonType(Array.Empty<NPTypeCode>(), new NPTypeCode[] {dtype_left, dtype_right});
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
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(NPTypeCode[] array_types, NPTypeCode[] scalar_types)
        {
            NPTypeCode maxa = _can_coerce_all(array_types);
            NPTypeCode maxsc = _can_coerce_all(scalar_types);

            if (maxa == NPTypeCode.Empty)
                return maxsc;

            if (maxsc == NPTypeCode.Empty)
                return maxa;

            int index_a;
            int index_sc;
            try
            {
                index_a = maxa != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxa]) : -1;
                index_sc = maxsc != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxsc]) : -1;
            }
            catch (Exception)
            {
                return NPTypeCode.Empty;
            }

            if (index_sc > index_a)
                return _nptypemap_arr_scalar[(maxa, maxsc)];
            else
                return maxa;
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType(List<NPTypeCode> array_types, List<NPTypeCode> scalar_types)
        {
            NPTypeCode maxa = _can_coerce_all(array_types);
            NPTypeCode maxsc = _can_coerce_all(scalar_types);

            if (maxa == NPTypeCode.Empty)
                return maxsc;

            if (maxsc == NPTypeCode.Empty)
                return maxa;

            int index_a;
            int index_sc;
            try
            {
                index_a = maxa != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxa]) : -1;
                index_sc = maxsc != NPTypeCode.Empty ? Array.IndexOf(_kind_list, DType._kind_list_map[maxsc]) : -1;
            }
            catch (Exception)
            {
                return NPTypeCode.Empty;
            }

            if (index_sc > index_a)
                return _find_common_coerce(maxsc, maxa);
            else
                return maxa;
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType_Scalar(params NPTypeCode[] scalar_types)
        {
            return _can_coerce_all(scalar_types);
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
        [MethodImpl(Optimize)]
        internal static NPTypeCode _FindCommonType_Array(params NPTypeCode[] array_types)
        {
            return _can_coerce_all(array_types);
        }

        /// <summary>
        ///     Resolves to which type should the output be.
        /// </summary>
        /// <remarks>This function relys on <see cref="NPTypeCode"/> being ordered numerically by size.</remarks>
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
        ///     Resolves to which type should the output be.
        /// </summary>
        [MethodImpl(Optimize)]
        public static NPTypeCode find_common_type(params string[] involvedTypes)
        {
            return _can_coerce_all(involvedTypes.Select(s => dtype(s).typecode).ToArray());
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

        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist)
        {
            int N = dtypelist.Length;

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;
            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist)
        {
            int N = dtypelist.Count;

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;

            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(NPTypeCode[] dtypelist, int start)
        {
            int N = dtypelist.Length;
            if (start > 0)
            {
                var len = N - start;
                var sub = new NPTypeCode[len];
                Array.Copy(dtypelist, start, sub, len, len);
                dtypelist = sub;
                N = sub.Length;
            }

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;
            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        [MethodImpl(Optimize)]
        private static NPTypeCode _can_coerce_all(List<NPTypeCode> dtypelist, int start)
        {
            int N = dtypelist.Count;
            if (start > 0)
            {
                var len = N - start;
                var sub = new List<NPTypeCode>(len);
                for (int i = start; i < N; i++)
                    sub[i - start] = dtypelist[i];
                dtypelist = sub;
                N = sub.Count;
            }

            if (N == 0)
                return NPTypeCode.Empty;
            if (N == 1)
                return dtypelist[0];
            NPTypeCode ret = default;

            while (N >= 2)
            {
                ret = _nptypemap_arr_arr[(dtypelist[N - 1], dtypelist[N - 2])];
                N -= 2;
            }

            if (N == 1)
            {
                ret = _nptypemap_arr_arr[(ret, dtypelist[0])];
            }

            return ret;
        }

        // Keep incrementing until a common type both can be coerced to
        //  is found.  Otherwise, return None
        private static NPTypeCode _find_common_coerce(NPTypeCode a, NPTypeCode b)
        {
            if (a > b)
                return a;
            if (a == NPTypeCode.Empty)
                return b;
            if (b == NPTypeCode.Empty)
                return a;

            int thisind;
            try
            {
                thisind = __test_types.IndexOf((char)a.ToTYPECHAR());
            }
            catch
            {
                return NPTypeCode.Empty;
            }

            return _can_coerce_all(new NPTypeCode[] {a, b}, thisind);
        }

        #endregion
    }
}
