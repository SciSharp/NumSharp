using System;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Compact identity of a production inner-loop kernel (the Tier-3B kernels the ufunc
    ///     routes drive through <c>NDIterRef.ExecuteElementWise*</c>): the ufunc family, its op
    ///     and up to three dtypes packed into one 64-bit value.
    /// </summary>
    /// <remarks>
    ///     The routes used to identify their kernel with an interpolated string —
    ///     <c>$"npy_binop_{op}_{lhs}_{rhs}_{res}"</c> — rebuilt on EVERY call: four enum formats
    ///     and a string allocation, measured 45 ns + 96 B of garbage per call against ~10 ns for a
    ///     dictionary probe on a prebuilt key. On a small array that string was a larger share of
    ///     the call than the iterator's own construction. This key is the allocation-free
    ///     replacement; <see cref="ToCacheKey"/> reproduces the exact string each route used to
    ///     build, so the string-keyed cache and the generated DynamicMethod names are unchanged —
    ///     the struct cache is purely a front cache over them.
    /// </remarks>
    public readonly record struct InnerLoopKernelKey(ulong Packed)
    {
        /// <summary>The ufunc family a key belongs to (selects the string format and op enum).</summary>
        public enum KernelFamily : byte
        {
            /// <summary>Binary arithmetic ufunc: <c>npy_binop_{op}_{lhs}_{rhs}_{result}</c>.</summary>
            Binary = 1,
            /// <summary>Comparison ufunc: <c>npy_cmp_{op}_{lhs}_{rhs}</c>.</summary>
            Comparison = 2,
            /// <summary>Unary ufunc: <c>npy_unop_{op}_{input}_{output}</c>.</summary>
            Unary = 3,
            /// <summary>Left shift: <c>npy_shift_L_{value}_{result}</c>.</summary>
            ShiftLeft = 4,
            /// <summary>Right shift: <c>npy_shift_R_{value}_{result}</c>.</summary>
            ShiftRight = 5,
        }

        // Layout (low → high): t2:8 | t1:8 | t0:8 | (unused):8 | op:8 | family:8.
        // Every NPTypeCode (max Complex = 128) and every kernel op enum fits in a byte.
        private const int T2Shift = 0, T1Shift = 8, T0Shift = 16, OpShift = 32, FamilyShift = 40;

        private static ulong Pack(KernelFamily family, int op, NPTypeCode t0, NPTypeCode t1, NPTypeCode t2)
            => ((ulong)(byte)family << FamilyShift)
             | ((ulong)(byte)op << OpShift)
             | ((ulong)(byte)t0 << T0Shift)
             | ((ulong)(byte)t1 << T1Shift)
             | ((ulong)(byte)t2 << T2Shift);

        /// <summary>The family this key was built for.</summary>
        public KernelFamily Family => (KernelFamily)(byte)(Packed >> FamilyShift);

        /// <summary>The raw op ordinal (interpret through <see cref="Family"/>).</summary>
        public int Op => (byte)(Packed >> OpShift);

        /// <summary>First dtype (lhs / input / shift value).</summary>
        public NPTypeCode Type0 => (NPTypeCode)(byte)(Packed >> T0Shift);

        /// <summary>Second dtype (rhs / output / shift result).</summary>
        public NPTypeCode Type1 => (NPTypeCode)(byte)(Packed >> T1Shift);

        /// <summary>Third dtype (binary result; <see cref="NPTypeCode.Empty"/> otherwise).</summary>
        public NPTypeCode Type2 => (NPTypeCode)(byte)(Packed >> T2Shift);

        /// <summary>Key of the binary-arithmetic kernel <c>npy_binop_{op}_{lhs}_{rhs}_{result}</c>.</summary>
        public static InnerLoopKernelKey Binary(BinaryOp op, NPTypeCode lhs, NPTypeCode rhs, NPTypeCode result)
            => new(Pack(KernelFamily.Binary, (int)op, lhs, rhs, result));

        /// <summary>Key of the comparison kernel <c>npy_cmp_{op}_{lhs}_{rhs}</c> (bool output).</summary>
        public static InnerLoopKernelKey Comparison(ComparisonOp op, NPTypeCode lhs, NPTypeCode rhs)
            => new(Pack(KernelFamily.Comparison, (int)op, lhs, rhs, NPTypeCode.Empty));

        /// <summary>Key of the unary kernel <c>npy_unop_{op}_{input}_{output}</c>.</summary>
        public static InnerLoopKernelKey Unary(UnaryOp op, NPTypeCode input, NPTypeCode output)
            => new(Pack(KernelFamily.Unary, (int)op, input, output, NPTypeCode.Empty));

        /// <summary>Key of the shift kernel <c>npy_shift_{L|R}_{value}_{result}</c>.</summary>
        public static InnerLoopKernelKey Shift(bool isLeftShift, NPTypeCode valueLoopType, NPTypeCode result)
            => new(Pack(isLeftShift ? KernelFamily.ShiftLeft : KernelFamily.ShiftRight, 0, valueLoopType, result, NPTypeCode.Empty));

        /// <summary>
        ///     The string key this kernel is registered under in the string-keyed inner-loop
        ///     cache — byte-identical to the interpolated key the route built before the packed
        ///     key existed. Only evaluated on a cache miss.
        /// </summary>
        public string ToCacheKey()
        {
            switch (Family)
            {
                case KernelFamily.Binary:
                    return $"npy_binop_{(BinaryOp)Op}_{Type0}_{Type1}_{Type2}";
                case KernelFamily.Comparison:
                    return $"npy_cmp_{(ComparisonOp)Op}_{Type0}_{Type1}";
                case KernelFamily.Unary:
                    return $"npy_unop_{(UnaryOp)Op}_{Type0}_{Type1}";
                case KernelFamily.ShiftLeft:
                    return $"npy_shift_L_{Type0}_{Type1}";
                case KernelFamily.ShiftRight:
                    return $"npy_shift_R_{Type0}_{Type1}";
                default:
                    throw new InvalidOperationException($"Unknown inner-loop kernel family {(byte)Family}.");
            }
        }

        /// <inheritdoc/>
        public override string ToString() => ToCacheKey();
    }
}
