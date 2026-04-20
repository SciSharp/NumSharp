using System;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Edge-case coverage for the three-tier custom-op API:
    ///   • Size boundaries (empty / 1 / VC / unroll / unroll±N / large)
    ///   • Non-contiguous layouts (slice, transpose, reverse)
    ///   • Broadcast inputs (stride 0)
    ///   • All 12 dtypes including SIMD-forbidden (Boolean, Char, Decimal)
    ///   • Mixed-type promotion (scalar path only)
    ///   • NpyExpr corners (deep nesting, input reuse, constant-only)
    ///   • Cache behavior + argument validation
    /// </summary>
    [TestClass]
    public unsafe class NpyIterCustomOpEdgeCaseTests
    {
        // =====================================================================
        // Common helpers
        // =====================================================================

        private static NpyIterRef Iter(NDArray input, NDArray output)
            => NpyIterRef.MultiNew(2, new[] { input, output },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

        private static NpyIterRef Iter(NDArray a, NDArray b, NDArray c)
            => NpyIterRef.MultiNew(3, new[] { a, b, c },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.WRITEONLY });

        private static int VectorCountFloat32()
        {
            // Matches ILKernelGenerator.GetVectorCount(NPTypeCode.Single).
            int bits = Vector512.IsHardwareAccelerated ? 512 :
                       Vector256.IsHardwareAccelerated ? 256 :
                       Vector128.IsHardwareAccelerated ? 128 : 32;
            return bits / 8 / 4;
        }

        // =====================================================================
        // Size-boundary: all via Tier C: out = 2*in + 1
        // =====================================================================

        private static void RunLinear(int count)
        {
            var input = count == 0
                ? np.empty(new Shape(0), np.float32)
                : np.arange(count).astype(np.float32);
            var output = np.empty(new Shape(count), np.float32);

            using var iter = Iter(input, output);
            var expr = NpyExpr.Input(0) * NpyExpr.Const(2.0f) + NpyExpr.Const(1.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_linear_f32_v1");

            for (int i = 0; i < count; i++)
                Assert.AreEqual(2f * i + 1f, output.GetSingle(i), 1e-5f, $"[{count}] i={i}");
        }

        [TestMethod] public void Size_0_Empty() => RunLinear(0);
        [TestMethod] public void Size_1_ScalarTailOnly() => RunLinear(1);
        [TestMethod] public void Size_3_BelowVector() => RunLinear(3);
        [TestMethod] public void Size_OneVector() => RunLinear(VectorCountFloat32());
        [TestMethod] public void Size_OneVectorPlus1() => RunLinear(VectorCountFloat32() + 1);
        [TestMethod] public void Size_OneVectorMinus1() => RunLinear(VectorCountFloat32() - 1);
        [TestMethod] public void Size_TwoVectors() => RunLinear(VectorCountFloat32() * 2);
        [TestMethod] public void Size_ThreeVectors() => RunLinear(VectorCountFloat32() * 3);
        [TestMethod] public void Size_ExactlyUnroll() => RunLinear(VectorCountFloat32() * 4);
        [TestMethod] public void Size_UnrollPlus1() => RunLinear(VectorCountFloat32() * 4 + 1);
        [TestMethod] public void Size_UnrollPlus7() => RunLinear(VectorCountFloat32() * 4 + 7);
        [TestMethod] public void Size_TenUnrollsPlusTail() => RunLinear(VectorCountFloat32() * 40 + 3);
        [TestMethod] public void Size_1M() => RunLinear(1_000_000);

        // =====================================================================
        // Non-contiguous: slice, transpose, reverse
        // =====================================================================

        [TestMethod]
        public void Strided_EveryOther_ScalarFallback()
        {
            var big = np.arange(64).astype(np.float32);
            var sliced = big["::2"];                         // 32 elements, stride 2
            var output = np.empty(new Shape(32), np.float32);

            using var iter = Iter(sliced, output);
            var expr = NpyExpr.Input(0) * NpyExpr.Input(0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_square_f32_strided");

            for (int i = 0; i < 32; i++)
            {
                float src = 2f * i;
                Assert.AreEqual(src * src, output.GetSingle(i), 1e-5f);
            }
        }

        [TestMethod]
        public void Strided_EveryFourth()
        {
            var big = np.arange(80).astype(np.float32);
            var sliced = big["::4"];                         // 20 elements, stride 4
            var output = np.empty(new Shape(20), np.float32);

            using var iter = Iter(sliced, output);
            iter.ExecuteElementWiseUnary(
                NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    il.Emit(OpCodes.Ldc_R4, 3.0f);
                    il.Emit(OpCodes.Add);
                },
                vectorBody: il =>
                {
                    il.Emit(OpCodes.Ldc_R4, 3.0f);
                    ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
                },
                cacheKey: "edge_add3_f32");

            for (int i = 0; i < 20; i++)
                Assert.AreEqual(4f * i + 3f, output.GetSingle(i), 1e-5f);
        }

        [TestMethod]
        public void Transposed_2D_TriggersGeneralPath()
        {
            // 4×3 transposed → 3×4 view with stride [1,3]. Inner stride=3, not 1.
            // Kernel's runtime contig check fails → strided fallback.
            var a = np.arange(12).astype(np.float32).reshape(4, 3);
            var t = a.T;                                     // shape (3,4), strides (1,3)*4
            var output = np.empty(new Shape(3, 4), np.float32);

            using var iter = Iter(t, output);
            iter.ExecuteElementWiseUnary(
                NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    il.Emit(OpCodes.Ldc_R4, 10.0f);
                    il.Emit(OpCodes.Add);
                },
                vectorBody: null,               // force scalar-only
                cacheKey: "edge_add10_f32_noSimd");

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                {
                    float expected = a.GetSingle(j, i) + 10f;
                    Assert.AreEqual(expected, output.GetSingle(i, j), 1e-5f, $"[{i},{j}]");
                }
        }

        [TestMethod]
        public void Broadcast_StrideZero_Input()
        {
            // A 0-d scalar broadcast to shape (8,) — stride 0 on the input.
            var scalar = np.full(new Shape(), 7.0f, NPTypeCode.Single);
            var output = np.empty(new Shape(8), np.float32);

            using var iter = NpyIterRef.AdvancedNew(
                nop: 2,
                op: new[] { scalar, output },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY },
                opDtypes: null, opAxesNDim: -1, opAxes: null,
                iterShape: new long[] { 8 });

            var expr = NpyExpr.Input(0) * NpyExpr.Const(3.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_broadcast_scalar_x3");

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(21f, output.GetSingle(i), 1e-5f);
        }

        // =====================================================================
        // All SIMD-capable dtypes
        // =====================================================================

        [TestMethod]
        public void Dtype_Byte_Add()
        {
            var a = np.arange(16).astype(np.uint8);
            var b = np.full(new Shape(16), (byte)5, NPTypeCode.Byte);
            var c = np.empty(new Shape(16), np.uint8);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Byte, NPTypeCode.Byte, NPTypeCode.Byte,
                scalarBody: il => il.Emit(OpCodes.Add),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Byte),
                cacheKey: "edge_byte_add");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual((byte)(i + 5), c.GetByte(i));
        }

        [TestMethod]
        public void Dtype_Int16_Subtract()
        {
            var a = np.arange(20).astype(np.int16);
            var b = np.full(new Shape(20), (short)10, NPTypeCode.Int16);
            var c = np.empty(new Shape(20), np.int16);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Int16, NPTypeCode.Int16, NPTypeCode.Int16,
                scalarBody: il => il.Emit(OpCodes.Sub),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Subtract, NPTypeCode.Int16),
                cacheKey: "edge_i16_sub");

            for (int i = 0; i < 20; i++)
                Assert.AreEqual((short)(i - 10), c.GetInt16(i));
        }

        [TestMethod]
        public void Dtype_UInt32_BitwiseAnd()
        {
            var a = np.arange(16).astype(np.uint32);
            var b = np.full(new Shape(16), (uint)0x0F, NPTypeCode.UInt32);
            var c = np.empty(new Shape(16), np.uint32);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.UInt32, NPTypeCode.UInt32, NPTypeCode.UInt32,
                scalarBody: il => il.Emit(OpCodes.And),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.BitwiseAnd, NPTypeCode.UInt32),
                cacheKey: "edge_u32_and");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual((uint)(i & 0x0F), c.GetUInt32(i));
        }

        [TestMethod]
        public void Dtype_Int64_Multiply()
        {
            var a = np.arange(12).astype(np.int64);
            var b = np.arange(12, 24).astype(np.int64);
            var c = np.empty(new Shape(12), np.int64);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Int64, NPTypeCode.Int64, NPTypeCode.Int64,
                scalarBody: il => il.Emit(OpCodes.Mul),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Int64),
                cacheKey: "edge_i64_mul");

            for (int i = 0; i < 12; i++)
                Assert.AreEqual((long)i * (long)(i + 12), c.GetInt64(i));
        }

        [TestMethod]
        public void Dtype_Double_Divide()
        {
            var a = np.arange(1, 17).astype(np.float64);
            var b = np.full(new Shape(16), 2.0, NPTypeCode.Double);
            var c = np.empty(new Shape(16), np.float64);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double,
                scalarBody: il => il.Emit(OpCodes.Div),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Divide, NPTypeCode.Double),
                cacheKey: "edge_f64_div");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual((i + 1) / 2.0, c.GetDouble(i), 1e-9);
        }

        // =====================================================================
        // SIMD-forbidden dtypes (Boolean, Char, Decimal)
        // =====================================================================

        [TestMethod]
        public void Dtype_Boolean_ScalarOnly_LogicalAnd()
        {
            // bool AND via BitwiseAnd (since bool is 1-byte, & works as logical-and).
            var a = np.array(new bool[] { true, false, true, true, false, true });
            var b = np.array(new bool[] { true, true, false, true, true, false });
            var c = np.empty(new Shape(6), np.@bool);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Boolean, NPTypeCode.Boolean, NPTypeCode.Boolean,
                scalarBody: il => il.Emit(OpCodes.And),
                vectorBody: null,        // Boolean is not SIMD-capable
                cacheKey: "edge_bool_and");

            bool[] expected = { true, false, false, true, false, false };
            for (int i = 0; i < 6; i++)
                Assert.AreEqual(expected[i], c.GetBoolean(i));
        }

        [TestMethod]
        public void Dtype_Decimal_ScalarOnly_Add()
        {
            var a = np.array(new decimal[] { 1m, 2m, 3m, 4m, 5m });
            var b = np.full(new Shape(5), 10m, NPTypeCode.Decimal);
            var c = np.empty(new Shape(5), np.@decimal);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Decimal, NPTypeCode.Decimal, NPTypeCode.Decimal,
                scalarBody: il => ILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Decimal),
                vectorBody: null,        // Decimal is not SIMD-capable
                cacheKey: "edge_decimal_add");

            for (int i = 0; i < 5; i++)
                Assert.AreEqual((decimal)(i + 1 + 10), c.GetDecimal(i));
        }

        // =====================================================================
        // Mixed-type promotion: int32 + float32 → float32 via scalar path
        // =====================================================================

        [TestMethod]
        public void MixedType_Int32PlusFloat32_ReturnsFloat32()
        {
            var a = np.arange(16).astype(np.int32);
            var b = np.arange(16, 32).astype(np.float32);
            var c = np.empty(new Shape(16), np.float32);

            using var iter = Iter(a, b, c);
            // All-same-type SIMD gating fails → only scalar path.
            // Scalar body must convert both operands to float before adding.
            iter.ExecuteElementWise(
                new[] { NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Single },
                scalarBody: il =>
                {
                    // Stack: [int_a, float_b]
                    var locB = il.DeclareLocal(typeof(float));
                    il.Emit(OpCodes.Stloc, locB);               // Stack: [int_a]
                    il.Emit(OpCodes.Conv_R4);                    // Stack: [float_a]
                    il.Emit(OpCodes.Ldloc, locB);                // Stack: [float_a, float_b]
                    il.Emit(OpCodes.Add);
                },
                vectorBody: null,
                cacheKey: "edge_mixed_i32_f32_add");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual((float)i + (float)(i + 16), c.GetSingle(i), 1e-5f);
        }

        // =====================================================================
        // NpyExpr tree corners
        // =====================================================================

        [TestMethod]
        public void NpyExpr_DeeplyNested_TenAdditions()
        {
            // ((((((((((a+1)+2)+3)+4)+5)+6)+7)+8)+9)+10) = a + 55
            var a = np.arange(16).astype(np.float32);
            var b = np.empty(new Shape(16), np.float32);

            using var iter = Iter(a, b);

            NpyExpr e = NpyExpr.Input(0);
            for (int k = 1; k <= 10; k++)
                e = e + NpyExpr.Const((float)k);

            iter.ExecuteExpression(e, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(i + 55f, b.GetSingle(i), 1e-4f);
        }

        [TestMethod]
        public void NpyExpr_InputReusedThreeTimes()
        {
            // a*a + a = a² + a
            var a = np.arange(16).astype(np.float32);
            var b = np.empty(new Shape(16), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Input(0) + NpyExpr.Input(0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_reuse_a2_plus_a");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(i * i + i, b.GetSingle(i), 1e-4f);
        }

        [TestMethod]
        public void NpyExpr_ConstantOnly_IgnoresInput()
        {
            // Output = 42; input is still iterated but unused.
            var a = np.arange(8).astype(np.float32);
            var b = np.empty(new Shape(8), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteExpression(NpyExpr.Const(42.0f),
                new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_const_only_42");

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(42f, b.GetSingle(i));
        }

        [TestMethod]
        public void NpyExpr_NegativeConstant()
        {
            var a = np.arange(8).astype(np.float32);
            var b = np.empty(new Shape(8), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) + NpyExpr.Const(-100.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_const_neg100");

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(i - 100f, b.GetSingle(i), 1e-5f);
        }

        [TestMethod]
        public void NpyExpr_DivideByConstant()
        {
            var a = np.arange(1, 17).astype(np.float64);
            var b = np.empty(new Shape(16), np.float64);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) / NpyExpr.Const(4.0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double,
                cacheKey: "edge_div_4");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual((i + 1) / 4.0, b.GetDouble(i), 1e-9);
        }

        [TestMethod]
        public void NpyExpr_UnaryChain_AbsThenNegate()
        {
            var a = np.array(new float[] { -3, 4, -5, 6, -7, 8 });
            var b = np.empty(new Shape(6), np.float32);

            using var iter = Iter(a, b);
            var expr = -NpyExpr.Abs(NpyExpr.Input(0));                // -|a|
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_neg_abs");

            float[] expected = { -3, -4, -5, -6, -7, -8 };
            for (int i = 0; i < 6; i++)
                Assert.AreEqual(expected[i], b.GetSingle(i), 1e-5f);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void NpyExpr_InputIndexOutOfRange_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            // Iter has 1 input but expression references Input(5).
            var expr = NpyExpr.Input(5);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void NpyExpr_InputNegativeIndex_ThrowsOnConstruction()
        {
            NpyExpr.Input(-1);
        }

        // =====================================================================
        // Auto-derived cache key (Tier C) & cache behavior
        // =====================================================================

        [TestMethod]
        public void Cache_AutoDerivedKey_StructurallyEquivalentTreesShareDelegate()
        {
            // Clear cache so we can observe growth precisely.
            InvokeClearCache();
            int before = GetInnerLoopCacheCount();

            var a1 = np.arange(4).astype(np.float32);
            var b1 = np.empty(new Shape(4), np.float32);
            var a2 = np.arange(4).astype(np.float32);
            var b2 = np.empty(new Shape(4), np.float32);

            // Two structurally identical expressions built from distinct instances.
            var e1 = NpyExpr.Input(0) * NpyExpr.Const(5.0f);
            var e2 = NpyExpr.Input(0) * NpyExpr.Const(5.0f);

            using (var it1 = Iter(a1, b1))
                it1.ExecuteExpression(e1, new[] { NPTypeCode.Single }, NPTypeCode.Single);
            int afterFirst = GetInnerLoopCacheCount();

            using (var it2 = Iter(a2, b2))
                it2.ExecuteExpression(e2, new[] { NPTypeCode.Single }, NPTypeCode.Single);
            int afterSecond = GetInnerLoopCacheCount();

            Assert.AreEqual(before + 1, afterFirst, "First call should add 1 entry.");
            Assert.AreEqual(afterFirst, afterSecond,
                "Structurally equal trees should share the cached delegate.");
        }

        [TestMethod]
        public void Cache_DistinctStructure_DistinctEntries()
        {
            InvokeClearCache();
            int before = GetInnerLoopCacheCount();

            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            var e1 = NpyExpr.Input(0) * NpyExpr.Const(2.0f);
            var e2 = NpyExpr.Input(0) * NpyExpr.Const(3.0f);      // different constant
            var e3 = NpyExpr.Input(0) + NpyExpr.Const(2.0f);      // different op

            using (var it = Iter(a, b)) it.ExecuteExpression(e1, new[] { NPTypeCode.Single }, NPTypeCode.Single);
            using (var it = Iter(a, b)) it.ExecuteExpression(e2, new[] { NPTypeCode.Single }, NPTypeCode.Single);
            using (var it = Iter(a, b)) it.ExecuteExpression(e3, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            int after = GetInnerLoopCacheCount();
            Assert.AreEqual(before + 3, after, "Three distinct expressions should add three entries.");
        }

        [TestMethod]
        public void Cache_SameTreeDifferentInputTypes_DistinctEntries()
        {
            InvokeClearCache();
            int before = GetInnerLoopCacheCount();

            var af = np.arange(4).astype(np.float32);
            var ad = np.arange(4).astype(np.float64);
            var bf = np.empty(new Shape(4), np.float32);
            var bd = np.empty(new Shape(4), np.float64);

            var tree = NpyExpr.Input(0) + NpyExpr.Const(1.0);

            using (var it = Iter(af, bf))
                it.ExecuteExpression(tree, new[] { NPTypeCode.Single }, NPTypeCode.Single);
            using (var it = Iter(ad, bd))
                it.ExecuteExpression(tree, new[] { NPTypeCode.Double }, NPTypeCode.Double);

            int after = GetInnerLoopCacheCount();
            Assert.AreEqual(before + 2, after, "Same tree + different dtypes = different cache keys.");
        }

        // =====================================================================
        // Argument validation
        // =====================================================================

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validate_NullScalarBody_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteElementWise(
                new[] { NPTypeCode.Single, NPTypeCode.Single },
                scalarBody: null!,
                vectorBody: null,
                cacheKey: "edge_null_scalar");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validate_NullOperandTypes_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteElementWise(
                operandTypes: null!,
                scalarBody: il => il.Emit(OpCodes.Nop),
                vectorBody: null,
                cacheKey: "edge_null_ops");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Validate_OperandTypesTooShort_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteElementWise(
                new[] { NPTypeCode.Single },    // need >= 2 entries
                scalarBody: il => il.Emit(OpCodes.Nop),
                vectorBody: null,
                cacheKey: "edge_too_short");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validate_NullExpression_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteExpression(null!, new[] { NPTypeCode.Single }, NPTypeCode.Single);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validate_TierA_NullBody_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteRawIL(null!, "edge_null_raw_body");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Validate_TierA_NullKey_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            iter.ExecuteRawIL(il => il.Emit(OpCodes.Ret), null!);
        }

        // =====================================================================
        // Multi-dim coalescing
        // =====================================================================

        [TestMethod]
        public void MultiDim_Contiguous3D_CoalescesToSimd()
        {
            var a = np.arange(24).astype(np.float32).reshape(2, 3, 4);
            var b = np.empty(new Shape(2, 3, 4), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Const(2.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_3d_mul2");

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                    {
                        int idx = i * 12 + j * 4 + k;
                        Assert.AreEqual(2f * idx, b.GetSingle(i, j, k), 1e-5f);
                    }
        }

        // =====================================================================
        // Stress: pattern aggressively mixes unroll/remainder/tail
        // =====================================================================

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(5)]
        [DataRow(7)]
        [DataRow(8)]
        [DataRow(9)]
        [DataRow(15)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(31)]
        [DataRow(32)]
        [DataRow(33)]
        [DataRow(47)]
        [DataRow(63)]
        [DataRow(64)]
        [DataRow(65)]
        [DataRow(127)]
        [DataRow(255)]
        [DataRow(256)]
        [DataRow(257)]
        [DataRow(1023)]
        [DataRow(1024)]
        [DataRow(1025)]
        public void Stress_VariousSizes(int n)
        {
            var a = np.arange(n).astype(np.float32);
            var b = np.empty(new Shape(n), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Input(0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_stress_square");

            for (int i = 0; i < n; i++)
                Assert.AreEqual((float)i * i, b.GetSingle(i), 1e-4f, $"n={n}, i={i}");
        }

        // =====================================================================
        // Reverse-stride slicing
        // =====================================================================

        [TestMethod]
        public void ReverseStride_TriggersScalarFallback()
        {
            // [::-1] gives a view with negative stride. NpyIter flips these
            // internally under K-order (default); the kernel sees positive
            // strides but possibly with rebased pointers.
            var big = np.arange(16).astype(np.float32);
            var reversed = big["::-1"];                      // [15,14,...,0]
            var output = np.empty(new Shape(16), np.float32);

            using var iter = Iter(reversed, output);
            var expr = NpyExpr.Input(0) + NpyExpr.Const(100.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_rev_add100");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(reversed.GetSingle(i) + 100f, output.GetSingle(i), 1e-5f, $"i={i}");
        }

        // =====================================================================
        // Strided output path
        // =====================================================================

        [TestMethod]
        public void StridedOutput_ViewOfEveryOther()
        {
            // Output is a slice (::2) of a larger array — write stride = 2.
            // The kernel's contig check sees output stride != 4, takes the
            // scalar-strided path.
            var input = np.arange(8).astype(np.float32);
            var outBig = np.zeros(new Shape(16), np.float32);
            var outView = outBig["::2"];                     // 8 elements, stride 2

            using var iter = Iter(input, outView);
            var expr = NpyExpr.Input(0) * NpyExpr.Const(-1.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single,
                cacheKey: "edge_neg_stridedOut");

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(-(float)i, outView.GetSingle(i), 1e-5f);
            // Verify the untouched slots in outBig remain 0.
            for (int i = 1; i < 16; i += 2)
                Assert.AreEqual(0f, outBig.GetSingle(i), 1e-5f, $"outBig[{i}] should be untouched");
        }

        // =====================================================================
        // Multi-D with mixed contig + strided operands
        // =====================================================================

        [TestMethod]
        public void MixedContigAndStrided_ScalarFallback()
        {
            // Input a: contig 12 floats. Input b: transposed (non-contig).
            // Output c: contig. Mixed layout → contig check fails → scalar.
            var a = np.arange(12).astype(np.float32);
            var bMat = np.arange(12).astype(np.float32).reshape(3, 4);
            var bT = bMat.T.flatten();                       // [0,4,8,1,5,9,2,6,10,3,7,11]
            var c = np.empty(new Shape(12), np.float32);

            using var iter = Iter(a, bT, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il => il.Emit(OpCodes.Add),
                vectorBody: il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single),
                cacheKey: "edge_mixedlayout_add");

            float[] expectedB = { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
            for (int i = 0; i < 12; i++)
                Assert.AreEqual(i + expectedB[i], c.GetSingle(i), 1e-5f);
        }

        // =====================================================================
        // Integer Tier C
        // =====================================================================

        [TestMethod]
        public void NpyExpr_Int32_ArithmeticChain()
        {
            var a = np.arange(16).astype(np.int32);
            var b = np.empty(new Shape(16), np.int32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Const(3) + NpyExpr.Const(7);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Int32 }, NPTypeCode.Int32,
                cacheKey: "edge_i32_3x_plus_7");

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(i * 3 + 7, b.GetInt32(i));
        }

        [TestMethod]
        public void NpyExpr_Int16_OverflowWraps()
        {
            // Int16 max is 32767. 200 * 200 = 40000 wraps in int16.
            var a = np.full(new Shape(4), (short)200, NPTypeCode.Int16);
            var b = np.empty(new Shape(4), np.int16);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Input(0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Int16 }, NPTypeCode.Int16,
                cacheKey: "edge_i16_square_overflow");

            // C# `short * short` widens to int, so 200*200 = 40000. But when
            // stored as Int16 the value wraps. Vector.Multiply<short> wraps
            // directly. Either way the result is 40000 mod 65536 = 40000,
            // reinterpreted as signed = -25536.
            short expected = unchecked((short)40000);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected, b.GetInt16(i));
        }

        [TestMethod]
        public void NpyExpr_UpcastIntToFloat_ViaInputConversion()
        {
            // Input int32, output float32 — expression auto-converts via EmitConvertTo.
            var a = np.arange(8).astype(np.int32);
            var b = np.empty(new Shape(8), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) * NpyExpr.Const(0.5f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Int32 }, NPTypeCode.Single,
                cacheKey: "edge_i32toF32_half");

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(i * 0.5f, b.GetSingle(i), 1e-5f);
        }

        // =====================================================================
        // Expression: 30-level deep (stress local-slot allocation in DynamicMethod)
        // =====================================================================

        [TestMethod]
        public void NpyExpr_30LevelNested()
        {
            var a = np.arange(8).astype(np.float32);
            var b = np.empty(new Shape(8), np.float32);

            using var iter = Iter(a, b);

            NpyExpr e = NpyExpr.Input(0);
            for (int k = 1; k <= 30; k++) e = e + NpyExpr.Const(1.0f);   // adds 30

            iter.ExecuteExpression(e, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            for (int i = 0; i < 8; i++)
                Assert.AreEqual(i + 30f, b.GetSingle(i), 1e-4f);
        }

        // =====================================================================
        // Decimal (previously buggy due to NPTypeCode.SizeOf(Decimal)=32).
        // Now that's been fixed to 16, the scalar-strided decimal path works.
        // =====================================================================

        [TestMethod]
        public void Dtype_Decimal_Add_AfterSizeFix()
        {
            var a = np.array(new decimal[] { 1m, 2m, 3m, 4m, 5m });
            var b = np.full(new Shape(5), 10m, NPTypeCode.Decimal);
            var c = np.empty(new Shape(5), np.@decimal);

            using var iter = Iter(a, b, c);
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Decimal, NPTypeCode.Decimal, NPTypeCode.Decimal,
                scalarBody: il => ILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Decimal),
                vectorBody: null,
                cacheKey: "edge_decimal_add_postfix");

            for (int i = 0; i < 5; i++)
                Assert.AreEqual((decimal)(i + 1 + 10), c.GetDecimal(i));
        }

        // =====================================================================
        // Char dtype (SIMD-forbidden, 2-byte)
        // =====================================================================

        // (Dtype_Char_ScalarOnly skipped — NumSharp rejects 1-D char arrays
        //  with "Please use char with extra dimension". The custom-op API is
        //  fine with NPTypeCode.Char; the restriction is upstream.)

        // =====================================================================
        // NpyExpr: auto-derived cache key with default null argument
        // =====================================================================

        [TestMethod]
        public void NpyExpr_AutoKey_NullParam_ProducesValidDelegate()
        {
            // Calling without cacheKey param (so it's null) should use
            // the auto-derived structural key and NOT throw.
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = Iter(a, b);
            var expr = NpyExpr.Input(0) + NpyExpr.Const(1.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            for (int i = 0; i < 4; i++)
                Assert.AreEqual(i + 1f, b.GetSingle(i), 1e-5f);
        }

        // =====================================================================
        // Reflection helpers for internal cache count
        // =====================================================================

        private static PropertyInfo _cacheCountProp = typeof(ILKernelGenerator)
            .GetProperty("InnerLoopCachedCount", BindingFlags.Static | BindingFlags.NonPublic)!;

        private static MethodInfo _clearCacheMethod = typeof(ILKernelGenerator)
            .GetMethod("ClearInnerLoopCache", BindingFlags.Static | BindingFlags.NonPublic)!;

        private static int GetInnerLoopCacheCount() => (int)_cacheCountProp.GetValue(null)!;

        private static void InvokeClearCache() => _clearCacheMethod.Invoke(null, null);
    }
}
