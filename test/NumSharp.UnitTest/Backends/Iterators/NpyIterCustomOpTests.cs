using System;
using System.Reflection.Emit;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Exercises the three-tier custom-op API on NpyIterRef:
    ///   Tier A — ExecuteRawIL (user emits entire inner-loop body)
    ///   Tier B — ExecuteElementWise (user supplies scalar + vector body emitters)
    ///   Tier C — ExecuteExpression (NpyExpr DSL compiled to inner-loop IL)
    /// </summary>
    [TestClass]
    public unsafe class NpyIterCustomOpTests
    {
        // =====================================================================
        // Tier A: Raw IL
        // =====================================================================

        [TestMethod]
        public void TierA_RawIL_AddsTwoInt32Arrays()
        {
            var a = np.arange(10).astype(np.int32);
            var b = np.arange(10, 20).astype(np.int32);
            var c = np.empty(new Shape(10), np.int32);

            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op: new[] { a, b, c },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            iter.ExecuteRawIL(il =>
            {
                // Signature: void(void** dataptrs, long* strides, long count, void* aux)
                // Args: arg0=dataptrs, arg1=strides, arg2=count

                // Load ptrs[0], ptrs[1], ptrs[2] and strides[0..2] once outside loop.
                var p0 = il.DeclareLocal(typeof(byte*));
                var p1 = il.DeclareLocal(typeof(byte*));
                var p2 = il.DeclareLocal(typeof(byte*));
                var s0 = il.DeclareLocal(typeof(long));
                var s1 = il.DeclareLocal(typeof(long));
                var s2 = il.DeclareLocal(typeof(long));
                var i = il.DeclareLocal(typeof(long));

                // p0 = dataptrs[0]
                il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, p0);
                // p1 = dataptrs[1]
                il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldc_I4, IntPtr.Size); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, p1);
                // p2 = dataptrs[2]
                il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldc_I4, 2 * IntPtr.Size); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I); il.Emit(OpCodes.Stloc, p2);

                // s0, s1, s2
                il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, s0);
                il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I4, sizeof(long)); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, s1);
                il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldc_I4, 2 * sizeof(long)); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stloc, s2);

                // for (i = 0; i < count; i++)
                il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);
                var lblTop = il.DefineLabel();
                var lblEnd = il.DefineLabel();
                il.MarkLabel(lblTop);
                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, lblEnd);

                // *(int*)(p2 + i*s2) = *(int*)(p0 + i*s0) + *(int*)(p1 + i*s1)
                il.Emit(OpCodes.Ldloc, p2); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, s2); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, p0); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, s0); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I4);
                il.Emit(OpCodes.Ldloc, p1); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, s1); il.Emit(OpCodes.Mul); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I4);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stind_I4);

                il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
                il.Emit(OpCodes.Br, lblTop);
                il.MarkLabel(lblEnd);
                il.Emit(OpCodes.Ret);
            }, cacheKey: "test_raw_int32_add_v1");

            for (int k = 0; k < 10; k++)
                Assert.AreEqual(k + (k + 10), c.GetInt32(k), $"c[{k}] wrong");
        }

        // =====================================================================
        // Tier B: Templated inner loop
        // =====================================================================

        [TestMethod]
        public void TierB_ElementWiseBinary_FusedMultiplyAdd_Float32()
        {
            // out = a * b + 1.0f
            var a = np.arange(16).astype(np.float32);
            var b = np.arange(16, 32).astype(np.float32);
            var c = np.empty(new Shape(16), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op: new[] { a, b, c },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            iter.ExecuteElementWiseBinary(
                NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    // Stack: [a, b]
                    il.Emit(OpCodes.Mul);                // a*b
                    il.Emit(OpCodes.Ldc_R4, 1.0f);
                    il.Emit(OpCodes.Add);                // a*b + 1
                },
                vectorBody: il =>
                {
                    // Stack: [va, vb]
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);
                    il.Emit(OpCodes.Ldc_R4, 1.0f);
                    ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
                },
                cacheKey: "test_fma_f32_const1");

            for (int k = 0; k < 16; k++)
            {
                float expected = (float)k * (float)(k + 16) + 1.0f;
                Assert.AreEqual(expected, c.GetSingle(k), 1e-5f, $"c[{k}] wrong");
            }
        }

        [TestMethod]
        public void TierB_ElementWiseUnary_Sqrt_Float32_Simd()
        {
            var input = np.arange(1, 33).astype(np.float32);      // 32 floats -> full Vector256 occupancy
            var output = np.empty(new Shape(32), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { input, output },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            iter.ExecuteElementWiseUnary(
                NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    ILKernelGenerator.EmitUnaryScalarOperation(il, UnaryOp.Sqrt, NPTypeCode.Single);
                },
                vectorBody: il =>
                {
                    ILKernelGenerator.EmitUnaryVectorOperation(il, UnaryOp.Sqrt, NPTypeCode.Single);
                },
                cacheKey: "test_sqrt_f32");

            for (int k = 0; k < 32; k++)
                Assert.AreEqual((float)Math.Sqrt(k + 1), output.GetSingle(k), 1e-5f, $"out[{k}] wrong");
        }

        [TestMethod]
        public void TierB_Ternary_Float32()
        {
            // out = a*b + c
            var a = np.arange(8).astype(np.float32);
            var b = np.arange(8, 16).astype(np.float32);
            var c = np.arange(16, 24).astype(np.float32);
            var d = np.empty(new Shape(8), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 4,
                op: new[] { a, b, c, d },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            iter.ExecuteElementWiseTernary(
                NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    // Stack: [a, b, c]
                    // Need: c + a*b   — but a*b needs a on the stack just below b, with c on top.
                    // We have [a, b, c]. Do: (a*b + c) via store c, mul, load c, add.
                    var tmpC = il.DeclareLocal(typeof(float));
                    il.Emit(OpCodes.Stloc, tmpC);        // stack: [a,b]
                    il.Emit(OpCodes.Mul);                 // stack: [a*b]
                    il.Emit(OpCodes.Ldloc, tmpC);         // stack: [a*b, c]
                    il.Emit(OpCodes.Add);                 // stack: [a*b + c]
                },
                vectorBody: il =>
                {
                    var tmpC = il.DeclareLocal(ILKernelGenerator.GetVectorType(typeof(float)));
                    il.Emit(OpCodes.Stloc, tmpC);
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);
                    il.Emit(OpCodes.Ldloc, tmpC);
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
                },
                cacheKey: "test_fma_ternary_f32");

            for (int k = 0; k < 8; k++)
            {
                float expected = (float)k * (float)(k + 8) + (float)(k + 16);
                Assert.AreEqual(expected, d.GetSingle(k), 1e-4f, $"d[{k}] wrong");
            }
        }

        [TestMethod]
        public void TierB_StridedInput_UsesScalarFallback()
        {
            // Slice every other element — inner stride = 2*elemSize, not elemSize.
            // The iterator keeps EXTERNAL_LOOP so ForEach runs a single inner-loop
            // call of count=16, and the emitted kernel's runtime contig check
            // fails (s_input != 4) → scalar-strided fallback inside the kernel.
            var big = np.arange(32).astype(np.float32);
            var sliced = big["::2"];                         // 16 elements, stride 8 bytes
            var output = np.empty(new Shape(16), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { sliced, output },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            iter.ExecuteElementWiseUnary(
                NPTypeCode.Single, NPTypeCode.Single,
                scalarBody: il =>
                {
                    il.Emit(OpCodes.Ldc_R4, 10.0f);
                    il.Emit(OpCodes.Add);                     // out = in + 10
                },
                vectorBody: il =>
                {
                    il.Emit(OpCodes.Ldc_R4, 10.0f);
                    ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
                    ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
                },
                cacheKey: "test_add10_f32");

            for (int k = 0; k < 16; k++)
                Assert.AreEqual(2 * k + 10.0f, output.GetSingle(k), 1e-5f, $"out[{k}] wrong");
        }

        [TestMethod]
        public void TierB_CacheReuse_SameKeyReturnsIdenticalDelegate()
        {
            // Two distinct iters calling ExecuteElementWise with the same
            // cacheKey should hit the same compiled delegate.
            ILKernelGenerator.ClearInnerLoopCache();

            var a1 = np.arange(4).astype(np.float32);
            var b1 = np.arange(4).astype(np.float32);
            var c1 = np.empty(new Shape(4), np.float32);
            var a2 = np.arange(4).astype(np.float32);
            var b2 = np.arange(4).astype(np.float32);
            var c2 = np.empty(new Shape(4), np.float32);

            Action<ILGenerator> scalar = il => il.Emit(OpCodes.Add);
            Action<ILGenerator> vec = il => ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);

            using (var iter = NpyIterRef.MultiNew(3, new[] { a1, b1, c1 },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }))
            {
                iter.ExecuteElementWiseBinary(NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
                    scalar, vec, "test_reuse_add_f32");
            }
            int afterFirst = (int)typeof(ILKernelGenerator)
                .GetProperty("InnerLoopCachedCount", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(null)!;

            using (var iter2 = NpyIterRef.MultiNew(3, new[] { a2, b2, c2 },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }))
            {
                iter2.ExecuteElementWiseBinary(NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
                    scalar, vec, "test_reuse_add_f32");  // same key
            }
            int afterSecond = (int)typeof(ILKernelGenerator)
                .GetProperty("InnerLoopCachedCount", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(null)!;

            Assert.AreEqual(afterFirst, afterSecond, "Second call should not have grown the cache.");
        }

        // =====================================================================
        // Tier C: Expression DSL
        // =====================================================================

        [TestMethod]
        public void TierC_Expression_AddConstant()
        {
            var a = np.arange(12).astype(np.float32);
            var b = np.empty(new Shape(12), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            var expr = NpyExpr.Input(0) + NpyExpr.Const(5.0f);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            for (int k = 0; k < 12; k++)
                Assert.AreEqual(k + 5.0f, b.GetSingle(k), 1e-5f);
        }

        [TestMethod]
        public void TierC_Expression_CompoundFma()
        {
            // out = (a + b) * c + 1
            var a = np.arange(8).astype(np.float32);
            var b = np.arange(8, 16).astype(np.float32);
            var c = np.arange(16, 24).astype(np.float32);
            var d = np.empty(new Shape(8), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 4, op: new[] { a, b, c, d },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            var expr = (NpyExpr.Input(0) + NpyExpr.Input(1)) * NpyExpr.Input(2) + NpyExpr.Const(1.0f);
            iter.ExecuteExpression(expr,
                new[] { NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single },
                NPTypeCode.Single);

            for (int k = 0; k < 8; k++)
            {
                float expected = ((float)k + (float)(k + 8)) * (float)(k + 16) + 1.0f;
                Assert.AreEqual(expected, d.GetSingle(k), 1e-3f, $"d[{k}] wrong");
            }
        }

        [TestMethod]
        public void TierC_Expression_SqrtOfSumSquares()
        {
            // out = sqrt(a^2 + b^2)   — hypot, single-kernel
            var a = np.array(new float[] { 3, 6, 5, 8 });
            var b = np.array(new float[] { 4, 8, 12, 15 });
            var c = np.empty(new Shape(4), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 3, op: new[] { a, b, c },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY
                });

            var expr = NpyExpr.Sqrt(NpyExpr.Square(NpyExpr.Input(0)) + NpyExpr.Square(NpyExpr.Input(1)));
            iter.ExecuteExpression(expr,
                new[] { NPTypeCode.Single, NPTypeCode.Single }, NPTypeCode.Single);

            float[] expected = { 5f, 10f, 13f, 17f };
            for (int k = 0; k < 4; k++)
                Assert.AreEqual(expected[k], c.GetSingle(k), 1e-4f, $"c[{k}] wrong");
        }

        [TestMethod]
        public void TierC_Expression_NegateAndAbs()
        {
            var a = np.array(new float[] { 3, -4, 5, -6 });
            var b = np.empty(new Shape(4), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            // out = -|a|
            var expr = -NpyExpr.Abs(NpyExpr.Input(0));
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            float[] expected = { -3f, -4f, -5f, -6f };
            for (int k = 0; k < 4; k++)
                Assert.AreEqual(expected[k], b.GetSingle(k), 1e-5f);
        }

        [TestMethod]
        public void TierC_Expression_DoubleDtype()
        {
            var a = np.arange(10).astype(np.float64);
            var b = np.empty(new Shape(10), np.float64);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            var expr = NpyExpr.Input(0) * NpyExpr.Const(2.0) + NpyExpr.Const(3.0);
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double);

            for (int k = 0; k < 10; k++)
                Assert.AreEqual(2.0 * k + 3.0, b.GetDouble(k), 1e-9);
        }

        [TestMethod]
        public void TierC_Expression_StridedPath()
        {
            // Expression tree must also work on strided views (kernel's
            // runtime contig check routes to the scalar-strided fallback).
            var big = np.arange(20).astype(np.float32);
            var sliced = big["::2"];                          // 10 elements, stride=2*4=8 bytes
            var output = np.empty(new Shape(10), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { sliced, output },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            var expr = NpyExpr.Input(0) * NpyExpr.Input(0);   // square
            iter.ExecuteExpression(expr, new[] { NPTypeCode.Single }, NPTypeCode.Single);

            for (int k = 0; k < 10; k++)
            {
                float src = 2f * k;
                Assert.AreEqual(src * src, output.GetSingle(k), 1e-5f, $"out[{k}] wrong");
            }
        }

        // =====================================================================
        // Argument validation
        // =====================================================================

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TierB_WrongOperandCount_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            // Iterator has 2 operands, we claim 3 types.
            iter.ExecuteElementWise(
                new[] { NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single },
                scalarBody: il => il.Emit(OpCodes.Add),
                vectorBody: null,
                cacheKey: "test_bad_nop");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TierC_WrongInputCount_Throws()
        {
            var a = np.arange(4).astype(np.float32);
            var b = np.empty(new Shape(4), np.float32);

            using var iter = NpyIterRef.MultiNew(
                nop: 2, op: new[] { a, b },
                flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            // Iter has NOp=2 → expects inputTypes.Length == 1, but we pass 2.
            iter.ExecuteExpression(
                NpyExpr.Input(0),
                new[] { NPTypeCode.Single, NPTypeCode.Single },
                NPTypeCode.Single);
        }
    }
}
