using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     Pins <see cref="InnerLoopKernelKey"/>, the packed identity the production ufunc
    ///     routes look their Tier-3B inner-loop kernels up by. Two properties matter: the key
    ///     must reproduce the legacy interpolated string EXACTLY (the string cache and the
    ///     DynamicMethod names keep their identity across the change), and distinct
    ///     (family, op, dtypes) must never collide.
    /// </summary>
    [TestClass]
    public class InnerLoopKernelKeyTests
    {
        [TestMethod]
        public void ToCacheKey_Reproduces_Legacy_String_Formats()
        {
            Assert.AreEqual("npy_binop_Add_Double_Double_Double",
                InnerLoopKernelKey.Binary(BinaryOp.Add, NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double).ToCacheKey());
            Assert.AreEqual("npy_binop_Power_Int32_Single_Double",
                InnerLoopKernelKey.Binary(BinaryOp.Power, NPTypeCode.Int32, NPTypeCode.Single, NPTypeCode.Double).ToCacheKey());
            Assert.AreEqual("npy_cmp_Less_Int64_Double",
                InnerLoopKernelKey.Comparison(ComparisonOp.Less, NPTypeCode.Int64, NPTypeCode.Double).ToCacheKey());
            Assert.AreEqual("npy_unop_Sqrt_Int32_Double",
                InnerLoopKernelKey.Unary(UnaryOp.Sqrt, NPTypeCode.Int32, NPTypeCode.Double).ToCacheKey());
            Assert.AreEqual("npy_shift_L_Int32_Int32",
                InnerLoopKernelKey.Shift(true, NPTypeCode.Int32, NPTypeCode.Int32).ToCacheKey());
            Assert.AreEqual("npy_shift_R_Byte_Byte",
                InnerLoopKernelKey.Shift(false, NPTypeCode.Byte, NPTypeCode.Byte).ToCacheKey());
        }

        [TestMethod]
        public void Fields_Round_Trip()
        {
            var key = InnerLoopKernelKey.Binary(BinaryOp.Multiply, NPTypeCode.Complex, NPTypeCode.Half, NPTypeCode.Complex);
            Assert.AreEqual(InnerLoopKernelKey.KernelFamily.Binary, key.Family);
            Assert.AreEqual((int)BinaryOp.Multiply, key.Op);
            Assert.AreEqual(NPTypeCode.Complex, key.Type0);   // 128 — the largest NPTypeCode, must survive the byte pack
            Assert.AreEqual(NPTypeCode.Half, key.Type1);
            Assert.AreEqual(NPTypeCode.Complex, key.Type2);
        }

        [TestMethod]
        public void Distinct_Identities_Never_Collide()
        {
            var types = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            };
            var seen = new HashSet<InnerLoopKernelKey>();
            var seenStrings = new HashSet<string>();
            int expected = 0;

            foreach (BinaryOp op in System.Enum.GetValues(typeof(BinaryOp)))
                foreach (var l in types)
                    foreach (var r in types)
                    {
                        var k = InnerLoopKernelKey.Binary(op, l, r, r);
                        Assert.IsTrue(seen.Add(k), k.ToString());
                        Assert.IsTrue(seenStrings.Add(k.ToCacheKey()), k.ToString());
                        expected++;
                    }
            foreach (ComparisonOp op in System.Enum.GetValues(typeof(ComparisonOp)))
                foreach (var l in types)
                    foreach (var r in types)
                    {
                        var k = InnerLoopKernelKey.Comparison(op, l, r);
                        Assert.IsTrue(seen.Add(k), k.ToString());
                        Assert.IsTrue(seenStrings.Add(k.ToCacheKey()), k.ToString());
                        expected++;
                    }
            foreach (UnaryOp op in System.Enum.GetValues(typeof(UnaryOp)))
                foreach (var i in types)
                    foreach (var o in types)
                    {
                        var k = InnerLoopKernelKey.Unary(op, i, o);
                        Assert.IsTrue(seen.Add(k), k.ToString());
                        Assert.IsTrue(seenStrings.Add(k.ToCacheKey()), k.ToString());
                        expected++;
                    }
            foreach (var isLeft in new[] { true, false })
                foreach (var v in types)
                    foreach (var r in types)
                    {
                        var k = InnerLoopKernelKey.Shift(isLeft, v, r);
                        Assert.IsTrue(seen.Add(k), k.ToString());
                        Assert.IsTrue(seenStrings.Add(k.ToCacheKey()), k.ToString());
                        expected++;
                    }

            Assert.AreEqual(expected, seen.Count);
        }

        [TestMethod]
        public void Production_Route_Registers_Kernel_Under_Both_Keys()
        {
            // An int16 add with a provided out takes the NDIter ufunc route; afterwards the
            // kernel must be reachable through the packed key AND the legacy string key —
            // the same delegate — so GeneratedDelegates.InnerLoopCount and the string cache
            // keep reporting one kernel per identity.
            var a = np.arange(6).astype(np.int16);
            var b = np.arange(6).astype(np.int16);
            var o = np.empty(new Shape(6), np.int16);
            np.add(a, b, o);
            Assert.AreEqual((short)10, o.GetInt16(5));

            var key = InnerLoopKernelKey.Binary(BinaryOp.Add, NPTypeCode.Int16, NPTypeCode.Int16, NPTypeCode.Int16);
            Assert.IsTrue(DirectILKernelGenerator.TryGetInnerLoop(key, out var packed), "packed-key cache must hold the kernel");
            Assert.IsTrue(DirectILKernelGenerator._innerLoopCache.TryGetValue(key.ToCacheKey(), out var byString), "string cache must hold the kernel");
            Assert.AreSame(byString, packed);
        }
    }
}
