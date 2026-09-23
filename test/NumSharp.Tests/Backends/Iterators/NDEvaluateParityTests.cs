using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    /// np.evaluate / NDExpr — the Phase 0 parity contract of docs/plans/ndexpr-evaluate.md
    /// (G1–G8, G10). Every expectation below is pinned to a NumPy 2.4.2 probe (the values in the
    /// comments are NumPy's own output), so these tests are the executable form of the plan's
    /// semantics table, not a description of what NumSharp happened to do.
    /// </summary>
    [TestClass]
    public class NDEvaluateParityTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        // =====================================================================
        // G1 — sub-32-bit integer zero test (Where / LogicalNot over int8 threw
        // "Zero-push unsupported for SByte")
        // =====================================================================

        [TestMethod]
        public void G1_Where_And_LogicalNot_Over_NarrowIntegerCondition()
        {
            var a = np.array(new double[] { 1, 2 });
            var b = np.array(new double[] { 10, 20 });

            foreach (var tc in new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char })
            {
                var cond = np.array(new int[] { 0, 5 }).astype(tc);

                // np.where(np.array([0,5], int8), a, b) -> [10., 2.]
                var w = np.evaluate(NDExpr.Where(NDExpr.Arr(cond), NDExpr.Arr(a), NDExpr.Arr(b)));
                Assert.AreEqual(NPTypeCode.Double, w.typecode, tc.ToString());
                Assert.AreEqual(10.0, w.GetDouble(0), tc.ToString());
                Assert.AreEqual(2.0, w.GetDouble(1), tc.ToString());

                // np.logical_not(np.array([0,5], int8)) -> [ True False]
                var n = np.evaluate(NDExpr.LogicalNot(NDExpr.Arr(cond)));
                Assert.AreEqual(NPTypeCode.Boolean, n.typecode, tc.ToString());
                Assert.IsTrue(n.GetBoolean(0), tc.ToString());
                Assert.IsFalse(n.GetBoolean(1), tc.ToString());
            }
        }

        // =====================================================================
        // G2 — np.absolute's complex loop is D->d (float64 magnitude)
        // =====================================================================

        [TestMethod]
        public void G2_Abs_Complex_IsFloat64Magnitude()
        {
            var z = np.array(new[] { new Complex(3, -4), Complex.Zero, new Complex(-1, 1), new Complex(double.NaN, 0) });

            // np.abs(z) -> float64 [5., 0., 1.41421356, nan]
            var r = np.evaluate(NDExpr.Abs(NDExpr.Arr(z)));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(5.0, r.GetDouble(0));
            Assert.AreEqual(0.0, r.GetDouble(1));
            Assert.AreEqual(System.Math.Sqrt(2.0), r.GetDouble(2), 0.0);
            Assert.IsTrue(double.IsNaN(r.GetDouble(3)));

            // Bit-identical to the engine's own np.abs(complex) (which is NumPy-exact incl. NaN sign).
            var u = np.abs(z);
            for (long i = 0; i < 4; i++)
                Assert.AreEqual(BitConverter.DoubleToInt64Bits(u.GetDouble(i)), BitConverter.DoubleToInt64Bits(r.GetDouble(i)), $"[{i}]");

            // And it composes: abs(z) * 2 is a float64 tree.
            var t = np.evaluate(NDExpr.Abs(NDExpr.Arr(z)) * 2);
            Assert.AreEqual(NPTypeCode.Double, t.typecode);
            Assert.AreEqual(10.0, t.GetDouble(0));
        }

        // =====================================================================
        // G4 — complex min/max: lexicographic (real, imag), first NaN sticks
        // =====================================================================

        [TestMethod]
        public void G4_Complex_MinMax_Reductions_AreLexicographic_NaNSticks()
        {
            var z = np.array(new[] { new Complex(3, -4), Complex.Zero, new Complex(-1, 1) });
            var zn = np.array(new[] { new Complex(3, -4), Complex.Zero, new Complex(-1, 1), new Complex(double.NaN, 0) });

            // np.max([3-4j, 0, -1+1j]) -> (3-4j); np.min -> (-1+1j)
            Assert.AreEqual(new Complex(3, -4), np.evaluate(NDExpr.Max(NDExpr.Arr(z))).GetComplex(0));
            Assert.AreEqual(new Complex(-1, 1), np.evaluate(NDExpr.Min(NDExpr.Arr(z))).GetComplex(0));

            // np.min([3-4j, 0, -1+1j, nan+0j]) -> (nan+0j)
            var mn = np.evaluate(NDExpr.Min(NDExpr.Arr(zn))).GetComplex(0);
            Assert.IsTrue(double.IsNaN(mn.Real) && mn.Imaginary == 0.0, mn.ToString());

            // np.max([1+1j, nanj]) -> nanj (the NaN element wins whenever it is met)
            var zi = np.array(new[] { new Complex(1, 1), new Complex(0, double.NaN) });
            var mx = np.evaluate(NDExpr.Max(NDExpr.Arr(zi))).GetComplex(0);
            Assert.IsTrue(mx.Real == 0.0 && double.IsNaN(mx.Imaginary), mx.ToString());

            // Elementwise np.maximum / np.minimum over complex compose too.
            var w = np.array(new[] { new Complex(3, 5), new Complex(0, -1), new Complex(-2, 9) });
            var emax = np.evaluate(NDExpr.Max(NDExpr.Arr(z), NDExpr.Arr(w)));
            Assert.AreEqual(NPTypeCode.Complex, emax.typecode);
            Assert.AreEqual(new Complex(3, 5), emax.GetComplex(0));   // same real, larger imag
            Assert.AreEqual(Complex.Zero, emax.GetComplex(1));        // 0 > 0-1j
            Assert.AreEqual(new Complex(-1, 1), emax.GetComplex(2));  // -1 > -2
        }

        [TestMethod]
        public void MinMax_SignedZeroTie_SecondOperandWins_LikeNumPy()
        {
            var neg = np.array(new double[] { -0.0 });
            var pos = np.array(new double[] { 0.0 });

            // np.maximum([-0.], [0.]) -> [0.] ; np.maximum([0.], [-0.]) -> [-0.]
            Assert.AreEqual(0L, BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Max(NDExpr.Arr(neg), NDExpr.Arr(pos))).GetDouble(0)));
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(-0.0), BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Max(NDExpr.Arr(pos), NDExpr.Arr(neg))).GetDouble(0)));

            // np.max([0., -0.]) -> -0. ; np.max([-0., 0.]) -> 0.
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(-0.0), BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Max(NDExpr.Arr(np.array(new[] { 0.0, -0.0 })))).GetDouble(0)));
            Assert.AreEqual(0L, BitConverter.DoubleToInt64Bits(np.evaluate(NDExpr.Max(NDExpr.Arr(np.array(new[] { -0.0, 0.0 })))).GetDouble(0)));
        }

        // =====================================================================
        // G5 — negative integer exponent inside an exponent ARRAY raises per element
        // =====================================================================

        [TestMethod]
        public void G5_Power_NegativeIntegerExponentArray_RaisesLikeNumPy()
        {
            var b32 = np.array(new int[] { 2, 3 });
            var e32 = np.array(new int[] { 1, -1 });
            var ex = Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Power(NDExpr.Arr(b32), NDExpr.Arr(e32))));
            Assert.AreEqual("Integers to negative integer powers are not allowed.", ex.Message);

            var b64 = np.array(new long[] { 2, 3 });
            var e64 = np.array(new long[] { 2, -2 });
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Power(NDExpr.Arr(b64), NDExpr.Arr(e64))));

            var b8 = np.array(new sbyte[] { 2, 3 });
            var e8 = np.array(new sbyte[] { 2, -1 });
            Assert.ThrowsException<ArgumentException>(() => np.evaluate(NDExpr.Power(NDExpr.Arr(b8), NDExpr.Arr(e8))));

            // A non-negative exponent array computes; an unsigned exponent can never be negative.
            var ok = np.evaluate(NDExpr.Power(NDExpr.Arr(b32), NDExpr.Arr(np.array(new int[] { 3, 2 }))));
            Assert.AreEqual(8, ok.GetInt32(0));
            Assert.AreEqual(9, ok.GetInt32(1));
            var u = np.evaluate(NDExpr.Power(NDExpr.Arr(np.array(new byte[] { 2, 3 })), NDExpr.Arr(np.array(new byte[] { 7, 5 }))));
            Assert.AreEqual((byte)128, u.GetByte(0));
            Assert.AreEqual((byte)243, u.GetByte(1));

            // Float loops never check (2.0 ** -1 = 0.5).
            var f = np.evaluate(NDExpr.Power(NDExpr.Arr(np.array(new double[] { 2 })), NDExpr.Arr(np.array(new int[] { -1 }))));
            Assert.AreEqual(0.5, f.GetDouble(0));
        }

        // =====================================================================
        // G6 — every literal the DSL can now spell, with its NEP50 kind
        // =====================================================================

        [TestMethod]
        public void G6_UInt64Literal_StaysUInt64_AndWraps()
        {
            var u = np.array(new ulong[] { 1UL });

            // np.array([1], uint64) + 18446744073709551615 -> uint64 [0]
            var r = np.evaluate((NDExpr)u + 18446744073709551615UL);
            Assert.AreEqual(NPTypeCode.UInt64, r.typecode);
            Assert.AreEqual(0UL, r.GetUInt64(0));

            // A ulong that fits long is an ordinary Python int: uint64 + 5 stays uint64.
            Assert.AreEqual(NPTypeCode.UInt64, np.evaluate((NDExpr)u + 5UL).typecode);

            // np.array([1], int64) + 2**63 -> OverflowError
            var s = np.array(new long[] { 1L });
            var ex = Assert.ThrowsException<OverflowException>(() => np.evaluate((NDExpr)s + 9223372036854775808UL));
            Assert.AreEqual("Python integer 9223372036854775808 out of bounds for int64", ex.Message);

            // A float array adopts it as a value (no range check).
            var f = np.array(new double[] { 1.0 });
            Assert.AreEqual(NPTypeCode.Double, np.evaluate((NDExpr)f + 18446744073709551615UL).typecode);
        }

        [TestMethod]
        public void G6_BoolLiteral_IsWeak_AdoptsEveryDtype()
        {
            // bool+True -> bool ; f4+True -> float32 ; i1+True -> int8 (probed)
            var b = np.array(new bool[] { false, true });
            var rb = np.evaluate((NDExpr)b + true);
            Assert.AreEqual(NPTypeCode.Boolean, rb.typecode);
            Assert.IsTrue(rb.GetBoolean(0) && rb.GetBoolean(1)); // logical or

            var f4 = np.array(new float[] { 1.5f });
            var rf = np.evaluate((NDExpr)f4 + true);
            Assert.AreEqual(NPTypeCode.Single, rf.typecode);
            Assert.AreEqual(2.5f, rf.GetSingle(0));

            var i1 = np.array(new sbyte[] { 5 });
            var ri = np.evaluate((NDExpr)i1 + true);
            Assert.AreEqual(NPTypeCode.SByte, ri.typecode);
            Assert.AreEqual((sbyte)6, ri.GetSByte(0));

            // np.where([True, False], True, 2.5) -> float64 [1. , 2.5]
            var cond = np.array(new bool[] { true, false });
            var w = np.evaluate(NDExpr.Where(NDExpr.Arr(cond), NDExpr.Const(true), NDExpr.Const(2.5)));
            Assert.AreEqual(NPTypeCode.Double, w.typecode);
            Assert.AreEqual(1.0, w.GetDouble(0));
            Assert.AreEqual(2.5, w.GetDouble(1));
        }

        [TestMethod]
        public void Bool_Add_Multiply_VectorPath_IsLogicalOrAnd()
        {
            // 64 elements run the SIMD block (32 byte lanes): bool add/multiply must be the
            // normalized logical or/and there too — Vector{N}<bool> arithmetic threw NotSupported.
            var a = np.array(Enumerable.Range(0, 64).Select(i => i % 2 == 0).ToArray());
            var b = np.array(Enumerable.Range(0, 64).Select(i => i % 3 == 0).ToArray());

            var or = np.evaluate(NDExpr.Add(NDExpr.Arr(a), NDExpr.Arr(b)));
            var and = np.evaluate(NDExpr.Multiply(NDExpr.Arr(a), NDExpr.Arr(b)));
            var orLit = np.evaluate((NDExpr)a + true);
            var andLit = np.evaluate((NDExpr)a * false);
            Assert.AreEqual(NPTypeCode.Boolean, or.typecode);
            for (int i = 0; i < 64; i++)
            {
                bool x = i % 2 == 0, y = i % 3 == 0;
                Assert.AreEqual(x || y, or.GetBoolean(i), $"or[{i}]");
                Assert.AreEqual(x && y, and.GetBoolean(i), $"and[{i}]");
                Assert.IsTrue(orLit.GetBoolean(i), $"orLit[{i}]");
                Assert.IsFalse(andLit.GetBoolean(i), $"andLit[{i}]");
            }
        }

        [TestMethod]
        public void G6_ComplexLiteral_IsWeak_ForcesComplex()
        {
            var f4 = np.array(new float[] { 1.5f });
            var i4 = np.array(new int[] { 2 });

            // f4 + 1j -> complex (NumPy complex64; NumSharp's single complex width) ; i4 + 1j -> complex128
            var rf = np.evaluate((NDExpr)f4 + new Complex(0, 1));
            Assert.AreEqual(NPTypeCode.Complex, rf.typecode);
            Assert.AreEqual(new Complex(1.5, 1), rf.GetComplex(0));

            var ri = np.evaluate((NDExpr)i4 * new Complex(0, 1));
            Assert.AreEqual(NPTypeCode.Complex, ri.typecode);
            Assert.AreEqual(new Complex(0, 2), ri.GetComplex(0));

            var z = np.array(new[] { new Complex(1, 1) });
            Assert.AreEqual(new Complex(0, 2), np.evaluate((NDExpr)z + new Complex(-1, 1)).GetComplex(0));
        }

        [TestMethod]
        public void G6_StrongLiterals_Half_Decimal_PromoteLikeZeroDArrays()
        {
            // np.float16(2) is a STRONG scalar: f8 + f16 -> f8 ; i1 + f16 -> f16 ; f4 + f16 -> f4
            var f8 = np.array(new double[] { 1.25 });
            var i1 = np.array(new sbyte[] { 3 });
            var f4 = np.array(new float[] { 0.5f });
            Assert.AreEqual(NPTypeCode.Double, np.evaluate((NDExpr)f8 + (Half)2).typecode);
            Assert.AreEqual(3.25, np.evaluate((NDExpr)f8 + (Half)2).GetDouble(0));
            var h = np.evaluate((NDExpr)i1 + (Half)2);
            Assert.AreEqual(NPTypeCode.Half, h.typecode);
            Assert.AreEqual((Half)5, h.GetHalf(0));
            Assert.AreEqual(NPTypeCode.Single, np.evaluate((NDExpr)f4 + (Half)2).typecode);

            // decimal: a strong NumSharp-only literal, emitted EXACTLY (0.1m stays 0.1m, no double detour).
            var d = np.array(new decimal[] { 1.0m, 2.5m });
            var rd = np.evaluate((NDExpr)d + 0.1m);
            Assert.AreEqual(NPTypeCode.Decimal, rd.typecode);
            Assert.AreEqual(1.1m, rd.GetDecimal(0));
            Assert.AreEqual(2.6m, rd.GetDecimal(1));
            Assert.AreEqual(NPTypeCode.Decimal, np.evaluate((NDExpr)f8 + 0.1m).typecode);

            // uint literal is a Python int (weak): i4 + 2u -> int32
            var i4 = np.array(new int[] { 40 });
            var ru = np.evaluate((NDExpr)i4 + 2u);
            Assert.AreEqual(NPTypeCode.Int32, ru.typecode);
            Assert.AreEqual(42, ru.GetInt32(0));
        }

        // =====================================================================
        // G7 — int64 vs uint64 compares exactly (NumPy's qQ / Qq loops), and an
        // out-of-range Python int compares instead of raising
        // =====================================================================

        [TestMethod]
        public void G7_Int64_UInt64_Comparison_IsExactPast2Pow53()
        {
            var u = np.array(new ulong[] { 9223372036854775809UL });   // 2^63 + 1
            var s = np.array(new long[] { 9223372036854775807L });     // 2^63 - 1

            // np.greater(u, s) -> True ; np.equal(u, s) -> False ; np.less(s, u) -> True
            Assert.IsTrue(np.evaluate(NDExpr.Greater(NDExpr.Arr(u), NDExpr.Arr(s))).GetBoolean(0));
            Assert.IsFalse(np.evaluate(NDExpr.Equal(NDExpr.Arr(u), NDExpr.Arr(s))).GetBoolean(0));
            Assert.IsTrue(np.evaluate(NDExpr.NotEqual(NDExpr.Arr(u), NDExpr.Arr(s))).GetBoolean(0));
            Assert.IsTrue(np.evaluate(NDExpr.Less(NDExpr.Arr(s), NDExpr.Arr(u))).GetBoolean(0));
            Assert.IsFalse(np.evaluate(NDExpr.GreaterEqual(NDExpr.Arr(s), NDExpr.Arr(u))).GetBoolean(0));

            // np.greater(uint64 [1], int64 [-1]) -> True (the float64 detour agreed here; the sign
            // test is what the exact loop must keep right for a negative signed operand).
            Assert.IsTrue(np.evaluate(NDExpr.Greater(NDExpr.Arr(np.array(new ulong[] { 1 })), NDExpr.Arr(np.array(new long[] { -1 })))).GetBoolean(0));

            // The shared engine fix: the comparison ufuncs / operators agree.
            Assert.IsTrue((u > s).GetBoolean(0));
            Assert.IsFalse((u == s).GetBoolean(0));
            Assert.IsTrue(np.greater(u, s).GetBoolean(0));
            Assert.IsTrue(np.less(s, u).GetBoolean(0));
        }

        [TestMethod]
        public void G7_OutOfRangeIntegerLiteral_ComparesInsteadOfRaising()
        {
            var u = np.array(new ulong[] { 1UL });
            var i1 = np.array(new sbyte[] { 5 });

            // uint64 [1] > -1 -> True ; == -1 -> False ; > 2**64-1 -> False
            Assert.IsTrue(np.evaluate(NDExpr.Greater(NDExpr.Arr(u), NDExpr.Const(-1))).GetBoolean(0));
            Assert.IsFalse(np.evaluate(NDExpr.Equal(NDExpr.Arr(u), NDExpr.Const(-1))).GetBoolean(0));
            Assert.IsFalse(np.evaluate(NDExpr.Greater(NDExpr.Arr(u), NDExpr.Const(18446744073709551615UL))).GetBoolean(0));
            Assert.IsTrue(np.evaluate(NDExpr.Less(NDExpr.Arr(u), NDExpr.Const(18446744073709551615UL))).GetBoolean(0));

            // int8 [5] > 300 -> False ; int8 [5] < 300 -> True (arithmetic int8 + 300 still raises)
            Assert.IsFalse(np.evaluate(NDExpr.Greater(NDExpr.Arr(i1), NDExpr.Const(300))).GetBoolean(0));
            Assert.IsTrue(np.evaluate(NDExpr.Less(NDExpr.Arr(i1), NDExpr.Const(300))).GetBoolean(0));
            Assert.ThrowsException<OverflowException>(() => np.evaluate((NDExpr)i1 + 300));

            // In range: the literal adopts the array dtype as before.
            Assert.IsTrue(np.evaluate(NDExpr.Greater(NDExpr.Arr(i1), NDExpr.Const(4))).GetBoolean(0));
        }

        // =====================================================================
        // G10 — Call accepts every dtype's CLR type
        // =====================================================================

        [TestMethod]
        public void G10_Call_Accepts_SByte_Half_Complex_Signatures()
        {
            Func<Half, Half> twiceH = h => (Half)((float)h * 2f);
            var f2 = np.array(new Half[] { (Half)1.5f, (Half)2 });
            var rh = np.evaluate(NDExpr.Call(twiceH, NDExpr.Arr(f2)));
            Assert.AreEqual(NPTypeCode.Half, rh.typecode);
            Assert.AreEqual((Half)3, rh.GetHalf(0));

            Func<sbyte, sbyte> negS = v => (sbyte)-v;
            var i1 = np.array(new sbyte[] { 5, -7 });
            var rs = np.evaluate(NDExpr.Call(negS, NDExpr.Arr(i1)));
            Assert.AreEqual(NPTypeCode.SByte, rs.typecode);
            Assert.AreEqual((sbyte)-5, rs.GetSByte(0));

            Func<Complex, Complex> conj = Complex.Conjugate;
            var z = np.array(new[] { new Complex(1, 2) });
            var rc = np.evaluate(NDExpr.Call(conj, NDExpr.Arr(z)));
            Assert.AreEqual(NPTypeCode.Complex, rc.typecode);
            Assert.AreEqual(new Complex(1, -2), rc.GetComplex(0));
        }

        // =====================================================================
        // Sweep: 15 dtypes × node kinds — no cell throws unless NumPy has no loop
        // =====================================================================

        [TestMethod]
        public void Sweep_AllDtypes_EveryNodeKind_ComputesUnlessNumPyHasNoLoop()
        {
            var failures = new System.Collections.Generic.List<string>();

            foreach (var tc in AllDtypes)
            {
                NDArray a = tc == NPTypeCode.Boolean
                    ? np.array(new bool[] { true, false, true, true, false, true })
                    : np.arange(1, 7).astype(tc);
                var x = NDExpr.Arr(a);
                bool isBool = tc == NPTypeCode.Boolean;
                bool isInt = NDExprTypeRulesProbe.IsIntegerKind(tc);
                bool isComplex = tc == NPTypeCode.Complex;

                var cells = new (string name, Func<NDExpr> build, bool numpyRaises)[]
                {
                    ("add", () => x + x, false),
                    ("subtract", () => x - x, isBool),                    // numpy boolean subtract
                    ("multiply", () => x * x, false),
                    ("divide", () => x / x, false),
                    ("negate", () => -x, isBool),                         // numpy boolean negative
                    ("abs", () => NDExpr.Abs(x), false),
                    ("sqrt", () => NDExpr.Sqrt(x), false),
                    ("exp", () => NDExpr.Exp(x), false),
                    ("floor", () => NDExpr.Floor(x), isComplex),          // no complex floor loop
                    ("round", () => NDExpr.Round(x), false),
                    ("square", () => NDExpr.Square(x), false),
                    ("sign", () => NDExpr.Sign(x), isBool),               // no bool sign loop
                    ("mod", () => x % x, isComplex),                      // no complex remainder loop
                    ("power", () => NDExpr.Power(x, x), false),
                    ("bitwise_and", () => x & x, !(isBool || isInt)),     // float/complex/decimal no loop
                    ("invert", () => ~x, !(isBool || isInt)),
                    ("equal", () => NDExpr.Equal(x, x), false),
                    ("greater", () => NDExpr.Greater(x, x), false),
                    ("logical_not", () => !x, false),
                    ("isnan", () => NDExpr.IsNaN(x), false),
                    ("isfinite", () => NDExpr.IsFinite(x), false),
                    ("where", () => NDExpr.Where(x, x, x), false),
                    ("minimum", () => NDExpr.Min(x, x), false),
                    ("maximum", () => NDExpr.Max(x, x), false),
                    ("clamp", () => NDExpr.Clamp(x, x, x), false),
                    ("sum", () => NDExpr.Sum(x), false),
                    ("prod", () => NDExpr.Prod(x), false),
                    ("min", () => NDExpr.Min(x), false),
                    ("max", () => NDExpr.Max(x), false),
                    ("mean", () => NDExpr.Mean(x), false),
                    ("sum_axis", () => NDExpr.Sum(x, 0), false),
                    ("max_axis_keep", () => NDExpr.Max(x, 0, keepdims: true), false),
                    ("lit_int", () => x + 2, false),
                    ("lit_float", () => x * 1.5, false),
                    ("compare_lit", () => NDExpr.Greater(x, 2), false),
                };

                foreach (var (name, build, numpyRaises) in cells)
                {
                    try
                    {
                        var r = np.evaluate(build());
                        if (numpyRaises)
                            failures.Add($"{tc}/{name}: computed ({r.typecode}) where NumPy has no loop");
                        else if (r.size == 0)
                            failures.Add($"{tc}/{name}: empty result");
                    }
                    catch (Exception e) when (!numpyRaises)
                    {
                        failures.Add($"{tc}/{name}: THREW {e.GetType().Name}: {e.Message}");
                    }
                    catch (Exception)
                    {
                        // expected: NumPy raises too
                    }
                }
            }

            Assert.AreEqual(0, failures.Count, string.Join("\n", failures));
        }

        /// <summary>Test-side mirror of the private integer-kind predicate (keeps the sweep self-contained).</summary>
        private static class NDExprTypeRulesProbe
        {
            public static bool IsIntegerKind(NPTypeCode t)
                => t == NPTypeCode.Byte || t == NPTypeCode.SByte || t == NPTypeCode.Int16 || t == NPTypeCode.UInt16 ||
                   t == NPTypeCode.Char || t == NPTypeCode.Int32 || t == NPTypeCode.UInt32 || t == NPTypeCode.Int64 ||
                   t == NPTypeCode.UInt64;
        }
    }
}
