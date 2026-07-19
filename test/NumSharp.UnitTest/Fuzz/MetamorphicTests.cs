using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Metamorphic / oracle-free invariants. These assert mathematical relationships that
    ///     must hold REGARDLESS of NumPy (no oracle), so they catch internal-consistency bugs the
    ///     differential corpus can't: round-trips, involutions, identities, order invariance.
    ///
    ///     Each invariant runs across several dtypes/shapes/layouts. A failure is collected (not
    ///     fatal) and surfaced at the end so the whole property is visible at once. Any genuine
    ///     violation is a NumSharp bug.
    /// </summary>
    [TestClass]
    public class MetamorphicTests
    {
        private static NDArray Arange(int n, NPTypeCode tc) => np.arange(n).astype(tc);

        private static bool BytesEqual(NDArray a, NDArray b)
        {
            if (a.typecode != b.typecode) return false;
            var ba = FuzzCorpus.ResultBytes(a);
            var bb = FuzzCorpus.ResultBytes(b);
            if (ba.Length != bb.Length) return false;
            for (int i = 0; i < ba.Length; i++)
                if (ba[i] != bb[i]) return false;
            return true;
        }

        private static void Run(string property, IEnumerable<(string label, Func<bool> check)> cases)
        {
            var failures = new List<string>();
            foreach (var (label, check) in cases)
            {
                bool ok;
                try { ok = check(); }
                catch (Exception e) { ok = false; label.ToString(); failures.Add($"{label}: THREW {e.GetType().Name}: {e.Message}"); continue; }
                if (!ok) failures.Add(label);
            }
            if (failures.Count > 0)
                Assert.Fail($"{property}: {failures.Count} invariant violation(s):\n  " + string.Join("\n  ", failures));
        }

        private static readonly NPTypeCode[] IntFloat =
            { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double };
        private static readonly NPTypeCode[] AllNumeric =
            { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Byte, NPTypeCode.UInt32 };
        // G17: the previously untested dtype tails. Values stay small integer-valued (arange), so
        // Half is exact and Complex/Decimal arithmetic is exact -- the invariants genuinely hold.
        // bool joins only where pure data movement or ==-reflexivity keeps the invariant meaningful.
        private static readonly NPTypeCode[] HalfComplexDecimal =
            { NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal };

        // -(-a) == a
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Negate_Involution()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in IntFloat)
                cases.Add(($"-(-a)==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, -(-a)); }));
            foreach (var tc in HalfComplexDecimal)                              // G17
                cases.Add(($"-(-a)==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, -(-a)); }));
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Double })   // G17: strided view
                cases.Add(($"-(-a)==a strided [{tc}]", () => { var a = Arange(16, tc)["::2"]; return BytesEqual(a, -(-a)); }));
            Run("negate involution", cases);
        }

        // (a + b) - b == a  (exact for these finite integer-valued operands)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void AdditiveInverse()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in IntFloat)
                cases.Add(($"(a+b)-b==a [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    var b = (Arange(12, tc).reshape(3, 4) + Arange(1, tc)); // b = a+1, distinct
                    return BytesEqual(a, (a + b) - b);
                }));
            foreach (var tc in HalfComplexDecimal)                              // G17 (values exact)
                cases.Add(($"(a+b)-b==a [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    var b = (Arange(12, tc).reshape(3, 4) + Arange(1, tc));
                    return BytesEqual(a, (a + b) - b);
                }));
            Run("additive inverse", cases);
        }

        // a.T.T == a   (involution, values and shape)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Transpose_Involution()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in AllNumeric)
            {
                cases.Add(($"(a.T).T==a 2d [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a.transpose().transpose()); }));
                cases.Add(($"(a.T).T==a 3d [{tc}]", () => { var a = Arange(24, tc).reshape(2, 3, 4); return BytesEqual(a, a.transpose().transpose()); }));
            }
            foreach (var tc in new[] { NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal, NPTypeCode.Boolean })  // G17
                cases.Add(($"(a.T).T==a 2d [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a.transpose().transpose()); }));
            cases.Add(("(v.T).T==v strided [Double]", () =>                     // G17: strided view
            { var v = Arange(24, NPTypeCode.Double).reshape(4, 6)[":, ::2"]; return BytesEqual(v, v.transpose().transpose()); }));
            Run("transpose involution", cases);
        }

        // a.reshape(flat).reshape(shape) == a   (reshape round-trip preserves C-order)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Reshape_RoundTrip()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in AllNumeric)
                cases.Add(($"reshape round-trip [{tc}]", () =>
                {
                    var a = Arange(24, tc).reshape(2, 3, 4);
                    return BytesEqual(a, a.reshape(24).reshape(2, 3, 4));
                }));
            foreach (var tc in new[] { NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal, NPTypeCode.Boolean })  // G17
                cases.Add(($"reshape round-trip [{tc}]", () =>
                {
                    var a = Arange(24, tc).reshape(2, 3, 4);
                    return BytesEqual(a, a.reshape(24).reshape(2, 3, 4));
                }));
            Run("reshape round-trip", cases);
        }

        // widening cast round-trip: int32 -> int64 -> int32 == a  (lossless)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Cast_WideningRoundTrip()
        {
            var cases = new List<(string, Func<bool>)>
            {
                ("int32->int64->int32", () => { var a = Arange(12, NPTypeCode.Int32).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Int64).astype(NPTypeCode.Int32)); }),
                ("uint8->int32->uint8", () => { var a = Arange(12, NPTypeCode.Byte).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Int32).astype(NPTypeCode.Byte)); }),
                ("int16->int64->int16", () => { var a = Arange(12, NPTypeCode.Int16).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Int64).astype(NPTypeCode.Int16)); }),
                ("single->double->single", () => { var a = Arange(12, NPTypeCode.Single).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Double).astype(NPTypeCode.Single)); }),
                // G17: half widens losslessly into double; bool/char survive an int32 round trip.
                ("half->double->half", () => { var a = Arange(12, NPTypeCode.Half).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Double).astype(NPTypeCode.Half)); }),
                ("bool->int32->bool", () => { var a = Arange(12, NPTypeCode.Boolean).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Int32).astype(NPTypeCode.Boolean)); }),
                ("char->int32->char", () => { var a = Arange(12, NPTypeCode.Char).reshape(3, 4);
                    return BytesEqual(a, a.astype(NPTypeCode.Int32).astype(NPTypeCode.Char)); }),
            };
            Run("widening cast round-trip", cases);
        }

        // a * 1 == a   and   a + 0 == a
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MultiplicativeAndAdditiveIdentity()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in IntFloat)
            {
                cases.Add(($"a*1==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a * np.array(1).astype(tc)); }));
                cases.Add(($"a+0==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a + np.array(0).astype(tc)); }));
            }
            foreach (var tc in HalfComplexDecimal)                              // G17
            {
                cases.Add(($"a*1==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a * np.array(1).astype(tc)); }));
                cases.Add(($"a+0==a [{tc}]", () => { var a = Arange(12, tc).reshape(3, 4); return BytesEqual(a, a + np.array(0).astype(tc)); }));
            }
            Run("identity elements", cases);
        }

        // abs(abs(a)) == abs(a)   (idempotent)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Abs_Idempotent()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Single, NPTypeCode.Double })
                cases.Add(($"abs(abs(a))==abs(a) [{tc}]", () =>
                {
                    var a = (Arange(12, tc).reshape(3, 4) - np.array(6).astype(tc)); // span negatives
                    var aa = np.abs(a);
                    return BytesEqual(aa, np.abs(aa));
                }));
            foreach (var tc in HalfComplexDecimal)                              // G17
                cases.Add(($"abs(abs(a))==abs(a) [{tc}]", () =>
                {
                    var a = (Arange(12, tc).reshape(3, 4) - np.array(6).astype(tc));
                    var aa = np.abs(a);                     // Complex: float64 magnitude, then fixpoint
                    return BytesEqual(aa, np.abs(aa));
                }));
            Run("abs idempotent", cases);
        }

        // sum over all axes == sum of the flattened array (axis order invariance)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void SumAll_Equals_FlatSum()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Double,
                                       NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal })  // G17
                cases.Add(($"sum(a)==sum(ravel(a)) [{tc}]", () =>
                {
                    var a = Arange(24, tc).reshape(2, 3, 4);
                    return BytesEqual(np.sum(a), np.sum(np.ravel(a)));
                }));
            Run("sum-all == flat sum", cases);
        }

        // concatenate([a, b], axis=0) first half == a, second half == b
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Concatenate_SplitFree()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in AllNumeric)
                cases.Add(($"concat halves [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    var b = (Arange(12, tc).reshape(3, 4) + np.array(100).astype(tc));
                    var cat = np.concatenate((a, b), 0);
                    return BytesEqual(a, cat["0:3"]) && BytesEqual(b, cat["3:6"]);
                }));
            foreach (var tc in HalfComplexDecimal)                              // G17
                cases.Add(($"concat halves [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    var b = (Arange(12, tc).reshape(3, 4) + np.array(100).astype(tc));
                    var cat = np.concatenate((a, b), 0);
                    return BytesEqual(a, cat["0:3"]) && BytesEqual(b, cat["3:6"]);
                }));
            Run("concatenate split-free", cases);
        }

        // argsort of an ascending array is the identity index vector 0..n-1
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Argsort_OfSorted_IsIdentity()
        {
            var cases = new List<(string, Func<bool>)>
            {
                ("argsort(arange) int32", () => { var a = Arange(8, NPTypeCode.Int32);
                    var idx = np.argsort<int>(a); var expect = np.arange(8).astype(idx.typecode); return BytesEqual(idx, expect); }),
                ("argsort(arange) double", () => { var a = Arange(8, NPTypeCode.Double);
                    var idx = np.argsort<double>(a); var expect = np.arange(8).astype(idx.typecode); return BytesEqual(idx, expect); }),
                // G17: the untested tails (ascending => identity permutation holds for each).
                ("argsort(arange) half", () => { var a = Arange(8, NPTypeCode.Half);
                    var idx = np.argsort<System.Half>(a); var expect = np.arange(8).astype(idx.typecode); return BytesEqual(idx, expect); }),
                ("argsort(arange) decimal", () => { var a = Arange(8, NPTypeCode.Decimal);
                    var idx = np.argsort<decimal>(a); var expect = np.arange(8).astype(idx.typecode); return BytesEqual(idx, expect); }),
                ("argsort(arange) complex", () => { var a = Arange(8, NPTypeCode.Complex);
                    var idx = np.argsort<System.Numerics.Complex>(a); var expect = np.arange(8).astype(idx.typecode); return BytesEqual(idx, expect); }),
            };
            Run("argsort of sorted is identity", cases);
        }

        // a == a is all-true for finite data (comparison reflexivity)
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Equality_Reflexive()
        {
            var cases = new List<(string, Func<bool>)>();
            foreach (var tc in AllNumeric)
                cases.Add(($"(a==a).all() [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    return (bool)np.all(a == a);
                }));
            foreach (var tc in new[] { NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Decimal,
                                       NPTypeCode.Boolean, NPTypeCode.Char })   // G17
                cases.Add(($"(a==a).all() [{tc}]", () =>
                {
                    var a = Arange(12, tc).reshape(3, 4);
                    return (bool)np.all(a == a);
                }));
            Run("equality reflexive", cases);
        }
    }
}
