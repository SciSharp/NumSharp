using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    [TestClass]
    public class np_clip_test
    {
        [TestMethod]
        public void Case1()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, 3, max).Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case2()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, max, null).Should().BeOfValues(8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case3()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, null, max).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }

        // Regression for W6-D: np.clip PROPAGATES NaN in NumPy (clip(NaN, lo, hi) = NaN), it does NOT
        // clamp NaN to a_min. NumSharp clips via the SIMD min/max kernel whose hardware MAXPS/MINPD
        // dropped the NaN; the NaN-aware float path now restores propagation. Verified vs NumPy 2.4.2.

        private static void AssertElems(NDArray actual, double[] expected, string because)
        {
            actual.size.Should().Be(expected.Length, because);
            var f = actual.astype(NPTypeCode.Double);
            for (int i = 0; i < expected.Length; i++)
            {
                double v = f.GetDouble(i);
                if (double.IsNaN(expected[i]))
                    double.IsNaN(v).Should().BeTrue($"element {i} should be NaN ({because})");
                else
                    v.Should().Be(expected[i], $"element {i} ({because})");
            }
        }

        [TestMethod]
        public void Clip_ScalarBounds_PropagatesNaN()
        {
            double nan = double.NaN;
            var d = np.array(new double[] { nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, nan, 11 });
            // NumPy: np.clip(d, 2, 5) = [nan, 2, 2, 3, 4, 5, 5, 5, 5, 5, nan, 5]
            AssertElems(np.clip(d, (NDArray)2.0, (NDArray)5.0),
                new double[] { nan, 2, 2, 3, 4, 5, 5, 5, 5, 5, nan, 5 },
                "clip(NaN, lo, hi) must preserve NaN, not clamp to a_min");
        }

        [TestMethod]
        public void Clip_ArrayBounds_PropagatesNaN()
        {
            double nan = double.NaN;
            // NaN in the value AND in both bounds (the clip fuzz-op shape, array a_min/a_max).
            var a = np.array(new double[] { nan, 1, 5, nan, 9, 2, nan, 8, 3, 10, 0, nan });
            var lo = np.array(new double[] { 0, 2, nan, 1, 3, nan, 4, 0, nan, 5, 1, 2 });
            var hi = np.array(new double[] { 3, nan, 7, 8, nan, 6, 9, nan, 5, 8, nan, 9 });
            // NumPy: every element with a NaN in value/lo/hi -> NaN; only index 9 (10 clipped to [5,8]) -> 8.
            AssertElems(np.clip(a, lo, hi),
                new double[] { nan, nan, nan, nan, nan, nan, nan, nan, nan, 8, nan, nan },
                "clip with array bounds must propagate NaN from value or either bound");
        }

        [TestMethod]
        public void Clip_OutAlias_PropagatesNaN()
        {
            double nan = double.NaN;
            var a = np.array(new double[] { nan, 1, 5, nan, 9, 2, nan, 8, 3, 10, 0, nan });
            var lo = np.array(new double[] { 0, 2, nan, 1, 3, nan, 4, 0, nan, 5, 1, 2 });
            var hi = np.array(new double[] { 3, nan, 7, 8, nan, 6, 9, nan, 5, 8, nan, 9 });
            np.clip(a, lo, hi, a); // out = a (aliases the input)
            AssertElems(a,
                new double[] { nan, nan, nan, nan, nan, nan, nan, nan, nan, 8, nan, nan },
                "clip(a, lo, hi, out=a) must propagate NaN through the aliased out= write");
        }

        // Regression: np.clip on a BOOLEAN array works on the NON-CONTIGUOUS path (transposed /
        // strided / F-order / reversed), matching NumPy 2.4.2. Previously the strided clip kernel
        // (ClipStrided) threw "clip not supported for Boolean" while the contiguous path worked —
        // it omitted the Boolean case. Bool now rides the Byte Min/Max kernel there (bool storage
        // is 0/1, so unsigned Byte ordering reproduces false<true). Also fuzz-gated: gen_oracle
        // CLIP_DTYPES now includes bool across every stat layout. Values verified vs NumPy 2.4.2.
        [TestMethod]
        public void Clip_Bool_Transposed()
        {
            var a = np.array(new bool[] { true, false, false, true, true, false }).reshape(2, 3).T; // (3,2), non-contiguous
            // clip(a, True, True) -> all True (lo=hi=True saturates every element).
            var allTrue = np.clip(a, NDArray.Scalar(true), NDArray.Scalar(true));
            allTrue.typecode.Should().Be(NPTypeCode.Boolean);
            allTrue.Should().BeOfValues(true, true, true, true, true, true).And.BeShaped(3, 2);

            // Identity clip (False, True) leaves values unchanged — this verifies the strided read
            // maps each element to the right C-order slot (a.T in C-order = [T,T,F,T,F,F]).
            np.clip(a, NDArray.Scalar(false), NDArray.Scalar(true))
                .Should().BeOfValues(true, true, false, true, false, false).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Clip_Bool_Strided()
        {
            var a = np.array(new bool[] { true, false, false, true, true, false, false, true })["::2"]; // [T,F,T,F]
            var allTrue = np.clip(a, NDArray.Scalar(true), NDArray.Scalar(true));
            allTrue.typecode.Should().Be(NPTypeCode.Boolean);
            allTrue.Should().BeOfValues(true, true, true, true).And.BeShaped(4);

            // Identity (False, True): the strided [::2] view [T,F,T,F] passes through unchanged.
            np.clip(a, NDArray.Scalar(false), NDArray.Scalar(true))
                .Should().BeOfValues(true, false, true, false).And.BeShaped(4);
        }

        // ---- Rarer non-contiguous bool clip edge cases (every value verified vs NumPy 2.4.2) ----

        [TestMethod]
        public void Clip_Bool_Strided_ArrayBounds()
        {
            // Per-element ARRAY bounds on a strided view — a NON-constant result, the strongest check
            // that BOTH the strided source AND the strided-paired bounds resolve the right C-order
            // slots. NumPy: clip([T,F,T,F], lo=[F,T,F,T], hi=[T,T,F,F]) = [T,T,F,F].
            var a = np.array(new bool[] { true, false, false, true, true, false, false, true })["::2"]; // [T,F,T,F]
            var lo = np.array(new bool[] { false, true, false, true });
            var hi = np.array(new bool[] { true, true, false, false });
            np.clip(a, lo, hi).Should().BeOfValues(true, true, false, false).And.BeShaped(4);
        }

        [TestMethod]
        public void Clip_Bool_BroadcastBound_Transposed()
        {
            // A (1,)-shaped lo bound broadcasts (stride=0) across the transposed (3,2) view; hi=True.
            // lo=False, hi=True -> identity. NumPy C-order: [T,T,F,T,F,F].
            var a = np.array(new bool[] { true, false, false, true, true, false }).reshape(2, 3).T;
            np.clip(a, np.array(new bool[] { false }), NDArray.Scalar(true))
                .Should().BeOfValues(true, true, false, true, false, false).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Clip_Bool_MinGreaterThanMax_AllFalse()
        {
            // clip applies Max(v, lo) THEN Min(., hi); lo=True > hi=False saturates every element to
            // False (NumPy min>max semantics), through non-contiguous views.
            var a = np.array(new bool[] { true, false, false, true, true, false }).reshape(2, 3).T;
            np.clip(a, NDArray.Scalar(true), NDArray.Scalar(false))
                .Should().BeOfValues(false, false, false, false, false, false).And.BeShaped(3, 2);
            var s = np.array(new bool[] { true, false, false, true })["::2"]; // [T,F]
            np.clip(s, NDArray.Scalar(true), NDArray.Scalar(false)).Should().BeOfValues(false, false).And.BeShaped(2);
        }

        [TestMethod]
        public void Clip_Bool_MinOnly_MaxOnly_Transposed()
        {
            var a = np.array(new bool[] { true, false, false, true, true, false }).reshape(2, 3).T; // C-order [T,T,F,T,F,F]
            // min-only clip(a, True, null) = Max(v, True) = all True.
            np.clip(a, NDArray.Scalar(true), null)
                .Should().BeOfValues(true, true, true, true, true, true).And.BeShaped(3, 2);
            // max-only clip(a, null, False) = Min(v, False) = all False.
            np.clip(a, null, NDArray.Scalar(false))
                .Should().BeOfValues(false, false, false, false, false, false).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Clip_Bool_3D_Transposed_Identity()
        {
            // Higher-rank (4,2,3) transposed view; identity clip verifies the multi-axis strided read.
            // arange(24).reshape(2,3,4).astype(bool) is all True except element 0; transpose(2,0,1).
            var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Boolean).transpose(new int[] { 2, 0, 1 });
            var r = np.clip(a, NDArray.Scalar(false), NDArray.Scalar(true));
            r.typecode.Should().Be(NPTypeCode.Boolean);
            r.Should().BeShaped(4, 2, 3);
            r.GetBoolean(0, 0, 0).Should().BeFalse("only original [0,0,0]=0 is False");
            r.GetBoolean(0, 0, 1).Should().BeTrue();
            r.GetBoolean(3, 1, 2).Should().BeTrue("last element");
        }

        [TestMethod]
        public void Clip_Bool_NegativeStride_2D_Identity()
        {
            // Doubly-reversed 2-D view. NumPy: reshape(2,3)[::-1,::-1] C-order = [F,T,F,T,F,T]; identity.
            var a = np.array(new bool[] { true, false, true, false, true, false }).reshape(2, 3)["::-1,::-1"];
            np.clip(a, NDArray.Scalar(false), NDArray.Scalar(true))
                .Should().BeOfValues(false, true, false, true, false, true).And.BeShaped(2, 3);
        }

        [TestMethod]
        public void Clip_Bool_InPlace_StridedOut()
        {
            // out= aliases a STRIDED view: clip(False, False) writes False back through the ::2
            // positions of the base (odd positions untouched). NumPy base: [F,F,F,T,F,F,F,T].
            var basev = np.array(new bool[] { true, false, false, true, true, false, false, true });
            var v = basev["::2"]; // strided view over indices 0,2,4,6 = [T,F,T,F]
            np.clip(v, NDArray.Scalar(false), NDArray.Scalar(false), v); // out = v (aliases base)
            basev.Should().BeOfValues(false, false, false, true, false, false, false, true).And.BeShaped(8);
        }

        [TestMethod]
        public void Clip_Bool_Empty_And_Singleton_NonContiguous()
        {
            // Empty transposed bool -> empty result, dtype preserved.
            var empty = np.clip(np.zeros(new Shape(0, 3), NPTypeCode.Boolean).T,
                NDArray.Scalar(false), NDArray.Scalar(true));
            empty.typecode.Should().Be(NPTypeCode.Boolean);
            empty.size.Should().Be(0);
            empty.Should().BeShaped(3, 0);
            // 1-element strided view.
            np.clip(np.array(new bool[] { true, false })["::2"], NDArray.Scalar(true), NDArray.Scalar(true))
                .Should().BeOfValues(true).And.BeShaped(1);
        }
    }
}
