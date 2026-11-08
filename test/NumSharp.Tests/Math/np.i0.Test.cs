using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.i0 — modified Bessel function of the first kind, order 0 (I₀), element-wise. All expected
    ///     values are probed from NumPy 2.4.2 (win-amd64).
    ///     <para>
    ///     Result dtype follows the INPUT float precision (NumPy's NEP50 weak-scalar rule): bool / every
    ///     integer width / Char promote to float64; float16/float32/float64 are PRESERVED and the cephes
    ///     routine runs at that precision; Decimal is preserved via the double bridge; Complex is refused.
    ///     Every float dtype is BIT-EXACT with NumPy (gated bit-for-bit by the differential-fuzz "i0"
    ///     tier — verified 0 diffs over 10 518 adversarial float64/float32 inputs and ALL 65 536 float16
    ///     bit patterns). These unit tests pin the headline values and every edge/dtype/layout branch.
    ///     </para>
    /// </summary>
    [TestClass]
    public class NpI0Test
    {
        /// <summary>The four docstring values <c>i0([0,1,2,3]) = [1, 1.2660658777520082, 2.279585302336067, 4.880792585865024]</c>, bit-exact vs NumPy 2.4.2.</summary>
        [TestMethod]
        public void I0_DocstringValues_Float64()
        {
            var r = np.i0(np.array(new[] { 0.0, 1.0, 2.0, 3.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(1.0);
            r.GetAtIndex<double>(1).Should().Be(1.2660658777520082);
            r.GetAtIndex<double>(2).Should().Be(2.279585302336067);
            r.GetAtIndex<double>(3).Should().Be(4.880792585865024);
        }

        /// <summary>The x &lt;= 8 / x &gt; 8 cephes branch split, incl. the boundary itself and the overflow region.</summary>
        [TestMethod]
        public void I0_BranchSplitAndOverflow_Float64()
        {
            var r = np.i0(np.array(new[] { 0.5, 4.0, 5.0, 8.0, 8.001, 10.0, 30.0, 700.0 }));
            r.GetAtIndex<double>(0).Should().Be(1.0634833707413234);
            r.GetAtIndex<double>(1).Should().Be(11.30192195213633);
            r.GetAtIndex<double>(2).Should().Be(27.239871823604442);
            r.GetAtIndex<double>(3).Should().Be(427.56411572180474);   // x == 8.0 → still the x<=8 branch
            r.GetAtIndex<double>(4).Should().Be(427.9641777084088);    // x == 8.001 → the x>8 branch
            r.GetAtIndex<double>(5).Should().Be(2815.716628466254);
            r.GetAtIndex<double>(6).Should().Be(781672297823.9775);
            r.GetAtIndex<double>(7).Should().Be(1.5295933476718735e+302);
        }

        /// <summary>i0 is even — it takes |x| first, so a negative and its magnitude give the same value.</summary>
        [TestMethod]
        public void I0_IsEven_AbsOfInput()
        {
            var r = np.i0(np.array(new[] { -1.0, -2.0, -8.5 }));
            r.GetAtIndex<double>(0).Should().Be(1.2660658777520082);
            r.GetAtIndex<double>(1).Should().Be(2.279585302336067);
            r.GetAtIndex<double>(2).Should().Be(np.i0(np.array(new[] { 8.5 })).GetAtIndex<double>(0));
        }

        /// <summary>±inf and NaN all map to NaN — the x&gt;8 branch computes <c>inf*finite/sqrt(inf) = inf/inf = nan</c>.</summary>
        [TestMethod]
        public void I0_NonFinite_AllNaN()
        {
            var r = np.i0(np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN }));
            double.IsNaN(r.GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNaN(r.GetAtIndex<double>(1)).Should().BeTrue();
            double.IsNaN(r.GetAtIndex<double>(2)).Should().BeTrue();
        }

        /// <summary>
        ///     float32 is PRESERVED and computed IN float32 — <c>i0(0f)</c> is the float just below 1
        ///     (0x3F7FFFFF), NOT 1.0, because the Chebyshev recurrence accumulates float32 rounding.
        ///     A float64-computed-then-cast result would be exactly 1.0f, so this pins native precision.
        /// </summary>
        [TestMethod]
        public void I0_Float32_NativePrecision_BitExact()
        {
            var r = np.i0(np.array(new[] { 0f, 1f, 2f, 3f }));
            r.typecode.Should().Be(NPTypeCode.Single);
            BitConverter.SingleToUInt32Bits(r.GetAtIndex<float>(0)).Should().Be(1065353215u); // 0.99999994f, not 1.0f
            r.GetAtIndex<float>(0).Should().Be(0.9999999403953552f);
            r.GetAtIndex<float>(1).Should().Be(1.266066074371338f);
            r.GetAtIndex<float>(2).Should().Be(2.279585123062134f);
            r.GetAtIndex<float>(3).Should().Be(4.880792617797852f);
        }

        /// <summary>float16 is PRESERVED and computed IN float16 — bit-exact vs NumPy's half loop (probed).</summary>
        [TestMethod]
        public void I0_Float16_NativePrecision()
        {
            var r = np.i0(np.array(new[] { (Half)0, (Half)1, (Half)2, (Half)3 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(1.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(1.265625);
            ((double)r.GetAtIndex<Half>(2)).Should().Be(2.28125);
            ((double)r.GetAtIndex<Half>(3)).Should().Be(4.87890625);
        }

        /// <summary>bool / integer inputs promote to float64 (NumPy's <c>astype(float)</c> for kind != 'f').</summary>
        [TestMethod]
        public void I0_IntAndBool_PromoteToFloat64()
        {
            var ri = np.i0(np.array(new[] { 0, 1, 2, 3 }));
            ri.typecode.Should().Be(NPTypeCode.Double);
            ri.GetAtIndex<double>(1).Should().Be(1.2660658777520082);   // == i0(1.0)

            var rb = np.i0(np.array(new[] { true, false, true }));
            rb.typecode.Should().Be(NPTypeCode.Double);
            rb.GetAtIndex<double>(0).Should().Be(1.2660658777520082);   // i0(1)
            rb.GetAtIndex<double>(1).Should().Be(1.0);                  // i0(0)
        }

        /// <summary>Decimal is a NumSharp extension (no NumPy analog): preserved, computed via the double bridge.</summary>
        [TestMethod]
        public void I0_Decimal_PreservedViaDoubleBridge()
        {
            var r = np.i0(np.array(new[] { 0m, 1m, 2m }));
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(1m);
            ((double)r.GetAtIndex<decimal>(2)).Should().BeApproximately(2.279585302336067, 1e-12);
        }

        /// <summary>A Decimal whose I₀ overflows the decimal range (|x| ≳ 72) throws, like the sibling decimal transcendentals.</summary>
        [TestMethod]
        public void I0_Decimal_Overflow_Throws()
        {
            Action act = () => np.i0(np.array(new[] { 100m }));
            act.Should().Throw<OverflowException>();
        }

        /// <summary>Complex has no I₀ loop — NumPy raises <c>TypeError("i0 not supported for complex values")</c>; NumSharp raises the same message.</summary>
        [TestMethod]
        public void I0_Complex_Throws()
        {
            Action act = () => np.i0(np.array(new[] { new Complex(1, 2) }));
            act.Should().Throw<IncorrectTypeException>().WithMessage("i0 not supported for complex values");
        }

        /// <summary>A null argument is a caller error.</summary>
        [TestMethod]
        public void I0_Null_Throws()
        {
            Action act = () => np.i0((NDArray)null);
            act.Should().Throw<ArgumentNullException>();
        }

        /// <summary>A 0-d input yields a 0-d result; <c>i0(0)</c> is exactly 1.0.</summary>
        [TestMethod]
        public void I0_Scalar_ZeroDimensional()
        {
            var r = np.i0(NDArray.Scalar(0.0));
            r.ndim.Should().Be(0);
            r.GetAtIndex<double>(0).Should().Be(1.0);
        }

        /// <summary>An empty input yields an empty float64 array (shape preserved).</summary>
        [TestMethod]
        public void I0_Empty()
        {
            var r = np.i0(np.array(new double[] { }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>The result has the input's shape (2-D preserved).</summary>
        [TestMethod]
        public void I0_ShapePreserved_2D()
        {
            var r = np.i0(np.zeros(new Shape(2, 3), np.float64));
            r.ndim.Should().Be(2);
            r.shape[0].Should().Be(2);
            r.shape[1].Should().Be(3);
        }

        /// <summary>
        ///     Non-contiguous inputs read through their own strides — a transposed / stepped / reversed /
        ///     broadcast view gives the same per-element result as its contiguous copy (the fused pass is
        ///     layout-agnostic). Values depend only on the element, not its position.
        /// </summary>
        [TestMethod]
        public void I0_NonContiguousLayouts_MatchContiguousCopy()
        {
            var b = (np.arange(24).reshape(4, 6).astype(np.float64)) / 2.0;   // values span the 8.0 branch split
            void Check(NDArray view)
            {
                np.array_equal(np.i0(view), np.i0(view.copy())).Should().BeTrue();
            }
            Check(b.T);                 // F-order / transposed
            Check(b["::2, :"]);         // row step
            Check(b[":, ::2"]);         // column step (non-unit inner stride)
            Check(b["::-1, ::-1"]);     // reversed (negative strides)
            Check(b["1:3, 2:5"]);       // sliced, non-zero offset
            Check(np.broadcast_to(np.arange(4).astype(np.float64).reshape(4, 1), new Shape(4, 6))); // stride-0 broadcast
        }
    }
}
