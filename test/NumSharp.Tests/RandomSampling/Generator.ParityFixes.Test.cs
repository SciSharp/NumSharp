using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity pins for the PCG64 <see cref="Generator"/> fixes verified against NumPy
    ///     2.4.2 during the parity audit: float32-scalar dtype, weighted sampling without replacement,
    ///     <c>permuted</c>, the full uint64 range, <c>shuffle</c> on a 0-d array, and the <c>out=</c>
    ///     validation contract (message/order/F-contiguous acceptance).
    /// </summary>
    [TestClass]
    public class GeneratorParityFixesTest
    {
        // ---- float32 scalar returns a float64 scalar (NumPy float_fill widens for size=None) ----

        [TestMethod]
        public void Random_Float32_Scalar_ReturnsFloat64_ByteExact()
        {
            var a = np.random.default_rng(42).random(dtype: np.float32);
            a.ndim.Should().Be(0);
            a.dtype.Should().Be(typeof(double));
            Convert.ToDouble(a.GetAtIndex(0)).Should().Be(0.08925092220306396);
        }

        [TestMethod]
        public void StandardNormal_Float32_Scalar_ReturnsFloat64_ByteExact()
        {
            var a = np.random.default_rng(42).standard_normal(dtype: np.float32);
            a.dtype.Should().Be(typeof(double));
            Convert.ToDouble(a.GetAtIndex(0)).Should().Be(0.14190717041492462);
        }

        [TestMethod]
        public void StandardExponential_Float32_Scalar_ReturnsFloat64_ByteExact()
        {
            var a = np.random.default_rng(42).standard_exponential(dtype: np.float32);
            a.dtype.Should().Be(typeof(double));
            Convert.ToDouble(a.GetAtIndex(0)).Should().Be(0.08859460800886154);
        }

        [TestMethod]
        public void StandardGamma_Float32_Scalar_ReturnsFloat64_ByteExact()
        {
            var a = np.random.default_rng(42).standard_gamma(2.5, dtype: np.float32);
            a.dtype.Should().Be(typeof(double));
            Convert.ToDouble(a.GetAtIndex(0)).Should().Be(2.3823325634002686);
        }

        [TestMethod]
        public void Random_Float32_Sized_StaysFloat32()
        {
            var a = np.random.default_rng(42).random(new Shape(3), np.float32);
            a.dtype.Should().Be(typeof(float));
            a.ndim.Should().Be(1);
        }

        // ---- weighted sampling without replacement (choice replace=False, p=...) ----

        [TestMethod]
        public void Choice_WithoutReplacement_Weighted_ByteExact()
        {
            var p = np.array(new[] { 0.05, 0.1, 0.15, 0.2, 0.25, 0.1, 0.15 });
            var a = np.random.default_rng(42).choice(7L, new Shape(3), replace: false, p: p);
            a.size.Should().Be(3);
            Convert.ToInt64(a.GetAtIndex(0)).Should().Be(5);
            Convert.ToInt64(a.GetAtIndex(1)).Should().Be(3);
            Convert.ToInt64(a.GetAtIndex(2)).Should().Be(6);

            var b = np.random.default_rng(42).choice(7L, new Shape(5), replace: false, p: p);
            long[] expB = { 5, 3, 6, 4, 1 };
            for (int i = 0; i < 5; i++) Convert.ToInt64(b.GetAtIndex(i)).Should().Be(expB[i]);
        }

        [TestMethod]
        public void Choice_WithoutReplacement_Weighted_FewerNonzeroThanSize_Throws()
        {
            var p = np.array(new[] { 0.0, 0.5, 0.0, 0.5, 0.0 });
            Action act = () => np.random.default_rng(0).choice(5L, new Shape(3), replace: false, p: p);
            act.Should().Throw<ValueError>().WithMessage("Fewer non-zero entries in p than size");
        }

        // ---- permuted ----

        [TestMethod]
        public void Permuted_1D_ByteExact()
        {
            var a = np.random.default_rng(42).permuted(np.arange(6), axis: 0);
            long[] exp = { 3, 2, 5, 4, 1, 0 };
            for (int i = 0; i < 6; i++) Convert.ToInt64(a.GetAtIndex(i)).Should().Be(exp[i]);
        }

        [TestMethod]
        public void Permuted_2D_Axis1_ByteExact()
        {
            var a = np.random.default_rng(42).permuted(np.arange(12).reshape(3, 4), axis: 1);
            long[] exp = { 3, 2, 1, 0, 7, 6, 4, 5, 8, 11, 10, 9 };
            for (int i = 0; i < 12; i++) Convert.ToInt64(a.GetAtIndex(i)).Should().Be(exp[i]);
        }

        [TestMethod]
        public void Permuted_Out_IsInPlace_And_Returned()
        {
            var x = np.arange(12).reshape(3, 4);
            var y = np.random.default_rng(42).permuted(x, axis: 1, @out: x);
            ReferenceEquals(y, x).Should().BeTrue();
            long[] exp = { 3, 2, 1, 0, 7, 6, 4, 5, 8, 11, 10, 9 };
            for (int i = 0; i < 12; i++) Convert.ToInt64(x.GetAtIndex(i)).Should().Be(exp[i]);
        }

        // ---- full uint64 range (ulong overload) ----

        [TestMethod]
        public void Integers_UInt64_FullRange_ByteExact()
        {
            var a = np.random.default_rng(42).integers(0UL, ulong.MaxValue, new Shape(4), np.uint64, endpoint: true);
            a.dtype.Should().Be(typeof(ulong));
            ulong[] exp = { 14276969152011380360UL, 8095878257575067585UL, 15838336090824644132UL, 12864169557245331597UL };
            for (int i = 0; i < 4; i++) Convert.ToUInt64(a.GetAtIndex(i)).Should().Be(exp[i]);
        }

        [TestMethod]
        public void Integers_UInt64_UpperHalf_ByteExact()
        {
            var a = np.random.default_rng(42).integers(9223372036854775808UL, ulong.MaxValue, new Shape(4), np.uint64, endpoint: true);
            ulong[] exp = { 16361856612860465988UL, 13271311165642309600UL, 17142540082267097874UL, 15655456815477441606UL };
            for (int i = 0; i < 4; i++) Convert.ToUInt64(a.GetAtIndex(i)).Should().Be(exp[i]);
        }

        [TestMethod]
        public void Integers_UInt64_LargeHigh_NonUInt64Dtype_Throws()
        {
            Action act = () => np.random.default_rng(0).integers(0UL, ulong.MaxValue, new Shape(4), np.int64, endpoint: true);
            act.Should().Throw<ValueError>().WithMessage("high is out of bounds for int64");
        }

        // ---- shuffle on a 0-d array raises NumPy's len() TypeError ----

        [TestMethod]
        public void Shuffle_ZeroD_ThrowsTypeError()
        {
            Action act = () => np.random.default_rng(0).shuffle(np.array(5));
            act.Should().Throw<TypeError>().WithMessage("len() of unsized object");
        }

        // ---- out= validation (NumPy check_output contract) ----

        [TestMethod]
        public void Random_Out_NonContiguous_ThrowsNumpyMessage()
        {
            var noncontig = np.empty(new Shape(6), np.float64)["::2"];
            Action act = () => np.random.default_rng(0).random(new Shape(3), null, noncontig);
            act.Should().Throw<ValueError>()
               .WithMessage("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.");
        }

        [TestMethod]
        public void Random_Out_ContiguityCheckedBeforeDtype()
        {
            // A non-contiguous AND wrong-dtype out reports the contiguity error first (NumPy order).
            var badBoth = np.empty(new Shape(6), np.float32)["::2"];
            Action act = () => np.random.default_rng(0).random(new Shape(3), np.float64, badBoth);
            act.Should().Throw<ValueError>()
               .WithMessage("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.");
        }
    }
}
