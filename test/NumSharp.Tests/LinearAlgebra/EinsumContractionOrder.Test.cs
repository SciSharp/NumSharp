using System;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The default <c>np.einsum</c> (<c>optimize=False</c>) contracts pairwise through <see cref="np.matmul"/> /
    ///     BLAS, where NumPy's <c>c_einsum</c> multiplies every operand of a term left to right and adds the terms in
    ///     iteration order: a documented divergence the owner chose to keep on 2026-11-12 ("Keep the BLAS route", the
    ///     OptunaSharp brief's item 3). The values agree to a few ULP; the bits of a summed contraction are not NumPy's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Why kept: a bit-exact default would port <c>c_einsum</c> and its sum-of-products kernels (the 2- and 3-operand
    ///     specializations unroll their sums) and give up BLAS on the default path. NumPy's own default is slow for large
    ///     contractions (<c>'ij,jk->ik'</c> over 500 x 500: 29.7 ms, against 2.0 ms with <c>optimize=True</c>), and
    ///     NumPy's <c>optimize=True</c> itself contracts pairwise and differs from its default in the last bits (26 of the
    ///     60 elements below). NumSharp's default is NumPy's <c>optimize=</c> structure for every value of <c>optimize</c>.
    ///     </para>
    ///     <para>
    ///     The example is Optuna NSGA-III's reference-point association, <c>"ni,pi,p,pm->npm"</c>, over formula inputs;
    ///     the expected bits are NumPy 2.4.2's default einsum, which equals the left-to-right sum over i of
    ///     <c>((a[n,i] * b[p,i]) * c[p]) * d[p,m]</c>. On the session's host NumSharp matched 19 of 60 elements bit for bit
    ///     and every element within 2 ULP; the bound below allows another BLAS backend's order. The zero-sign half of
    ///     this divergence (a bare product of a -0.0 is +0.0 in NumPy's zero-filled output) and the output layout
    ///     (QB-45) stay open pins of their own.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class EinsumContractionOrderTests
    {
        /// <summary>NumPy 2.4.2's <c>np.einsum("ni,pi,p,pm->npm", o, r, n2, r)</c> for the inputs below, as float64 bits in C
        ///     order (shape (4, 5, 3)).</summary>
        private static readonly ulong[] NumPyBits =
        {
            0x3FBB3F86A0472EF0, 0x3FE2730327303273, 0x3FB22A59C02F74A1, 0x3FD57C2F8EBDD169, 0x3FE4138491F8B715,
            0x3FD4138491F8B716, 0x3FE12F4589A0C677, 0x3FD08F4C516C6225, 0x3FE08F4C516C622A, 0x3FD394DCFC3204ED,
            0x3FE461784A684E44, 0x3FD1FBA65FC57238, 0x3FE1334C799D224B, 0x3FCCC8C91F1BDB5D, 0x3FE07F927F195521,
            0x3FBADDF42406A4B5, 0x3FE230F2A30F2A32, 0x3FB1E94D6D59C325, 0x3FDC5F2B233D4B1A, 0x3FEA82E529527456,
            0x3FDA82E529527456,
            0x3FED8E5FFE0EAEF4, 0x3FDC7B3BEB828356, 0x3FEC7B3BEB82835E, 0x3FD92D43B8266F82, 0x3FEA34562828011E,
            0x3FD71F1ED8234C42, 0x3FEE2822F5F98A30, 0x3FD93BB4C35DDC22, 0x3FECED0769529EA8, 0x3FAF59C1F59C1F4C,
            0x3FD53A1B53A1B532, 0x3FA4E6814E6814E0, 0x3FD3C0AFDEB343DE, 0x3FE27519D87D8AF3, 0x3FD27519D87D8AF4,
            0x3FE6ACCE1DB11F88, 0x3FD5D9B8A83FA0B2, 0x3FE5D9B8A83FA0B8, 0x3FD14D4AD58C1263, 0x3FE202146B53185C,
            0x3FCFC76F53FC0CD6, 0x3FE741CA2CED3118, 0x3FD375BE10B19194, 0x3FE64EC725DE4933, 0x3FCF8538BD3FDA97,
            0x3FF5578915789157, 0x3FC5037B28D53C66, 0x3FE40E46354344F2, 0x3FF2BD99BC430E0F, 0x3FE2BD99BC430E10,
            0x3FE911F5AC00120B, 0x3FD828944D4F31F4, 0x3FE828944D4F31FB, 0x3FE2BBC8940D763D, 0x3FF37F879F51EDFF,
            0x3FE1344A7D8486B0, 0x3FE8AA14D71EF487, 0x3FD4A336025DD1D8, 0x3FE7A85D21EEABD8,
        };

        /// <summary>
        ///     NumSharp's default einsum of the NSGA-III association is within 4 ULP of NumPy's <c>c_einsum</c> in every
        ///     element (the documented order divergence: pairwise contraction, not NumPy's sum of products).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void DefaultEinsum_ContractsPairwise_WithinAFewUlpOfNumPysSumOfProducts()
        {
            // The formula inputs NumPy's bits were computed from: o (4, 3), r (5, 3) and n2 = 1 / sum(r * r, axis=1).
            var o = new double[12];
            for (int i = 0; i < 12; i++) o[i] = (i * 0.37 + 0.11) % 1.3;
            var r = new double[15];
            for (int i = 0; i < 15; i++) r[i] = (i * 0.53 + 0.07) % 1.1 + 0.05;
            using var O = np.array(o).reshape(4, 3);
            using var R = np.array(r).reshape(5, 3);
            using var rr = R * R;
            using var sums = np.sum(rr, axis: 1);
            using var N2 = 1.0 / sums;

            using var e = np.einsum("ni,pi,p,pm->npm", O, R, N2, R);
            e.shape.Should().Equal(4L, 5L, 3L);
            for (int i = 0; i < NumPyBits.Length; i++)
            {
                long got = BitConverter.DoubleToInt64Bits(e.GetAtIndex<double>(i));
                System.Math.Abs(got - (long)NumPyBits[i]).Should().BeLessThanOrEqualTo(4L, $"element {i} is NumPy's value to a few ULP");
            }
        }
    }
}
