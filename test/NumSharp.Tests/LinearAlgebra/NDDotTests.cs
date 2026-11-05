using System;
using System.Security.Cryptography;
using NumSharp.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     NumPy 2.4.2's bits for np.dot over its N-D route, where every output is one call of the dtype's dot function
    ///     (<c>FLOAT_dot</c> / <c>DOUBLE_dot</c> / <c>CDOUBLE_dot</c>) and the stride class of the contracted runs picks the
    ///     arithmetic (<see cref="NDDot"/>): two unit strides reach scipy-openblas' Haswell kernels, other positive strides
    ///     cblas' strided loops, a reversed or broadcast run the dot function's own loop. The expected values come from
    ///     NumPy (scipy-openblas at one thread, an AVX2 host) over operands drawn from an integer generator both sides
    ///     reproduce exactly (<see cref="Lcg"/>).
    /// </summary>
    /// <remarks>
    ///     Each row: the operands' dtypes, the stride class, the contracted length K, the rhs's last dimension M (3 makes
    ///     one call per output; 5 a row of four plus one, the row walk), the generator seed, whether NaN / inf / -inf /
    ///     -0.0 are planted, whether the rhs is 3-D, NumPy's result dtype, and the expectation: <c>sha:</c> and the first
    ///     32 hex digits of SHA-256 over the result's bytes, or <c>hex:</c> and every byte (special rows, where a NaN slot
    ///     compares as NaN for NaN: which NaN survives when several meet follows RyuJIT's operand order, see
    ///     <see cref="NDDot"/>). Generator: the PA3a scratch script gen_nddot_tests.py of the np-layer dtype plan.
    /// </remarks>
    [TestClass]
    public class NDDotTests : TestClass
    {
        /// <summary>
        ///     The test generator: a 64-bit LCG whose draws become doubles with a random sign, an exponent in [-10, 10] and
        ///     a full mantissa, the same integer arithmetic as the Python script that recorded NumPy's answers.
        /// </summary>
        private sealed class Lcg
        {
            private ulong _s;

            /// <summary>Starts the stream at <paramref name="seed"/>.</summary>
            /// <param name="seed">The row's seed.</param>
            public Lcg(ulong seed) => _s = seed;

            /// <summary>Advances the state one step (Knuth's MMIX constants).</summary>
            /// <returns>The new state.</returns>
            private ulong Step() => _s = unchecked(_s * 6364136223846793005UL + 1442695040888963407UL);

            /// <summary>One double: sign from the first draw's top bit, exponent from its bits 52..56 (mod 21) minus 10,
            ///     mantissa the second draw's top 52 bits.</summary>
            /// <returns>The value.</returns>
            public double Value()
            {
                ulong a = Step();
                ulong b = Step();
                ulong sign = a >> 63;
                long e = (long)((a >> 52) & 0x1F) % 21 - 10;
                ulong mant = b >> 12;
                return BitConverter.Int64BitsToDouble((long)((sign << 63) | ((ulong)(e + 1023) << 52) | mant));
            }

            /// <summary>One integer in [-1000, 1000]: the draw's top 24 bits modulo 2001, minus 1000.</summary>
            /// <returns>The value.</returns>
            public long Integer() => (long)((Step() >> 40) % 2001) - 1000;

            /// <summary>An array of <paramref name="dtype"/> filled in C order (complex: real draw, then imaginary).
            ///     float32 values are the doubles rounded to float32, as NumPy's astype rounds them; integer, bool and
            ///     float16 values are exact in their dtype (an <see cref="Integer"/>, the draw's top bit, an
            ///     <see cref="Integer"/> over 8), so no conversion rounding has to agree between the two sides.</summary>
            /// <param name="shape">The shape.</param>
            /// <param name="dtype">float32, float64, complex128, int32, int64, bool or float16.</param>
            /// <returns>The fresh C-contiguous array.</returns>
            public NDArray Array(long[] shape, string dtype)
            {
                long n = 1;
                foreach (long d in shape)
                    n *= d;
                switch (dtype)
                {
                    case "int32":
                    {
                        var v = new int[n];
                        for (long i = 0; i < n; i++)
                            v[i] = (int)Integer();
                        return np.array(v).reshape(shape);
                    }
                    case "int64":
                    {
                        var v = new long[n];
                        for (long i = 0; i < n; i++)
                            v[i] = Integer();
                        return np.array(v).reshape(shape);
                    }
                    case "bool":
                    {
                        var v = new bool[n];
                        for (long i = 0; i < n; i++)
                            v[i] = (Step() >> 63) == 1;
                        return np.array(v).reshape(shape);
                    }
                    case "float16":
                    {
                        var v = new Half[n];
                        for (long i = 0; i < n; i++)
                            v[i] = (Half)(Integer() / 8.0);
                        return np.array(v).reshape(shape);
                    }
                    case "complex128":
                    {
                        var v = new System.Numerics.Complex[n];
                        for (long i = 0; i < n; i++)
                        {
                            double re = Value();
                            double im = Value();
                            v[i] = new System.Numerics.Complex(re, im);
                        }
                        return np.array(v).reshape(shape);
                    }
                    case "float32":
                    {
                        var v = new float[n];
                        for (long i = 0; i < n; i++)
                            v[i] = (float)Value();
                        return np.array(v).reshape(shape);
                    }
                    default:
                    {
                        var v = new double[n];
                        for (long i = 0; i < n; i++)
                            v[i] = Value();
                        return np.array(v).reshape(shape);
                    }
                }
            }
        }

        /// <summary>Plants NaN, -0.0, inf and -inf at flat positions 1, 2, 5 and 9 of a C-contiguous array (the
        ///     positions the generator plants them at; complex: in the real part, imaginary 0).</summary>
        /// <param name="x">The array (C-contiguous, written in place).</param>
        private static void Specials(NDArray x)
        {
            var flat = x.reshape(-1);
            void Set(long i, double v)
            {
                if (x.typecode == NPTypeCode.Complex) flat.SetAtIndex(new System.Numerics.Complex(v, 0), i);
                else if (x.typecode == NPTypeCode.Single) flat.SetAtIndex((float)v, i);
                else flat.SetAtIndex(v, i);
            }
            Set(1, BitConverter.Int64BitsToDouble(0x7FF8000000000000));   // np.nan is the POSITIVE quiet NaN; .NET's double.NaN is negative
            Set(2, -0.0);
            if (x.size > 7) Set(5, double.PositiveInfinity);
            if (x.size > 11) Set(9, double.NegativeInfinity);
        }

        /// <summary>
        ///     NumPy 2.4.2's bits for np.dot over its N-D route (a 3-D first operand), one row per stride class, contracted
        ///     length, row length and dtype pair (see the class remarks for the columns).
        /// </summary>
        /// <param name="dtypeA">The first operand's dtype.</param>
        /// <param name="dtypeB">The second operand's dtype.</param>
        /// <param name="kind">unit (<c>b.T</c>: two unit runs), strided (a C-ordered b), rev (<c>a[..., ::-1]</c>), bcast
        ///     (b broadcast along K: stride 0).</param>
        /// <param name="K">The contracted length.</param>
        /// <param name="M">The rhs's last dimension (3: one call per output; 5: a row of four plus one).</param>
        /// <param name="seed">The generator seed.</param>
        /// <param name="special">Whether NaN / inf / -inf / -0.0 are planted.</param>
        /// <param name="rhs3d">Whether the rhs is 3-D (the row walk advances an outer axis).</param>
        /// <param name="resultDtype">NumPy's result dtype.</param>
        /// <param name="expected">sha: or hex: and NumPy's result bytes.</param>
        [DataTestMethod]
        [DataRow("float32", "float32", "unit", 3, 3, 1001UL, false, false, "float32", "sha:5ca3dc4f637c702e04a3a1a83a3dc083")]
        [DataRow("float32", "float32", "unit", 3, 5, 1002UL, false, false, "float32", "sha:99439d6aae2639c54826d80c8121b550")]
        [DataRow("float32", "float32", "unit", 17, 3, 1003UL, false, false, "float32", "sha:f767e08902815b40a5bed4268d4f4a19")]
        [DataRow("float32", "float32", "unit", 17, 5, 1004UL, false, false, "float32", "sha:8e0f3332c5ef94bda0dcbe88a1bc5a6d")]
        [DataRow("float32", "float32", "unit", 31, 3, 1005UL, false, false, "float32", "sha:2ac63ce8af9fb5c287a3285b1ce107b3")]
        [DataRow("float32", "float32", "unit", 31, 5, 1006UL, false, false, "float32", "sha:05e3012aacdf4eb5a9290892c82537e4")]
        [DataRow("float32", "float32", "unit", 33, 3, 1007UL, false, false, "float32", "sha:8681be8f56ecaa809778e2f42b71f0a8")]
        [DataRow("float32", "float32", "unit", 33, 5, 1008UL, false, false, "float32", "sha:4491c821b42acaa4a37d8f7af18bd1a5")]
        [DataRow("float32", "float32", "unit", 32, 5, 1009UL, true, false, "float32", "hex:0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f540a89c63658ed485fae72456fe44cc80000c07f289a60c450c106c3ec6a984289b382c50000c07f1a83e7c47d0da8c42820e4419a4facc70000c07f32ffbc47f73dcdc38be18e465b33e5470000c07fc983054979356946141521c6bcaaac47")]
        [DataRow("float32", "float32", "strided", 3, 3, 1010UL, false, false, "float32", "sha:f3e934441a2dde981926ad30747a7e14")]
        [DataRow("float32", "float32", "strided", 3, 5, 1011UL, false, false, "float32", "sha:27ba13d7e6508a1de08fb870b15fe973")]
        [DataRow("float32", "float32", "strided", 17, 3, 1012UL, false, false, "float32", "sha:9e6c04d46581794caa01195f11db286f")]
        [DataRow("float32", "float32", "strided", 17, 5, 1013UL, false, false, "float32", "sha:e229f1d807715b194c5cc10dc3d4a725")]
        [DataRow("float32", "float32", "strided", 31, 3, 1014UL, false, false, "float32", "sha:1eeed27de2072e24a4fa1ba3480d9a2c")]
        [DataRow("float32", "float32", "strided", 31, 5, 1015UL, false, false, "float32", "sha:bc0800821eef7277c98753749625497d")]
        [DataRow("float32", "float32", "strided", 33, 3, 1016UL, false, false, "float32", "sha:a554e701102e2908d3be95930957714e")]
        [DataRow("float32", "float32", "strided", 33, 5, 1017UL, false, false, "float32", "sha:4e3f5a28d1c36eecd8daa81ddb160f26")]
        [DataRow("float32", "float32", "strided", 32, 5, 1018UL, true, false, "float32", "hex:0000c07f0000c07f0000c07f0000c07f0000c07f000080ff0000c07fd98c4047899fa5470000807f000080ff0000c07fda9f8bc4ebcfe8c60000807f000080ff0000c07fa56f3ac8c7069f470000807f000080ff0000c07f46933249b45ea9490000807f0000807f0000c07fee4e86c77bd8abc7000080ff")]
        [DataRow("float32", "float32", "rev", 3, 3, 1019UL, false, false, "float32", "sha:7ca3b3dbbd096b10ae45970c0982905c")]
        [DataRow("float32", "float32", "rev", 3, 5, 1020UL, false, false, "float32", "sha:6e45ae809578ba3ff095e0c9f45687fe")]
        [DataRow("float32", "float32", "rev", 17, 3, 1021UL, false, false, "float32", "sha:dfe5c4c8b509685ccdcf2a103106f938")]
        [DataRow("float32", "float32", "rev", 17, 5, 1022UL, false, false, "float32", "sha:b4d6d81f2a168a190aae2aa4b780d0c4")]
        [DataRow("float32", "float32", "rev", 31, 3, 1023UL, false, false, "float32", "sha:fcd468f91d901406ba43e98d87f1c013")]
        [DataRow("float32", "float32", "rev", 31, 5, 1024UL, false, false, "float32", "sha:080c965b99fd055b3a27cf54e33a0e86")]
        [DataRow("float32", "float32", "rev", 33, 3, 1025UL, false, false, "float32", "sha:511f13f2d8d7df99a46c5ec24fd1ba26")]
        [DataRow("float32", "float32", "rev", 33, 5, 1026UL, false, false, "float32", "sha:3e4b3b2c8593cc46256f7c4b523435fe")]
        [DataRow("float32", "float32", "rev", 32, 5, 1027UL, true, false, "float32", "hex:0000c0ff0000c07f0000c0ff0000c0ff0000c0ff0000807f0000c07fb405b8c3b1e332c9000080ff0000807f0000c07fdc3b24c4ca9cb746000080ff000080ff0000c07f6e984846481b25440000807f0000807f0000c07f4543e5c5c8f7b546000080ff0000807f0000c07f52844cc9cf4087c6000080ff")]
        [DataRow("float32", "float32", "bcast", 3, 3, 1028UL, false, false, "float32", "sha:b87691ef72b25d62ad57235eccc047ee")]
        [DataRow("float32", "float32", "bcast", 3, 5, 1029UL, false, false, "float32", "sha:076a462e9f37c89f3eee9a46dc3b51d4")]
        [DataRow("float32", "float32", "bcast", 17, 3, 1030UL, false, false, "float32", "sha:3a57b5da32a92afc43d34e95cb9fc5fe")]
        [DataRow("float32", "float32", "bcast", 17, 5, 1031UL, false, false, "float32", "sha:b637be8a234c5604d3239a88141b92c5")]
        [DataRow("float32", "float32", "bcast", 31, 3, 1032UL, false, false, "float32", "sha:c810814b3a21ccd8998b500fda50c9cc")]
        [DataRow("float32", "float32", "bcast", 31, 5, 1033UL, false, false, "float32", "sha:a6e72a93b8c139ca45b814e9c39017e0")]
        [DataRow("float32", "float32", "bcast", 33, 3, 1034UL, false, false, "float32", "sha:6fe9ee29929a77031cc5cc7762c87487")]
        [DataRow("float32", "float32", "bcast", 33, 5, 1035UL, false, false, "float32", "sha:cdfea7143e9981fcb30e54c3a43bd7fc")]
        [DataRow("float32", "float32", "bcast", 32, 5, 1036UL, true, false, "float32", "hex:0000c07f0000c07f0000c07f0000c07f0000c07f5fd108488f1039448ba8c44523b7b8c3dff7b6c32605e9c687981dc3e577a7c4564c9d4275cf9b426ce49945ff28d0415d335d438ec44fc169cd4dc13cfe0348df89324431b9bd459c33b2c31e84b0c326a531c7fa4970c3a557ffc4e5d5ef422991ed42")]
        [DataRow("float32", "float32", "strided", 17, 6, 1037UL, false, true, "float32", "sha:1606c776b56af7e77f6703e04ebbbb6f")]
        [DataRow("float64", "float64", "unit", 3, 3, 1038UL, false, false, "float64", "sha:44a7ac78dccb5041a8e93c7d6b300ee6")]
        [DataRow("float64", "float64", "unit", 3, 5, 1039UL, false, false, "float64", "sha:023076fd53a1e97f28c7ba7acc28c34d")]
        [DataRow("float64", "float64", "unit", 17, 3, 1040UL, false, false, "float64", "sha:d20853fbc8eb9be7123f40ed7d493a7b")]
        [DataRow("float64", "float64", "unit", 17, 5, 1041UL, false, false, "float64", "sha:1740b32b4778be3459e8153be46fb625")]
        [DataRow("float64", "float64", "unit", 31, 3, 1042UL, false, false, "float64", "sha:9d56fde74a470b575456ddb016e4289e")]
        [DataRow("float64", "float64", "unit", 31, 5, 1043UL, false, false, "float64", "sha:47579911413348b128bcac93aefbba3b")]
        [DataRow("float64", "float64", "unit", 33, 3, 1044UL, false, false, "float64", "sha:34b0080a80caecc689128edb88b25cc0")]
        [DataRow("float64", "float64", "unit", 33, 5, 1045UL, false, false, "float64", "sha:5fb239b851d0a30a1fff8fbeab1d7f82")]
        [DataRow("float64", "float64", "unit", 32, 5, 1046UL, true, false, "float64", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87fa4ace1ad5fc040c0636403c3423e204107f3816a9fac1741915ad28e6c4d1bc1000000000000f87f518be1be84f3f5c0eb14ca8d13c7c840f095412ee9bdfbc0792698eeaebf1141000000000000f87ffcfcc11cabf1bec0ba508d366df5f3c010e9a2da89764dc1147a0ab08adef3c0000000000000f87f7623719e6ad75f405bad6bc70759ec4017a42954595403416a775743b35cac40000000000000f87fee46e19eb3c1b440b83958a5a79483c08e4cb3b1e4820c41e0e73e878f88cfc0")]
        [DataRow("float64", "float64", "strided", 3, 3, 1047UL, false, false, "float64", "sha:33cf17daedb015e1dc69630c3375e979")]
        [DataRow("float64", "float64", "strided", 3, 5, 1048UL, false, false, "float64", "sha:9f77e85fb49d4a5d184ed6b0107dbfad")]
        [DataRow("float64", "float64", "strided", 17, 3, 1049UL, false, false, "float64", "sha:355e1c6fb69776746fdedd5b510ad6c1")]
        [DataRow("float64", "float64", "strided", 17, 5, 1050UL, false, false, "float64", "sha:4262b38e8bf1ff01c6da548d1332c2b5")]
        [DataRow("float64", "float64", "strided", 31, 3, 1051UL, false, false, "float64", "sha:3908cc3d0bbc2d3d8b72d56c2e793e64")]
        [DataRow("float64", "float64", "strided", 31, 5, 1052UL, false, false, "float64", "sha:3ac93fc9cd5706b9b75513e75cfe71b3")]
        [DataRow("float64", "float64", "strided", 33, 3, 1053UL, false, false, "float64", "sha:65f2f47591bfe3762192559c82676733")]
        [DataRow("float64", "float64", "strided", 33, 5, 1054UL, false, false, "float64", "sha:26cf8c99b55d83d224b02c020e4928e4")]
        [DataRow("float64", "float64", "strided", 32, 5, 1055UL, true, false, "float64", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f0ff000000000000f87f1bf82598ec46ea40dd9e4b42f908b0c0000000000000f07f000000000000f0ff000000000000f87f9048507c619f25412c97c8fea0bfd1c0000000000000f07f000000000000f0ff000000000000f87fb012473f5c471141d2a15dfc2ee2f1c0000000000000f07f000000000000f07f000000000000f87f8ccb18d90c65d8409374b48bfa80e1c0000000000000f0ff000000000000f0ff000000000000f87f3bf624a9bc93b3c0f8b4c03d3b2df540000000000000f07f")]
        [DataRow("float64", "float64", "rev", 3, 3, 1056UL, false, false, "float64", "sha:a9160f8299187d5533418fc5edc9858f")]
        [DataRow("float64", "float64", "rev", 3, 5, 1057UL, false, false, "float64", "sha:66e6050d37cfe9a239cb8d4e88904771")]
        [DataRow("float64", "float64", "rev", 17, 3, 1058UL, false, false, "float64", "sha:007f4ff4e9855811584e2835324d9de1")]
        [DataRow("float64", "float64", "rev", 17, 5, 1059UL, false, false, "float64", "sha:fd758c5e9294d2ee90556993136b0ed4")]
        [DataRow("float64", "float64", "rev", 31, 3, 1060UL, false, false, "float64", "sha:2f201daa4505467aa151a0d2f074268a")]
        [DataRow("float64", "float64", "rev", 31, 5, 1061UL, false, false, "float64", "sha:30ff338acc6c49683e1813f3ff1bf3d8")]
        [DataRow("float64", "float64", "rev", 33, 3, 1062UL, false, false, "float64", "sha:ac1124dc387d6f7528a0bebe91328034")]
        [DataRow("float64", "float64", "rev", 33, 5, 1063UL, false, false, "float64", "sha:acaa42a7bb31077db32f9034a1c4343e")]
        [DataRow("float64", "float64", "rev", 32, 5, 1064UL, true, false, "float64", "hex:000000000000f8ff000000000000f87f000000000000f87f000000000000f87f000000000000f8ff000000000000f0ff000000000000f87f6e53a9cfa8858d40760b5453e3a33641000000000000f07f000000000000f0ff000000000000f87fda308adcb140d5c0ca33a18811911041000000000000f07f000000000000f0ff000000000000f87f517f03e87ce7edc0afbbcd963c5c5840000000000000f07f000000000000f0ff000000000000f87f0a13d8e7ce662d4124102caf76692141000000000000f07f000000000000f0ff000000000000f87fd4d9fe993f54f040803feb56f6e6b1c0000000000000f07f")]
        [DataRow("float64", "float64", "bcast", 3, 3, 1065UL, false, false, "float64", "sha:a29226c2ebf28d93b45d3084c35ec056")]
        [DataRow("float64", "float64", "bcast", 3, 5, 1066UL, false, false, "float64", "sha:b228a59611b9a7cb1822d49f9d861499")]
        [DataRow("float64", "float64", "bcast", 17, 3, 1067UL, false, false, "float64", "sha:5b19f6f20a70456a848b36f6eb11a85d")]
        [DataRow("float64", "float64", "bcast", 17, 5, 1068UL, false, false, "float64", "sha:2e68b8ce0db8a14027ce0cdfd0b9ab4a")]
        [DataRow("float64", "float64", "bcast", 31, 3, 1069UL, false, false, "float64", "sha:59c4fdae880511d60937aa7636a01600")]
        [DataRow("float64", "float64", "bcast", 31, 5, 1070UL, false, false, "float64", "sha:eca0297e7dfca6a151617982e1d0df45")]
        [DataRow("float64", "float64", "bcast", 33, 3, 1071UL, false, false, "float64", "sha:cb65af2116abebbd84fed4e76983a3e5")]
        [DataRow("float64", "float64", "bcast", 33, 5, 1072UL, false, false, "float64", "sha:0187c7c69fb61d544e38e7bd327575b0")]
        [DataRow("float64", "float64", "bcast", 32, 5, 1073UL, true, false, "float64", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f83b1faf73428944073291810c1769040a503ce49ce4ac440c1d1ee7a5c060241a559397d4ad83540b8d24bb90d9b8c4082038d00535d87401725097f27ccbc40822033266c94f940f7064abe3d002f4054f5ad0de26297c063822f7bf71993c0ca59a155068bc7c0d73a8a7b97e904c15f664f18305839c0edcc2da69d9d9e404716fabe9b0199406fe36ba62ad2ce40959cbdeb8d600b412e5be0e0f19640406ad8f64cb42382c000e9cf2dc5a17dc03f0fa50fd742b2c0b724de827e38f0c0483684238aa823c0")]
        [DataRow("float64", "float64", "strided", 17, 6, 1074UL, false, true, "float64", "sha:9a62abc80f7d4983251007074c5a0657")]
        [DataRow("complex128", "complex128", "unit", 3, 3, 1075UL, false, false, "complex128", "sha:3053fb3f8b55c730da8007d33f5cb863")]
        [DataRow("complex128", "complex128", "unit", 3, 5, 1076UL, false, false, "complex128", "sha:1e5f6e988e324211ba0185bb2e902115")]
        [DataRow("complex128", "complex128", "unit", 17, 3, 1077UL, false, false, "complex128", "sha:9ef980c3de4461e4f462f84f9ff81338")]
        [DataRow("complex128", "complex128", "unit", 17, 5, 1078UL, false, false, "complex128", "sha:11cb5284f06a96f9d5ddcf7fb67b25f8")]
        [DataRow("complex128", "complex128", "unit", 31, 3, 1079UL, false, false, "complex128", "sha:244691aa969a3ede3cdcaa663612bcaa")]
        [DataRow("complex128", "complex128", "unit", 31, 5, 1080UL, false, false, "complex128", "sha:33ee11b49894d21537d9e200524d1a3c")]
        [DataRow("complex128", "complex128", "unit", 33, 3, 1081UL, false, false, "complex128", "sha:97826bbae86701846aa9f8d74a7a0bd1")]
        [DataRow("complex128", "complex128", "unit", 33, 5, 1082UL, false, false, "complex128", "sha:14b9afac0ebe646f5e7efa72232a711e")]
        [DataRow("complex128", "complex128", "unit", 32, 5, 1083UL, true, false, "complex128", "hex:000000000000f87f000000000000f8ff000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f0d916cc1cdcce240f03fc0bf541dfcc0b038fd5980ed13c13c55c1e0b57e2841bb8073134e59e2407c1be21a513bc2c043f3d08df354f0c0e3f16021704c9c40000000000000f87f000000000000f87f770f3e08dab7e5c0d068d99a6c508140987fcdce8bbee340756140723ef154c1d5eb58a1ac9ac6c08be7d3933a5b0641302d655307d74541af9c4547824f34c1000000000000f87f000000000000f87f6dfc41e6ac99ac404806cf4375fcbac04c5c6b417b3a12c175980dc5cc4635411b053d363613cec05e0e92de5fd0bcc08b75d9185096fac0659e8686671f3441000000000000f87f000000000000f87f5d027c6d696822c1c41a18dbf451dbc04f56911a3cedf6c01c994e3d1e95d9c05c2994f8da96e0407b432b3e05d3d240dd8abf631ee5f440c19fcb6b1744dcc0000000000000f87f000000000000f87f38350b108d60e54034bd6ea1604de9406f06b9c1cee70241703573a1c7a41841e83205e41bd8d34005f1e7edbc5a82c0d45591b5ee88d140d83dd0777c7d1a41")]
        [DataRow("complex128", "complex128", "strided", 3, 3, 1084UL, false, false, "complex128", "sha:8e0d86c45b9111871a464fe55787e2bf")]
        [DataRow("complex128", "complex128", "strided", 3, 5, 1085UL, false, false, "complex128", "sha:9edab2cd08a9740cc8cb5d2b7b74bf80")]
        [DataRow("complex128", "complex128", "strided", 17, 3, 1086UL, false, false, "complex128", "sha:afc7e1cab8ae1067182189e049fea0d1")]
        [DataRow("complex128", "complex128", "strided", 17, 5, 1087UL, false, false, "complex128", "sha:09c1a024a60fd1d39d799e586f5c37ee")]
        [DataRow("complex128", "complex128", "strided", 31, 3, 1088UL, false, false, "complex128", "sha:cf4287af6bea06761e21dda72a907ebe")]
        [DataRow("complex128", "complex128", "strided", 31, 5, 1089UL, false, false, "complex128", "sha:1f94368260c8853887181b1e5cb664f4")]
        [DataRow("complex128", "complex128", "strided", 33, 3, 1090UL, false, false, "complex128", "sha:8da17402dbb08214afecc606d7accd55")]
        [DataRow("complex128", "complex128", "strided", 33, 5, 1091UL, false, false, "complex128", "sha:5694b967226f26b1d9398fc6682da73e")]
        [DataRow("complex128", "complex128", "strided", 32, 5, 1092UL, true, false, "complex128", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f8ff000000000000f0ff000000000000f87f000000000000f87fc31ff97fc7ae054194d7d0aab79b0341d8149f3d81f6104112838c8c266decc0000000000000f8ff000000000000f07f000000000000f8ff000000000000f0ff000000000000f87f000000000000f87fc152f7554eebf740c818bfdd02e03cc17807a38ee13ec940e6562f002960f4c0000000000000f8ff000000000000f07f000000000000f8ff000000000000f07f000000000000f87f000000000000f87febeb18fa8e310141251a5e1faed114c1c4e0cb4d1e9b2141cb76513e6d5edf40000000000000f8ff000000000000f0ff000000000000f8ff000000000000f0ff000000000000f87f000000000000f87fe9a224774280f040e6d8bf597fe3f1407935ca239223c4c046ed19f3266c0a41000000000000f8ff000000000000f07f000000000000f8ff000000000000f0ff000000000000f87f000000000000f87f00ddb8b5e3229740ce677774c7c74c41b9f915b0e8051341570e600c84dc0141000000000000f8ff000000000000f07f")]
        [DataRow("complex128", "complex128", "rev", 3, 3, 1093UL, false, false, "complex128", "sha:93fa950ac97ab8eaa06fc9077bbbbf6d")]
        [DataRow("complex128", "complex128", "rev", 3, 5, 1094UL, false, false, "complex128", "sha:b0c6a88b1f2e31e386b158c8858d7125")]
        [DataRow("complex128", "complex128", "rev", 17, 3, 1095UL, false, false, "complex128", "sha:a576f9dd970450f3572103f7839ead4f")]
        [DataRow("complex128", "complex128", "rev", 17, 5, 1096UL, false, false, "complex128", "sha:870d356456c2c5b1c0c6a2bcdd400c1e")]
        [DataRow("complex128", "complex128", "rev", 31, 3, 1097UL, false, false, "complex128", "sha:b4a3a9712e2128bf95bcec1a75b54d29")]
        [DataRow("complex128", "complex128", "rev", 31, 5, 1098UL, false, false, "complex128", "sha:880ba04966d0d56c65eaf9eac20a84d1")]
        [DataRow("complex128", "complex128", "rev", 33, 3, 1099UL, false, false, "complex128", "sha:ca7760a8965f7b65310661da831ec244")]
        [DataRow("complex128", "complex128", "rev", 33, 5, 1100UL, false, false, "complex128", "sha:d142ed3ad29899d3d7fcc8e59667988b")]
        [DataRow("complex128", "complex128", "rev", 32, 5, 1101UL, true, false, "complex128", "hex:000000000000f8ff000000000000f8ff000000000000f87f000000000000f87f000000000000f8ff000000000000f8ff000000000000f8ff000000000000f87f000000000000f8ff000000000000f8ff000000000000f07f000000000000f0ff000000000000f87f000000000000f87fcb0b4c21e086c3c0107e54539f6613c1766b532fedae25412c53a6ad624618c1000000000000f0ff000000000000f07f000000000000f07f000000000000f07f000000000000f87f000000000000f87febde6a8ed3b3d040fb07a273c2f41bc1b03129574af3c240e845ea96b75fd3c0000000000000f0ff000000000000f0ff000000000000f0ff000000000000f0ff000000000000f87f000000000000f87f8cf9195ad0ce42415f5101b44867a4401ba941f705d636c17ff1d302f80c10c1000000000000f07f000000000000f07f000000000000f0ff000000000000f07f000000000000f87f000000000000f87fef3cb90059afbcc0a7a94b116838e2c097b80974ca7304c1c60cfe74c754fd40000000000000f07f000000000000f0ff000000000000f07f000000000000f0ff000000000000f87f000000000000f87fe7d3574f6e9507c11a6c813af247bf4043089591dc9ac4c067a516c34386f540000000000000f0ff000000000000f07f")]
        [DataRow("complex128", "complex128", "bcast", 3, 3, 1102UL, false, false, "complex128", "sha:9320d092f79cf69ca51a678486376588")]
        [DataRow("complex128", "complex128", "bcast", 3, 5, 1103UL, false, false, "complex128", "sha:afec3866d851a678352aa0d276697ac9")]
        [DataRow("complex128", "complex128", "bcast", 17, 3, 1104UL, false, false, "complex128", "sha:8a8d1c0ffcf088ebad91f9fbea16a634")]
        [DataRow("complex128", "complex128", "bcast", 17, 5, 1105UL, false, false, "complex128", "sha:f0c02cdd63446400c6fbf29ef7a0e808")]
        [DataRow("complex128", "complex128", "bcast", 31, 3, 1106UL, false, false, "complex128", "sha:3ca156b7cd65f56b52bb9a4d897dd84b")]
        [DataRow("complex128", "complex128", "bcast", 31, 5, 1107UL, false, false, "complex128", "sha:9b922e29ca7ed803623155a95719fdbe")]
        [DataRow("complex128", "complex128", "bcast", 33, 3, 1108UL, false, false, "complex128", "sha:3966c2a2cf88c3b807b2b0e6ea116185")]
        [DataRow("complex128", "complex128", "bcast", 33, 5, 1109UL, false, false, "complex128", "sha:9ec45c3fde6d9058f896724332be1050")]
        [DataRow("complex128", "complex128", "bcast", 32, 5, 1110UL, true, false, "complex128", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f000000000000f87f2daa7b8eadf18f405264e5c661ba9240066282387e6cd0c0c29497435b90cc401fdd62c8412541c1c13cc03ac15042412728fe2b4e3beb4025f0a1d97db8f540751034c7768d86c0981b291ddb90abc0dd39b2591d747740b91736315b6394c0019a2229cb13d24080a60cf1650bb4407b13f805d75145410c833b05e7e31c4142fb09c92945e2409a2338c13a1ef4c0232a3d188821a1c069b267631c2aa140f5e1297e680c92c0b1adb5385938ac4062f0246e8407e9c0e205baf623f5cec086c3037ffe9c5dc1c829bbddb6ac37c1f67664662229fbc08b71636261b50b410958affff39eb840accded029d41b7c0b2c43743b2a6824088d2e7fd7e119340e9124a25fdc7d0c016183966dfd0c040b2a6ee1cbc3f42c103eddc20a73e3741fc365288eedcda40782cfbeb1510f540cb540d0bbd386340c4b3e96a123ca8c0f733035a731a9240d17b15fad03690c04684aa373dfbcc4003a314d91abdcf403fa1ff76457d42411510f981317c4041c83a02801a1df5401d50a2608febebc08845f58c0a13abc03248e92cf8e68840")]
        [DataRow("complex128", "complex128", "strided", 17, 6, 1111UL, false, true, "complex128", "sha:08bc56030b2a357b7ce1b847d529b178")]
        [DataRow("float32", "float64", "strided", 20, 5, 1112UL, false, false, "float64", "sha:1cda958492e9d63dd706d4de9b52a969")]
        [DataRow("float64", "float32", "strided", 20, 5, 1113UL, false, false, "float64", "sha:bfece5d930be0c2ba187bffab0945269")]
        [DataRow("float32", "complex128", "strided", 20, 5, 1114UL, false, false, "complex128", "sha:15a2a5abf24c8e9b646bc241601093e5")]
        public void NDRoute_MatchesNumPyBits(string dtypeA, string dtypeB, string kind, int K, int M, ulong seed,
            bool special, bool rhs3d, string resultDtype, string expected)
        {
            // The operands exactly as the generator built them: a (2, 3, K), then b by stride class, then the specials.
            var g = new Lcg(seed);
            var a = g.Array(new long[] { 2, 3, K }, dtypeA);
            NDArray b, bv;
            switch (kind)
            {
                case "unit":
                    b = g.Array(new long[] { M, K }, dtypeB);
                    bv = b.T;                                    // both contracted runs at unit stride
                    break;
                case "bcast":
                    b = g.Array(new long[] { 1, M }, dtypeB);
                    bv = np.broadcast_to(b, new Shape(K, M));    // the contracted run has stride 0: the own loop
                    break;
                default:
                    b = g.Array(rhs3d ? new long[] { 2, K, M } : new long[] { K, M }, dtypeB);
                    bv = b;                                      // the contracted run at stride M (or reversed lhs)
                    break;
            }
            if (special)
            {
                Specials(a);
                if (kind != "bcast")
                    Specials(b);
            }
            NDArray av = kind == "rev" ? a[":, :, ::-1"] : a;

            AssertNumPyBits(np.dot(av, bv), resultDtype, expected, $"{dtypeA}.{dtypeB} {kind} K={K} M={M}");
        }

        /// <summary>
        ///     NumPy 2.4.2's bits for np.dot of two 1-D arrays, the route of <c>cblas_matrixproduct</c> for float32 /
        ///     float64 / complex128 (both operands converted to their common dtype first; a length-1 pair is the plain
        ///     product; an operand with a negative or zero stride is copied, which the managed route reads in place;
        ///     the dtype's dot function computes the rest) and of the dot function itself for the others.
        /// </summary>
        /// <param name="dtypeA">The first operand's dtype.</param>
        /// <param name="dtypeB">The second operand's dtype.</param>
        /// <param name="kind">c (both contiguous), rev (<c>a[::-1]</c>), revb (<c>b[::-1]</c>), step (<c>a[::2]</c>,
        ///     <c>b[::2]</c>: cblas' strided loops), stepr (<c>a[::2]</c>, <c>b[::-1]</c>: one copied operand beside a
        ///     stepped one), bcast (b broadcast from one element: stride 0), rev2 (<c>a[::-2]</c>: the one layout the
        ///     route copies).</param>
        /// <param name="n">The length.</param>
        /// <param name="seed">The generator seed.</param>
        /// <param name="special">Whether NaN / inf / -inf / -0.0 are planted.</param>
        /// <param name="resultDtype">NumPy's result dtype.</param>
        /// <param name="expected">sha: or hex: and NumPy's result bytes (see the class remarks).</param>
        [DataTestMethod]
        [DataRow("float32", "float32", "c", 1, 5001UL, false, "float32", "sha:99d0d37b1947e031c876063648bc02bd")]
        [DataRow("float32", "float32", "c", 2, 5002UL, false, "float32", "sha:7b6dc596154ca47898694551d47aa76c")]
        [DataRow("float32", "float32", "c", 7, 5003UL, false, "float32", "sha:a3038459cd33de59e64b27a0276a4240")]
        [DataRow("float32", "float32", "c", 15, 5004UL, false, "float32", "sha:65fd377636adc9cc8a626dc388ddc669")]
        [DataRow("float32", "float32", "c", 16, 5005UL, false, "float32", "sha:326d07cbde70945e8a2488359178b108")]
        [DataRow("float32", "float32", "c", 17, 5006UL, false, "float32", "sha:4aeca6db8f715e41f8b912487af4d300")]
        [DataRow("float32", "float32", "c", 31, 5007UL, false, "float32", "sha:003640d53e41603352c7d5b3bd731277")]
        [DataRow("float32", "float32", "c", 33, 5008UL, false, "float32", "sha:5e423580b6f99efd630cfa523211d171")]
        [DataRow("float32", "float32", "c", 40, 5009UL, false, "float32", "sha:0c98c3bbaea9b9efb231889eb23f45be")]
        [DataRow("float32", "float32", "rev", 1, 5010UL, false, "float32", "sha:9ae7b4c46e91d6ae579176d87431492f")]
        [DataRow("float32", "float32", "rev", 2, 5011UL, false, "float32", "sha:2e126115966a4dca640ab392f8ce0ea9")]
        [DataRow("float32", "float32", "rev", 7, 5012UL, false, "float32", "sha:31662d9de81de505e994256d6f2c8243")]
        [DataRow("float32", "float32", "rev", 15, 5013UL, false, "float32", "sha:b4b586e41720a573619cd8864b65e198")]
        [DataRow("float32", "float32", "rev", 16, 5014UL, false, "float32", "sha:5511e4af89c6d074670b060d4d59bb5c")]
        [DataRow("float32", "float32", "rev", 17, 5015UL, false, "float32", "sha:30c45fbfe42986d711520e2c08268dfc")]
        [DataRow("float32", "float32", "rev", 31, 5016UL, false, "float32", "sha:d403af0a446dea61f11e2a975913c4b2")]
        [DataRow("float32", "float32", "rev", 33, 5017UL, false, "float32", "sha:d990408891fc7415ae037e2468b14970")]
        [DataRow("float32", "float32", "rev", 40, 5018UL, false, "float32", "sha:90962c7990c70a2e26207b4bb49de778")]
        [DataRow("float32", "float32", "revb", 1, 5019UL, false, "float32", "sha:fda6c3a69c185df4615e63c08baf1abd")]
        [DataRow("float32", "float32", "revb", 2, 5020UL, false, "float32", "sha:6d91bc279e2bfdb48717cd499b14579f")]
        [DataRow("float32", "float32", "revb", 7, 5021UL, false, "float32", "sha:29a46f65dd0a9dcd508981af3ed3ad33")]
        [DataRow("float32", "float32", "revb", 15, 5022UL, false, "float32", "sha:1f0ecbda42ef6a720fea342e5d63e0ed")]
        [DataRow("float32", "float32", "revb", 16, 5023UL, false, "float32", "sha:8ee32f070c07f34ad88f4c78a6e8f088")]
        [DataRow("float32", "float32", "revb", 17, 5024UL, false, "float32", "sha:96a3d2fb07fb5f4c842319058e812fb3")]
        [DataRow("float32", "float32", "revb", 31, 5025UL, false, "float32", "sha:6a32cb9d1877df05633f9fe749348a88")]
        [DataRow("float32", "float32", "revb", 33, 5026UL, false, "float32", "sha:0808986e2f86bb634503868f9a4b48d5")]
        [DataRow("float32", "float32", "revb", 40, 5027UL, false, "float32", "sha:1c84fce842433f02b2a6e01b6c274c9c")]
        [DataRow("float32", "float32", "step", 1, 5028UL, false, "float32", "sha:64b8f4e4d1e9ebe91c4999252f92d8fa")]
        [DataRow("float32", "float32", "step", 2, 5029UL, false, "float32", "sha:a98cb2698e511ab3661645226e7c3682")]
        [DataRow("float32", "float32", "step", 7, 5030UL, false, "float32", "sha:216b701751488bc7705fcf19f99e8c78")]
        [DataRow("float32", "float32", "step", 15, 5031UL, false, "float32", "sha:b2a65dfc8a9733797d87ba8bb68649e6")]
        [DataRow("float32", "float32", "step", 16, 5032UL, false, "float32", "sha:8b62cc6e0130fcf16d0badef0e95c53a")]
        [DataRow("float32", "float32", "step", 17, 5033UL, false, "float32", "sha:f1f5c4b9d8315ba798c550de59d7eac2")]
        [DataRow("float32", "float32", "step", 31, 5034UL, false, "float32", "sha:616be0f44e4439a80c57aa8c097c20b1")]
        [DataRow("float32", "float32", "step", 33, 5035UL, false, "float32", "sha:28f6f05b6e225303319b389b50c7a817")]
        [DataRow("float32", "float32", "step", 40, 5036UL, false, "float32", "sha:a2c806e657cc87ec4547c71488134ac6")]
        [DataRow("float32", "float32", "stepr", 1, 5037UL, false, "float32", "sha:7475e0549861ef9a8d2a0616a60bb0ee")]
        [DataRow("float32", "float32", "stepr", 2, 5038UL, false, "float32", "sha:7defa38c1dcd48ba34595566c902db86")]
        [DataRow("float32", "float32", "stepr", 7, 5039UL, false, "float32", "sha:ad22d3c0a11a83d1cad247318ab3ed94")]
        [DataRow("float32", "float32", "stepr", 15, 5040UL, false, "float32", "sha:f57724c483140ddba88ca8a03fb1165c")]
        [DataRow("float32", "float32", "stepr", 16, 5041UL, false, "float32", "sha:694f150856b16f6bbc4b0ddd0f9ce50e")]
        [DataRow("float32", "float32", "stepr", 17, 5042UL, false, "float32", "sha:aa5b5a53a5a3d2d17342bd4091ffe110")]
        [DataRow("float32", "float32", "stepr", 31, 5043UL, false, "float32", "sha:d807e9f3340474c4b4850ded4aa3d70a")]
        [DataRow("float32", "float32", "stepr", 33, 5044UL, false, "float32", "sha:fbc518cee5b8d30b7a23527ee2cae4e8")]
        [DataRow("float32", "float32", "stepr", 40, 5045UL, false, "float32", "sha:4bb815dda70e35d37b8aacd7d5df5319")]
        [DataRow("float32", "float32", "bcast", 1, 5046UL, false, "float32", "sha:845100e77872b088561835359d5fdfc1")]
        [DataRow("float32", "float32", "bcast", 2, 5047UL, false, "float32", "sha:0697da7f01b8909310aa88a6896d3505")]
        [DataRow("float32", "float32", "bcast", 7, 5048UL, false, "float32", "sha:4ac1d08073cd39d34bca5294dc4d10c7")]
        [DataRow("float32", "float32", "bcast", 15, 5049UL, false, "float32", "sha:9228297c354713ac4c21c7a04d4fb1df")]
        [DataRow("float32", "float32", "bcast", 16, 5050UL, false, "float32", "sha:bb9fc659d6b3d3df08017b3c9919bead")]
        [DataRow("float32", "float32", "bcast", 17, 5051UL, false, "float32", "sha:70266f67dc2c38481c1bd8a0d31a2284")]
        [DataRow("float32", "float32", "bcast", 31, 5052UL, false, "float32", "sha:4f3873c8c2c8ea88b2e345d14439e1e6")]
        [DataRow("float32", "float32", "bcast", 33, 5053UL, false, "float32", "sha:2d6afe8b0060f3efbb9565c58435a195")]
        [DataRow("float32", "float32", "bcast", 40, 5054UL, false, "float32", "sha:445bf4682cebc456c8446a8fde8b96c3")]
        [DataRow("float32", "float32", "rev2", 1, 5055UL, false, "float32", "sha:84a6eff995e9d1538fd8109889a4567e")]
        [DataRow("float32", "float32", "rev2", 2, 5056UL, false, "float32", "sha:a6c51a45bf9231549dc06d4fc007f1ff")]
        [DataRow("float32", "float32", "rev2", 7, 5057UL, false, "float32", "sha:45c10c58cb7d9f3a0d13c62cc314c2f7")]
        [DataRow("float32", "float32", "rev2", 15, 5058UL, false, "float32", "sha:d44572562d2ed1d16c9c807095a929df")]
        [DataRow("float32", "float32", "rev2", 16, 5059UL, false, "float32", "sha:94ed4b38d002f8b6739daab7721a8dd9")]
        [DataRow("float32", "float32", "rev2", 17, 5060UL, false, "float32", "sha:d5aa2af56d1620455d522a01801e4e10")]
        [DataRow("float32", "float32", "rev2", 31, 5061UL, false, "float32", "sha:f10ec769b89e3c20ecf87ebd32bf056a")]
        [DataRow("float32", "float32", "rev2", 33, 5062UL, false, "float32", "sha:4a53f9baf3e9a8aa4e408c8f2dc69611")]
        [DataRow("float32", "float32", "rev2", 40, 5063UL, false, "float32", "sha:ec1690e0edb174a69456fe5942bc0603")]
        [DataRow("float32", "float32", "c", 32, 5064UL, true, "float32", "hex:0000c07f")]
        [DataRow("float32", "float32", "rev", 32, 5065UL, true, "float32", "hex:0000c07f")]
        [DataRow("float32", "float32", "bcast", 32, 5066UL, true, "float32", "hex:0000c07f")]
        [DataRow("float64", "float64", "c", 1, 5067UL, false, "float64", "sha:784ba200b4c6a4236481b1474cc5f5d2")]
        [DataRow("float64", "float64", "c", 2, 5068UL, false, "float64", "sha:9fbfcc7a6cd99a296e7ac40c9414e9d8")]
        [DataRow("float64", "float64", "c", 7, 5069UL, false, "float64", "sha:7329342ab6e087fcb2a0608124c105e7")]
        [DataRow("float64", "float64", "c", 15, 5070UL, false, "float64", "sha:ecd4516f21cc3fa9b6709197b88cfa5b")]
        [DataRow("float64", "float64", "c", 16, 5071UL, false, "float64", "sha:dfc1778a20f6c9a7e0d3bb28f2b9201c")]
        [DataRow("float64", "float64", "c", 17, 5072UL, false, "float64", "sha:b0241ac18a9beb0b7a8b02e26266f023")]
        [DataRow("float64", "float64", "c", 31, 5073UL, false, "float64", "sha:845aa411805e2a44f244ad31d2f380ed")]
        [DataRow("float64", "float64", "c", 33, 5074UL, false, "float64", "sha:ba55611232c277955546df0b392e3848")]
        [DataRow("float64", "float64", "c", 40, 5075UL, false, "float64", "sha:4d2a9f46c189c4bc55edaa26359810fd")]
        [DataRow("float64", "float64", "rev", 1, 5076UL, false, "float64", "sha:4ad6aa1f2b621e4fa321b022853ad9e3")]
        [DataRow("float64", "float64", "rev", 2, 5077UL, false, "float64", "sha:2460fc20ee54edd7ff1843a9aa39b182")]
        [DataRow("float64", "float64", "rev", 7, 5078UL, false, "float64", "sha:9c7e2d5f067c24955a02c2aa8cbcc578")]
        [DataRow("float64", "float64", "rev", 15, 5079UL, false, "float64", "sha:006e3edfffb33cfc93240ce92b632b25")]
        [DataRow("float64", "float64", "rev", 16, 5080UL, false, "float64", "sha:72b17607c7e74286b73b19e96152e083")]
        [DataRow("float64", "float64", "rev", 17, 5081UL, false, "float64", "sha:12c6a45cb0a075c80169efbbbbe872ba")]
        [DataRow("float64", "float64", "rev", 31, 5082UL, false, "float64", "sha:6ba8c8f42496e9ac7ccea849e622da73")]
        [DataRow("float64", "float64", "rev", 33, 5083UL, false, "float64", "sha:f9b781a4c3d8b0bc85d47058c294ddeb")]
        [DataRow("float64", "float64", "rev", 40, 5084UL, false, "float64", "sha:3d3efbaa68024d7568ce57ed568fb84e")]
        [DataRow("float64", "float64", "revb", 1, 5085UL, false, "float64", "sha:6ae0624a4028a3eb68b7c50f5a250a23")]
        [DataRow("float64", "float64", "revb", 2, 5086UL, false, "float64", "sha:02b581d21297a12fa36282f97dcd6277")]
        [DataRow("float64", "float64", "revb", 7, 5087UL, false, "float64", "sha:5b99c154b5906b5fb0371fb39e52cba6")]
        [DataRow("float64", "float64", "revb", 15, 5088UL, false, "float64", "sha:6bb79356d9d0003262d0773cf9fac857")]
        [DataRow("float64", "float64", "revb", 16, 5089UL, false, "float64", "sha:140c45ba846dca0ef708e341399c140a")]
        [DataRow("float64", "float64", "revb", 17, 5090UL, false, "float64", "sha:2be29779593cbb683a871cdfc70edd7f")]
        [DataRow("float64", "float64", "revb", 31, 5091UL, false, "float64", "sha:55ae23c5bdfd5e3433287aaa4d766f70")]
        [DataRow("float64", "float64", "revb", 33, 5092UL, false, "float64", "sha:b91cba854e5efa00e0636b1e9d5243f0")]
        [DataRow("float64", "float64", "revb", 40, 5093UL, false, "float64", "sha:763be166846263595c5dc549bcb676c3")]
        [DataRow("float64", "float64", "step", 1, 5094UL, false, "float64", "sha:fba4b8800979f0b6ad52d51b536a5bb7")]
        [DataRow("float64", "float64", "step", 2, 5095UL, false, "float64", "sha:1d83dacd4e8cc6dcc8d6b2d0dc913a4c")]
        [DataRow("float64", "float64", "step", 7, 5096UL, false, "float64", "sha:0c07f948dd7bdb1dd55da6917c07551b")]
        [DataRow("float64", "float64", "step", 15, 5097UL, false, "float64", "sha:550ed722b4d75afca3f40d5c049f693e")]
        [DataRow("float64", "float64", "step", 16, 5098UL, false, "float64", "sha:e613f5ba3e9da3f6cb18ddd168d25a7b")]
        [DataRow("float64", "float64", "step", 17, 5099UL, false, "float64", "sha:efa4ea8d69d1905fbdeba51a852a9a37")]
        [DataRow("float64", "float64", "step", 31, 5100UL, false, "float64", "sha:d9038b151fe7af815c3c139e7815685b")]
        [DataRow("float64", "float64", "step", 33, 5101UL, false, "float64", "sha:b77cf849db8419412c583027a8ee9b9a")]
        [DataRow("float64", "float64", "step", 40, 5102UL, false, "float64", "sha:eca9bbf7f705acbc24ffc57f50607913")]
        [DataRow("float64", "float64", "stepr", 1, 5103UL, false, "float64", "sha:1ab6ffc95e1e5042922739b7c6c9da53")]
        [DataRow("float64", "float64", "stepr", 2, 5104UL, false, "float64", "sha:f4cdb77d0f9b27be7d13b93fe610008a")]
        [DataRow("float64", "float64", "stepr", 7, 5105UL, false, "float64", "sha:db262d5394678f083cbe721ab0a22cf5")]
        [DataRow("float64", "float64", "stepr", 15, 5106UL, false, "float64", "sha:71e2edd17e3e69db4873c3e9c94545d8")]
        [DataRow("float64", "float64", "stepr", 16, 5107UL, false, "float64", "sha:d7f616aedd0ce9522dd5eb883cc237fb")]
        [DataRow("float64", "float64", "stepr", 17, 5108UL, false, "float64", "sha:2bb47c4d387e6298939acba61c62c09b")]
        [DataRow("float64", "float64", "stepr", 31, 5109UL, false, "float64", "sha:f22da8ae8a74cb1ff5c02b0ee3d6b6f3")]
        [DataRow("float64", "float64", "stepr", 33, 5110UL, false, "float64", "sha:719661188e25949e34d07ea0572b8395")]
        [DataRow("float64", "float64", "stepr", 40, 5111UL, false, "float64", "sha:0c07d9e118c5533bec949ef41aa3788c")]
        [DataRow("float64", "float64", "bcast", 1, 5112UL, false, "float64", "sha:229657f8834ee3d00cc23aa35e1322d7")]
        [DataRow("float64", "float64", "bcast", 2, 5113UL, false, "float64", "sha:1594b0fbdf48cf38cb10dc0fb8224ddb")]
        [DataRow("float64", "float64", "bcast", 7, 5114UL, false, "float64", "sha:93988cabe3aebf06d340eacd0b10f407")]
        [DataRow("float64", "float64", "bcast", 15, 5115UL, false, "float64", "sha:ee11b8eda08593f682f3ac5bf5924f58")]
        [DataRow("float64", "float64", "bcast", 16, 5116UL, false, "float64", "sha:818b5403fb4f0c47232a33a41bbfac20")]
        [DataRow("float64", "float64", "bcast", 17, 5117UL, false, "float64", "sha:4ec2183189cdc3f1663426d9db868f73")]
        [DataRow("float64", "float64", "bcast", 31, 5118UL, false, "float64", "sha:0c3fcdc7fdb10a6b64a1d26e36dd60cf")]
        [DataRow("float64", "float64", "bcast", 33, 5119UL, false, "float64", "sha:0158706fe6318781692cb0371281bad6")]
        [DataRow("float64", "float64", "bcast", 40, 5120UL, false, "float64", "sha:dfa54126ecb18224ae7988b91d3ba3b6")]
        [DataRow("float64", "float64", "rev2", 1, 5121UL, false, "float64", "sha:211014468de6e28469d6f6bfe0e28b40")]
        [DataRow("float64", "float64", "rev2", 2, 5122UL, false, "float64", "sha:a0c62e2533d270d727ba776b88d02e21")]
        [DataRow("float64", "float64", "rev2", 7, 5123UL, false, "float64", "sha:4e35b2c0081f2c7e5c315274dcba20bc")]
        [DataRow("float64", "float64", "rev2", 15, 5124UL, false, "float64", "sha:81efd4a59be422b6c72e9bdcef9f19ef")]
        [DataRow("float64", "float64", "rev2", 16, 5125UL, false, "float64", "sha:3346cbcb19dc82fdbbd5da7e934d3940")]
        [DataRow("float64", "float64", "rev2", 17, 5126UL, false, "float64", "sha:2ffa02909cb16121bed0e75091499a3d")]
        [DataRow("float64", "float64", "rev2", 31, 5127UL, false, "float64", "sha:95ca68589373e2a2e4efb43f01476a1c")]
        [DataRow("float64", "float64", "rev2", 33, 5128UL, false, "float64", "sha:a3ed62d8730d69b19d829b216d0204ca")]
        [DataRow("float64", "float64", "rev2", 40, 5129UL, false, "float64", "sha:e7e74e8690d9365763f8b456d3443289")]
        [DataRow("float64", "float64", "c", 32, 5130UL, true, "float64", "hex:000000000000f87f")]
        [DataRow("float64", "float64", "rev", 32, 5131UL, true, "float64", "hex:000000000000f87f")]
        [DataRow("float64", "float64", "bcast", 32, 5132UL, true, "float64", "hex:000000000000f87f")]
        [DataRow("complex128", "complex128", "c", 1, 5133UL, false, "complex128", "sha:f97702225bbbbc7fd19feec4e71f4848")]
        [DataRow("complex128", "complex128", "c", 2, 5134UL, false, "complex128", "sha:04bf8ab25973498bbc9cf901378dae1e")]
        [DataRow("complex128", "complex128", "c", 7, 5135UL, false, "complex128", "sha:fc45ee0198da4c792622d0ee5637a0bd")]
        [DataRow("complex128", "complex128", "c", 15, 5136UL, false, "complex128", "sha:40a09edc0b5c083646708181e00c220b")]
        [DataRow("complex128", "complex128", "c", 16, 5137UL, false, "complex128", "sha:9f247736b2591610018e3ae57bac3b9f")]
        [DataRow("complex128", "complex128", "c", 17, 5138UL, false, "complex128", "sha:ab6bb581007f922cedf48121e1022a25")]
        [DataRow("complex128", "complex128", "c", 31, 5139UL, false, "complex128", "sha:73a2e23b269b72c8e4f644cb2f1fd204")]
        [DataRow("complex128", "complex128", "c", 33, 5140UL, false, "complex128", "sha:0ccd22b133e8cab5c7f1a97ba951d78e")]
        [DataRow("complex128", "complex128", "c", 40, 5141UL, false, "complex128", "sha:ae3e04f4e365075fb28bbb367a5808c5")]
        [DataRow("complex128", "complex128", "rev", 1, 5142UL, false, "complex128", "sha:133de571a98fad31e469af2f79c820d7")]
        [DataRow("complex128", "complex128", "rev", 2, 5143UL, false, "complex128", "sha:b3d5b48d19ac9b69baafe70376592855")]
        [DataRow("complex128", "complex128", "rev", 7, 5144UL, false, "complex128", "sha:14fda109e4b501c86a910324a1e4f82b")]
        [DataRow("complex128", "complex128", "rev", 15, 5145UL, false, "complex128", "sha:87424a47a77810a388ed5b47a503f158")]
        [DataRow("complex128", "complex128", "rev", 16, 5146UL, false, "complex128", "sha:f9918715fea2fcf23f46894749b0a656")]
        [DataRow("complex128", "complex128", "rev", 17, 5147UL, false, "complex128", "sha:1158722e8cab6861d3d212e2bb87edb7")]
        [DataRow("complex128", "complex128", "rev", 31, 5148UL, false, "complex128", "sha:c1f7054b84b25a72bc1cf4fd8a53d9e0")]
        [DataRow("complex128", "complex128", "rev", 33, 5149UL, false, "complex128", "sha:2dad1ecece809cfd4446b93d4fddc9d2")]
        [DataRow("complex128", "complex128", "rev", 40, 5150UL, false, "complex128", "sha:75001be3ce28ff6848decdbf4794c3f7")]
        [DataRow("complex128", "complex128", "revb", 1, 5151UL, false, "complex128", "sha:55ad7c2acc8f84da0e1c3ebf81bb36cc")]
        [DataRow("complex128", "complex128", "revb", 2, 5152UL, false, "complex128", "sha:03009e0e0df223965da144f1b6c6b2db")]
        [DataRow("complex128", "complex128", "revb", 7, 5153UL, false, "complex128", "sha:92628a8a4a5efb6b1c3f12f0dcaa9fa2")]
        [DataRow("complex128", "complex128", "revb", 15, 5154UL, false, "complex128", "sha:7e1d419a949cd80d8b77ad0435fe1933")]
        [DataRow("complex128", "complex128", "revb", 16, 5155UL, false, "complex128", "sha:4bde1025f7e35fbdc89ea5da02265a10")]
        [DataRow("complex128", "complex128", "revb", 17, 5156UL, false, "complex128", "sha:1e3c4235f034b468d0517327dbf4e793")]
        [DataRow("complex128", "complex128", "revb", 31, 5157UL, false, "complex128", "sha:459e880b55b6d746710596b1bec8b2bb")]
        [DataRow("complex128", "complex128", "revb", 33, 5158UL, false, "complex128", "sha:0171e95d0ba5869aee300f51587d13d8")]
        [DataRow("complex128", "complex128", "revb", 40, 5159UL, false, "complex128", "sha:5cd8dc27e8e1664d2dcaef85a3658f7c")]
        [DataRow("complex128", "complex128", "step", 1, 5160UL, false, "complex128", "sha:ee98d63568e40a98674d0757d9e70202")]
        [DataRow("complex128", "complex128", "step", 2, 5161UL, false, "complex128", "sha:7aee06464b3e20c5379017aea2408f2a")]
        [DataRow("complex128", "complex128", "step", 7, 5162UL, false, "complex128", "sha:d0679091bb09928d89d7d32cc559f53e")]
        [DataRow("complex128", "complex128", "step", 15, 5163UL, false, "complex128", "sha:2af1f3aaec745985a0b9baf80440dfbd")]
        [DataRow("complex128", "complex128", "step", 16, 5164UL, false, "complex128", "sha:b84b0641ca21609d256a4c7591791fb9")]
        [DataRow("complex128", "complex128", "step", 17, 5165UL, false, "complex128", "sha:47034e66c07defb8cbb546715611fb3e")]
        [DataRow("complex128", "complex128", "step", 31, 5166UL, false, "complex128", "sha:b69cf7b48e8afdcedb216ea556e1909a")]
        [DataRow("complex128", "complex128", "step", 33, 5167UL, false, "complex128", "sha:95fe8438d4027b80eb67636b81c26f9f")]
        [DataRow("complex128", "complex128", "step", 40, 5168UL, false, "complex128", "sha:060696bd5d50700f5385d8b783cdcd50")]
        [DataRow("complex128", "complex128", "stepr", 1, 5169UL, false, "complex128", "sha:c8e36c96d2a4c9151670b6f3a9007874")]
        [DataRow("complex128", "complex128", "stepr", 2, 5170UL, false, "complex128", "sha:49366885293217bcefe0053ccd6ffd93")]
        [DataRow("complex128", "complex128", "stepr", 7, 5171UL, false, "complex128", "sha:8aa440265723fc86964922ce6f82a0ff")]
        [DataRow("complex128", "complex128", "stepr", 15, 5172UL, false, "complex128", "sha:1a65be642826a4d264c74f0e2ddd6c03")]
        [DataRow("complex128", "complex128", "stepr", 16, 5173UL, false, "complex128", "sha:05c12745ccf4065c7be5d5f8a70ee6a5")]
        [DataRow("complex128", "complex128", "stepr", 17, 5174UL, false, "complex128", "sha:f8785f55874a7136f41ce1513ba56a4a")]
        [DataRow("complex128", "complex128", "stepr", 31, 5175UL, false, "complex128", "sha:6377235a8115f1be0fe4a90078f6f775")]
        [DataRow("complex128", "complex128", "stepr", 33, 5176UL, false, "complex128", "sha:3719740d56fc80b9b11c285bd3b1d4e4")]
        [DataRow("complex128", "complex128", "stepr", 40, 5177UL, false, "complex128", "sha:4574417af5922d62687e745550cc3e7a")]
        [DataRow("complex128", "complex128", "bcast", 1, 5178UL, false, "complex128", "sha:9a57d4911447e21d599e775037684379")]
        [DataRow("complex128", "complex128", "bcast", 2, 5179UL, false, "complex128", "sha:f995d458d79372bf1ab5c73ba5846530")]
        [DataRow("complex128", "complex128", "bcast", 7, 5180UL, false, "complex128", "sha:f33bbc159f138959c8a9f43c51d8d16b")]
        [DataRow("complex128", "complex128", "bcast", 15, 5181UL, false, "complex128", "sha:20df63ac5445f759dc575c974d048e93")]
        [DataRow("complex128", "complex128", "bcast", 16, 5182UL, false, "complex128", "sha:31b4c4c911a2367287c162df844527e3")]
        [DataRow("complex128", "complex128", "bcast", 17, 5183UL, false, "complex128", "sha:d79a1062275752388287d61b7ea5e5c2")]
        [DataRow("complex128", "complex128", "bcast", 31, 5184UL, false, "complex128", "sha:a4e671853bb48af9e891bebdbfb83497")]
        [DataRow("complex128", "complex128", "bcast", 33, 5185UL, false, "complex128", "sha:c732d023b95e54c1d8d4970e43c7289e")]
        [DataRow("complex128", "complex128", "bcast", 40, 5186UL, false, "complex128", "sha:45ebdacd808bf5a3e3bcf72bec2b6eb7")]
        [DataRow("complex128", "complex128", "rev2", 1, 5187UL, false, "complex128", "sha:fbd11f13e25a2d3422a0e02d06621786")]
        [DataRow("complex128", "complex128", "rev2", 2, 5188UL, false, "complex128", "sha:c288ee4e62df314bcf6ab7e791cd2e83")]
        [DataRow("complex128", "complex128", "rev2", 7, 5189UL, false, "complex128", "sha:8fe33b408c78ac3f816e870a4e536fa9")]
        [DataRow("complex128", "complex128", "rev2", 15, 5190UL, false, "complex128", "sha:af04b7d652b0cfec6c492d09ad7655dc")]
        [DataRow("complex128", "complex128", "rev2", 16, 5191UL, false, "complex128", "sha:f916c457acbfbb03dd4c05749df553aa")]
        [DataRow("complex128", "complex128", "rev2", 17, 5192UL, false, "complex128", "sha:76c85112913eb5d97342c2bfa67b53af")]
        [DataRow("complex128", "complex128", "rev2", 31, 5193UL, false, "complex128", "sha:f0d06d8c02e88a4bb3736f15d3e9e459")]
        [DataRow("complex128", "complex128", "rev2", 33, 5194UL, false, "complex128", "sha:bd08e86f51f87771a1c00234338415ce")]
        [DataRow("complex128", "complex128", "rev2", 40, 5195UL, false, "complex128", "sha:1b0e2c668579aac57e40d4b591b299fd")]
        [DataRow("complex128", "complex128", "c", 32, 5196UL, true, "complex128", "hex:000000000000f87f000000000000f8ff")]
        [DataRow("complex128", "complex128", "rev", 32, 5197UL, true, "complex128", "hex:000000000000f87f000000000000f87f")]
        [DataRow("complex128", "complex128", "bcast", 32, 5198UL, true, "complex128", "hex:000000000000f87f000000000000f87f")]
        [DataRow("float32", "float64", "c", 1, 5199UL, false, "float64", "sha:3d8d9ffead41f05561e1d2203070da3b")]
        [DataRow("float32", "float64", "rev", 1, 5200UL, false, "float64", "sha:9fc6ef70d2bd28129e9acb1752cc61fb")]
        [DataRow("float32", "float64", "c", 20, 5201UL, false, "float64", "sha:1191c750d3de1955c9a8975ad1f46b48")]
        [DataRow("float32", "float64", "rev", 20, 5202UL, false, "float64", "sha:3c461347995fa382b304be947dafaa7b")]
        [DataRow("float32", "float64", "c", 40, 5203UL, false, "float64", "sha:63d5095d36e74e49feb13ee5570d19b6")]
        [DataRow("float32", "float64", "rev", 40, 5204UL, false, "float64", "sha:3ac37775c3b51000564ac8699768677a")]
        [DataRow("int32", "float32", "c", 1, 5205UL, false, "float64", "sha:e5d55fe427c5ad3c851cae7e8d8de458")]
        [DataRow("int32", "float32", "rev", 1, 5206UL, false, "float64", "sha:ac802a1c48e54abf616c35a7e9d26e11")]
        [DataRow("int32", "float32", "c", 20, 5207UL, false, "float64", "sha:77b2c3e5bab36c48adf92a520cd7ddea")]
        [DataRow("int32", "float32", "rev", 20, 5208UL, false, "float64", "sha:37c905eca378872e9149e57eb8999d41")]
        [DataRow("int32", "float32", "c", 40, 5209UL, false, "float64", "sha:3418aa3fec9dbc0f8915375735e0bd63")]
        [DataRow("int32", "float32", "rev", 40, 5210UL, false, "float64", "sha:8aea2d99fe7a7323490eff9cbcf01fc2")]
        [DataRow("float16", "float32", "c", 1, 5211UL, false, "float32", "sha:f48406e19a8ff58ccf10e9be428b2fbf")]
        [DataRow("float16", "float32", "rev", 1, 5212UL, false, "float32", "sha:80e50522530df79ed692210685194980")]
        [DataRow("float16", "float32", "c", 20, 5213UL, false, "float32", "sha:02d3059a694edac10673efa11b2ce2a1")]
        [DataRow("float16", "float32", "rev", 20, 5214UL, false, "float32", "sha:454ab819fb01c3eda6869b430a6b1a81")]
        [DataRow("float16", "float32", "c", 40, 5215UL, false, "float32", "sha:0f0adbeb6545367aac1c349d37b1296e")]
        [DataRow("float16", "float32", "rev", 40, 5216UL, false, "float32", "sha:dd5e2ff00c741f1bf215a73fabd8ecd3")]
        [DataRow("int64", "complex128", "c", 1, 5217UL, false, "complex128", "sha:e84686d0046d5ca28fbf11bf9ec6a6eb")]
        [DataRow("int64", "complex128", "rev", 1, 5218UL, false, "complex128", "sha:0d3fa4a6b34690ae01bcbf40a9093f58")]
        [DataRow("int64", "complex128", "c", 20, 5219UL, false, "complex128", "sha:5144836daeb1dd3476063d463abccd6b")]
        [DataRow("int64", "complex128", "rev", 20, 5220UL, false, "complex128", "sha:360133cba53afbc9fa52bf4bcccfa7c4")]
        [DataRow("int64", "complex128", "c", 40, 5221UL, false, "complex128", "sha:c356b1ab2b281c02f63bddbf0a24042d")]
        [DataRow("int64", "complex128", "rev", 40, 5222UL, false, "complex128", "sha:6cfd66c6b600eeb6e8765a1bf9796e71")]
        [DataRow("bool", "float64", "c", 1, 5223UL, false, "float64", "sha:e6ad6c9a3a3b7658c35bacf6553fcb8f")]
        [DataRow("bool", "float64", "rev", 1, 5224UL, false, "float64", "sha:e6ad6c9a3a3b7658c35bacf6553fcb8f")]
        [DataRow("bool", "float64", "c", 20, 5225UL, false, "float64", "sha:47d74147687ffeab020129ad547c2703")]
        [DataRow("bool", "float64", "rev", 20, 5226UL, false, "float64", "sha:553f72ad128c15a0e0790a3e5c9ef9f6")]
        [DataRow("bool", "float64", "c", 40, 5227UL, false, "float64", "sha:1face5ec61c13e3a079964182d797ea8")]
        [DataRow("bool", "float64", "rev", 40, 5228UL, false, "float64", "sha:148bba074e29e147b4ebf13a21174855")]
        [DataRow("float32", "complex128", "c", 1, 5229UL, false, "complex128", "sha:6b822bd089e3cf38c56d69f318f9dadc")]
        [DataRow("float32", "complex128", "rev", 1, 5230UL, false, "complex128", "sha:3e2a7c9e33c44db20a4628e9d5766b1c")]
        [DataRow("float32", "complex128", "c", 20, 5231UL, false, "complex128", "sha:35593a988c72e8408a0ef4437bf7a5d7")]
        [DataRow("float32", "complex128", "rev", 20, 5232UL, false, "complex128", "sha:665c89cd772d7c96fb87c52abd60de78")]
        [DataRow("float32", "complex128", "c", 40, 5233UL, false, "complex128", "sha:01aa123d02c3691d4c650bdf77f8db51")]
        [DataRow("float32", "complex128", "rev", 40, 5234UL, false, "complex128", "sha:422a4b1a9e98958794edee9726baba55")]
        [DataRow("int32", "int64", "c", 1, 5235UL, false, "int64", "sha:3522dfbc8845dc81133b5c0d6aad7fb8")]
        [DataRow("int32", "int64", "rev", 1, 5236UL, false, "int64", "sha:a235b9df2f4f5db062b2e1011503ac62")]
        [DataRow("int32", "int64", "c", 20, 5237UL, false, "int64", "sha:5697511bef10a50f628e517937984438")]
        [DataRow("int32", "int64", "rev", 20, 5238UL, false, "int64", "sha:f88547a7a17e40d924c38d1be74b8308")]
        [DataRow("int32", "int64", "c", 40, 5239UL, false, "int64", "sha:bd9814fbbc7dbb622a34699c3bfa21cc")]
        [DataRow("int32", "int64", "rev", 40, 5240UL, false, "int64", "sha:c604354e589e60126150c81c13d1fe83")]
        public void Route1D_MatchesNumPyBits(string dtypeA, string dtypeB, string kind, int n, ulong seed, bool special,
            string resultDtype, string expected)
        {
            // The operands as the generator built them: a, then b, from one stream; the stepped / reversed-by-2 kinds
            // draw twice as many elements for the operand they step.
            var g = new Lcg(seed);
            long na = kind is "step" or "stepr" or "rev2" ? 2L * n : n;
            long nb = kind == "step" ? 2L * n : kind == "bcast" ? 1 : n;
            var a = g.Array(new[] { na }, dtypeA);
            var b = g.Array(new[] { nb }, dtypeB);
            if (special)
            {
                Specials(a);
                if (kind != "bcast")
                    Specials(b);
            }
            var (av, bv) = kind switch
            {
                "rev" => (a["::-1"], b),
                "revb" => (a, b["::-1"]),
                "step" => (a["::2"], b["::2"]),
                "stepr" => (a["::2"], b["::-1"]),
                "bcast" => (a, np.broadcast_to(b, new Shape(n))),
                "rev2" => (a["::-2"], b),
                _ => (a, b),
            };

            AssertNumPyBits(np.dot(av, bv), resultDtype, expected, $"{dtypeA}.{dtypeB} {kind} n={n}");
        }

        /// <summary>
        ///     Compares a result with NumPy's: the dtype, then either the first 32 hex digits of SHA-256 over the result's
        ///     bytes (<c>sha:</c>) or every byte (<c>hex:</c>), where a slot NumPy fills with a NaN only has to hold a NaN
        ///     (which NaN survives when several meet follows RyuJIT's operand order: see <see cref="NDDot"/>).
        /// </summary>
        /// <param name="r">NumSharp's result.</param>
        /// <param name="resultDtype">NumPy's result dtype.</param>
        /// <param name="expected">sha: or hex: and NumPy's bytes.</param>
        /// <param name="what">The case, for the failure message.</param>
        private static void AssertNumPyBits(NDArray r, string resultDtype, string expected, string what)
        {
            r.dtype.name.Should().Be(resultDtype, what);
            byte[] bytes = r.tobytes();
            if (expected.StartsWith("sha:", StringComparison.Ordinal))
            {
                Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant().Substring(0, 32)
                    .Should().Be(expected.Substring(4), "np.dot must give NumPy's bits for {0}", what);
                return;
            }

            byte[] want = Convert.FromHexString(expected.Substring(4));
            bytes.Length.Should().Be(want.Length, what);
            int part = resultDtype == "float32" ? 4 : 8;   // complex compares its two doubles separately
            for (int i = 0; i < want.Length; i += part)
            {
                bool wantNaN = part == 4
                    ? float.IsNaN(BitConverter.ToSingle(want, i))
                    : double.IsNaN(BitConverter.ToDouble(want, i));
                bool gotNaN = part == 4
                    ? float.IsNaN(BitConverter.ToSingle(bytes, i))
                    : double.IsNaN(BitConverter.ToDouble(bytes, i));
                if (wantNaN)
                {
                    gotNaN.Should().BeTrue("slot {0} of {1}: NumPy gives NaN", i / part, what);
                    continue;
                }
                bytes.AsSpan(i, part).SequenceEqual(want.AsSpan(i, part)).Should().BeTrue(
                    "slot {0} of {1}: NumPy {2}, NumSharp {3}", i / part, what,
                    Convert.ToHexString(want, i, part), Convert.ToHexString(bytes, i, part));
            }
        }

        /// <summary>
        ///     The models' own arithmetic at the edges NumPy's tests pin: no terms give +0, a sum of -0.0 products is +0
        ///     (every sum starts at +0), and the unit kernels' block boundaries (sdot 32, ddot 16, zdotu 8) carry the tail
        ///     after the kernel.
        /// </summary>
        [TestMethod]
        public unsafe void Models_EmptyAndNegativeZero()
        {
            float* f = stackalloc float[4] { -0.0f, -0.0f, -0.0f, -0.0f };
            float* one = stackalloc float[4] { 1f, 1f, 1f, 1f };
            BitConverter.SingleToInt32Bits(NDDot.FloatDot(f, 1, one, 1, 0)).Should().Be(0);
            BitConverter.SingleToInt32Bits(NDDot.FloatDot(f, 1, one, 1, 4)).Should().Be(0, "+0 + -0 is +0");
            BitConverter.SingleToInt32Bits(NDDot.FloatDot(f, 2, one, 2, 2)).Should().Be(0);
            BitConverter.SingleToInt32Bits(NDDot.FloatDot(f + 3, -1, one, 1, 4)).Should().Be(0);

            double* d = stackalloc double[4] { -0.0, -0.0, -0.0, -0.0 };
            double* oned = stackalloc double[4] { 1, 1, 1, 1 };
            BitConverter.DoubleToInt64Bits(NDDot.DoubleDot(d, 1, oned, 1, 0)).Should().Be(0);
            BitConverter.DoubleToInt64Bits(NDDot.DoubleDot(d, 1, oned, 1, 4)).Should().Be(0);
            BitConverter.DoubleToInt64Bits(NDDot.DoubleDot(d + 3, -1, oned, 1, 4)).Should().Be(0);
        }

        /// <summary>
        ///     A row function gives each output exactly what the per-output function gives it (the rows only run four
        ///     outputs' sums side by side): every class, a count that leaves a remainder, lengths around the kernel blocks.
        /// </summary>
        [TestMethod]
        public unsafe void RowFunctions_EqualPerOutputFunctions()
        {
            var g = new Lcg(77);
            foreach (long n in new long[] { 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 47, 64 })
            {
                const long count = 7, yStep = 70;
                var xd = (double[])g.Array(new long[] { 70 }, "float64").ToArray<double>();
                var yd = (double[])g.Array(new long[] { 70 * 8 }, "float64").ToArray<double>();
                var xf = (float[])g.Array(new long[] { 70 }, "float32").ToArray<float>();
                var yf = (float[])g.Array(new long[] { 70 * 8 }, "float32").ToArray<float>();
                var xc = (System.Numerics.Complex[])g.Array(new long[] { 70 }, "complex128").ToArray<System.Numerics.Complex>();
                var yc = (System.Numerics.Complex[])g.Array(new long[] { 70 * 8 }, "complex128").ToArray<System.Numerics.Complex>();
                fixed (double* x = xd, y = yd)
                fixed (float* x32 = xf, y32 = yf)
                fixed (System.Numerics.Complex* x128 = xc, y128 = yc)
                {
                    var od = new double[count];
                    var of = new float[count];
                    var oc = new System.Numerics.Complex[count];
                    fixed (double* o = od)
                    fixed (float* o32 = of)
                    fixed (System.Numerics.Complex* o128 = oc)
                    {
                        // unit rows
                        NDDot.DoubleDotUnitRow(x, 1, y, 1, yStep, o, count, n);
                        NDDot.FloatDotUnitRow(x32, 1, y32, 1, yStep, o32, count, n);
                        NDDot.ComplexDotUnitRow(x128, 1, y128, 1, yStep, o128, count, n);
                        for (long j = 0; j < count; j++)
                        {
                            BitConverter.DoubleToInt64Bits(od[j]).Should().Be(
                                BitConverter.DoubleToInt64Bits(NDDot.DoubleDotUnit(x, 1, y + j * yStep, 1, n)));
                            BitConverter.SingleToInt32Bits(of[j]).Should().Be(
                                BitConverter.SingleToInt32Bits(NDDot.FloatDotUnit(x32, 1, y32 + j * yStep, 1, n)));
                            oc[j].Should().Be(NDDot.ComplexDotUnit(x128, 1, y128 + j * yStep, 1, n));
                        }

                        // own-loop rows: a reversed first operand and a stepped second one
                        long nn = System.Math.Min(n, 9);
                        NDDot.DoubleDotOwnRow(x + 69, -1, y, 7, 1, o, count, nn);
                        NDDot.FloatDotOwnRow(x32 + 69, -1, y32, 7, 1, o32, count, nn);
                        NDDot.ComplexDotOwnRow(x128 + 69, -1, y128, 7, 1, o128, count, nn);
                        for (long j = 0; j < count; j++)
                        {
                            BitConverter.DoubleToInt64Bits(od[j]).Should().Be(
                                BitConverter.DoubleToInt64Bits(NDDot.DoubleDotOwn(x + 69, -1, y + j, 7, nn)));
                            BitConverter.SingleToInt32Bits(of[j]).Should().Be(
                                BitConverter.SingleToInt32Bits(NDDot.FloatDotOwn(x32 + 69, -1, y32 + j, 7, nn)));
                            oc[j].Should().Be(NDDot.ComplexDotOwn(x128 + 69, -1, y128 + j, 7, nn));
                        }
                    }
                }
            }
        }
    }
}
